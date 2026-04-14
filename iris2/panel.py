from __future__ import annotations
import json
import os
import re
import time
import threading
import urllib.request
import urllib.parse
from typing import List, Optional

from PyQt5.QtCore import Qt, QEvent, QPoint, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QPushButton, QScrollArea, QToolButton, QInputDialog,
    QSizePolicy, QFrame,
)
from PyQt5.QtGui import QPixmap, QColor, QPalette, QTextCursor, QFont

from .agent import IrisAgentWorker, IrisLocalWorker, IrisSmartWorker, get_api_key, save_api_key
from .app_state import state
from .tools import (
    tool_get_app_state, tool_get_scan_results, tool_generate_report,
    tool_llm_log_audit, tool_run_scan
)
from .event_bus import bus, AppEvent, EventType


# ── Colours ────────────────────────────────────────────────────────────────

_BG        = "rgba(18, 18, 28, 220)"
_BG_USER   = "rgba(30, 50, 90, 200)"
_BG_IRIS   = "rgba(28, 38, 55, 200)"
_TEXT      = "#e8eaf6"
_TEXT_DIM  = "#9fa8c0"
_ACCENT    = "#7c4dff"
_ACCENT_H  = "#9c6fff"
_ACCENT_L  = "#110f17"
_FRAME_BTN = "#1de9b6"   # clickable frame number colour
_BORDER    = "rgba(255,255,255,15)"

_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "iris_memory.json")


class ChatInput(QTextEdit):
    """Enter sends. Shift+Enter inserts newline."""
    sendRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_h = 30
        self._max_h = 120
        self.textChanged.connect(self._update_height)
        self._update_height()

    def _update_height(self):
        doc_h = self.document().size().height()
        h = int(doc_h) + 12
        h = max(self._min_h, min(self._max_h, h))
        self.setFixedHeight(h)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                event.accept()
                self.sendRequested.emit()
            return
        super().keyPressEvent(event)


# ── Message bubble with clickable frame numbers ────────────────────────────

class MessageBubble(QFrame):
    """
    A single message bubble. Frame numbers like **561** are rendered as
    highlighted monospace spans — visually distinct but not clickable.
    The user navigates manually using the frame slider.
    """

    def __init__(
        self,
        text: str,
        role: str,
        mode: str = "",
        model_name: str = "",
        timestamp: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        bg = _BG_USER if role == "user" else _BG_IRIS
        border_left = f"2px solid {'#5c6bc0' if role == 'user' else _ACCENT}"

        self.setStyleSheet(f"""
            MessageBubble {{
                background: {bg};
                border-radius: 8px;
                border-left: {border_left};
                padding: 1px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(3)

        # Header row: role label + mode dot inline
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(1)
        role_lbl = QLabel("You" if role == "user" else "Iris", self)
        role_lbl.setStyleSheet(
            f"color: {_TEXT_DIM}; font-size: 9px; font-weight: bold; background: transparent;")
        header.addWidget(role_lbl)
        if role != "user" and mode:
            dot = QLabel("●", self)
            if mode in ("rule", "local"):
                dot.setStyleSheet("color: #8a8a8a; font-size: 9px; background: transparent;")
                dot.setToolTip("Rule-based (no LLM/API)")
            elif mode == "ollama":
                dot.setStyleSheet("color: #43a047; font-size: 9px; background: transparent;")
                dot.setToolTip("Ollama")
            elif mode == "api":
                dot.setStyleSheet(f"color: {_ACCENT_H}; font-size: 9px; background: transparent;")
                dot.setToolTip("API")
            else:
                dot.setStyleSheet("color: #6b6b6b; font-size: 9px; background: transparent;")
            header.addWidget(dot)
        if role != "user" and model_name:
            model_lbl = QLabel(f" {model_name}", self)
            model_lbl.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 8px; background: transparent;")
            header.addWidget(model_lbl)
        header.addStretch()
        if timestamp:
            ts_lbl = QLabel(timestamp, self)
            ts_lbl.setStyleSheet(
                f"color: {_TEXT_DIM}; font-size: 8px; background: transparent;")
            header.addWidget(ts_lbl)
        layout.addLayout(header)

        # Render text — **NUMBER** becomes styled inline HTML span
        rendered = self._render_frame_numbers(text)
        lbl = QLabel(self)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setOpenExternalLinks(False)
        lbl.setTextFormat(Qt.RichText)
        lbl.setText(rendered)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 10px; background: transparent; line-height: 1.4;")
        layout.addWidget(lbl)

    @staticmethod
    def _render_frame_numbers(text: str) -> str:
        """
        Convert **NUMBER** markers to styled HTML spans.
        e.g. **561** → <span style="color:#1de9b6;font-family:monospace;
                         background:rgba(29,233,182,12);
                         padding:0 3px;border-radius:2px;">561</span>
        """
        import html
        # Escape HTML first so other content is safe
        safe = html.escape(text)
        # Allow long paths/strings to wrap
        safe = safe.replace("/", "/<wbr>").replace("\\", "\\<wbr>")
        safe = safe.replace("═", "═<wbr>").replace("─", "─<wbr>")
        # Replace **NUMBER** with styled span
        safe = re.sub(
            r"\*\*(\d+)\*\*",
            r'<span style="color:#1de9b6;font-family:monospace;font-weight:bold;'
            r'background:rgba(29,233,182,10);padding:0 3px;border-radius:2px;">\1</span>',
            safe,
        )

        styled_lines = []
        active_issue = None
        for line in safe.split("\n"):
            plain = re.sub(r"<[^>]+>", "", line)
            stripped = plain.strip()
            is_summary_heading = re.match(
                r"^[🔴🟡⚪✅❌⚠]\s+[A-Z][A-Z ]+\s*:\s*\d+\s*$",
                stripped
            ) is not None
            starts_issue = (
                stripped.startswith("- ")
                or stripped.startswith("• ")
                or stripped.startswith("◦ ")
                or re.match(r"^[🔴🟡⚪✅❌⚠]\s+\[", stripped) is not None
                or "LOG ERROR:" in stripped.upper()
                or "[W01]" in stripped.upper()
            )
            is_continuation = stripped.startswith("→") or stripped.startswith("&rarr;")
            upper = stripped.upper()
            if is_summary_heading:
                active_issue = None
            elif starts_issue and ("[CRITICAL]" in upper or "🔴" in upper or "❌" in upper):
                active_issue = "critical"
                line = (
                    '<span style="background-color:#4a1f24;color:#ffb4ab;">'
                    '&nbsp;&nbsp;<b>'
                    f"{line}"
                    '</b>&nbsp;&nbsp;</span>'
                )
            elif starts_issue and ("[WARNING]" in upper or "🟡" in upper or "⚠" in upper):
                active_issue = "warning"
                line = (
                    '<span style="background-color:#4a3a12;color:#ffe082;">'
                    '&nbsp;&nbsp;<b>'
                    f"{line}"
                    '</b>&nbsp;&nbsp;</span>'
                )
            elif is_continuation and active_issue == "critical":
                line = (
                    '<span style="background-color:#3b1a1e;color:#ffd0cb;">'
                    '&nbsp;&nbsp;&nbsp;&nbsp;'
                    f"{line}"
                    '&nbsp;&nbsp;</span>'
                )
            elif is_continuation and active_issue == "warning":
                line = (
                    '<span style="background-color:#40320f;color:#ffecb3;">'
                    '&nbsp;&nbsp;&nbsp;&nbsp;'
                    f"{line}"
                    '&nbsp;&nbsp;</span>'
                )
            elif stripped:
                active_issue = None
            styled_lines.append(line)

        safe = "<br>".join(styled_lines)
        return f'<div style="white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;">{safe}</div>'


# ── Choice widget — interactive multi-select for compare/report flows ──────

class ChoiceWidget(QFrame):
    """
    Rendered when Iris needs the user to pick from a list of datasets.
    Shows checkboxes (multi-select) or radio buttons (single-select).
    On confirm, sends the selection as a new user message back to Iris.
    """
    confirmed = pyqtSignal(str)   # emits the composed user reply

    def __init__(self, prompt: str, options: list, multi: bool = False, parent=None):
        super().__init__(parent)
        self._multi   = multi
        self._options = options
        self._checks  = []
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"""
            ChoiceWidget {{
                background: rgba(20, 30, 50, 210);
                border-radius: 8px;
                border-left: 2px solid {_ACCENT};
                padding: 2px;
            }}
        """)

        from PyQt5.QtWidgets import QCheckBox, QRadioButton, QButtonGroup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(4)

        # Prompt
        lbl = QLabel(prompt, self)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 10px; font-weight: bold; background: transparent;")
        layout.addWidget(lbl)

        if not multi:
            self._btn_group = QButtonGroup(self)

        for i, opt in enumerate(options):
            if multi:
                cb = QCheckBox(opt, self)
                cb.setStyleSheet(f"color: {_TEXT}; font-size: 10px; background: transparent;")
                layout.addWidget(cb)
                self._checks.append(cb)
            else:
                from PyQt5.QtWidgets import QRadioButton
                rb = QRadioButton(opt, self)
                rb.setStyleSheet(f"color: {_TEXT}; font-size: 10px; background: transparent;")
                layout.addWidget(rb)
                self._btn_group.addButton(rb, i)
                self._checks.append(rb)

        # Confirm button
        btn = QPushButton("Confirm selection", self)
        btn.setFixedHeight(26)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {_ACCENT}; color: white;
                border-radius: 4px; font-size: 9px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {_ACCENT_H}; }}
        """)
        btn.clicked.connect(self._on_confirm)
        layout.addWidget(btn)

    def _on_confirm(self):
        selected = [opt for cb, opt in zip(self._checks, self._options)
                    if cb.isChecked()]
        if not selected:
            return
        if self._multi:
            reply = "Compare: " + " vs ".join(selected)
        else:
            reply = f"Report for: {selected[0]}"
        self.confirmed.emit(reply)
        # Grey out widget after selection
        self.setEnabled(False)
        self.setStyleSheet(self.styleSheet() + "opacity: 0.5;")


class FolderPickWidget(QFrame):
    """Inline chat widget: lets user choose a folder when Iris requests a path."""
    requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"""
            FolderPickWidget {{
                background: rgba(20, 30, 50, 210);
                border-radius: 8px;
                border-left: 2px solid {_ACCENT};
                padding: 2px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(4)

        lbl = QLabel("Pick a folder to continue:", self)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {_TEXT}; font-size: 10px; background: transparent;")
        layout.addWidget(lbl)

        btn = QPushButton("Select Folder", self)
        btn.setFixedHeight(26)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {_ACCENT}; color: white;
                border-radius: 4px; font-size: 9px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {_ACCENT_H}; }}
        """)
        btn.clicked.connect(self._on_click)
        layout.addWidget(btn)

    def _on_click(self):
        self.requested.emit()
        self.setEnabled(False)


# ── Tool-call indicator ────────────────────────────────────────────────────

_TOOL_LABELS = {
    "get_app_state":            "📋 Reading app state…",
    "list_folder":              "📁 Listing folder…",
    "read_logs":                "📄 Reading logs…",
    "run_scan":                 "🔍 Scanning dataset…",
    "refresh_scan":             "🔄 Re-scanning dataset (fresh)…",
    "get_scan_results":         "📊 Loading scan results…",
    "find_anomaly_frames":      "🎯 Finding anomaly frames…",
    "get_frame_info":           "🖼️ Reading frame info…",
    "navigate_to_frame":        "➡️ Navigating to frame…",
    "open_dataset":             "📂 Opening dataset…",
    "set_zoom":                 "🔎 Setting zoom…",
    "generate_report":          "📝 Generating report…",
    "compare_datasets":         "⚖️ Comparing datasets…",
    "browse_folder_tree":       "🌳 Scanning folder tree…",
    "get_histogram_state":      "📊 Reading histogram…",
    "list_open_datasets":       "📋 Listing open datasets…",
    "detect_repeating_pattern": "🧩 Detecting repeating pattern…",
    "session_report":           "📋 Generating session report…",
    # Memory
    "save_memory":              "🧠 Saving to memory…",
    "recall_memory":            "🧠 Recalling past findings…",
    "memory_summary":           "🧠 Reading memory…",
    # Knowledge base
    "knowledge_search":         "📚 Searching knowledge base…",
    "index_knowledge":          "📥 Indexing knowledge files…",
    "knowledge_status":         "📋 Checking knowledge base…",
}


# ── Main panel ─────────────────────────────────────────────────────────────

class IrisPanel(QWidget):
    """
    Floating overlay panel. Parented to the main window so it sits on top.
    Call toggle() to show/hide.
    """
    dbCheckFinished = pyqtSignal(str)
    uiCall = pyqtSignal(object)

    def __init__(self, main_window):
        super().__init__(main_window)
        self._host    = main_window
        self._enabled = False
        self._width   = 260
        self._workers: List[IrisAgentWorker] = []
        self._history: List[dict] = []        # legacy cache (kept for compatibility)
        self._history_by_tab: dict = {}       # tab_index -> [{role, content}]
        self._stream_bubble: Optional[MessageBubble] = None
        self._pending_bubble: Optional[QLabel] = None
        self._pending_text = ""
        self._spinner_frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
        self._spinner_idx = 0
        self._pending_timer = QTimer(self)
        self._pending_timer.setInterval(120)
        self._pending_timer.timeout.connect(self._tick_pending_spinner)
        self._input_locked = False
        self._api_prompted = False
        self._welcome_shown = False
        self._last_assistant_text = ""
        self._last_assistant_ts = 0.0
        self._last_local_scan = None
        self._local_task_status = ""
        self._scan_progress = None
        self._scan_status_label = None
        self._scan_status_last_ts = 0.0
        self._scan_status_last_text = ""
        self._response_mode = "rule"
        self._response_model = ""
        self._assist_mode = "smart"
        self._memory = self._load_memory()
        self._prune_dataset_memory()
        self._apply_memory_settings()
        self.dbCheckFinished.connect(self._on_db_check_finished)
        self.uiCall.connect(self._run_ui_call)

        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.Widget)
        self.hide()

        self._build_ui()
        self._host.installEventFilter(self)

        # Wire close-tab events so we can clean up
        bus.subscribe(EventType.TAB_CLOSED, lambda e: None)
        bus.subscribe(EventType.DATASET_LOADED, self._on_dataset_loaded_event)
        bus.subscribe(EventType.TAB_ACTIVATED, self._on_tab_activated_event)

    # ── UI Construction ────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Outer container with background ───────────────────────────────
        container = QWidget(self)
        container.setStyleSheet(f"""
            QWidget {{
                background: {_BG};
                border-radius: 10px;
                border: 1px solid {_BORDER};
            }}
        """)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(6, 6, 6, 6)
        c_layout.setSpacing(4)
        layout.addWidget(container)

        # ── Header ─────────────────────────────────────────────────────────
        header = QWidget(container)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(4, 0, 0, 0)

        title = QLabel("✦ Iris", header)
        title.setStyleSheet(f"color: {_ACCENT_H}; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(title)

        self._api_enabled = True
        self._last_mode = "idle"
        self._mode_dot = QToolButton(header)
        self._mode_dot.setText("●")
        self._mode_dot.setCursor(Qt.PointingHandCursor)
        self._mode_dot.setStyleSheet("color: #6b6b6b; font-size: 10px; border: none; background: transparent;")
        self._mode_dot.setToolTip("Idle (click to disable API)")
        self._mode_dot.clicked.connect(self._toggle_api_enabled)
        h_layout.addWidget(self._mode_dot)
        h_layout.addStretch()

        close_btn = QToolButton(header)
        close_btn.setText("×")
        close_btn.setFixedSize(18, 18)
        close_btn.setStyleSheet(f"""
            QToolButton {{ background: transparent; color: {_TEXT_DIM};
                           border: none; font-size: 14px; }}
            QToolButton:hover {{ color: #ef5350; }}
        """)
        close_btn.clicked.connect(self.hide_panel)
        h_layout.addWidget(close_btn)
        c_layout.addWidget(header)

        # ── Scroll area for messages ───────────────────────────────────────
        self._scroll = QScrollArea(container)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical { width: 4px; background: transparent; }
            QScrollBar::handle:vertical { background: rgba(255,255,255,30); border-radius: 2px; }
        """)
        self._msg_container = QWidget()
        self._msg_container.setStyleSheet("background: transparent;")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(0, 0, 0, 0)
        self._msg_layout.setSpacing(6)
        self._msg_layout.addStretch()
        self._scroll.setWidget(self._msg_container)
        c_layout.addWidget(self._scroll, 1)

        # ── Input row ──────────────────────────────────────────────────────
        input_row = QWidget(container)
        i_layout = QHBoxLayout(input_row)
        i_layout.setContentsMargins(0, 0, 0, 0)
        i_layout.setSpacing(4)

        self._input = ChatInput(input_row)
        self._input.setPlaceholderText("Ask Iris…")
        self._input.sendRequested.connect(self._on_send)
        self._input.setAcceptRichText(False)
        self._input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._input.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._input.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(255,255,255,8);
                color: {_TEXT};
                border: 1px solid rgba(255,255,255,20);
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
            }}
            QTextEdit:focus {{
                border-color: {_ACCENT};
            }}
        """)
        i_layout.addWidget(self._input, 1)

        send_btn = QPushButton("↑", input_row)
        send_btn.setFixedSize(28, 28)
        send_btn.clicked.connect(self._on_send)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {_ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {_ACCENT_H}; }}
        """)
        self._send_btn = send_btn
        i_layout.addWidget(send_btn)

        self._screenshot_btn = QToolButton(input_row)
        self._screenshot_btn.setText("📷")
        self._screenshot_btn.setCheckable(True)
        self._screenshot_btn.setFixedSize(28, 28)
        self._screenshot_btn.setToolTip("Include screenshot with next message")
        self._screenshot_btn.setStyleSheet(f"""
            QToolButton {{
                background: rgba(255,255,255,8);
                border: 1px solid rgba(255,255,255,15);
                border-radius: 6px;
                font-size: 12px;
            }}
            QToolButton:checked {{ background: rgba(124,77,255,100); border-color: {_ACCENT}; }}
        """)
        i_layout.addWidget(self._screenshot_btn)
        c_layout.addWidget(input_row)

        # ── Mode row (below input) ───────────────────────────────────────
        mode_row = QWidget(container)
        m_layout = QHBoxLayout(mode_row)
        m_layout.setContentsMargins(0, 2, 0, 0)
        m_layout.setSpacing(6)

        mode_lbl = QLabel("Mode", mode_row)
        mode_lbl.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 9px;")
        m_layout.addWidget(mode_lbl)

        from PyQt5.QtWidgets import QButtonGroup
        self._mode_group = QButtonGroup(mode_row)
        self._mode_group.setExclusive(True)

        def _make_mode_btn(text: str, tooltip: str) -> QToolButton:
            btn = QToolButton(mode_row)
            btn.setText(text)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(tooltip)
            btn.setStyleSheet("""
                QToolButton {
                    color: #d0d5f0;
                    background: rgba(255,255,255,6);
                    border: 1px solid rgba(255,255,255,15);
                    border-radius: 4px;
                    padding: 0 6px;
                    font-size: 9px;
                }
                QToolButton:checked {
                    background: rgba(124,77,255,120);
                    border-color: rgba(124,77,255,180);
                    color: #0b0a10;
                }
                QToolButton:hover { border-color: rgba(255,255,255,40); }
            """)
            return btn

        self._mode_btn_local = _make_mode_btn("Local", "Local only (offline)")
        self._mode_btn_smart = _make_mode_btn("Smart", "Local first, online when needed")
        self._mode_btn_online = _make_mode_btn("Online", "Always online response")

        self._mode_group.addButton(self._mode_btn_local)
        self._mode_group.addButton(self._mode_btn_smart)
        self._mode_group.addButton(self._mode_btn_online)
        m_layout.addWidget(self._mode_btn_local)
        m_layout.addWidget(self._mode_btn_smart)
        m_layout.addWidget(self._mode_btn_online)
        m_layout.addStretch()
        c_layout.addWidget(mode_row)

        self._mode_btn_local.clicked.connect(lambda: self._set_assist_mode("local"))
        self._mode_btn_smart.clicked.connect(lambda: self._set_assist_mode("smart"))
        self._mode_btn_online.clicked.connect(lambda: self._set_assist_mode("online"))
        self._sync_mode_buttons()

        self._task_status_lbl = QLabel("Scan: idle", mode_row)
        self._task_status_lbl.setStyleSheet("color: #8ea0c9; font-size: 9px;")
        m_layout.addWidget(self._task_status_lbl)

    # ── Toggle button (lives in main window's toolbar) ─────────────────────

    def make_toggle_button(self, parent) -> QPushButton:
        """Create and return the toolbar button that opens/closes the panel."""
        btn = QPushButton("✦ Iris", parent)
        btn.setCheckable(True)
        btn.setFixedHeight(22)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: rgba(17, 15, 23, 1);
                color: {_ACCENT_H};
                border: 1px solid rgba(124,77,255,80);
                border-radius: 4px;
                padding: 0px 8px;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:checked {{
                background: rgba(80,40,140,220);
                color: {_ACCENT_L};
                border-color: {_ACCENT};
            }}
            QPushButton:hover {{ border-color: {_ACCENT_H}; }}
        """)
        btn.toggled.connect(self.toggle)
        self._toggle_btn = btn
        return btn

    # ── Visibility ─────────────────────────────────────────────────────────

    def toggle(self, on: bool):
        self._enabled = on
        if on:
            self.show()
            is_wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
            if not is_wayland:
                self.raise_()
                self._input.setFocus()
            if not self._welcome_shown:
                self._add_message("iris",
                    "How can I help you today?")
                self._welcome_shown = True
        else:
            self.hide()
        self._reposition()

    def hide_panel(self):
        self._enabled = False
        self.hide()
        if hasattr(self, "_toggle_btn"):
            self._toggle_btn.blockSignals(True)
            self._toggle_btn.setChecked(False)
            self._toggle_btn.blockSignals(False)

    # ── Input handling ─────────────────────────────────────────────────────

    def _on_send(self):
        if self._input_locked:
            return
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()

        # Handle slash commands
        if text.startswith("?"):
            self._handle_command(text)
            return

        self._add_message("user", text)

        # ── Mode routing ──────────────────────────────────────────────────
        if self._assist_mode == "local":
            self._run_local(text)
            return

        if self._assist_mode == "smart":
            # Ollama is boss — decides when to use rules vs API
            self._run_smart(text)
            return

        # online mode — API is boss
        self._set_mode_indicator("api")
        self._run_agent(text)

    def _is_task_kind(self, kind: str) -> bool:
        """Kinds that should execute immediately via rules/tools."""
        return kind in {
            "open", "open_prompt", "navigate", "zoom",
            "scan_start", "report",
        }

    def _on_pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose Folder to Scan")
        if not folder:
            return
        try:
            bus.emit(AppEvent(EventType.OPEN_DATASET, {"folder": folder}, source="iris-panel"))
        except Exception:
            pass
        user_text = f"Scan this folder and report anomalies:\n{folder}"
        self._add_message("user", user_text)
        self._run_agent(user_text)

    def _should_show_folder_picker(self, text: str) -> bool:
        t = (text or "").lower()
        # Only show the picker for an explicit picker prompt, not generic chat.
        return (
            "pick a folder to continue" in t
            or "click select folder" in t
        )

    def _should_capture_screenshot(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in (
            "see", "look", "screen", "screenshot", "image", "viewer",
            "histogram", "band", "frame", "pixel", "exposure",
            "saturation", "dn", "contrast", "tab", "panel", "ui",
        ))

    def _active_mode(self) -> str:
        """Best-effort active mode: band/raw/video/live/tiled."""
        try:
            tab = state.active_tab
            if tab and getattr(tab, "mode", None):
                return str(tab.mode).lower()
        except Exception:
            pass
        return "band"

    def _try_local_response(self, text: str) -> tuple[str, str]:
        """
        Fast local fallback for common UI/help questions.
        Returns (response_text, kind) if handled, else ("", "").
        """
        def _normalize_command_text(raw: str) -> str:
            s = (raw or "").lower().strip()
            if not s:
                return s
            replacements = {
                "chnage": "change",
                "cahnge": "change",
                "chnage": "change",
                "enhnace": "enhance",
                "enhace": "enhance",
                "ehnance": "enhance",
                "enhacne": "enhance",
                "contrsat": "contrast",
                "constrast": "contrast",
                "contast": "contrast",
                "histgram": "histogram",
                "histogra": "histogram",
                "individula": "individual",
                "indivdual": "individual",
                "magnifer": "magnifier",
                "magnfier": "magnifier",
                "maginifier": "magnifier",
            }
            for wrong, right in replacements.items():
                s = re.sub(rf"\b{re.escape(wrong)}\b", right, s)
            s = re.sub(r"\bto\s+to\b", "to", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        t = _normalize_command_text(text)
        if not t:
            return "", ""

        if self._should_load_last_dataset(t):
            path = self._get_last_dataset_path()
            if path and os.path.isdir(path):
                try:
                    bus.emit(AppEvent(EventType.OPEN_DATASET, {"folder": path}, source="iris-local"))
                    self._remember_action("open", path, "opened last dataset")
                    self._remember_dataset(path)
                    return f"Opened last dataset: {os.path.basename(path)}", "open"
                except Exception:
                    return f"Could not open last dataset: {path}", "open"
            return "No recent dataset found. Use 'open /path/to/folder'.", "open"

        if ("individual band" in t or "individual bands" in t) and not any(k in t for k in ("open", "close", "show", "hide")):
            return (
                "Individual Bands quick use:\n"
                "1. Enable Individual Bands in Display Modes.\n"
                "2. Tick band checkboxes at the top.\n"
                "3. Click the new band tab (with * ) to load it.\n"
                "4. Use the frame slider/Play to scrub.\n"
                "If blank: no band checkbox enabled or you didn’t click the tab."
            ), "help"

        if "histogram" in t and not any(k in t for k in ("open", "close", "show", "hide")):
            return (
                "Histogram quick use:\n"
                "1. Open Histogram tab.\n"
                "2. Single Frame vs Frame Range controls what’s plotted.\n"
                "3. Use Min/Max or Auto for display scaling.\n"
                "If nothing shows, load a dataset and move the frame slider."
            ), "help"

        if ("rgb" in t or "fusion" in t) and not any(k in t for k in ("open", "close", "show", "hide")):
            return (
                "RGB Fusion quick use:\n"
                "1. Open RGB Fusion tab.\n"
                "2. Pick bands for R/G/B.\n"
                "3. Use offsets if alignment is off.\n"
                "4. Adjust contrast if preview looks flat."
            ), "help"

        if "list tabs" in t or "open tabs" in t or "list datasets" in t:
            info = tool_get_app_state()
            tabs = info.get("open_tabs", [])
            if not tabs:
                return "No datasets are open.", "status"
            lines = ["Open tabs:"]
            for tab in tabs:
                idx = tab.get("tab_index", "?")
                name = tab.get("dataset_name", "?")
                frames = tab.get("frame_count", "?")
                active = " (active)" if tab.get("is_active") else ""
                lines.append(f"- {idx}: {name} · {frames} frames{active}")
            return "\n".join(lines), "status"

        if "status" in t or "app state" in t or "where am i" in t or "what's open" in t:
            info = tool_get_app_state()
            folder = info.get("active_folder") or "None"
            frame = info.get("active_frame", 0)
            scan = "yes" if info.get("scan_available") else "no"
            last_action = self._last_action_summary()
            return (
                f"Active folder: {folder}\n"
                f"Active frame: {frame}\n"
                f"Scan available: {scan}\n"
                f"Last action: {last_action}"
            ), "status"

        if "what is displayed" in t or "what am i seeing" in t or "on screen" in t:
            info = tool_get_app_state()
            tabs = info.get("open_tabs", [])
            active = next((x for x in tabs if x.get("is_active")), None)
            if not active:
                return "No dataset is open right now.", "status"
            return (
                f"Showing: {active.get('dataset_name')} "
                f"({active.get('frame_count')} frames, mode {active.get('mode')}).\n"
                f"Active frame: {active.get('current_frame')}."
            ), "status"

        if "scan result" in t or "scan results" in t:
            res = tool_get_scan_results("")
            if res.get("status") == "NO_RESULTS" or res.get("error"):
                return "No scan results found. Run a scan first.", "report"
            msg = (
                f"Scan results: {res.get('scan_type')} · "
                f"health {res.get('health_score')} · "
                f"{len(res.get('anomaly_frames', []))} anomaly frames."
            )
            self._remember_report(res.get("folder") or "", msg)
            return msg, "report"

        if re.search(r"\breport\b", t) and (
            re.search(r"\b(generate|make|create|show)\b", t) or
            t == "report" or t.startswith("report ")
        ):
            loaded_folder = self._resolve_loaded_folder()
            if not loaded_folder:
                chosen = QFileDialog.getExistingDirectory(self, "Select Dataset Folder for Report")
                if not chosen:
                    return "No dataset loaded. Select Folder first.", "status"
                try:
                    bus.emit(AppEvent(EventType.OPEN_DATASET, {"folder": chosen}, source="iris-panel"))
                except Exception:
                    pass
                return (
                    f"Loaded dataset folder: {os.path.basename(chosen)}\n"
                    "Run 'report' again, or say 'scan' if you want a fresh scan first."
                ), "status"
            include_examples = ("example" in t or "examples" in t or "detailed" in t or "detail" in t)
            rep = tool_generate_report(loaded_folder, include_examples=include_examples, enable_template_comparison=False)
            if rep.get("error"):
                return rep.get("error"), "report"
            report_text = rep.get("report", "")
            report_text = self._maybe_append_llm_log_audit(report_text, rep.get("folder") or "")
            self._remember_report(rep.get("folder") or "", report_text)
            return report_text, "report"

        if self._looks_like_dataset_scan_request(text):
            # Run a local scan in a background thread
            scan_mode = self._choose_scan_mode(text)
            loaded_folder = self._resolve_loaded_folder()
            if not loaded_folder:
                try:
                    chosen = QFileDialog.getExistingDirectory(self, "Select Dataset Folder to Scan")
                    if not chosen:
                        return "No dataset loaded. Select Folder first.", "status"
                    try:
                        bus.emit(AppEvent(EventType.OPEN_DATASET, {"folder": chosen}, source="iris-panel"))
                    except Exception:
                        pass
                    return (
                        f"Loaded dataset folder: {os.path.basename(chosen)}\n"
                        "Run 'scan' again, or say 'report' if you want the cached report."
                    ), "status"
                except Exception as e:
                    return f"Could not open folder picker: {e}", "status"
            def _do_scan():
                self._set_local_task_status("SCAN_RUNNING")
                folder = self._resolve_loaded_folder() or self._resolve_active_folder()
                if not folder:
                    result = {"error": "No active dataset folder resolved for current tab."}
                else:
                    def _progress_cb(msg: str):
                        pct, text = self._parse_progress_msg(msg)
                        if pct is not None:
                            self.uiCall.emit(lambda: self._set_scan_progress(pct))
                        if text:
                            self.uiCall.emit(lambda t=text, p=pct: self._update_scan_status_line(t, p))
                    result = tool_run_scan(folder, scan_mode, progress_cb=_progress_cb)
                result = self._coerce_result_dict(result, "scan result")
                self._last_local_scan = result
                self._set_local_task_status("SCAN_DONE")
                def _done():
                    self._stop_pending()
                    self._finish_scan_status_line("Scan complete.")
                    if result.get("error"):
                        self._add_message("iris", f"Scan failed: {result['error']}")
                    else:
                        folder = (
                            result.get("folder", "")
                            or self._resolve_active_folder()
                            or state.active_folder
                            or ""
                        )
                        text = (
                            f"Scan complete: health {result.get('health_score')} · "
                            f"{result.get('anomaly_count')} anomalies."
                        )
                        # Always show completion summary first, then auto-post full report.
                        self._add_message("iris", text, mode="local")
                        rep = self._coerce_result_dict(
                            tool_generate_report(folder, enable_template_comparison=False), "report result")
                        if rep.get("report"):
                            report_text = rep.get("report", "")
                            report_text = self._maybe_append_llm_log_audit(report_text, rep.get("folder") or folder)
                            self._add_message("iris", report_text, mode="local")
                            self._remember_report(folder, report_text)
                        elif rep.get("error"):
                            self._add_message("iris", f"Report generation failed: {rep['error']}", mode="local")
                            self._remember_report(folder, text)
                self.uiCall.emit(_done)
            threading.Thread(target=_do_scan, daemon=True).start()
            self._start_pending(f"Scanning dataset ({scan_mode})…")
            self._remember_action("scan", "", "started local scan")
            return f"Started scan ({scan_mode}). I’ll post results when done.", "scan_start"

        if t in ("done", "done?", "status", "scan status", "scan done"):
            if self._local_task_status == "SCAN_RUNNING":
                return "Scan is still running.", "status"
            if self._local_task_status == "SCAN_TREE_RUNNING":
                return "Folder scan is still running.", "status"
            if self._local_task_status == "SCAN_DONE" and self._last_local_scan:
                r = self._coerce_result_dict(self._last_local_scan, "scan result")
                if r.get("error"):
                    return f"Last scan failed: {r['error']}", "status"
                return (
                    f"Last scan complete: health {r.get('health_score')} · "
                    f"{r.get('anomaly_count')} anomalies."
                ), "status"
            if self._local_task_status == "SCAN_TREE_DONE" and self._last_local_scan:
                r = self._coerce_result_dict(self._last_local_scan, "folder scan result")
                if r.get("error"):
                    return f"Last folder scan failed: {r['error']}", "status"
                return f"Last folder scan complete: {r.get('datasets_found', 0)} dataset(s).", "status"
            return "No recent local scan.", "status"

        if "open" in t or "load" in t:
            # Accept a path on the next line or after keyword
            path = ""
            lines = text.splitlines()
            if len(lines) > 1:
                path = lines[-1].strip()
            if not path:
                m = re.search(r"(?:open|load)\s+(.+)$", text, re.I)
                if m:
                    path = m.group(1).strip()
            if path and os.path.isdir(path):
                try:
                    bus.emit(AppEvent(EventType.OPEN_DATASET, {"folder": path}, source="iris-local"))
                    self._remember_action("open", path, "opened by path")
                    self._remember_dataset(path)
                    return f"Opened dataset: {os.path.basename(path)}", "open"
                except Exception:
                    return f"Could not open: {path}", "open"
            if any(k in t for k in ("dataset", "folder", "data")):
                mode = self._active_mode()
                if mode == "raw":
                    return (
                        "You are in Raw Mode. Use the **Load** button (top-left in Raw tab) "
                        "to open a `.raw` file, or press `Ctrl+O`."
                    ), "raw_open_prompt"
                return (
                    "Use the picker below to choose your dataset folder "
                    "(the one with `.band0/.band1/...` files)."
                ), "open_prompt"

        return "", ""

    def _add_folder_picker_widget(self):
        widget = FolderPickWidget(parent=self._msg_container)
        widget.requested.connect(self._on_pick_folder)
        count = self._msg_layout.count()
        self._msg_layout.insertWidget(count - 1, widget)
        self._scroll_to_bottom()

    def _handle_command(self, text: str):
        lower = text.lower().strip()

        if lower.startswith("?apikey"):
            parts = text.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                save_api_key(parts[1].strip())
                self._add_message("iris", "API key saved.")
            else:
                self._prompt_api_key()
            return

        if lower.startswith("?mode"):
            parts = lower.split()
            if len(parts) > 1 and parts[1] in ("local", "smart", "online"):
                self._set_assist_mode(parts[1])
            else:
                self._add_message("iris", "Usage: ?mode local | ?mode smart | ?mode online", mode="local")
            return

        if lower == "?clear":
            for i in reversed(range(self._msg_layout.count())):
                item = self._msg_layout.itemAt(i)
                if item and item.widget():
                    item.widget().deleteLater()
            self._msg_layout.addStretch()
            idx = self._active_tab_index()
            self._history_by_tab[idx] = []
            self._history = self._history_by_tab[idx]
            return

        if lower == "?status":
            ctx = state.context_for_claude()
            key = get_api_key()
            key_status = "Set ✓" if key else "Not set — use ?apikey"
            self._add_message("iris", f"API key: {key_status}\n\n{ctx}")
            return

        if lower == "?help":
            self._add_message("iris",
                "Commands:\n"
                "  ?apikey [key]          — set API key\n"
                "  ?status                — show current app state\n"
                "  ?db                    — check Supabase DB connection\n"
                "  ?clear                 — clear chat\n"
                "  ?mode local|smart|online — set response mode\n"
                "\n"
                "Just type naturally for everything else:\n"
                "  'scan this dataset'\n"
                "  'open /path/to/folder'\n"
                "  'generate a report'")
            return

        if lower == "?db":
            self._start_pending("Checking Supabase connection…")
            import threading
            def _do_db_check():
                try:
                    msg = self._db_connection_status_text()
                except Exception as e:
                    msg = f"❌ DB check failed: {e}"
                self.dbCheckFinished.emit(msg)
            threading.Thread(target=_do_db_check, daemon=True).start()
            QTimer.singleShot(20000, self._db_check_timeout_guard)
            return

        self._add_message("iris", "Unknown command. Type ?help for options.")

    # ── Memory + Mode helpers ─────────────────────────────────────────────

    def _load_memory(self) -> dict:
        try:
            if os.path.exists(_MEMORY_PATH):
                with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "version": 1,
            "settings": {"assist_mode": "smart"},
            "last_active_dataset": None,
            "recent_datasets": [],
            "last_actions": [],
            "last_report": None,
        }

    def _save_memory(self):
        try:
            with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self._memory, f, indent=2)
        except Exception:
            pass

    def _apply_memory_settings(self):
        settings = self._memory.get("settings", {})
        mode = settings.get("assist_mode") or settings.get("report_mode")
        if mode == "hybrid":
            mode = "smart"
        if mode in ("local", "smart", "online"):
            self._assist_mode = mode

    def _remember_dataset(self, path: str):
        if not path:
            return
        try:
            names = [n.lower() for n in os.listdir(path)] if os.path.isdir(path) else []
        except Exception:
            names = []
        looks_like_dataset = any(re.search(r"\.band\d+$", n) for n in names)
        if not looks_like_dataset:
            return
        name = os.path.basename(path)
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        item = {"path": path, "name": name, "last_opened": now}
        recents = [r for r in self._memory.get("recent_datasets", [])
                   if r.get("path") != path]
        recents.insert(0, item)
        self._memory["recent_datasets"] = recents[:10]
        self._memory["last_active_dataset"] = item
        self._save_memory()

    def _prune_dataset_memory(self):
        def _is_band_dataset(path: str) -> bool:
            if not path or not os.path.isdir(path):
                return False
            try:
                names = [n.lower() for n in os.listdir(path)]
            except Exception:
                return False
            return any(re.search(r"\.band\d+$", n) for n in names)

        recents = []
        for item in self._memory.get("recent_datasets", []):
            path = (item.get("path") or "").strip()
            if _is_band_dataset(path):
                recents.append(item)
        self._memory["recent_datasets"] = recents[:10]

        last = self._memory.get("last_active_dataset") or {}
        if not _is_band_dataset((last.get("path") or "").strip()):
            self._memory["last_active_dataset"] = recents[0] if recents else None

        actions = []
        for entry in self._memory.get("last_actions", []):
            if entry.get("action") == "open":
                path = (entry.get("dataset") or "").strip()
                if path and not _is_band_dataset(path):
                    continue
            actions.append(entry)
        self._memory["last_actions"] = actions[:20]
        self._save_memory()

    def _remember_action(self, action: str, dataset_path: str, details: str):
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        entry = {
            "ts": now,
            "action": action,
            "dataset": dataset_path,
            "details": details,
        }
        actions = self._memory.get("last_actions", [])
        actions.insert(0, entry)
        self._memory["last_actions"] = actions[:20]
        if dataset_path:
            self._remember_dataset(dataset_path)
        self._save_memory()

    def _remember_report(self, dataset_path: str, summary: str):
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._memory["last_report"] = {
            "ts": now,
            "dataset": dataset_path,
            "summary": summary[:16000],
        }
        if dataset_path:
            self._remember_dataset(dataset_path)
        self._save_memory()

    def _looks_like_report_followup(self, text: str) -> bool:
        t = (text or "").lower().strip()
        if not t:
            return False
        if any(k in t for k in (
            "compare", "open", "close", "show", "hide", "set ", "change ",
            "scan", "reload", "refresh", "magnifier", "contrast", "histogram",
            "theme", "zoom", "frame", "tab",
        )):
            return False
        followup_phrases = (
            "what went wrong", "whats wrong", "what's wrong", "what is wrong",
            "the report", "that report", "above one", "above report",
            "the one you gave", "the one you generated", "you gave just now",
            "you generated", "explain this", "explain that", "what happened",
            "what does that mean", "why is that", "what is the issue",
            "what's the issue", "tell me what went wrong",
        )
        if any(p in t for p in followup_phrases):
            return True
        # Short, vague follow-ups like "why?" or "what happened?" often refer
        # to the previous answer, but longer task/control messages should not.
        short_followups = {
            "why", "why?", "how", "how?", "what happened", "what happened?",
            "explain", "explain?", "what does that mean", "what does that mean?",
        }
        return t in short_followups

    def _messages_for_llm(self, user_text: str, limit: int = 6) -> list:
        messages = []
        for m in self._tab_history()[-limit:]:
            messages.append({"role": m.get("role", ""), "content": m.get("content", "")})

        last_report = self._memory.get("last_report") or {}
        report_text = (last_report.get("summary") or "").strip()
        report_dataset = (last_report.get("dataset") or "").strip()
        active_folder = (self._resolve_active_folder() or "").strip()
        same_dataset = bool(report_dataset and active_folder and report_dataset == active_folder)
        if report_text and same_dataset and self._looks_like_report_followup(user_text):
            messages.append({
                "role": "assistant",
                "content": (
                    "Previous report context for follow-up question:\n\n"
                    + report_text
                ),
            })
        return messages

    def _maybe_append_llm_log_audit(self, report_text: str, folder: str) -> str:
        if not report_text or not folder:
            return report_text
        if self._response_mode != "ollama":
            return report_text
        if not self._has_local_llm():
            return report_text

        audit = tool_llm_log_audit(folder)
        if audit.get("error"):
            return report_text + "\n\n[LLM LOG AUDIT]\n" + audit["error"]

        audit_text = (audit.get("audit") or "").strip()
        if not audit_text:
            audit_text = "No additional concerns found."

        return report_text + "\n\n" + "\n".join([
            "══════════════════════════════════════════════════════════════════",
            "LLM LOG AUDIT (OLLAMA)",
            "══════════════════════════════════════════════════════════════════",
            audit_text,
        ])

    def _last_action_summary(self) -> str:
        actions = self._memory.get("last_actions", [])
        if not actions:
            return "None"
        a = actions[0]
        action = a.get("action", "unknown")
        details = a.get("details", "")
        return f"{action} ({details})".strip()

    def _get_last_dataset_path(self) -> str:
        last = self._memory.get("last_active_dataset") or {}
        return last.get("path") or ""

    def _should_load_last_dataset(self, text: str) -> bool:
        return any(k in text for k in (
            "load that data", "open that data", "load that dataset", "open that dataset",
            "load last", "open last", "resume last", "load previous", "open previous",
            "open recent", "load recent", "open last data", "load last data",
        ))

    def _sync_mode_buttons(self):
        if not hasattr(self, "_mode_btn_local"):
            return
        self._mode_btn_local.setChecked(self._assist_mode == "local")
        self._mode_btn_smart.setChecked(self._assist_mode == "smart")
        self._mode_btn_online.setChecked(self._assist_mode == "online")

    def _set_assist_mode(self, mode: str):
        if mode not in ("local", "smart", "online"):
            return
        self._assist_mode = mode
        self._memory.setdefault("settings", {})["assist_mode"] = self._assist_mode
        self._save_memory()
        self._sync_mode_buttons()

    def _set_local_task_status(self, status: str):
        self._local_task_status = status
        if hasattr(self, "_task_status_lbl") and self._task_status_lbl:
            label = "Scan: idle"
            if status == "SCAN_RUNNING":
                if self._scan_progress is not None:
                    label = f"Scan: running {self._scan_progress:.0f}%"
                else:
                    label = "Scan: running"
            elif status == "SCAN_DONE":
                label = "Scan: done"
                self._scan_progress = None
            elif status == "SCAN_TREE_RUNNING":
                label = "Scan: folder running"
            elif status == "SCAN_TREE_DONE":
                label = "Scan: folder done"
            self._task_status_lbl.setText(label)

    def _parse_progress_msg(self, msg: str):
        if (msg or "").startswith("PROGRESS:"):
            try:
                body = msg.split(":", 1)[1]
                pct_str, text = body.split("|", 1)
                return float(pct_str), text
            except Exception:
                return None, msg
        return None, msg

    def _set_scan_progress(self, pct: float):
        try:
            self._scan_progress = max(0.0, min(100.0, float(pct)))
        except Exception:
            self._scan_progress = None
        if self._local_task_status == "SCAN_RUNNING":
            self._set_local_task_status("SCAN_RUNNING")

    def _choose_scan_mode(self, text: str) -> str:
        """
        Default to quick (log/meta/ephem only). Use full only when user asks
        explicitly for pixel/frame-level analysis.
        """
        t = (text or "").lower()
        if any(k in t for k in ("pixel", "frame", "frames", "full scan", "full", "deep", "anomaly frames")):
            return "full"
        if "sample" in t:
            return "sample"
        return "quick"

    def _looks_like_folder_scan_request(self, text: str) -> bool:
        t = (text or "").lower().strip()
        if not t:
            return False
        strong_phrases = (
            "group scan", "scan group", "scan all", "scan tree", "scan folder tree",
            "scan root", "root folder", "scan root folder", "scan this root folder",
            "scan folder", "scan this folder", "scan folders", "few folders",
            "multiple folders", "many folders", "all datasets", "all folders",
            "scan datasets", "scan all datasets",
        )
        if any(p in t for p in strong_phrases):
            return True
        return (
            "scan" in t and
            any(k in t for k in ("root", "folder", "folders", "datasets", "dataset tree"))
        )

    def _looks_like_dataset_scan_request(self, text: str) -> bool:
        t = (text or "").lower().strip()
        if not t or "scan" not in t:
            return False
        if self._looks_like_folder_scan_request(text):
            return False
        return True

    def _update_scan_status_line(self, text: str, pct: Optional[float] = None):
        if not text:
            return
        now = time.time()
        if text == self._scan_status_last_text and (now - self._scan_status_last_ts) < 0.5:
            return
        self._scan_status_last_text = text
        self._scan_status_last_ts = now
        msg = f"Scan {pct:.0f}% — {text}" if pct is not None else f"Scan — {text}"
        if self._scan_status_label is None:
            self._scan_status_label = self._add_plain_label(msg)
        else:
            try:
                self._scan_status_label.setText(msg)
            except Exception:
                self._scan_status_label = self._add_plain_label(msg)

    def _finish_scan_status_line(self, text: str = "Scan complete."):
        if self._scan_status_label is None:
            return
        try:
            self._scan_status_label.setText(text)
        except Exception:
            pass
        lbl = self._scan_status_label
        self._scan_status_label = None
        try:
            QTimer.singleShot(4000, lambda: lbl.deleteLater() if lbl else None)
        except Exception:
            pass

    def _run_report_batch(self, folders: List[str], title: str = "Group report",
                          summary_rows: Optional[List[dict]] = None,
                          errors: Optional[List[dict]] = None):
        if not folders:
            self._add_message("iris", "No datasets to report.", mode="local")
            return
        def _do():
            reports = []
            errors = []
            for i, folder in enumerate(folders, start=1):
                name = os.path.basename(folder) if folder else "?"
                self.uiCall.emit(lambda n=name, idx=i, total=len(folders):
                                 self._update_scan_status_line(f"Report {idx}/{total}: {n}"))
                rep = self._coerce_result_dict(tool_generate_report(folder, enable_template_comparison=False), "report result")
                if rep.get("report"):
                    report_text = rep.get("report", "")
                    reports.append((name, folder, report_text))
                    self.uiCall.emit(lambda f=folder, r=report_text: self._remember_report(f, r))
                else:
                    errors.append(f"{name}: {rep.get('error','unknown error')}")

            def _post():
                # Build a single combined report
                width = 70
                lines = []
                lines.append("╔" + "═" * width + "╗")
                lines.append(f"  IRIS MULTI-DATASET REPORT")
                lines.append(f"  Datasets : {len(reports)} acquisitions")
                lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M')}")
                lines.append("╚" + "═" * width + "╝")
                lines.append("")
                if errors:
                    lines.append("  ⚠ Some datasets could not be scanned or reported:")
                    for e in errors:
                        if isinstance(e, dict):
                            lines.append(f"    - {os.path.basename(e.get('folder','?'))}: {e.get('error','?')}")
                        else:
                            lines.append(f"    - {e}")
                    lines.append("")
                if summary_rows:
                    lines.append("  FLEET SUMMARY")
                    lines.append("  " + "─" * 60)
                    lines.append("  Dataset                          Health   Findings")
                    lines.append("  " + "─" * 60)
                    for r in summary_rows:
                        name = r.get("name") or os.path.basename(r.get("folder","?"))
                        health = r.get("health_score")
                        if health is None:
                            health = "?"
                        else:
                            health = f"{health:.0f}"
                        crit = r.get("critical", 0)
                        warn = r.get("warnings", 0)
                        findings = f"{crit}🔴 {warn}🟡"
                        lines.append(f"  {name[:30]:<30}   {health:>5}   {findings}")
                    lines.append("")
                # Append each dataset report
                for idx, (name, folder, report_text) in enumerate(reports, start=1):
                    lines.append("═" * width)
                    lines.append(f"DATASET {idx} — {name}")
                    lines.append("═" * width)
                    lines.append(report_text)
                    lines.append("")
                self._add_message("iris", "\n".join(lines), mode="local")
                self._finish_scan_status_line("Report generation complete.")

            self.uiCall.emit(_post)
        threading.Thread(target=_do, daemon=True).start()

    def _build_group_summary_report(self, folders: List[str],
                                    summary_rows: Optional[List[dict]] = None,
                                    errors: Optional[List[dict]] = None) -> str:
        width = 70
        lines = []
        # Header (Fleet style)
        sat = None
        orbit = None
        for folder in folders:
            scan = state.get_scan_result(folder)
            meta = scan.meta_summary if scan else {}
            if meta:
                if meta.get("sat_id"):
                    sat = meta.get("sat_id") if sat in (None, meta.get("sat_id")) else "mixed"
                if meta.get("orbit_number"):
                    orbit = meta.get("orbit_number") if orbit in (None, meta.get("orbit_number")) else "mixed"
        sat = sat or "?"
        orbit = orbit or "?"

        lines.append("╔" + "═" * width + "╗")
        lines.append(f"  IRIS FLEET REPORT  —  {len(folders)} Acquisitions")
        lines.append(f"  Satellite: {sat}  |  Orbit: {orbit}")
        lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M')}")
        lines.append("╚" + "═" * width + "╝")
        lines.append("")

        # Aggregate counts + detail breakdown
        crit_total = warn_total = info_total = 0
        crit_msgs = {}
        warn_msgs = {}
        info_msgs = {}
        for folder in folders:
            scan = state.get_scan_result(folder)
            if not scan:
                continue
            ds_name = os.path.basename(folder)
            for f in scan.findings:
                sev = f.get("severity")
                msg = (f.get("message") or f.get("type") or "Unknown issue").strip()
                msg = msg.replace("\n", " ")
                if len(msg) > 110:
                    msg = msg[:107] + "..."
                target = None
                if sev == "CRITICAL":
                    crit_total += 1
                    target = crit_msgs
                elif sev == "WARNING":
                    warn_total += 1
                    target = warn_msgs
                elif sev == "INFO":
                    info_total += 1
                    target = info_msgs
                if target is not None:
                    entry = target.setdefault(msg, {"count": 0, "names": []})
                    entry["count"] += 1
                    if ds_name not in entry["names"]:
                        entry["names"].append(ds_name)

        def _render_issue_lines(bucket: dict, limit: int):
            for msg, meta in sorted(bucket.items(), key=lambda x: (-x[1]["count"], x[0]))[:limit]:
                names = meta.get("names", [])
                lines.append(f"    - {msg} (x{meta['count']})")
                if names:
                    prefix = "      Datasets: "
                    current = prefix
                    for idx, name in enumerate(names):
                        item = name if idx == 0 else f", {name}"
                        if len(current) + len(item) > width:
                            lines.append(current)
                            current = "      " + name
                        else:
                            current += item
                    if current.strip():
                        lines.append(current)

        lines.append(f"  🔴 CRITICAL   : {crit_total}")
        if crit_total:
            lines.append("    Details:")
            _render_issue_lines(crit_msgs, 6)
        else:
            lines.append("    Details: none")

        lines.append(f"  🟡 WARNING    : {warn_total}")
        if warn_total:
            lines.append("    Details:")
            _render_issue_lines(warn_msgs, 8)
        else:
            lines.append("    Details: none")

        lines.append(f"  ⚪ INFO       : {info_total}")
        if info_total:
            lines.append("    Details:")
            _render_issue_lines(info_msgs, 8)
        else:
            lines.append("    Details: none")
        lines.append("")

        # Session index
        lines.append("  Session index:")
        for i, folder in enumerate(folders, start=1):
            name = os.path.basename(folder)
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            proc = (log.get("procmode") or {}).get("decoded", {}) if log else {}
            tdi = (log.get("procmode") or {}).get("tdi_decoded", {}) if log else {}
            bs = (log.get("procmode") or {}).get("band_selection", {}) if log else {}
            fps = proc.get("fps_requested") or "?"
            tdi_mode = tdi.get("mode") or ("No-TDI" if tdi.get("tdi_on") is False else "?")
            bands = bs.get("active_count") or "?"
            frames = (log.get("frame_accounting") or {}).get("captured_count") or "?"
            test_pat = log.get("test_pattern")
            test_tag = "  TEST" if test_pat not in (None, 0) else ""
            date = (log.get("param_date_info") or {}).get("i28_date") or "?"
            lines.append(f"    [{i}] {name[:9]:<9}  {date}  {fps}fps  {tdi_mode}  {bands}-band  {frames} frames{test_tag}")
        lines.append("")

        if errors:
            lines.append("  ⚠ Some datasets could not be scanned:")
            for e in errors:
                if isinstance(e, dict):
                    lines.append(f"    - {os.path.basename(e.get('folder','?'))}: {e.get('error','?')}")
                else:
                    lines.append(f"    - {e}")
            lines.append("")

        # Section 1: Parameter file integrity
        lines.append("═" * width)
        lines.append("1. PARAMETER FILE INTEGRITY")
        lines.append("═" * width)
        all_14 = True
        i07_common = None
        rows = []
        arg_labels = {
            1: "OrbitID",
            2: "TaskID",
            3: "JsonID",
            4: "Date",
            5: "UTC Time",
            6: "Duration",
            7: "BandSelection",
            8: "TDI byte",
            9: "FPS",
            10: "ExposureTime",
            11: "Gain",
            12: "XShift",
            13: "Binning byte",
            14: "TDIYShift",
        }
        arg_issues = []
        for folder in folders:
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            raw_cnt = log.get("raw_arg_count")
            proc_cnt = log.get("proc_arg_count")
            if raw_cnt != 14 or proc_cnt != 14:
                all_14 = False
            pdi = log.get("param_date_info") or {}
            i07 = pdi.get("i07_date")
            i28 = pdi.get("i28_date")
            stale = pdi.get("discrepancy_days", 0)
            if i07:
                i07_common = i07 if i07_common in (None, i07) else "mixed"
            rows.append((os.path.basename(folder), i28 or "?", stale))
            if raw_cnt not in (None, 14):
                missing_positions = list(range(raw_cnt + 1, 15)) if raw_cnt < 14 else []
                missing_names = [arg_labels.get(pos, f"Arg{pos}") for pos in missing_positions]
                arg_issues.append(
                    f"{os.path.basename(folder)}: only {raw_cnt}/14 raw args found"
                    + (f" ; missing: {', '.join(missing_names)}" if missing_names else "")
                )
            elif raw_cnt == 14 and proc_cnt not in (None, 14):
                arg_issues.append(
                    f"{os.path.basename(folder)}: raw args OK (14/14) but processed count is {proc_cnt}/14"
                )
        lines.append(f"  {'All sessions received and processed all 14 arguments. ✅' if all_14 else '⚠ Some sessions missing arguments.'}")
        if arg_issues:
            lines.append("  Missing/processed detail:")
            for issue in arg_issues:
                lines.append(f"    - {issue}")
        if i07_common:
            lines.append(f"  I07 raw date: {i07_common} — present in ALL sessions.")
        lines.append("")
        lines.append("  Session     I28 corrected date/time         Stale by")
        lines.append("  " + "─" * 52)
        for name, i28, stale in rows:
            lines.append(f"  {name[:10]:<10}  {i28:<24}  {stale} days")
        lines.append("")

        # Section 2: Requested vs Applied
        lines.append("═" * width)
        lines.append("2. REQUESTED vs APPLIED")
        lines.append("═" * width)
        lines.append("  Session       FPS req→app        Exposure req→app        MaxExpTime")
        lines.append("  " + "─" * 60)
        for folder in folders:
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            proc = (log.get("procmode") or {}).get("decoded", {}) if log else {}
            p_app = log.get("parameters_applied", {}) if log else {}
            capinfo = log.get("capture_info", {}) if log else {}
            fps_r = proc.get("fps_requested")
            fps_a = p_app.get("FPS")
            exp_r = proc.get("exposure_time")
            exp_a = p_app.get("ExposureTime") or p_app.get("Exposure_Time")
            max_exp = capinfo.get("max_exp_time_us")
            name = os.path.basename(folder)
            lines.append(f"  {name[:10]:<10}  {str(fps_r):>6} → {str(fps_a):<6}    "
                         f"{str(exp_r):>5} → {str(exp_a):<7}       {str(round(max_exp, 1)) if isinstance(max_exp, (int, float)) else '—'}")
        lines.append("")

        # Section 3: Frame accounting
        lines.append("═" * width)
        lines.append("3. FRAME ACCOUNTING")
        lines.append("═" * width)
        lines.append("  Session       Expected   Captured   Drops")
        lines.append("  " + "─" * 44)
        for folder in folders:
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            fa = log.get("frame_accounting", {}) if log else {}
            exp = fa.get("total_frames_expected") or "?"
            cap = fa.get("captured_count") or "?"
            drops = fa.get("frames_lost") or 0
            name = os.path.basename(folder)
            lines.append(f"  {name[:10]:<10}  {exp!s:<9} {cap!s:<9} {drops}")
        lines.append("")

        # Section 4: Temperature
        lines.append("═" * width)
        lines.append("4. TEMPERATURE")
        lines.append("═" * width)
        lines.append("  Session       Sensor start   Sensor end   Core   Firmware   FW Status")
        lines.append("  " + "─" * 68)
        for folder in folders:
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            temps = log.get("temperatures", {}) if log else {}
            fw = log.get("firmware_version") or "?"
            s0 = temps.get("sensor_before_C", "?")
            s1 = temps.get("sensor_after_C", "?")
            c0 = temps.get("core_before_C", "?")
            fw_status = "BAD" if fw == "2.3.2" else "GOOD" if fw in ("2.2.2", "2.4.2") else "-"
            name = os.path.basename(folder)
            lines.append(f"  {name[:10]:<10}  {s0!s:<12} {s1!s:<11} {c0!s:<5} {fw:<8} {fw_status}")
        lines.append("")

        # Section 5: Trigger timing
        lines.append("═" * width)
        lines.append("5. TRIGGER TIMING")
        lines.append("═" * width)
        lines.append("  Session       Corrected wait (ms)    Outcome")
        lines.append("  " + "─" * 52)
        for folder in folders:
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            tt = log.get("trigger_timing", {}) if log else {}
            wait = tt.get("waiting_time_msec_2") or tt.get("waiting_time_msec") or "?"
            name = os.path.basename(folder)
            lines.append(f"  {name[:10]:<10}  {str(wait):<20}   ✅ OK")
        lines.append("")

        # Section 6: FPS stability
        lines.append("═" * width)
        lines.append("6. FPS STABILITY")
        lines.append("═" * width)
        lines.append("  Session       FPS applied   TimeDiff (ms)   Stability")
        lines.append("  " + "─" * 58)
        for folder in folders:
            scan = state.get_scan_result(folder)
            log = scan.log_summary if scan else {}
            fps = log.get("fps_stability", {}) if log else {}
            mean = fps.get("mean_fps") or "?"
            td = fps.get("time_diff_mean_ms") or "?"
            stab = fps.get("timing_stability") or "?"
            name = os.path.basename(folder)
            lines.append(f"  {name[:10]:<10}  {str(mean):<12} {str(td):<14} {stab}")
        lines.append("")

        return "\n".join(lines)

        return "\n".join(lines)

    def _coerce_result_dict(self, obj, label: str = "result") -> dict:
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (tuple, list)):
            for x in obj:
                if isinstance(x, dict):
                    return x
            print(f"[Iris] Unexpected {label} type: {type(obj).__name__}")
            return {"error": f"Unexpected {label} type: {type(obj).__name__}"}
        print(f"[Iris] Unexpected {label} type: {type(obj).__name__}")
        return {"error": f"Unexpected {label} type: {type(obj).__name__}"}

    def _should_enhance_with_api(self, kind: str, user_text: str) -> bool:
        if not getattr(self, "_api_enabled", True):
            return False
        if kind in ("report",):
            return True
        return self._user_wants_online(user_text)

    def _user_wants_online(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in (
            "analyze", "reason", "insight", "explain", "why", "deep", "online",
        ))

    def _compose_hybrid_prompt(self, user_text: str, local_text: str) -> str:
        """
        Build the prompt sent to Claude after a local scan/report completes.
        Enriches the structured report with knowledge-base and memory context
        from Supabase so Claude can write a narrative explanation with spec refs.
        """
        folder      = self._resolve_active_folder()
        dataset     = os.path.basename(folder) if folder else ""
        kb_context  = ""
        mem_context = ""

        # Derive a focused KB query from finding types in the scan result
        kb_query = user_text
        try:
            from .app_state import state as _state
            scan = _state.get_scan_result(folder) if folder else None
            if scan and scan.findings:
                top_types = list({
                    f.get("type", "") for f in scan.findings
                    if f.get("severity") in ("CRITICAL", "WARNING")
                })[:4]
                if top_types:
                    kb_query = f"{dataset} {' '.join(top_types)}".strip()
        except Exception:
            pass

        # 1. Knowledge-base: sensor specs, manuals, SOPs
        try:
            from .retrieval import search_knowledge
            kb = search_knowledge(query=kb_query, top_k=4, threshold=0.28)
            if kb.get("total_found", 0) > 0:
                kb_context = kb["context_text"]
                print(f"[Iris KB] {kb['total_found']} chunks fetched for query: {kb_query!r}")
            else:
                print(f"[Iris KB] 0 chunks returned for query: {kb_query!r}")
        except Exception as _kb_err:
            print(f"[Iris KB] search_knowledge failed: {_kb_err}")

        # 2. Long-term memory: past findings for this dataset
        try:
            from .retrieval import recall_memory
            mem = recall_memory(
                query=f"{dataset} {user_text}".strip(),
                top_k=3,
                threshold=0.35,
            )
            if mem.get("total_found", 0) > 0:
                mem_context = mem["context_text"]
                print(f"[Iris Mem] {mem['total_found']} memories recalled")
            else:
                print("[Iris Mem] no memories found")
        except Exception as _mem_err:
            print(f"[Iris Mem] recall_memory failed: {_mem_err}")

        # Assemble: KB -> memory -> structured report -> instruction
        parts = []
        if kb_context:
            parts.append(kb_context)
        if mem_context:
            parts.append(mem_context)
        parts.append(
            "Structured scan report (Iris rule engine output — factual base):\n\n"
            + local_text
        )
        context_block = "\n\n---\n\n".join(parts)

        return (
            f"{context_block}\n\n"
            "Using the structured report and knowledge-base excerpts above, "
            "write a clear narrative report for this dataset. "
            "Explain what each finding means in plain language, "
            "cross-reference the knowledge base where relevant, "
            "state whether each issue is confirmed or requires further investigation, "
            "and end with prioritised recommended actions. "
            "Do not re-scan. Do not repeat raw numbers without context."
        )

    def _has_local_llm(self) -> bool:
        try:
            from .ollama import available_models
            return bool(available_models())
        except Exception:
            return False

    def _run_local(self, user_text: str):
        """Local mode: rule-first, then local LLM."""
        local, kind = self._try_local_response(user_text)
        if local:
            self._set_mode_indicator("local")
            self._add_message("iris", local, mode="local")
            if kind == "open_prompt":
                self._add_folder_picker_widget()
            return

        if not self._has_local_llm():
            self._set_mode_indicator("disabled")
            self._add_message("iris", "No local LLM or API available. Start Ollama or enable API.", mode="local")
            return

        # Ollama path
        messages = self._messages_for_llm(user_text)

        self._start_pending("thinking…")
        self._response_mode = "ollama"
        self._response_model = ""

        worker = IrisSmartWorker(
            messages, "", self._resolve_active_folder(), allow_api=False, parent=self
        )
        worker.token.connect(self._on_token)
        worker.tool_call.connect(self._on_tool_call)
        worker.completed.connect(self._on_response)
        worker.failed.connect(self._on_error)
        worker.needs_api.connect(
            lambda _q: self._add_message(
                "iris",
                "API escalation is disabled in Local mode. Switch to Online if needed.",
                mode="local",
            )
        )
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    def _run_smart(self, user_text: str):
        """Smart mode: rule-first, then LLM/API."""
        local, kind = self._try_local_response(user_text)
        if local:
            self._set_mode_indicator("local")
            self._add_message("iris", local, mode="local")
            if kind == "open_prompt":
                self._add_folder_picker_widget()
            return

        if self._has_local_llm():
            # Ollama path: casual chat + analysis questions + API escalation
            messages = self._messages_for_llm(user_text)

            self._start_pending("thinking…")
            self._response_mode = "ollama"
            self._response_model = ""

            worker = IrisSmartWorker(messages, get_api_key(), self._resolve_active_folder(), parent=self)
            worker.token.connect(self._on_token)
            worker.tool_call.connect(self._on_tool_call)
            worker.completed.connect(self._on_response)
            worker.failed.connect(self._on_error)
            worker.needs_api.connect(self._on_smart_escalate)
            worker.finished.connect(lambda: self._cleanup_worker(worker))
            self._workers.append(worker)
            worker.start()
            return

        # No local LLM: try API directly
        if getattr(self, "_api_enabled", True) and get_api_key():
            self._set_mode_indicator("api")
            self._run_agent(user_text)
            return

        # No LLM + no API
        self._set_mode_indicator("disabled")
        self._add_message("iris", "No local LLM or API available. Start Ollama or set API key.", mode="local")
        return

    def _on_smart_escalate(self, question: str):
        """Smart mode decided to escalate to Claude API."""
        self._set_mode_indicator("api")
        self._run_agent(question)

    def _run_agent(self, user_text: str, _injected_prompt: str = ""):
        """
        Send a message to the Claude API worker.

        user_text:        The text shown to the user (added to history, displayed).
        _injected_prompt: If provided, used as the actual content of the final
                          user message sent to Claude instead of user_text.
                          Used by scan completion handlers to pass KB-enriched
                          prompts without polluting chat history with raw context.
        """
        if not getattr(self, "_api_enabled", True):
            local, kind = self._try_local_response(user_text)
            if local:
                self._set_mode_indicator("local")
                self._add_message("iris", local, mode="local")
                if kind == "open_prompt":
                    self._add_folder_picker_widget()
                return
            if self._has_local_llm():
                self._run_local(user_text)
                return
            self._set_mode_indicator("disabled")
            self._add_message("iris", "API is disabled. Try a local command like: scan, report, go to frame 21.", mode="local")
            return
        key = get_api_key()
        if not key:
            local, kind = self._try_local_response(user_text)
            if local:
                self._set_mode_indicator("local")
                self._add_message("iris", local, mode="local")
                if kind == "open_prompt":
                    self._add_folder_picker_widget()
                return
            if self._has_local_llm():
                self._run_local(user_text)
                return
            key = self._prompt_api_key()
        if not key:
            self._add_message("iris", "Please set your API key first: /apikey sk-ant-...")
            return

        # Build message list from tab history (conversation context).
        # If _injected_prompt is provided, replace the last user turn with it —
        # this carries KB + memory context to Claude without polluting chat history.
        messages       = []
        last_user_text = None
        for m in self._tab_history()[-12:]:
            role    = m.get("role", "")
            content = m.get("content", "")
            if role == "user" and content == last_user_text:
                continue
            if role == "user":
                last_user_text = content
            messages.append({"role": role, "content": content})

        if _injected_prompt:
            # Replace last user message content with the enriched prompt.
            # This is the KB-enriched version; the visible history still shows
            # the short human-readable message.
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i] = {"role": "user", "content": _injected_prompt}
                    break
            else:
                messages.append({"role": "user", "content": _injected_prompt})

        # Capture screenshot only when needed (visual questions or explicit toggle)
        img = None
        if self._screenshot_btn.isChecked() or self._should_capture_screenshot(user_text):
            img = self._capture_screenshot()
        self._screenshot_btn.setChecked(False)

        # Pending indicator
        self._start_pending("thinking…")
        self._response_mode = "api"
        self._response_model = "claude"

        _folder = self._resolve_active_folder()
        worker = IrisAgentWorker(key, messages, img, folder=_folder, parent=self)
        worker.token.connect(self._on_token)
        worker.tool_call.connect(self._on_tool_call)
        worker.completed.connect(self._on_response)
        worker.failed.connect(self._on_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._workers.append(worker)
        worker.start()

    # ── Agent signal handlers ──────────────────────────────────────────────

    def _on_token(self, tok: str):
        if self._stream_bubble is None:
            self._remove_pending()
            self._stream_bubble = self._add_message(
                "iris", "", mode=self._response_mode, model_name=self._response_model
            )
        # Update the label inside the bubble with accumulated text
        if self._stream_bubble:
            labels = self._stream_bubble.findChildren(QLabel)
            text_labels = [l for l in labels
                           if l.text() not in ("Iris", "You") and l.wordWrap()]
            if text_labels:
                last = text_labels[-1]
                # Re-render with updated text including frame number styling
                current = getattr(last, "_raw_text", "")
                current += tok
                last._raw_text = current
                last.setText(MessageBubble._render_frame_numbers(current))
        self._scroll_to_bottom()

    def _on_tool_call(self, name: str):
        if (name or "").startswith("model:"):
            self._response_model = name.split(":", 1)[1].strip()
            return
        label = _TOOL_LABELS.get(name, f"⚙️ {name}…")
        if self._pending_bubble is None:
            self._start_pending(label)
        else:
            self._set_pending_text(label)

    def _on_response(self, text: str):
        mode = self._response_mode or "rule"
        stream = self._stream_bubble
        self._stream_bubble = None
        self._stop_pending()
        self._set_mode_indicator("api" if mode == "api" else "local")

        now = time.time()
        clean_text = (text or "").strip()
        if (clean_text and clean_text == self._last_assistant_text
                and (now - self._last_assistant_ts) < 2.0):
            self._scroll_to_bottom()
            return

        # Detect CHOICE block — Iris embeds this to trigger interactive picker
        # Format: [CHOICE:multi|single|prompt="..."|opts="A||B||C"]
        choice_match = re.search(
            r'\[CHOICE:(?P<mode>multi|single)\|prompt="(?P<prompt>[^"]+)"\|opts="(?P<opts>[^"]+)"\]',
            text)

        if choice_match:
            # Strip the marker from the visible text
            visible_text = text[:choice_match.start()].strip()
            if visible_text:
                if stream:
                    labels = stream.findChildren(QLabel)
                    text_labels = [l for l in labels
                                   if l.text() not in ("Iris", "You") and l.wordWrap()]
                    if text_labels:
                        last = text_labels[-1]
                        last._raw_text = visible_text
                        last.setText(MessageBubble._render_frame_numbers(visible_text))
                    else:
                        self._add_message("iris", visible_text, mode=mode, model_name=self._response_model)
                else:
                    self._add_message("iris", visible_text, mode=mode, model_name=self._response_model)

            mode    = choice_match.group("mode")
            prompt  = choice_match.group("prompt")
            options = choice_match.group("opts").split("||")

            widget = ChoiceWidget(
                prompt  = prompt,
                options = [o.strip() for o in options if o.strip()],
                multi   = (mode == "multi"),
                parent  = self._msg_container,
            )
            widget.confirmed.connect(self._on_choice_confirmed)
            count = self._msg_layout.count()
            self._msg_layout.insertWidget(count - 1, widget)
            self._scroll_to_bottom()
        else:
            if stream:
                labels = stream.findChildren(QLabel)
                text_labels = [l for l in labels
                               if l.text() not in ("Iris", "You") and l.wordWrap()]
                if text_labels:
                    last = text_labels[-1]
                    last._raw_text = text
                    last.setText(MessageBubble._render_frame_numbers(text))
                else:
                    self._add_message("iris", text, mode=mode, model_name=self._response_model)
            else:
                self._add_message("iris", text, mode=mode, model_name=self._response_model)

        if self._should_show_folder_picker(text):
            self._add_folder_picker_widget()

        self._append_tab_history("assistant", text)
        self._last_assistant_text = clean_text
        self._last_assistant_ts = now
        self._scroll_to_bottom()

    def _on_choice_confirmed(self, reply: str):
        """User picked from a ChoiceWidget — inject as a new user message."""
        self._add_message("user", reply)
        self._run_agent(reply)

    def _on_error(self, err: str):
        self._stream_bubble = None
        self._stop_pending()
        self._set_mode_indicator("idle")
        mode = self._response_mode if self._response_mode in ("api", "ollama") else "rule"
        self._add_message("iris", f"⚠ {err}", mode=mode)

    def _cleanup_worker(self, worker):
        try:
            self._workers.remove(worker)
        except ValueError:
            pass
        worker.deleteLater()

    # ── Message helpers ────────────────────────────────────────────────────

    def _add_message(
        self, role: str, text: str, streaming: bool = False, mode: str = "", model_name: str = ""
    ) -> MessageBubble:
        ts = time.strftime("%H:%M", time.localtime())
        bubble = MessageBubble(
            text,
            role,
            mode=mode,
            model_name=model_name,
            timestamp=ts,
            parent=self._msg_container,
        )
        # Insert before the trailing stretch
        count = self._msg_layout.count()
        self._msg_layout.insertWidget(count - 1, bubble)
        if role == "user":
            self._append_tab_history("user", text)
        self._trim_messages()
        self._scroll_to_bottom()
        return bubble

    def _add_plain_label(self, text: str) -> QLabel:
        lbl = QLabel(text, self._msg_container)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 9px; padding: 4px 8px;")
        count = self._msg_layout.count()
        self._msg_layout.insertWidget(count - 1, lbl)
        self._scroll_to_bottom()
        return lbl

    def _set_mode_indicator(self, mode: str):
        if not hasattr(self, "_mode_dot") or self._mode_dot is None:
            return
        if not getattr(self, "_api_enabled", True):
            self._last_mode = "disabled"
            self._mode_dot.setStyleSheet("color: #e53935; font-size: 10px; border: none; background: transparent;")
            self._mode_dot.setToolTip("API disabled (click to enable)")
            return
        self._last_mode = mode
        if mode == "local":
            self._mode_dot.setStyleSheet("color: #43a047; font-size: 10px;")
            self._mode_dot.setToolTip("Handled locally (click to disable API)")
        elif mode == "api":
            self._mode_dot.setStyleSheet(f"color: {_ACCENT_H}; font-size: 10px; border: none; background: transparent;")
            self._mode_dot.setToolTip("Using API (click to disable)")
        else:
            self._mode_dot.setStyleSheet("color: #6b6b6b; font-size: 10px; border: none; background: transparent;")
            self._mode_dot.setToolTip("Idle (click to disable API)")

    def _toggle_api_enabled(self):
        self._api_enabled = not getattr(self, "_api_enabled", True)
        if not self._api_enabled:
            self._set_mode_indicator("disabled")
        else:
            self._set_mode_indicator(self._last_mode or "idle")

    def _tick_pending_spinner(self):
        if not self._pending_bubble:
            return
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        self._pending_bubble.setText(f"Iris: {frame} {self._pending_text}")

    def _start_pending(self, text: str):
        self._pending_text = text
        self._spinner_idx = 0
        # Keep input OPEN while Iris is thinking — user can type the next
        # message without waiting. The worker queues naturally since messages
        # are processed sequentially via the _workers list.
        self._set_input_locked(False)
        if self._pending_bubble is None:
            self._pending_bubble = self._add_plain_label("")
        self._tick_pending_spinner()
        if not self._pending_timer.isActive():
            self._pending_timer.start()

    def _set_pending_text(self, text: str):
        self._pending_text = text
        self._tick_pending_spinner()

    def _db_connection_status_text(self) -> str:
        cfg = {}
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "iris_knowledge.cfg")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#") or "=" not in s:
                            continue
                        k, v = s.split("=", 1)
                        cfg[k.strip()] = v.strip()
        except Exception:
            pass

        url = os.environ.get("SUPABASE_URL") or cfg.get("supabase_url", "")
        key = os.environ.get("SUPABASE_SERVICE_KEY") or cfg.get("supabase_service_key", "")
        if not url or not key:
            return (
                "❌ Supabase not configured.\n"
                "Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or update iris_knowledge.cfg)."
            )

        checks = []
        all_ok = True
        base = url.rstrip("/") + "/rest/v1/"
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }

        for table in ("iris_knowledge", "iris_memory"):
            try:
                qs = urllib.parse.urlencode({"select": "id", "limit": 1})
                req = urllib.request.Request(base + table + "?" + qs, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=8) as resp:
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}")
                checks.append(f"  ✓ {table}")
            except Exception as e:
                all_ok = False
                checks.append(f"  ✗ {table}: {e}")

        if all_ok:
            return "✅ Supabase connected.\nTables reachable:\n" + "\n".join(checks)
        return "⚠ Supabase reachable, but some tables failed:\n" + "\n".join(checks)

    def _on_db_check_finished(self, msg: str):
        self._stop_pending()
        self._add_message("iris", msg)

    def _db_check_timeout_guard(self):
        if self._pending_bubble and "Checking Supabase connection" in (self._pending_text or ""):
            self._stop_pending()
            self._add_message("iris", "⚠ DB check timed out. Check internet/SUPABASE_URL and try `?db` again.")

    def _run_ui_call(self, fn):
        try:
            fn()
        except Exception as e:
            self._stop_pending()
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            self._add_message("iris", f"⚠ UI callback error: {e}", mode="rule")

    def _stop_pending(self):
        if self._pending_timer.isActive():
            self._pending_timer.stop()
        if self._pending_bubble:
            try:
                self._msg_layout.removeWidget(self._pending_bubble)
                self._pending_bubble.deleteLater()
            except Exception:
                pass
            self._pending_bubble = None
        self._pending_text = ""
        self._set_input_locked(False)

    def _set_input_locked(self, locked: bool):
        self._input_locked = bool(locked)
        try:
            self._input.setEnabled(not self._input_locked)
            self._send_btn.setEnabled(not self._input_locked)
            self._input.setPlaceholderText("Iris is processing…" if self._input_locked else "Ask Iris…")
        except Exception:
            pass

    def _remove_pending(self):
        self._stop_pending()

    def _trim_messages(self, max_visible: int = 30):
        """Remove oldest bubbles if too many."""
        count = self._msg_layout.count() - 1  # -1 for stretch
        while count > max_visible:
            item = self._msg_layout.itemAt(0)
            if item and item.widget():
                item.widget().deleteLater()
            count -= 1

    def _active_tab_index(self) -> int:
        try:
            t = state.active_tab
            if t is not None:
                return int(getattr(t, "tab_index", -1))
        except Exception:
            pass
        return -1

    def _tab_history(self) -> list:
        idx = self._active_tab_index()
        if idx not in self._history_by_tab:
            self._history_by_tab[idx] = []
        return self._history_by_tab[idx]

    def _append_tab_history(self, role: str, content: str):
        hist = self._tab_history()
        hist.append({"role": role, "content": content})
        if len(hist) > 60:
            del hist[:-60]
        self._history = hist

    def _sync_last_dataset_from_state(self):
        folder = ""
        try:
            tab = state.active_tab
            if tab:
                folder = (getattr(tab, "folder", "") or "").strip()
        except Exception:
            folder = ""
        if folder:
            self._remember_dataset(folder)

    def _on_dataset_loaded_event(self, event: AppEvent):
        try:
            folder = (event.payload.get("folder") or "").strip()
        except Exception:
            folder = ""
        if folder:
            QTimer.singleShot(0, lambda f=folder: self._remember_dataset(f))

    def _on_tab_activated_event(self, event: AppEvent):
        QTimer.singleShot(0, self._sync_last_dataset_from_state)

    def _resolve_active_folder(self) -> str:
        """
        Resolve the folder for the currently visible/active tab.
        Falls back through state, widget introspection, then recent session folders.
        """
        def _to_folder(path_val: str) -> str:
            p = (path_val or "").strip()
            if not p:
                return ""
            if os.path.isdir(p):
                return p
            if os.path.isfile(p):
                d = os.path.dirname(p)
                return d if os.path.isdir(d) else ""
            return ""

        try:
            f = _to_folder(state.active_folder or "")
            if f:
                return f
        except Exception:
            pass

        try:
            t = state.active_tab
            f = _to_folder((getattr(t, "folder", "") or "").strip()) if t else ""
            if f:
                return f
        except Exception:
            pass

        try:
            tw = getattr(self._host, "tab_widget", None)
            w = tw.currentWidget() if tw is not None else None
            if w is not None:
                candidates = [w]
                try:
                    candidates.extend(w.findChildren(QWidget))
                except Exception:
                    pass
                for obj in candidates:
                    for attr in ("current_folder", "folder", "last_file_path"):
                        f = _to_folder(getattr(obj, attr, "") or "")
                        if f:
                            return f
        except Exception:
            pass

        try:
            for f in state.session_folders():
                if f and os.path.isdir(f):
                    return f
        except Exception:
            pass
        return ""

    def _resolve_loaded_folder(self) -> str:
        """
        Resolve only a genuinely loaded/open dataset.
        Do not fall back to session history or cache, because commands like
        'report' should match what the UI currently has loaded.
        """
        def _to_folder(path_val: str) -> str:
            p = (path_val or "").strip()
            if not p:
                return ""
            if os.path.isdir(p):
                return p
            if os.path.isfile(p):
                d = os.path.dirname(p)
                return d if os.path.isdir(d) else ""
            return ""

        try:
            active = state.active_tab
            f = _to_folder(getattr(active, "folder", "") or "")
            if f:
                return f
        except Exception:
            pass

        try:
            tabs = tool_get_app_state().get("open_tabs", [])
            active = next((tab for tab in tabs if tab.get("is_active")), None)
            f = _to_folder((active or {}).get("folder", ""))
            if f:
                return f
        except Exception:
            pass

        try:
            tw = getattr(self._host, "tab_widget", None)
            w = tw.currentWidget() if tw is not None else None
            if w is not None:
                candidates = [w]
                try:
                    candidates.extend(w.findChildren(QWidget))
                except Exception:
                    pass
                for obj in candidates:
                    for attr in ("current_folder", "folder", "last_file_path"):
                        f = _to_folder(getattr(obj, attr, "") or "")
                        if f:
                            return f
        except Exception:
            pass

        return ""

    def _scroll_to_bottom(self):
        QTimer.singleShot(30, lambda: (
            self._scroll.verticalScrollBar().setValue(
                self._scroll.verticalScrollBar().maximum()
            ) if self._scroll else None
        ))

    # ── Screenshot ─────────────────────────────────────────────────────────────

    def _capture_screenshot(self) -> Optional[bytes]:
        """
        Capture the full main window so Iris can see the active frame/histogram,
        sidebar state, open tabs, and the current UI context — everything the
        user is looking at right now.
        """
        try:
            from PyQt5.QtCore import QBuffer, QIODevice
            from PyQt5.QtGui import QPixmap
            # Grab the entire main window (self._host), not just the active tab.
            # This gives Iris full visual context: image viewer, histogram,
            # band controls, open dataset tabs, and the chat itself.
            pixmap: QPixmap = self._host.grab()
            buf = QBuffer()
            buf.open(QIODevice.WriteOnly)
            pixmap.save(buf, "PNG")
            return bytes(buf.data())
        except Exception as e:
            print(f"[Iris] Screenshot error: {e}")
            return None

    # ── API key prompt ─────────────────────────────────────────────────────

    def _prompt_api_key(self) -> str:
        if self._api_prompted:
            return ""
        self._api_prompted = True
        try:
            key, ok = QInputDialog.getText(
                self._host, "Iris — API Key",
                "Enter your Anthropic API key (sk-ant-…):",
                QLineEdit.Password)
            if ok and key.strip():
                save_api_key(key.strip())
                self._api_prompted = False
                return key.strip()
        except Exception:
            pass
        self._api_prompted = False
        return ""

    # ── Layout ─────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self._host and event.type() in (
                QEvent.Resize, QEvent.Move, QEvent.Show):
            self._reposition()
        return super().eventFilter(obj, event)

    def _reposition(self):
        try:
            hr  = self._host.rect()
            # Scale panel width with window size to avoid cramped wrapping
            w = min(max(self._width, int(hr.width() * 0.28)), 460)
            margin = 8
            x   = hr.width() - w - margin
            # Try to position below the toolbar
            y   = 40
            h   = hr.height() - y - margin - 60
            self.setGeometry(x, y, w, max(200, h))
        except Exception:
            pass
