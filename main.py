from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QGridLayout,
    QLabel, QTabWidget, QSizePolicy, QToolButton, QShortcut, QTabBar, QProgressDialog,
    QDialog, QRadioButton, QButtonGroup, QMessageBox, QCheckBox, QSplitter, QToolTip
)
from PyQt5.QtCore import Qt, QTimer, QCoreApplication, QThread, QObject, QEvent, QPoint, qInstallMessageHandler
from PyQt5.QtGui import QPalette, QColor, QTransform, QKeySequence, QCloseEvent, QCursor, QHelpEvent
from types import SimpleNamespace
import sys
import time
from band_app import BandStitchProApp
import gc
import json
import os
import psutil
from ui_components import *
from utils import TerminalWidget
from help_tab import create_help_tab
try:
    from iris2 import Iris
except Exception as _e:
    print(f"[Iris2] Not available: {_e}")
    Iris = None
try:
    from Live_Display_AppVZ import VideoModeHandler
except ImportError:
    VideoModeHandler = None  
try:
    from video_mode import PlaybackApp
except ImportError:
    PlaybackApp = None
try:
    from raw_mode import RawViewer
except ImportError:
    RawViewer = None
try:
    from tiled_viewer import TiledDisplay
except Exception as e:
    import traceback
    print(f"[TiledDisplay Import Error] {e}", file=sys.stderr)
    traceback.print_exc()
    TiledDisplay = None
try:
    from editor_tab import EditorTab
except Exception as e:
    import traceback
    print(f"[EditorTab Import Error] {e}", file=sys.stderr)
    traceback.print_exc()
    EditorTab = None


def _qt_message_handler(mode, context, message):
    # Suppress known benign startup warning emitted by some backends.
    if "QSocketNotifier: Can only be used with threads started with QThread" in message:
        return
    print(message, file=sys.stderr)

# --- Light/Dark palette helpers  ---
def set_light_palette(app: QApplication):
    try:
        p = QPalette()
        p.setColor(QPalette.Window, QColor(245, 245, 247))  # Soft off-white background
        p.setColor(QPalette.WindowText, QColor(33, 33, 33))  # Dark gray text
        p.setColor(QPalette.Base, QColor(255, 255, 255))  # Pure white for inputs
        p.setColor(QPalette.AlternateBase, QColor(240, 240, 242))  # Subtle alternate rows
        p.setColor(QPalette.Text, QColor(33, 33, 33))
        p.setColor(QPalette.Button, QColor(255, 255, 255))
        p.setColor(QPalette.ButtonText, QColor(33, 33, 33))  # white text on teal buttons
        p.setColor(QPalette.Highlight, QColor(38, 166, 154))    # teal 400
        p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        p.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        p.setColor(QPalette.ToolTipText, QColor(33, 33, 33))
        app.setPalette(p)

        # Medium pale teal buttons
        stylesheet = """
            * { font-family: 'Segoe UI', Arial, sans-serif; font-size:11px; }
            QPushButton {
                background-color: #26A69A;   /* medium pale teal */
                color: white;
                border: none;
                padding: 2px 6px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #00897B;   /* slightly darker teal on hover */
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
            }
            QToolButton[class="tab-close"] {
                background-color: transparent;
                color: #757575;  /* Neutral gray matching light theme */
                font-size: 11pt;
                font-weight: bold;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QToolButton[class="tab-add"] {
                background: transparent;
                color: #616161;
                font-size: 11pt;
                font-weight: 600;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QToolButton[class="tab-add"]:hover {
                color: #00897B;
            }
            QToolButton[class="tab-nav"] {
                background: transparent;
                color: #555;
                font-size: 11pt;
                font-weight: 700;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QToolButton[class="tab-nav"]:hover {
                color: #222;
            }
            QToolButton[class="tab-nav"]:disabled {
                color: #bcbcbc;
            }
            QTabBar QToolButton#ScrollLeftButton,
            QTabBar QToolButton#ScrollRightButton {
                border: none;
                background: transparent;
                margin: 0px;
                padding: 0px;
                color: transparent;
            }
            QToolButton[class="tab-close"]:hover {
                background-color: #ffebee;  /* Soft red bg for hover */
                color: #f44336;  /* Trendy red text on hover */
                border-radius: 7px;
            }
            """
        app.setStyleSheet(stylesheet)
    except Exception as e:
        print(f"Error setting light palette: {e}")


def set_dark_palette(app: QApplication):
    try:
        app.setStyle('Fusion')  # Fusion plays nicely with palette tweaks
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.black)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(44, 44, 44))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(148, 36, 227))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        app.setPalette(dark_palette)

        dark_stylesheet = """
        * { font-family: 'Segoe UI', Arial, sans-serif; font-size:11px; }
         QGraphicsView, QGraphicsView::viewport {
            background-color: black;
        }
        QToolButton[class="tab-close"] {
            background-color: transparent;
            color: #bdbdbd;  /* Neutral gray matching dark theme */
            font-size: 11pt;
            font-weight: bold;
            border: none;
            padding: 0px;
            margin: 0px;
        }
        QToolButton[class="tab-add"] {
            background: transparent;
            color: #cfcfcf;
            font-size: 11pt;
            font-weight: 600;
            border: none;
            padding: 0px;
            margin: 0px;
        }
        QToolButton[class="tab-add"]:hover {
            color: #4db6ac;
        }
        QToolButton[class="tab-nav"] {
            background: transparent;
            color: #e0e0e0;
            font-size: 11pt;
            font-weight: 700;
            border: none;
            padding: 0px;
            margin: 0px;
        }
        QToolButton[class="tab-nav"]:hover {
            color: #fff;
        }
        QToolButton[class="tab-nav"]:disabled {
            color: #8f8f8f;
        }
        QTabBar QToolButton#ScrollLeftButton,
        QTabBar QToolButton#ScrollRightButton {
            border: none;
            background: transparent;
            margin: 0px;
            padding: 0px;
            color: transparent;
        }
        QToolButton[class="tab-close"]:hover {
            color: #b53c44;  /* Trendy red text on hover */
            border-radius: 7px;
        }
        """
        app.setStyleSheet(dark_stylesheet)
    except Exception:
        pass


class ModeSelectionDialog(QDialog):
    """Simple dialog to select which mode to open."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Display Mode")
        self.setFixedSize(520, 360)
        layout = QVBoxLayout()
        self.setLayout(layout)

        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        select_page = QWidget()
        select_layout = QVBoxLayout()
        select_page.setLayout(select_layout)
        tabs.addTab(select_page, "Select Mode")

        # Title
        title = QLabel("Select Data Format")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        select_layout.addWidget(title)

        # Mode selection group
        self.mode_group = QButtonGroup(self)
        
        self.band_radio = QRadioButton("Band Mode")
        self.mode_group.addButton(self.band_radio)
        select_layout.addWidget(self.band_radio)

        self.raw_radio = QRadioButton("Raw Mode")
        self.mode_group.addButton(self.raw_radio)
        select_layout.addWidget(self.raw_radio)

        self.video_radio = QRadioButton("Video Mode")
        self.mode_group.addButton(self.video_radio)
        select_layout.addWidget(self.video_radio)

        self.live_radio = QRadioButton("Live Display Mode")
        self.mode_group.addButton(self.live_radio)
        select_layout.addWidget(self.live_radio)

        self.tiled_radio = QRadioButton("Tiled Mode")
        self.mode_group.addButton(self.tiled_radio)
        select_layout.addWidget(self.tiled_radio)

        # Default to Band Mode
        self.band_radio.setChecked(True)
        select_layout.addStretch()

        # Onboarding help tab
        try:
            onboarding_help = create_help_tab(main_app=parent, mode="onboarding")
            tabs.addTab(onboarding_help, "Help")
        except Exception as e:
            fallback = QLabel(f"Help unavailable: {e}")
            fallback.setWordWrap(True)
            tabs.addTab(fallback, "Help")

        # Buttons
        buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setToolTip("Open selected mode")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Close mode selector")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        layout.addLayout(buttons)

    def get_selected_mode(self):
        """Return the selected mode: 'band', 'raw', 'video', 'live', or 'tiled'."""
        if self.band_radio.isChecked():
            return "band"
        elif self.raw_radio.isChecked():
            return "raw"
        elif self.video_radio.isChecked():
            return "video"
        elif self.live_radio.isChecked():
            return "live"
        elif self.tiled_radio.isChecked():
            return "tiled"
        return "band"  # Fallback

class _TooltipFilter(QObject):
    """Force tooltips to appear at current cursor target and hide stale ones."""
    def eventFilter(self, watched, event):
        et = event.type()
        if et == QEvent.Enter:
            QToolTip.hideText()
        elif et == QEvent.Leave:
            QToolTip.hideText()
        elif et == QEvent.ToolTip:
            try:
                tip = watched.toolTip() if hasattr(watched, "toolTip") else ""
            except Exception:
                tip = ""
            if tip:
                if isinstance(event, QHelpEvent):
                    pos = event.globalPos()
                else:
                    pos = QCursor.pos()
                QToolTip.showText(pos, tip, watched)
                return True
        return False


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.session_file = "last_session.json"
        self.setWindowTitle("DisplayGround-Gold")
        self.resize(1280, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        button_layout = QHBoxLayout()

        self.ram_label = QLabel("RAM: 0/0 GB")
        button_layout.addWidget(self.ram_label)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        self.tab_widget = QTabWidget()
        tab_bar = CustomTabBar()
        self.tab_widget.setTabBar(tab_bar)  # From ui_components
        #self.tab_widget.tabCloseRequested.connect(self.close_tab)

        self._tab_nav_left = QToolButton(self.tab_widget)
        self._tab_nav_left.setText("<")
        self._tab_nav_left.setFixedSize(20, 20)
        self._tab_nav_left.setAutoRaise(True)
        self._tab_nav_left.setProperty("class", "tab-nav")
        self._tab_nav_left.setToolTip("Scroll tab strip left")
        self._tab_nav_left.clicked.connect(lambda: self._scroll_tab_strip(-1))

        self._tab_left_corner = QWidget(self.tab_widget)
        left_corner_layout = QHBoxLayout(self._tab_left_corner)
        left_corner_layout.setContentsMargins(0, 0, 0, 0)
        left_corner_layout.setSpacing(0)
        left_corner_layout.addWidget(self._tab_nav_left)
        self.tab_widget.setCornerWidget(self._tab_left_corner, Qt.TopLeftCorner)

        self._tab_right_corner = QWidget(self.tab_widget)
        right_corner_layout = QHBoxLayout(self._tab_right_corner)
        right_corner_layout.setContentsMargins(0, 0, 0, 0)
        right_corner_layout.setSpacing(0)

        self._tab_nav_right = QToolButton(self._tab_right_corner)
        self._tab_nav_right.setText(">")
        self._tab_nav_right.setFixedSize(20, 20)
        self._tab_nav_right.setAutoRaise(True)
        self._tab_nav_right.setProperty("class", "tab-nav")
        self._tab_nav_right.setToolTip("Scroll tab strip right")
        self._tab_nav_right.clicked.connect(lambda: self._scroll_tab_strip(1))

        self._tab_right_gap = QWidget(self._tab_right_corner)
        self._tab_right_gap.setFixedWidth(2)

        right_corner_layout.addWidget(self._tab_right_gap)
        self._tab_add_corner_button = QToolButton(self._tab_right_corner)
        self._tab_add_corner_button.setText("+")
        self._tab_add_corner_button.setFixedSize(16, 16)
        self._tab_add_corner_button.setAutoRaise(True)
        self._tab_add_corner_button.setProperty("class", "tab-add")
        self._tab_add_corner_button.setToolTip("Open a new dataset tab")
        self._tab_add_corner_button.clicked.connect(self.add_new_tab)
        right_corner_layout.addWidget(self._tab_add_corner_button)
        right_corner_layout.addWidget(self._tab_nav_right)
        self._tab_add_corner_button.hide()

        self.tab_widget.setCornerWidget(self._tab_right_corner, Qt.TopRightCorner)

        self._add_tab_button = QToolButton(self.tab_widget)
        self._add_tab_button.setText("+")
        self._add_tab_button.setFixedSize(16, 16)
        self._add_tab_button.setAutoRaise(True)
        self._add_tab_button.setProperty("class", "tab-add")
        self._add_tab_button.setToolTip("Open a new dataset tab")
        self._add_tab_button.clicked.connect(self.add_new_tab)

        main_layout.addWidget(self.tab_widget)
        self.tab_switch_timer = QTimer()
        self._pending_tab_change_index = -1
        self._last_real_tab_index = 0
        self.ram_timer = QTimer(self)
        self.ram_timer.timeout.connect(self.update_ram_label)
        self.ram_timer.start(1000)
        self.tab_switch_timer.setSingleShot(True)
        self.tab_switch_timer.setInterval(100)  # 100ms debounce
        self.tab_switch_timer.timeout.connect(self._flush_tab_change)
        self.tab_widget.currentChanged.connect(self._debounced_tab_change)
        tab_bar.layoutChanged.connect(self._update_tab_navigation_controls)
        try:
            tab_bar.tabMoved.connect(lambda *_args: self._update_tab_navigation_controls())
        except Exception:
            pass

        # Initialize Iris EARLY (before add_new_tab is called)
        self.iris = None
        if Iris is not None:
            try:
                self.iris = Iris(self)
                button_layout.addWidget(self.iris.button)
            except Exception as e:
                print(f"Iris disabled: {e}")

        add_live_btn = QPushButton("Live Tab")
        add_live_btn.setToolTip("Open live tab")
        add_live_btn.clicked.connect(self.add_live_tab)
        button_layout.addWidget(add_live_btn)

        self.add_new_tab()

        app = QApplication.instance()
        app.setStyle('Fusion')  # Set Fusion style initially for consistency
        set_light_palette(app)  # Apply light palette initially
        self._tooltip_filter = _TooltipFilter(self)
        app.installEventFilter(self._tooltip_filter)

        self._is_dark_mode = False

        self.dark_mode_button = QToolButton()
        self.dark_mode_button.setCheckable(True)
        self.dark_mode_button.setToolTip("Switch light/dark theme")
        self.dark_mode_button.setText("🌞")  # default: light mode
        self.dark_mode_button.setStyleSheet("""
            font-size: 16pt;
            font-family: "Segoe UI Emoji", "Noto Color Emoji", "Apple Color Emoji", "EmojiOne Color", sans-serif;
        """)  # Updated for color emojis
        self.dark_mode_button.toggled.connect(self._on_dark_mode_toggled)
        button_layout.addWidget(self.dark_mode_button)

        # Keep a reference and use application-level context so the shortcut
        # works even when focus is inside child widgets.
        self.restore_shortcut = QShortcut(QKeySequence("Ctrl+Shift+P"), self)
        self.restore_shortcut.setContext(Qt.ApplicationShortcut)
        self.restore_shortcut.activated.connect(self.restore_session)
        self.load_dark_mode()  # Moved here to ensure dark_mode_button exists
        QTimer.singleShot(0, self._fit_window_to_screen)
        QTimer.singleShot(0, self._update_tab_navigation_controls)

    def _fit_window_to_screen(self):
        """Ensure main window starts fully inside the available screen area."""
        try:
            if self.isFullScreen() or self.isMaximized():
                return

            screen = self.screen() if hasattr(self, "screen") else None
            if screen is None:
                app = QApplication.instance()
                if app is not None:
                    screen = app.primaryScreen()
            if screen is None:
                return

            avail = screen.availableGeometry()
            margin = 8
            max_w = max(320, avail.width() - (margin * 2))
            max_h = max(240, avail.height() - (margin * 2))

            pref_w = 1920
            pref_h = 1080
            new_w = min(pref_w, max_w)
            new_h = min(pref_h, max_h)

            self.resize(new_w, new_h)
            new_x = avail.x() + ((avail.width() - new_w) // 2)
            new_y = avail.y() + ((avail.height() - new_h) // 2)
            self.move(new_x, new_y)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_tab_navigation_controls)

    def showEvent(self, event):
        super().showEvent(event)
        # Defer so Qt finishes laying out the tab bar before we measure it
        QTimer.singleShot(0, self._update_tab_navigation_controls)
        QTimer.singleShot(100, self._update_tab_navigation_controls)

    def _on_dark_mode_toggled(self, checked: bool):
        app = QApplication.instance()
        if app is None:
            return

        if checked:
            # turn on dark mode
            self.dark_mode_button.setText("🌙")
            set_dark_palette(app)
            self._is_dark_mode = True
        else:
            # turn on light mode 
            self.dark_mode_button.setText("🌞")
            set_light_palette(app)
            self._is_dark_mode = False

        QApplication.instance().processEvents()
        try:
            data = {}
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = {}
            data['dark_mode'] = self._is_dark_mode
            with open(self.session_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving dark mode preference: {e}")


    def load_dark_mode(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                checked = bool(data.get('dark_mode', False))

                if hasattr(self, 'dark_mode_button') and self.dark_mode_button is not None:
                    self.dark_mode_button.blockSignals(True)
                    self.dark_mode_button.setChecked(checked)
                    self.dark_mode_button.blockSignals(False)
                    self.dark_mode_button.setText("🌙" if checked else "🌞")

                app = QApplication.instance()
                if checked:
                    set_dark_palette(app)
                    self._is_dark_mode = True
                else:
                    set_light_palette(app)
                    self._is_dark_mode = False

            except Exception as e:
                print(f"Error loading dark mode: {e}")

    def _trigger_iris_analysis(self, widget):
        """Trigger Iris background analysis on folder load."""
        try:
            if not self.iris or not hasattr(self.iris, 'analyze_folder_on_load'):
                return
            
            # Try to extract folder path from widget
            folder_path = None
            if isinstance(widget, BandStitchProApp):
                if hasattr(widget, 'current_folder') and widget.current_folder:
                    folder_path = widget.current_folder
            
            if folder_path:
                # remember the path so Iris can answer follow-ups even if the
                # UI later loses a reference to this widget
                try:
                    self.iris._last_loaded_folder = folder_path
                except Exception:
                    pass
                self.iris.analyze_folder_on_load(folder_path)
        except Exception as e:
            print(f"[Iris] Folder analysis trigger error: {e}")

    def _trigger_iris_frame_analysis(self, widget, frame_index=None):
        """Send current frame sample to Iris for live anomaly checks."""
        try:
            if not self.iris or not hasattr(self.iris, 'on_frame_changed'):
                return
            if widget is None or not isinstance(widget, BandStitchProApp):
                return

            # Resolve frame index
            idx = frame_index
            if idx is None:
                idx = getattr(widget, "current_frame_index", 0)
            try:
                idx = int(idx)
            except Exception:
                idx = 0

            band_frames = getattr(widget, "band_frames", None) or {}
            if not band_frames:
                return
            keys = [k for k in sorted(band_frames.keys()) if band_frames.get(k) is not None]
            if not keys:
                return
            frames = band_frames.get(keys[0])
            if frames is None or idx < 0 or idx >= len(frames):
                return
            frame_array = frames[idx]
            self.iris.on_frame_changed(idx, frame_array)
        except Exception as e:
            print(f"[Iris] Frame analysis trigger error: {e}")

    def save_session(self):
        data = {
            'dark_mode': self._is_dark_mode,
            'modes': {
                'band': [],
                'raw': [],
                'video': [],
                'live': [],
                'tiled': []
            }
        }
        for i in range(self.tab_widget.count()):
            widget = self.tab_widget.widget(i)
            title = self.tab_widget.tabText(i)
            # Band mode
            if isinstance(widget, BandStitchProApp):
                try:
                    st = widget.save_state()
                except Exception:
                    st = {}
                data['modes']['band'].append({'title': title, 'state': st})
                continue

            # Raw mode (container with RawViewer child)
            try:
                raw_child = None
                if RawViewer is not None:
                    try:
                        raw_child = widget.findChild(RawViewer)
                    except Exception:
                        raw_child = None
                if raw_child:
                    try:
                        st = raw_child.save_state()
                    except Exception:
                        st = {}
                    data['modes']['raw'].append({'title': title, 'state': st})
                    continue
            except Exception:
                pass

            # Video mode (PlaybackApp)
            try:
                playback_child = None
                if PlaybackApp is not None:
                    try:
                        playback_child = widget.findChild(PlaybackApp)
                    except Exception:
                        playback_child = None
                if playback_child:
                    try:
                        st = playback_child.save_state()
                    except Exception:
                        st = {}
                    data['modes']['video'].append({'title': title, 'state': st})
                    continue
            except Exception:
                pass

            # Live mode 
            try:
                if VideoModeHandler is not None:
                    vm = widget.findChild(VideoModeHandler)
                    if vm:
                        st = {}
                        try:
                            if hasattr(vm, 'save_state'):
                                st = vm.save_state()
                        except Exception:
                            st = {}
                        data['modes']['live'].append({'title': title, 'state': st})
            except Exception:
                pass

            # Tiled mode 
            try:
                tiled_child = None
                if TiledDisplay is not None:
                    try:
                        tiled_child = widget.findChild(TiledDisplay)
                    except Exception:
                        tiled_child = None
                if tiled_child:
                    try:
                        st = tiled_child.settings if hasattr(tiled_child, 'settings') else {}
                    except Exception:
                        st = {}
                    data['modes']['tiled'].append({'title': title, 'state': st})
                    continue
            except Exception:
                pass

        try:
            with open(self.session_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving session: {e}")


    def restore_session(self):
        if not os.path.exists(self.session_file):
            QMessageBox.information(self, "No Session", "No previous session found.")
            return
        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)

            # determine current mode of active tab
            cur_idx = self.tab_widget.currentIndex()
            cur_widget = self.tab_widget.widget(cur_idx)
            current_mode = None
            if isinstance(cur_widget, BandStitchProApp):
                current_mode = 'band'
            else:
                try:
                    if RawViewer is not None and cur_widget.findChild(RawViewer):
                        current_mode = 'raw'
                except Exception:
                    pass
                if current_mode is None:
                    try:
                        if PlaybackApp is not None and cur_widget.findChild(PlaybackApp):
                            current_mode = 'video'
                    except Exception:
                        pass
                if current_mode is None:
                    try:
                        if VideoModeHandler is not None and cur_widget.findChild(VideoModeHandler):
                            current_mode = 'live'
                    except Exception:
                        pass
                if current_mode is None:
                    try:
                        if TiledDisplay is not None and cur_widget.findChild(TiledDisplay):
                            current_mode = 'tiled'
                    except Exception:
                        pass

            if current_mode is None:
                QMessageBox.information(self, "Restore", "Cannot determine current tab mode to restore.")
                return

            saved = data.get('modes', {}).get(current_mode, [])
            if not saved:
                QMessageBox.information(self, "Restore", f"No saved tabs for mode: {current_mode}")
                return

            # Load only into current tab; never create a new tab via Ctrl+Shift+P.
            entry = saved[0]
            title = entry.get('title', 'Dataset')
            state = entry.get('state', {})
            try:
                if current_mode == 'band':
                    app = cur_widget if isinstance(cur_widget, BandStitchProApp) else None
                    if app is None:
                        QMessageBox.information(self, "Restore", "Current tab is not Band mode.")
                        return
                    app.load_state(state)
                    self.update_tab_name(app, title)
                elif current_mode == 'raw':
                    container = cur_widget
                    app = container.findChild(RawViewer) if (container and RawViewer is not None) else None
                    if app and hasattr(app, 'load_state'):
                        app.load_state(state)
                    else:
                        QMessageBox.information(self, "Restore", "Current tab cannot load Raw state.")
                        return
                    idx_tab = self.tab_widget.indexOf(container)
                    if idx_tab != -1:
                        self.tab_widget.setTabText(idx_tab, title)
                elif current_mode == 'video':
                    container = cur_widget
                    app = container.findChild(PlaybackApp) if (container and PlaybackApp is not None) else None
                    if app and hasattr(app, 'load_state'):
                        app.load_state(state)
                    else:
                        QMessageBox.information(self, "Restore", "Current tab cannot load Video state.")
                        return
                    idx_tab = self.tab_widget.indexOf(container)
                    if idx_tab != -1:
                        self.tab_widget.setTabText(idx_tab, title)
                elif current_mode == 'live':
                    container = cur_widget
                    app = container.findChild(VideoModeHandler) if (container and VideoModeHandler is not None) else None
                    if app and hasattr(app, 'load_state'):
                        try:
                            app.load_state(state)
                        except Exception:
                            pass
                    else:
                        QMessageBox.information(self, "Restore", "Current tab cannot load Live state.")
                        return
                    idx_tab = self.tab_widget.indexOf(container)
                    if idx_tab != -1:
                        self.tab_widget.setTabText(idx_tab, title)
                elif current_mode == 'tiled':
                    container = cur_widget
                    app = container.findChild(TiledDisplay) if (container and TiledDisplay is not None) else None
                    if app and state:
                        app.settings = state
                        app.load_frames()
                    else:
                        QMessageBox.information(self, "Restore", "Current tab cannot load Tiled state.")
                        return
                    idx_tab = self.tab_widget.indexOf(container)
                    if idx_tab != -1:
                        self.tab_widget.setTabText(idx_tab, title)
            except Exception as load_err:
                QMessageBox.warning(self, "Restore", f"Failed to load state in current tab: {load_err}")
                return

            # Apply dark mode
            self.dark_mode_button.setChecked(data.get('dark_mode', False))

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to restore session: {e}")


    def _stop_threads_in_widget(self, widget):
        try:
            for name, val in vars(widget).items():
                try:
                    if isinstance(val, QThread):
                        if val.isRunning():
                            try:
                                val.requestInterruption()
                            except Exception:
                                pass
                            try:
                                val.quit()
                            except Exception:
                                pass
                            try:
                                if not val.wait(2000):
                                    print(f"[DEBUG] Thread {name} did not quit gracefully, terminating")
                                    val.terminate()
                                    val.wait(1000)
                            except Exception:
                                pass
                except Exception:
                    pass
                # Recurse into child widgets
                try:
                    if isinstance(val, QWidget):
                        self._stop_threads_in_widget(val)
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent):
        # Show progress dialog for shutdown
        tab_count = self.tab_widget.count()
        if tab_count > 0:
            progress = QProgressDialog("Closing application...", None, 0, tab_count, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)  # No cancel button
            progress.setAutoReset(False)
            progress.setAutoClose(False)
            progress.show()
            
            # Perform cleanup on all tab widgets
            for i in range(tab_count):
                w = self.tab_widget.widget(i)
                if w is not None:
                    progress.setLabelText(f"Closing tab {i+1} of {tab_count}...")
                    progress.setValue(i)
                    QApplication.processEvents()  # Allow UI updates
                    self._cleanup_tab_widget(w)
            
            progress.setLabelText("Saving session...")
            progress.setValue(tab_count)
            QApplication.processEvents()
        
        self.save_session()  # Auto-save on app close
        self.ram_timer.stop()
        
        if tab_count > 0:
            progress.setLabelText("Shutdown complete")
            progress.setValue(tab_count)
            QApplication.processEvents()
            progress.close()
        
        super().closeEvent(event)       


    def _cleanup_tab_widget(self, widget):
        """Perform cleanup on a tab widget without calling close()."""
        try:
            # Check for BandStitchProApp
            if hasattr(widget, 'memory_monitor'):
                # This is a BandStitchProApp - call its closeEvent logic
                if hasattr(widget, '_is_closing'):
                    widget._is_closing = True
                
                # Stop memory monitor
                if widget.memory_monitor:
                    try:
                        widget.memory_monitor.stop()
                        widget.memory_monitor.wait(2000)
                    except Exception:
                        pass
                
                # Stop RGB fusion worker
                try:
                    if hasattr(widget, '_rgb_worker') and widget._rgb_worker:
                        if widget._rgb_worker.isRunning():
                            widget._rgb_worker.requestInterruption()
                            widget._rgb_worker.quit()
                            if not widget._rgb_worker.wait(2000):
                                widget._rgb_worker.terminate()
                                widget._rgb_worker.wait(1000)
                except Exception:
                    pass
                
                # Stop individual band workers
                try:
                    if hasattr(widget, 'individual_bands_notebook'):
                        for i in range(widget.individual_bands_notebook.count()):
                            w = widget.individual_bands_notebook.widget(i)
                            if hasattr(w, 'worker') and w.worker:
                                try:
                                    if w.worker.isRunning():
                                        w.worker.requestInterruption()
                                        w.worker.quit()
                                        if not w.worker.wait(2000):
                                            w.worker.terminate()
                                            w.worker.wait(1000)
                                except Exception:
                                    pass
                except Exception:
                    pass
                
                # Stop view_worker
                try:
                    if hasattr(widget, 'view_worker') and widget.view_worker and widget.view_worker.isRunning():
                        widget.view_worker.requestInterruption()
                        widget.view_worker.quit()
                        widget.view_worker.wait(2000)
                        if widget.view_worker.isRunning():
                            widget.view_worker.terminate()
                            widget.view_worker.wait(1000)
                except Exception:
                    pass
                
                # Stop main load worker
                try:
                    if hasattr(widget, 'worker') and widget.worker and widget.worker.isRunning():
                        widget.worker.requestInterruption()
                        widget.worker.quit()
                        widget.worker.wait(2000)
                        if widget.worker.isRunning():
                            widget.worker.terminate()
                            widget.worker.wait(1000)
                except Exception:
                    pass
            
            # Check for RawViewer
            elif hasattr(widget, 'loading_thread') or hasattr(widget, 'stack_thread'):
                # This is a RawViewer - call its closeEvent logic
                try:
                    if hasattr(widget, 'loading_thread') and widget.loading_thread and widget.loading_thread.isRunning():
                        widget.loading_thread.requestInterruption()
                        widget.loading_thread.quit()
                        widget.loading_thread.wait(2000)
                        if widget.loading_thread.isRunning():
                            widget.loading_thread.terminate()
                            widget.loading_thread.wait(1000)
                except Exception:
                    pass
                
                try:
                    if hasattr(widget, 'stack_thread') and widget.stack_thread and widget.stack_thread.isRunning():
                        widget.stack_thread.requestInterruption()
                        widget.stack_thread.quit()
                        widget.stack_thread.wait(2000)
                        if widget.stack_thread.isRunning():
                            widget.stack_thread.terminate()
                            widget.stack_thread.wait(1000)
                except Exception:
                    pass
            
            # Check for VideoModeHandler (live display)
            elif hasattr(widget, 'video_mode'):
                # This is a VideoModeHandler - call its on_close logic
                try:
                    widget.on_close()
                except Exception:
                    pass
            
            # For any widget, stop threads as safety net
            self._stop_threads_in_widget(widget)
            
        except Exception as e:
            print(f"[DEBUG] Error during tab cleanup: {e}")


    def update_ram_label(self):
        try:
            vm = psutil.virtual_memory()
            used_gb = vm.used / (1024 ** 3)
            total_gb = vm.total / (1024 ** 3)
            self.ram_label.setText(f"RAM: {used_gb:.1f}/{total_gb:.1f} GB")
        except Exception:
            self.ram_label.setText("RAM: N/A")    

    def repolish_widgets(self, widget):
        """Recursively repolish and update widgets to apply palette changes."""
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()
        for child in widget.findChildren(QWidget):
            self.repolish_widgets(child)

    def _debounced_tab_change(self, index):
        self.tab_switch_timer.stop()
        self._pending_tab_change_index = index
        self.tab_switch_timer.start()

    def _flush_tab_change(self):
        index = self._pending_tab_change_index
        self._pending_tab_change_index = -1
        if index >= 0:
            self._handle_tab_change(index)

    def _handle_tab_change(self, index):
        self._last_real_tab_index = index
        self._update_tab_navigation_controls()
        if self.iris:
            widget = self.tab_widget.widget(index)
            mode = "band"
            if hasattr(widget, "band_frames"):
                mode = "band"
            elif RawViewer is not None and hasattr(widget, "findChild") and widget.findChild(RawViewer):
                mode = "raw"
            elif PlaybackApp is not None and hasattr(widget, "findChild") and widget.findChild(PlaybackApp):
                mode = "video"
            elif VideoModeHandler is not None and hasattr(widget, "findChild") and widget.findChild(VideoModeHandler):
                mode = "live"
            elif hasattr(widget, "original_tiles_per_frame"):
                mode = "tiled"
            elif TiledDisplay is not None and hasattr(widget, "findChild") and widget.findChild(TiledDisplay):
                mode = "tiled"
            self.iris.notify_tab_activated(index, widget, mode)

        for i in range(self.tab_widget.count()):
            if i != index:
                app = self.tab_widget.widget(i)
                if hasattr(app, 'view_cache'):
                    app.view_cache.clear()

    def _is_real_tab_index(self, index: int) -> bool:
        return 0 <= index < self.tab_widget.count()

    def _real_tab_count(self) -> int:
        return self.tab_widget.count()

    def _tab_overflow_active(self) -> bool:
        tab_bar = self.tab_widget.tabBar()
        if not isinstance(tab_bar, CustomTabBar):
            return False
        total = sum(tab_bar.tabRect(i).width() for i in range(tab_bar.count()))
        return total > tab_bar.width()

    def _position_add_tab_button(self):
        try:
            if getattr(self, '_tab_add_corner_button', None) is not None and self._tab_add_corner_button.isVisible():
                self._add_tab_button.hide()
                return

            tab_bar = self.tab_widget.tabBar()
            if tab_bar is None:
                return

            gap = 6
            bar_pos = tab_bar.mapTo(self.tab_widget, QPoint(0, 0))
            btn_h = self._add_tab_button.height()
            btn_w = self._add_tab_button.width()
            y = bar_pos.y() + max(0, (tab_bar.height() - btn_h) // 2)

            right_edge = bar_pos.x()
            if tab_bar.count() > 0:
                last_rect = tab_bar.tabRect(tab_bar.count() - 1)
                right_edge = bar_pos.x() + last_rect.right()

            if self._tab_right_corner.isVisible():
                rc_pos = self._tab_right_corner.mapTo(self.tab_widget, QPoint(0, 0))
                right_limit = rc_pos.x() - btn_w - gap
            else:
                right_limit = self.tab_widget.width() - gap

            left_limit = bar_pos.x() + gap
            x = max(left_limit, right_edge + gap)
            x = min(x, right_limit)

            self._add_tab_button.move(x, y)
            self._add_tab_button.show()
            self._add_tab_button.raise_()
        except Exception:
            pass

    def _update_tab_navigation_controls(self):
        tab_bar = self.tab_widget.tabBar()
        overflow = self._tab_overflow_active()

        if overflow and isinstance(tab_bar, CustomTabBar):
            self._tab_left_corner.setVisible(True)
            self._tab_right_corner.setVisible(True)
            self._tab_nav_left.show()
            self._tab_nav_right.show()
            self._tab_add_corner_button.show()
            self._add_tab_button.hide()

            self._tab_nav_left.setEnabled(tab_bar.can_scroll_left())
            self._tab_nav_right.setEnabled(tab_bar.can_scroll_right())

            nav_w = self._tab_nav_right.width()
            add_w = self._tab_add_corner_button.width()
            safe_margin = 2
            self._tab_left_corner.setFixedWidth(self._tab_nav_left.width())
            self._tab_right_gap.setFixedWidth(safe_margin)
            self._tab_right_corner.setFixedWidth(nav_w + add_w + safe_margin)
        else:
            self._tab_left_corner.setVisible(False)
            self._tab_right_corner.setVisible(False)
            self._tab_nav_left.hide()
            self._tab_nav_right.hide()
            self._tab_add_corner_button.hide()
            self._add_tab_button.show()
            self._tab_left_corner.setFixedWidth(0)
            self._tab_right_corner.setFixedWidth(0)

        self._position_add_tab_button()

    def _scroll_tab_strip(self, direction: int):
        tab_bar = self.tab_widget.tabBar()
        if not isinstance(tab_bar, CustomTabBar):
            return
        if direction < 0:
            tab_bar.scroll_strip_left()
        else:
            tab_bar.scroll_strip_right()
        tab_bar.update()
        QTimer.singleShot(0, self._update_tab_navigation_controls)

    def _insert_real_tab(self, widget, title: str) -> int:
        tab_index = self.tab_widget.addTab(widget, title)
        self._set_custom_close_button(tab_index)
        self.tab_widget.setCurrentIndex(tab_index)
        self._last_real_tab_index = tab_index
        # Use two deferred calls: first let Qt lay out the new tab, then update
        QTimer.singleShot(0, self._update_tab_navigation_controls)
        QTimer.singleShot(50, self._update_tab_navigation_controls)
        return tab_index

    def _attach_bottom_terminal(self, host_widget, parent_layout, content_widget):
        """Attach a collapsible + resizable terminal below content for a mode tab."""
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        parent_layout.addWidget(splitter, 1)

        splitter.addWidget(content_widget)

        terminal_panel = QWidget(host_widget)
        terminal_panel_layout = QVBoxLayout()
        terminal_panel_layout.setContentsMargins(0, 0, 0, 0)
        terminal_panel_layout.setSpacing(2)
        terminal_panel.setLayout(terminal_panel_layout)

        term_btn = QPushButton("Terminal ↑")
        term_btn.setToolTip("Show or hide terminal panel")
        terminal_panel_layout.addWidget(term_btn)

        terminal_widget = TerminalWidget(host_widget)
        terminal_widget.setMaximumHeight(16777215)
        terminal_widget.hide()
        terminal_panel_layout.addWidget(terminal_widget, 1)
        splitter.addWidget(terminal_panel)

        btn_h = max(28, term_btn.sizeHint().height())
        splitter.setSizes([1000, btn_h])
        splitter.handle(1).setEnabled(False)

        state = {"expanded": False}

        def toggle_terminal():
            total = max(240, splitter.height())
            if state["expanded"]:
                terminal_widget.hide()
                splitter.setSizes([max(120, total - btn_h), btn_h])
                splitter.handle(1).setEnabled(False)
                term_btn.setText("Terminal ↑")
                state["expanded"] = False
            else:
                terminal_widget.show()
                bottom = max(160, int(total * 0.30))
                top = max(120, total - bottom)
                splitter.setSizes([top, bottom])
                splitter.handle(1).setEnabled(True)
                term_btn.setText("Terminal ↓")
                state["expanded"] = True
                try:
                    terminal_widget.focus_input()
                except Exception:
                    pass

        term_btn.clicked.connect(toggle_terminal)
        host_widget._terminal_btn = term_btn
        host_widget._terminal_splitter = splitter
        host_widget._terminal_widget = terminal_widget

    def add_new_tab(self):
        """Show mode selection dialog and create appropriate tab."""
        dialog = ModeSelectionDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return

        mode = dialog.get_selected_mode()

        if mode == "band":
            self._add_band_tab()
        elif mode == "raw":
            self._add_raw_tab()
        elif mode == "video":
            self._add_video_tab()
        elif mode == "live":
            self._add_live_tab()
        elif mode == "tiled":
            self._add_tiled_tab()

    def _add_band_tab(self):
        """Create a new Band Mode tab."""
        app = BandStitchProApp(main_app=self)
        tab_index = self._insert_real_tab(app, "Band Mode")
        if self.iris:
            self.iris.notify_tab_activated(tab_index, app, "band")
        
        # Trigger Iris analysis on folder load (after brief delay for UI)
        if self.iris:
            QTimer.singleShot(1000, lambda: self._trigger_iris_analysis(app))
            # Trigger Iris frame analysis as the frame slider changes.
            try:
                if hasattr(app, "frame_slider") and app.frame_slider is not None:
                    app.frame_slider.valueChanged.connect(
                        lambda v, w=app: self._trigger_iris_frame_analysis(w, v)
                    )
            except Exception as e:
                print(f"[Iris] Failed to wire frame-slider integration: {e}")

    def _add_raw_tab(self):
        """Create a new Raw Mode tab."""
        if RawViewer is None:
            QMessageBox.warning(self, "Warning", "Raw Mode not available (missing raw_mode module)")
            return
        
        raw_widget = QWidget()
        layout = QVBoxLayout()
        raw_widget.setLayout(layout)

        content = QWidget(raw_widget)
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        raw_viewer = RawViewer(parent=content)
        raw_viewer._main_app = self
        content_layout.addWidget(raw_viewer)

        self._attach_bottom_terminal(raw_widget, layout, content)
        
        self._insert_real_tab(raw_widget, "Raw Mode")

    def _add_video_tab(self):
        """Create a new Video Mode tab."""
        if PlaybackApp is None:
            QMessageBox.warning(self, "Warning", "Video Mode not available (missing video_mode module)")
            return
        
        video_widget = QWidget()
        layout = QVBoxLayout()
        video_widget.setLayout(layout)

        content = QWidget(video_widget)
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        content_layout.addWidget(QLabel("Video Mode - Play and Analyze Videos"))
        video_app = PlaybackApp(parent=content)
        video_app._main_app = self
        content_layout.addWidget(video_app)

        self._attach_bottom_terminal(video_widget, layout, content)
        
        self._insert_real_tab(video_widget, "Video Mode")

    def _add_tiled_tab(self):
        """Create a new Tiled Mode tab."""
        if TiledDisplay is None:
            QMessageBox.warning(self, "Warning", "Tiled Mode not available (missing tiled_viewer module)")
            return
        
        tiled_widget = QWidget()
        layout = QVBoxLayout()
        tiled_widget.setLayout(layout)

        content = QWidget(tiled_widget)
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        content_layout.addWidget(QLabel("Tiled Mode - Tiled RAW Viewer"))
        tiled_app = TiledDisplay(parent=content)
        tiled_app._main_app = self
        content_layout.addWidget(tiled_app)

        self._attach_bottom_terminal(tiled_widget, layout, content)
        
        self._insert_real_tab(tiled_widget, "Tiled Mode")

    def _add_live_tab(self):
        """Create a new Live Display tab."""
        self.add_live_tab()  # Reuse existing logic
    
    def update_tab_name(self, app, name):
        self.update_tab_name_for_widget(app, name)

    def update_tab_name_for_widget(self, widget, name):
        if widget is None or not name:
            return
        try:
            w = widget
            while w is not None:
                index = self.tab_widget.indexOf(w)
                if index != -1:
                    self.tab_widget.setTabText(index, str(name))
                    return
                w = w.parentWidget()
        except Exception:
            pass

    def _set_custom_close_button(self, index):
        close_btn = QToolButton()
        close_btn.setText("×")  # Trendy Unicode cross
        close_btn.setFixedSize(16, 16)  # Slightly smaller close affordance
        close_btn.setProperty("class", "tab-close")  # For stylesheet targeting
        close_btn.setToolTip("Close this tab")
        close_btn.clicked.connect(lambda _checked=False, btn=close_btn: self._close_tab_from_button(btn))
        self.tab_widget.tabBar().setTabButton(index, QTabBar.RightSide, close_btn)

    def _close_tab_from_button(self, button):
        if button is None:
            return
        tab_bar = self.tab_widget.tabBar()
        for index in range(self.tab_widget.count()):
            if tab_bar.tabButton(index, QTabBar.RightSide) is button:
                self.close_tab(index)
                return
    
    def add_live_tab(self):
        live_widget = QWidget()
        layout = QVBoxLayout()
        live_widget.setLayout(layout)
        content = QWidget(live_widget)
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        content_layout.addWidget(QLabel("Live Display"))

        if VideoModeHandler:
            video_mode = VideoModeHandler(content, filepath=None)
            video_mode._main_app = self
            content_layout.addWidget(video_mode)
        else:
            content_layout.addWidget(QLabel("Live Display: Module not available"))

        self._attach_bottom_terminal(live_widget, layout, content)

        dataset_app = None
        cur = self.tab_widget.currentWidget()

        if cur and cur.__class__.__name__ == "BandStitchProApp":
            dataset_app = cur
        else:
            for i in range(self.tab_widget.count()):
                candidate = self.tab_widget.widget(i)
                if candidate and candidate.__class__.__name__ == "BandStitchProApp":
                    dataset_app = candidate
                    break

        if dataset_app is None:
            self._insert_real_tab(live_widget, "Live Display")
            return

        try:
            notebook = getattr(dataset_app, 'individual_bands_notebook', None)
            if notebook:
                while notebook.count():
                    widget = notebook.widget(0)
                    notebook.removeTab(0)
                    try:
                        widget.deleteLater()
                    except Exception:
                        pass


            gc.collect()

            band_checkbox_layout = getattr(dataset_app, 'band_checkbox_layout', None)
            if band_checkbox_layout:
                while band_checkbox_layout.count():
                    item = band_checkbox_layout.takeAt(0)
                    if item and item.widget():
                        try:
                            item.widget().deleteLater()
                        except Exception:
                            pass

            band_frames = getattr(dataset_app, 'band_frames', None)
            if not band_frames:
                self._insert_real_tab(live_widget, "Live Display")
                return

            band_enabled_map = getattr(dataset_app, 'band_enabled', {}) or {}
            for key in sorted(band_frames.keys()):
                cb = QCheckBox(f"Band {key[1:]}")
                prev_checked = False
                prev_cb = band_enabled_map.get(key)
                try:
                    prev_checked = bool(prev_cb.isChecked())
                except Exception:
                    prev_checked = False
                cb.setChecked(prev_checked)
                cb.stateChanged.connect(lambda state, k=key: dataset_app.toggle_band(k, state) if hasattr(dataset_app, 'toggle_band') else None)
                if band_checkbox_layout:
                    band_checkbox_layout.addWidget(cb)

            # Handle pan checkbox if applicable
            unbinned_keys_for_check = [k for k in band_frames.keys() if k.endswith(('_left', '_right'))]
            if unbinned_keys_for_check:
                pan_cb = QCheckBox("Pan")
                prev_pan_checked = False
                prev_pan = band_enabled_map.get("pan")
                try:
                    prev_pan_checked = bool(prev_pan.isChecked())
                except Exception:
                    prev_pan_checked = False
                pan_cb.setChecked(prev_pan_checked)
                pan_cb.stateChanged.connect(lambda state: dataset_app.toggle_band("pan", state) if hasattr(dataset_app, 'toggle_band') else None)
                if band_checkbox_layout:
                    band_checkbox_layout.addWidget(pan_cb)
                # store back on dataset_app if needed
                try:
                    dataset_app.band_enabled["pan"] = pan_cb
                except Exception:
                    pass

            if band_checkbox_layout:
                band_checkbox_layout.addStretch()

            # Build list of checked keys and placeholder tabs (lazy loaded)
            keys = []
            for k in sorted(band_frames.keys()):
                cb = dataset_app.band_enabled.get(k) if hasattr(dataset_app, 'band_enabled') else None
                if cb is not None and getattr(cb, 'isChecked', lambda: False)() and band_frames.get(k) is not None:
                    keys.append(k)

            dataset_app.individual_band_keys = keys

            # Add placeholder tabs for checked keys
            notebook = getattr(dataset_app, 'individual_bands_notebook', None)
            if notebook:
                for i, key in enumerate(keys):
                    placeholder = QWidget()
                    placeholder.setObjectName("placeholder")
                    placeholder.key = key
                    base_key = key.rsplit('_', 1)[0] if '_' in key else key
                    side = key.split('_')[-1] if '_' in key else ''
                    tab_text = f"Band {base_key[1:]} {side}".strip()
                    notebook.addTab(placeholder, tab_text)

                # Add pan tab if applicable
                unbinned_keys = [k for k in keys if k.endswith(('_left', '_right'))]
                if unbinned_keys and dataset_app.band_enabled.get("pan") and dataset_app.band_enabled["pan"].isChecked():
                    placeholder = QWidget()
                    placeholder.setObjectName("pan_placeholder")
                    placeholder.unbinned_keys = unbinned_keys
                    notebook.addTab(placeholder, "Pan")

                # connect lazy loader if not connected already
                if not getattr(dataset_app, '_individual_connected', False):
                    try:
                        notebook.currentChanged.connect(dataset_app.lazy_load_individual_tab)
                    except Exception:
                        pass
                    dataset_app._individual_connected = True

        except Exception as e:
            # don't raise UI-level exceptions; log to console
            print("add_live_tab (safe) encountered:", e)

        # finally add the live tab to main tab widget
        self._insert_real_tab(live_widget, "Live Display")

    def add_video_tab(self, folder: str = None, width: int = None,
                      height: int = None, bitdepth: int = None):
        if PlaybackApp is None:
            QMessageBox.warning(self, "Error", "PlaybackApp module not available.")
            return

        video_widget = QWidget()
        layout = QVBoxLayout()
        video_widget.setLayout(layout)

        content = QWidget(video_widget)
        content_layout = QVBoxLayout()
        content.setLayout(content_layout)
        content_layout.addWidget(QLabel("Video Mode"))
        video_app = PlaybackApp(parent=content)
        video_app._main_app = self
        content_layout.addWidget(video_app)
        self._attach_bottom_terminal(video_widget, layout, content)

        # Add to tab widget
        tab_index = self._insert_real_tab(video_widget, "Video Mode")
        try:
            if height is not None:
                video_app.height_entry.setText(str(int(height)))
                video_app.validate_height()
            if width is not None:
                video_app.width_entry.setText(str(int(width)))
                video_app.validate_width()
            if bitdepth is not None:
                video_app.bitdepth_var.setCurrentText(str(int(bitdepth)))
            if folder:
                video_app.open_folder(folder)
        except Exception as e:
            print("add_video_tab (safe) encountered:", e)
    
    def open_editor_tab(self, source_viewer):
        """Open editor tab with the source viewer's image."""
        if not source_viewer.current_pil_image:
            QMessageBox.warning(self, "No Image", "No image loaded in the source viewer.")
            return
        if EditorTab is None:
            QMessageBox.warning(self, "Error", "EditorTab module not available.")
            return
        editor = EditorTab(source_viewer, self)
        if editor.original_array is None:
            return
        # Get a descriptive name for the tab
        tab_name = "Edited Image"
        try:
            for i in range(self.tab_widget.count()):
                if self.tab_widget.widget(i) == source_viewer.parent():
                    tab_name = f"Edited {self.tab_widget.tabText(i)}"
                    break
        except Exception:
            pass
        self._insert_real_tab(editor, tab_name)
        
    def close_tab(self, index):
        if self._real_tab_count() <= 1:
            QMessageBox.warning(self, "Warning", "Cannot close the last tab")
            return

        if index < 0 or index >= self.tab_widget.count():
            print(f"[DEBUG] Ignoring invalid close_tab index: {index}")
            return

        self.save_session()
        widget = self.tab_widget.widget(index)
        if widget is None:
            print(f"[DEBUG] No widget found for tab index: {index}")
            return

        def finalize_close():
            current_index = self.tab_widget.indexOf(widget)
            if current_index == -1:
                try:
                    widget.close()
                except Exception as e:
                    print(f"[DEBUG] Error closing detached tab widget: {e}")
                widget.deleteLater()
                return
            if self.iris:
                self.iris.notify_tab_closed(current_index)
            try:
                widget.close()
            except Exception as e:
                print(f"[DEBUG] Error closing tab widget at index {current_index}: {e}")
            self.tab_widget.removeTab(current_index)
            fallback_index = min(current_index, max(0, self._real_tab_count() - 1))
            if self._is_real_tab_index(fallback_index):
                self.tab_widget.setCurrentIndex(fallback_index)
            try:
                widget.setParent(None)
            except Exception:
                pass
            widget.deleteLater()
            self._last_real_tab_index = self.tab_widget.currentIndex() if self._is_real_tab_index(self.tab_widget.currentIndex()) else max(0, self._real_tab_count() - 1)
            QTimer.singleShot(0, self._update_tab_navigation_controls)
            
        if hasattr(widget, 'band_enabled'):
            for key in list(widget.band_enabled.keys()):
                try:
                    del widget.band_enabled[key]
                except Exception:
                    pass
            try:
                widget.band_enabled.clear()
                del widget.band_enabled
            except Exception:
                pass
        
        if hasattr(widget, 'tdi_worker') and widget.tdi_worker.isRunning():
            widget.tdi_worker.requestInterruption()
            widget.tdi_worker.wait()
        
        video_mode = None
        if VideoModeHandler and hasattr(widget, "findChild"):
            video_mode = widget.findChild(VideoModeHandler)
        
        if video_mode:
            if video_mode.connected:
                # Trigger disconnect logic
                video_mode.connectCamera()
            
            # Show countdown dialog
            progress = QProgressDialog("Safely closing Live Display...\nRemaining: 2 seconds", None, 0, 2, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.setCancelButton(None)  # No cancel button
            progress.show()
            progress.setAutoReset(False)
            progress.setAutoClose(False)
            
            # Timer for countdown
            timer = QTimer(self)
            def update_progress():
                progress.setValue(progress.value() + 1)
                remaining = progress.maximum() - progress.value()
                sec_text = "second" if remaining == 1 else "seconds"
                if remaining > 0:
                    progress.setLabelText(f"Safely closing Live Display...\nRemaining: {remaining} {sec_text}")
                else:
                    progress.setLabelText("Safely closing Live Display...\nClosing now.")
                
                if progress.value() >= progress.maximum():
                    timer.stop()
                    if hasattr(video_mode, 'process') and video_mode.process and video_mode.process.poll() is None:
                        video_mode.process.kill()
                    finalize_close()
                    progress.close()
            
            timer.timeout.connect(update_progress)
            timer.start(1000)
        else:
            if PlaybackApp and hasattr(widget, "findChild"):
                playback_app = widget.findChild(PlaybackApp)
                if playback_app:
                    playback_app.closeEvent(QCloseEvent())
            finalize_close()

if __name__ == "__main__":
    qInstallMessageHandler(_qt_message_handler)
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.processEvents()
    sys.exit(app.exec_())
