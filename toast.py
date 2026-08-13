"""
Modern, native-matching, non-blocking toast notification system for PyQt5.
Customized to seamlessly match the GroundDisplayApp-Gold theme (Charcoal & Gold/Teal in dark mode,
and Clean White & Teal in light mode) with crisp vector badges, animated progress bar,
hover-pause, and smooth stacking.
"""

from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QGraphicsOpacityEffect, QGraphicsDropShadowEffect, QApplication
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QPointF, QRect, QRectF, QParallelAnimationGroup,
    QObject, pyqtSignal, QEvent
)
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QBrush, QPen, QCursor, QPainterPath,
    QLinearGradient, QPalette
)

import sys
import os


class ToastType:
    WARNING = "warning"
    INFO = "info"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"


def is_dark_mode():
    """Detect whether the application is currently using dark mode."""
    try:
        app = QApplication.instance()
        if app:
            return app.palette().color(QPalette.Window).lightness() < 128
    except Exception:
        pass
    return True


THEME_CONFIG = {
    ToastType.WARNING: {
        "dark": {
            "accent": QColor("#e0a800"),        # App Gold/Amber
            "accent_light": QColor("#ffd54f"),
            "badge_bg": QColor(224, 168, 0, 40),
            "badge_border": QColor(224, 168, 0, 110),
            "title_fg": "#ffd54f",
            "msg_fg": "#e6e6e6",
        },
        "light": {
            "accent": QColor("#d97706"),
            "accent_light": QColor("#f59e0b"),
            "badge_bg": QColor(217, 119, 6, 25),
            "badge_border": QColor(217, 119, 6, 90),
            "title_fg": "#b45309",
            "msg_fg": "#374151",
        },
        "default_title": "Warning",
    },
    ToastType.INFO: {
        "dark": {
            "accent": QColor("#29b6f6"),        # Sky/Cyan
            "accent_light": QColor("#81d4fa"),
            "badge_bg": QColor(41, 182, 246, 40),
            "badge_border": QColor(41, 182, 246, 110),
            "title_fg": "#81d4fa",
            "msg_fg": "#e6e6e6",
        },
        "light": {
            "accent": QColor("#0284c7"),
            "accent_light": QColor("#38bdf8"),
            "badge_bg": QColor(2, 132, 199, 25),
            "badge_border": QColor(2, 132, 199, 90),
            "title_fg": "#0369a1",
            "msg_fg": "#374151",
        },
        "default_title": "Information",
    },
    ToastType.ERROR: {
        "dark": {
            "accent": QColor("#e53935"),        # Soft Crimson
            "accent_light": QColor("#ff8a80"),
            "badge_bg": QColor(229, 57, 53, 40),
            "badge_border": QColor(229, 57, 53, 110),
            "title_fg": "#ff8a80",
            "msg_fg": "#e6e6e6",
        },
        "light": {
            "accent": QColor("#dc2626"),
            "accent_light": QColor("#ef4444"),
            "badge_bg": QColor(220, 38, 38, 25),
            "badge_border": QColor(220, 38, 38, 90),
            "title_fg": "#b91c1c",
            "msg_fg": "#374151",
        },
        "default_title": "Error",
    },
    ToastType.CRITICAL: {
        "dark": {
            "accent": QColor("#e53935"),
            "accent_light": QColor("#ff8a80"),
            "badge_bg": QColor(229, 57, 53, 40),
            "badge_border": QColor(229, 57, 53, 110),
            "title_fg": "#ff8a80",
            "msg_fg": "#e6e6e6",
        },
        "light": {
            "accent": QColor("#dc2626"),
            "accent_light": QColor("#ef4444"),
            "badge_bg": QColor(220, 38, 38, 25),
            "badge_border": QColor(220, 38, 38, 90),
            "title_fg": "#b91c1c",
            "msg_fg": "#374151",
        },
        "default_title": "Critical Error",
    },
    ToastType.SUCCESS: {
        "dark": {
            "accent": QColor("#26A69A"),        # App signature Teal
            "accent_light": QColor("#80cbc4"),
            "badge_bg": QColor(38, 166, 154, 40),
            "badge_border": QColor(38, 166, 154, 110),
            "title_fg": "#80cbc4",
            "msg_fg": "#e6e6e6",
        },
        "light": {
            "accent": QColor("#00897B"),
            "accent_light": QColor("#26A69A"),
            "badge_bg": QColor(0, 137, 123, 25),
            "badge_border": QColor(0, 137, 123, 90),
            "title_fg": "#00695c",
            "msg_fg": "#374151",
        },
        "default_title": "Success",
    },
}


class ToastIconWidget(QWidget):
    """
    Renders vector icon badges in harmony with the application palette.
    """
    def __init__(self, kind=ToastType.INFO, is_dark=True, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.is_dark = is_dark
        theme_node = THEME_CONFIG.get(self.kind, THEME_CONFIG[ToastType.INFO])
        self.theme = theme_node["dark" if self.is_dark else "light"]
        self.setFixedSize(26, 26)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        cx = rect.center().x()
        cy = rect.center().y()

        # Circular badge backdrop
        painter.setBrush(QBrush(self.theme["badge_bg"]))
        painter.setPen(QPen(self.theme["badge_border"], 1.0))
        painter.drawEllipse(rect)

        accent = self.theme["accent_light"] if self.is_dark else self.theme["accent"]

        if self.kind in (ToastType.ERROR, ToastType.CRITICAL):
            # Crisp Cross Icon ✕
            pen = QPen(accent, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            r = 4.0
            painter.drawLine(QPointF(cx - r, cy - r), QPointF(cx + r, cy + r))
            painter.drawLine(QPointF(cx + r, cy - r), QPointF(cx - r, cy + r))

        elif self.kind == ToastType.WARNING:
            # Crisp Exclamation !
            pen = QPen(accent, 2.0, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(cx, cy - 4.5), QPointF(cx, cy + 0.5))
            painter.setBrush(QBrush(accent))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, cy + 4.5), 1.1, 1.1)

        elif self.kind == ToastType.SUCCESS:
            # Crisp Checkmark ✓
            path = QPainterPath()
            path.moveTo(cx - 4.5, cy + 0.5)
            path.lineTo(cx - 1.2, cy + 3.8)
            path.lineTo(cx + 4.8, cy - 3.0)
            pen = QPen(accent, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

        else:  # ToastType.INFO
            # Crisp 'i' icon
            painter.setBrush(QBrush(accent))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, cy - 4.2), 1.2, 1.2)
            pen = QPen(accent, 2.0, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(cx, cy - 0.8), QPointF(cx, cy + 4.2))

        painter.end()


class ToastCloseButton(QPushButton):
    """
    Clean, theme-matching close button with subtle hover and press feedback.
    """
    def __init__(self, is_dark=True, parent=None):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setFixedSize(22, 22)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setToolTip("Dismiss")
        self._hovered = False
        self._pressed = False

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hovered = False
        self._pressed = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()

        if self._pressed:
            bg_color = QColor(255, 255, 255, 45) if self.is_dark else QColor(0, 0, 0, 30)
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)
        elif self._hovered:
            bg_color = QColor(255, 255, 255, 28) if self.is_dark else QColor(0, 0, 0, 18)
            painter.setBrush(QBrush(bg_color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

        cx = rect.center().x()
        cy = rect.center().y()
        r = 3.5

        if self.is_dark:
            color = QColor("#ffffff") if (self._hovered or self._pressed) else QColor("#a8a8a8")
        else:
            color = QColor("#111111") if (self._hovered or self._pressed) else QColor("#666666")

        pen = QPen(color, 1.8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(int(cx - r), int(cy - r), int(cx + r), int(cy + r))
        painter.drawLine(int(cx + r), int(cy - r), int(cx - r), int(cy + r))
        painter.end()


class ToastNotification(QWidget):
    """
    Native-themed in-window floating toast notification card for GroundDisplayApp.
    Matches the Fusion Dark/Charcoal aesthetic with 5px radius and accent styling.
    Runs as an in-window overlay to completely avoid OS window-manager popups and switcher artifacts.
    """
    closing = pyqtSignal(object)
    dismissed = pyqtSignal(object)

    def __init__(self, title, message, kind=ToastType.WARNING, duration=3500, parent=None):
        parent_window = parent.window() if (parent and hasattr(parent, "window")) else parent
        super().__init__(parent_window)  # In-window child overlay
        self.parent_target = parent_window
        self.kind = str(kind).lower() if kind else ToastType.WARNING
        if self.kind not in THEME_CONFIG:
            self.kind = ToastType.WARNING

        self.is_dark = is_dark_mode()
        theme_node = THEME_CONFIG[self.kind]
        self.theme = theme_node["dark" if self.is_dark else "light"]
        self.duration = max(1200, int(duration))
        self.title_text = str(title) if title else theme_node["default_title"]
        self.message_text = str(message) if message is not None else ""
        self._is_closing = False
        self._hovered = False
        self._progress = 1.0

        self._setup_ui()
        self._setup_animation()

    def _setup_ui(self):
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.setMinimumWidth(310)
        self.setMaximumWidth(460)

        # Opacity effect for fade transitions
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self.opacity_effect)

        # Main Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(14, 10, 10, 11)
        main_layout.setSpacing(10)

        # Vector Icon Badge
        self.icon_widget = ToastIconWidget(self.kind, is_dark=self.is_dark, parent=self)
        main_layout.addWidget(self.icon_widget, 0, Qt.AlignTop)

        # Text Layout
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)

        self.title_label = QLabel(self.title_text, self)
        self.title_label.setObjectName("ToastTitle")
        p_title = self.title_label.palette()
        p_title.setColor(QPalette.WindowText, QColor(self.theme["title_fg"]))
        self.title_label.setPalette(p_title)
        self.title_label.setStyleSheet(f"""
            QLabel#ToastTitle {{
                color: {self.theme["title_fg"]} !important;
                font-family: 'Segoe UI', Arial, sans-serif !important;
                font-size: 11.5px !important;
                font-weight: bold !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
                padding: 0px !important;
                margin: 0px !important;
            }}
        """)
        text_layout.addWidget(self.title_label)

        self.message_label = QLabel(self.message_text, self)
        self.message_label.setObjectName("ToastMessage")
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        p_msg = self.message_label.palette()
        p_msg.setColor(QPalette.WindowText, QColor(self.theme["msg_fg"]))
        self.message_label.setPalette(p_msg)
        self.message_label.setStyleSheet(f"""
            QLabel#ToastMessage {{
                color: {self.theme["msg_fg"]} !important;
                font-family: 'Segoe UI', Arial, sans-serif !important;
                font-size: 11px !important;
                line-height: 1.38 !important;
                font-weight: normal !important;
                background: transparent !important;
                border: none !important;
                outline: none !important;
                padding: 0px !important;
                margin: 0px !important;
            }}
        """)
        text_layout.addWidget(self.message_label)
        main_layout.addLayout(text_layout, 1)

        # Close Button
        self.close_btn = ToastCloseButton(is_dark=self.is_dark, parent=self)
        self.close_btn.clicked.connect(self.fade_out_and_close)
        main_layout.addWidget(self.close_btn, 0, Qt.AlignTop)

        # Timer & Progress update
        self._timer_interval = 25
        self._elapsed_ms = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(self._timer_interval)
        self.progress_timer.timeout.connect(self._on_timer_tick)

    def _setup_animation(self):
        # Entry Animation Group
        self.fade_in_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_in_anim.setDuration(220)
        self.fade_in_anim.setStartValue(0.0)
        self.fade_in_anim.setEndValue(1.0)
        self.fade_in_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.slide_in_anim = QPropertyAnimation(self, b"pos")
        self.slide_in_anim.setDuration(220)
        self.slide_in_anim.setEasingCurve(QEasingCurve.OutCubic)

        self.entry_group = QParallelAnimationGroup(self)
        self.entry_group.addAnimation(self.fade_in_anim)
        self.entry_group.addAnimation(self.slide_in_anim)

        # Exit Animation Group
        self.fade_out_anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_out_anim.setDuration(170)
        self.fade_out_anim.setStartValue(1.0)
        self.fade_out_anim.setEndValue(0.0)
        self.fade_out_anim.setEasingCurve(QEasingCurve.InCubic)

        self.slide_out_anim = QPropertyAnimation(self, b"pos")
        self.slide_out_anim.setDuration(170)
        self.slide_out_anim.setStartValue(curr_pos := self.pos())
        self.slide_out_anim.setEndValue(curr_pos)
        self.slide_out_anim.setEasingCurve(QEasingCurve.InCubic)

        self.exit_group = QParallelAnimationGroup(self)
        self.exit_group.addAnimation(self.fade_out_anim)
        self.exit_group.addAnimation(self.slide_out_anim)
        self.exit_group.finished.connect(self._on_exit_finished)

    def paintEvent(self, event):
        """Custom painting matching the exact GroundDisplayApp fusion charcoal/white card."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = self.rect().adjusted(1, 1, -1, -1)
        r = 5.0  # Matches QGroupBox border-radius in app

        card_path = QPainterPath()
        card_path.addRoundedRect(rect.x(), rect.y(), rect.width(), rect.height(), r, r)

        # Card Background matching application palette
        if self.is_dark:
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, QColor(48, 48, 48, 252))
            gradient.setColorAt(1.0, QColor(38, 38, 38, 254))
            painter.fillPath(card_path, QBrush(gradient))

            # Crisp neutral border matching QGroupBox / QFrame in app
            painter.setPen(QPen(QColor(72, 72, 72), 1.0))
        else:
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, QColor(255, 255, 255, 254))
            gradient.setColorAt(1.0, QColor(246, 246, 248, 254))
            painter.fillPath(card_path, QBrush(gradient))

            painter.setPen(QPen(QColor(204, 204, 204), 1.0))

        painter.setBrush(Qt.NoBrush)
        painter.drawPath(card_path)

        # Left 3px accent strip
        accent_color = self.theme["accent"]
        strip_w = 3.0
        strip_x = rect.x() + 3.0
        strip_y = rect.y() + 6.0
        strip_h = max(0.0, rect.height() - 12.0)
        strip_path = QPainterPath()
        strip_path.addRoundedRect(strip_x, strip_y, strip_w, strip_h, 1.5, 1.5)
        painter.fillPath(strip_path, QBrush(accent_color))

        # Bottom sleek progress bar
        if self._progress > 0.001 and not self._is_closing:
            bar_h = 2.0
            bar_y = rect.y() + rect.height() - bar_h
            bar_w = (rect.width() - 2 * r) * self._progress
            prog_path = QPainterPath()
            prog_path.addRoundedRect(rect.x() + r, bar_y, max(0.0, bar_w), bar_h, 1.0, 1.0)
            prog_color = QColor(accent_color.red(), accent_color.green(), accent_color.blue(), 180 if self.is_dark else 150)
            painter.fillPath(prog_path, QBrush(prog_color))

        painter.end()

    def _on_timer_tick(self):
        if self._hovered or self._is_closing:
            return
        self._elapsed_ms += self._timer_interval
        self._progress = max(0.0, 1.0 - (self._elapsed_ms / float(self.duration)))
        self.update()

        if self._elapsed_ms >= self.duration:
            self.fade_out_and_close()

    def enterEvent(self, event):
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def pop_up(self, target_pos):
        self.adjustSize()
        self.raise_()
        start_pos = QPoint(target_pos.x() + 25, target_pos.y())
        self.move(start_pos)
        self.show()
        self.raise_()

        self.slide_in_anim.setStartValue(start_pos)
        self.slide_in_anim.setEndValue(target_pos)
        self.entry_group.start()
        self.progress_timer.start()

    def animate_to_pos(self, new_pos):
        if self._is_closing:
            return
        self.raise_()
        reposition_anim = QPropertyAnimation(self, b"pos", self)
        reposition_anim.setDuration(170)
        reposition_anim.setStartValue(self.pos())
        reposition_anim.setEndValue(new_pos)
        reposition_anim.setEasingCurve(QEasingCurve.OutCubic)
        reposition_anim.start()

    def fade_out_and_close(self):
        if self._is_closing:
            return
        self._is_closing = True
        self.progress_timer.stop()
        self.closing.emit(self)

        curr_pos = self.pos()
        end_pos = QPoint(curr_pos.x() + 30, curr_pos.y())
        self.slide_out_anim.setStartValue(curr_pos)
        self.slide_out_anim.setEndValue(end_pos)
        self.fade_out_anim.setStartValue(self.opacity_effect.opacity())
        self.fade_out_anim.setEndValue(0.0)

        self.exit_group.start()

    def _on_exit_finished(self):
        self.dismissed.emit(self)
        self.hide()
        # Defer deletion safely to prevent animation group race conditions
        QTimer.singleShot(50, self.deleteLater)


class ToastManager(QObject):
    """
    Singleton manager tracking active toasts, positioning, and smooth stacking.
    Positions toasts at the Top Right of the parent window as an in-window overlay.
    """
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ToastManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        self._active_toasts = []
        self._max_toasts = 5
        self._margin_x = 24
        self._margin_y = 75
        self._spacing = 10
        self._observed_windows = set()

    def show(self, title, message, kind=ToastType.WARNING, duration=3500, parent=None):
        target_widget = parent
        if target_widget is None:
            target_widget = QApplication.activeWindow()
        if target_widget is None:
            for w in QApplication.topLevelWidgets():
                if w.isVisible():
                    target_widget = w
                    break

        if target_widget is not None:
            top_win = target_widget.window() if hasattr(target_widget, 'window') else target_widget
            if top_win and top_win not in self._observed_windows:
                top_win.installEventFilter(self)
                self._observed_windows.add(top_win)

        toast = ToastNotification(
            title=title,
            message=message,
            kind=kind,
            duration=duration,
            parent=target_widget
        )
        toast.closing.connect(self._on_toast_closing)
        toast.dismissed.connect(self._on_toast_dismissed)

        if len(self._active_toasts) >= self._max_toasts:
            oldest = self._active_toasts[0]
            oldest.fade_out_and_close()

        self._active_toasts.append(toast)
        self._reposition_all(new_toast=toast)
        return toast

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Move):
            self._reposition_all()
        return super().eventFilter(obj, event)

    def _reposition_all(self, new_toast=None):
        if not self._active_toasts:
            return

        ref_widget = self._active_toasts[0].parent_target if self._active_toasts else None
        if ref_widget is None:
            ref_widget = QApplication.activeWindow()

        parent_w = ref_widget.width() if ref_widget else 1200
        right_x = parent_w - self._margin_x
        current_y = self._margin_y

        for toast in list(self._active_toasts):
            if getattr(toast, '_is_closing', False):
                continue
            toast.adjustSize()
            toast_w = toast.width()
            toast_h = toast.height()

            target_pos = QPoint(right_x - toast_w, current_y)
            current_y += (toast_h + self._spacing)

            if toast is new_toast:
                toast.pop_up(target_pos)
            else:
                toast.animate_to_pos(target_pos)

    def _on_toast_closing(self, toast):
        if toast in self._active_toasts:
            self._active_toasts.remove(toast)
        self._reposition_all()

    def _on_toast_dismissed(self, toast):
        if toast in self._active_toasts:
            self._active_toasts.remove(toast)


def show_toast(title, message, kind=ToastType.WARNING, duration=3500, parent=None):
    return ToastManager.instance().show(
        title=title,
        message=message,
        kind=kind,
        duration=duration,
        parent=parent
    )

def show_warning(parent=None, title="Warning", message="", duration=3500):
    return show_toast(title=title, message=message, kind=ToastType.WARNING, duration=duration, parent=parent)

def show_info(parent=None, title="Info", message="", duration=3000):
    return show_toast(title=title, message=message, kind=ToastType.INFO, duration=duration, parent=parent)

def show_error(parent=None, title="Error", message="", duration=4500):
    return show_toast(title=title, message=message, kind=ToastType.ERROR, duration=duration, parent=parent)

def show_success(parent=None, title="Success", message="", duration=3000):
    return show_toast(title=title, message=message, kind=ToastType.SUCCESS, duration=duration, parent=parent)


def install_toast_message_box_hook():
    from PyQt5.QtWidgets import QMessageBox

    if getattr(QMessageBox, "_toast_hooked", False):
        return

    _orig_warning = QMessageBox.warning
    _orig_info = QMessageBox.information
    _orig_critical = QMessageBox.critical

    def _hooked_warning(parent=None, title="", text="", buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        if buttons == QMessageBox.Ok or buttons == QMessageBox.NoButton:
            show_warning(parent, title, text)
            return QMessageBox.Ok
        return _orig_warning(parent, title, text, buttons, defaultButton)

    def _hooked_info(parent=None, title="", text="", buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        if buttons == QMessageBox.Ok or buttons == QMessageBox.NoButton:
            if title and "success" in title.lower():
                show_success(parent, title, text)
            else:
                show_info(parent, title, text)
            return QMessageBox.Ok
        return _orig_info(parent, title, text, buttons, defaultButton)

    def _hooked_critical(parent=None, title="", text="", buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        if buttons == QMessageBox.Ok or buttons == QMessageBox.NoButton:
            show_error(parent, title, text)
            return QMessageBox.Ok
        return _orig_critical(parent, title, text, buttons, defaultButton)

    QMessageBox.warning = staticmethod(_hooked_warning)
    QMessageBox.information = staticmethod(_hooked_info)
    QMessageBox.critical = staticmethod(_hooked_critical)
    QMessageBox._toast_hooked = True
