
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QStyle, QShortcut,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSizePolicy,
    QTabWidget, QLabel, QPushButton, QSpinBox, QComboBox, QCheckBox, QToolButton,
    QRadioButton, QGroupBox, QScrollArea, QTextEdit, QSlider, QLineEdit,
    QFileDialog, QMessageBox, QDoubleSpinBox, QFormLayout, QTabBar, QButtonGroup, QTreeWidget, QTreeWidgetItem,
    QGraphicsDropShadowEffect, QGraphicsItemGroup, QMenu, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsTextItem
)
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF, QPoint, QPointF, QProcess, pyqtSignal, QPropertyAnimation, QEvent, QBuffer, QByteArray, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPainterPath,QPen,QCursor,QColor,QTextCursor,QKeySequence,QTransform, QPalette, QIcon, QFont
import math
import platform
from PIL import Image
import numpy as np
from io import BytesIO
import sys
from time import time
import json

from utils import image_coords_to_latlon, check_memory_requirement


def pil_to_qimage(pil_image):
    if pil_image.mode == 'RGB':
        data = pil_image.tobytes()
        qimage = QImage(data, pil_image.size[0], pil_image.size[1], pil_image.size[0] * 3, QImage.Format_RGB888)
    elif pil_image.mode == 'L':
        data = pil_image.tobytes()
        qimage = QImage(data, pil_image.size[0], pil_image.size[1], pil_image.size[0], QImage.Format_Grayscale8)
    elif pil_image.mode == 'I':
        data = pil_image.tobytes()
        qimage = QImage(data, pil_image.size[0], pil_image.size[1], pil_image.size[0] * 2, QImage.Format_Grayscale16)
    else:
        pil_image = pil_image.convert('RGB')
        data = pil_image.tobytes()
        qimage = QImage(data, pil_image.size[0], pil_image.size[1], pil_image.size[0] * 3, QImage.Format_RGB888)
    return qimage
def qimage_to_pil(qimg):
    buf = QBuffer()
    buf.open(QBuffer.ReadWrite)
    qimg.save(buf, b"PNG")
    data = buf.data() # returns a QByteArray
    buf.close()
    img_bytes = bytes(data)
    pil_image = Image.open(BytesIO(img_bytes))
    pil_image.load()
    return pil_image


class ToolboxButton(QToolButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_text = False

    def set_internal_text(self, text):
        self._allow_text = True
        try:
            super().setText(text)
        finally:
            self._allow_text = False

    def setText(self, text):
        # Ignore external setText calls unless explicitly allowed
        if getattr(self, '_allow_text', False):
            return super().setText(text)
        return
class MagnifierGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        if platform.system() == 'Darwin':
            self.setAttribute(Qt.WA_MacMetalStyle, True)
        self.magnifier_enabled = False
        self.magnifier_center = None # QPointF in item local coordinates (original image)
        self.magnifier_radius = 100
        self.mouse_zoom_enabled = False
        self.magnifier_zoom = 8.0
        self.torch_enabled = False
        self.dragging_magnifier = False
        self.resizing_magnifier = False
        self.drag_offset = QPoint()
        self.start_mouse_view = QPoint()
        self.start_radius = 0
        self.pan_start_pos = None
        self.cached_pixmap = None # Cache for magnified pixmap
        self.cached_source_scene = None # Cache source scene rect as tuple (left, top, width, height)
        self.cached_zoom = None # Cache zoom level
        self.cached_torch = None # Cache torch state
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate) # Further optimize updates
        self.last_update_time = 0
        self.update_interval = 32
        self.setCacheMode(QGraphicsView.CacheBackground) # Cache background for faster redraws
        self.interaction_mode = "off"  # off | measure | calculate
        self.measure_enabled = False
        self.calculate_enabled = False
        # For resolving collisions when both measure+calculate are enabled
        self.pending_interaction = None  # None | 'undecided' | 'calculate'
        self.pending_press_scene = None
        self.pending_press_view = None
        self.click_move_threshold = 6  # pixels
        self.measure_points = []
        self.last_measure_points = []
        self.calculate_drag_active = False
        self.calculate_start_scene = None
        self.calculate_current_scene = None
        self.last_calculate_rect = None
        self.grid_enabled = False
        self.grid_divisions = 6
        self.grid_spacing_x = None
        self.grid_spacing_y = None
        self.grid_offset_x = 0.0
        self.grid_offset_y = 0.0
        self.grid_show_box = True
        self.grid_show_columns = True
        self.grid_show_rows = True
        self.grid_drag_axis = None
        self.grid_drag_start_orig = None
        self.grid_drag_start_spacing = 0.0
        self.grid_line_pick_tolerance = 6
        
        # Mouse click mode line
        self.mouse_click_line_enabled = False
        self.mouse_click_line_pos = None  # QPointF for current mouse position
        self.mouse_click_line_direction = "vertical"  # "vertical" or "horizontal"
        self.mouse_click_cut_pos = None  # QPointF for last clicked cut position
        self.mouse_click_line_clickable = False
        self.crop_box_enabled = False
        self.crop_box_rect = None  # QRectF in original image coordinates
        self.crop_box_drag_mode = None
        self.crop_box_drag_start = None
        self.crop_box_initial = None
        
        # === Annotation Layer ===
        self.annotations = []  # List of annotation dicts: {"type": "arrow"|"text"|"rect"|"measure", "points": [...], "color": ..., "text": ...}
        self.annotations_visible = True
        self.annotation_color = QColor(255, 0, 0, 200)  # Red with alpha
        self.annotation_pen_width = 2

    def set_interaction_mode(self, mode: str):
        mode = str(mode).lower()
        # allow a combined 'both' mode to enable both measure and calculate
        if mode not in ("off", "measure", "calculate", "both"):
            mode = "off"
        self.interaction_mode = mode
        self.measure_enabled = (mode == "measure" or mode == "both")
        self.calculate_enabled = (mode == "calculate" or mode == "both")
        if not self.measure_enabled:
            self.measure_points = []
        if not self.calculate_enabled:
            self.calculate_drag_active = False
            self.calculate_start_scene = None
            self.calculate_current_scene = None
            self.last_calculate_rect = None
        self.viewport().update()
    def toggle_magnifier(self, enabled):
        self.magnifier_enabled = enabled
        if enabled and not self.magnifier_center:
            vp_rect = self.viewport().rect()
            center_view = vp_rect.center()
            center_scene = self.mapToScene(center_view)
            x, y = self.parent().get_original_coords(center_scene)
            self.magnifier_center = QPointF(x, y)
        self.cached_pixmap = None # Invalidate cache
        self.cached_source_scene = None
        self.viewport().update()
    def toggle_torch(self, enabled):
        self.torch_enabled = enabled
        self.cached_pixmap = None # Invalidate cache
        self.cached_source_scene = None
        self.viewport().update()
    def set_magnifier_zoom(self, value):
        self.magnifier_zoom = value / 10.0
        self.cached_pixmap = None # Invalidate cache
        self.cached_source_scene = None
        self.viewport().update()
    def toggle_grid(self, enabled):
        self.grid_enabled = bool(enabled)
        if self.grid_enabled:
            self._ensure_grid_defaults()
        else:
            self.grid_drag_axis = None
            self.grid_drag_start_orig = None
        self.viewport().update()

    def set_easy_grid_options(self, show_box=None, show_columns=None, show_rows=None):
        if show_box is not None:
            self.grid_show_box = bool(show_box)
        if show_columns is not None:
            self.grid_show_columns = bool(show_columns)
        if show_rows is not None:
            self.grid_show_rows = bool(show_rows)
        if not self.grid_show_box:
            self.grid_show_columns = False
            self.grid_show_rows = False
        self.viewport().update()
    def set_grid_divisions(self, divisions):
        self.grid_divisions = max(1, min(24, int(divisions)))
        self._sync_grid_spacing()
        self.viewport().update()
    def set_mouse_click_mode(self, enabled, direction="vertical", clickable=False):
        """Enable/disable mouse click mode with line following cursor."""
        self.mouse_click_line_enabled = bool(enabled)
        self.mouse_click_line_direction = str(direction).lower()
        self.mouse_click_line_clickable = bool(clickable)
        if not enabled:
            self.mouse_click_line_pos = None
            self.mouse_click_cut_pos = None
        self.viewport().update()

    def set_crop_box_mode(self, enabled):
        self.crop_box_enabled = bool(enabled)
        if self.crop_box_enabled and (self.crop_box_rect is None or self.crop_box_rect.isEmpty()):
            self._init_crop_box_rect()
        if not self.crop_box_enabled:
            self.crop_box_drag_mode = None
            self.crop_box_drag_start = None
            self.crop_box_initial = None
        self.viewport().update()

    def _init_crop_box_rect(self):
        parent = self.parent()
        if parent is None:
            return
        fw = int(getattr(parent, "full_width", 0))
        fh = int(getattr(parent, "full_height", 0))
        if fw <= 0 or fh <= 0:
            return
        inset_x = min(max(12.0, float(fw) * 0.1), max(1.0, float(fw) / 3.0))
        inset_y = min(max(12.0, float(fh) * 0.1), max(1.0, float(fh) / 3.0))
        self.crop_box_rect = QRectF(
            inset_x,
            inset_y,
            max(1.0, float(fw) - 2.0 * inset_x),
            max(1.0, float(fh) - 2.0 * inset_y),
        )
        self._clamp_crop_box()

    def _pick_crop_box_hit(self, x, y):
        if not self.crop_box_enabled or self.crop_box_rect is None:
            return None
        rect = self.crop_box_rect
        scale = max(0.05, abs(self.transform().m11()))
        tol = max(8.0 / scale, min(40.0 / scale, max(rect.width(), rect.height()) * 0.02))
        left = abs(x - rect.left()) <= tol
        right = abs(x - rect.right()) <= tol
        top = abs(y - rect.top()) <= tol
        bottom = abs(y - rect.bottom()) <= tol
        if left and top:
            return "top-left"
        if right and top:
            return "top-right"
        if left and bottom:
            return "bottom-left"
        if right and bottom:
            return "bottom-right"
        if left:
            return "left"
        if right:
            return "right"
        if top:
            return "top"
        if bottom:
            return "bottom"
        return None

    def _cursor_for_crop_hit(self, hit):
        if hit in ("top-left", "bottom-right"):
            return Qt.SizeFDiagCursor
        if hit in ("top-right", "bottom-left"):
            return Qt.SizeBDiagCursor
        if hit in ("left", "right"):
            return Qt.SizeHorCursor
        if hit in ("top", "bottom"):
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def _clamp_crop_box(self):
        parent = self.parent()
        if parent is None or self.crop_box_rect is None:
            return
        fw = int(getattr(parent, "full_width", 0))
        fh = int(getattr(parent, "full_height", 0))
        if fw <= 0 or fh <= 0:
            return
        r = QRectF(self.crop_box_rect)
        if r.left() < 0:
            r.setLeft(0)
        if r.top() < 0:
            r.setTop(0)
        if r.right() > fw:
            r.setRight(fw)
        if r.bottom() > fh:
            r.setBottom(fh)
        if r.width() < 1:
            r.setRight(r.left() + 1)
        if r.height() < 1:
            r.setBottom(r.top() + 1)
        self.crop_box_rect = r

    def get_crop_box_bounds(self):
        if not self.crop_box_enabled or self.crop_box_rect is None:
            return None
        self._clamp_crop_box()
        rect = self.crop_box_rect.normalized()
        left = int(math.floor(rect.left()))
        top = int(math.floor(rect.top()))
        right = int(math.ceil(rect.right()))
        bottom = int(math.ceil(rect.bottom()))
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        return (left, top, right, bottom)

    def _get_grid_rect(self):
        parent = self.parent()
        if parent is None:
            return None
        fw = int(getattr(parent, "full_width", 0))
        fh = int(getattr(parent, "full_height", 0))
        if fw <= 0 or fh <= 0:
            return None
        if self.grid_show_box and self.crop_box_enabled and self.crop_box_rect is not None:
            rect = self.crop_box_rect.normalized()
            return (
                float(rect.left()),
                float(rect.top()),
                float(rect.right()),
                float(rect.bottom()),
            )
        return (0.0, 0.0, float(fw - 1), float(fh - 1))

    def _update_crop_box_drag(self, image_x, image_y):
        if not self.crop_box_enabled or self.crop_box_drag_mode is None or self.crop_box_initial is None:
            return
        rect = QRectF(self.crop_box_initial)
        dx = image_x - self.crop_box_drag_start.x()
        dy = image_y - self.crop_box_drag_start.y()
        mode = self.crop_box_drag_mode
        min_size = 1.0
        if "left" in mode:
            new_left = self.crop_box_initial.left() + dx
            if new_left > self.crop_box_initial.right() - min_size:
                new_left = self.crop_box_initial.right() - min_size
            rect.setLeft(new_left)
        if "right" in mode:
            new_right = self.crop_box_initial.right() + dx
            if new_right < self.crop_box_initial.left() + min_size:
                new_right = self.crop_box_initial.left() + min_size
            rect.setRight(new_right)
        if "top" in mode:
            new_top = self.crop_box_initial.top() + dy
            if new_top > self.crop_box_initial.bottom() - min_size:
                new_top = self.crop_box_initial.bottom() - min_size
            rect.setTop(new_top)
        if "bottom" in mode:
            new_bottom = self.crop_box_initial.bottom() + dy
            if new_bottom < self.crop_box_initial.top() + min_size:
                new_bottom = self.crop_box_initial.top() + min_size
            rect.setBottom(new_bottom)
        self.crop_box_rect = rect
        self._clamp_crop_box()
        self.viewport().update()
    def _sync_grid_spacing(self):
        bounds = self._get_grid_rect()
        if bounds is None:
            return
        left, top, right, bottom = bounds
        width = max(1.0, float(right) - float(left))
        height = max(1.0, float(bottom) - float(top))
        if width > 1:
            self.grid_spacing_x = max(2.0, width / float(max(1, self.grid_divisions)))
        if height > 1:
            self.grid_spacing_y = max(2.0, height / float(max(1, self.grid_divisions)))
        self.grid_offset_x = 0.0
        self.grid_offset_y = 0.0

    def _ensure_grid_defaults(self):
        bounds = self._get_grid_rect()
        if bounds is None:
            return
        left, top, right, bottom = bounds
        width = max(1.0, float(right) - float(left))
        height = max(1.0, float(bottom) - float(top))
        if width > 1 and (self.grid_spacing_x is None or self.grid_spacing_x <= 1):
            self.grid_spacing_x = max(2.0, width / float(max(1, self.grid_divisions)))
        if height > 1 and (self.grid_spacing_y is None or self.grid_spacing_y <= 1):
            self.grid_spacing_y = max(2.0, height / float(max(1, self.grid_divisions)))
        self.grid_offset_x = 0.0
        self.grid_offset_y = 0.0
    def _iter_grid_positions(self, max_value, spacing, offset):
        if spacing is None or spacing <= 1 or max_value <= 1:
            return []
        positions = []
        x = float(offset) % float(spacing)
        if x < 1.0:
            x += float(spacing)
        count = 0
        while x < max_value and count < 512:
            positions.append(x)
            x += float(spacing)
            count += 1
        return positions
    def _nearest_grid_distance(self, value, max_value, spacing, offset):
        if spacing is None or spacing <= 1 or max_value <= 1:
            return None
        rel = (float(value) - float(offset)) % float(spacing)
        if rel <= (float(spacing) - rel):
            dist = rel
            nearest = float(value) - rel
        else:
            dist = float(spacing) - rel
            nearest = float(value) + dist
        # Allow selection of outer border lines as well.
        if nearest < 0.0 or nearest > max_value:
            return None
        return abs(dist)
    def _pick_grid_axis(self, ox, oy):
        parent = self.parent()
        if parent is None:
            return None
        fw = int(getattr(parent, "full_width", 0))
        fh = int(getattr(parent, "full_height", 0))
        if fw <= 1 or fh <= 1:
            return None
        bounds = self._get_grid_rect()
        if bounds is None:
            return None
        self._ensure_grid_defaults()
        scale = max(0.05, abs(self.transform().m11()))
        tol_orig = max(1.2, float(self.grid_line_pick_tolerance) / scale)
        left, top, right, bottom = bounds
        dx = None
        dy = None
        if self.grid_show_columns and top - tol_orig <= oy <= bottom + tol_orig:
            dx = self._nearest_grid_distance(ox, right, self.grid_spacing_x, left)
        if self.grid_show_rows and left - tol_orig <= ox <= right + tol_orig:
            dy = self._nearest_grid_distance(oy, bottom, self.grid_spacing_y, top)
        if dx is None and dy is None:
            return None
        if dx is not None and dx <= tol_orig and (dy is None or dx <= dy):
            return "x"
        if dy is not None and dy <= tol_orig:
            return "y"
        return None
    def _draw_grid_overlay(self, painter):
        parent = self.parent()
        if parent is None:
            return
        fw = int(getattr(parent, "full_width", 0))
        fh = int(getattr(parent, "full_height", 0))
        if fw <= 1 or fh <= 1:
            return
        bounds = self._get_grid_rect()
        if bounds is None:
            return
        self._ensure_grid_defaults()
        line_shadow = QPen(QColor(0, 0, 0, 80), 1)
        line_main = QPen(QColor(255, 255, 255, 125), 1)
        border_shadow = QPen(QColor(0, 0, 0, 235), 6)
        border_main = QPen(QColor(255, 255, 255, 140), 1)

        def draw_line(ox1, oy1, ox2, oy2, shadow_pen, main_pen):
            p1_scene = parent.map_original_to_scene(float(ox1), float(oy1))
            p2_scene = parent.map_original_to_scene(float(ox2), float(oy2))
            p1_view = QPointF(self.mapFromScene(p1_scene))
            p2_view = QPointF(self.mapFromScene(p2_scene))
            painter.setPen(shadow_pen)
            painter.drawLine(p1_view + QPointF(0.5, 0.5), p2_view + QPointF(0.5, 0.5))
            painter.setPen(main_pen)
            painter.drawLine(p1_view, p2_view)

        left, top, right, bottom = bounds
        if self.grid_show_box:
            draw_line(left, top, right, top, border_shadow, border_main)
            draw_line(left, bottom, right, bottom, border_shadow, border_main)
            draw_line(left, top, left, bottom, border_shadow, border_main)
            draw_line(right, top, right, bottom, border_shadow, border_main)

        if self.grid_show_columns:
            x_positions = self._iter_grid_positions(right, self.grid_spacing_x, left)
            for x in x_positions:
                if left < x < right:
                    draw_line(x, top, x, bottom, line_shadow, line_main)
        if self.grid_show_rows:
            y_positions = self._iter_grid_positions(bottom, self.grid_spacing_y, top)
            for y in y_positions:
                if top < y < bottom:
                    draw_line(left, y, right, y, line_shadow, line_main)

    def _draw_crop_box_overlay(self, painter):
        if not self.crop_box_enabled or self.crop_box_rect is None:
            return
        parent = self.parent()
        if parent is None:
            return
        rect = self.crop_box_rect
        top_left_scene = parent.map_original_to_scene(rect.left(), rect.top())
        bottom_right_scene = parent.map_original_to_scene(rect.right(), rect.bottom())
        top_left_view = QPointF(self.mapFromScene(top_left_scene))
        bottom_right_view = QPointF(self.mapFromScene(bottom_right_scene))
        view_rect = QRectF(top_left_view, bottom_right_view).normalized()
        overlay_brush = QColor(0, 200, 255, 35)
        painter.fillRect(view_rect, overlay_brush)
        pen = QPen(QColor(0, 220, 255, 230), 2)
        painter.setPen(pen)
        painter.drawRect(view_rect)
        handle_size = 8
        half = handle_size / 2.0
        handles = [
            view_rect.topLeft(),
            view_rect.topRight(),
            view_rect.bottomLeft(),
            view_rect.bottomRight(),
            QPointF(view_rect.center().x(), view_rect.top()),
            QPointF(view_rect.center().x(), view_rect.bottom()),
            QPointF(view_rect.left(), view_rect.center().y()),
            QPointF(view_rect.right(), view_rect.center().y()),
        ]
        painter.setBrush(QColor(0, 220, 255, 220))
        for handle in handles:
            painter.drawRect(QRectF(handle.x() - half, handle.y() - half, handle_size, handle_size))

    def wheelEvent(self, event):
        # allow zoom with Ctrl+wheel OR when the viewer's Mouse Zoom checkbox is enabled
        mouse_zoom_flag = getattr(self.parent(), "mouse_zoom_enabled", False)
        if (event.modifiers() & Qt.ControlModifier) or mouse_zoom_flag:
            delta = event.angleDelta().y()
            zoom_factor = 1.25 if delta > 0 else 0.8
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(zoom_factor, zoom_factor)
            # keep magnifier/tile state consistent
            if self.magnifier_enabled:
                mouse_scene = self.mapToScene(event.pos())
                x, y = self.parent().get_original_coords(mouse_scene)
                self.magnifier_center = QPointF(x, y)
                self.cached_pixmap = None # Invalidate cache
                self.cached_source_scene = None
            self.viewport().update()
            event.accept()
            # update parent viewer's zoom state and position label
            try:
                self.parent().zoom = self.transform().m11()
                self.parent().update_position_label(event.pos())
            except Exception:
                pass
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            mouse_scene = self.mapToScene(event.pos())
            local_x, local_y = self.parent().get_original_coords(mouse_scene)
            x, y = math.floor(local_x), math.floor(local_y)
            if self.crop_box_enabled and not (self.measure_enabled or self.calculate_enabled):
                hit = self._pick_crop_box_hit(local_x, local_y)
                if hit is not None:
                    self.crop_box_drag_mode = hit
                    self.crop_box_drag_start = QPointF(local_x, local_y)
                    self.crop_box_initial = QRectF(self.crop_box_rect) if self.crop_box_rect is not None else None
                    self.setCursor(self._cursor_for_crop_hit(hit))
                    event.accept()
                    return
            if self.mouse_click_line_enabled and self.mouse_click_line_clickable and not self.grid_enabled and not self.measure_enabled and not self.calculate_enabled:
                self.mouse_click_cut_pos = QPointF(local_x, local_y)
                parent_viewer = self.parent()
                if hasattr(parent_viewer, 'on_guru_cut'):
                    parent_viewer.on_guru_cut(self.mouse_click_line_direction, x, y)
                self.viewport().update()
                event.accept()
                return
            if self.grid_enabled and not (self.measure_enabled or self.calculate_enabled):
                axis = self._pick_grid_axis(local_x, local_y)
                if axis is not None:
                    self.grid_drag_axis = axis
                    self.grid_drag_start_orig = (float(local_x), float(local_y))
                    self.grid_drag_start_spacing = self.grid_spacing_x if axis == "x" else self.grid_spacing_y
                    self.setCursor(Qt.SizeHorCursor if axis == "x" else Qt.SizeVerCursor)
                    event.accept()
                    return
            full_width = self.parent().full_width
            full_height = self.parent().full_height
            if 0 <= x < full_width and 0 <= y < full_height:
                self.parent().selection_pos = (x, y)
                self.parent()._emit_pixel_info_at(x, y)
            if self.parent().click_callback:
                self.parent().click_callback(self.parent(), x, y)
            # --- Flip system ---
            flip_mode = getattr(self.parent(), "flip_mode", 0)
            if flip_mode in (1, 2): # 1=Select, 2=Select All
                
                menu = QMenu(self)
                act_v = menu.addAction("Flip Vertical")
                act_h = menu.addAction("Flip Horizontal")
            
                act_global_v = None
                act_global_h = None
                if flip_mode == 2: # Only add global options in "Select All" mode
                    menu.addSeparator()
                    act_global_v = menu.addAction("Global Flip Vertical")
                    act_global_h = menu.addAction("Global Flip Horizontal")
                chosen = menu.exec_(event.globalPos())
                if chosen is None:
                    return
                vertical = chosen in (act_v, act_global_v)
                horizontal = chosen in (act_h, act_global_h)
                is_global = (flip_mode == 2 and chosen in (act_global_v, act_global_h))
                all_flag = (flip_mode == 2)
                if vertical or horizontal:
                    self.parent().apply_flip(vertical=vertical, horizontal=horizontal, all=all_flag, click_pos=(x, y), global_flip=is_global)
                    return # stop here so magnifier logic doesn’t also run
            
            # --- Magnifier logic (only interaction, no auto-set on left click) ---
            if self.magnifier_enabled:
                # Get visual center for interaction
                scene_center = self.parent().map_original_to_scene(self.magnifier_center.x(), self.magnifier_center.y())
                center_view = self.mapFromScene(scene_center)
                mouse_view = event.pos()
                dx = mouse_view.x() - center_view.x()
                dy = mouse_view.y() - center_view.y()
                dist = math.sqrt(dx**2 + dy**2)
                radius = self.magnifier_radius
                tol = 10
                if abs(dist - radius) <= tol:
                    self.resizing_magnifier = True
                    self.start_mouse_view = mouse_view
                    self.start_radius = radius
                    event.accept()
                    return
                elif dist < radius - tol:
                    self.dragging_magnifier = True
                    self.drag_offset = mouse_view - center_view
                    event.accept()
                    return
            
            if (self.measure_enabled and self.calculate_enabled):
                # Defer decision until movement or release.
                self.pending_interaction = 'undecided'
                self.pending_press_scene = self.mapToScene(event.pos())
                self.pending_press_view = event.pos()
                self.viewport().update()
                event.accept()
                return
            # Single-mode behaviors (unchanged)
            if self.measure_enabled and event.button() == Qt.LeftButton:
                scene_pos = self.mapToScene(event.pos())
                if len(self.measure_points) == 0 and len(self.last_measure_points) > 0:
                    self.last_measure_points = []
                    self.viewport().update()
                self.measure_points.append(scene_pos)
                if len(self.measure_points) == 2:
                    self.calculate_measurements()
                    self.last_measure_points = self.measure_points[:]
                    self.measure_points = []
                self.viewport().update()
                event.accept()
                return

            if self.calculate_enabled and event.button() == Qt.LeftButton:
                scene_pos = self.mapToScene(event.pos())
                self.calculate_drag_active = True
                self.calculate_start_scene = scene_pos
                self.calculate_current_scene = scene_pos
                self.last_calculate_rect = None
                self.viewport().update()
                event.accept()
                return
        
        elif event.button() == Qt.RightButton:
            mouse_scene = self.mapToScene(event.pos())
            local_x, local_y = self.parent().get_original_coords(mouse_scene)
            if not self.grid_enabled and not (self.measure_enabled or self.calculate_enabled):
                self.grid_enabled = True
                self.grid_divisions = max(1, self.grid_divisions)
                self._sync_grid_spacing()
                self.viewport().update()
                event.accept()
                return
            if self.grid_enabled and not (self.measure_enabled or self.calculate_enabled):
                if not self.grid_show_columns and not self.grid_show_rows:
                    event.accept()
                    return
                axis = self._pick_grid_axis(local_x, local_y)
                if axis is not None:
                    if self.grid_divisions > 1:
                        self.grid_divisions -= 1
                        self._sync_grid_spacing()
                    self.viewport().update()
                    event.accept()
                    return
                else:
                    if self.grid_divisions < 24:
                        self.grid_divisions += 1
                        self._sync_grid_spacing()
                        self.viewport().update()
                        event.accept()
                        return
            if self.magnifier_enabled:
                self.magnifier_center = QPointF(local_x, local_y)
                self.cached_pixmap = None # Invalidate cache
                self.cached_source_scene = None
                self.viewport().update()
            x, y = math.floor(local_x), math.floor(local_y)
        
        elif event.button() == Qt.MiddleButton:
            self.pan_start_pos = event.globalPos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.magnifier_enabled:
            mouse_scene = self.mapToScene(event.pos())
            local_x, local_y = self.parent().get_original_coords(mouse_scene)
            self.magnifier_center = QPointF(local_x, local_y)
            self.cached_pixmap = None # Invalidate cache
            self.cached_source_scene = None
            self.viewport().update()
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        
        current_time = time() * 1000
        if current_time - self.last_update_time < self.update_interval and not self.calculate_drag_active:
            return
        mouse_scene = self.mapToScene(event.pos())
        image_x, image_y = self.parent().get_original_coords(mouse_scene)
        
        # Update mouse click line position
        if self.mouse_click_line_enabled:
            self.mouse_click_line_pos = QPointF(image_x, image_y)
            self.viewport().update()
        
        if self.crop_box_enabled and self.crop_box_drag_mode and (event.buttons() & Qt.LeftButton):
            self._update_crop_box_drag(image_x, image_y)
            self.setCursor(self._cursor_for_crop_hit(self.crop_box_drag_mode))
            event.accept()
            return

        if self.grid_enabled and self.grid_drag_axis and (event.buttons() & Qt.LeftButton):
            self._ensure_grid_defaults()
            start = self.grid_drag_start_orig or (float(image_x), float(image_y))
            bounds = self._get_grid_rect()
            if bounds is None:
                return
            left, top, right, bottom = bounds
            width = max(1.0, float(right) - float(left))
            height = max(1.0, float(bottom) - float(top))
            if self.grid_drag_axis == "x" and self.grid_spacing_x and self.grid_spacing_x > 0 and width > 1:
                min_spacing = max(2.0, width / 24.0)
                max_spacing = max(min_spacing, width)
                delta = float(image_x) - float(start[0])
                spacing_axis = max(min_spacing, min(max_spacing, float(self.grid_drag_start_spacing) + delta))
                ratio = width / max(1.0, spacing_axis)
                self.grid_divisions = int(max(1, min(24, math.ceil(ratio - 1e-9))))
                self._sync_grid_spacing()
                self.setCursor(Qt.SizeHorCursor)
            elif self.grid_drag_axis == "y" and self.grid_spacing_y and self.grid_spacing_y > 0 and height > 1:
                min_spacing = max(2.0, height / 24.0)
                max_spacing = max(min_spacing, height)
                delta = float(image_y) - float(start[1])
                spacing_axis = max(min_spacing, min(max_spacing, float(self.grid_drag_start_spacing) + delta))
                ratio = height / max(1.0, spacing_axis)
                self.grid_divisions = int(max(1, min(24, math.ceil(ratio - 1e-9))))
                self._sync_grid_spacing()
                self.setCursor(Qt.SizeVerCursor)
            self.viewport().update()
            self.last_update_time = current_time
            event.accept()
            return
        self.parent().update_position_label(event.pos())
        if not (self.measure_enabled or self.calculate_enabled):
            self.parent()._emit_pixel_info_at(math.floor(image_x), math.floor(image_y))
        magnifier_cursor = Qt.ArrowCursor
        grid_cursor = Qt.ArrowCursor
        if self.grid_enabled and not (self.measure_enabled or self.calculate_enabled):
            axis = self._pick_grid_axis(image_x, image_y)
            if axis == "x":
                grid_cursor = Qt.SizeHorCursor
            elif axis == "y":
                grid_cursor = Qt.SizeVerCursor
        if self.crop_box_enabled and not self.crop_box_drag_mode:
            crop_hit = self._pick_crop_box_hit(image_x, image_y)
            if crop_hit is not None:
                grid_cursor = self._cursor_for_crop_hit(crop_hit)
        if self.magnifier_enabled:
            # Get visual center for interaction
            scene_center = self.parent().map_original_to_scene(self.magnifier_center.x(), self.magnifier_center.y())
            center_view = self.mapFromScene(scene_center)
            mouse_view = event.pos()
            dx = mouse_view.x() - center_view.x()
            dy = mouse_view.y() - center_view.y()
            dist = math.sqrt(dx**2 + dy**2)
            radius = self.magnifier_radius
            tol = 10
            if abs(dist - radius) <= tol:
                magnifier_cursor = Qt.SizeFDiagCursor
            elif dist < radius - tol:
                magnifier_cursor = Qt.SizeAllCursor
            # Set cursor for hint
            self.setCursor(magnifier_cursor)
            # Only perform drag/resize if not measuring
            if not (self.measure_enabled or self.calculate_enabled):
                if self.dragging_magnifier:
                    new_center_view = mouse_view - self.drag_offset
                    new_scene = self.mapToScene(new_center_view)
                    x, y = self.parent().get_original_coords(new_scene)
                    self.magnifier_center = QPointF(x, y)
                    self.cached_pixmap = None # Invalidate cache on move
                    self.cached_source_scene = None
                    self.viewport().update()
                elif self.resizing_magnifier:
                    dx = mouse_view.x() - center_view.x()
                    dy = mouse_view.y() - center_view.y()
                    current_dist = math.sqrt(dx**2 + dy**2)
                    start_dx = self.start_mouse_view.x() - center_view.x()
                    start_dy = self.start_mouse_view.y() - center_view.y()
                    start_dist = math.sqrt(start_dx**2 + start_dy**2)
                    new_radius = self.start_radius + (current_dist - start_dist)
                    self.magnifier_radius = max(1, int(new_radius))
                    self.cached_pixmap = None # Invalidate cache on resize
                    self.cached_source_scene = None
                    self.viewport().update()
        if event.buttons() & Qt.MiddleButton and self.pan_start_pos:
            current_pos = event.globalPos()
            delta = current_pos - self.pan_start_pos
            self.pan_start_pos = current_pos
            h_scroll = self.horizontalScrollBar()
            v_scroll = self.verticalScrollBar()
            h_scroll.setValue(h_scroll.value() - delta.x())
            v_scroll.setValue(v_scroll.value() - delta.y())
            event.accept()
        # If both enabled and user has a pending press, decide on drag -> calculate
        if (self.measure_enabled and self.calculate_enabled and self.pending_interaction == 'undecided'):
            if event.buttons() & Qt.LeftButton and self.pending_press_view is not None:
                dv = event.pos() - self.pending_press_view
                dist = math.hypot(dv.x(), dv.y())
                if dist >= self.click_move_threshold:
                    # Treat as calculate drag start
                    self.pending_interaction = 'calculate'
                    self.calculate_drag_active = True
                    self.calculate_start_scene = self.pending_press_scene
                    self.calculate_current_scene = mouse_scene
                    self.last_calculate_rect = None
                    self.viewport().update()
        if self.calculate_enabled and self.calculate_drag_active:
            self.calculate_current_scene = mouse_scene
            self.viewport().update()
        # Final cursor decision
        if self.measure_enabled or self.calculate_enabled:
            if magnifier_cursor != Qt.ArrowCursor:
                # Provide hint with magnifier cursor even during measure
                self.setCursor(magnifier_cursor)
            else:
                self.setCursor(Qt.CrossCursor)
        else:
            if magnifier_cursor != Qt.ArrowCursor:
                self.setCursor(magnifier_cursor)
            elif grid_cursor != Qt.ArrowCursor:
                self.setCursor(grid_cursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        self.last_update_time = current_time
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.crop_box_drag_mode is not None:
                self.crop_box_drag_mode = None
                self.crop_box_drag_start = None
                self.crop_box_initial = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            if self.grid_drag_axis is not None:
                self.grid_drag_axis = None
                self.grid_drag_start_orig = None
                self.setCursor(Qt.ArrowCursor)
                event.accept()
                return
            self.dragging_magnifier = False
            self.resizing_magnifier = False
            # If we were in combined-mode and pending undecided, resolve as click -> measure
            if self.pending_interaction == 'undecided':
                scene_pos = self.pending_press_scene or self.mapToScene(event.pos())
                # Append measure point
                if len(self.measure_points) == 0 and len(self.last_measure_points) > 0:
                    self.last_measure_points = []
                    self.viewport().update()
                self.measure_points.append(scene_pos)
                if len(self.measure_points) == 2:
                    self.calculate_measurements()
                    self.last_measure_points = self.measure_points[:]
                    self.measure_points = []
                # clear pending
                self.pending_interaction = None
                self.pending_press_scene = None
                self.pending_press_view = None
                self.viewport().update()
            elif self.calculate_enabled and self.calculate_drag_active:
                self.calculate_drag_active = False
                self.calculate_current_scene = self.mapToScene(event.pos())
                self.finalize_calculate_rect()
        if event.button() == Qt.MiddleButton:
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def finalize_calculate_rect(self):
        if not self.calculate_start_scene or not self.calculate_current_scene:
            return
        p1 = self.calculate_start_scene
        p2 = self.calculate_current_scene
        self.last_calculate_rect = (p1, p2)
        self.viewport().update()
        self.calculate_region_statistics(p1, p2)

    def calculate_region_statistics(self, p1, p2):
        try:
            # Get float bounds from original coordinates
            ox1, oy1 = self.parent().get_original_coords(p1)
            ox2, oy2 = self.parent().get_original_coords(p2)
            
            # Convert to integer indices using floor for min and ceil for max
            x0 = int(math.floor(min(ox1, ox2)))
            x1 = int(math.ceil(max(ox1, ox2)))
            y0 = int(math.floor(min(oy1, oy2)))
            y1 = int(math.ceil(max(oy1, oy2)))
            
            fw = int(getattr(self.parent(), "full_width", 0))
            fh = int(getattr(self.parent(), "full_height", 0))
            if fw <= 0 or fh <= 0:
                return
            
            # Clamp to image bounds
            x0 = max(0, min(fw, x0))
            x1 = max(0, min(fw, x1))
            y0 = max(0, min(fh, y0))
            y1 = max(0, min(fh, y1))
            
            if x1 <= x0 or y1 <= y0:
                return
            
            data_src = self.parent().original_raw_data if getattr(self.parent(), "original_raw_data", None) is not None else self.parent().original_image_data
            if data_src is None:
                return
            
            # Use half-open interval [y0:y1, x0:x1)
            roi = data_src[y0:y1, x0:x1]
            if roi is None or getattr(roi, "size", 0) == 0:
                return
            
            roi_f = roi.astype(np.float64, copy=False).ravel()
            mean_val = float(np.mean(roi_f))
            var_val = float(np.var(roi_f))
            std_val = float(np.std(roi_f))
            try:
                min_val = float(np.min(roi_f))
            except Exception:
                min_val = None
            try:
                max_val = float(np.max(roi_f))
            except Exception:
                max_val = None
            try:
                # Compute width and height directly from indices
                width = x1 - x0
                height = y1 - y0
                count_val = int(width * height)
            except Exception:
                count_val = None
            self.parent()._emit_pixel_info_at((x0 + x1) // 2, (y0 + y1) // 2)
            app = None
            try:
                app = self.parent().get_app()
            except Exception:
                app = None
            # Respect viewer toolbox state: if viewer has both tools checked, set 'both'
            try:
                parent_viewer = self.parent()
                mode_for_ui = 'calculate'
                if parent_viewer is not None and hasattr(parent_viewer, 'action_measure') and hasattr(parent_viewer, 'action_calculate'):
                    m_checked = parent_viewer.action_measure.isChecked()
                    c_checked = parent_viewer.action_calculate.isChecked()
                    if m_checked and c_checked:
                        mode_for_ui = 'both'
                    elif m_checked:
                        mode_for_ui = 'measure'
                    elif c_checked:
                        mode_for_ui = 'calculate'
                if app and hasattr(app, 'pixel_info_box'):
                    try:
                        app.pixel_info_box.set_interaction_mode(mode_for_ui)
                        app.pixel_info_box.update_calculations(mean_val, var_val, std_val, min_val, max_val, (count_val, width, height))
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                parent = self.parent()
                if parent is not None and hasattr(parent, 'pixel_info_box_overlay') and parent.pixel_info_box_overlay is not None:
                    try:
                        parent.pixel_info_box_overlay.set_interaction_mode(mode_for_ui)
                        parent.pixel_info_box_overlay.update_calculations(mean_val, var_val, std_val, min_val, max_val, (count_val, width, height))
                    except Exception:
                        pass
                if hasattr(self, 'pixel_info_box_overlay') and self.pixel_info_box_overlay is not None:
                    try:
                        self.pixel_info_box_overlay.set_interaction_mode(mode_for_ui)
                        self.pixel_info_box_overlay.update_calculations(mean_val, var_val, std_val, min_val, max_val, (count_val, width, height))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
    def calculate_measurements(self):
        if len(self.measure_points) != 2:
            return
        p1, p2 = self.measure_points
        ox1, oy1 = self.parent().get_original_coords(p1)
        ox2, oy2 = self.parent().get_original_coords(p2)
        # Assume O is (ox1, oy2), A (ox1, oy1), B (ox2, oy2)
        oa = abs(oy1 - oy2)
        ob = abs(ox2 - ox1)
        ab = math.ceil(math.sqrt(oa**2 + ob**2))

        try:
            mid_ox = int(round((ox1 + ox2) / 2.0))
            mid_oy = int(round((oy1 + oy2) / 2.0))
            # Prefer using the viewer's helper so sampling + callback behavior is consistent
            if hasattr(self.parent(), '_emit_pixel_info_at'):
                # clamp to image bounds
                mx = max(0, min(self.parent().full_width - 1, mid_ox))
                my = max(0, min(self.parent().full_height - 1, mid_oy))
                try:
                    self.parent()._emit_pixel_info_at(mx, my)
                except Exception:
                    pass
        except Exception:
            pass

        # Get the app using existing get_app and ensure central PixelInfoBox is in measure mode
        try:
            app = self.parent().get_app()
            if app and hasattr(app, 'pixel_info_box'):
                try:
                    # Respect viewer toolbox state: may be 'both' if both checked
                    parent_viewer = self.parent()
                    mode_for_ui = 'measure'
                    if parent_viewer is not None and hasattr(parent_viewer, 'action_measure') and hasattr(parent_viewer, 'action_calculate'):
                        m_checked = parent_viewer.action_measure.isChecked()
                        c_checked = parent_viewer.action_calculate.isChecked()
                        if m_checked and c_checked:
                            mode_for_ui = 'both'
                        elif m_checked:
                            mode_for_ui = 'measure'
                        elif c_checked:
                            mode_for_ui = 'calculate'
                    app.pixel_info_box.set_interaction_mode(mode_for_ui)
                except Exception:
                    pass
        except Exception:
            app = None

        # Update app-level PixelInfoBox if present (BandStitchProApp)
        if app and hasattr(app, 'pixel_info_box'):
            try:
                app.pixel_info_box.update_measurements(oa, ob, ab)
            except Exception:
                pass

        try:
            parent = self.parent()
            if parent is not None and hasattr(parent, 'pixel_info_box_overlay') and parent.pixel_info_box_overlay is not None:
                try:
                    # Use same mode resolution as above
                    parent_viewer = parent
                    mode_for_ui = 'measure'
                    if parent_viewer is not None and hasattr(parent_viewer, 'action_measure') and hasattr(parent_viewer, 'action_calculate'):
                        m_checked = parent_viewer.action_measure.isChecked()
                        c_checked = parent_viewer.action_calculate.isChecked()
                        if m_checked and c_checked:
                            mode_for_ui = 'both'
                        elif m_checked:
                            mode_for_ui = 'measure'
                        elif c_checked:
                            mode_for_ui = 'calculate'
                    parent.pixel_info_box_overlay.set_interaction_mode(mode_for_ui)
                    parent.pixel_info_box_overlay.set_measure_mode(True)
                    parent.pixel_info_box_overlay.update_measurements(oa, ob, ab)
                except Exception:
                    pass
            # RawViewer attaches the overlay to the GraphicsImageViewer instance itself
            if hasattr(self, 'pixel_info_box_overlay') and self.pixel_info_box_overlay is not None:
                # Ensure measure-mode is enabled so the tuple is rendered even if no recent pixel-value update
                try:
                    # Mirror the same resolution
                    pv = self
                    mode_for_ui = 'measure'
                    if pv is not None and hasattr(pv, 'action_measure') and hasattr(pv, 'action_calculate'):
                        m_checked = pv.action_measure.isChecked()
                        c_checked = pv.action_calculate.isChecked()
                        if m_checked and c_checked:
                            mode_for_ui = 'both'
                        elif m_checked:
                            mode_for_ui = 'measure'
                        elif c_checked:
                            mode_for_ui = 'calculate'
                    self.pixel_info_box_overlay.set_interaction_mode(mode_for_ui)
                    self.pixel_info_box_overlay.set_measure_mode(True)
                    self.pixel_info_box_overlay.update_measurements(oa, ob, ab)
                except Exception:
                    # As a last-resort fallback, force the measurement to be shown
                    try:
                        self.pixel_info_box_overlay.force_show_measurements(oa, ob, ab)
                    except Exception:
                        pass
        except Exception:
            pass

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.grid_enabled:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                self._draw_grid_overlay(painter)
            finally:
                painter.end()
        if self.crop_box_enabled and self.crop_box_rect is not None:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                self._draw_crop_box_overlay(painter)
            finally:
                painter.end()
        if not self.magnifier_enabled or not self.scene().items():
            pass
        else:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                # Map local center to scene and then to view
                scene_center = self.parent().map_original_to_scene(self.magnifier_center.x(), self.magnifier_center.y())
                center_view = self.mapFromScene(scene_center)
                radius = self.magnifier_radius
                source_half_view = radius / self.magnifier_zoom
                source_view_rect = QRectF(center_view.x() - source_half_view,
                                        center_view.y() - source_half_view,
                                        2 * source_half_view, 2 * source_half_view)
                if source_view_rect.isEmpty() or source_view_rect.width() <= 0 or source_view_rect.height() <= 0:
                    pass
                else:
                    # Map the four corners of source_view_rect to scene coordinates and compute bounding rect
                    tl = source_view_rect.topLeft()
                    tr = source_view_rect.topRight()
                    bl = source_view_rect.bottomLeft()
                    br = source_view_rect.bottomRight()
                    scene_tl = self.mapToScene(QPoint(int(tl.x()), int(tl.y())))
                    scene_tr = self.mapToScene(QPoint(int(tr.x()), int(tr.y())))
                    scene_bl = self.mapToScene(QPoint(int(bl.x()), int(bl.y())))
                    scene_br = self.mapToScene(QPoint(int(br.x()), int(br.y())))
                    min_x = min(scene_tl.x(), scene_tr.x(), scene_bl.x(), scene_br.x())
                    max_x = max(scene_tl.x(), scene_tr.x(), scene_bl.x(), scene_br.x())
                    min_y = min(scene_tl.y(), scene_tr.y(), scene_bl.y(), scene_br.y())
                    max_y = max(scene_tl.y(), scene_tr.y(), scene_bl.y(), scene_br.y())
                    source_scene_rect = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)
                    source_scene_tuple = (source_scene_rect.left(), source_scene_rect.top(),
                                        source_scene_rect.width(), source_scene_rect.height())
                    # Check cache
                    if (self.cached_pixmap and
                        self.cached_source_scene == source_scene_tuple and
                        self.cached_zoom == self.magnifier_zoom and
                        self.cached_torch == self.torch_enabled):
                        scaled_pixmap = self.cached_pixmap
                    else:
                        target_width = int(2 * radius)
                        target_height = int(2 * radius)
                        temp_pixmap = QPixmap(target_width, target_height)
                        temp_painter = QPainter(temp_pixmap)
                        self.scene().render(temp_painter, QRectF(0, 0, target_width, target_height),
                                            source_scene_rect, Qt.KeepAspectRatio)
                        temp_painter.end()
                        scaled_pixmap = temp_pixmap
                        if self.torch_enabled:
                            qimg = temp_pixmap.toImage()
                            sub_image = qimage_to_pil(qimg)
                            img_array = np.array(sub_image)
                            if img_array.size == 0:
                                scaled_pixmap = QPixmap()
                            else:
                                if img_array.ndim == 3 and img_array.shape[2] >= 3: # RGB/RGBA image
                                    rgb = img_array[..., :3]
                                    min_val = np.min(rgb, axis=(0, 1), keepdims=True)
                                    max_val = np.max(rgb, axis=(0, 1), keepdims=True)
                                    scale = np.where(max_val - min_val > 0, 255 / (max_val - min_val + 1e-5), 1)
                                    enhanced = ((rgb.astype(np.float32) - min_val) * scale).clip(0, 255).astype(np.uint8)
                                    enhanced_pil = Image.fromarray(enhanced)
                                    qimg_enh = pil_to_qimage(enhanced_pil)
                                    scaled_pixmap = QPixmap.fromImage(qimg_enh)
                                else: # Grayscale
                                    min_val = np.min(img_array)
                                    max_val = np.max(img_array)
                                    scale = 255 / (max_val - min_val + 1e-5) if max_val - min_val > 0 else 1
                                    enhanced = ((img_array.astype(np.float32) - min_val) * scale).clip(0, 255).astype(np.uint8)
                                    enhanced_pil = Image.fromarray(enhanced, mode='L')
                                    qimg_enh = pil_to_qimage(enhanced_pil)
                                    scaled_pixmap = QPixmap.fromImage(qimg_enh)
                        self.cached_pixmap = scaled_pixmap
                        self.cached_source_scene = source_scene_tuple
                        self.cached_zoom = self.magnifier_zoom
                        self.cached_torch = self.torch_enabled
                    path = QPainterPath()
                    path.addEllipse(center_view, radius, radius)
                    painter.setClipPath(path)
                    painter.drawPixmap(
                        QRectF(center_view.x() - radius, center_view.y() - radius, 2 * radius, 2 * radius),
                        scaled_pixmap,
                        QRectF(0, 0, scaled_pixmap.width(), scaled_pixmap.height())
                    )
                    painter.setClipping(False)
                    pen = QPen(Qt.white, 4)
                    painter.setPen(pen)
                    painter.drawEllipse(center_view, radius, radius)
                    pen = QPen(Qt.black, 2)
                    painter.setPen(pen)
                    painter.drawEllipse(center_view, radius, radius)
                    pen = QPen(QColor(255, 255, 255, 128), 2)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(center_view.x() - 10, center_view.y()), QPointF(center_view.x() + 10, center_view.y()))
                    painter.drawLine(QPointF(center_view.x(), center_view.y() - 10), QPointF(center_view.x(), center_view.y() + 10))
        
            finally:
                painter.end()
        if self.measure_enabled:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                points = self.measure_points if self.measure_points else self.last_measure_points
                for p_scene in points:
                    p_view = self.mapFromScene(p_scene)
                    pen = QPen(Qt.red, 2)
                    painter.setPen(pen)
                    painter.drawLine(QPointF(p_view.x() - 5, p_view.y()), QPointF(p_view.x() + 5, p_view.y()))
                    painter.drawLine(QPointF(p_view.x(), p_view.y() - 5), QPointF(p_view.x(), p_view.y() + 5))
                    painter.setBrush(QColor(255, 0, 0))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(p_view, 3, 3)
                if len(points) == 2:
                    p1_scene, p2_scene = points
                    p1_view = self.mapFromScene(p1_scene)
                    p2_view = self.mapFromScene(p2_scene)
                    # Get original image coordinates
                    ox1, oy1 = self.parent().get_original_coords(p1_scene)
                    ox2, oy2 = self.parent().get_original_coords(p2_scene)
                    # Compute O in original coords: (ox1, oy2)
                    o_orig = QPointF(ox1, oy2)
                    # Map O to scene, then to view
                    o_scene = self.parent().map_original_to_scene(o_orig.x(), o_orig.y())
                    o_view = self.mapFromScene(o_scene)
                    # Draw hypotenuse AB: thick yellow
                    pen_ab = QPen(QColor(255, 255, 0), 3, Qt.SolidLine) # Yellow, thicker
                    painter.setPen(pen_ab)
                    painter.drawLine(p1_view, p2_view)
                    # Draw vertical OA (A to O): green, dashed for distinction
                    pen_oa = QPen(QColor(0, 255, 0), 2, Qt.DashLine) # Green, dashed
                    painter.setPen(pen_oa)
                    painter.drawLine(p1_view, o_view)
                    # Draw horizontal OB (O to B): blue, dashed
                    pen_ob = QPen(QColor(0, 0, 255), 2, Qt.DashLine) # Blue, dashed
                    painter.setPen(pen_ob)
                    painter.drawLine(o_view, p2_view)
            finally:
                painter.end()
        
        # === Draw Mouse Click Line ===
        if self.mouse_click_line_enabled and self.mouse_click_line_pos is not None:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                # Draw line following mouse cursor
                line_color = QColor(255, 0, 255, 255)  # Magenta with full alpha
                pen = QPen(line_color, 4, Qt.DashLine)  # Thicker line
                painter.setPen(pen)
                
                # Convert image coordinates to view coordinates
                line_scene = self.parent().map_original_to_scene(
                    self.mouse_click_line_pos.x(), self.mouse_click_line_pos.y()
                )
                line_view = self.mapFromScene(line_scene)
                
                if self.mouse_click_line_direction == "vertical":
                    painter.drawLine(
                        QPointF(line_view.x(), 0),
                        QPointF(line_view.x(), self.viewport().height())
                    )
                else:  # horizontal
                    painter.drawLine(
                        QPointF(0, line_view.y()),
                        QPointF(self.viewport().width(), line_view.y())
                    )
            finally:
                painter.end()

        if self.mouse_click_cut_pos is not None:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                cut_color = QColor(255, 0, 255, 255)
                cut_pen = QPen(cut_color, 3, Qt.SolidLine)
                painter.setPen(cut_pen)
                cut_scene = self.parent().map_original_to_scene(
                    self.mouse_click_cut_pos.x(), self.mouse_click_cut_pos.y()
                )
                cut_view = self.mapFromScene(cut_scene)
                if self.mouse_click_line_direction == "vertical":
                    painter.drawLine(
                        QPointF(cut_view.x(), 0),
                        QPointF(cut_view.x(), self.viewport().height())
                    )
                else:
                    painter.drawLine(
                        QPointF(0, cut_view.y()),
                        QPointF(self.viewport().width(), cut_view.y())
                    )
            finally:
                painter.end()
        
        # === Draw Annotations ===
        if self.annotations and self.annotations_visible:
            painter = QPainter(self.viewport())
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                for ann in self.annotations:
                    self.draw_annotation(painter, ann)
            finally:
                painter.end()
    
    # ============ Annotation Layer ============
    def add_annotation(self, annotation_dict):
        """Add annotation: {"type": "arrow"|"text"|"rect"|"measure", "p1": ..., "p2": ..., "color": ..., "text": ...}"""
        self.annotations.append(annotation_dict)
        self.viewport().update()
    
    def clear_annotations(self):
        """Clear all annotations."""
        self.annotations = []
        self.viewport().update()
    
    def toggle_annotations(self):
        """Toggle annotation visibility."""
        self.annotations_visible = not self.annotations_visible
        self.viewport().update()
    
    def draw_annotation(self, painter, ann):
        """Draw a single annotation on the painter."""
        if not self.annotations_visible or not ann:
            return
        
        ann_type = ann.get("type", "arrow")
        color = QColor(ann.get("color", "#FF0000"))
        pen = QPen(color, self.annotation_pen_width)
        painter.setPen(pen)
        
        try:
            if ann_type == "arrow":
                # Draw line with arrowhead
                p1 = ann.get("p1", QPointF(0, 0))
                p2 = ann.get("p2", QPointF(10, 10))
                painter.drawLine(p1, p2)
                # Simple arrowhead
                angle = math.atan2(p2.y() - p1.y(), p2.x() - p1.x())
                arrow_size = 15
                p2_1 = QPointF(p2.x() - arrow_size * math.cos(angle - math.pi / 6),
                               p2.y() - arrow_size * math.sin(angle - math.pi / 6))
                p2_2 = QPointF(p2.x() - arrow_size * math.cos(angle + math.pi / 6),
                               p2.y() - arrow_size * math.sin(angle + math.pi / 6))
                painter.drawLine(p2, p2_1)
                painter.drawLine(p2, p2_2)
            
            elif ann_type == "rect":
                rect = QRectF(ann.get("rect", QRectF(0, 0, 50, 50)))
                painter.drawRect(rect)
            
            elif ann_type == "measure":
                # Line with distance text
                p1 = ann.get("p1", QPointF(0, 0))
                p2 = ann.get("p2", QPointF(10, 10))
                painter.drawLine(p1, p2)
                dist = math.sqrt((p2.x() - p1.x())**2 + (p2.y() - p1.y())**2)
                mid = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
                font = QFont()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(int(mid.x()), int(mid.y()), f"{dist:.1f}px")
            
            elif ann_type == "text":
                font = QFont()
                font.setPointSize(10)
                painter.setFont(font)
                pos = ann.get("pos", QPointF(0, 0))
                text = ann.get("text", "Label")
                painter.drawText(int(pos.x()), int(pos.y()), text)
        except Exception:
            pass  # Silently skip malformed annotations
    
    def save_annotations_json(self, filepath):
        """Save annotations to JSON file (JSON-serializable dicts)."""
        try:
            serializable = []
            for ann in self.annotations:
                sann = ann.copy()
                # Convert QPointF/QRectF to dicts for JSON
                if "p1" in sann and isinstance(sann["p1"], QPointF):
                    sann["p1"] = {"x": sann["p1"].x(), "y": sann["p1"].y()}
                if "p2" in sann and isinstance(sann["p2"], QPointF):
                    sann["p2"] = {"x": sann["p2"].x(), "y": sann["p2"].y()}
                if "pos" in sann and isinstance(sann["pos"], QPointF):
                    sann["pos"] = {"x": sann["pos"].x(), "y": sann["pos"].y()}
                if "rect" in sann and isinstance(sann["rect"], QRectF):
                    r = sann["rect"]
                    sann["rect"] = {"x": r.x(), "y": r.y(), "w": r.width(), "h": r.height()}
                serializable.append(sann)
            with open(filepath, 'w') as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            print(f"Failed to save annotations: {e}")
    
    def load_annotations_json(self, filepath):
        """Load annotations from JSON file."""
        try:
            with open(filepath, 'r') as f:
                serializable = json.load(f)
            self.annotations = []
            for sann in serializable:
                ann = sann.copy()
                # Convert dicts back to QPointF/QRectF
                if isinstance(ann.get("p1"), dict):
                    ann["p1"] = QPointF(ann["p1"]["x"], ann["p1"]["y"])
                if isinstance(ann.get("p2"), dict):
                    ann["p2"] = QPointF(ann["p2"]["x"], ann["p2"]["y"])
                if isinstance(ann.get("pos"), dict):
                    ann["pos"] = QPointF(ann["pos"]["x"], ann["pos"]["y"])
                if isinstance(ann.get("rect"), dict):
                    r = ann["rect"]
                    ann["rect"] = QRectF(r["x"], r["y"], r["w"], r["h"])
                self.annotations.append(ann)
            self.viewport().update()
        except Exception as e:
            print(f"Failed to load annotations: {e}")

class GraphicsImageViewer(QWidget):
    def __init__(self, parent=None, pixel_info_callback=None, matrix_size_var=None, click_callback=None):
        super().__init__(parent)
        self.click_callback = click_callback
        self.pixel_info_callback = pixel_info_callback
        self.matrix_size_var = matrix_size_var
        self.original_image_data = None
        self.current_pil_image = None
        self.selection_pos = None
        self.full_width = 0
        self.full_height = 0
        self.zoom = 1.0
        self.global_rotation = 0
        self.local_rotation = 0
        self.geo_info = None
        self.global_vertical_flipped = False
        self.global_horizontal_flipped = False
        # Only RGB Fusion should render RGB pixel info formatting.
        self.is_rgb_fusion = False
        self._bottom_layout = None # To store the bottom_bar layout reference
        self.frame_items = [] # For local rotation frames
        self.frame_group = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 10)
      
        # Control bar
        control_bar = QHBoxLayout()
        layout.addLayout(control_bar)
        control_bar.addStretch()
        self.position_label = QLabel("X: 0, Y: 0 | Zoom: 100% ")
        control_bar.addWidget(self.position_label)
        # Graphics view setup
        self.scene = QGraphicsScene(self)
        self.graphics_view = MagnifierGraphicsView(self)
        self.graphics_view.setScene(self.scene)
        # Ensure scrollbars are available when rotated/zoomed content exceeds the viewport
        self.graphics_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.graphics_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.graphics_view)
        self.mouse_zoom_enabled = False # toggled by Mouse Zoom checkbox
       
        self.bottom_bar_widget = QWidget(self)
        self.bottom_bar_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        bottom_bar = QHBoxLayout(self.bottom_bar_widget)
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        self._bottom_layout = bottom_bar
        bottom_bar.addStretch()

        
        self.interaction_modes = ["off", "measure", "calculate"]
        self.interaction_mode_labels = {
            "off": "Off",
            "measure": "Measure",
            "calculate": "Calculate",
            "both": "Measure+Calc",
        }
        self.interaction_mode_styles = {
            "off": "background-color: #E57373; color: white;",
            "measure": "background-color: #4CAF50; color: white;",
            "calculate": "background-color: #FF9800; color: white;",
            "both": "background-color: #8E24AA; color: white;",
        }
        def _apply_interaction_mode(mode: str):
            mode = str(mode).lower()
            if mode not in self.interaction_mode_labels:
                mode = "off"
            try:
                self.graphics_view.set_interaction_mode(mode)
                # update toolbox visual (if exists)
                if hasattr(self, 'toolbox_btn'):
                    # Keep the button labeled 'Toolbox' and update a nearby status label
                    if hasattr(self, 'toolbox_status_label'):
                        try:
                            self.toolbox_status_label.setText(self.interaction_mode_labels[mode])
                            # Try to apply style if possible
                            self.toolbox_status_label.setStyleSheet(self.interaction_mode_styles[mode] + " padding-left:6px; font-weight:600;")
                        except Exception:
                            pass
                    try:
                        self.toolbox_btn.setStyleSheet(self.interaction_mode_styles[mode])
                    except Exception:
                        pass
                parent = self.parent()
                app = parent.get_app() if (parent is not None and hasattr(parent, 'get_app')) else None
                if app and hasattr(app, 'pixel_info_box'):
                    app.pixel_info_box.set_interaction_mode(mode)
                if parent and hasattr(parent, 'pixel_info_box_overlay') and parent.pixel_info_box_overlay is not None:
                    parent.pixel_info_box_overlay.set_interaction_mode(mode)
                if hasattr(self, 'pixel_info_box_overlay') and self.pixel_info_box_overlay is not None:
                    self.pixel_info_box_overlay.set_interaction_mode(mode)
            except Exception:
                pass

        
        self.toolbox_btn = ToolboxButton()
        self.toolbox_btn.set_internal_text("Toolbox")
        self.toolbox_btn.setToolTip("Toolbox: choose Measure and/or Calculate")
        self.toolbox_btn.setStyleSheet(self.interaction_mode_styles['off'])
        bottom_bar.addWidget(self.toolbox_btn)
        # status label next to toolbox showing current selection
        self.toolbox_status_label = QLabel(self.interaction_mode_labels['off'])
        self.toolbox_status_label.setStyleSheet("color: white; padding-left:6px; font-weight:600;")
        bottom_bar.addWidget(self.toolbox_status_label)
        # Backwards compatibility: some external code references `measure_mode_btn`
        self.measure_mode_btn = self.toolbox_btn
        self.toolbox_menu = QMenu(self)
        self.action_measure = self.toolbox_menu.addAction("Measure")
        self.action_measure.setCheckable(True)
        self.action_calculate = self.toolbox_menu.addAction("Calculate")
        self.action_calculate.setCheckable(True)

        def _on_tool_toggled(tool, checked):
            try:
                # determine resulting mode
                m = self.action_measure.isChecked()
                c = self.action_calculate.isChecked()
                if m and c:
                    mode = 'both'
                elif m:
                    mode = 'measure'
                elif c:
                    mode = 'calculate'
                else:
                    mode = 'off'
                _apply_interaction_mode(mode)
            except Exception:
                pass

        self.action_measure.toggled.connect(lambda ch: _on_tool_toggled('measure', ch))
        self.action_calculate.toggled.connect(lambda ch: _on_tool_toggled('calculate', ch))

        def _show_tool_menu():
            menu = self.toolbox_menu
            pos = self.toolbox_btn.mapToGlobal(QPoint(0, -menu.sizeHint().height()))
            menu.exec_(pos)

        self.toolbox_btn.clicked.connect(_show_tool_menu)
        _apply_interaction_mode("off")
        self.magnifier_toggle = QCheckBox("Magnifier")
        bottom_bar.addWidget(self.magnifier_toggle)    
        self.torch_toggle = QCheckBox("Torch")
        bottom_bar.addWidget(self.torch_toggle)
        self.torch_toggle.setChecked(False)
        self.torch_toggle.setEnabled(False)  # Only enabled when magnifier is ON
        def _on_magnifier_toggled(checked: bool):
            if not checked:
                self.torch_toggle.setChecked(False)
                try:
                    self.graphics_view.toggle_torch(False)
                except Exception:
                    pass
            self.torch_toggle.setEnabled(checked)
            if not checked:
                self.torch_toggle.setChecked(False)
            try:
                self.graphics_view.toggle_magnifier(checked)
            except Exception:
                pass
        self.magnifier_toggle.toggled.connect(_on_magnifier_toggled)
        self.magnifier_zoom_label = QLabel("Magnifier Zoom:")
        bottom_bar.addWidget(self.magnifier_zoom_label)
        self.magnifier_zoom_slider = QSlider(Qt.Horizontal)
        self.magnifier_zoom_slider.setRange(10, 500) # value/10 -> 1.0x .. 50.0x
        self.magnifier_zoom_slider.setValue(80) # default 8.0x
        bottom_bar.addWidget(self.magnifier_zoom_slider)
        bottom_bar.addSpacing(20)
        # zoom buttons
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setToolTip("Zoom out")
        bottom_bar.addWidget(self.zoom_out_btn)
        self.reset_zoom_btn = QPushButton("Reset")
        self.reset_zoom_btn.setToolTip("Reset to 100%")
        bottom_bar.addWidget(self.reset_zoom_btn)
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setToolTip("Zoom in")
        bottom_bar.addWidget(self.zoom_in_btn)
        # Mouse Zoom toggle
        self.mouse_zoom_btn = QPushButton("Mouse Zoom: Off")
        self.mouse_zoom_btn.setCheckable(True)
        self.mouse_zoom_btn.setToolTip("Toggle wheel zoom")
        self.mouse_zoom_btn.setStyleSheet("background-color: #E57373; color: white;") # default red (OFF)
        bottom_bar.addWidget(self.mouse_zoom_btn)
        def toggle_mouse_zoom():
            self.mouse_zoom_enabled = not self.mouse_zoom_enabled
            if self.mouse_zoom_enabled:
                self.mouse_zoom_btn.setText("Mouse Zoom: On")
                self.mouse_zoom_btn.setStyleSheet("background-color: #4CAF50; color: white;") # green (ON)
            else:
                self.mouse_zoom_btn.setText("Mouse Zoom: Off")
                self.mouse_zoom_btn.setStyleSheet("background-color: #E57373; color: white;") # red (OFF)
        self.mouse_zoom_btn.clicked.connect(toggle_mouse_zoom)
        # Flip mode cycle
        self.flip_mode_btn = QPushButton("Flip: Off")
        self.flip_mode_btn.setCheckable(True)
        self.flip_mode_btn.setToolTip("Cycle flip mode")
        self.flip_mode_btn.setStyleSheet("background-color: #E57373; color: white;") # default red (OFF)
        bottom_bar.addWidget(self.flip_mode_btn)
        self.flip_mode = 0
        self.flip_labels = ["Flip: Off", "Flip: Select", "Flip: Select All"]
        def cycle_flip_mode():
            self.flip_mode = (self.flip_mode + 1) % 3
            self.flip_mode_btn.setText(self.flip_labels[self.flip_mode])
            if self.flip_mode == 0: # Off
                self.flip_mode_btn.setStyleSheet("background-color: #E57373; color: white;") # red
            elif self.flip_mode == 1: # Select
                self.flip_mode_btn.setStyleSheet("background-color: #81C784; color: white;") # pale green
            else: # Select All
                self.flip_mode_btn.setStyleSheet("background-color: #2E7D32; color: white;") # dark green
        self.flip_mode_btn.clicked.connect(cycle_flip_mode)
        # Rotation toggle
        self.rotation_mode_btn = QPushButton("Rotate: Off")
        self.rotation_mode_btn.setCheckable(True)
        self.rotation_mode_btn.setToolTip("Cycle rotation mode")
        self.rotation_mode_btn.setStyleSheet("background-color: #E57373; color: white;") # default red (OFF)
        bottom_bar.addWidget(self.rotation_mode_btn)
        self.rotation_mode = 0
        self.rotation_labels = ["Rotate: Off", "Rotate: Global", "Rotate: Local"]
        def cycle_rotation_mode():
            prev_mode = self.rotation_mode
            self.rotation_mode = (self.rotation_mode + 1) % 3
            self.rotation_mode_btn.setText(self.rotation_labels[self.rotation_mode])
            if self.rotation_mode == 0:
                self.rotation_mode_btn.setStyleSheet("background-color: #E57373; color: white;")
                self.rotation_overlay.hide()
            elif self.rotation_mode == 1:
                self.rotation_mode_btn.setStyleSheet("background-color: #4CAF50; color: white;")
                self.rotation_overlay.show()
                self._reposition_overlay()
                if prev_mode == 2:
                    self.combine_frames()
                self.global_rotation_slider.blockSignals(True)
                self.global_rotation_slider.setValue(self.global_rotation)
                self.global_rotation_slider.blockSignals(False)
                self.global_rotation_label.show()
                self.global_rotation_slider.show()
                self.global_rotation_value_label.show()
                self.local_rotation_label.hide()
                self.local_rotation_slider.hide()
                self.local_rotation_value_label.hide()
                self._apply_item_transform()
            else: # 2: Local
                self.rotation_mode_btn.setStyleSheet("background-color: #2196F3; color: white;") # blue for local
                self.rotation_overlay.show()
                self._reposition_overlay()
                if prev_mode != 2:
                    self.split_into_frames()
                self.global_rotation_slider.blockSignals(True)
                self.global_rotation_slider.setValue(self.global_rotation)
                self.global_rotation_slider.blockSignals(False)
                self.local_rotation_slider.blockSignals(True)
                self.local_rotation_slider.setValue(self.local_rotation)
                self.local_rotation_slider.blockSignals(False)
                self.global_rotation_label.hide()
                self.global_rotation_slider.hide()
                self.global_rotation_value_label.hide()
                self.local_rotation_label.show()
                self.local_rotation_slider.show()
                self.local_rotation_value_label.show()
                self._apply_local_rotation()
        self.rotation_mode_btn.clicked.connect(cycle_rotation_mode)
        bottom_bar.addStretch()
        layout.addWidget(self.bottom_bar_widget, 0, alignment=Qt.AlignHCenter)
        self._overlay_btn_margin = 12
        self.grid_btn = QPushButton("#", self.graphics_view.viewport())
        self.grid_btn.setFixedSize(22, 22)
        self.grid_btn.setCheckable(True)
        self.grid_btn.setToolTip("Toggle grid")
        self.grid_btn.clicked.connect(self._on_grid_toggled)
        self.grid_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0,0,0,0.7);
                color: white;
                border-radius: 4px;
                border: 1px solid rgba(255,255,255,0.12);
                font-weight: 700;
                font-size: 10px;
                padding: 0px;
            }
            QPushButton:checked {
                background: rgba(34, 114, 67, 0.78);
                border: 1px solid rgba(180,255,200,0.65);
            }
        """)
        self.grid_btn.show()
        self.grid_btn.raise_()
        self.fs_btn = QPushButton("⛶", self.graphics_view.viewport())
        self.fs_btn.setFixedSize(40, 40)
        self.fs_btn.setToolTip("Toggle fullscreen")
        self.fs_btn.setStyleSheet("background: rgba(0,0,0,0.7); color: white; border-radius: 6px;")
        self.fs_btn.clicked.connect(self.toggle_fullscreen)
        self.fs_btn.show()
        self.fs_btn.raise_()
        self._fs_margin = self._overlay_btn_margin
        QTimer.singleShot(0, self._reposition_fs_btn)
        self.graphics_view.viewport().installEventFilter(self)
        hbar = self.graphics_view.horizontalScrollBar()
        vbar = self.graphics_view.verticalScrollBar()
        try:
            hbar.valueChanged.connect(self._reposition_fs_btn)
            vbar.valueChanged.connect(self._reposition_fs_btn)
        except Exception:
            pass
        try:
            self.graphics_view.viewport().installEventFilter(self)
        except Exception:
            pass
        # Editor button (right side of bottom bar)
        self.editor_btn = QPushButton("Open Editor")
        self.editor_btn.setToolTip("Open editor")
        self.editor_btn.clicked.connect(self.open_editor)
        bottom_bar.addWidget(self.editor_btn)
        bottom_bar.addStretch()  # This already exists, but ensure it's after
        # connect signals (use existing graphics_view handlers for magnifier/torch/slider)
        self.torch_toggle.stateChanged.connect(lambda state: self.graphics_view.toggle_torch(state == Qt.Checked))
        self.magnifier_zoom_slider.valueChanged.connect(self.graphics_view.set_magnifier_zoom)
        # zoom buttons operate on this viewer's QGraphicsView transform
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        # Dynamic Island Overlay (floating widget)
        self.rotation_overlay = QWidget(self)
        self.rotation_overlay.setAttribute(Qt.WA_TranslucentBackground)
        self.rotation_overlay.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        # clean single background + rounded pill
        self.rotation_overlay.setStyleSheet("""
            QWidget {
                background-color: rgba(20, 24, 28, 220); /* darker solid translucent */
                border-radius: 14px;
                padding: 6px;
            }
        """)
        # subtle drop shadow to separate from image
        shadow = QGraphicsDropShadowEffect(self.rotation_overlay)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 200))
        self.rotation_overlay.setGraphicsEffect(shadow)
        overlay_layout = QHBoxLayout(self.rotation_overlay)
        overlay_layout.setContentsMargins(10, 6, 10, 6)
        overlay_layout.setSpacing(10)
        # Reset button (black dot)
        self.reset_rotation_btn = QPushButton("•")
        self.reset_rotation_btn.setFixedSize(24, 24)
        self.reset_rotation_btn.setStyleSheet("""
            QPushButton {
                background-color: black;
                color: white;
                border-radius: 12px;
                font-size: 16px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:pressed {
                background-color: #555555;
            }
        """)
        self.reset_rotation_btn.setToolTip("Reset rotation to 0°")
        self.reset_rotation_btn.clicked.connect(self.reset_rotation)
        overlay_layout.addWidget(self.reset_rotation_btn)
        self.global_rotation_label = QLabel("Global:")
        self.global_rotation_label.setStyleSheet("color: white;")
        overlay_layout.addWidget(self.global_rotation_label)
        self.global_rotation_label.hide()
        # Global Rotation slider
        self.global_rotation_slider = QSlider(Qt.Horizontal)
        self.global_rotation_slider.setRange(-180, 180)
        self.global_rotation_slider.setValue(0)
        self.global_rotation_slider.installEventFilter(self)
        # allow the slider to expand to take available width
        self.global_rotation_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.global_rotation_slider.setFixedHeight(20)
        self.global_rotation_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 5px;
                margin: 0px;
                border-radius: 5px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #2a7bd5, stop:1 #00d4ff);
            }
            QSlider::sub-page:horizontal { background: #4CAF50; border-radius: 5px; }
            QSlider::add-page:horizontal { background: rgba(255,255,255,0.08); border-radius: 5px; }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #0b78d1;
                width: 18px;
                height: 18px;
                margin: -9px 0; /* center handle */
                border-radius: 9px; /* makes it perfectly round */
            }
        """)
        overlay_layout.addWidget(self.global_rotation_slider)
        self.global_rotation_slider.installEventFilter(self)
        self.global_rotation_slider.hide()
        # Global angle label
        self.global_rotation_value_label = QLabel("0°")
        self.global_rotation_value_label.setStyleSheet("color: white; font-weight: 600;")
        overlay_layout.addWidget(self.global_rotation_value_label)
        self.global_rotation_value_label.hide()
        self.local_rotation_label = QLabel("Local:")
        self.local_rotation_label.setStyleSheet("color: white;")
        overlay_layout.addWidget(self.local_rotation_label)
        self.local_rotation_label.hide()
        # Local Rotation slider
        self.local_rotation_slider = QSlider(Qt.Horizontal)
        self.local_rotation_slider.setRange(-180, 180)
        self.local_rotation_slider.setValue(0)
        self.local_rotation_slider.installEventFilter(self)
        # allow the slider to expand to take available width
        self.local_rotation_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.local_rotation_slider.setFixedHeight(20)
        self.local_rotation_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 5px;
                margin: 0px;
                border-radius: 5px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #2a7bd5, stop:1 #00d4ff);
            }
            QSlider::sub-page:horizontal { background: #4CAF50; border-radius: 5px; }
            QSlider::add-page:horizontal { background: rgba(255,255,255,0.08); border-radius: 5px; }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 2px solid #0b78d1;
                width: 18px;
                height: 18px;
                margin: -9px 0; /* center handle */
                border-radius: 9px; /* makes it perfectly round */
            }
        """)
        overlay_layout.addWidget(self.local_rotation_slider)
        self.local_rotation_slider.installEventFilter(self)
        self.local_rotation_slider.hide()
        # Local angle label
        self.local_rotation_value_label = QLabel("0°")
        self.local_rotation_value_label.setStyleSheet("color: white; font-weight: 600;")
        overlay_layout.addWidget(self.local_rotation_value_label)
        self.local_rotation_value_label.hide()
        self.rotation_overlay.resize(600, 40)
        self.rotation_overlay.hide()
        self.global_rotation_slider.valueChanged.connect(self.set_global_rotation)
        self.local_rotation_slider.valueChanged.connect(self.set_local_rotation)

    def set_primary_monitor_control_width(self, width=None):
        try:
            bar = getattr(self, "bottom_bar_widget", None)
            if bar is None:
                return
            if width is None:
                bar.setMinimumWidth(0)
                bar.setMaximumWidth(16777215)
            else:
                safe_width = max(320, int(width))
                bar.setMinimumWidth(0)
                bar.setMaximumWidth(safe_width)
        except Exception:
            pass

    def set_global_rotation(self, degrees):
        self.global_rotation = degrees
        self.global_rotation_value_label.setText(f"{degrees}°")
        if self.frame_items:
            self._apply_local_rotation()
        else:
            self._apply_item_transform()
        self.graphics_view.cached_pixmap = None
        self.graphics_view.cached_source_scene = None
        self.graphics_view.viewport().update()
    def set_local_rotation(self, degrees):
        self.local_rotation = degrees
        self.local_rotation_value_label.setText(f"{degrees}°")
        if self.frame_items:
            self._apply_local_rotation()
        # Removed auto-combine to prevent issues when sliding through 0°
        self.graphics_view.cached_pixmap = None
        self.graphics_view.cached_source_scene = None
        self.graphics_view.viewport().update()

    def reset_rotation(self):
        self.global_rotation_slider.setValue(0)
        self.local_rotation_slider.setValue(0)

    def show_status(self, text, timeout_ms=None):
        """Show a brief textual status centered over the display viewport.
        If timeout_ms is provided the status will auto-clear after that many milliseconds.
        """
        try:
            if not text:
                self.clear_status()
                return
            # Update overlay text and reposition
            self.status_overlay.setText(text)
            self.status_overlay.adjustSize()
            vp = self.graphics_view.viewport()
            # size to at most 90% of viewport width, capped to 400px
            max_w = min(400, int(vp.width() * 0.9))
            self.status_overlay.setFixedWidth(max_w)
            self.status_overlay.move(int((vp.width() - self.status_overlay.width()) / 2), int((vp.height() - self.status_overlay.height()) / 2))
            self.status_overlay.show()
            self.status_overlay.raise_()
            if timeout_ms:
                QTimer.singleShot(int(timeout_ms), self.clear_status)
        except Exception:
            pass

    def clear_status(self):
        try:
            if hasattr(self, 'status_overlay'):
                self.status_overlay.hide()
        except Exception:
            pass

    def _reposition_status_overlay(self):
        try:
            if hasattr(self, 'status_overlay') and self.status_overlay.isVisible():
                vp = self.graphics_view.viewport()
                self.status_overlay.move(int((vp.width() - self.status_overlay.width()) / 2), int((vp.height() - self.status_overlay.height()) / 2))
                self.status_overlay.raise_()
        except Exception:
            pass

    def open_editor(self):
        """Open the Editor tab for this viewer's current image."""
        app = self.get_app()  # Uses your existing get_app() method
        if app:
            app.open_editor_tab(self)
        else:
            print("Warning: No app found to open editor.")

    def map_original_to_scene(self, orig_x, orig_y):
        if not self.frame_items:
            return self.pixmap_item.mapToScene(QPointF(orig_x, orig_y))
        else:
            for item in self.frame_items:
                item_h = getattr(item, 'local_height', item.pixmap().height())
                if item.orig_y <= orig_y < item.orig_y + item_h:
                    local_y = orig_y - item.orig_y
                    return item.mapToScene(QPointF(orig_x, local_y))
            # fallback to last item
            if self.frame_items:
                item = self.frame_items[-1]
                item_h = getattr(item, 'local_height', item.pixmap().height())
                local_y = orig_y - item.orig_y
                return item.mapToScene(QPointF(orig_x, local_y))
            return QPointF(orig_x, orig_y)
        
    def get_app(self):
        p = self
        # First try to find BandStitchProApp (has band_frames)
        while p and not hasattr(p, 'band_frames'): # band_frames is on BandStitchProApp
            p = p.parent()
        if p and hasattr(p, 'band_frames'):
            return p
        
        # If not found, try to find MainApp (has tab_widget and view_tabs for editor)
        p = self
        while p and not (hasattr(p, 'tab_widget') or hasattr(p, 'view_tabs')):
            p = p.parent()
        return p
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_fs_btn()
        if self.rotation_mode != 0:
            self._reposition_overlay()
        # Reposition pixel info box overlay to stay at bottom-left inside graphics view viewport
        if hasattr(self, 'pixel_info_box_overlay') and self.pixel_info_box_overlay is not None:
            try:
                margin = 10
                gv = self.graphics_view
                viewport = gv.viewport()
                viewport_rect = viewport.rect()
               
                # Position inside the graphics_view viewport, accounting for scrollbars
                # Map viewport bottom-left to parent coordinates
                px_x = viewport.mapToParent(viewport_rect.bottomLeft()).x() + margin
                py_y = viewport.mapToParent(viewport_rect.bottomLeft()).y() - self.pixel_info_box_overlay.height() - margin
               
                self.pixel_info_box_overlay.move(int(px_x), int(py_y))
            except Exception as e:
                print(f"Error repositioning pixel info box: {e}")

    def get_original_coords(self, scene_point):
        items = self.scene.items(scene_point)
        if items:
            item = items[0] # topmost
            if isinstance(item, QGraphicsPixmapItem):
                local_point = item.mapFromScene(scene_point)
                return local_point.x(), local_point.y() + getattr(item, 'orig_y', 0)
        # Outside item bounds: still map through the image transform so we can
        # report signed out-of-bounds coordinates instead of forcing (0, 0).
        if self.frame_items:
            best = None
            best_dist = None
            for item in self.frame_items:
                try:
                    local = item.mapFromScene(scene_point)
                    lx, ly = local.x(), local.y()
                    w = float(item.pixmap().width())
                    h = float(getattr(item, 'local_height', item.pixmap().height()))
                    max_x = max(0.0, w - 1.0)
                    max_y = max(0.0, h - 1.0)
                    dx = 0.0 if (0.0 <= lx <= max_x) else min(abs(lx - 0.0), abs(lx - max_x))
                    dy = 0.0 if (0.0 <= ly <= max_y) else min(abs(ly - 0.0), abs(ly - max_y))
                    dist = dx * dx + dy * dy
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best = (lx, ly + getattr(item, 'orig_y', 0.0))
                except Exception:
                    continue
            if best is not None:
                return best
        if self.pixmap_item is not None:
            local_point = self.pixmap_item.mapFromScene(scene_point)
            return local_point.x(), local_point.y()
        return 0.0, 0.0

    def _normalize_pixel_query(self, x, y):
        """Return display coords and border-clamped sampling coords."""
        try:
            x = int(x)
            y = int(y)
        except Exception:
            x, y = 0, 0
        w = int(getattr(self, 'full_width', 0) or 0)
        h = int(getattr(self, 'full_height', 0) or 0)
        if w <= 0 or h <= 0:
            return x, y, 0, 0
        sample_x = max(0, min(w - 1, x))
        sample_y = max(0, min(h - 1, y))
        if x < 0:
            display_x = x
        elif x >= w:
            display_x = -(x - (w - 1))
        else:
            display_x = x
        if y < 0:
            display_y = y
        elif y >= h:
            display_y = -(y - (h - 1))
        else:
            display_y = y
        return display_x, display_y, sample_x, sample_y
    
    def _apply_item_transform(self):
        if self.global_rotation == 0:
            self.pixmap_item.setTransform(QTransform())
        else:
            w, h = float(self.full_width), float(self.full_height)
            cx, cy = w / 2.0, h / 2.0
            transform = QTransform().translate(cx, cy).rotate(self.global_rotation).translate(-cx, -cy)
            self.pixmap_item.setTransform(transform)
        self._update_scene_rect()

    def _apply_local_rotation(self):
        for item in self.frame_items:
            w = float(item.pixmap().width())
            h = float(getattr(item, 'local_height', item.pixmap().height()))
            cx = w / 2.0
            cy = h / 2.0
            local_t = QTransform().translate(cx, cy).rotate(self.local_rotation).translate(-cx, -cy)
            item.setTransform(local_t)
        full_w, full_h = float(self.full_width), float(self.full_height)
        full_cx, full_cy = full_w / 2.0, full_h / 2.0
        global_t = QTransform().translate(full_cx, full_cy).rotate(self.global_rotation).translate(-full_cx, -full_cy)
        if self.frame_group:
            self.frame_group.setTransform(global_t)
        self._update_scene_rect()
        self.graphics_view.viewport().update()

    def _update_scene_rect(self):
        """Update scene rect to include transformed item bounds (fixes missing scrollbars after rotation)."""
        try:
            rect = self.scene.itemsBoundingRect()
            if rect.isValid():
                pad = 2
                rect = rect.adjusted(-pad, -pad, pad, pad)
                self.scene.setSceneRect(rect)
        except Exception:
            pass

    def _reposition_overlay(self):
        if self.rotation_mode == 0 or not self._bottom_layout:
            return
        try:
            ol = self.rotation_overlay
            bottom_item = self.layout().itemAt(self.layout().count() - 1)
            bottom_geom = bottom_item.geometry()
            y = bottom_geom.y() - ol.height() - 30
            x = max(0, (self.width() - ol.width()) // 2)
            ol.move(x, y)
            ol.raise_()
        except Exception:
            pass

    def split_into_frames(self):
        if not self.current_pil_image or self.frame_items:
            return
        stitch_sequence = self.get_stitch_sequence()
        if not stitch_sequence and getattr(self, 'is_individual', False):
            per_block_h = self.frame_h
            stitch_sequence = [{'start_y': 0, 'per_block_h': per_block_h, 'base': 'individual'}]
            frame_h = per_block_h
        else:
            if not stitch_sequence:
                return
            first_start = min(e['start_y'] for e in stitch_sequence)
            last_end = max(e['start_y'] + e['per_block_h'] for e in stitch_sequence)
            frame_h = last_end - first_start
        gap = getattr(self, 'gap', 0)
        # Improved n_frames calculation: exact for stacked frames with uniform gaps
        if gap == 0 or frame_h == 0:
            n_frames = self.full_height // frame_h if frame_h > 0 else 0
        else:
            n_frames = round((self.full_height + gap) / (frame_h + gap))
        if n_frames < 1:
            n_frames = 1
        if n_frames == 0:
            return
        gap_between_frames = (self.full_height - n_frames * frame_h) // (n_frames - 1) if n_frames > 1 else 0
        self.pixmap_item.setVisible(False)
        self.frame_group = QGraphicsItemGroup()
        self.scene.addItem(self.frame_group)
        for f in range(n_frames):
            base_y = f * (frame_h + gap_between_frames)
            for entry in stitch_sequence:
                y0 = base_y + entry['start_y']
                h = entry['per_block_h']
                y1 = y0 + h
                if y1 > self.full_height: continue
                part = self.current_pil_image.crop((0, int(y0), self.full_width, int(y1)))
                frame_raw = self.original_image_data[int(y0):int(y1), :]
                qimg = pil_to_qimage(part)
                pix = QPixmap.fromImage(qimg)
                item = QGraphicsPixmapItem(pix)
                item.setPos(0, y0)
                item.setZValue(y0)
                item.orig_y = y0
                item.local_height = h
                item.original_data = frame_raw
                self.frame_group.addToGroup(item)
                self.frame_items.append(item)
        self._apply_local_rotation()

    def combine_frames(self):
        if not self.frame_items:
            return
        for item in self.frame_items:
            self.scene.removeItem(item)
        if self.frame_group:
            self.scene.removeItem(self.frame_group)
        self.frame_items = []
        self.frame_group = None
        self.pixmap_item.setVisible(True)
        self._apply_item_transform() # Restore global if needed

        
    def get_stitch_sequence(self):
        # Extracted logic from apply_flip
        parent_app = self.parent()
        while parent_app and not (hasattr(parent_app, 'bands_info') and hasattr(parent_app, 'gap_var') and hasattr(parent_app, 'height_entry')):
            parent_app = parent_app.parent()
        bands_info = getattr(parent_app, 'bands_info', None) if parent_app else None
        geo_info = getattr(self, 'geo_info', None)
        gap = 0
        if parent_app and getattr(parent_app, 'gap_var', None):
            try:
                gap = int(parent_app.gap_var.value())
            except Exception:
                gap = 0
        orig_band_h = None
        if parent_app and getattr(parent_app, 'height_entry', None):
            try:
                orig_band_h = int(parent_app.height_entry.text())
            except Exception:
                orig_band_h = None
        if orig_band_h is None and geo_info is not None:
            try:
                orig_band_h = int(geo_info[3])
            except Exception:
                orig_band_h = None
        if orig_band_h is None:
            orig_band_h = 384
        full_w, full_h = self.full_width, self.full_height
        # --- Determine if this is Individual view or All Bands view ---
        is_individual = getattr(self, 'is_individual', False)
        if is_individual:
            return [] # No split for individual
        # --- Existing logic for All Bands (multi-band stitch) ---
        def find_in_obj(obj):
            if obj is None:
                return (None, None)
            try:
                if isinstance(obj, dict):
                    bf = obj.get("band_frames") or obj.get("bands") or obj.get("frames") or None
                    bi = obj.get("bands_info") or obj.get("band_info") or None
                    return (bf, bi)
                bf = getattr(obj, "band_frames", None)
                if not bf:
                    for name in ("bands", "frames", "bandFrames", "band_frames_local"):
                        bf = getattr(obj, name, None)
                        if bf:
                            break
                bi = getattr(obj, "bands_info", None) or getattr(obj, "band_info", None)
                return (bf, bi)
            except Exception:
                return (None, None)
        band_frames = None
        bands_info_found = None
        def bands_info_score(bi):
            if not isinstance(bi, dict):
                return 0
            score = 0
            try:
                for k, v in bi.items():
                    if isinstance(v, dict):
                        if v.get("binned"):
                            score += 5
                        if int(v.get("bin_factor", 1)) > 1:
                            score += 3
                        if any("_binned" in str(x).lower() for x in v.get("variants", [])):
                            score += 2
                score += min(3, len(bi))
            except Exception:
                score = len(bi) if isinstance(bi, dict) else 0
            return score
        candidates = [self, parent_app, getattr(self, "window", None), getattr(self, "main_window", None)]
        candidates.append(globals())
        try:
            
            candidates.append(sys.modules.get(__name__, None))
        except Exception:
            pass
        for c in candidates:
            bf, bi = find_in_obj(c)
            if bf:
                try:
                    valid_bf = False
                    if isinstance(bf, dict):
                        if len(bf) > 0:
                            valid_bf = True
                    else:
                        try:
                            if hasattr(bf, "__len__") and len(bf) > 0:
                                valid_bf = True
                        except Exception:
                            valid_bf = True
                    if valid_bf:
                        band_frames = bf
                except Exception:
                    band_frames = bf
            if bi:
                if bands_info_found is None:
                    bands_info_found = bi
                else:
                    try:
                        if bands_info_score(bi) > bands_info_score(bands_info_found):
                            bands_info_found = bi
                    except Exception:
                        bands_info_found = bi
            if band_frames and isinstance(bands_info_found, dict) and bands_info_score(bands_info_found) > 0:
                break
        if band_frames is None:
            band_frames = {}
        try:
            _ = list(band_frames.keys())
        except Exception:
            try:
                band_frames = {k: getattr(band_frames, k) for k in dir(band_frames) if not k.startswith("__")}
            except Exception:
                band_frames = {}
        if not bands_info_found:
            keys = list(band_frames.keys())
            inferred = {}
            if keys:
                bases_seen = []
                for k in keys:
                    base = k.split('_', 1)[0]
                    if base not in bases_seen:
                        bases_seen.append(base)
                for idx, base in enumerate(bases_seen):
                    variants = [k for k in keys if k.startswith(base + "_")]
                    measured_hs = []
                    for v in variants:
                        f = band_frames.get(v)
                        mh = 0
                        try:
                            if isinstance(f, list) and len(f) > 0:
                                mh = int(getattr(f[0], "shape", (None,))[0] or f[0].shape[0])
                            else:
                                if hasattr(f, "h"):
                                    mh = int(getattr(f, "h"))
                                elif hasattr(f, "__len__"):
                                    try:
                                        if len(f) > 0:
                                            first = f[0]
                                            mh = int(getattr(first, "shape", (None,))[0] or first.shape[0])
                                    except Exception:
                                        mh = 0
                        except Exception:
                            mh = 0
                        if mh:
                            measured_hs.append(mh)
                    if measured_hs:
                        mh = sorted(measured_hs)[len(measured_hs) // 2]
                        binned = (abs(mh - (orig_band_h // 2)) <= 3)
                        bin_factor = 2 if binned else 1
                    else:
                        binned = any("binned" in v.lower() for v in variants)
                        bin_factor = 2 if binned else 1
                    split = any(v.lower().endswith("_left") or v.lower().endswith("_right") for v in variants)
                    inferred[base] = {'index': idx, 'variants': variants or [base], 'binned': binned, 'split': split, 'bin_factor': bin_factor}
            else:
                half_h = orig_band_h // 2
                n_full = max(1, int(round(full_h / orig_band_h)))
                expected_full = orig_band_h * n_full + gap * (n_full - 1) if n_full > 0 else 0
                diff_full = abs(expected_full - full_h)
                n_half = max(1, int(round(full_h / half_h)))
                expected_half = half_h * n_half + gap * (n_half - 1) if n_half > 0 else 0
                diff_half = abs(expected_half - full_h)
                if diff_full <= diff_half:
                    n = n_full
                    per_h = orig_band_h
                    bin_factor = 1
                    binned = False
                    if n % 2 == 0:
                        split = True
                        m = n // 2
                    else:
                        split = False
                        m = n
                else:
                    n = n_half
                    per_h = half_h
                    bin_factor = 2
                    binned = True
                    split = False
                    m = n
                inferred = {}
                for i in range(m):
                    base = f"b{i}"
                    inferred[base] = {'index': i, 'variants': [base], 'binned': binned, 'split': split, 'bin_factor': bin_factor}
            bands_info_found = inferred
        # Build stitch_sequence (same as original)
        stitch_sequence = []
        ordered_bases = sorted(bands_info_found.keys(), key=lambda k: bands_info_found[k]['index'])
        binned_bases = [b for b in ordered_bases if bands_info_found[b].get('binned', False)]
        unbinned_bases = [b for b in ordered_bases if not bands_info_found[b].get('binned', False)]
        for base in binned_bases:
            info = bands_info_found[base]
            bin_factor = int(info.get('bin_factor', 1)) or 1
            per_block_h = max(1, int(round(float(orig_band_h) / float(bin_factor))))
            if parent_app and getattr(parent_app, "ENABLE_1_TO_4_LAYOUT", False):
                if hasattr(parent_app, "_binned_upsample_factor"):
                    per_block_h = int(per_block_h * parent_app._binned_upsample_factor(bin_factor))
            stitch_sequence.append({
                'base': base, 'kind': 'full_binned', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': False, 'side': None
            })
        merge_lr = False
        if parent_app and getattr(parent_app, "ENABLE_1_TO_4_LAYOUT", False):
            merge_lr = True
        for base in unbinned_bases:
            info = bands_info_found[base]
            bin_factor = int(info.get('bin_factor', 1)) or 1
            per_block_h = int(orig_band_h)
            if info.get('split', False):
                if merge_lr:
                    stitch_sequence.append({
                        'base': base, 'kind': 'full_unbinned', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': True, 'side': None
                    })
                else:
                    stitch_sequence.append({
                        'base': base, 'kind': 'half_left', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': True, 'side': 'left'
                    })
                    stitch_sequence.append({
                        'base': base, 'kind': 'half_right', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': True, 'side': 'right'
                    })
            else:
                stitch_sequence.append({
                    'base': base, 'kind': 'full_unbinned', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': False, 'side': None
                })
        # Compute start_y
        cur_y = 0
        for entry in stitch_sequence:
            entry['start_y'] = cur_y
            cur_y += int(entry['per_block_h']) + gap
        return stitch_sequence
   
    def _split_vertical(self):
        if not self.current_pil_image or not self.is_tdi:
            return []
        stitch_sequence = self.get_stitch_sequence()
        if not stitch_sequence:
            return []
        first_start = min(e['start_y'] for e in stitch_sequence)
        last_end = max(e['start_y'] + e['per_block_h'] for e in stitch_sequence)
        frame_h = last_end - first_start
        if frame_h <= 0 or self.full_height < frame_h:
            return []
        n_frames = self.full_height // frame_h # assuming exact division
        gap_between_frames = (self.full_height - n_frames * frame_h) // max(1, n_frames - 1) if n_frames > 1 else 0
        frames = []
        cur_y = 0
        for f in range(n_frames):
            y0 = cur_y
            y1 = y0 + frame_h
            frame_pil = self.current_pil_image.crop((0, y0, self.full_width, y1))
            frame_raw = self.original_image_data[y0:y1, :]
            frames.append((frame_pil, frame_raw, y0))
            cur_y += frame_h + gap_between_frames
        return frames
    def show_image(self, pil_image, fit_to_screen=False, raw_pil=None):
        if raw_pil is None:
            raw_pil = pil_image
        # Defensive guard: this viewer can be referenced briefly after Qt has
        # already deleted underlying C++ objects during tab unload/reload.
        try:
            _ = self.graphics_view
            _ = self.pixmap_item
            _ = self.scene
        except RuntimeError:
            return
        if pil_image is None:
            try:
                self.pixmap_item.setPixmap(QPixmap())
            except RuntimeError:
                return
            try:
                self.torch_toggle.setChecked(False)
                self.torch_toggle.setEnabled(False)
            except RuntimeError:
                return
            try:
                self.graphics_view.toggle_torch(False)
            except Exception:
                pass
            self.current_pil_image = None
            self.original_image_data = None
            # Raw (unscaled) data for pixel info, if provided by caller
            self.original_raw_data = None
            self.full_width = self.full_height = 0
            try:
                self.scene.setSceneRect(QRectF())
            except RuntimeError:
                return
            return
        # Save current view state for preservation
        old_zoom = self.zoom
        try:
            vp_rect = self.graphics_view.viewport().rect()
            old_center_view = vp_rect.center()
            old_center_scene = self.graphics_view.mapToScene(old_center_view)
            old_h_scroll = self.graphics_view.horizontalScrollBar().value()
            old_v_scroll = self.graphics_view.verticalScrollBar().value()
        except RuntimeError:
            return
        # Keep safe copies
        pil_copy = pil_image.copy()
        raw_copy = raw_pil.copy()
        # Normalize display image to RGB or Grayscale
        if pil_copy.mode not in ('RGB', 'L'):
            pil_copy = pil_copy.convert('RGB')
        self.current_pil_image = pil_copy
        # Original data for pixel info. Preserve native dimensionality for grayscale inputs
        # (including high bit depth), and only strip alpha channels when present.
        raw_array = np.array(raw_copy)
        if raw_array.ndim == 3:
            if raw_array.shape[2] == 4:
                raw_array = raw_array[..., :3]
            elif raw_array.shape[2] == 2:
                raw_array = raw_array[..., 0]
        self.original_image_data = raw_array
        # Clear any stale raw data cache; callers can set this after show_image.
        self.original_raw_data = None
        self.full_width, self.full_height = pil_copy.size
        # Ensure grid defaults if grid is enabled
        if self.graphics_view.grid_enabled:
            self.graphics_view._ensure_grid_defaults()
        if self.graphics_view.crop_box_enabled and (
            self.graphics_view.crop_box_rect is None or self.graphics_view.crop_box_rect.isEmpty()
        ):
            self.graphics_view._init_crop_box_rect()
        # Compute frame_h
        parent_app = self.parent()
        while parent_app and not (hasattr(parent_app, 'gap_var') and hasattr(parent_app, 'height_entry') and hasattr(parent_app, 'bands_info')):
            parent_app = parent_app.parent()
        gap = int(getattr(parent_app.gap_var, 'value', lambda: 0)()) if parent_app else 0
        orig_band_h_str = getattr(parent_app.height_entry, 'text', lambda: '')() if parent_app else ''
        orig_band_h = int(orig_band_h_str) if orig_band_h_str.isdigit() else 384
        bands_info = getattr(parent_app, 'bands_info', None) if parent_app else None
        stitch_sequence = self.get_stitch_sequence()
        if stitch_sequence:
            first_start = min(e['start_y'] for e in stitch_sequence)
            last_end = max(e['start_y'] + e['per_block_h'] for e in stitch_sequence)
            frame_h = last_end - first_start
        else:
            # For individual band tabs or RGB Fusion
            if hasattr(self, 'is_rgb_fusion') and self.is_rgb_fusion:
                # Use pre-set frame_h for RGB Fusion (per-frame fused height including offsets)
                frame_h = getattr(self, 'frame_h', self.full_height) # Fallback to full height if not set
            else:
                # Original individual band inference logic
                print(f"DEBUG show_image individual: is_individual={getattr(self, 'is_individual', False)}, key={getattr(self, 'key', None)}")
                print(f"DEBUG bands_info: {bands_info}")
                print(f"DEBUG full_height={self.full_height}, orig_band_h={orig_band_h}, gap={gap}")
                frame_h = orig_band_h
                key = getattr(self, 'key', None)
                if key is None:
                    try:
                        # Climb up to find the QTabWidget
                        p = self
                        tabwidget = None
                        while p is not None:
                            if isinstance(p, QTabWidget):
                                tabwidget = p
                                break
                            p = p.parent()
                        if tabwidget:
                            # Now check each tab to see if self is a descendant of its widget
                            for i in range(tabwidget.count()):
                                tab_child = tabwidget.widget(i)
                                q = self
                                is_descendant = False
                                while q is not None:
                                    if q == tab_child:
                                        is_descendant = True
                                        break
                                    q = q.parent()
                                if is_descendant:
                                    key = tabwidget.tabText(i)
                                    self.key = key
                                    print(f"DEBUG inferred key from tab text: {key}")
                                    break
                    except Exception as e:
                        print(f"DEBUG failed to infer key: {e}")
                bin_factor = 1
                matched = False
                if key and bands_info:
                    for base, info in bands_info.items():
                        variants = info.get('variants', [])
                        if key == base or key in variants:
                            bin_factor = info.get('bin_factor', 1)
                            matched = True
                            break
                print(f"DEBUG after bands_info match: matched={matched}, bin_factor={bin_factor}")
                if not matched:
                    if key and 'binned' in key.lower():
                        bin_factor = 2
                print(f"DEBUG after not matched 'binned' check: bin_factor={bin_factor}")
                # Additional fallback: use dimension matching if bin_factor still 1
                if bin_factor == 1:
                    print("DEBUG entering dimension check")
                    half_h = orig_band_h // 2
                    def compute_expected(per_h, full_h, gap):
                        if per_h == 0:
                            return float('inf'), 0
                        n = max(1, int(round(full_h / per_h)))
                        expected = per_h * n + gap * (n - 1)
                        diff = abs(expected - full_h)
                        return diff, n
                    diff_full, n_full = compute_expected(orig_band_h, self.full_height, gap)
                    diff_half, n_half = compute_expected(half_h, self.full_height, gap)
                    print(f"DEBUG diff_full={diff_full}, n_full={n_full}, diff_half={diff_half}, n_half={n_half}")
                    if diff_full < diff_half:
                        bin_factor = 1
                    elif diff_half < diff_full:
                        bin_factor = 2
                    else:
                        # Equal diffs: use key to break tie
                        if key and 'binned' in key.lower():
                            bin_factor = 2
                        else:
                            bin_factor = 1
                print(f"DEBUG final bin_factor={bin_factor}, frame_h={orig_band_h // bin_factor}")
                frame_h = orig_band_h // bin_factor
                self.per_block_h = frame_h
        self.frame_h = frame_h
        self.gap = gap
        self.is_frame_stack = (self.full_height > frame_h) if frame_h > 0 else False
        self.overall_center_y = self.full_height / 2.0
        # Display array for QImage
        display_array = np.array(pil_copy)
        bytes_per_line = display_array.strides[0]
        if display_array.ndim == 3 and display_array.shape[2] == 3:
            qimg_format = QImage.Format_RGB888
        elif display_array.ndim == 2:
            qimg_format = QImage.Format_Grayscale8
        else:
            # Fallback to RGB
            pil_copy = pil_copy.convert('RGB')
            display_array = np.array(pil_copy)
            bytes_per_line = display_array.strides[0]
            qimg_format = QImage.Format_RGB888
        qimg = QImage(display_array.data, self.full_width, self.full_height, bytes_per_line, qimg_format)
        qimg = qimg.copy() # Own the data
        pixmap = QPixmap.fromImage(qimg)
        try:
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.setTransform(QTransform()) # Reset
            self.pixmap_item.setPos(0, 0)
            # Reset frames
            if self.frame_items:
                self.combine_frames()
            self.scene.setSceneRect(self.pixmap_item.sceneBoundingRect())
            if self.rotation_mode == 2 and self.is_frame_stack:
                self.split_into_frames()
            if self.rotation_mode == 2:
                self._apply_local_rotation()
            else:
                self._apply_item_transform()
            # Torch is available only while magnifier is enabled.
            self.torch_toggle.setEnabled(bool(self.magnifier_toggle.isChecked()))
            def _do_fit():
                try:
                    self.graphics_view.resetTransform()
                    if not self.scene.sceneRect().isValid():
                        self.scene.setSceneRect(self.pixmap_item.sceneBoundingRect())
                    self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                    self.zoom = self.graphics_view.transform().m11()
                except Exception as e:
                    print("DEBUG: fitInView failed:", e)
                finally:
                    self.graphics_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
                    self.graphics_view.viewport().update()
            if fit_to_screen:
                QTimer.singleShot(0, _do_fit)
            else:
                # Restore zoom and center
                def _restore_view():
                    try:
                        transform = QTransform().scale(old_zoom, old_zoom)
                        self.graphics_view.setTransform(transform)
                        self.zoom = old_zoom
                        # Restore center
                        new_center_view = vp_rect.center() # Viewport may have resized, but use old relative
                        new_center_scene = self.graphics_view.mapToScene(new_center_view)
                        delta_scene = old_center_scene - new_center_scene
                        self.graphics_view.centerOn(new_center_scene + delta_scene)
                        # Fallback to scroll values if needed
                        self.graphics_view.horizontalScrollBar().setValue(old_h_scroll)
                        self.graphics_view.verticalScrollBar().setValue(old_v_scroll)
                    except Exception as e:
                        print("DEBUG: view restore failed:", e)
                        _do_fit() # Fallback to fit
                    finally:
                        self.graphics_view.viewport().update()
                QTimer.singleShot(0, _restore_view)
            # Adjust magnifier proportionally if size changed
            if self.graphics_view.magnifier_enabled and self.graphics_view.magnifier_center:
                old_height = getattr(self, '_last_height', self.full_height)
                if old_height != self.full_height:
                    rel_y = self.graphics_view.magnifier_center.y() / old_height
                    self.graphics_view.magnifier_center.setY(rel_y * self.full_height)
                self._last_height = self.full_height
                self.graphics_view.cached_pixmap = None
                self.graphics_view.viewport().update()
        except RuntimeError:
            return
           
    def zoom_in(self):
        """Zoom in around the view center."""
        self._apply_zoom(1.25)
    def zoom_out(self):
        """Zoom out around the view center."""
        self._apply_zoom(0.8)
    def reset_zoom(self):
        """Reset transform to identity (100% / actual size)."""
        try:
            self.graphics_view.resetTransform()
        except Exception:
            pass
        self.zoom = 1.0
        # Update position label using viewport center so zoom % updates
        center = self.graphics_view.viewport().rect().center()
        try:
            self.update_position_label(center)
        except Exception:
            pass
        self.graphics_view.viewport().update()
    def _apply_zoom(self, factor):
        """Internal helper: scale view around viewport center and update state."""
        try:
            # scale around the viewport center (keeps view stable)
            self.graphics_view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            self.graphics_view.scale(factor, factor)
            # store zoom factor (m11 is the horizontal scale)
            self.zoom = self.graphics_view.transform().m11()
            center = self.graphics_view.viewport().rect().center()
            try:
                self.update_position_label(center)
            except Exception:
                pass
            self.graphics_view.viewport().update()
        except Exception as e:
            # defensive: don't crash UI on scaling error
            print("Zoom failed:", e)
    def apply_flip(self, vertical=False, horizontal=False, all=False, click_pos=None, global_flip=False):
        if self.original_image_data is None:
            return
        if global_flip:
            key = 'vertical' if vertical else 'horizontal' if horizontal else None
            if key:
                attr = f'global_{key}_flipped'
                current = getattr(self, attr)
                setattr(self, attr, not current)
                axis = 0 if vertical else 1
                self.original_image_data = np.flip(self.original_image_data, axis=axis)
            if self.original_image_data.ndim == 3:
                pil_image = Image.fromarray(self.original_image_data)
            else:
                pil_image = Image.fromarray(self.original_image_data, mode='L')
            self.current_pil_image = pil_image
            self.pixmap_item.setPixmap(QPixmap.fromImage(pil_to_qimage(pil_image)))
            self._apply_item_transform()
            self.scene.setSceneRect(self.pixmap_item.sceneBoundingRect())
            self.graphics_view.viewport().update()
            return
        parent_app = self.parent()
        while parent_app and not (hasattr(parent_app, 'bands_info') and hasattr(parent_app, 'gap_var') and hasattr(parent_app, 'height_entry')):
            parent_app = parent_app.parent()
        bands_info = getattr(parent_app, 'bands_info', None) if parent_app else None
        geo_info = getattr(self, 'geo_info', None)
        gap = 0
        if parent_app and getattr(parent_app, 'gap_var', None):
            try:
                gap = int(parent_app.gap_var.value())
            except Exception:
                gap = 0
        orig_band_h = None
        if parent_app and getattr(parent_app, 'height_entry', None):
            try:
                orig_band_h = int(parent_app.height_entry.text())
            except Exception:
                orig_band_h = None
        if orig_band_h is None and geo_info is not None:
            try:
                orig_band_h = int(geo_info[3])
            except Exception:
                orig_band_h = None
        if orig_band_h is None:
            orig_band_h = 384
        full_w, full_h = self.full_width, self.full_height
        # --- Determine if this is Individual view or All Bands view ---
        is_individual = getattr(self, 'is_individual', False)
        if is_individual:
            method = Image.FLIP_TOP_BOTTOM if vertical else Image.FLIP_LEFT_RIGHT if horizontal else None
            if method is None:
                return
            if not getattr(self, 'is_frame_stack', False):
                # Single frame: flip the whole image
                flip_rect = (0, 0, full_w, full_h)
                part = self.current_pil_image.crop(flip_rect)
                flipped = part.transpose(method)
                self.current_pil_image.paste(flipped, flip_rect)
            else:
                # Frame stack: flip each frame block (or selected if not all)
                per_h = getattr(self, 'per_block_h', orig_band_h)
                gap = getattr(self, 'gap', 0)
                if all:
                    cur_y = 0
                    while cur_y < full_h:
                        y0 = cur_y
                        y1 = min(full_h, y0 + per_h)
                        flip_rect = (0, y0, full_w, y1)
                        part = self.current_pil_image.crop(flip_rect)
                        flipped = part.transpose(method)
                        self.current_pil_image.paste(flipped, flip_rect)
                        cur_y = y1 + gap
                else:
                    if click_pos is None:
                        return
                    cx, cy = click_pos
                    # Map click_pos considering global flips for correct band selection
                    if self.global_vertical_flipped:
                        logical_cy = full_h - 1 - cy
                    else:
                        logical_cy = cy
                    if self.global_horizontal_flipped:
                        cx = full_w - 1 - cx
                    block_idx = int(logical_cy) // (per_h + gap)
                    logical_y0 = block_idx * (per_h + gap)
                    logical_y1 = min(full_h, logical_y0 + per_h)
                    if self.global_vertical_flipped:
                        y0 = full_h - logical_y1
                        y1 = full_h - logical_y0
                    else:
                        y0 = logical_y0
                        y1 = logical_y1
                    flip_rect = (0, y0, full_w, y1)
                    part = self.current_pil_image.crop(flip_rect)
                    flipped = part.transpose(method)
                    self.current_pil_image.paste(flipped, flip_rect)
            self.show_image(self.current_pil_image, fit_to_screen=False)
            return
        # --- Existing logic for All Bands (multi-band stitch) ---
        def find_in_obj(obj):
            if obj is None:
                return (None, None)
            try:
                if isinstance(obj, dict):
                    bf = obj.get("band_frames") or obj.get("bands") or obj.get("frames") or None
                    bi = obj.get("bands_info") or obj.get("band_info") or None
                    return (bf, bi)
                bf = getattr(obj, "band_frames", None)
                if not bf:
                    for name in ("bands", "frames", "bandFrames", "band_frames_local"):
                        bf = getattr(obj, name, None)
                        if bf:
                            break
                bi = getattr(obj, "bands_info", None) or getattr(obj, "band_info", None)
                return (bf, bi)
            except Exception:
                return (None, None)
        band_frames = None
        bands_info_found = None
        def bands_info_score(bi):
            if not isinstance(bi, dict):
                return 0
            score = 0
            try:
                for k, v in bi.items():
                    if isinstance(v, dict):
                        if v.get("binned"):
                            score += 5
                        if int(v.get("bin_factor", 1)) > 1:
                            score += 3
                        if any("_binned" in str(x).lower() for x in v.get("variants", [])):
                            score += 2
                score += min(3, len(bi))
            except Exception:
                score = len(bi) if isinstance(bi, dict) else 0
            return score
        candidates = [self, parent_app, getattr(self, "window", None), getattr(self, "main_window", None)]
        candidates.append(globals())
        try:
            
            candidates.append(sys.modules.get(__name__, None))
        except Exception:
            pass
        for c in candidates:
            bf, bi = find_in_obj(c)
            if bf:
                try:
                    valid_bf = False
                    if isinstance(bf, dict):
                        if len(bf) > 0:
                            valid_bf = True
                    else:
                        try:
                            if hasattr(bf, "__len__") and len(bf) > 0:
                                valid_bf = True
                        except Exception:
                            valid_bf = True
                    if valid_bf:
                        band_frames = bf
                except Exception:
                    band_frames = bf
            if bi:
                if bands_info_found is None:
                    bands_info_found = bi
                else:
                    try:
                        if bands_info_score(bi) > bands_info_score(bands_info_found):
                            bands_info_found = bi
                    except Exception:
                        bands_info_found = bi
            if band_frames and isinstance(bands_info_found, dict) and bands_info_score(bands_info_found) > 0:
                break
        if band_frames is None:
            band_frames = {}
        try:
            _ = list(band_frames.keys())
        except Exception:
            try:
                band_frames = {k: getattr(band_frames, k) for k in dir(band_frames) if not k.startswith("__")}
            except Exception:
                band_frames = {}
        if not bands_info_found:
            keys = list(band_frames.keys())
            inferred = {}
            if keys:
                bases_seen = []
                for k in keys:
                    base = k.split('_', 1)[0]
                    if base not in bases_seen:
                        bases_seen.append(base)
                for idx, base in enumerate(bases_seen):
                    variants = [k for k in keys if k.startswith(base + "_")]
                    measured_hs = []
                    for v in variants:
                        f = band_frames.get(v)
                        mh = 0
                        try:
                            if isinstance(f, list) and len(f) > 0:
                                mh = int(getattr(f[0], "shape", (None,))[0] or f[0].shape[0])
                            else:
                                if hasattr(f, "h"):
                                    mh = int(getattr(f, "h"))
                                elif hasattr(f, "__len__"):
                                    try:
                                        if len(f) > 0:
                                            first = f[0]
                                            mh = int(getattr(first, "shape", (None,))[0] or first.shape[0])
                                    except Exception:
                                        mh = 0
                        except Exception:
                            mh = 0
                        if mh:
                            measured_hs.append(mh)
                    if measured_hs:
                        mh = sorted(measured_hs)[len(measured_hs) // 2]
                        binned = (abs(mh - (orig_band_h // 2)) <= 3)
                        bin_factor = 2 if binned else 1
                    else:
                        binned = any("binned" in v.lower() for v in variants)
                        bin_factor = 2 if binned else 1
                    split = any(v.lower().endswith("_left") or v.lower().endswith("_right") for v in variants)
                    inferred[base] = {'index': idx, 'variants': variants or [base], 'binned': binned, 'split': split, 'bin_factor': bin_factor}
            else:
                half_h = orig_band_h // 2
                n_full = max(1, int(round(full_h / orig_band_h)))
                expected_full = orig_band_h * n_full + gap * (n_full - 1) if n_full > 0 else 0
                diff_full = abs(expected_full - full_h)
                n_half = max(1, int(round(full_h / half_h)))
                expected_half = half_h * n_half + gap * (n_half - 1) if n_half > 0 else 0
                diff_half = abs(expected_half - full_h)
                if diff_full <= diff_half:
                    n = n_full
                    per_h = orig_band_h
                    bin_factor = 1
                    binned = False
                    if n % 2 == 0:
                        split = True
                        m = n // 2
                    else:
                        split = False
                        m = n
                else:
                    n = n_half
                    per_h = half_h
                    bin_factor = 2
                    binned = True
                    split = False
                    m = n
                inferred = {}
                for i in range(m):
                    base = f"b{i}"
                    inferred[base] = {'index': i, 'variants': [base], 'binned': binned, 'split': split, 'bin_factor': bin_factor}
            bands_info_found = inferred
        # Build stitch_sequence (same as original)
        stitch_sequence = []
        ordered_bases = sorted(bands_info_found.keys(), key=lambda k: bands_info_found[k]['index'])
        binned_bases = [b for b in ordered_bases if bands_info_found[b].get('binned', False)]
        unbinned_bases = [b for b in ordered_bases if not bands_info_found[b].get('binned', False)]
        for base in binned_bases:
            info = bands_info_found[base]
            bin_factor = int(info.get('bin_factor', 1)) or 1
            per_block_h = max(1, int(round(float(orig_band_h) / float(bin_factor))))
            stitch_sequence.append({
                'base': base, 'kind': 'full_binned', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': False, 'side': None
            })
        for base in unbinned_bases:
            info = bands_info_found[base]
            bin_factor = int(info.get('bin_factor', 1)) or 1
            per_block_h = int(orig_band_h)
            if info.get('split', False):
                stitch_sequence.append({
                    'base': base, 'kind': 'half_left', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': True, 'side': 'left'
                })
                stitch_sequence.append({
                    'base': base, 'kind': 'half_right', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': True, 'side': 'right'
                })
            else:
                stitch_sequence.append({
                    'base': base, 'kind': 'full_unbinned', 'per_block_h': per_block_h, 'bin_factor': bin_factor, 'is_split': False, 'side': None
                })
        # Compute start_y
        cur_y = 0
        for entry in stitch_sequence:
            entry['start_y'] = cur_y
            cur_y += int(entry['per_block_h']) + gap
        print("DEBUG stitch_sequence (order,height,start_y):")
        for e in stitch_sequence:
            print(f" {e['kind']:12} base={e['base']:6} h={e['per_block_h']:4} start_y={e['start_y']:5} side={e.get('side')}")
        # --- Robust frame height calculation (handles binned and unbinned bands) ---
        if not stitch_sequence:
            return
        # Calculate frame_stitch_height from stitch_sequence entries (covers gaps implicitly)
        first_start = min(entry['start_y'] for entry in stitch_sequence)
        last_end = max(entry['start_y'] + entry['per_block_h'] for entry in stitch_sequence)
        frame_stitch_height = max(1, int(last_end - first_start))
        # Auto-detect binned vs unbinned entries (useful debugging/flags)
        unique_heights = sorted({entry['per_block_h'] for entry in stitch_sequence})
        if len(unique_heights) >= 2:
            # threshold halfway between smallest and largest height
            threshold = (unique_heights[0] + unique_heights[-1]) / 2.0
            for entry in stitch_sequence:
                entry['binned'] = (entry['per_block_h'] <= threshold)
        else:
            for entry in stitch_sequence:
                entry['binned'] = False
        # Helper to flip one rect
        def flip_rect(box, vertical, horizontal):
            left, top, right, bottom = [int(max(0, min(v, (full_w if i % 2 == 0 else full_h)))) for i, v in enumerate(box)]
            if right <= left or bottom <= top:
                return
            part = self.current_pil_image.crop((left, top, right, bottom))
            if vertical:
                flipped = part.transpose(Image.FLIP_TOP_BOTTOM)
            elif horizontal:
                flipped = part.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return
            self.current_pil_image.paste(flipped, (left, top))
        # --- Flip system (unchanged) ---
        def flip_region_for_entry_abs_y(y0, y1, vertical, horizontal):
            flip_rect((0, int(y0), full_w, int(y1)), vertical=vertical, horizontal=horizontal)
        # If "all" requested -> flip matching blocks across all stacked frames
        if all:
            if full_h > frame_stitch_height:
                n_frames_stacked = max(1, full_h // frame_stitch_height)
                for fidx in range(n_frames_stacked):
                    for entry in stitch_sequence:
                        # Logical positions (unflipped)
                        logical_y0 = fidx * frame_stitch_height + entry['start_y']
                        logical_y1 = logical_y0 + entry['per_block_h']
                        # Map to visual if globally flipped
                        if self.global_vertical_flipped:
                            y0 = full_h - logical_y1
                            y1 = full_h - logical_y0
                        else:
                            y0 = logical_y0
                            y1 = logical_y1
                        # Ensure y0 < y1 and bounds
                        if y0 >= y1 or y1 <= 0 or y0 >= full_h:
                            continue
                        y0 = max(0, y0)
                        y1 = min(full_h, y1)
                        flip_region_for_entry_abs_y(y0, y1, vertical, horizontal)
            else:
                # Single frame: same logical-to-visual mapping
                for entry in stitch_sequence:
                    # Logical positions (unflipped)
                    logical_y0 = entry['start_y']
                    logical_y1 = logical_y0 + entry['per_block_h']
                    # Map to visual if globally flipped
                    if self.global_vertical_flipped:
                        y0 = full_h - logical_y1
                        y1 = full_h - logical_y0
                    else:
                        y0 = logical_y0
                        y1 = logical_y1
                    # Ensure y0 < y1 and bounds
                    if y0 >= y1 or y1 <= 0 or y0 >= full_h:
                        continue
                    y0 = max(0, y0)
                    y1 = min(full_h, y1)
                    flip_region_for_entry_abs_y(y0, y1, vertical, horizontal)
        else:
            # Single-click behavior: map click to the correct stacked frame and the correct band inside it
            if click_pos is None:
                return
            cx, cy = click_pos
            if self.global_vertical_flipped:
                logical_cy = full_h - 1 - cy
            else:
                logical_cy = cy
            if frame_stitch_height > 0 and full_h > frame_stitch_height:
                frame_idx = int(logical_cy) // frame_stitch_height
                local_cy = int(logical_cy) - frame_idx * frame_stitch_height
                # find the entry whose local range contains local_cy
                found = None
                for entry in stitch_sequence:
                    y0_local = entry['start_y']
                    y1_local = y0_local + entry['per_block_h']
                    if local_cy >= y0_local and local_cy < y1_local:
                        found = entry
                        break
                # if not found (edge cases), choose nearest by distance
                if found is None:
                    # fallback to nearest entry by absolute distance to center of entry
                    best = None
                    best_d = None
                    for entry in stitch_sequence:
                        center = entry['start_y'] + entry['per_block_h'] / 2.0
                        d = abs(local_cy - center)
                        if best is None or d < best_d:
                            best = entry
                            best_d = d
                    found = best
                # compute logical absolute positions
                logical_local_y0 = found['start_y']
                logical_local_y1 = logical_local_y0 + found['per_block_h']
                logical_y0 = frame_idx * frame_stitch_height + logical_local_y0
                logical_y1 = frame_idx * frame_stitch_height + logical_local_y1
                # map to visual
                if self.global_vertical_flipped:
                    y0 = full_h - logical_y1
                    y1 = full_h - logical_y0
                else:
                    y0 = logical_y0
                    y1 = logical_y1
                flip_region_for_entry_abs_y(y0, y1, vertical, horizontal)
            else:
                # Single stitched image (not stacked) — original behavior but uses real per_block_h
                found = None
                for entry in stitch_sequence:
                    y0 = entry['start_y']
                    y1 = y0 + entry['per_block_h']
                    if logical_cy >= y0 and logical_cy < y1:
                        found = entry
                        break
                if found is None:
                    found = stitch_sequence[-1]
                logical_y0 = found['start_y']
                logical_y1 = logical_y0 + found['per_block_h']
                # map to visual
                if self.global_vertical_flipped:
                    y0 = full_h - logical_y1
                    y1 = full_h - logical_y0
                else:
                    y0 = logical_y0
                    y1 = logical_y1
                flip_region_for_entry_abs_y(y0, y1, vertical, horizontal)
        self.show_image(self.current_pil_image, fit_to_screen=False)
           
    def _reposition_fs_btn(self):
        """Place bottom overlay buttons in the graphics_view.viewport()."""
        try:
            vp = self.graphics_view.viewport()
            fs_x = max(0, vp.width() - self.fs_btn.width() - self._fs_margin)
            fs_y = max(0, vp.height() - self.fs_btn.height() - self._fs_margin)
            self.fs_btn.move(fs_x, fs_y)
            self.fs_btn.raise_()
            if hasattr(self, "grid_btn") and self.grid_btn is not None:
                grid_x = max(0, self._fs_margin)
                grid_y = max(0, vp.height() - self.grid_btn.height() - self._fs_margin)
                self.grid_btn.move(grid_x, grid_y)
                self.grid_btn.raise_()
        except Exception:
            pass
    def _on_grid_toggled(self, checked):
        try:
            self.graphics_view.toggle_grid(checked)
            self._reposition_fs_btn()
        except Exception:
            pass
    def eventFilter(self, obj, event):
        """Catch viewport Resize and reposition the fullscreen button."""
        try:
            if obj is self.graphics_view.viewport() and event.type() == QEvent.Resize:
                self._reposition_fs_btn()
                try:
                    self._reposition_status_overlay()
                except Exception:
                    pass
        except Exception:
            pass
        return super().eventFilter(obj, event)
    def _get_band_index_from_click(self, pos):
        if self.geo_info is None:
            return None
        try:
            parent_app = self.parent()
            merge_lr = False
            x_mapped, y_mapped = pos[0], pos[1]
            if parent_app and hasattr(parent_app, "_map_display_coords_for_geo"):
                x_mapped, y_mapped, merge_lr = parent_app._map_display_coords_for_geo(pos[0], pos[1])
            lat, lon, band_idx = image_coords_to_latlon(
                x_mapped, y_mapped,
                self.geo_info,
                bands_info=getattr(self.parent(), "bands_info", None),
                gap=getattr(self.parent(), "gap_var", 0),
                orig_band_h=self.parent().height_entry.text(),
                merge_lr=merge_lr
            )
            return band_idx
        except Exception:
            return None
    def toggle_fullscreen(self):
        self.is_fullscreen = not getattr(self, "is_fullscreen", False)
        # start from an explicit parent_viewer if present, otherwise climb parents
        parent_app = getattr(self, "parent_viewer", None) or self.parent()
        # climb until we find the main app object that owns left_scroll
        while parent_app is not None and not hasattr(parent_app, "left_scroll"):
            parent_app = parent_app.parent()
        if not parent_app:
            print("Error: Could not find main app for full-screen toggle")
            return
        main_window = parent_app.window() if hasattr(parent_app, 'window') else parent_app
        if self.is_fullscreen:
            # save UI state we will modify
            parent_app._saved_ui_state = {
                "left_scroll_visible": getattr(parent_app.left_scroll, "isVisible", lambda: True)(),
                "left_scroll_width": getattr(parent_app.left_scroll, "width", lambda: 0)()
            }
            # Save main window geometry
            main_window._saved_geometry = main_window.saveGeometry()
            # hide left panel (best-effort)
            try:
                parent_app.left_scroll.hide()
            except Exception:
                pass
            # make sure viewer's controls we want remain visible in fullscreen
            try:
                # these widgets exist on the viewer itself
                if hasattr(self, "magnifier_toggle"):
                    self.magnifier_toggle.show()
                if hasattr(self, "magnifier_zoom_slider"):
                    self.magnifier_zoom_slider.show()
            except Exception:
                pass
            # On an extended desktop, span the window across the full virtual
            # geometry instead of fullscreening only the current monitor.
            try:
                target_screen = None
                if main_window.windowHandle() is not None:
                    target_screen = main_window.windowHandle().screen()
                if target_screen is None:
                    target_screen = main_window.screen() if hasattr(main_window, "screen") else None

                virtual_geom = None
                if target_screen is not None:
                    siblings = target_screen.virtualSiblings()
                    if siblings and len(siblings) > 1:
                        virtual_geom = target_screen.virtualGeometry()

                if virtual_geom is not None and not virtual_geom.isNull():
                    main_window._saved_window_flags = int(main_window.windowFlags())
                    main_window._spanned_fullscreen = True
                    main_window.setWindowFlag(Qt.FramelessWindowHint, True)
                    main_window.show()
                    main_window.setGeometry(virtual_geom)
                    main_window.raise_()
                else:
                    main_window._spanned_fullscreen = False
                    main_window.showFullScreen()
            except Exception as e:
                print(f"Error entering fullscreen: {e}")
        else:
            # restore UI state
            try:
                state = getattr(parent_app, "_saved_ui_state", None)
                if state is not None:
                    if state.get("left_scroll_visible", True):
                        try:
                            parent_app.left_scroll.show()
                        except Exception:
                            pass
                    try:
                        parent_app.left_scroll.setFixedWidth(state.get("left_scroll_width", parent_app.left_scroll.width()))
                    except Exception:
                        pass
                    try:
                        del parent_app._saved_ui_state
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error restoring UI after fullscreen: {e}")
            # Restore main window geometry
            try:
                if hasattr(main_window, '_saved_geometry') and main_window._saved_geometry:
                    main_window.restoreGeometry(main_window._saved_geometry)
                    del main_window._saved_geometry
                else:
                    # Fallback to original size
                    main_window.resize(1920, 1080)
            except Exception as e:
                print(f"Error restoring geometry: {e}")
                main_window.resize(1920, 1080)
            # exit fullscreen
            try:
                if bool(getattr(main_window, "_spanned_fullscreen", False)):
                    saved_flags = getattr(main_window, "_saved_window_flags", None)
                    if saved_flags is not None:
                        main_window.setWindowFlags(Qt.WindowFlags(saved_flags))
                    main_window.showNormal()
                    if hasattr(main_window, "_saved_window_flags"):
                        del main_window._saved_window_flags
                    del main_window._spanned_fullscreen
                else:
                    main_window.showNormal()
            except Exception as e:
                print(f"Error exiting fullscreen: {e}")
       
    def fit_to_screen(self):
        # single, correct implementation of fit_to_screen for this viewer
        if not self.scene or not self.scene.items():
            return
        # clear any old transform so fitInView starts from identity
        try:
            self.graphics_view.resetTransform()
        except Exception:
            pass
        # make sure scene rect is correct (defensive)
        if self.pixmap_item and not self.scene.sceneRect().isValid():
            self.scene.setSceneRect(self.pixmap_item.sceneBoundingRect())
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom = self.graphics_view.transform().m11()
        self.graphics_view.viewport().update()
    def update_position_label(self, pos):
        mouse_scene = self.graphics_view.mapToScene(pos)
        ix, iy = self.get_original_coords(mouse_scene)
        ix_raw, iy_raw = math.floor(ix), math.floor(iy)
        ix_visual, iy_visual, _, _ = self._normalize_pixel_query(ix_raw, iy_raw)
        zoom_pct = int(self.zoom * 100)
        text = f"X: {ix_visual}, Y: {iy_visual} | Zoom: {zoom_pct}% | |"
        if self.geo_info and 0 <= ix_visual < self.full_width and 0 <= iy_visual < self.full_height:
            try:
                # Map to logical coordinates for geo
                ix_mapped, iy_mapped = ix_visual, iy_visual
                if self.global_horizontal_flipped:
                    ix_mapped = self.full_width - 1 - ix_visual
                if self.global_vertical_flipped:
                    iy_mapped = self.full_height - 1 - iy_visual
                # Climb parent hierarchy to find the main app with bands_info, gap_var, height_entry
                parent_app = self.parent()
                while parent_app and not (hasattr(parent_app, 'bands_info') and hasattr(parent_app, 'gap_var') and hasattr(parent_app, 'height_entry')):
                    parent_app = parent_app.parent()
                if parent_app:
                    bands_info = getattr(parent_app, 'bands_info', None)
                    gap = getattr(parent_app.gap_var, 'value', lambda: 0)()
                    orig_band_h_str = getattr(parent_app.height_entry, 'text', lambda: '')()
                    try:
                        orig_band_h = int(orig_band_h_str) if orig_band_h_str else None
                    except ValueError:
                        orig_band_h = None
                    merge_lr = False
                    ix_geo, iy_geo = ix_mapped, iy_mapped
                    if hasattr(parent_app, "_map_display_coords_for_geo"):
                        ix_geo, iy_geo, merge_lr = parent_app._map_display_coords_for_geo(ix_mapped, iy_mapped)
                    lat, lon, _ = image_coords_to_latlon(
                        ix_geo, iy_geo, self.geo_info,
                        bands_info=bands_info,
                        gap=gap,
                        orig_band_h=orig_band_h,
                        merge_lr=merge_lr
                    )
                    text += f" | Lat: {lat:.8f} | Lon: {lon:.8f}"
            except Exception:
                pass # Skip lat/lon if computation fails
        self.position_label.setText(text)
    def actual_size(self):
        self.graphics_view.resetTransform()
        self.zoom = 1.0
        self.graphics_view.viewport().update()
    def _emit_pixel_info_at(self, x, y):
        # Prefer caller-provided raw data for pixel info when available
        data_src = self.original_raw_data if getattr(self, 'original_raw_data', None) is not None else self.original_image_data
        if data_src is not None:
            display_x, display_y, sample_x, sample_y = self._normalize_pixel_query(x, y)
            try:
                data_h = int(data_src.shape[0])
                data_w = int(data_src.shape[1])
            except Exception:
                return
            if data_h <= 0 or data_w <= 0:
                return
            # Handle case where matrix_size_var is None (e.g., in editor tab)
            if self.matrix_size_var is not None:
                size = self.matrix_size_var.value()
            else:
                size = 5  # Default matrix size
            half = size // 2

            def _axis_window(center, limit, win_size):
                if limit <= 0:
                    return np.array([0] * max(1, win_size), dtype=int)
                start = int(center) - (win_size // 2)
                if start < 0:
                    start = 0
                max_start = max(0, int(limit) - int(win_size))
                if start > max_start:
                    start = max_start
                idx = np.arange(start, start + win_size, dtype=int)
                return np.clip(idx, 0, max(0, int(limit) - 1))

            safe_sample_x = max(0, min(data_w - 1, int(sample_x)))
            safe_sample_y = max(0, min(data_h - 1, int(sample_y)))
            x_idx = _axis_window(safe_sample_x, data_w, size)
            y_idx = _axis_window(safe_sample_y, data_h, size)
            values = np.take(np.take(data_src, y_idx, axis=0), x_idx, axis=1)
            # Treat pixel-info as RGB only for RGB Fusion view.
            is_rgb = (
                getattr(self, 'is_rgb_fusion', False) and
                getattr(data_src, 'ndim', 0) == 3 and
                data_src.shape[2] >= 3
            )
           
            # Only update overlay directly if there's NO external callback
            # (For raw_mode, the callback will handle the overlay update with raw data)
            if self.pixel_info_callback is None:
                if hasattr(self, 'pixel_info_box_overlay') and self.pixel_info_box_overlay is not None:
                    try:
                        self.pixel_info_box_overlay.update_info(int(display_x), int(display_y), values, is_rgb=is_rgb)
                    except Exception as e:
                        print(f"Error updating pixel info box overlay: {e}")
           
            # Call the external callback if provided (raw_mode will update overlay with raw data)
            if self.pixel_info_callback:
                self.pixel_info_callback(display_x, display_y, values, is_rgb)

    
