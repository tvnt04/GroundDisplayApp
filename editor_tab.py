from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QScrollArea, QMessageBox, QMenu, QUndoStack, QUndoCommand,
    QStackedWidget, QDialog, QFormLayout, QDialogButtonBox,
    QLineEdit, QCheckBox, QGraphicsView, QRadioButton, QGroupBox, QSlider, QApplication,
    QInputDialog
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QEvent, QMimeData
from PyQt5.QtGui import QPixmap, QTransform, QCursor, QDrag, QImage
import numpy as np
from PIL import Image
from image_viewer import GraphicsImageViewer, pil_to_qimage
from raw_mode import RawViewer
from tiled_viewer import TiledDisplay

# ---------------- Undo Commands ----------------
class SwapCommand(QUndoCommand):
    def __init__(self, editor, a, b):
        super().__init__("Swap Tiles")
        self.editor = editor
        self.a = a
        self.b = b
    def redo(self):
        self.editor._swap(self.a, self.b)
    def undo(self):
        self.editor._swap(self.a, self.b)

class FlipCommand(QUndoCommand):
    def __init__(self, editor, indices, lr=False, tb=False):
        count = len(indices)
        action = "Flip Tile" if count == 1 else f"Flip {count} Tiles"
        if lr:
            action += " Horizontally"
        if tb:
            action += " Vertically"
        super().__init__(action)
        self.editor = editor
        self.indices = indices
        self.lr = lr
        self.tb = tb
    def redo(self):
        self.editor._apply_flip(self.indices, self.lr, self.tb)
    def undo(self):
        self.editor._apply_flip(self.indices, self.lr, self.tb)

class RotateCommand(QUndoCommand):
    def __init__(self, editor, indices, k):
        super().__init__("Rotate Tiles")
        self.editor = editor
        self.indices = indices
        self.k = k
    def redo(self):
        self.editor._apply_rotate(self.indices, self.k)
    def undo(self):
        self.editor._apply_rotate(self.indices, -self.k)

class RotateOriginalCommand(QUndoCommand):
    def __init__(self, editor, k):
        super().__init__("Rotate Image")
        self.editor = editor
        self.k = k
        self.prev_array = self.editor.original_array.copy()
    def redo(self):
        self.editor.original_array = np.rot90(self.prev_array, self.k)
        self.editor.preview.show_image(Image.fromarray(self.editor.original_array), fit_to_screen=True)
    def undo(self):
        self.editor.original_array = self.prev_array.copy()
        self.editor.preview.show_image(Image.fromarray(self.editor.original_array), fit_to_screen=True)

class SplitRowCommand(QUndoCommand):
    def __init__(self, editor, row, prev_tiles, new_tiles):
        super().__init__("Split Row into Tiles")
        self.editor = editor
        self.row = row
        self.prev = prev_tiles
        self.new = new_tiles
    def redo(self):
        self.editor.tiles[self.row] = [t.copy() for t in self.new]
        self.editor.show_grid()
        self.editor.update_preview()
    def undo(self):
        self.editor.tiles[self.row] = [t.copy() for t in self.prev]
        self.editor.show_grid()
        self.editor.update_preview()

class SplitVerticalCommand(QUndoCommand):
    def __init__(self, editor, prev_tiles, new_tiles):
        super().__init__("Split Columns into Rows")
        self.editor = editor
        self.prev = [t.copy() for t in prev_tiles]
        self.new = [[tt.copy() for tt in row] for row in new_tiles]
    def redo(self):
        self.editor.tiles = [[tt.copy() for tt in row] for row in self.new]
        self.editor.show_grid()
        self.editor.update_preview()
    def undo(self):
        self.editor.tiles = [[t.copy() for t in self.prev]]
        self.editor.show_grid()
        self.editor.update_preview()

class DeleteTileCommand(QUndoCommand):
    def __init__(self, editor, r, c, tile):
        super().__init__("Delete Tile")
        self.editor = editor
        self.r = r
        self.c = c
        self.tile = tile
    def redo(self):
        del self.editor.tiles[self.r][self.c]
        self.editor.show_grid()
        self.editor.update_preview()
    def undo(self):
        self.editor.tiles[self.r].insert(self.c, self.tile.copy())
        self.editor.show_grid()
        self.editor.update_preview()

class DeleteRowCommand(QUndoCommand):
    def __init__(self, editor, row, tiles):
        super().__init__("Delete Row")
        self.editor = editor
        self.row = row
        self.tiles = tiles
    def redo(self):
        del self.editor.tiles[self.row]
        self.editor.show_grid()
        self.editor.update_preview()
    def undo(self):
        self.editor.tiles.insert(self.row, [t.copy() for t in self.tiles])
        self.editor.show_grid()
        self.editor.update_preview()

class DeleteColumnCommand(QUndoCommand):
    def __init__(self, editor, col, removed_tiles):
        super().__init__("Delete Column")
        self.editor = editor
        self.col = col
        self.removed_tiles = removed_tiles
    def redo(self):
        for r in range(len(self.editor.tiles)):
            if self.col < len(self.editor.tiles[r]):
                del self.editor.tiles[r][self.col]
        self.editor.show_grid()
        self.editor.update_preview()
    def undo(self):
        for r in range(len(self.editor.tiles)):
            if r < len(self.removed_tiles):
                self.editor.tiles[r].insert(self.col, self.removed_tiles[r].copy())
        self.editor.show_grid()
        self.editor.update_preview()

class CropCommand(QUndoCommand):
    def __init__(self, editor, prev_array, new_array, is_tile=False, index=None):
        super().__init__("Crop Image")
        self.editor = editor
        self.prev = prev_array.copy()
        self.new = new_array.copy()
        self.is_tile = is_tile
        self.index = index
    def redo(self):
        if not self.is_tile:
            self.editor.original_array = self.new.copy()
        else:
            r, c = self.index
            self.editor.tiles[r][c] = self.new.copy()
        self.editor.show_grid()
        self.editor.update_preview()
    def undo(self):
        if not self.is_tile:
            self.editor.original_array = self.prev.copy()
        else:
            r, c = self.index
            self.editor.tiles[r][c] = self.prev.copy()
        self.editor.show_grid()
        self.editor.update_preview()

# ---------------- Tile Widget ----------------
class TileWidget(QLabel):
    def __init__(self, editor, index):
        super().__init__()
        self.editor = editor
        self.index = index
        self.drag_start_pos = None
        self.setAlignment(Qt.AlignCenter)
        self.update_style()
        self.setAcceptDrops(True)
        self.update_pixmap()
        self.setMouseTracking(True)

    def update_style(self):
        if self.editor.seamless_view.isChecked():
            self.setStyleSheet("QLabel { border: none; background: transparent; padding: 0px; margin: 0px; }")
        else:
            self.setStyleSheet("QLabel { border: 1px solid #555; background: black; padding: 2px; }")

    def update_pixmap(self):
        r, c = self.index
        tile = self.editor.tiles[r][c]
        pil = Image.fromarray(tile)
        zoom = getattr(self.editor, 'grid_zoom', 1.0)
        display_h = max(1, int(pil.height * zoom + 0.5))
        forced_w = getattr(self, 'forced_width', None)
        if display_h > 0 and pil.height > 0:
            scale = display_h / pil.height
            if forced_w is None:
                new_w = max(1, int(pil.width * scale + 0.5))
            else:
                new_w = max(1, int(forced_w))
            resample = Image.NEAREST if display_h < 75 else Image.LANCZOS
            pil = pil.resize((new_w, display_h), resample)
        self.setPixmap(QPixmap.fromImage(pil_to_qimage(pil)))
        if self.editor.seamless_view.isChecked():
            self.setFixedHeight(display_h)
        else:
            try:
                self.setFixedHeight(self.sizeHint().height())
            except Exception:
                pass

    def mousePressEvent(self, event):
        if hasattr(self.editor, 'crop_mode') and self.editor.crop_mode.isChecked() and event.button() == Qt.LeftButton:
            self.editor.handle_manual_crop_click(self.index, event.pos(), local=True)
            return
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
        elif event.button() == Qt.RightButton:
            self.show_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if hasattr(self.editor, 'crop_mode') and self.editor.crop_mode.isChecked():
            self.setCursor(QCursor(Qt.CrossCursor))
        if not (event.buttons() & Qt.LeftButton) or not self.drag_start_pos:
            return
        if (event.pos() - self.drag_start_pos).manhattanLength() < 30:
            return
        if not self.editor.uniform_grid:
            return
        self.start_drag()
        self.drag_start_pos = None

    def start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(f"{self.index[0]}:{self.index[1]}")
        drag.setMimeData(mime)
        pixmap = self.pixmap()
        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.FastTransformation)
            drag.setPixmap(scaled)
            drag.setHotSpot(scaled.rect().center())
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, e):
        if e.mimeData().hasText() and self.editor.uniform_grid:
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if self.editor.uniform_grid:
            e.acceptProposedAction()

    def dropEvent(self, e):
        if e.mimeData().hasText() and self.editor.uniform_grid:
            text = e.mimeData().text()
            parts = text.split(':')
            if len(parts) == 2:
                try:
                    sr, sc = int(parts[0]), int(parts[1])
                    source = (sr, sc)
                    target = self.index
                    if source != target:
                        src_tile = self.editor.tiles[sr][sc]
                        tgt_tile = self.editor.tiles[target[0]][target[1]]
                        if src_tile.shape == tgt_tile.shape:
                            self.editor.swap_tiles(source, target)
                except Exception:
                    pass
            e.acceptProposedAction()

    def show_menu(self, pos):
        self.editor.show_tile_menu(self.index, pos)

# ---------------- Editor ----------------
class EditorTab(QWidget):
    def __init__(self, source_viewer, parent=None):
        super().__init__(parent)
        self.source_viewer = source_viewer
        self.original_array = np.array(
            source_viewer.current_pil_image.convert("L"),
            copy=True
        )
        self._preview_canvas_shape = tuple(int(v) for v in self.original_array.shape[:2])
        self.tiles = []
        self.undo_stack = QUndoStack(self)
        self.uniform_grid = False
        self.notified = False
        self.notified_force = False
        layout = QHBoxLayout(self)
        self.stack = QStackedWidget()
        self.preview = GraphicsImageViewer(parent)
        self.stack.addWidget(self.preview)
        self.grid_scroll = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QVBoxLayout(self.grid_widget)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(0)
        self.grid_scroll.setWidget(self.grid_widget)
        self.grid_scroll.setWidgetResizable(True)
        self.stack.addWidget(self.grid_scroll)
        # put the stack and a dedicated bottom-bar holder into a vertical container
        content_v = QVBoxLayout()
        content_v.setContentsMargins(0, 0, 0, 0)
        content_v.setSpacing(0)
        content_v.addWidget(self.stack)
        self._editor_bottom_holder = QWidget()
        bh_layout = QHBoxLayout(self._editor_bottom_holder)
        bh_layout.setContentsMargins(0, 0, 0, 0)
        bh_layout.setSpacing(6)
        self._editor_bottom_holder.setVisible(False)
        content_v.addWidget(self._editor_bottom_holder)
        layout.addLayout(content_v, 4)
        controls = QVBoxLayout()
        self.row_height = QSpinBox()
        self.row_height.setRange(1, self.original_array.shape[0])
        self.row_height.setValue(200)
        self.col_width = QSpinBox()
        self.col_width.setRange(1, self.original_array.shape[1])
        self.col_width.setValue(172)
        self.black_threshold = QSpinBox()
        self.black_threshold.setRange(0, 255)
        self.black_threshold.setValue(50)
        split_rows_btn = QPushButton("Split Image into Rows")
        split_rows_btn.setToolTip("Split image into row tiles")
        split_rows_btn.clicked.connect(self.show_split_image_rows_dialog)
        split_cols_btn = QPushButton("Split Image into Columns")
        split_cols_btn.setToolTip("Split image into column tiles")
        split_cols_btn.clicked.connect(self.show_split_image_columns_dialog)
        batch_split_btn = QPushButton("Batch Split Rows...")
        batch_split_btn.setToolTip("Batch split rows for many files")
        batch_split_btn.clicked.connect(self.show_batch_split_dialog)
        batch_vertical_btn = QPushButton("Batch Split Columns...")
        batch_vertical_btn.setToolTip("Batch split columns for many files")
        batch_vertical_btn.clicked.connect(self.show_batch_vertical_split_dialog)
        auto_crop_btn = QPushButton("Auto-Crop Borders")
        auto_crop_btn.setToolTip("Auto-crop dark borders")
        auto_crop_btn.clicked.connect(self.auto_crop)
        self.crop_mode = QCheckBox("Crop Mode (Click to Crop)")
        self.crop_mode.stateChanged.connect(self.toggle_crop_mode)
        crop_dir_group = QGroupBox("Crop Direction")
        crop_dir_layout = QHBoxLayout()
        self.crop_horizontal = QRadioButton("Horizontal")
        self.crop_vertical = QRadioButton("Vertical")
        self.crop_vertical.setChecked(True)
        crop_dir_layout.addWidget(self.crop_horizontal)
        crop_dir_layout.addWidget(self.crop_vertical)
        crop_dir_group.setLayout(crop_dir_layout)
        self.seamless_view = QCheckBox("Seamless editor view (no borders/spacing)")
        self.seamless_view.setChecked(False)
        self.seamless_view.hide()
        self.force_grid = QCheckBox("Force Grid Mode (swap equals only)")
        self.force_grid.stateChanged.connect(self.show_grid)
        self.remove_separators = QCheckBox("Remove near-black separator lines on apply")
        self.remove_separators.setChecked(False)
        self.remove_separators.hide()
        self.update_source = QCheckBox("Update original viewer on apply")
        self.update_source.setChecked(True)
        apply_btn = QPushButton("Apply & Create New Tab")
        apply_btn.setToolTip("Apply edits and create new tab")
        apply_btn.clicked.connect(self.apply)
        undo_btn = QPushButton("Undo")
        undo_btn.setToolTip("Undo last edit")
        undo_btn.clicked.connect(self.undo_stack.undo)
        redo_btn = QPushButton("Redo")
        redo_btn.setToolTip("Redo last edit")
        redo_btn.clicked.connect(self.undo_stack.redo)
        reset_btn = QPushButton("Reset to Original")
        reset_btn.setToolTip("Reset all edits")
        reset_btn.clicked.connect(self.reset)
        rotate_entire_btn = QPushButton("Rotate Entire 90° CW")
        rotate_entire_btn.setToolTip("Rotate full image 90° clockwise")
        rotate_entire_btn.clicked.connect(self.rotate_entire)
        controls.addWidget(QLabel("Row Height (px)"))
        controls.addWidget(self.row_height)
        controls.addWidget(split_rows_btn)
        controls.addWidget(batch_split_btn)
        controls.addWidget(QLabel("Column Width (px)"))
        controls.addWidget(self.col_width)
        controls.addWidget(split_cols_btn)
        controls.addWidget(batch_vertical_btn)
        controls.addWidget(QLabel("Black Threshold (0-255)"))
        controls.addWidget(self.black_threshold)
        controls.addWidget(auto_crop_btn)
        controls.addWidget(self.crop_mode)
        controls.addWidget(crop_dir_group)
        controls.addWidget(self.force_grid)
        controls.addWidget(self.update_source)
        controls.addWidget(apply_btn)
        controls.addWidget(undo_btn)
        controls.addWidget(redo_btn)
        controls.addWidget(reset_btn)
        controls.addWidget(rotate_entire_btn)
        self.grid_zoom = 1.0
        self.editor_view_mode = "canvas"
        # Keep these attributes for backward compatibility in helper methods.
        self.grid_zoom_in_btn = None
        self.grid_zoom_out_btn = None
        self.grid_zoom_reset_btn = None
        # small persistent badge to indicate Grid Mode (shown only when uniform grid active)
        self._grid_mode_badge = QLabel("GRID MODE — drag tiles to rearrange", self.grid_scroll.viewport())
        self._grid_mode_badge.setStyleSheet(
            "QLabel { background: rgba(40, 120, 40, 200); color: white; padding: 6px 10px; border-radius: 12px; font-weight: bold; }")
        self._grid_mode_badge.setVisible(False)
        self._grid_mode_badge.setFixedHeight(26)
        self._grid_mode_badge.setContentsMargins(8, 2, 8, 2)
        self._grid_mode_badge.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        # clicking badge shows short help
        self._grid_mode_badge.mousePressEvent = lambda ev: QMessageBox.information(self, "Grid mode",
            "Grid mode is active — tiles are uniform so you can drag to reorder.\nTurn off 'Seamless editor view' to see borders.")
        # Right-panel preview zoom controls removed by request.
        controls.addStretch()
        layout.addLayout(controls, 1)
        self.preview.show_image(
            Image.fromarray(self.original_array),
            fit_to_screen=True
        )
        # Install event filters last
        self.preview.installEventFilter(self)
        try:
            self.preview.graphics_view.viewport().installEventFilter(self)
        except Exception:
            pass
        self.grid_scroll.installEventFilter(self)
        self._canvas_drag_source = None
        self._canvas_drag_start_global = None
        self._canvas_drag_active = False
        self._canvas_drag_target = None
        self._drag_override_cursor_active = False
        self._drag_cursor_pixmap = None  # Keep pixmap alive during drag
        self._set_editor_view("canvas")

    def eventFilter(self, obj, event):
        if obj == self.grid_scroll and event.type() == QEvent.Wheel:
            return True
        preview_targets = [self.preview]
        try:
            preview_targets.append(self.preview.graphics_view.viewport())
        except Exception:
            pass
        if obj in preview_targets and self.tiles and not self.crop_mode.isChecked():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                src = self._tile_index_from_preview_click(event.globalPos())
                if src is not None:
                    self._canvas_drag_source = src
                    self._canvas_drag_start_global = event.globalPos()
                    self._canvas_drag_active = False
                    self._canvas_drag_target = None
                    self._set_override_cursor(QCursor(Qt.OpenHandCursor))
                    self.update_preview()
                return False
            if event.type() == QEvent.MouseMove and self._canvas_drag_source is not None:
                try:
                    tgt = self._tile_index_from_preview_click(event.globalPos())
                    if tgt != self._canvas_drag_target:
                        self._canvas_drag_target = tgt
                        self.update_preview()
                    if self._canvas_drag_start_global is not None:
                        if (event.globalPos() - self._canvas_drag_start_global).manhattanLength() >= 8:
                            if not self._canvas_drag_active:
                                self._canvas_drag_active = True
                                if not self._set_drag_cursor_from_source():
                                    self._set_override_cursor(QCursor(Qt.ClosedHandCursor))
                except Exception:
                    pass
                return False
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                try:
                    if self._canvas_drag_source is not None and self._canvas_drag_active:
                        tgt = self._tile_index_from_preview_click(event.globalPos())
                        src = self._canvas_drag_source
                        if tgt is not None and tgt != src:
                            if self.force_grid.isChecked():
                                s = self.tiles[src[0]][src[1]].shape
                                t = self.tiles[tgt[0]][tgt[1]].shape
                                if s != t:
                                    QMessageBox.warning(self, "Swap Blocked", "Force Grid Mode: only equal-size tiles can be swapped.")
                                    return True
                            self.swap_tiles(src, tgt)
                            return True
                finally:
                    self._canvas_drag_source = None
                    self._canvas_drag_start_global = None
                    self._canvas_drag_active = False
                    self._canvas_drag_target = None
                    self._clear_override_cursor()
                    self.update_preview()
        if obj in preview_targets and self.tiles and event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            try:
                idx = self._tile_index_from_preview_click(event.globalPos())
                if idx is not None:
                    self.show_tile_menu(idx, event.globalPos())
                    return True
            except Exception:
                pass
        if obj in preview_targets and self.crop_mode.isChecked() and event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.handle_manual_crop_click(None, event.pos(), local=False)
            return True
        return super().eventFilter(obj, event)

    def _tile_index_from_original_coords(self, x, y):
        if not self.tiles:
            return None
        xx = int(x)
        yy = int(y)
        if xx < 0 or yy < 0:
            return None
        cur_y = 0
        for r, row_tiles in enumerate(self.tiles):
            if not row_tiles:
                continue
            row_h = max(1, max(int(t.shape[0]) for t in row_tiles))
            if cur_y <= yy < (cur_y + row_h):
                cur_x = 0
                for c, tile in enumerate(row_tiles):
                    w = int(tile.shape[1])
                    if cur_x <= xx < (cur_x + w):
                        return (r, c)
                    cur_x += w
                return None
            cur_y += row_h
        return None

    def _tile_rect_from_index(self, index):
        """Get bounding rect of tile in original coords (used for visual feedback)."""
        if not self.tiles or index is None:
            return None
        r, c = index
        if r < 0 or r >= len(self.tiles) or c < 0 or c >= len(self.tiles[r]):
            return None
        y = 0
        for rr in range(r):
            row_tiles = self.tiles[rr]
            if row_tiles:
                y += max(1, max(int(t.shape[0]) for t in row_tiles))
        x = 0
        for cc in range(c):
            x += int(self.tiles[r][cc].shape[1])
        tile = self.tiles[r][c]
        return (x, y, int(tile.shape[1]), int(tile.shape[0]))

    def _set_override_cursor(self, cursor):
        try:
            if self._drag_override_cursor_active:
                QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(cursor)
            self._drag_override_cursor_active = True
        except Exception:
            pass

    def _clear_override_cursor(self):
        try:
            if self._drag_override_cursor_active:
                QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self._drag_override_cursor_active = False

    def _set_drag_cursor_from_source(self):
        """Set cursor to dragged tile. Return False to use default ClosedHandCursor instead."""
        # Don't create a pixmap preview - just use standard cursors
        # The pixmap approach causes rendering artifacts/noise
        return False

    def _tile_index_from_preview_click(self, global_pos):
        gv = getattr(self.preview, 'graphics_view', None)
        if gv is None:
            return None
        vp = gv.viewport()
        local = vp.mapFromGlobal(global_pos)
        if not vp.rect().contains(local):
            return None
        scene_pos = gv.mapToScene(local)
        ox, oy = self.preview.get_original_coords(scene_pos)
        return self._tile_index_from_original_coords(ox, oy)

    def _prompt_swap_target(self, source):
        if not self.tiles:
            return None
        text, ok = QInputDialog.getText(
            self, "Swap Tile", "Target tile as row,col (0-based), e.g. 2,3:"
        )
        if not ok:
            return None
        try:
            parts = [p.strip() for p in str(text).split(",")]
            if len(parts) != 2:
                raise ValueError("bad format")
            r = int(parts[0])
            c = int(parts[1])
        except Exception:
            QMessageBox.warning(self, "Invalid Input", "Use format row,col (0-based), e.g. 2,3")
            return None
        if r < 0 or r >= len(self.tiles) or c < 0 or c >= len(self.tiles[r]):
            QMessageBox.warning(self, "Invalid Target", "Target tile index is out of range.")
            return None
        if self.force_grid.isChecked():
            s = self.tiles[source[0]][source[1]].shape
            t = self.tiles[r][c].shape
            if s != t:
                QMessageBox.warning(self, "Swap Blocked", "Force Grid Mode: only equal-size tiles can be swapped.")
                return None
        if (r, c) == source:
            return None
        return (r, c)

    def show_tile_menu(self, index, pos):
        r, c = index
        if r < 0 or r >= len(self.tiles) or c < 0 or c >= len(self.tiles[r]):
            return
        m = QMenu(self)
        num_tiles = len(self.tiles[r])
        num_rows = len(self.tiles)
        if self.is_full_row(r):
            m.addAction("Split this row into tiles...", lambda: self.show_split_row_dialog(r))
        if num_rows == 1 and self.is_full_col(c):
            m.addAction("Split all columns into rows...", lambda: self.show_batch_vertical_split_dialog())
        m.addSeparator()
        m.addAction("Flip this tile Left ↔ Right", lambda: self.flip_tiles([index], lr=True))
        m.addAction("Flip this tile Top ↔ Bottom", lambda: self.flip_tiles([index], tb=True))
        m.addAction("Rotate this tile 90° CW", lambda: self.rotate_tiles([index], -1))
        m.addAction("Rotate this tile 180°", lambda: self.rotate_tiles([index], 2))
        m.addAction("Rotate this tile 90° CCW", lambda: self.rotate_tiles([index], 1))
        m.addAction(
            "Swap with tile...",
            lambda: (lambda tgt: self.swap_tiles(index, tgt) if tgt is not None else None)(self._prompt_swap_target(index))
        )
        if num_tiles > 1:
            m.addSeparator()
            m.addAction("Flip ENTIRE ROW Left ↔ Right", lambda: self.flip_row(r, lr=True))
            m.addAction("Flip ENTIRE ROW Top ↔ Bottom", lambda: self.flip_row(r, tb=True))
            m.addAction("Rotate ENTIRE ROW 90° CW", lambda: self.rotate_row(r, -1))
            m.addAction("Rotate ENTIRE ROW 180°", lambda: self.rotate_row(r, 2))
            m.addAction("Rotate ENTIRE ROW 90° CCW", lambda: self.rotate_row(r, 1))
        m.addSeparator()
        m.addAction("Delete this tile", lambda: self.delete_tile(index))
        m.addAction("Delete this row", lambda: self.delete_row(r))
        m.addAction("Delete this column", lambda: self.delete_column(c))
        m.exec_(pos)

    def toggle_crop_mode(self, state):
        cursor = QCursor(Qt.CrossCursor) if state else QCursor(Qt.ArrowCursor)
        self.preview.setCursor(cursor)
        for tw in self.grid_widget.findChildren(TileWidget):
            tw.setCursor(cursor)

    def handle_manual_crop_click(self, index, pos, local):
        direction = 'horizontal' if self.crop_horizontal.isChecked() else 'vertical'
        if local and index is not None:
            dialog_title = f"Crop Tile ({direction.capitalize()})"
            size = self.tiles[index[0]][index[1]].shape[0 if direction == 'horizontal' else 1]
        else:
            dialog_title = f"Crop Global ({direction.capitalize()})"
            size = self.original_array.shape[0 if direction == 'horizontal' else 1]
        dialog = QDialog(self)
        dialog.setWindowTitle(dialog_title)
        form = QFormLayout(dialog)
        left_top = QSpinBox()
        left_top.setRange(0, size)
        left_top.setValue(0)
        right_bottom = QSpinBox()
        right_bottom.setRange(0, size)
        right_bottom.setValue(0)
        if direction == 'horizontal':
            form.addRow("Crop Top (px):", left_top)
            form.addRow("Crop Bottom (px):", right_bottom)
        else:
            form.addRow("Crop Left (px):", left_top)
            form.addRow("Crop Right (px):", right_bottom)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        crop_lt = left_top.value()
        crop_rb = right_bottom.value()
        if crop_lt + crop_rb >= size:
            QMessageBox.warning(self, "Error", "Crop amounts too large")
            return
        if local and index is not None:
            r, c = index
            prev = self.tiles[r][c].copy()
            if direction == 'horizontal':
                new = prev[crop_lt:prev.shape[0] - crop_rb, :]
            else:
                new = prev[:, crop_lt:prev.shape[1] - crop_rb]
            command = CropCommand(self, prev, new, is_tile=True, index=index)
            self.undo_stack.push(command)
        else:
            prev = self.original_array.copy()
            if direction == 'horizontal':
                new = prev[crop_lt:prev.shape[0] - crop_rb, :]
            else:
                new = prev[:, crop_lt:prev.shape[1] - crop_rb]
            command = CropCommand(self, prev, new)
            self.undo_stack.push(command)
            self.original_array = new.copy()
            if self.tiles:
                for rr in range(len(self.tiles)):
                    for cc in range(len(self.tiles[rr])):
                        t = self.tiles[rr][cc]
                        if direction == 'horizontal':
                            self.tiles[rr][cc] = t[crop_lt:t.shape[0] - crop_rb, :]
                        else:
                            self.tiles[rr][cc] = t[:, crop_lt:t.shape[1] - crop_rb]
                self.show_grid()
                self.update_preview()

    def auto_crop(self):
        prev = self.original_array.copy()
        new = self.enhanced_crop(prev)
        if new.shape == prev.shape:
            QMessageBox.information(self, "Auto-Crop", "No borders detected to crop.")
            return
        command = CropCommand(self, prev, new)
        self.undo_stack.push(command)
        self.original_array = new.copy()
        if self.tiles:
            for r in range(len(self.tiles)):
                for c in range(len(self.tiles[r])):
                    self.tiles[r][c] = self.enhanced_crop(self.tiles[r][c])
            self.show_grid()
            self.update_preview()

    def enhanced_crop(self, array):
        if array.size == 0:
            return array
        def is_border_col(col_data):
            mean = np.mean(col_data)
            var = np.var(col_data)
            return mean <= self.black_threshold.value() and var < 50
        def is_border_row(row_data):
            mean = np.mean(row_data)
            var = np.var(row_data)
            return mean <= self.black_threshold.value() and var < 50
        good_cols = np.logical_not(np.array([is_border_col(array[:, c]) for c in range(array.shape[1])]))
        if np.any(good_cols):
            x_min = np.argmax(good_cols)
            x_max = array.shape[1] - np.argmax(good_cols[::-1]) - 1
            array = array[:, x_min:x_max + 1]
        good_rows = np.logical_not(np.array([is_border_row(array[r, :]) for r in range(array.shape[0])]))
        if np.any(good_rows):
            y_min = np.argmax(good_rows)
            y_max = array.shape[0] - np.argmax(good_rows[::-1]) - 1
            array = array[y_min:y_max + 1, :]
        return array

    def delete_row(self, row):
        if QMessageBox.question(self, "Confirm", "Delete this row? (Undoable)") != QMessageBox.Yes:
            return
        prev_tiles = [t.copy() for t in self.tiles[row]]
        command = DeleteRowCommand(self, row, prev_tiles)
        self.undo_stack.push(command)

    def delete_column(self, col):
        if QMessageBox.question(self, "Confirm", "Delete this column? (Undoable)") != QMessageBox.Yes:
            return
        removed = []
        max_cols = max((len(row) for row in self.tiles), default=0)
        if col >= max_cols:
            QMessageBox.warning(self, "Error", "Invalid column index")
            return
        for r in range(len(self.tiles)):
            if col < len(self.tiles[r]):
                removed.append(self.tiles[r][col].copy())
            else:
                removed.append(np.zeros_like(self.tiles[r][0][:, :0]) if self.tiles[r] else np.array([]))
        command = DeleteColumnCommand(self, col, removed)
        self.undo_stack.push(command)

    def zoom_in_grid(self):
        self._zoom_preview(1.25)

    def zoom_out_grid(self):
        self._zoom_preview(0.8)

    def reset_grid_zoom(self):
        self._zoom_preview('fit')

    def _zoom_grid(self, factor):
        if factor == 0:
            return
        self._zoom_preview(factor)

    def show_split_image_rows_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Split Image into Rows")
        form = QFormLayout(dialog)
        height_spin = QSpinBox()
        height_spin.setRange(1, self.original_array.shape[0])
        height_spin.setValue(self.row_height.value())
        form.addRow("Row Height:", height_spin)
        dir_group = QGroupBox("Direction")
        dir_layout = QHBoxLayout()
        top_to_bottom = QRadioButton("Top to Bottom")
        top_to_bottom.setChecked(True)
        bottom_to_top = QRadioButton("Bottom to Top")
        dir_layout.addWidget(top_to_bottom)
        dir_layout.addWidget(bottom_to_top)
        dir_group.setLayout(dir_layout)
        form.addRow(dir_group)
        trim_rem = QCheckBox("Trim remainder to make even (unlocks grid mode)")
        form.addRow(trim_rem)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        row_h = height_spin.value()
        trim_remainder = trim_rem.isChecked()
        ttb = top_to_bottom.isChecked()
        if self.tiles:
            if len(self.tiles) == 1 and all(self.is_full_col(c) for c in range(len(self.tiles[0]))):
                self.show_batch_vertical_split_dialog()
            else:
                QMessageBox.warning(self, "Error", "Image already split, use context menu for individual edits")
            return
        self.undo_stack.clear()
        self.tiles = []
        img_h, img_w = self.original_array.shape
        if row_h < 1 or row_h >= img_h:
            QMessageBox.warning(self, "Error", "Invalid row height")
            return
        num_full = img_h // row_h
        remain = img_h % row_h
        start = 0 if ttb or not trim_remainder else remain
        end = img_h if not trim_remainder else start + num_full * row_h
        new_rows = []
        y = start
        while y < end:
            slice_h = min(row_h, end - y)
            strip = self.original_array[y:y + slice_h, :].copy()
            new_rows.append([strip])
            y += row_h
        if not ttb:
            new_rows.reverse()
        self.tiles = new_rows
        self.show_grid()
        self.update_preview()
        self.stack.setCurrentWidget(self.preview)
        if remain > 0 and not trim_remainder:
            QMessageBox.information(self, "Note", "Remainder row included.")
        elif remain > 0:
            QMessageBox.information(self, "Note", "Remainder trimmed.")

    def show_split_image_columns_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Split Image into Columns")
        form = QFormLayout(dialog)
        width_spin = QSpinBox()
        width_spin.setRange(1, self.original_array.shape[1])
        width_spin.setValue(self.col_width.value())
        form.addRow("Column Width:", width_spin)
        dir_group = QGroupBox("Direction")
        dir_layout = QHBoxLayout()
        left_to_right = QRadioButton("Left to Right")
        left_to_right.setChecked(True)
        right_to_left = QRadioButton("Right to Left")
        dir_layout.addWidget(left_to_right)
        dir_layout.addWidget(right_to_left)
        dir_group.setLayout(dir_layout)
        form.addRow(dir_group)
        trim_rem = QCheckBox("Trim remainder to make even (unlocks grid mode)")
        form.addRow(trim_rem)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        col_w = width_spin.value()
        trim_remainder = trim_rem.isChecked()
        ltr = left_to_right.isChecked()
        if self.tiles:
            if all(self.is_full_row(r) for r in range(len(self.tiles))):
                self.show_batch_split_dialog()
            else:
                QMessageBox.warning(self, "Error", "Image already split, use context menu for individual edits")
            return
        self.undo_stack.clear()
        self.tiles = []
        img_h, img_w = self.original_array.shape
        if col_w < 1 or col_w >= img_w:
            QMessageBox.warning(self, "Error", "Invalid column width")
            return
        num_full = img_w // col_w
        remain = img_w % col_w
        start = 0 if ltr or not trim_remainder else remain
        end = img_w if not trim_remainder else start + num_full * col_w
        new_cols = []
        x = start
        while x < end:
            slice_w = min(col_w, end - x)
            strip = self.original_array[:, x:x + slice_w].copy()
            new_cols.append(strip)
            x += col_w
        if not ltr:
            new_cols.reverse()
        self.tiles = [new_cols]
        self.show_grid()
        self.update_preview()
        self.stack.setCurrentWidget(self.preview)
        if remain > 0 and not trim_remainder:
            QMessageBox.information(self, "Note", "Remainder column included.")
        elif remain > 0:
            QMessageBox.information(self, "Note", "Remainder trimmed.")

    def is_full_row(self, row):
        return len(self.tiles[row]) == 1 and self.tiles[row][0].shape[1] == self.original_array.shape[1]

    def is_full_col(self, col):
        if len(self.tiles) != 1:
            return False
        return len(self.tiles[0]) > col and self.tiles[0][col].shape[0] == self.original_array.shape[0]

    def _perform_split(self, row, tile_w, trim_remainder=False, left_to_right=True):
        """Split a full-width row into tiles without creating or removing pixels.

        Guarantees:
        - When trim_remainder is False (default) tiles form a strict partition of
          the entire row: sum(tile.width) == original_row.width and heights
          are unchanged. The final tile may be smaller than `tile_w` to absorb
          the remainder.
        - When trim_remainder is True only full tiles of width `tile_w` are
          produced and the leftover remainder is discarded (no remainder tile).
        - No padding, synthetic pixels, or UI filler are introduced.
        - Tiles are returned in left-to-right storage order (so np.hstack(tiles)
          reconstructs the original row or its trimmed prefix exactly).
        """
        if not self.is_full_row(row):
            return False
        previous_tiles = [t.copy() for t in self.tiles[row]]
        row_array = self.tiles[row][0].copy()
        h, total_w = row_array.shape

        # Validate
        if tile_w <= 0:
            QMessageBox.warning(self, "Error", "Invalid tile width")
            return False
        # Nothing to do if tile width is >= entire row
        if tile_w >= total_w:
            return False

        new_tiles = []
        if trim_remainder:
            # Produce only full tiles; remainder discarded.
            num_full = total_w // tile_w
            if num_full == 0:
                return False
            if left_to_right:
                widths = [tile_w] * num_full
            else:
                # remainder is on the left; storage remains left-to-right
                widths = [tile_w] * num_full
                # we'll slice from an offset below so left remainder is discarded
                offset = total_w - num_full * tile_w
        else:
            # Preserve full width. Build widths so they sum to total_w.
            full_count = total_w // tile_w
            rem = total_w % tile_w
            if rem == 0:
                widths = [tile_w] * full_count
            else:
                if left_to_right:
                    widths = [tile_w] * full_count + [rem]
                else:
                    widths = [rem] + [tile_w] * full_count

        # Build slices in left-to-right order
        if trim_remainder and not left_to_right:
            # start slicing after the left-side remainder that we discard
            offset = total_w - len(widths) * tile_w
            pos = offset
            for w in widths:
                new_tiles.append(row_array[:, pos:pos + w].copy())
                pos += w
        else:
            pos = 0
            for w in widths:
                new_tiles.append(row_array[:, pos:pos + w].copy())
                pos += w

        # Sanity checks: no zero-width tiles, heights unchanged, widths sum as expected
        if any(t.shape[1] == 0 for t in new_tiles):
            QMessageBox.warning(self, "Error", "Split produced a zero-width tile (internal)")
            return False
        if any(t.shape[0] != h for t in new_tiles):
            QMessageBox.warning(self, "Error", "Split changed tile height (internal)")
            return False

        expected_w = (total_w // tile_w) * tile_w if trim_remainder else total_w
        got_w = sum(t.shape[1] for t in new_tiles)
        if got_w != expected_w:
            QMessageBox.warning(self, "Error", "Split produced inconsistent widths (internal)")
            return False

        # Commit
        self.tiles[row] = [t.copy() for t in new_tiles]
        command = SplitRowCommand(self, row, previous_tiles, [t.copy() for t in new_tiles])
        self.undo_stack.push(command)
        return True

    def show_split_row_dialog(self, row):
        dialog = QDialog(self)
        dialog.setWindowTitle("Split Row into Tiles")
        form = QFormLayout(dialog)
        width_spin = QSpinBox()
        width_spin.setRange(1, self.tiles[row][0].shape[1])
        width_spin.setValue(self.col_width.value())
        form.addRow("Tile Width:", width_spin)
        dir_group = QGroupBox("Direction")
        dir_layout = QHBoxLayout()
        left_to_right = QRadioButton("Left to Right")
        left_to_right.setChecked(True)
        right_to_left = QRadioButton("Right to Left")
        dir_layout.addWidget(left_to_right)
        dir_layout.addWidget(right_to_left)
        dir_group.setLayout(dir_layout)
        form.addRow(dir_group)
        trim_rem = QCheckBox("Trim remainder to make even (unlocks grid mode)")
        form.addRow(trim_rem)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        tile_w = width_spin.value()
        trim_remainder = trim_rem.isChecked()
        ltr = left_to_right.isChecked()
        self._perform_split(row, tile_w, trim_remainder, ltr)
        self.show_grid()
        self.update_preview()

    def show_batch_split_dialog(self):
        if not self.tiles:
            QMessageBox.warning(self, "Error", "First split the image into rows")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Split Multiple Rows")
        form = QFormLayout(dialog)
        width_spin = QSpinBox()
        width_spin.setRange(1, self.original_array.shape[1])
        width_spin.setValue(self.col_width.value())
        form.addRow("Tile Width:", width_spin)
        dir_group = QGroupBox("Direction")
        dir_layout = QHBoxLayout()
        left_to_right = QRadioButton("Left to Right")
        left_to_right.setChecked(True)
        right_to_left = QRadioButton("Right to Left")
        dir_layout.addWidget(left_to_right)
        dir_layout.addWidget(right_to_left)
        dir_group.setLayout(dir_layout)
        form.addRow(dir_group)
        trim_rem = QCheckBox("Trim remainder to make even (unlocks grid mode)")
        form.addRow(trim_rem)
        row_input = QLineEdit()
        row_input.setPlaceholderText("e.g. 0,1,3-5,8,10-15 (blank for all)")
        form.addRow("Rows to split:", row_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        text = row_input.text().strip()
        if text:
            try:
                rows_to_split = set()
                for part in text.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        rows_to_split.update(range(start, end + 1))
                    else:
                        rows_to_split.add(int(part))
            except Exception:
                QMessageBox.warning(self, "Error", "Invalid row format")
                return
        else:
            rows_to_split = set(range(len(self.tiles)))
        tile_w = width_spin.value()
        trim_remainder = trim_rem.isChecked()
        ltr = left_to_right.isChecked()
        applied = 0
        for r in sorted(rows_to_split):
            if 0 <= r < len(self.tiles) and self._perform_split(r, tile_w, trim_remainder, ltr):
                applied += 1
        if applied:
            self.show_grid()
            self.update_preview()
        if applied:
            QMessageBox.information(self, "Done", f"Batch split applied to {applied} rows")
        else:
            QMessageBox.information(self, "Info", "No rows were split")

    def show_batch_vertical_split_dialog(self):
        if len(self.tiles) != 1:
            QMessageBox.warning(self, "Error", "First split the image into columns (must be one row)")
            return
        if not self.tiles[0]:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Split All Columns into Rows")
        form = QFormLayout(dialog)
        height_spin = QSpinBox()
        height_spin.setRange(1, self.tiles[0][0].shape[0])
        height_spin.setValue(self.row_height.value())
        form.addRow("Row Height:", height_spin)
        dir_group = QGroupBox("Direction")
        dir_layout = QHBoxLayout()
        top_to_bottom = QRadioButton("Top to Bottom")
        top_to_bottom.setChecked(True)
        bottom_to_top = QRadioButton("Bottom to Top")
        dir_layout.addWidget(top_to_bottom)
        dir_layout.addWidget(bottom_to_top)
        dir_group.setLayout(dir_layout)
        form.addRow(dir_group)
        trim_rem = QCheckBox("Trim remainder to make even (unlocks grid mode)")
        form.addRow(trim_rem)
        column_input = QLineEdit()
        column_input.setPlaceholderText("e.g. 0,1,3-5,8,10-15 (blank for all)")
        form.addRow("Columns to split:", column_input)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        text = column_input.text().strip()
        if text:
            try:
                columns_to_split = set()
                for part in text.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        columns_to_split.update(range(start, end + 1))
                    else:
                        columns_to_split.add(int(part))
            except Exception:
                QMessageBox.warning(self, "Error", "Invalid column format")
                return
        else:
            columns_to_split = set(range(len(self.tiles[0])))
        if columns_to_split != set(range(len(self.tiles[0]))):
            QMessageBox.warning(self, "Error", "Partial column selection not supported. Use blank for all.")
            return
        row_h = height_spin.value()
        trim_remainder = trim_rem.isChecked()
        ttb = top_to_bottom.isChecked()
        previous_tiles = self.tiles[0]
        total_h = previous_tiles[0].shape[0]
        num_full = total_h // row_h
        remain = total_h % row_h
        start = 0 if ttb or not trim_remainder else remain
        end = total_h if not trim_remainder else start + num_full * row_h
        new_tiles = []
        y = start
        while y < end:
            slice_h = min(row_h, end - y)
            new_row = []
            for cc in range(len(previous_tiles)):
                tile = previous_tiles[cc]
                sub_tile = tile[y:y + slice_h, :].copy()
                new_row.append(sub_tile)
            new_tiles.append(new_row)
            y += row_h
        if not ttb:
            new_tiles.reverse()
        command = SplitVerticalCommand(self, previous_tiles, new_tiles)
        self.undo_stack.push(command)

    def flip_row(self, row, lr=False, tb=False):
        indices = [(row, c) for c in range(len(self.tiles[row]))]
        self.flip_tiles(indices, lr=lr, tb=tb)

    def rotate_row(self, row, k):
        indices = [(row, c) for c in range(len(self.tiles[row]))]
        self.rotate_tiles(indices, k)

    def rotate_entire(self):
        if not self.tiles:
            command = RotateOriginalCommand(self, -1)
            self.undo_stack.push(command)
        else:
            indices = [(rr, cc) for rr in range(len(self.tiles)) for cc in range(len(self.tiles[rr]))]
            self.rotate_tiles(indices, -1)

    def reset(self):
        self.tiles = []
        self.undo_stack.clear()
        self.stack.setCurrentWidget(self.preview)
        self.show_grid()
        self.update_preview()
        try:
            self._move_bottom_bar(False)
        except Exception:
            pass

    def _move_bottom_bar(self, to_editor):
        """Show editor-side *proxy* controls (to_editor=True) or remove them (to_editor=False).
        IMPORTANT: we do NOT reparent the original viewer widgets anymore — that broke
        some viewer-local behavior. Instead we create small proxy controls that forward
        actions to the real widgets and stay synced with them.
        """
        if to_editor:
            self._ensure_bottom_proxies()
            self._editor_bottom_holder.setVisible(True)
        else:
            # destroy proxies and hide holder
            self._destroy_bottom_proxies()
            self._editor_bottom_holder.setVisible(False)

    def _ensure_bottom_proxies(self):
        """Create proxy widgets in the editor bottom-holder that forward to
        the `GraphicsImageViewer` controls and stay synchronized.
        """
        if getattr(self, '_bottom_proxies', None):
            return
        src = getattr(self.preview, '_bottom_layout', None)
        if src is None:
            return
        # ordered list of attribute names we want to proxy (keeps layout order)
        names = [
            'measure_mode_btn', 'magnifier_toggle', 'torch_toggle',
            'magnifier_zoom_label', 'magnifier_zoom_slider',
            'zoom_out_btn', 'reset_zoom_btn', 'zoom_in_btn',
            'mouse_zoom_btn', 'flip_mode_btn', 'rotation_mode_btn'
        ]
        self._bottom_proxies = {}
        tgt_layout = self._editor_bottom_holder.layout()
        # clear any existing items in holder (defensive)
        for i in reversed(range(tgt_layout.count())):
            it = tgt_layout.takeAt(i)
            w = it.widget()
            if w:
                w.deleteLater()
        for name in names:
            orig = getattr(self.preview, name, None)
            if orig is None:
                continue
            proxy = None
            # create matching proxy type and set initial state
            if isinstance(orig, QCheckBox):
                proxy = QCheckBox(orig.text())
                proxy.setChecked(orig.isChecked())
                proxy.setToolTip(orig.toolTip())
                # two-way sync
                proxy.toggled.connect(lambda v, o=orig: o.setChecked(v))
                orig.toggled.connect(proxy.setChecked)
            elif isinstance(orig, QPushButton):
                proxy = QPushButton(orig.text())
                proxy.setCheckable(orig.isCheckable())
                proxy.setChecked(orig.isChecked())
                proxy.setToolTip(orig.toolTip())
                # keep check-state in sync for checkable controls
                if orig.isCheckable():
                    orig.toggled.connect(proxy.setChecked)
                # special handling for zoom controls: center preview on image then delegate
                if name in ('zoom_in_btn', 'zoom_out_btn', 'reset_zoom_btn'):
                    def _center_and_delegate(o=orig, nm=name):
                        # In grid view, zoom controls operate on the tile grid only.
                        if self.stack.currentWidget() == self.grid_scroll:
                            try:
                                if nm == 'zoom_in_btn':
                                    self._zoom_grid(1.25)
                                elif nm == 'zoom_out_btn':
                                    self._zoom_grid(0.8)
                                else:
                                    self.reset_grid_zoom()
                            except Exception:
                                pass
                            return
                        try:
                            # center the viewer on the image (if available) so zoom anchors on image center
                            pi = getattr(self.preview, 'pixmap_item', None)
                            if pi is not None and not pi.pixmap().isNull():
                                center = pi.sceneBoundingRect().center()
                                try:
                                    self.preview.graphics_view.centerOn(center)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # call the real viewer button (keeps preview behavior)
                        try:
                            o.click()
                        except Exception:
                            pass
                    proxy.clicked.connect(_center_and_delegate)
                else:
                    # default forwarding for non-zoom buttons: keep original behavior
                    proxy.clicked.connect(lambda _checked=False, o=orig: o.click())
                    # rotation/flip toggles should at least update editor preview state (no-op by default)
                    if name == 'mouse_zoom_btn':
                        proxy.clicked.connect(lambda _checked=False: None) # placeholder for future editor behavior
            elif isinstance(orig, QLabel):
                proxy = QLabel(orig.text())
            elif isinstance(orig, QSlider):
                proxy = QSlider(Qt.Horizontal)
                proxy.setRange(orig.minimum(), orig.maximum())
                proxy.setValue(orig.value())
                proxy.setToolTip(orig.toolTip())
                proxy.valueChanged.connect(lambda v, o=orig: o.setValue(v))
                orig.valueChanged.connect(proxy.setValue)
            else:
                # fallback: show a disabled label so layout spacing remains
                proxy = QLabel(name)
                proxy.setEnabled(False)
            proxy.setEnabled(orig.isEnabled())
            proxy.setFixedHeight(orig.sizeHint().height())
            tgt_layout.addWidget(proxy)
            self._bottom_proxies[name] = (orig, proxy)
        tgt_layout.addStretch()

    def _destroy_bottom_proxies(self):
        if not getattr(self, '_bottom_proxies', None):
            return
        tgt_layout = self._editor_bottom_holder.layout()
        # disconnect and delete proxies
        for name, (orig, proxy) in list(self._bottom_proxies.items()):
            try:
                proxy.deleteLater()
            except Exception:
                pass
        self._bottom_proxies = {}
        # clear leftover layout items
        for i in reversed(range(tgt_layout.count())):
            it = tgt_layout.takeAt(i)
            w = it.widget()
            if w:
                w.deleteLater()

    def delete_tile(self, index):
        r, c = index
        if QMessageBox.question(self, "Confirm", "Delete this tile? (Undoable)") != QMessageBox.Yes:
            return
        prev_tile = self.tiles[r][c].copy()
        command = DeleteTileCommand(self, r, c, prev_tile)
        self.undo_stack.push(command)

    def show_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        if not self.tiles:
            # ensure preview is visible and bottom bar is restored
            self.stack.setCurrentWidget(self.preview)
            try:
                self._move_bottom_bar(False)
            except Exception:
                pass
            return
        # Canvas-only editor: keep tile model and actions, but always render in preview.
        all_shapes = set()
        for r in range(len(self.tiles)):
            for t in self.tiles[r]:
                all_shapes.add(tuple(t.shape))
        uniform = len(all_shapes) == 1
        self.uniform_grid = uniform or self.force_grid.isChecked()
        if self.force_grid.isChecked() and not uniform and not self.notified_force:
            self.notified_force = True
            QMessageBox.information(self, "Force Grid Mode", "Force grid mode enabled: can only swap tiles with equal shapes.")
        if self.uniform_grid and not self.notified:
            self.notified = True
            QMessageBox.information(self, "Grid Mode", "Grid mode unlocked! Drag tiles to rearrange.")
        self.stack.setCurrentWidget(self.preview)
        try:
            self._move_bottom_bar(False)
        except Exception:
            pass
        # Grid badge is not used in canvas-only mode.
        try:
            badge = getattr(self, '_grid_mode_badge', None)
            if badge is not None:
                badge.hide()
        except Exception:
            pass

    def _set_editor_view(self, mode):
        self.editor_view_mode = "canvas"
        self.stack.setCurrentWidget(self.preview)
        try:
            self._move_bottom_bar(False)
        except Exception:
            pass
        self.update_preview()

    def refresh_tiles_display(self):
        self.update_preview()

    def _preview_pad_value(self):
        """Return an 8-bit grayscale value to use for visual padding in the preview.
        Prefer the preview widget background so padding looks like UI background
        (visual-only — does NOT change the actual image data used by Apply).
        """
        try:
            # ask the preview widget for its background color and convert to luminance
            pal = self.preview.palette()
            col = pal.color(self.preview.backgroundRole())
            r, g, b, _ = col.getRgb()
            # convert to luma (ITU-R BT.601)
            luma = int(round(0.299 * r + 0.587 * g + 0.114 * b))
            return max(0, min(255, luma))
        except Exception:
            return 0

    def _compute_row_display_crop(self, row_index):
        """Compute a display-only vertical crop for one tile row in seamless mode."""
        try:
            row_tiles = self.tiles[row_index]
            if not row_tiles:
                return None
            max_h = max(int(t.shape[0]) for t in row_tiles)
            thr = float(self.black_threshold.value())
            non_empty = np.zeros(max_h, dtype=bool)
            for tile in row_tiles:
                if tile.size == 0:
                    continue
                h = int(tile.shape[0])
                if h <= 0:
                    continue
                means = np.mean(tile, axis=1)
                non_empty[:h] |= (means > thr)
            if not np.any(non_empty):
                return None
            y0 = int(np.argmax(non_empty))
            y1 = int(len(non_empty) - np.argmax(non_empty[::-1]))
            if y1 <= y0:
                return None
            return (y0, y1)
        except Exception:
            return None

    def _tile_for_grid_display(self, tile, r=None, c=None):
        """Display helper for grid mode.
        In seamless mode, trim empty top/bottom margins (display-only)
        to avoid artificial-looking gaps between split rows.
        """
        if tile is None or tile.size == 0 or not self.seamless_view.isChecked():
            return tile
        try:
            if r is None or r < 0 or r >= len(self.tiles):
                return tile
            crop = None
            crops = getattr(self, "_row_display_crops", None)
            if isinstance(crops, dict):
                crop = crops.get(r)
            if crop is None:
                crop = self._compute_row_display_crop(r)
            if crop is None:
                return tile
            y0, y1 = crop
            h = int(tile.shape[0])
            if h <= 0:
                return tile
            y0 = max(0, min(y0, h - 1))
            y1 = max(y0 + 1, min(y1, h))
            trimmed = tile[y0:y1, :]
            return trimmed if trimmed.size else tile
        except Exception:
            return tile

    def _sync_preview_canvas_from_original(self):
        """Align preview canvas geometry to current base image geometry."""
        try:
            h, w = self.original_array.shape[:2]
            if h > 0 and w > 0:
                self._preview_canvas_shape = (int(h), int(w))
        except Exception:
            pass

    def _compose_preview_canvas(self):
        """Render tile edits into a fixed preview canvas (stable world coordinates)."""
        self._sync_preview_canvas_from_original()
        canvas_h, canvas_w = self._preview_canvas_shape
        bg_val = self._preview_pad_value()
        # Split guides should be subtle and stable (not pure black).
        sep_val = int(max(0, min(255, 0.6 * bg_val + 80)))
        # Keep guides visible when zoomed out by increasing thickness in image pixels.
        guide_px = 1
        try:
            gv = getattr(self.preview, "graphics_view", None)
            z = abs(float(gv.transform().m11())) if gv is not None else abs(float(getattr(self.preview, "zoom", 1.0)))
            z = max(0.05, z)
            guide_px = max(1, min(12, int(round(1.25 / z))))
        except Exception:
            guide_px = 1
        canvas = np.full((canvas_h, canvas_w), bg_val, dtype=self.original_array.dtype)
        if not self.tiles:
            h = min(canvas_h, self.original_array.shape[0])
            w = min(canvas_w, self.original_array.shape[1])
            if h > 0 and w > 0:
                canvas[:h, :w] = self.original_array[:h, :w]
            return canvas
        row_boundaries = []
        col_bounds_sets = []
        used_w = 0
        y = 0
        for row_tiles in self.tiles:
            if not row_tiles:
                continue
            row_h = max(1, max(int(t.shape[0]) for t in row_tiles))
            x = 0
            row_col_bounds = []
            for tile in row_tiles:
                h = int(tile.shape[0])
                w = int(tile.shape[1])
                if h <= 0 or w <= 0:
                    x += max(0, w)
                    continue
                if y < canvas_h and x < canvas_w:
                    copy_h = min(h, canvas_h - y)
                    copy_w = min(w, canvas_w - x)
                    if copy_h > 0 and copy_w > 0:
                        canvas[y:y + copy_h, x:x + copy_w] = tile[:copy_h, :copy_w]
                x += w
                row_col_bounds.append(x)
            if row_col_bounds:
                # exclude row-end boundary
                col_bounds_sets.append(set(row_col_bounds[:-1]))
            used_w = max(used_w, min(x, canvas_w))
            y += row_h
            row_boundaries.append(y)
            if y >= canvas_h:
                break
        used_h = min(y, canvas_h)
        # Display-only split guides (1px) so users can clearly see row/column splits.
        for by in row_boundaries[:-1]:
            if 0 <= by < used_h and used_w > 0:
                y1 = min(used_h, by + guide_px)
                canvas[by:y1, :used_w] = sep_val
        # Draw vertical guides only for boundaries common to all rows (prevents patchy segments).
        common_col_bounds = set.intersection(*col_bounds_sets) if col_bounds_sets else set()
        for bx in sorted(common_col_bounds):
            if 0 <= bx < used_w and used_h > 0:
                x1 = min(used_w, bx + guide_px)
                canvas[:used_h, bx:x1] = sep_val
        # Drag feedback overlays (display-only): source and target tile outlines.
        try:
            src = self._canvas_drag_source
            tgt = self._canvas_drag_target
            src_rect = self._tile_rect_from_index(src)
            tgt_rect = self._tile_rect_from_index(tgt)
            src_val = int(max(0, min(255, sep_val - 30)))
            tgt_val = int(max(0, min(255, sep_val + 30)))
            outline = max(1, min(4, guide_px))
            def _draw_rect(rect, val):
                if rect is None:
                    return
                rx, ry, rw, rh = rect
                if rw <= 0 or rh <= 0:
                    return
                x0 = max(0, rx)
                y0 = max(0, ry)
                x1 = min(canvas_w, rx + rw)
                y1 = min(canvas_h, ry + rh)
                if x1 <= x0 or y1 <= y0:
                    return
                oy = min(outline, y1 - y0)
                ox = min(outline, x1 - x0)
                canvas[y0:y0 + oy, x0:x1] = val
                canvas[y1 - oy:y1, x0:x1] = val
                canvas[y0:y1, x0:x0 + ox] = val
                canvas[y0:y1, x1 - ox:x1] = val
            _draw_rect(src_rect, src_val)
            if tgt is not None and tgt != src:
                _draw_rect(tgt_rect, tgt_val)
        except Exception:
            pass
        return canvas

    def _swap(self, a, b):
        ra, ca = a
        rb, cb = b
        self.tiles[ra][ca], self.tiles[rb][cb] = self.tiles[rb][cb], self.tiles[ra][ca]
        self.show_grid()
        self.update_preview()

    def swap_tiles(self, a, b):
        self.undo_stack.push(SwapCommand(self, a, b))

    def _apply_flip(self, indices, lr, tb):
        for r, c in indices:
            t = self.tiles[r][c]
            if lr:
                t = np.fliplr(t)
            if tb:
                t = np.flipud(t)
            self.tiles[r][c] = t.copy()
        self.show_grid()
        self.update_preview()

    def flip_tiles(self, indices, lr=False, tb=False):
        self.undo_stack.push(FlipCommand(self, indices, lr, tb))

    def _apply_rotate(self, indices, k):
        for r, c in indices:
            t = self.tiles[r][c]
            t = np.rot90(t, k)
            self.tiles[r][c] = t.copy()
        self.show_grid()
        self.update_preview()

    def rotate_tiles(self, indices, k):
        self.undo_stack.push(RotateCommand(self, indices, k))

    def is_empty_col(self, array, col):
        return np.mean(array[:, col]) <= self.black_threshold.value()

    def is_empty_row(self, array, row):
        return np.mean(array[row, :]) <= self.black_threshold.value()

    def crop_left_right(self, array):
        if array.size == 0:
            return array
        good_cols = np.logical_not(np.array([self.is_empty_col(array, c) for c in range(array.shape[1])]))
        if not np.any(good_cols):
            return array[:, 0:0]
        x_min = np.argmax(good_cols)
        x_max = len(good_cols) - 1 - np.argmax(good_cols[::-1])
        return array[:, x_min:x_max + 1]

    def crop_top_bottom(self, array):
        if array.size == 0:
            return array
        good_rows = np.logical_not(np.array([self.is_empty_row(array, r) for r in range(array.shape[0])]))
        if not np.any(good_rows):
            return array[0:0, :]
        y_min = np.argmax(good_rows)
        y_max = len(good_rows) - 1 - np.argmax(good_rows[::-1])
        return array[y_min:y_max + 1, :]

    def remove_separator_lines(self, array):
        # Option removed from UI: do not auto-strip separator lines on apply.
        return array

    def update_preview(self):
        full = self._compose_preview_canvas()
        self.preview.show_image(Image.fromarray(full), fit_to_screen=False)

    def apply(self):
        row_images = []
        for row_tiles in self.tiles:
            if row_tiles:
                row_full = np.hstack(row_tiles)
                row_full = self.crop_left_right(row_full)
                row_images.append(row_full)
        if not row_images:
            final = self.original_array
        else:
            max_w = max((img.shape[1] for img in row_images), default=0)
            padded_rows = []
            for img in row_images:
                if img.shape[1] < max_w:
                    pad_w = max_w - img.shape[1]
                    pad = np.zeros((img.shape[0], pad_w), dtype=img.dtype)
                    img = np.hstack((img, pad))
                padded_rows.append(img)
            full = np.vstack(padded_rows)
            full = self.crop_top_bottom(full)
            full = self.remove_separator_lines(full)
            full = self.crop_top_bottom(self.crop_left_right(full))
            final = full
        pil_final = Image.fromarray(final)
        if self.update_source.isChecked():
            try:
                if hasattr(self.source_viewer, 'show_image'):
                    # Preferred path for GraphicsImageViewer / band viewers
                    self.source_viewer.show_image(pil_final, fit_to_screen=False)
                else:
                    st = type(self.source_viewer).__name__
                    if st == 'RawViewer':
                        sv = self.source_viewer
                        sv.raw_data = final.copy()
                        if getattr(sv, 'bitdepth', 8) > 8:
                            max_val = (1 << sv.bitdepth) - 1
                            sv.normalized_data = ((final.astype(np.float32) / max_val) * 255.0).clip(0, 255).astype(np.uint8)
                        else:
                            sv.normalized_data = final.astype(np.uint8)
                        sv.current_pil_image = pil_final
                        # Refresh display safely if methods exist
                        if hasattr(sv, 'update_contrast_controls'):
                            sv.update_contrast_controls()
                        if hasattr(sv, 'update_display'):
                            sv.update_display()
                        if hasattr(sv, 'update_histogram'):
                            sv.update_histogram()
                    elif st == 'TiledDisplay':
                        sv = self.source_viewer
                        sv.original_tiles_per_frame = [[final]]
                        sv.tile_rotations = [[0]]
                        if hasattr(self.source_viewer, 'settings') and isinstance(self.source_viewer.settings, dict):
                            sv.settings = self.source_viewer.settings.copy()
                        if hasattr(sv, 'matrix_size_var') and hasattr(self.source_viewer, 'matrix_size_var'):
                            try:
                                sv.matrix_size_var.setValue(self.source_viewer.matrix_size_var.value())
                            except Exception:
                                pass
                        sv.tile_w_eff = final.shape[1]
                        sv.tile_h_eff = final.shape[0]
                        sv.current_frame = 0
                        if hasattr(sv, 'show_frame'):
                            sv.show_frame()
                        try:
                            sv.lbl_status.setText("Edited single frame")
                        except Exception:
                            pass
                        for ctrl in ('play_btn','frame_slider','btn_prev','btn_next','lbl_frame'):
                            try:
                                w = getattr(sv, ctrl)
                                if ctrl == 'lbl_frame':
                                    w.setText("Frame 1 / 1")
                                else:
                                    w.setEnabled(False)
                            except Exception:
                                pass
                    else:
                        # Unknown viewer type — skip updating source
                        pass
            except Exception:
                # Non-fatal: prefer creating the edited tab even if source update fails
                pass
        # Find the main app *that has a tab container* (support both `view_tabs` and `tab_widget`).
        main_app = self.parent()
        max_iter = 12
        while main_app and max_iter > 0 and not (hasattr(main_app, 'view_tabs') or hasattr(main_app, 'tab_widget')):
            main_app = main_app.parent()
            max_iter -= 1
        if not main_app:
            # fallback to active window
            top_widget = QApplication.activeWindow()
            if top_widget and (hasattr(top_widget, 'view_tabs') or hasattr(top_widget, 'tab_widget')):
                main_app = top_widget
        if not main_app:
            print("[DEBUG] Could not find main app to create new tab.")
            return

        tab_container = getattr(main_app, 'view_tabs', None) or getattr(main_app, 'tab_widget', None)
        if tab_container is None:
            QMessageBox.warning(self, "Warning", "Main app does not expose a tab container.")
            return

        new_tab = QWidget()
        new_layout = QVBoxLayout(new_tab)
        source_type = type(self.source_viewer).__name__
        if source_type == 'RawViewer':
            new_viewer = RawViewer(main_app)
            new_viewer.bitdepth = self.source_viewer.bitdepth
            new_viewer.raw_data = final.copy()
            if new_viewer.bitdepth > 8:
                max_val = (1 << new_viewer.bitdepth) - 1
                new_viewer.normalized_data = ((final.astype(np.float32) / max_val) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                new_viewer.normalized_data = final.astype(np.uint8)
            new_viewer.current_pil_image = pil_final
            new_viewer.update_contrast_controls()
            new_viewer.update_display()
            new_viewer.update_histogram()
        elif source_type == 'TiledDisplay':
            new_viewer = TiledDisplay(main_app)
            new_viewer.original_tiles_per_frame = [[final]]
            new_viewer.tile_rotations = [[0]]
            if hasattr(self.source_viewer, 'settings'):
                try:
                    new_viewer.settings = self.source_viewer.settings.copy()
                except Exception:
                    pass
            try:
                new_viewer.matrix_size_var.setValue(self.source_viewer.matrix_size_var.value())
            except Exception:
                pass
            new_viewer.tile_w_eff = final.shape[1]
            new_viewer.tile_h_eff = final.shape[0]
            new_viewer.current_frame = 0
            if hasattr(new_viewer, 'show_frame'):
                new_viewer.show_frame()
            try:
                new_viewer.lbl_status.setText("Edited single frame")
            except Exception:
                pass
            for ctrl in ('play_btn', 'frame_slider', 'btn_prev', 'btn_next', 'lbl_frame'):
                try:
                    w = getattr(new_viewer, ctrl)
                    if ctrl == 'lbl_frame':
                        w.setText("Frame 1 / 1")
                    else:
                        w.setEnabled(False)
                except Exception:
                    pass
        else:
            new_viewer = GraphicsImageViewer(
                parent=main_app,
                pixel_info_callback=getattr(main_app, 'update_pixel_info', None),
                matrix_size_var=getattr(main_app, 'matrix_size_var', None)
            )
            new_viewer.show_image(pil_final, fit_to_screen=True)

        new_viewer.geo_info = getattr(self.source_viewer, 'geo_info', None)
        new_layout.addWidget(new_viewer)

        # Derive tab name from the source viewer's tab (use the same tab container API)
        source_tab_name = "Edited Image"
        try:
            source_parent = self.source_viewer.parent()
            while source_parent and not isinstance(source_parent, QWidget):
                source_parent = source_parent.parent()
            if source_parent:
                source_tab_idx = tab_container.indexOf(source_parent)
                if source_tab_idx >= 0:
                    source_tab_name = tab_container.tabText(source_tab_idx)
            new_tab_name = f"Edited {source_tab_name}"
        except Exception:
            new_tab_name = "Edited Image"

        tab_container.addTab(new_tab, new_tab_name)
        tab_container.setCurrentIndex(tab_container.count() - 1)
        try:
            if hasattr(main_app, '_set_custom_close_button'):
                main_app._set_custom_close_button(tab_container.count() - 1)
        except Exception:
            pass
        else:
            print("[DEBUG] Could not find main app to create new tab.")
            return
        QMessageBox.information(self, "Applied", "Image reconstructed — seamless with no gaps or separators. New tab created.")

    def _zoom_preview(self, factor):
        try:
            if self.stack.currentWidget() == self.grid_scroll:
                self.stack.setCurrentWidget(self.preview)
                try:
                    self._move_bottom_bar(False)
                except Exception:
                    pass
            gv = None
            if hasattr(self.preview, 'view'):
                gv = self.preview.view
            elif hasattr(self.preview, 'graphics_view'):
                gv = self.preview.graphics_view
            if gv and isinstance(gv, QGraphicsView):
                if factor == 'fit':
                    gv.setTransformationAnchor(QGraphicsView.NoAnchor)
                    gv.resetTransform()
                    gv.fitInView(gv.sceneRect(), Qt.KeepAspectRatio)
                    gv.centerOn(gv.sceneRect().center())
                    return
                else:
                    gv.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
                    gv.scale(factor, factor)
                return
            if hasattr(self.preview, 'scale_view'):
                self.preview.scale_view(factor)
            elif hasattr(self.preview, 'zoom'):
                self.preview.zoom *= factor
        except Exception:
            pass
