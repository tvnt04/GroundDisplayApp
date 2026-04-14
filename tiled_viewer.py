import sys
import os
import cv2
import numpy as np
import json
from utils import load_folder_params, save_params_for_path, add_recent, get_recents_for_mode, select_from_history
from help_tab import create_help_tab
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDialog, QFormLayout, QSpinBox,
    QComboBox, QLineEdit, QMessageBox, QScrollArea,
    QTabWidget, QCheckBox, QSlider, QProgressDialog, QMenu, QTextEdit, QToolButton, QShortcut
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QKeySequence
from PyQt5.QtCore import Qt, QEvent, QTimer

class TileOrder:
    ROW_MAJOR = "Row-Major (Left → Right, Top → Bottom)"
    COLUMN_MAJOR = "Column-Major (Top → Bottom, Left → Right)"
    ROW_SERPENTINE = "Row Serpentine (Zigzag Rows)"
    COLUMN_SERPENTINE = "Column Serpentine (Zigzag Columns)"

    @staticmethod
    def get_index(order, r, c, rows, cols):
        if order == TileOrder.ROW_MAJOR:
            return r * cols + c
        elif order == TileOrder.COLUMN_MAJOR:
            return c * rows + r
        elif order == TileOrder.ROW_SERPENTINE:
            return r * cols + (c if r % 2 == 0 else (cols - 1 - c))
        elif order == TileOrder.COLUMN_SERPENTINE:
            return c * rows + (r if c % 2 == 0 else (rows - 1 - r))
        return r * cols + c

class SettingsDialog(QDialog):
    def __init__(self, parent=None, existing_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Load & Settings")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.dir_edit = QLineEdit()
        dir_btn = QPushButton("Browse")
        dir_btn.setToolTip("Choose session folder")
        dir_btn.clicked.connect(self.browse)
        dir_hbox = QHBoxLayout()
        dir_hbox.addWidget(self.dir_edit)
        dir_hbox.addWidget(dir_btn)
        form.addRow("Session Folder:", dir_hbox)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Folder per Frame", "Flat Tiles"])
        form.addRow("Mode:", self.mode_combo)

        self.grid_cols = QSpinBox(); self.grid_cols.setRange(1, 50); self.grid_cols.setValue(6)
        form.addRow("Grid Columns:", self.grid_cols)
        self.grid_rows = QSpinBox(); self.grid_rows.setRange(1, 50); self.grid_rows.setValue(4)
        form.addRow("Grid Rows:", self.grid_rows)

        self.tile_w = QSpinBox(); self.tile_w.setRange(1, 16384); self.tile_w.setValue(1024)
        form.addRow("Tile Width (px):", self.tile_w)
        self.tile_h = QSpinBox(); self.tile_h.setRange(1, 16384); self.tile_h.setValue(1024)
        form.addRow("Tile Height (px):", self.tile_h)

        self.overlap = QSpinBox(); self.overlap.setRange(0, 1024); self.overlap.setValue(0)
        form.addRow("Overlap (px):", self.overlap)

        self.bit_depth = QSpinBox(); self.bit_depth.setRange(8, 32); self.bit_depth.setValue(8); self.bit_depth.setSingleStep(8)
        form.addRow("Bit Depth:", self.bit_depth)

        self.order_combo = QComboBox()
        self.order_combo.addItems([TileOrder.COLUMN_MAJOR, TileOrder.ROW_MAJOR,
                                   TileOrder.ROW_SERPENTINE, TileOrder.COLUMN_SERPENTINE])
        form.addRow("Tile Order:", self.order_combo)

        self.final_w = QSpinBox(); self.final_w.setRange(1, 32768); self.final_w.setValue(6480)
        form.addRow("Final Width:", self.final_w)
        self.final_h = QSpinBox(); self.final_h.setRange(1, 32768); self.final_h.setValue(4860)
        form.addRow("Final Height:", self.final_h)

        self.matrix_size = QSpinBox(); self.matrix_size.setRange(3, 11); self.matrix_size.setValue(5); self.matrix_size.setSingleStep(2)
        form.addRow("Pixel Matrix Size (odd):", self.matrix_size)

        layout.addLayout(form)

        btn_box = QHBoxLayout()
        ok = QPushButton("Load / Apply")
        ok.setToolTip("Load tiles with current settings")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.setToolTip("Close settings without applying")
        cancel.clicked.connect(self.reject)
        btn_box.addStretch()
        btn_box.addWidget(ok)
        btn_box.addWidget(cancel)
        layout.addLayout(btn_box)

        if existing_settings:
            self.load_existing(existing_settings)

    def browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select Session Folder")
        if d:
            self.dir_edit.setText(d)

    def get_values(self):
        return {
            'session': self.dir_edit.text().strip(),
            'folder_mode': self.mode_combo.currentText() == "Folder per Frame",
            'cols': self.grid_cols.value(),
            'rows': self.grid_rows.value(),
            'tile_w': self.tile_w.value(),
            'tile_h': self.tile_h.value(),
            'overlap': self.overlap.value(),
            'bit_depth': self.bit_depth.value(),
            'order': self.order_combo.currentText(),
            'final_w': self.final_w.value(),
            'final_h': self.final_h.value(),
            'matrix_size': self.matrix_size.value()
        }

    def load_existing(self, s):
        self.dir_edit.setText(s.get('session', ''))
        self.mode_combo.setCurrentIndex(0 if s.get('folder_mode', True) else 1)
        self.grid_cols.setValue(s.get('cols', 6))
        self.grid_rows.setValue(s.get('rows', 4))
        self.tile_w.setValue(s.get('tile_w', 1024))
        self.tile_h.setValue(s.get('tile_h', 1024))
        self.overlap.setValue(s.get('overlap', 0))
        self.bit_depth.setValue(s.get('bit_depth', 8))
        self.order_combo.setCurrentText(s.get('order', TileOrder.COLUMN_MAJOR))
        self.final_w.setValue(s.get('final_w', 6480))
        self.final_h.setValue(s.get('final_h', 4860))
        self.matrix_size.setValue(s.get('matrix_size', 5))

class PixelInfoWidget(QWidget):
    def __init__(self, parent=None, matrix_size_var=None):
        super().__init__(parent)
        self.matrix_size_var = matrix_size_var
        self.setFixedWidth(280)
        self.setFixedHeight(200)

        # Default to dark theme initially
        self.update_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Matrix Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["3", "5", "7", "9"])
        try:
            self.size_combo.setCurrentText(str(matrix_size_var.value()))
        except Exception:
            pass
        self.size_combo.currentTextChanged.connect(lambda v: matrix_size_var.setValue(int(v)))
        control_layout.addWidget(self.size_combo)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFontFamily("Consolas")
        self.info_text.setFontPointSize(9)
        self.info_text.setFixedHeight(150)
        self.info_text.setAcceptRichText(True)
        layout.addWidget(self.info_text)

    def update_theme(self):
        app = QApplication.instance()
        if app:
            pal = app.palette()
            bg_lightness = pal.color(QPalette.Window).lightness()
            text_lightness = pal.color(QPalette.WindowText).lightness()
            is_dark = text_lightness > bg_lightness
        else:
            is_dark = True  # Fallback to dark

        if is_dark:
            widget_bg = "rgba(0, 0, 0, 0.85)"
            widget_border = "rgba(255, 255, 255, 0.3)"
            widget_color = "white"
            inner_bg = "rgba(0, 0, 0, 0.8)"
            inner_border = "rgba(255, 255, 255, 0.3)"
        else:
            widget_bg = "rgba(255, 255, 255, 0.85)"
            widget_border = "rgba(0, 0, 0, 0.3)"
            widget_color = "black"
            inner_bg = "rgba(255, 255, 255, 0.9)"
            inner_border = "rgba(0, 0, 0, 0.3)"

        self.setStyleSheet(
            f"QWidget {{ background-color: {widget_bg}; border: 1px solid {widget_border}; border-radius: 4px; color: {widget_color}; }}"
            f"QTextEdit {{ background-color: {inner_bg}; border: none; color: {widget_color}; }}"
            f"QComboBox {{ background-color: {inner_bg}; border: 1px solid {inner_border}; color: {widget_color}; }}"
        )

    def update_info(self, text):
        if "<span" in text or "<div" in text:
            self.info_text.setHtml(text)
        else:
            self.info_text.setPlainText(text)

class TiledDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Initial theme detection
        app = QApplication.instance()
        pal = app.palette()
        bg_lightness = pal.color(QPalette.Window).lightness()
        text_lightness = pal.color(QPalette.WindowText).lightness()
        self.is_dark_theme = text_lightness > bg_lightness
        self.image_bg_color = "#111111" if self.is_dark_theme else "#FFFFFF"

        # Toolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(10, 5, 10, 5)
        self.btn_load = QPushButton("Load / Settings")
        self.btn_load.setToolTip("Load tiles and open tiled settings")
        self.btn_load.clicked.connect(self.open_settings)
        top_bar.addWidget(self.btn_load)
        # Recent history menu
        self.load_menu_btn = QToolButton()
        self.load_menu_btn.setArrowType(Qt.DownArrow)
        self.load_menu_btn.setMaximumWidth(22)
        self.load_menu_btn.setToolTip("Open recent tiled folders")
        self.load_menu_btn.clicked.connect(self._show_recent_menu)
        top_bar.addWidget(self.load_menu_btn)

        self.lbl_status = QLabel("No frames loaded")
        top_bar.addWidget(self.lbl_status)

        top_bar.addStretch()

        export_btn = QPushButton("Export Stitched")
        export_btn.setToolTip("Export current stitched image")
        export_btn.clicked.connect(self.export_current)
        top_bar.addWidget(export_btn)

        self.chk_stretch = QCheckBox("Auto Stretch")
        self.chk_stretch.stateChanged.connect(self.on_stretch_changed)
        top_bar.addWidget(self.chk_stretch)

        self.chk_tile_tab = QCheckBox("Show Tile View")
        self.chk_tile_tab.stateChanged.connect(self.toggle_tile_tab)
        top_bar.addWidget(self.chk_tile_tab)

        layout.addLayout(top_bar)

        # Navigation
        nav = QHBoxLayout()
        nav.setContentsMargins(10, 5, 10, 5)

        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Play or pause frame sequence")
        self.play_btn.clicked.connect(self.toggle_play)
        nav.addWidget(self.play_btn)

        nav.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.valueChanged.connect(self.update_timer_interval)
        nav.addWidget(self.fps_spin)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        nav.addWidget(self.frame_slider, 1)

        self.btn_prev = QPushButton("< Previous")
        self.btn_prev.setToolTip("Previous frame")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next = QPushButton("Next >")
        self.btn_next.setToolTip("Next frame")
        self.btn_next.clicked.connect(self.next_frame)
        self.lbl_frame = QLabel("Frame 0 / 0")

        zoom_in = QPushButton("Zoom +"); zoom_in.setToolTip("Zoom in"); zoom_in.clicked.connect(self.zoom_in)
        zoom_out = QPushButton("Zoom -"); zoom_out.setToolTip("Zoom out"); zoom_out.clicked.connect(self.zoom_out)
        zoom_fit = QPushButton("Fit"); zoom_fit.setToolTip("Fit image to viewport"); zoom_fit.clicked.connect(self.zoom_fit)
        zoom_100 = QPushButton("100%"); zoom_100.setToolTip("Reset zoom to 100%"); zoom_100.clicked.connect(self.zoom_100)

        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_frame)
        nav.addWidget(self.btn_next)
        nav.addStretch()
        nav.addWidget(zoom_in)
        nav.addWidget(zoom_out)
        nav.addWidget(zoom_fit)
        nav.addWidget(zoom_100)
        layout.addLayout(nav)

        # Tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget, 1)

        # Stitched Image Tab
        self.image_tab = QWidget()
        img_layout = QVBoxLayout(self.image_tab)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setStyleSheet(f"background-color: {self.image_bg_color};")
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_stitched_rotate_menu)
        self.image_scroll.setWidget(self.image_label)
        img_layout.addWidget(self.image_scroll)
        self.tab_widget.addTab(self.image_tab, "Stitched Image")

        # Tile View Tab
        self.tile_tab = QWidget()
        tile_layout = QVBoxLayout(self.tile_tab)
        self.tile_frame_tabs = QTabWidget()
        tile_layout.addWidget(self.tile_frame_tabs)
        self.tab_widget.addTab(self.tile_tab, "Tile View")
        self.tab_widget.setTabVisible(1, False)

        try:
            self.help_tab = create_help_tab(main_app=getattr(self, "_main_app", None), mode="tiled")
            self.tab_widget.addTab(self.help_tab, "Help")
        except Exception as e:
            print(f"Tiled help tab unavailable: {e}")

        # State
        self.original_tiles_per_frame = []  # Unrotated originals (memmap)
        self.tile_rotations = []  # Per frame per tile angles
        self.current_frame = 0
        self.current_scale = 0.0
        self.settings = None

        # Matrix size variable
        self.matrix_size_var = QSpinBox()
        self.matrix_size_var.setRange(1, 11)
        self.matrix_size_var.setValue(5)
        self.matrix_size_var.setMaximumWidth(50)

        self.tile_image_labels = []
        self.tile_nav_labels = []
        self.current_tile_per_frame = []

        # Grid params (cached for speed)
        self.tile_w_eff = 0
        self.tile_h_eff = 0

        # Events & Pixel Info
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.image_scroll.viewport().installEventFilter(self)

        self.pixel_info = PixelInfoWidget(None, matrix_size_var=self.matrix_size_var)
        viewport = self.image_scroll.viewport()
        self.pixel_info.setParent(viewport)
        self.pixel_info.show()
        self.position_pixel_info()
        viewport.installEventFilter(self)

        # Playback timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.auto_next)
        self.is_playing = False
        self.update_timer_interval()

        # Apply initial theme to everything
        self.update_image_backgrounds()

        # Keyboard shortcuts
        QShortcut(QKeySequence("Right"), self, self.next_frame)
        QShortcut(QKeySequence("Left"), self, self.prev_frame)
        QShortcut(QKeySequence("Ctrl+Space"), self, self.export_current)

    def _update_host_tab_name(self, folder_path):
        try:
            base = os.path.basename(folder_path.rstrip(os.sep)) if folder_path else ""
            if not base:
                return
            main_app = getattr(self, "_main_app", None)
            if main_app is not None and hasattr(main_app, "update_tab_name_for_widget"):
                main_app.update_tab_name_for_widget(self, base)
        except Exception:
            pass

    def changeEvent(self, event):
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self.update_image_backgrounds()
            self.pixel_info.update_theme()
        super().changeEvent(event)

    def update_image_backgrounds(self):
        app = QApplication.instance()
        if app:
            pal = app.palette()
            bg_lightness = pal.color(QPalette.Window).lightness()
            text_lightness = pal.color(QPalette.WindowText).lightness()
            is_dark = text_lightness > bg_lightness
        else:
            is_dark = True

        new_bg = "#111111" if is_dark else "#FFFFFF"
        if new_bg != getattr(self, 'image_bg_color', None):
            self.image_bg_color = new_bg
            self.image_label.setStyleSheet(f"background-color: {new_bg};")
            for label in self.tile_image_labels:
                label.setStyleSheet(f"background-color: {new_bg};")

    def position_pixel_info(self):
        viewport = self.image_scroll.viewport()
        margin = 10
        px_x = margin
        py_y = viewport.height() - self.pixel_info.height() - margin
        self.pixel_info.move(int(px_x), int(py_y))
        self.pixel_info.raise_()

    def update_timer_interval(self):
        fps = self.fps_spin.value()
        interval = max(1, 1000 // fps)
        self.timer.setInterval(interval)

    def toggle_play(self):
        if not self.original_tiles_per_frame or len(self.original_tiles_per_frame) <= 1:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.setText("Pause")
            self.timer.start()
        else:
            self.play_btn.setText("Play")
            self.timer.stop()

    def auto_next(self):
        if not self.original_tiles_per_frame:
            return
        self.current_frame = (self.current_frame + 1) % len(self.original_tiles_per_frame)
        self.frame_slider.setValue(self.current_frame)
        self.update_frame_label()
        self.show_frame()
        if self.chk_tile_tab.isChecked():
            self.tile_frame_tabs.setCurrentIndex(self.current_frame)
            self.display_tile(self.current_frame, self.current_tile_per_frame[self.current_frame])

    def on_stretch_changed(self):
        self.show_frame()
        if self.chk_tile_tab.isChecked():
            self.refresh_tile_views()

    def open_settings(self):
        # Prompt for a folder first (quick pick). If folder contains saved parameters, auto-apply and load.
        sel = QFileDialog.getExistingDirectory(self, "Select Session Folder")
        if sel:
            return self.open_folder(sel)

        # Fallback: open settings dialog (user wants to tweak existing settings)
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_values()
            self.matrix_size_var.setValue(self.settings['matrix_size'])
            try:
                save_params_for_path(self.settings['session'], self.settings, as_default=True)
            except Exception:
                pass
            self.load_frames()

    def open_folder(self, folder):
        self._update_host_tab_name(folder)
        # Apply folder defaults if present, otherwise prompt
        try:
            data = load_folder_params(folder)
        except Exception:
            data = None
        if data and data.get('default'):
            dflt = data.get('default', {})
            self.settings = {
                'session': folder,
                'folder_mode': dflt.get('folder_mode', True),
                'cols': dflt.get('cols', 6),
                'rows': dflt.get('rows', 4),
                'tile_w': dflt.get('tile_w', 1024),
                'tile_h': dflt.get('tile_h', 1024),
                'overlap': dflt.get('overlap', 0),
                'bit_depth': dflt.get('bit_depth', 8),
                'order': dflt.get('order', TileOrder.COLUMN_MAJOR),
                'final_w': dflt.get('final_w', 6480),
                'final_h': dflt.get('final_h', 4860),
                'matrix_size': dflt.get('matrix_size', 5)
            }
            self.matrix_size_var.setValue(self.settings['matrix_size'])
            self.load_frames()
            # Record recent
            try:
                add_recent(folder, 'tiled', self.settings)
            except Exception:
                pass
            return
        else:
            # No saved defaults - show settings dialog pre-filled with selected folder
            dialog = SettingsDialog(self, {'session': folder})
            if dialog.exec_() == QDialog.Accepted:
                self.settings = dialog.get_values()
                self.matrix_size_var.setValue(self.settings['matrix_size'])
                # Persist as folder default silently
                try:
                    save_params_for_path(self.settings['session'], self.settings, as_default=True)
                except Exception:
                    pass
                self.load_frames()
                try:
                    add_recent(folder, 'tiled', self.settings)
                except Exception:
                    pass
                return

        # Fallback: open settings dialog (user wants to tweak existing settings)
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec_() == QDialog.Accepted:
            self.settings = dialog.get_values()
            self.matrix_size_var.setValue(self.settings['matrix_size'])
            try:
                save_params_for_path(self.settings['session'], self.settings, as_default=True)
            except Exception:
                pass
            self.load_frames()
    def _show_recent_menu(self):
        try:
            menu = QMenu(self)
            recs = get_recents_for_mode('tiled', limit=7)
            if not recs:
                a = menu.addAction("No recent sessions")
                a.setEnabled(False)
            else:
                for r in recs:
                    ts = r.get('last_opened', '')
                    display = f"{os.path.basename(r.get('path',''))} — {ts[:19]}"
                    act = menu.addAction(display)
                    act.setToolTip(r.get('path'))
                    path = r.get('path')
                    act.triggered.connect(lambda checked, p=path: self.open_folder(p))
            all_recs = get_recents_for_mode('tiled')
            if len(all_recs) > 7:
                menu.addSeparator()
                vm = menu.addAction("View more...")
                vm.triggered.connect(lambda: self._open_full_history('tiled'))
            pos = self.load_menu_btn.mapToGlobal(self.load_menu_btn.rect().bottomLeft())
            menu.exec_(pos)
        except Exception:
            pass

    def _open_full_history(self, mode):
        try:
            sel = select_from_history(self, mode=mode)
            if sel:
                if mode == 'tiled':
                    self.open_folder(sel)
        except Exception:
            pass
    def load_frames(self):
        s = self.settings
        if not os.path.isdir(s['session']):
            QMessageBox.warning(self, "Error", "Invalid folder")
            return

        self.original_tiles_per_frame = []
        self.tile_rotations = []
        progress = QProgressDialog("Loading...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            if s['folder_mode']:
                subs = sorted([d for d in os.listdir(s['session']) if os.path.isdir(os.path.join(s['session'], d))])
                progress.setMaximum(len(subs))
                for i, sub in enumerate(subs):
                    progress.setValue(i)
                    if progress.wasCanceled(): return
                    path = os.path.join(s['session'], sub)
                    raws = self.get_raw_files(path)
                    tiles = self.read_tiles(raws, s)
                    if tiles and len(tiles) == s['cols'] * s['rows']:
                        self.original_tiles_per_frame.append(tiles)
                        self.tile_rotations.append([0] * len(tiles))
            else:
                raws = self.get_raw_files(s['session'])
                tpf = s['cols'] * s['rows']
                progress.setMaximum(len(raws) // tpf if tpf else 1)
                for i in range(0, len(raws), tpf):
                    if progress.wasCanceled(): return
                    group = raws[i:i+tpf]
                    tiles = self.read_tiles(group, s)
                    if tiles and len(tiles) == tpf:
                        self.original_tiles_per_frame.append(tiles)
                        self.tile_rotations.append([0] * len(tiles))

            if self.original_tiles_per_frame:
                self.current_frame = 0
                self.current_scale = 0.0
                self.tile_w_eff = self.settings['tile_w'] - self.settings['overlap']
                self.tile_h_eff = self.settings['tile_h'] - self.settings['overlap']
                self.frame_slider.setRange(0, len(self.original_tiles_per_frame)-1)
                self.lbl_status.setText(f"Loaded {len(self.original_tiles_per_frame)} frames")
                self.update_frame_label()
                self.show_frame()
                try:
                    main_app = getattr(self, '_main_app', None)
                    if main_app and getattr(main_app, 'iris', None):
                        w = self
                        tab_index = -1
                        while w:
                            tab_index = main_app.tab_widget.indexOf(w)
                            if tab_index != -1:
                                break
                            w = w.parent()
                        session_folder = self.settings.get('session', '')
                        frame_count = len(self.original_tiles_per_frame)
                        main_app.iris.notify_dataset_loaded(
                            tab_index, session_folder, frame_count, 0, {}, self)
                except Exception as e:
                    print(f"[Iris2] tiled notify error: {e}")

                self.play_btn.setEnabled(len(self.original_tiles_per_frame) > 1)
                if not self.play_btn.isEnabled() and self.is_playing:
                    self.toggle_play()

                if self.chk_tile_tab.isChecked():
                    self.setup_tile_tabs()
            else:
                QMessageBox.information(self, "Done", "No valid frames found")
        finally:
            progress.close()

    def get_raw_files(self, folder):
        return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.raw')])

    def read_tiles(self, paths, s):
        tiles = []
        for p in paths:
            tile = self.read_tile(p, s)
            if tile is None:
                return None
            tiles.append(tile)
        return tiles

    def read_tile(self, path, s):
        try:
            w, h, bpp = s['tile_w'], s['tile_h'], s['bit_depth'] // 8
            if bpp not in (1,2,4): raise ValueError("Unsupported bit depth")
            expected = w * h * bpp
            if os.path.getsize(path) != expected: raise ValueError("Size mismatch")
            dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[bpp]
            return np.memmap(path, dtype=dtype, mode='r', shape=(h, w))
        except Exception:
            return None

    def compose_stitched(self):
        if not self.original_tiles_per_frame: return np.zeros((1,1), dtype=np.uint8)
        tiles = self.original_tiles_per_frame[self.current_frame]
        rotations = self.tile_rotations[self.current_frame]
        s = self.settings
        canvas_h = self.tile_h_eff * s['rows']
        canvas_w = self.tile_w_eff * s['cols']
        canvas = np.zeros((canvas_h, canvas_w), dtype=tiles[0].dtype)
        for r in range(s['rows']):
            for c in range(s['cols']):
                idx = TileOrder.get_index(s['order'], r, c, s['rows'], s['cols'])
                tile = tiles[idx].copy()
                angle = rotations[idx]
                if angle == 90:
                    tile = np.rot90(tile, k=1)
                elif angle == 180:
                    tile = np.rot90(tile, k=2)
                elif angle == 270:
                    tile = np.rot90(tile, k=3)
                canvas[r*self.tile_h_eff:(r+1)*self.tile_h_eff, c*self.tile_w_eff:(c+1)*self.tile_w_eff] = tile[:self.tile_h_eff, :self.tile_w_eff]
        return canvas[:s['final_h'], :s['final_w']]

    def apply_stretch(self, img):
        if img.dtype != np.uint8:
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img8 = img
        return cv2.equalizeHist(img8)

    def show_frame(self):
        stitched = self.compose_stitched()
        disp = self.apply_stretch(stitched) if self.chk_stretch.isChecked() else (
            stitched if stitched.dtype == np.uint8 else cv2.normalize(stitched, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0], disp.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if self.current_scale <= 0:
            scaled = pix.scaled(self.image_scroll.viewport().size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        else:
            scaled = pix.scaled(int(disp.shape[1] * self.current_scale), int(disp.shape[0] * self.current_scale),
                                Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_label.setPixmap(scaled)

    def get_tile_index_from_pos(self, mx, my):
        s = self.settings
        if mx < 0 or my < 0 or mx >= s['final_w'] or my >= s['final_h']:
            return -1
        c = mx // self.tile_w_eff
        r = my // self.tile_h_eff
        if c >= s['cols'] or r >= s['rows']:
            return -1
        return TileOrder.get_index(s['order'], r, c, s['rows'], s['cols'])

    def show_stitched_rotate_menu(self, pos):
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull(): return
        pw, ph = pixmap.width(), pixmap.height()
        lw, lh = self.image_label.width(), self.image_label.height()
        offset_x = max(0, (lw - pw) // 2)
        offset_y = max(0, (lh - ph) // 2)
        sx = self.image_scroll.horizontalScrollBar().value()
        sy = self.image_scroll.verticalScrollBar().value()
        mx = pos.x() - offset_x + sx
        my = pos.y() - offset_y + sy
        if not (0 <= mx < pw and 0 <= my < ph): return
        scale_x = self.settings['final_w'] / pw
        scale_y = self.settings['final_h'] / ph
        stitched_x = int(round(mx * scale_x))
        stitched_y = int(round(my * scale_y))
        tile_idx = self.get_tile_index_from_pos(stitched_x, stitched_y)
        if tile_idx == -1: return

        menu = QMenu(self)
        act90cw = menu.addAction("Rotate Tile 90° Clockwise")
        act90ccw = menu.addAction("Rotate Tile 90° Counter-Clockwise")
        act180 = menu.addAction("Rotate Tile 180°")
        act_reset = menu.addAction("Reset Tile Rotation")
        action = menu.exec_(self.image_label.mapToGlobal(pos))
        delta = 0
        if action == act90cw:
            delta = 90
        elif action == act90ccw:
            delta = -90
        elif action == act180:
            delta = 180
        elif action == act_reset:
            delta = -self.tile_rotations[self.current_frame][tile_idx]
        if delta != 0:
            self.tile_rotations[self.current_frame][tile_idx] = (self.tile_rotations[self.current_frame][tile_idx] + delta) % 360
            self.show_frame()
            if self.chk_tile_tab.isChecked():
                self.refresh_tile_views()

    def zoom_in(self): self.current_scale = max(0.1, (self.current_scale or 1.0) * 1.25); self.show_frame()
    def zoom_out(self): self.current_scale = max(0.1, (self.current_scale or 1.0) * 0.8); self.show_frame()
    def zoom_fit(self): self.current_scale = 0.0; self.show_frame()
    def zoom_100(self): self.current_scale = 1.0; self.show_frame()

    def prev_frame(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.setValue(self.current_frame)
            self.update_frame_label()
            self.show_frame()
            if self.chk_tile_tab.isChecked():
                self.tile_frame_tabs.setCurrentIndex(self.current_frame)
                self.display_tile(self.current_frame, self.current_tile_per_frame[self.current_frame])

    def next_frame(self):
        if self.current_frame < len(self.original_tiles_per_frame)-1:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.update_frame_label()
            self.show_frame()
            if self.chk_tile_tab.isChecked():
                self.tile_frame_tabs.setCurrentIndex(self.current_frame)
                self.display_tile(self.current_frame, self.current_tile_per_frame[self.current_frame])

    def slider_changed(self, v):
        self.current_frame = v
        self.update_frame_label()
        self.show_frame()
        if self.chk_tile_tab.isChecked():
            self.tile_frame_tabs.setCurrentIndex(self.current_frame)
            self.display_tile(self.current_frame, self.current_tile_per_frame[self.current_frame])
        try:
            main_app = getattr(self, '_main_app', None)
            if main_app and getattr(main_app, 'iris', None):
                w = self
                tab_index = -1
                while w:
                    tab_index = main_app.tab_widget.indexOf(w)
                    if tab_index != -1:
                        break
                    w = w.parent()
                main_app.iris.notify_frame_changed(tab_index, v)
        except Exception:
            pass

    def update_frame_label(self):
        self.lbl_frame.setText(f"Frame {self.current_frame + 1} / {len(self.original_tiles_per_frame)}")

    def export_current(self):
        if not self.original_tiles_per_frame: return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "", "PNG (*.png)")
        if path:
            stitched = self.compose_stitched()
            out = self.apply_stretch(stitched) if self.chk_stretch.isChecked() else stitched
            cv2.imwrite(path, out if out.dtype == np.uint8 else cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    def toggle_tile_tab(self, state):
        self.tab_widget.setTabVisible(1, state == Qt.Checked)
        if state and self.original_tiles_per_frame:
            self.setup_tile_tabs()

    def refresh_tile_views(self):
        if not self.chk_tile_tab.isChecked(): return
        for fi in range(len(self.original_tiles_per_frame)):
            self.display_tile(fi, self.current_tile_per_frame[fi])

    def setup_tile_tabs(self):
        self.tile_frame_tabs.clear()
        self.tile_image_labels = []
        self.tile_nav_labels = []
        self.current_tile_per_frame = [0] * len(self.original_tiles_per_frame)
        for fi in range(len(self.original_tiles_per_frame)):
            tab = QWidget()
            lay = QVBoxLayout(tab)
            nav = QHBoxLayout()
            prev = QPushButton("< Prev Tile")
            prev.setToolTip("Previous tile in this frame")
            prev.clicked.connect(lambda _, fii=fi: self.change_tile(fii, -1))
            next = QPushButton("Next Tile >")
            next.setToolTip("Next tile in this frame")
            next.clicked.connect(lambda _, fii=fi: self.change_tile(fii, 1))
            lbl = QLabel(f"Tile 1 / {len(self.original_tiles_per_frame[fi])}")
            nav.addWidget(prev); nav.addWidget(lbl); nav.addWidget(next)
            lay.addLayout(nav)
            self.tile_nav_labels.append(lbl)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel(alignment=Qt.AlignCenter)
            label.setStyleSheet(f"background-color: {self.image_bg_color};")
            label.setMouseTracking(True)
            label.installEventFilter(self)
            scroll.setWidget(label)
            lay.addWidget(scroll)
            self.tile_image_labels.append(label)

            self.tile_frame_tabs.addTab(tab, f"Frame {fi+1}")

        self.tile_frame_tabs.setCurrentIndex(self.current_frame)
        self.display_tile(self.current_frame, 0)

        # Ensure newly created tile labels use current theme
        self.update_image_backgrounds()

    def display_tile(self, frame_idx, tile_idx):
        tile = self.original_tiles_per_frame[frame_idx][tile_idx].copy()
        angle = self.tile_rotations[frame_idx][tile_idx]
        if angle == 90:
            tile = np.rot90(tile, k=1)
        elif angle == 180:
            tile = np.rot90(tile, k=2)
        elif angle == 270:
            tile = np.rot90(tile, k=3)
        disp = self.apply_stretch(tile) if self.chk_stretch.isChecked() else (
            tile if tile.dtype == np.uint8 else cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0], disp.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.tile_image_labels[frame_idx].setPixmap(pix.scaled(self.tile_image_labels[frame_idx].size(), Qt.KeepAspectRatio, Qt.FastTransformation))
        self.tile_nav_labels[frame_idx].setText(f"Tile {tile_idx+1} (Rot: {angle}°) / {len(self.original_tiles_per_frame[frame_idx])}")

    def change_tile(self, frame_idx, delta):
        new = self.current_tile_per_frame[frame_idx] + delta
        if 0 <= new < len(self.original_tiles_per_frame[frame_idx]):
            self.current_tile_per_frame[frame_idx] = new
            self.display_tile(frame_idx, new)

    def eventFilter(self, obj, event):
        if obj == self.image_scroll.viewport() and event.type() == QEvent.Resize:
            self.position_pixel_info()

        stitched_img = pixmap = label_widget = scroll_area = None
        is_stitched = False

        if obj == self.image_label:
            stitched_img = self.compose_stitched()
            pixmap = self.image_label.pixmap()
            label_widget = self.image_label
            scroll_area = self.image_scroll
            is_stitched = True
        else:
            for i, lbl in enumerate(self.tile_image_labels):
                if obj == lbl:
                    tile = self.original_tiles_per_frame[i][self.current_tile_per_frame[i]].copy()
                    angle = self.tile_rotations[i][self.current_tile_per_frame[i]]
                    if angle == 90:
                        tile = np.rot90(tile, k=1)
                    elif angle == 180:
                        tile = np.rot90(tile, k=2)
                    elif angle == 270:
                        tile = np.rot90(tile, k=3)
                    stitched_img = tile
                    pixmap = lbl.pixmap()
                    label_widget = lbl
                    scroll_area = lbl.parent().parent()
                    break

        if stitched_img is None or pixmap is None or pixmap.isNull():
            return super().eventFilter(obj, event)

        if event.type() == QEvent.MouseMove:
            pos = event.pos()
            pw, ph = pixmap.width(), pixmap.height()
            lw, lh = label_widget.width(), label_widget.height()
            offset_x = max(0, (lw - pw) // 2)
            offset_y = max(0, (lh - ph) // 2)
            sx = scroll_area.horizontalScrollBar().value() if scroll_area else 0
            sy = scroll_area.verticalScrollBar().value() if scroll_area else 0
            mx = pos.x() - offset_x + sx
            my = pos.y() - offset_y + sy

            if 0 <= mx < pw and 0 <= my < ph:
                scale_x = stitched_img.shape[1] / pw
                scale_y = stitched_img.shape[0] / ph
                x = int(round(mx * scale_x))
                y = int(round(my * scale_y))

                if is_stitched:
                    tile_idx = self.get_tile_index_from_pos(x, y)
                    if tile_idx == -1:
                        return True
                    orig_tile = self.original_tiles_per_frame[self.current_frame][tile_idx]
                    angle = self.tile_rotations[self.current_frame][tile_idx]
                    local_x = x % self.tile_w_eff
                    local_y = y % self.tile_h_eff
                    if angle == 90:
                        orig_x, orig_y = orig_tile.shape[1] - 1 - local_y, local_x
                    elif angle == 180:
                        orig_x, orig_y = orig_tile.shape[1] - 1 - local_x, orig_tile.shape[0] - 1 - local_y
                    elif angle == 270:
                        orig_x, orig_y = local_y, orig_tile.shape[0] - 1 - local_x
                    else:
                        orig_x, orig_y = local_x, local_y
                    val = int(orig_tile[orig_y, orig_x])

                    half = self.matrix_size_var.value() // 2
                    orig_ys = max(0, orig_y - half)
                    orig_ye = min(orig_tile.shape[0], orig_y + half + 1)
                    orig_xs = max(0, orig_x - half)
                    orig_xe = min(orig_tile.shape[1], orig_x + half + 1)
                    mat = orig_tile[orig_ys:orig_ye, orig_xs:orig_xe]
                    center_row_idx = orig_y - orig_ys
                    center_col_idx = orig_x - orig_xs
                else:
                    return True

                lines = [f"Coord: ({x}, {y}) Value: {val}"]
                lines.append("")
                size = self.matrix_size_var.value()
                lines.append(f"{size}x{size} Pixel Matrix:")
                lines.append("")

                # Dynamic theme detection for matrix highlight
                app = QApplication.instance()
                if app is not None:
                    pal = app.palette()
                    bg_l = pal.color(QPalette.Window).lightness()
                    text_l = pal.color(QPalette.WindowText).lightness()
                    is_dark = text_l > bg_l
                else:
                    is_dark = self.is_dark_theme

                if is_dark:
                    highlight_bg = "#3399FF"  # Bright blue for dark mode
                    highlight_text = "white"
                else:
                    highlight_bg = "yellow"
                    highlight_text = "black"

                html_rows = []
                for r, row in enumerate(mat):
                    cells = []
                    for c, v in enumerate(row):
                        val_str = f"{int(v):3}"
                        if r == center_row_idx and c == center_col_idx:
                            cells.append(f"<span style='background-color:{highlight_bg}; color:{highlight_text}; padding: 2px;'>{val_str}</span>")
                        else:
                            cells.append(val_str)
                    row_str = " ".join(cells)
                    html_rows.append(f"<div style='font-family:monospace; font-size:11px; white-space:pre'>{row_str}</div>")
                matrix_html = "".join(html_rows)
                lines.append(matrix_html)

                self.pixel_info.update_info("\n".join(lines))
                return True

        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TiledDisplay()
    win.show()
    sys.exit(app.exec_())
