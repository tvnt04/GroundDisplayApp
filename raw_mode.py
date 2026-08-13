import os
import numpy as np
import math
from PIL import Image
import gc
import psutil
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QLabel, QProgressBar,
    QFileDialog, QMessageBox, QLineEdit, QFormLayout, QDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QToolButton, QMenu, QApplication, QShortcut, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QKeySequence
from image_viewer import GraphicsImageViewer
from ui_components import HistogramViewer, PixelInfoBox
from utils import (
    unpack_by_bitdepth, check_memory_requirement, LazyFrames, get_saved_params_for_file, save_params_for_path,
    add_recent, get_recents_for_mode, select_from_history
)
from help_tab import create_help_tab
try:
    from editor_tab import EditorTab
except ImportError:
    EditorTab = None

class UnloadThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, raw_viewer, parent=None):
        super().__init__(parent)
        self.raw_viewer = raw_viewer

    def run(self):
        try:
            self.progress.emit(20)
            if hasattr(self.raw_viewer, 'raw_data'):
                self.raw_viewer.raw_data = None
            if hasattr(self.raw_viewer, 'normalized_data'):
                self.raw_viewer.normalized_data = None
            if hasattr(self.raw_viewer, 'lazy_frames'):
                self.raw_viewer.lazy_frames = None

            self.progress.emit(50)
            gc.collect()
            
            self.progress.emit(80)
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass
            
            self.progress.emit(100)
        except Exception:
            pass
        finally:
            self.finished.emit()

class StackBuildThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # PIL Image
    error = pyqtSignal(str)

    def __init__(self, lazy_frames, parent=None, max_pixels=50_000_000, start_frame=0, end_frame=None):
        super().__init__(parent)
        self.lazy_frames = lazy_frames
        self.max_pixels = int(max_pixels)
        self.start_frame = int(start_frame)
        self.end_frame = (None if end_frame is None else int(end_frame))

    def run(self):
        try:
            total_frames = len(self.lazy_frames)
            # Normalize requested range
            start = max(0, min(self.start_frame, total_frames - 1))
            end = total_frames - 1 if self.end_frame is None else max(0, min(self.end_frame, total_frames - 1))
            if end < start:
                self.error.emit("Invalid stacking range")
                return
            num_frames = end - start + 1
            w = self.lazy_frames.w
            h = self.lazy_frames.h
            # Determine a reasonable target width to keep memory usage bounded
            target_w = min(w, 2048)
            scale = target_w / float(w)
            th = max(1, int(round(h * scale)))
            per_frame_pixels = target_w * th
            if per_frame_pixels <= 0:
                self.error.emit("Invalid frame dimensions for stacking")
                return
            max_frames_allowed = max(1, self.max_pixels // per_frame_pixels)
            if max_frames_allowed >= num_frames:
                step = 1
            else:
                step = int(math.ceil(num_frames / float(max_frames_allowed)))
            sampled_indices = [start + i for i in range(0, num_frames, step)]
            total = len(sampled_indices)
            # Create a new PIL image with L mode (vertical stack)
            stack_img = Image.new('L', (target_w, th * total))
            for i, idx in enumerate(sampled_indices):
                raw = self.lazy_frames.get_raw(idx)
                # Convert raw to uint8 for display
                if self.lazy_frames.bitdepth > 8:
                    max_val = (1 << self.lazy_frames.bitdepth) - 1
                    arr = ((raw.astype(np.float64) / max_val) * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = raw.astype(np.uint8)
                pil = Image.fromarray(arr)
                if pil.width != target_w or pil.height != th:
                    pil = pil.resize((target_w, th), resample=Image.BILINEAR)
                stack_img.paste(pil, (0, i * th))
                self.progress.emit(int(100.0 * (i + 1) / total))
            self.finished.emit(stack_img)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))

class RawParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Raw Image Parameters")
        layout = QFormLayout()
        self.setLayout(layout)
        self.width_entry = QLineEdit("8448")
        layout.addRow("Width:", self.width_entry)
        self.height_entry = QLineEdit("384")
        layout.addRow("Height:", self.height_entry)
        self.bitdepth_var = QComboBox()
        self.bitdepth_var.addItems(["8", "10", "12", "16", "32"])
        self.bitdepth_var.setCurrentIndex(1)  # Default to 10-bit
        layout.addRow("Bit Depth:", self.bitdepth_var)
        buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setToolTip("Apply parameters")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Close without changes")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addRow(buttons)

    def get_parameters(self):
        return {
            "width": int(self.width_entry.text()),
            "height": int(self.height_entry.text()),
            "bitdepth": int(self.bitdepth_var.currentText())
        }

class RawLoadingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)  # Original raw frame (uint16 for >8 bit)
    normalized = pyqtSignal(np.ndarray)  # 8-bit normalized for display
    error = pyqtSignal(str)

    def __init__(self, file_path, width, height, bitdepth, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.width = width
        self.height = height
        self.bitdepth = bitdepth

    def run(self):
        try:
            self.progress.emit(0)
            if not os.path.exists(self.file_path):
                self.error.emit("File not found")
                return
            with open(self.file_path, 'rb') as f:
                data = f.read()
            if len(data) == 0:
                self.error.emit("File is empty")
                return
            self.progress.emit(50)
            frames = unpack_by_bitdepth(data, self.width, self.height, self.bitdepth, return_raw=True)
            if frames is None:
                self.error.emit(f"Unpack failed for {self.bitdepth}-bit")
                return
            if isinstance(frames, np.ndarray):
                if frames.size == 0:
                    self.error.emit(f"Unpack failed for {self.bitdepth}-bit")
                    return
                raw_frame = frames if frames.ndim == 2 else frames[0]
            else:
                if len(frames) == 0:
                    self.error.emit(f"Unpack failed for {self.bitdepth}-bit")
                    return
                raw_frame = frames[0]
            if raw_frame is None or raw_frame.size == 0:
                self.error.emit(f"Unpack failed for {self.bitdepth}-bit")
                return
            if raw_frame.shape != (self.height, self.width):
                self.error.emit("Frame shape mismatch")
                return
            max_val = (1 << self.bitdepth) - 1 if self.bitdepth > 8 else 255
            display_frame = np.clip(
                raw_frame.astype(np.float64) * (255.0 / float(max_val)),
                0, 255
            ).astype(np.uint8)
            self.progress.emit(100)
            self.finished.emit(raw_frame)
            self.normalized.emit(display_frame)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Error: {str(e)}")

class RawViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw_data = None
        self.normalized_data = None
        self.last_params = {}
        self.bitdepth = 8
        self.last_file_path = None
        self.contrast_enhance = False
        self.contrast_min = 0
        self.contrast_max = 255
        self.current_pil_image = None  # For editor tab compatibility

        self.playing = False
        self.play_delay = 100
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_next_frame)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Minimize margins
        layout.setSpacing(3)
        self.setLayout(layout)

        # Hidden matrix control (used by pixel-info overlays; removed from top toolbar UI)
        self.matrix_size_var = QSpinBox()
        self.matrix_size_var.setRange(1, 11)
        self.matrix_size_var.setValue(5)
        self.matrix_size_var.setMaximumWidth(50)
        self.matrix_size_var.setVisible(False)

        # ===== GROUPED TOOLBAR =====
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(8)

        file_group = QGroupBox("")
        file_group_layout = QHBoxLayout()
        file_group_layout.setContentsMargins(6, 6, 6, 6)
        file_group_layout.setSpacing(6)

        self.load_btn = QPushButton("Load")
        self.load_btn.setMaximumWidth(60)
        self.load_btn.setToolTip("Load a .raw file")
        self.load_btn.clicked.connect(self.load_raw_file)
        file_group_layout.addWidget(self.load_btn)

        # Recent history menu (down arrow)
        self.load_menu_btn = QToolButton()
        self.load_menu_btn.setArrowType(Qt.DownArrow)
        self.load_menu_btn.setMaximumWidth(22)
        self.load_menu_btn.setToolTip("Open recent raw files")
        self.load_menu_btn.clicked.connect(self._show_recent_menu)
        file_group_layout.addWidget(self.load_menu_btn)

        self.params_btn = QPushButton("Params")
        self.params_btn.setMaximumWidth(60)
        self.params_btn.setToolTip("Edit image parameters")
        self.params_btn.clicked.connect(self.edit_params)
        file_group_layout.addWidget(self.params_btn)
        file_group.setLayout(file_group_layout)
        control_layout.addWidget(file_group)

        frame_group = QGroupBox("")
        frame_group_layout = QHBoxLayout()
        frame_group_layout.setContentsMargins(6, 6, 6, 6)
        frame_group_layout.setSpacing(6)
        frame_group_layout.addWidget(QLabel("Frame:"))
        self.prev_frame_btn = QPushButton("<")
        self.prev_frame_btn.setMaximumWidth(30)
        self.prev_frame_btn.setToolTip("Previous frame")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)
        frame_group_layout.addWidget(self.prev_frame_btn)

        self.frame_index_spin = QSpinBox()
        self.frame_index_spin.setMinimum(1)
        self.frame_index_spin.setMaximum(1)
        self.frame_index_spin.setValue(1)
        self.frame_index_spin.setMaximumWidth(80)
        self.frame_index_spin.setToolTip("Current frame number")
        self.frame_index_spin.setKeyboardTracking(False)
        self.frame_index_spin.valueChanged.connect(self.goto_frame)
        self.frame_index_spin.setEnabled(False)  # Disabled until a file is loaded
        frame_group_layout.addWidget(self.frame_index_spin)

        self.next_frame_btn = QPushButton(">")
        self.next_frame_btn.setMaximumWidth(30)
        self.next_frame_btn.setToolTip("Next frame")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        frame_group_layout.addWidget(self.next_frame_btn)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setToolTip("Play or pause frames")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        frame_group_layout.addWidget(self.play_btn)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x", "4.0x", "10.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setToolTip("Playback Speed")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        self.speed_combo.setEnabled(False)
        frame_group_layout.addWidget(self.speed_combo)

        frame_group.setLayout(frame_group_layout)
        control_layout.addWidget(frame_group)

        contrast_group = QGroupBox("")
        contrast_group_layout = QHBoxLayout()
        contrast_group_layout.setContentsMargins(6, 6, 6, 6)
        contrast_group_layout.setSpacing(6)
        self.enhance_cb = QCheckBox("Auto Contrast")
        self.enhance_cb.setChecked(False)
        self.enhance_cb.setToolTip("Apply Min/Max contrast")
        self.enhance_cb.stateChanged.connect(self.update_display)
        contrast_group_layout.addWidget(self.enhance_cb)
        # Min/Max controls
        contrast_group_layout.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(0)
        self.min_spin.setRange(0, 65535)
        self.min_spin.setMaximumWidth(105)
        self.min_spin.setValue(0)
        self.min_spin.setToolTip("Minimum value")
        self.min_spin.valueChanged.connect(self.update_display)
        contrast_group_layout.addWidget(self.min_spin)

        contrast_group_layout.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(0)
        self.max_spin.setRange(0, 65535)
        self.max_spin.setMaximumWidth(105)
        self.max_spin.setValue(255)
        self.max_spin.setToolTip("Maximum value")
        self.max_spin.valueChanged.connect(self.update_display)
        contrast_group_layout.addWidget(self.max_spin)
        contrast_group.setLayout(contrast_group_layout)
        control_layout.addWidget(contrast_group)

        range_group = QGroupBox("")
        range_group_layout = QHBoxLayout()
        range_group_layout.setContentsMargins(6, 6, 6, 6)
        range_group_layout.setSpacing(6)
        # Stack button
        self.stack_btn = QPushButton("Create Stack")
        self.stack_btn.setMaximumWidth(100)
        self.stack_btn.setToolTip("Create stacked preview")
        self.stack_btn.clicked.connect(self.create_stack_view)
        self.stack_btn.setEnabled(False)
        range_group_layout.addWidget(self.stack_btn)

         # Range selection (From / To)
        range_group_layout.addWidget(QLabel("From:"))
        self.range_start_spin = QSpinBox()
        self.range_start_spin.setMinimum(1)
        self.range_start_spin.setMaximum(1)
        self.range_start_spin.setValue(1)
        self.range_start_spin.setMaximumWidth(80)
        # Allow immediate feedback from arrows and typed edits; also ensure edits commit on Enter/FocusOut
        self.range_start_spin.setKeyboardTracking(True)
        self.range_start_spin.setFocusPolicy(Qt.StrongFocus)
        self.range_start_spin.setEnabled(True)  # Always enabled — user wants control
        self.range_start_spin.setToolTip("Start frame")
        # Support both spinning and typed edits reliably
        self.range_start_spin.valueChanged.connect(self._on_range_spin_changed)
        self.range_start_spin.editingFinished.connect(self._on_range_spin_changed)
        range_group_layout.addWidget(self.range_start_spin)

        range_group_layout.addWidget(QLabel("To:"))
        self.range_end_spin = QSpinBox()
        self.range_end_spin.setMinimum(1)
        self.range_end_spin.setMaximum(1)
        self.range_end_spin.setValue(1)
        self.range_end_spin.setMaximumWidth(80)
        # Allow immediate feedback from arrows and typed edits; also ensure edits commit on Enter/FocusOut
        self.range_end_spin.setKeyboardTracking(True)
        self.range_end_spin.setFocusPolicy(Qt.StrongFocus)
        self.range_end_spin.setEnabled(True)  # Always enabled
        self.range_end_spin.setToolTip("End frame")
        # Support both spinning and typed edits reliably
        self.range_end_spin.valueChanged.connect(self._on_range_spin_changed)
        self.range_end_spin.editingFinished.connect(self._on_range_spin_changed)
        range_group_layout.addWidget(self.range_end_spin)

        range_group.setLayout(range_group_layout)
        control_layout.addWidget(range_group)

        export_group = QGroupBox("")
        export_group_layout = QHBoxLayout()
        export_group_layout.setContentsMargins(6, 6, 6, 6)
        export_group_layout.setSpacing(6)
        self.export_btn = QPushButton("Export")
        self.export_btn.setMaximumWidth(70)
        self.export_btn.setToolTip("Export current tab view")
        self.export_btn.clicked.connect(self.export_current)
        self.export_btn.setEnabled(False)
        export_group_layout.addWidget(self.export_btn)
        export_group.setLayout(export_group_layout)
        control_layout.addWidget(export_group)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(12)
        layout.addWidget(self.progress_bar)

        # ===== TABBED DISPLAY =====
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Frames viewer tab (for stacked view)
        self.frames_display = GraphicsImageViewer(
            self,
            pixel_info_callback=self.on_pixel_info,
            matrix_size_var=self.matrix_size_var
        )
        frames_tab = QWidget()
        frames_layout = QVBoxLayout()
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.addWidget(self.frames_display)
        frames_tab.setLayout(frames_layout)
        self.tabs.addTab(frames_tab, "Frames")

        # Lazy frame attributes
        self.lazy_frames = None
        self.num_frames = 1
        self.current_frame_index = 0
        self.stack_thread = None

        # Main Display
        self.main_display = GraphicsImageViewer(
            self,
            pixel_info_callback=self.on_pixel_info,
            matrix_size_var=self.matrix_size_var
        )
        # RAW viewer: primary display should be treated as a single band by default
        try:
            self.main_display.is_individual = True
            self.main_display.is_frame_stack = False
            # ensure frames_display keeps its stacked semantics
            self.frames_display.is_individual = False
        except Exception:
            pass
        main_tab = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.main_display)
        main_tab.setLayout(main_layout)
        self.tabs.addTab(main_tab, "Display")

        # Histogram tab
        self.hist_viewer = HistogramViewer(self)
        try:
            self.hist_viewer.mode_changed.connect(self.update_histogram)
        except Exception:
            pass
        hist_tab = QWidget()
        hist_layout = QVBoxLayout()
        hist_layout.setContentsMargins(0, 0, 0, 0)
        hist_layout.addWidget(self.hist_viewer)
        hist_tab.setLayout(hist_layout)
        self.tabs.addTab(hist_tab, "Histogram")

        try:
            self.help_tab = create_help_tab(main_app=getattr(self, "_main_app", None), mode="raw")
            self.tabs.addTab(self.help_tab, "Help")
        except Exception as e:
            print(f"Raw help tab unavailable: {e}")

        # Ensure tab order is Display, Frames, Histogram (robust to saved tab indices)
        try:
            desired = ["Display", "Frames", "Histogram", "Help"]
            current = {self.tabs.tabText(i): self.tabs.widget(i) for i in range(self.tabs.count())}
            # remove all existing tabs
            for i in reversed(range(self.tabs.count())):
                self.tabs.removeTab(i)
            # re-add in desired order if available
            for name in desired:
                w = current.get(name)
                if w is not None:
                    self.tabs.addTab(w, name)
        except Exception:
            pass

        # Pixel info overlay
        try:
            self.main_display.pixel_info_box_overlay = PixelInfoBox(self.main_display, matrix_size_var=self.matrix_size_var)
            self.main_display.pixel_info_box_overlay.setFixedWidth(280)
            self.main_display.pixel_info_box_overlay.setFixedHeight(200)
            self.main_display.pixel_info_box_overlay.setStyleSheet(
                "QWidget { background-color: rgba(0, 0, 0, 0.85); border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 4px; color: white; }"
                "QTextEdit { background-color: rgba(0, 0, 0, 0.8); border: none; color: white; }"
            )
            try:
                if self.main_display.pixel_info_box_overlay.layout().count() > 0:
                    title_item = self.main_display.pixel_info_box_overlay.layout().itemAt(0)
                    if title_item and title_item.widget():
                        title_item.widget().hide()
            except Exception:
                pass
            self.main_display.pixel_info_box_overlay.show()
            self.main_display.pixel_info_box_overlay.raise_()
        except Exception as e:
            print(f"Warning: Could not create pixel info box overlay: {e}")
            self.main_display.pixel_info_box_overlay = None

        # Keyboard shortcuts
        QShortcut(QKeySequence("Right"), self, self.next_frame)
        QShortcut(QKeySequence("Left"), self, self.prev_frame)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_state)
        QShortcut(QKeySequence("Ctrl+Space"), self, self.export_current)

    def closeEvent(self, event):
        # Stop any running threads
        try:
            if hasattr(self, 'loading_thread') and self.loading_thread and self.loading_thread.isRunning():
                self.loading_thread.requestInterruption()
                self.loading_thread.quit()
                self.loading_thread.wait(3000)
        except Exception as e:
            print(f"[DEBUG] Error stopping loading_thread: {e}")
        
        try:
            if hasattr(self, 'stack_thread') and self.stack_thread and self.stack_thread.isRunning():
                self.stack_thread.requestInterruption()
                self.stack_thread.quit()
                self.stack_thread.wait(3000)
        except Exception as e:
            print(f"[DEBUG] Error stopping stack_thread: {e}")
        
        super().closeEvent(event)

    def load_raw_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select .raw File", "", "Raw Files (*.raw)")
        if not file_path:
            return

        # If existing data present, unload it first then proceed with load
        if getattr(self, 'raw_data', None) is not None or getattr(self, 'lazy_frames', None) is not None or (getattr(self, 'loading_thread', None) and getattr(self.loading_thread, 'isRunning', lambda: False)()):
            self._start_unload_and_then(lambda: self._do_load_raw_file(file_path))
        else:
            self._do_load_raw_file(file_path)

    def _update_host_tab_name(self, file_path):
        try:
            base = os.path.basename(file_path) if file_path else ""
            if not base:
                return
            main_app = getattr(self, "_main_app", None)
            if main_app is not None and hasattr(main_app, "update_tab_name_for_widget"):
                main_app.update_tab_name_for_widget(self, base)
        except Exception:
            pass

    def _do_load_raw_file(self, file_path, forced_params=None):
        # Actual load logic (extracted for chaining after unload)
        self.last_file_path = file_path
        self._update_host_tab_name(file_path)

        params = None
        est_bytes = None
        if forced_params is not None:
            try:
                params = {
                    "width": int(forced_params.get("width")),
                    "height": int(forced_params.get("height")),
                    "bitdepth": int(forced_params.get("bitdepth")),
                }
                est_bytes = params["width"] * params["height"] * max(1, params["bitdepth"] // 8)
            except Exception:
                QMessageBox.critical(self, "Parameters", "Invalid parameters provided for reload.")
                return
        else:
            # Try to auto-apply saved parameters for this file (silent)
            try:
                saved = get_saved_params_for_file(file_path)
            except Exception:
                saved = None
            use_saved = False
            if saved:
                try:
                    w = int(saved.get('width', saved.get('tile_w', 0)))
                    h = int(saved.get('height', saved.get('tile_h', 0)))
                    bd = int(saved.get('bitdepth', saved.get('bit_depth', saved.get('bitdepth', 8))))
                    params = {'width': w, 'height': h, 'bitdepth': bd}
                    est_bytes = params["width"] * params["height"] * max(1, params["bitdepth"] // 8)
                    if check_memory_requirement(est_bytes, self):
                        use_saved = True
                except Exception:
                    use_saved = False

            if not use_saved:
                dialog = RawParameterDialog(self)
                if dialog.exec_() != QDialog.Accepted:
                    return
                params = dialog.get_parameters()
                est_bytes = params["width"] * params["height"] * max(1, params["bitdepth"] // 8)

        self.last_params = params.copy()
        if not check_memory_requirement(est_bytes, self):
            QMessageBox.warning(self, "Memory", "File may exceed available RAM")
            return

        # Try LazyFrames first (for all files, single or multi-frame)
        try:
            lf = LazyFrames(file_path, params['width'], params['height'], params['bitdepth'])
            num_frames = len(lf)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to initialize LazyFrames: {e}")
            return

        if num_frames == 0:
            QMessageBox.critical(self, "Load Error", "File is empty or contains no frames.")
            return

        # Multi-frame (or single-frame) success
        self.lazy_frames = lf
        self.num_frames = num_frames
        self.current_frame_index = 0

        # Frame navigation controls
        self.frame_index_spin.setRange(1, self.num_frames)
        self.frame_index_spin.setValue(1)
        self.frame_index_spin.setEnabled(True)
        self.prev_frame_btn.setEnabled(self.num_frames > 1)
        self.next_frame_btn.setEnabled(self.num_frames > 1)
        self.play_btn.setEnabled(self.num_frames > 1)
        self.speed_combo.setEnabled(self.num_frames > 1)
        self.stack_btn.setEnabled(self.num_frames > 1)
        self.export_btn.setEnabled(True)

        # Range controls (always enabled, default full range)
        self.range_start_spin.blockSignals(True)
        self.range_end_spin.blockSignals(True)
        self.range_start_spin.setRange(1, self.num_frames)
        self.range_end_spin.setRange(1, self.num_frames)
        self.range_start_spin.setValue(1)
        self.range_end_spin.setValue(self.num_frames)
        self.range_start_spin.setEnabled(True)
        self.range_end_spin.setEnabled(True)
        self.range_start_spin.blockSignals(False)
        self.range_end_spin.blockSignals(False)

        # Load and display first frame
        try:
            raw = lf.get_raw(0)
            self.bitdepth = params['bitdepth']
            self.raw_data = raw
            if self.bitdepth > 8:
                max_val = self._current_max_dn()
                self.normalized_data = ((raw.astype(np.float64) / max_val) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                self.normalized_data = raw.astype(np.uint8)
            self.update_contrast_controls()
            self.update_display()
            self.update_histogram()
            # Clear any status overlay
            try:
                self.main_display.clear_status()
            except Exception:
                pass
            # Persist params silently for future auto-loads
            try:
                save_params_for_path(self.last_file_path, self.last_params)
            except Exception:
                pass
            # Add to recent history
            try:
                add_recent(self.last_file_path, 'raw', self.last_params)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to read first frame: {e}")
            # Fallback to single-frame-like UI state
            self.lazy_frames = None
            self.num_frames = 1
            self.current_frame_index = 0
            self.frame_index_spin.setRange(1, 1)
            self.frame_index_spin.setValue(1)
            self.frame_index_spin.setEnabled(True)
            self.prev_frame_btn.setEnabled(False)
            self.next_frame_btn.setEnabled(False)
            self.play_btn.setEnabled(False)
            self.speed_combo.setEnabled(False)
            self.stack_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

            # Range controls for fallback single-frame state
            self.range_start_spin.blockSignals(True)
            self.range_end_spin.blockSignals(True)
            self.range_start_spin.setRange(1, 1)
            self.range_end_spin.setRange(1, 1)
            self.range_start_spin.setValue(1)
            self.range_end_spin.setValue(1)
            self.range_start_spin.setEnabled(True)
            self.range_end_spin.setEnabled(True)
            self.range_start_spin.blockSignals(False)
            self.range_end_spin.blockSignals(False)

    def _start_unload_and_then(self, callback=None):
        # Re-use UnloadThread to release resources and optionally run callback when done
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            try:
                self.main_display.show_status("Unloading... 0%")
            except Exception:
                pass
            self._unloader = UnloadThread(self)
            self._unloader.progress.connect(self.progress_bar.setValue)
            self._unloader.progress.connect(lambda v: self.main_display.show_status(f"Unloading... {v}%"))
            def _on_finished():
                try:
                    self.progress_bar.setVisible(False)
                except Exception:
                    pass
                try:
                    self.main_display.clear_status()
                except Exception:
                    pass
                # update UI state after unload
                try:
                    self.frame_index_spin.setRange(1, 1)
                    self.frame_index_spin.setValue(1)
                    self.frame_index_spin.setEnabled(True)
                    self.prev_frame_btn.setEnabled(False)
                    self.next_frame_btn.setEnabled(False)
                    self.play_btn.setEnabled(False)
                    self.speed_combo.setEnabled(False)
                    self.stack_btn.setEnabled(False)
                    self.export_btn.setEnabled(False)
                    self.range_start_spin.setRange(1, 1)
                    self.range_end_spin.setRange(1, 1)
                    self.range_start_spin.setValue(1)
                    self.range_end_spin.setValue(1)
                except Exception:
                    pass
                if callback:
                    try:
                        callback()
                    except Exception:
                        pass
            self._unloader.finished.connect(_on_finished)
            self._unloader.start()
        except Exception:
            if callback:
                callback()

    def _show_recent_menu(self):
        try:
            menu = QMenu(self)
            recs = get_recents_for_mode('raw', limit=7)
            if not recs:
                a = menu.addAction("No recent files")
                a.setEnabled(False)
            else:
                for r in recs:
                    ts = r.get('last_opened', '')
                    display = f"{os.path.basename(r.get('path',''))} — {ts[:19]}"
                    act = menu.addAction(display)
                    act.setToolTip(r.get('path'))
                    path = r.get('path')
                    act.triggered.connect(lambda checked, p=path: self._start_unload_and_then(lambda: self._do_load_raw_file(p)))
                # view more
            all_recs = get_recents_for_mode('raw')
            if len(all_recs) > 7:
                menu.addSeparator()
                vm = menu.addAction("View more...")
                vm.triggered.connect(lambda: self._open_full_history('raw'))
            pos = self.load_menu_btn.mapToGlobal(self.load_menu_btn.rect().bottomLeft())
            menu.exec_(pos)
        except Exception:
            pass

    def _open_full_history(self, mode):
        try:
            sel = select_from_history(self, mode=mode)
            if sel:
                if mode == 'raw':
                    self._start_unload_and_then(lambda: self._do_load_raw_file(sel))
        except Exception:
            pass
    def unload_raw_file(self):
        # Public action to unload current file with progress
        if getattr(self, 'raw_data', None) is None and getattr(self, 'lazy_frames', None) is None and not (getattr(self, 'loading_thread', None) and getattr(self.loading_thread, 'isRunning', lambda: False)()):
            # nothing to do
            return
        self._start_unload_and_then()

    def edit_params(self):
        if not self.last_file_path:
            self.load_raw_file()
            return

        dialog = RawParameterDialog(self)
        # Prefer last-used file parameters instead of decoded frame shape,
        # which may already be wrong when user wants to fix configuration.
        try:
            p = self.last_params or {}
            w = int(p.get("width", self.raw_data.shape[1] if self.raw_data is not None else 8448))
            h = int(p.get("height", self.raw_data.shape[0] if self.raw_data is not None else 384))
            bd = int(p.get("bitdepth", self.bitdepth))
        except Exception:
            w = self.raw_data.shape[1] if self.raw_data is not None else 8448
            h = self.raw_data.shape[0] if self.raw_data is not None else 384
            bd = self.bitdepth
        dialog.width_entry.setText(str(w))
        dialog.height_entry.setText(str(h))
        dialog.bitdepth_var.setCurrentText(str(bd))

        if dialog.exec_() == QDialog.Accepted:
            params = dialog.get_parameters()
            self.last_params = params.copy()
            # Always force a real reload from disk with the new params.
            file_path = self.last_file_path
            if not file_path:
                QMessageBox.critical(self, "Load Error", "No raw file is selected.")
                return
            self._start_unload_and_then(lambda p=params, fp=file_path: self._do_load_raw_file(fp, forced_params=p))

    def on_load_finished(self, original_data):
        self.raw_data = original_data
        self.bitdepth = self.loading_thread.bitdepth
        self.export_btn.setEnabled(True)
        # Persist params silently for future auto-loads
        try:
            save_params_for_path(self.last_file_path, getattr(self, 'last_params', {}))
        except Exception:
            pass
        # Clear status overlay and hide progress
        try:
            self.main_display.clear_status()
        except Exception:
            pass
        self.progress_bar.setVisible(False)
        self.update_contrast_controls()
        self.update_display()
        self.update_histogram()
        QMessageBox.information(self, "Success", "Raw dataset loaded successfully.")
        gc.collect()

    def on_normalized_ready(self, display_data):
        try:
            self.normalized_data = display_data
            self.update_display()
        except Exception:
            pass

    def on_load_error(self, msg):
        self.progress_bar.setVisible(False)
        try:
            self.main_display.clear_status()
        except Exception:
            pass
        QMessageBox.critical(self, "Load Error", msg)

    def on_load_error(self, msg):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Load Error", msg)

    def update_display(self):
        if self.normalized_data is None:
            return
        data = self.normalized_data.copy()
        if self.enhance_cb.isChecked():
            self.contrast_min = int(self.min_spin.value())
            self.contrast_max = int(self.max_spin.value())
            if self.contrast_max > self.contrast_min:
                source = self.raw_data if self.raw_data is not None else self.normalized_data
                scale = 255.0 / max(1, self.contrast_max - self.contrast_min)
                data = np.clip((source.astype(np.float64) - self.contrast_min) * scale, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(data)
        self.current_pil_image = pil_img
        self.main_display.show_image(pil_img)
        # RAW mode: treat the displayed image as a single "band" (do not auto-split/stack)
        # This ensures flips/rotations operate on the whole image (not on a partial band).
        try:
            self.main_display.is_individual = True
            self.main_display.is_frame_stack = False
            # make frame_h explicit so downstream logic treats this as a single-band image
            self.main_display.frame_h = getattr(self.main_display, 'full_height', self.main_display.frame_h if hasattr(self.main_display, 'frame_h') else 0)
        except Exception:
            pass

    def update_contrast_controls(self):
        max_dn = float(self._current_max_dn())
        self.min_spin.setRange(0, max_dn)
        self.max_spin.setRange(0, max_dn)
        self.min_spin.setValue(0)
        self.max_spin.setValue(max_dn)
        self.contrast_min = 0
        self.contrast_max = int(max_dn)
        try:
            tip = f"Native DN range for {int(self.bitdepth)}-bit data: 0-{int(max_dn)}"
            self.min_spin.setToolTip(tip)
            self.max_spin.setToolTip(tip)
        except Exception:
            pass
        self.enhance_cb.setChecked(False)

    def _current_max_dn(self):
        try:
            bd = int(getattr(self, "bitdepth", 8))
        except Exception:
            bd = 8
        return (1 << bd) - 1 if bd > 8 else 255

    def update_histogram(self):
        if self.raw_data is None:
            return

        # Respect the user's histogram selection in the HistogramViewer radios.
        try:
            # SINGLE: always show only the current frame
            if hasattr(self, 'hist_viewer') and getattr(self.hist_viewer, 'single_frame_radio', None) and self.hist_viewer.single_frame_radio.isChecked():
                self.hist_viewer.update_histogram({'raw': [self.raw_data]}, self.current_frame_index, "Single", ignore_extremes=False)
                return

            # RANGE: use user-specified range if available, prefer streaming LazyFrames
            if hasattr(self, 'hist_viewer') and getattr(self.hist_viewer, 'frame_range_radio', None) and self.hist_viewer.frame_range_radio.isChecked():
                # default internal indices
                sf = 0
                ef = max(0, self.num_frames - 1)
                try:
                    # UI is 1-based; convert to 0-based internal indices
                    sf_ui = int(self.range_start_spin.value())
                    ef_ui = int(self.range_end_spin.value())
                    sf_ui = max(1, min(sf_ui, max(1, self.num_frames)))
                    ef_ui = max(1, min(ef_ui, max(1, self.num_frames)))
                    if sf_ui > ef_ui:
                        sf_ui, ef_ui = ef_ui, sf_ui
                        self.range_start_spin.blockSignals(True)
                        self.range_end_spin.blockSignals(True)
                        self.range_start_spin.setValue(sf_ui)
                        self.range_end_spin.setValue(ef_ui)
                        self.range_start_spin.blockSignals(False)
                        self.range_end_spin.blockSignals(False)
                    sf = sf_ui - 1
                    ef = ef_ui - 1
                except Exception:
                    pass

                if getattr(self, 'lazy_frames', None) is not None:
                    try:
                        self.hist_viewer.update_histogram({'raw': self.lazy_frames}, self.current_frame_index, "Range", start_frame=sf, end_frame=ef, ignore_extremes=False)
                        return
                    except Exception:
                        pass
                # Fallback: synthesize range from single-frame list
                try:
                    self.hist_viewer.update_histogram({'raw': [self.raw_data]}, self.current_frame_index, "Single", ignore_extremes=False)
                    return
                except Exception:
                    pass
        except Exception:
            # If anything goes wrong reading the radio state, fall back to safe single-frame path
            pass

        # Default behavior: single-frame histogram (safe fallback)
        self.hist_viewer.update_histogram({'raw': [self.raw_data]}, self.current_frame_index, "Single", ignore_extremes=False)

    def _on_range_spin_changed(self, _val=None):
        """Validate start/end spins and trigger a range histogram if selected."""
        try:
            # UI is 1-based; convert to internal 0-based indices
            sf_ui = int(self.range_start_spin.value())
            ef_ui = int(self.range_end_spin.value())
            # clamp UI values to valid 1..num_frames
            sf_ui = max(1, min(sf_ui, max(1, self.num_frames)))
            ef_ui = max(1, min(ef_ui, max(1, self.num_frames)))
            if sf_ui > ef_ui:
                # keep UI consistent: make start <= end
                sf_ui, ef_ui = ef_ui, sf_ui
                self.range_start_spin.blockSignals(True)
                self.range_end_spin.blockSignals(True)
                self.range_start_spin.setValue(sf_ui)
                self.range_end_spin.setValue(ef_ui)
                self.range_start_spin.blockSignals(False)
                self.range_end_spin.blockSignals(False)

            sf = sf_ui - 1
            ef = ef_ui - 1

            # If histogram is in range mode, update it
            if hasattr(self, 'hist_viewer') and getattr(self.hist_viewer, 'frame_range_radio', None) and self.hist_viewer.frame_range_radio.isChecked():
                # Re-run histogram over the new subrange
                try:
                    if getattr(self, 'lazy_frames', None) is not None:
                        self.hist_viewer.update_histogram({'raw': self.lazy_frames}, self.current_frame_index, "Range", start_frame=sf, end_frame=ef, ignore_extremes=False)
                    else:
                        # single-frame fallback
                        self.hist_viewer.update_histogram({'raw': [self.raw_data]}, self.current_frame_index, "Single", ignore_extremes=False)
                except Exception:
                    pass
        except Exception:
            pass
    # ---------------- Frame navigation ----------------
    def prev_frame(self):
        if not self.lazy_frames:
            return
        idx = max(0, self.current_frame_index - 1)
        if idx != self.current_frame_index:
            # frame_index_spin is 1-based in the UI
            try:
                self.frame_index_spin.setValue(idx + 1)
            except Exception:
                self.frame_index_spin.setValue(max(1, idx + 1))

    def next_frame(self):
        if not self.lazy_frames:
            return
        idx = min(self.num_frames - 1, self.current_frame_index + 1)
        if idx != self.current_frame_index:
            try:
                self.frame_index_spin.setValue(idx + 1)
            except Exception:
                self.frame_index_spin.setValue(max(1, idx + 1))

    def goto_frame(self, idx):
        """Accept either UI (1-based) or internal (0-based) idx and navigate to the frame."""
        try:
            idx = int(idx)
        except Exception:
            return
        if not self.lazy_frames:
            return
        # If value looks like UI (1..N) convert to 0-based, otherwise accept 0-based
        if 1 <= idx <= self.num_frames:
            idx0 = idx - 1
        else:
            idx0 = idx
        if idx0 < 0 or idx0 >= self.num_frames:
            return
        if idx0 == self.current_frame_index:
            return
        self.set_current_frame(idx0)

    def set_current_frame(self, idx):
        is_playing = getattr(self, 'playing', False)
        if not is_playing:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(f"Loading frame {idx + 1}/{self.num_frames}...")
        try:
            raw = self.lazy_frames.get_raw(idx)
            self.raw_data = raw
            self.bitdepth = self.lazy_frames.bitdepth
            if self.bitdepth > 8:
                max_val = (1 << self.bitdepth) - 1
                if self.bitdepth == 16:
                    actual_max = raw.max()
                    if actual_max <= 1023:
                        max_val = 1023
                    elif actual_max <= 4095:
                        max_val = 4095
                self.normalized_data = ((raw.astype(np.float64) / max_val) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                self.normalized_data = raw.astype(np.uint8)
            self.current_frame_index = idx
            self.update_display()
            if not is_playing:
                self.update_histogram()

            self.frame_index_spin.blockSignals(True)
            try:
                self.frame_index_spin.setValue(idx + 1)
            except Exception:
                self.frame_index_spin.setValue(max(1, idx + 1))
            self.frame_index_spin.blockSignals(False)
        except Exception as e:
            QMessageBox.critical(self, "Frame Error", f"Error loading frame {idx + 1}: {str(e)}")
            if is_playing:
                self.toggle_play()
        finally:
            if not is_playing:
                self.progress_bar.setVisible(False)
                gc.collect()

    # ---------------- Stack creation ----------------
    def create_stack_view(self):
        if not self.lazy_frames:
            QMessageBox.warning(self, "Warning", "No multi-frame raw loaded to create stack.")
            return

        # Estimate a safe pixel budget from available RAM (conservative fraction)
        try:
            avail = psutil.virtual_memory().available
            # use up to 35% of available RAM for the stacked image buffer (conservative)
            safe_bytes = int(avail * 0.35)
            # stacked image uses 1 byte per pixel ('L' mode) + PIL overhead — be conservative
            safe_pixels = max(10_000_000, min(safe_bytes // 1, 1_200_000_000))
        except Exception:
            safe_pixels = 50_000_000

        # Allow the existing default to act as a minimum; caller can still override in future
        effective_max_pixels = max(50_000_000, int(safe_pixels))

        # Compute how many frames will be sampled with that budget (same logic as StackBuildThread.run)
        w = getattr(self.lazy_frames, 'w', None) or getattr(self.lazy_frames, 'width', None)
        h = getattr(self.lazy_frames, 'h', None) or getattr(self.lazy_frames, 'height', None)
        num_frames = len(self.lazy_frames)
        if not w or not h:
            QMessageBox.information(self, "Stacking", "Could not determine frame size; using conservative sampling.")
            effective_max_pixels = 50_000_000

        # Provide informative message to the user about sampling
        try:
            target_w = min(w, 2048)
            scale = target_w / float(w)
            th = max(1, int(round(h * scale)))
            per_frame_pixels = target_w * th
            max_frames_allowed = max(1, effective_max_pixels // max(1, per_frame_pixels))
            if max_frames_allowed < num_frames:
                info = f"Stack will sample {max_frames_allowed} of {num_frames} frames (stride applied) to limit memory use)."
            else:
                info = f"Stack will include all {num_frames} frames (within memory budget)."
        except Exception:
            info = "Frames will be sampled/downscaled as needed to limit memory usage."

        QMessageBox.information(self, "Stacking", info)

        self.stack_btn.setEnabled(False)
        # determine requested start/end (if spinboxes are present)
        try:
            # UI is 1-based; convert to 0-based internal indices
            sf_ui = int(self.range_start_spin.value())
            ef_ui = int(self.range_end_spin.value())
            sf_ui = max(1, min(sf_ui, max(1, num_frames)))
            ef_ui = max(1, min(ef_ui, max(1, num_frames)))
            sf = sf_ui - 1
            ef = ef_ui - 1
        except Exception:
            sf = 0
            ef = max(0, num_frames - 1)

        # Pass the RAM-aware pixel budget and requested subrange to the worker
        self.stack_thread = StackBuildThread(self.lazy_frames, parent=self, max_pixels=effective_max_pixels, start_frame=sf, end_frame=ef)
        self.stack_thread.progress.connect(lambda v: self.progress_bar.setValue(v))
        self.stack_thread.finished.connect(self.on_stack_finished)
        self.stack_thread.error.connect(self.on_stack_error)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.stack_thread.start()

    def on_stack_finished(self, pil_img):
        try:
            self.progress_bar.setVisible(False)
            self.stack_btn.setEnabled(True)
            self.frames_display.show_image(pil_img)
            # Frames tab is a stacked view — ensure it's treated as such
            try:
                self.frames_display.is_individual = False
                self.frames_display.is_frame_stack = True
            except Exception:
                pass
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Frames":
                    self.tabs.setCurrentIndex(i)
                    break
        except Exception as e:
            print(f"Error displaying stack: {e}")

    def on_stack_error(self, msg):
        self.progress_bar.setVisible(False)
        self.stack_btn.setEnabled(True)
        QMessageBox.critical(self, "Stack Error", msg)

    def export_current(self):
        if not hasattr(self, 'tabs') or self.tabs.count() == 0:
            return

        tab_idx = self.tabs.currentIndex()
        tab_name = self.tabs.tabText(tab_idx) if tab_idx >= 0 else "Display"
        default_name = f"raw_{tab_name.lower()}.png"

        if tab_name == "Display":
            img = getattr(self.main_display, 'current_pil_image', None)
            if img is None:
                QMessageBox.warning(self, "Export Warning", "Nothing to export in Display tab.")
                return
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export", default_name, "PNG (*.png);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
            )
            if filename:
                img.save(filename)
                QMessageBox.information(self, "Success", f"Exported to {filename}")
            return

        if tab_name == "Frames":
            img = getattr(self.frames_display, 'current_pil_image', None)
            if img is None:
                QMessageBox.warning(self, "Export Warning", "Nothing to export in Frames tab.")
                return
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export", default_name, "PNG (*.png);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
            )
            if filename:
                img.save(filename)
                QMessageBox.information(self, "Success", f"Exported to {filename}")
            return

        current_tab = self.tabs.currentWidget()
        if current_tab is None:
            return
        pixmap = current_tab.grab()
        if pixmap.isNull():
            QMessageBox.warning(self, "Export Warning", "Nothing visible to export in this tab.")
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export", default_name, "PNG (*.png)")
        if filename:
            pixmap.save(filename)
            QMessageBox.information(self, "Success", f"Exported to {filename}")

    def on_pixel_info(self, x, y, values, is_rgb):
        if self.raw_data is None or not hasattr(self.main_display, 'pixel_info_box_overlay'):
            return
        try:
            x = int(np.clip(x, 0, self.raw_data.shape[1] - 1))
            y = int(np.clip(y, 0, self.raw_data.shape[0] - 1))
            size = self.matrix_size_var.value()
            half = size // 2
            y0 = max(0, y - half)
            y1 = min(self.raw_data.shape[0], y + half + 1)
            x0 = max(0, x - half)
            x1 = min(self.raw_data.shape[1], x + half + 1)
            raw_matrix = self.raw_data[y0:y1, x0:x1]

            pad_top = max(0, half - (y - y0))
            pad_bottom = max(0, (y + half + 1) - y1)
            pad_left = max(0, half - (x - x0))
            pad_right = max(0, (x + half + 1) - x1)
            if pad_top or pad_bottom or pad_left or pad_right:
                raw_matrix = np.pad(
                    raw_matrix,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant',
                    constant_values=0
                )
            raw_matrix = raw_matrix[:size, :size]

            overlay = self.main_display.pixel_info_box_overlay
            if overlay is not None:
                overlay.update_info(x, y, raw_matrix, is_rgb=False)

            margin = 10
            gv = self.main_display.graphics_view
            viewport = gv.viewport()
            viewport_rect = viewport.rect()
            px_x = viewport.mapToParent(viewport_rect.bottomLeft()).x() + margin
            py_y = viewport.mapToParent(viewport_rect.bottomLeft()).y() - overlay.height() - margin
            current_pos = overlay.pos()
            if current_pos.x() != px_x or current_pos.y() != py_y:
                overlay.move(int(px_x), int(py_y))
        except Exception as e:
            print(f"Error in on_pixel_info: {e}")

    def open_editor_tab(self):
        if not self.current_pil_image:
            QMessageBox.warning(self, "No Image", "No image loaded. Please load a raw file first.")
            return
        if EditorTab is None:
            QMessageBox.warning(self, "Error", "EditorTab module not available.")
            return
        main_app = self.parent()
        max_iterations = 10
        while main_app and not hasattr(main_app, 'view_tabs') and max_iterations > 0:
            main_app = main_app.parent()
            max_iterations -= 1
        if not main_app or not hasattr(main_app, 'view_tabs'):
            top_widget = QApplication.activeWindow()
            if top_widget and hasattr(top_widget, 'view_tabs'):
                main_app = top_widget
        if not main_app or not hasattr(main_app, 'view_tabs'):
            QMessageBox.warning(self, "Error", "Could not find main app to create editor tab.")
            return
        editor = EditorTab(self, main_app)
        if editor.original_array is None:
            return
        tab_name = f"Editor – Raw"
        idx = main_app.view_tabs.addTab(editor, tab_name)
        main_app.view_tabs.setCurrentIndex(idx)
        try:
            main_app._set_custom_close_button(idx)
        except Exception:
            pass

    def save_state(self):
        try:
            tab_name = self.tabs.tabText(self.tabs.currentIndex()) if hasattr(self, 'tabs') and self.tabs.count() > 0 else "Display"
        except Exception:
            tab_name = "Display"
        return {
            'last_file_path': self.last_file_path,
            'last_params': self.last_params,
            'bitdepth': getattr(self, 'bitdepth', 8),
            'contrast_enhance': bool(self.enhance_cb.isChecked()) if hasattr(self, 'enhance_cb') else False,
            'contrast_min': int(self.min_spin.value()) if hasattr(self, 'min_spin') else 0,
            'contrast_max': int(self.max_spin.value()) if hasattr(self, 'max_spin') else 255,
            'matrix_size': int(self.matrix_size_var.value()) if hasattr(self, 'matrix_size_var') else 5,
            'tab_index': int(self.tabs.currentIndex()) if hasattr(self, 'tabs') else 0,
            'tab_name': tab_name,
            'current_frame_index': int(getattr(self, 'current_frame_index', 0)),
        }

    def load_state(self, data):
        try:
            lp = data.get('last_file_path')
            params = data.get('last_params', {}) or {}

            # Restore UI controls
            if hasattr(self, 'matrix_size_var') and 'matrix_size' in data:
                try:
                    self.matrix_size_var.setValue(int(data.get('matrix_size', self.matrix_size_var.value())))
                except Exception:
                    pass
            if hasattr(self, 'enhance_cb'):
                try:
                    self.enhance_cb.setChecked(bool(data.get('contrast_enhance', False)))
                except Exception:
                    pass
            if hasattr(self, 'min_spin') and 'contrast_min' in data:
                try:
                    self.min_spin.setValue(int(data.get('contrast_min', 0)))
                except Exception:
                    pass
            if hasattr(self, 'max_spin') and 'contrast_max' in data:
                try:
                    self.max_spin.setValue(int(data.get('contrast_max', 255)))
                except Exception:
                    pass

            if lp and params:
                self.last_file_path = lp
                self._update_host_tab_name(lp)
                self.last_params = params.copy()
                try:
                    lf = LazyFrames(lp, params.get('width', 0), params.get('height', 0), params.get('bitdepth', 8))
                    if lf and len(lf) >= 1:
                        self.lazy_frames = lf
                        self.num_frames = len(lf)
                        self.current_frame_index = int(data.get('current_frame_index', 0))
                        self.frame_index_spin.setRange(1, max(1, self.num_frames))
                        try:
                            self.frame_index_spin.setValue(self.current_frame_index + 1)
                        except Exception:
                            self.frame_index_spin.setValue(1)
                        self.prev_frame_btn.setEnabled(self.num_frames > 1)
                        self.next_frame_btn.setEnabled(self.num_frames > 1)
                        self.play_btn.setEnabled(self.num_frames > 1)
                        self.speed_combo.setEnabled(self.num_frames > 1)
                        self.frame_index_spin.setEnabled(True)
                        self.stack_btn.setEnabled(self.num_frames > 1)
                        self.export_btn.setEnabled(True)
                        try:
                            raw = lf.get_raw(self.current_frame_index)
                            self.raw_data = raw
                            self.bitdepth = lf.bitdepth
                            if self.bitdepth > 8:
                                max_val = self._current_max_dn()
                                self.normalized_data = ((raw.astype(np.float64) / max_val) * 255.0).clip(0, 255).astype(np.uint8)
                            else:
                                self.normalized_data = raw.astype(np.uint8)
                            self.update_contrast_controls()
                            self.update_display()
                            self.update_histogram()
                        except Exception as e:
                            print(f"RawViewer.load_state: failed to read frame: {e}")
                except Exception as e:
                    print(f"RawViewer.load_state: failed to start loader: {e}")

            try:
                # Prefer tab name (robust across reorder); fall back to numeric index for older saved states
                tname = data.get('tab_name')
                if tname and hasattr(self, 'tabs'):
                    for i in range(self.tabs.count()):
                        if self.tabs.tabText(i) == tname:
                            self.tabs.setCurrentIndex(i)
                            break
                else:
                    ti = int(data.get('tab_index', 0))
                    if hasattr(self, 'tabs') and 0 <= ti < self.tabs.count():
                        self.tabs.setCurrentIndex(ti)
            except Exception:
                pass
        except Exception as e:
            print(f"RawViewer.load_state error: {e}")

    def toggle_play(self):
        if getattr(self, 'num_frames', 0) <= 1:
            return
        
        self.playing = not getattr(self, 'playing', False)
        if self.playing:
            self.play_btn.setText("⏸ Pause")
            try:
                self.play_timer.start(self.play_delay)
            except Exception:
                self.play_next_frame()
        else:
            self.play_btn.setText("▶ Play")
            try:
                self.play_timer.stop()
            except Exception:
                pass
                
    def change_speed(self, text):
        try:
            rate = float(text[:-1])
            self.play_delay = int(100 / rate)
            if getattr(self, 'playing', False) and getattr(self, 'play_timer', None) and self.play_timer.isActive():
                self.play_timer.setInterval(self.play_delay)
        except Exception as e:
            print(f"Speed change error: {e}")
            
    def play_next_frame(self):
        if not getattr(self, 'playing', False) or getattr(self, 'num_frames', 0) <= 1:
            self.play_btn.setText("▶ Play")
            self.playing = False
            return
            
        current = self.frame_index_spin.value()
        if current < self.num_frames:
            self.frame_index_spin.setValue(current + 1)
        else:
            self.frame_index_spin.setValue(1)
