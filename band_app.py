from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QSpinBox, QComboBox, QGridLayout,
    QCheckBox, QRadioButton, QSlider, QLineEdit, QFileDialog, QMessageBox, QDoubleSpinBox, QSizePolicy,
    QFormLayout, QButtonGroup, QTabWidget, QToolButton, QScrollArea, QTextEdit, QDialog, QTabBar, QProgressDialog, QApplication, QMenu, QShortcut, QProgressBar, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QThread
from PyQt5.QtGui import QKeySequence, QTransform
import json
import hashlib
import psutil
import time
import gc
import re
import os
import numpy as np
from PIL import Image
from image_viewer import GraphicsImageViewer
from ui_components import PixelInfoBox, HistogramViewer, ParameterDialog, CustomTabBar
from utils import (
    meters_per_degree, image_coords_to_latlon, check_memory_requirement, unpack_by_bitdepth, LazyFrames, TerminalWidget, save_params_for_path, add_recent, get_recents_for_mode, select_from_history, get_saved_params_for_file, load_folder_params, infer_dataset_image_params, normalize_tdi_stage
)
from band_views import BandViewsMixin
try:
    import sip
except Exception:
    sip = None
try:
    from help_tab import create_help_tab
except ImportError:
    pass
try:
    from editor_tab import EditorTab
except ImportError:
    EditorTab = None
class LoadWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int) # Emit progress percentage
    def __init__(self, folder, width, height, bitdepth, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.width = width
        self.height = height
        self.bitdepth = bitdepth
    def run(self):
        try:
            self.progress.emit(0) # Start
            def interrupted():
                return self.isInterruptionRequested()
            # List all files once for efficiency
            self.progress.emit(int(1 * 0.8)) # 0.8
            files = os.listdir(self.folder)
            self.progress.emit(int(5 * 0.8)) # 4
            # Meta file parsing with granular progress
            meta_file = None
            for i, f in enumerate(files):
                if interrupted():
                    self.error.emit("Loading cancelled")
                    return
                self.progress.emit(int((5 + int((i + 1) / len(files) * 15)) * 0.8)) # 4-16
                if f.endswith(".meta"):
                    meta_file = os.path.join(self.folder, f)
                    break
            center_lat = None
            center_lon = None
            pixel_size_m = 1.5
            geo_info = None
            if meta_file and os.path.exists(meta_file):
                self.progress.emit(int(21 * 0.8)) # 16.8 ~17
                with open(meta_file, 'r', errors='ignore') as mf:
                    text = mf.read()
                self.progress.emit(int(25 * 0.8)) # 20
                # Parse lat/lon with sub-steps
                lat_patterns = [
                    r'center[_\s-]*lat(?:itude)?\s*[=:]\s*([+-]?\d+\.\d+)',
                    r'lat(?:itude)?[_\s-]*center\s*[=:]\s*([+-]?\d+\.\d+)',
                    r'lat(?:itude)?\s*[=:]\s*([+-]?\d+\.\d+)',
                    r'latitude\s*=\s*([+-]?\d+\.\d+)',
                ]
                lon_patterns = [
                    r'center[_\s-]*lon(?:gitude)?\s*[=:]\s*([+-]?\d+\.\d+)',
                    r'lon(?:gitude)?[_\s-]*center\s*[=:]\s*([+-]?\d+\.\d+)',
                    r'lon(?:gitude)?\s*[=:]\s*([+-]?\d+\.\d+)',
                    r'longitude\s*=\s*([+-]?\d+\.\d+)',
                ]
                # Parse lat patterns granularly
                for step, p in enumerate(lat_patterns):
                    if interrupted():
                        self.error.emit("Loading cancelled")
                        return
                    m = re.search(p, text, re.IGNORECASE)
                    if m:
                        center_lat = float(m.group(1))
                        break
                    self.progress.emit(int((25 + (step + 1) * 2) * 0.8)) # 20-26.4
                # Parse lon patterns
                for step, p in enumerate(lon_patterns):
                    if interrupted():
                        self.error.emit("Loading cancelled")
                        return
                    m = re.search(p, text, re.IGNORECASE)
                    if m:
                        center_lon = float(m.group(1))
                        break
                    self.progress.emit(int((33 + (step + 1) * 2) * 0.8)) # 26.4-32.8
                if center_lat is None or center_lon is None:
                    self.progress.emit(int(42 * 0.8)) # 33.6 ~34
                    mlat = re.search(r'lat(?:itude)?[^0-9+-]*([+-]?\d+\.\d+)', text, re.IGNORECASE)
                    mlon = re.search(r'lon(?:gitude)?[^0-9+-]*([+-]?\d+\.\d+)', text, re.IGNORECASE)
                    if mlat and center_lat is None:
                        center_lat = float(mlat.group(1))
                    if mlon and center_lon is None:
                        center_lon = float(mlon.group(1))
                    self.progress.emit(int(45 * 0.8)) # 36
                if center_lat is None or center_lon is None:
                    self.progress.emit(int(46 * 0.8)) # 36.8 ~37
                    floats = [float(tok) for tok in re.findall(r'([+-]?\d+\.\d+)', text)]
                    if len(floats) >= 2:
                        a, b = floats[0], floats[1]
                        if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
                            if center_lat is None: center_lat = a
                            if center_lon is None: center_lon = b
                        else:
                            for i in range(len(floats)-1):
                                if interrupted():
                                    self.error.emit("Loading cancelled")
                                    return
                                a, b = floats[i], floats[i+1]
                                if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
                                    if center_lat is None: center_lat = a
                                    if center_lon is None: center_lon = b
                                    break
                                self.progress.emit(int((46 + int((i + 1) / len(floats) * 4)) * 0.8)) # 37-40
                    self.progress.emit(int(50 * 0.8)) # 40
                if center_lat is not None and center_lon is not None:
                    geo_info = (center_lat, center_lon, self.width, self.height, pixel_size_m)
            self.progress.emit(int(20 * 0.8)) # 16 (adjusted)
            # Base name extraction with granular progress
            base_name = None
            self.progress.emit(int(21 * 0.8)) # 16.8 ~17
            # Check JSON files
            json_files = [f for f in files if f.endswith(".json") and f != "parameters.json"]
            for i, f in enumerate(json_files):
                if interrupted():
                    self.error.emit("Loading cancelled")
                    return
                self.progress.emit(int((21 + int((i + 1) / max(1, len(json_files)) * 4)) * 0.8)) # 17-20
                with open(os.path.join(self.folder, f), "r") as jf:
                    try:
                        config = json.load(jf)
                    except Exception:
                        config = {}
                    base_name = os.path.splitext(f)[0]
                if base_name:
                    break
            if not base_name:
                self.progress.emit(int(25 * 0.8)) # 20
                log_files = [f for f in files if f.endswith(".log")]
                for i, f in enumerate(log_files):
                    if interrupted():
                        self.error.emit("Loading cancelled")
                        return
                    self.progress.emit(int((25 + int((i + 1) / max(1, len(log_files)) * 5)) * 0.8)) # 20-24
                    with open(os.path.join(self.folder, f), "r") as lf:
                        for line_num, line in enumerate(lf):
                            if "Arguments received from parameter file" in line:
                                base_name = os.path.splitext(f)[0]
                                break
                            # Granular per log line if large (throttle)
                            if line_num % 100 == 0:
                                self.progress.emit(int((25 + int((i + 1) / max(1, len(log_files)) * 5) + 0.1) * 0.8))
                        if base_name:
                            break
            if not base_name:
                self.progress.emit(int(30 * 0.8)) # 24
                candidates = set()
                for i, f in enumerate(files):
                    if interrupted():
                        self.error.emit("Loading cancelled")
                        return
                    self.progress.emit(int((30 + int((i + 1) / len(files) * 2)) * 0.8)) # 24-25.6
                    if '.band' in f:
                        parts = f.split('.band')
                        if len(parts) == 2 and parts[1] and len(parts[1]) <= 3:
                            candidates.add(parts[0])
                if len(candidates) == 1:
                    base_name = list(candidates)[0]
                elif len(candidates) > 1:
                    base_name = sorted(candidates)[0]
                else:
                    raise ValueError("No .bandXX files found in folder.")
                self.progress.emit(int(32 * 0.8)) # 25.6
            self.progress.emit(int(30 * 0.8)) # 24
            # Load band frames with granular progress per band and sub-checks
            band_frames = {}
            files_checked = []
            num_bands = 7
            for i in range(num_bands):
                if interrupted():
                    self.error.emit("Loading cancelled")
                    return
                band_key = f"b{i}"
                base_progress = int((30 + (i * 6)) * 0.8) # 24 to 52.8
                self.progress.emit(base_progress) # Band start
                # Check full fname
                fname_full = os.path.join(self.folder, f"{base_name}.band{i}")
                files_checked.append(fname_full)
                self.progress.emit(int((base_progress + 0.5) * 0.8)) # slight
                if os.path.exists(fname_full) and os.path.getsize(fname_full) > 0:
                    self.progress.emit(int((base_progress + 1) * 0.8))
                    # Granular for LazyFrames init (simulate sub-steps if slow)
                    for sub_step in range(5): # e.g., open, read header, validate, unpack, init
                        if interrupted():
                            self.error.emit("Loading cancelled")
                            return
                        time.sleep(0.005) # Simulate; remove if not needed
                        self.progress.emit(int((base_progress + 1 + (sub_step * 0.8)) * 0.8))
                    band_frames[band_key] = LazyFrames(fname_full, self.width, self.height, self.bitdepth)
                    self.progress.emit(int((base_progress + 5.5) * 0.8))
                    continue
                # Check binned
                self.progress.emit(int((base_progress + 0.5) * 0.8))
                fname_binned = os.path.join(self.folder, f"{base_name}.band{i}2")
                files_checked.append(fname_binned)
                if os.path.exists(fname_binned) and os.path.getsize(fname_binned) > 0:
                    self.progress.emit(int((base_progress + 1) * 0.8))
                    for sub_step in range(5):
                        if interrupted():
                            self.error.emit("Loading cancelled")
                            return
                        time.sleep(0.005)
                        self.progress.emit(int((base_progress + 1 + (sub_step * 0.8)) * 0.8))
                    band_frames[f"{band_key}_binned"] = LazyFrames(fname_binned, self.width // 2, self.height // 2, self.bitdepth)
                    self.progress.emit(int((base_progress + 5.5) * 0.8))
                    continue
                # Check left/right (split)
                self.progress.emit(int((base_progress + 0.5) * 0.8))
                lfile = os.path.join(self.folder, f"{base_name}.band{i}0")
                files_checked.append(lfile)
                if os.path.exists(lfile) and os.path.getsize(lfile) > 0:
                    self.progress.emit(int((base_progress + 1) * 0.8))
                    for sub_step in range(3):
                        if interrupted():
                            self.error.emit("Loading cancelled")
                            return
                        time.sleep(0.005)
                        self.progress.emit(int((base_progress + 1 + (sub_step * 1.2)) * 0.8))
                    band_frames[f"{band_key}_left"] = LazyFrames(lfile, self.width // 2, self.height, self.bitdepth)
                self.progress.emit(int((base_progress + 2.5) * 0.8))
                rfile = os.path.join(self.folder, f"{base_name}.band{i}1")
                files_checked.append(rfile)
                if os.path.exists(rfile) and os.path.getsize(rfile) > 0:
                    for sub_step in range(3):
                        if interrupted():
                            self.error.emit("Loading cancelled")
                            return
                        time.sleep(0.005)
                        self.progress.emit(int((base_progress + 2.5 + (sub_step * 1.2)) * 0.8))
                    band_frames[f"{band_key}_right"] = LazyFrames(rfile, self.width // 2, self.height, self.bitdepth)
                self.progress.emit(int((base_progress + 5.5) * 0.8)) # Band complete
            self.progress.emit(int(72 * 0.8)) # 57.6 ~58
            if not band_frames:
                raise ValueError(f"No valid band frames loaded. Checked files: {', '.join(files_checked)}")
            # Build bands_info with granular progress
            self.progress.emit(int(73 * 0.8)) # 58.4 ~58
            base_keys = []
            for i, k in enumerate(band_frames.keys()):
                if interrupted():
                    self.error.emit("Loading cancelled")
                    return
                base = k.split('_')[0]
                if base not in base_keys:
                    base_keys.append(base)
                self.progress.emit(int((73 + int((i + 1) / len(band_frames) * 7)) * 0.8)) # 58-64
            bands_info = {}
            for idx, base in enumerate(base_keys):
                self.progress.emit(int((80 + (idx * 2)) * 0.8)) # 64 +
                variants = [k for k in band_frames.keys() if k.startswith(base)]
                has_left = any('left' in v.lower() for v in variants)
                has_right = any('right' in v.lower() for v in variants)
                is_split = has_left and has_right
                bin_factor = 1
                explicit_binned_present = False
                for j, variant_name in enumerate(variants):
                    if interrupted():
                        self.error.emit("Loading cancelled")
                        return
                    vn = variant_name.lower()
                    m = re.search(r'bin(?:ned)?(?:[_\-]?)(\d+)', variant_name, re.IGNORECASE)
                    if m and m.group(1):
                        bf = int(m.group(1))
                        if bf > bin_factor:
                            bin_factor = bf
                    if 'binned' in vn or ('bin' in vn and re.search(r'bin\b', vn) and not re.search(r'bin(?:ned)?(?:[_\-]?\d+)', vn)):
                        explicit_binned_present = True
                    self.progress.emit(int((80 + (idx * 2) + int((j + 1) / len(variants) * 2)) * 0.8)) # sub
                if explicit_binned_present and bin_factor == 1:
                    bin_factor = 2
                is_binned = (bin_factor > 1) or explicit_binned_present
                is_split = is_split and not (len(variants) == 1 and '_' not in variants[0])
                bands_info[base] = {
                    'index': idx,
                    'variants': variants,
                    'binned': bool(is_binned),
                    'split': bool(is_split),
                    'bin_factor': int(bin_factor)
                }
                self.progress.emit(int((82 + (idx * 2)) * 0.8)) # 65.6 +
            self.progress.emit(int(90 * 0.8)) # 72
            result = {
                'band_frames': band_frames,
                'geo_info': geo_info,
                'bands_info': bands_info,
                'base_name': base_name
            }
            self.progress.emit(int(95 * 0.8)) # 76
            gc.collect() # Clean up
            self.progress.emit(80) # Complete loading phase
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
class ViewUpdateWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    images_ready = pyqtSignal(dict)
    def __init__(self, app_instance, full_refresh=False):
        super().__init__()
        self.app = app_instance
        self.full_refresh = full_refresh
        self.base_progress = 80
        self.scale_factor = 20
       
    def run(self):
        images = {} # Initialize here
        try:
            # Prepare All Bands with granular progress (scaled to 80-86)
            self.progress.emit(self.base_progress + int(1 * self.scale_factor / 100)) # ~80
            try:
                self.progress.emit(self.base_progress + int(5 * self.scale_factor / 100)) # ~81
                stitch_seq = self.app.build_stitch_sequence() # Assuming this exists
                for i, _ in enumerate(stitch_seq):
                    if self.isInterruptionRequested():
                        self.error.emit("Update cancelled")
                        return
                    self.progress.emit(self.base_progress + int((5 + int((i + 1) / len(stitch_seq) * 20)) * self.scale_factor / 100)) # 81-85
                    time.sleep(0.01) # Simulate processing each band in stitch
                images['all_bands'] = self.app._prepare_all_bands_images()
                self.progress.emit(self.base_progress + int(30 * self.scale_factor / 100)) # ~86
            except Exception as e:
                print(f"All bands prep error: {e}")
                self.progress.emit(self.base_progress + int(25 * self.scale_factor / 100)) # ~85
            # Prepare RGB with granular progress (scaled to 86-98)
            self.progress.emit(self.base_progress + int(31 * self.scale_factor / 100)) # ~86.2
            try:
                channels = ["R", "G", "B"]
                for i, ch in enumerate(channels):
                    if self.isInterruptionRequested():
                        self.error.emit("Update cancelled")
                        return
                    self.progress.emit(self.base_progress + int((31 + int((i + 1) / len(channels) * 49)) * self.scale_factor / 100)) # 86.2-98
                    time.sleep(0.02) # Replace with actual sub-work
                images['rgb'] = self.app._prepare_rgb_images()
                self.progress.emit(self.base_progress + int(80 * self.scale_factor / 100)) # ~98
            except Exception as e:
                print(f"RGB prep error: {e}")
                self.progress.emit(self.base_progress + int(50 * self.scale_factor / 100)) # ~90
            # Individual is lazy, skip precompute (to 100)
            self.progress.emit(self.base_progress + int(95 * self.scale_factor / 100)) # ~99.5
            gc.collect()
            self.images_ready.emit(images)
            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
class MemoryMonitor(QThread):
    check_pressure = pyqtSignal()
    unload_request = pyqtSignal(str) # NEW: Signal to request unload on main thread (key as str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
    def run(self):
        while self.running:
            time.sleep(1.0) # Check every second
            usage = psutil.virtual_memory().percent
            # Trigger memory pressure handling when usage crosses 90%
            if usage >= 90.0:
                self.check_pressure.emit() # Existing: Triggers pressure handler
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
class BandStitchProApp(BandViewsMixin, QWidget):
    def __init__(self, parent=None, main_app=None):
        super().__init__(parent)
        self.main_app = main_app
        self.band_frames = {}
        self.bitdepth = 10
        self.base_name = ""
        self.current_frame_index = 0
        self.flip_flags = {}
        self.play_delay = 100
        self.play_frame_count = 0
        # Playback timer (used to advance frames while playing)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_next_frame)
        self.band_offsets = {f"b{i}": {"x": 0, "y": 0} for i in range(7)}
        self.band_enabled = {}
        self.band_gaps = 10
        self.rgb_bands = {"R": "b0", "G": "b1", "B": "b2"}
        self.current_folder = None
        self.playback_mode = False
        self.playing = False
        self.individual_band_keys = []
        self.last_play_frame_time = None
        self.loaded_band_memories = {} # Reserved/predicted memory per band/tab
        self.loaded_individual_bands = set()
        self.access_times = {} # {key: last_access_time}
        self.unloaded_keys = set() # Track unloaded but enabled keys
        self.ENABLE_1_TO_4_LAYOUT = False
        self.memory_monitor = MemoryMonitor(self)
       
        self.memory_monitor.check_pressure.connect(self.handle_memory_pressure)
        self.memory_monitor.unload_request.connect(self._perform_unload_on_main)
        self._last_tab_index = -1 # For on_tab_changed
        self._individual_tab_connected = False # For setup_individual_tab_loading_signal
        self._is_reloading_band = False
        self._is_closing = False
        self._load_generation = 0
        self.tab_mem_per_type = {'binned': None, 'unbinned': None, 'pan': None}
        self._progress_session_active = False
        self._progress_last_value = 0
       
       
        self.matrix_size_var = QSpinBox()
        self.matrix_size_var.setRange(3, 9)
        self.matrix_size_var.setSingleStep(2)
        self.matrix_size_var.setValue(3)
       
        self.width_entry = QLineEdit("8448")
        self.height_entry = QLineEdit("384")
        self.raw_height = 384
        self.tdi_stage = 0
        self.bitdepth_var = QComboBox()
        # Support 8, 10, 12 and 16-bit sources
        self.bitdepth_var.addItems(["8", "10", "12", "16"])
        self.bitdepth_var.setCurrentIndex(1)
       
        self.gap_var = QSpinBox()
        self.gap_var.setRange(0, 50)
        self.gap_var.setValue(0)
       
        self.contrast_enhance_var = QCheckBox("Contrast Enhancement")
        self.contrast_enhance_var.setChecked(False)
        self.contrast_min_var = QDoubleSpinBox()
        self.contrast_min_var.setRange(0, 1023)
        self.contrast_min_var.setValue(0)
        self.contrast_min_var.setMaximumWidth(80)
        self.contrast_max_var = QDoubleSpinBox()
        self.contrast_max_var.setRange(0, 1023)
        self.contrast_max_var.setValue(1023)
        self.contrast_max_var.setMaximumWidth(80)
        self.contrast_enhance_var.stateChanged.connect(self._invalidate_cache)
        self.bitdepth_var.currentTextChanged.connect(lambda _: self._sync_contrast_range_to_bitdepth(clamp_only=True))
        self.frame_mode_var = QButtonGroup()
        self.frame_mode_single = QRadioButton("Single Frame")
        self.frame_mode_single.setChecked(True)
        self.frame_mode_range = QRadioButton("Frame Range")
        self.frame_mode_var.addButton(self.frame_mode_single, 0)
        self.frame_mode_var.addButton(self.frame_mode_range, 1)
       
        self.rgb_frame_mode_var = QButtonGroup()
        self.rgb_frame_mode_single = QRadioButton("Selected Frame")
        self.rgb_frame_mode_single.setChecked(True)
        self.rgb_frame_mode_all = QRadioButton("All Frames")
        self.rgb_frame_mode_var.addButton(self.rgb_frame_mode_single, 0)
        self.rgb_frame_mode_var.addButton(self.rgb_frame_mode_all, 1)
       
        self.fit_mode_var = QButtonGroup()
        self.fit_mode_screen = QRadioButton("Fit to Screen")
        self.fit_mode_actual = QRadioButton("Actual Size")
        self.fit_mode_actual.setChecked(True)
        self.fit_mode_var.addButton(self.fit_mode_screen, 0)
        self.fit_mode_var.addButton(self.fit_mode_actual, 1)
       
        self.start_frame_entry = QSpinBox()
        self.start_frame_entry.setRange(1, 1000)
        self.start_frame_entry.setValue(1)
        self.end_frame_entry = QSpinBox()
        self.end_frame_entry.setRange(1, 1000)
        self.end_frame_entry.setValue(1)
        self.view_cache = {} # {tab_name: {'frame_index': int, 'pil_image': Image, 'raw_data': np.array, 'hash': str}}
        self.param_hash = self._compute_param_hash() # Initial hash of offsets, contrast, etc.
        # Shared across all individual band viewers
        self.shared_mouse_zoom_enabled = False # For mouse wheel zoom
        self.shared_magnifier_enabled = False
        self.shared_magnifier_zoom = 8.0 # Default zoom level
        self.shared_magnifier_radius = 100
        self.shared_magnifier_torch = False
        self.shared_flip_vertical = False # Global flip states (apply to all on toggle)
        self.shared_flip_horizontal = False
        # Per-band state preservation (zoom, scroll, etc.)
        self.viewer_states = {} # {band_key: {'zoom': float, 'rotation': float, 'scroll_x': int, 'scroll_y': int, ...}}
        self._individual_bands_built_hash = None  # param hash when individual band tabs were last built
        # Shared controls for individual bands (add to UI in init_ui)
        self.shared_mouse_zoom_cb = QCheckBox("Mouse Zoom (All Tabs)")
        self.shared_mouse_zoom_cb.stateChanged.connect(self._sync_shared_mouse_zoom)
        self.shared_magnifier_toggle = QToolButton()
        self.shared_magnifier_toggle.setText("🔍") # Or icon
        self.shared_magnifier_toggle.setToolTip("Toggle magnifier for all tabs")
        self.shared_magnifier_toggle.clicked.connect(self._toggle_shared_magnifier)
        self.shared_magnifier_zoom_slider = QSlider(Qt.Horizontal)
        self.shared_magnifier_zoom_slider.setRange(10, 100) # 1.0 to 10.0 *10
        self.shared_magnifier_zoom_slider.setValue(int(self.shared_magnifier_zoom * 10))
        self.shared_magnifier_zoom_slider.valueChanged.connect(self._sync_shared_magnifier_zoom)
        self.shared_flip_v_btn = QPushButton("Flip V (All)")
        self.shared_flip_v_btn.setToolTip("Flip all open viewers vertically")
        self.shared_flip_v_btn.clicked.connect(lambda: self._apply_shared_flip(vertical=True))
        self.shared_flip_h_btn = QPushButton("Flip H (All)")
        self.shared_flip_h_btn.setToolTip("Flip all open viewers horizontally")
        self.shared_flip_h_btn.clicked.connect(lambda: self._apply_shared_flip(vertical=False))
       
        # Keyboard shortcuts
        QShortcut(QKeySequence("Shift+N"), self, self.main_app.add_new_tab)
        QShortcut(QKeySequence("Shift+Q"), self, lambda: self.main_app.close_tab(self.main_app.tab_widget.indexOf(self)))
        QShortcut(QKeySequence("Shift+Return"), self, self.select_folder)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_parameters)
        QShortcut(QKeySequence("Tab"), self, self.cycle_view_tabs_forward)
        QShortcut(QKeySequence("Shift+Tab"), self, self.cycle_view_tabs_backward)
        QShortcut(QKeySequence("Right"), self, lambda: self.change_frame(1))
        QShortcut(QKeySequence("Left"), self, lambda: self.change_frame(-1))
        QShortcut(QKeySequence("Ctrl+Up"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+Down"), self, self.zoom_out)
        QShortcut(QKeySequence("Space"), self, self.toggle_play)
        QShortcut(QKeySequence("Return"), self, self.update_views)
        QShortcut(QKeySequence("Ctrl+Return"), self, self.apply_contrast_enhancement)
        QShortcut(QKeySequence("Ctrl+Space"), self, self.export_current_image)
        #QShortcut(QKeySequence("F"), self, activated=lambda: self.toggle_fullscreen(self.image_viewer))
        self.init_ui()

    def _qt_alive(self, obj) -> bool:
        if obj is None:
            return False
        try:
            if sip is not None and hasattr(sip, "isdeleted") and sip.isdeleted(obj):
                return False
        except Exception:
            return False
        try:
            obj.parent()
            return True
        except RuntimeError:
            return False
        except Exception:
            return False

    def _stop_thread(self, worker, wait_ms=1500):
        if worker is None:
            return
        try:
            worker.requestInterruption()
        except Exception:
            pass
        try:
            worker.quit()
        except Exception:
            pass
        try:
            if worker.isRunning() and not worker.wait(wait_ms):
                worker.terminate()
                worker.wait(1000)
        except Exception:
            pass
    def _perform_unload_on_main(self, key):
        """NEW: Slot to run unload on main thread (connected to memory_monitor.unload_request)."""
        print(f"[MAIN THREAD] Performing unload for {key}")
        # Individual band sub-tab or pan view: use per-band unload helper
        if key in getattr(self, 'band_frames', {}) or key == 'pan':
            self._unload_data_only(key) # Now safe: Runs on main thread
            # If the unloaded sub-tab is currently visible, trigger a refresh so
            # the user sees the loading placeholder next time they visit it.
            if hasattr(self, 'individual_bands_notebook'):
                for i in range(self.individual_bands_notebook.count()):
                    widget = self.individual_bands_notebook.widget(i)
                    if (hasattr(widget, 'key') and widget.key == key) or (key == 'pan' and getattr(widget, 'key', None) == 'pan'):
                        if self.individual_bands_notebook.currentIndex() == i:
                            QTimer.singleShot(100, lambda: self.refresh_current_individual_band(full_reload=True))
                        break
            # Track as unloaded so LRU logic skips it until reloaded
            self.unloaded_keys.add(key)
            return
        # View-level keys (display modes): free large images/caches for that mode
        if key == 'all_bands':
            try:
                widget = getattr(self, 'all_bands_tab', None)
            except Exception:
                widget = None
            # Clears All Bands viewer images and related state
            self.unload_view_widget(widget, "All Bands")
            self.unloaded_keys.add(key)
            return
        if key in ('rgb_fusion', 'rgb'):
            try:
                widget = getattr(self, 'fusion_tab', None)
            except Exception:
                widget = None
            # Clears RGB Fusion preview and releases its images
            self.unload_view_widget(widget, "RGB Fusion")
            self.unloaded_keys.add(key)
            return
        if key == 'histogram':
            try:
                widget = getattr(self, 'histogram_tab', None)
            except Exception:
                widget = None
            # Clears histogram worker and plot (cheap but frees worker buffers)
            self.unload_view_widget(widget, "Histogram")
            self.unloaded_keys.add(key)
            return
   
    def _sync_shared_mouse_zoom(self, state):
        self.shared_mouse_zoom_enabled = bool(state)
        # Apply to all loaded viewers
        for key in self.viewer_states:
            viewer = self._get_viewer_for_key(key)
            if viewer and hasattr(viewer.graphics_view, 'mouse_zoom_enabled'):
                viewer.graphics_view.mouse_zoom_enabled = self.shared_mouse_zoom_enabled # Assuming you add this attr to GraphicsImageViewer
    def _toggle_shared_magnifier(self):
        self.shared_magnifier_enabled = not self.shared_magnifier_enabled
        # Apply to all
        for key in self.viewer_states:
            viewer = self._get_viewer_for_key(key)
            if viewer:
                viewer.graphics_view.toggle_magnifier(self.shared_magnifier_enabled)
                viewer.graphics_view.magnifier_zoom = self.shared_magnifier_zoom
                viewer.graphics_view.magnifier_radius = self.shared_magnifier_radius
                viewer.graphics_view.torch_enabled = self.shared_magnifier_torch
    def _sync_shared_magnifier_zoom(self, value):
        self.shared_magnifier_zoom = value / 10.0
        # Apply to all
        for key in self.viewer_states:
            viewer = self._get_viewer_for_key(key)
            if viewer:
                viewer.graphics_view.set_magnifier_zoom(value)
    def _apply_shared_flip(self, vertical=False, horizontal=False, rot180=False):
        if rot180:
            self.shared_flip_vertical = not self.shared_flip_vertical
            self.shared_flip_horizontal = not self.shared_flip_horizontal
        else:
            if vertical:
                self.shared_flip_vertical = not self.shared_flip_vertical
            if horizontal:
                self.shared_flip_horizontal = not self.shared_flip_horizontal
        for key in self.band_frames:
            self.flip_flags[key] = {'vertical': self.shared_flip_vertical, 'horizontal': self.shared_flip_horizontal}
            if key in self.viewer_states:
                self._flip_viewer_image(key, self.shared_flip_vertical, self.shared_flip_horizontal)
        self.update_views()

    def open_editor_tab(self, source_viewer):
        """Create and open a new Editor tab based on the source viewer's image."""
        if not source_viewer.current_pil_image:
            QMessageBox.warning(self, "No Image", "No image loaded in the source viewer.")
            return
        if EditorTab is None:
            QMessageBox.warning(self, "Error", "EditorTab module not available.")
            return
        editor = EditorTab(source_viewer, self)
        if editor.original_array is None:  # Early return if init failed
            return
        tab_name = f"Editor – {self.view_tabs.tabText(self.view_tabs.currentIndex())}"
        idx = self.view_tabs.addTab(editor, tab_name)
        self.view_tabs.setCurrentIndex(idx)
        # Optional: Set custom close button
        try:
            self._set_custom_close_button(idx)
        except Exception:
            pass
   
    def setup_tab_connections(self):
        self.view_tabs.currentChanged.connect(self.on_tab_changed)
        # Consistently use on_individual_tab_changed for all sub-tab logic
        if hasattr(self, 'individual_bands_notebook'):
            self.individual_bands_notebook.currentChanged.connect(self.on_individual_tab_changed)
    def _effective_height_for_stage(self, raw_height=None, tdi_stage=None):
        try:
            raw = int(self.raw_height if raw_height is None else raw_height)
        except Exception:
            raw = 384
        stage = normalize_tdi_stage(self.tdi_stage if tdi_stage is None else tdi_stage)
        return raw if stage <= 0 else max(1, raw // stage)
    def _apply_image_params(self, width, raw_height, bit_depth, tdi_stage):
        try:
            width_i = int(width)
        except Exception:
            width_i = 8448
        try:
            raw_height_i = int(raw_height)
        except Exception:
            raw_height_i = 384
        try:
            bit_depth_i = int(bit_depth)
        except Exception:
            bit_depth_i = 10
        tdi_stage_i = normalize_tdi_stage(tdi_stage)
        self.raw_height = raw_height_i
        self.tdi_stage = tdi_stage_i
        self.width_entry.setText(str(width_i))
        self.height_entry.setText(str(self._effective_height_for_stage(raw_height_i, tdi_stage_i)))
        self.bitdepth_var.setCurrentText(str(bit_depth_i))
        self._sync_contrast_range_to_bitdepth(clamp_only=True)

    def _build_parameter_dialog(self, folder=None):
        inferred = {}
        if folder:
            try:
                inferred = infer_dataset_image_params(folder)
            except Exception:
                inferred = {}
        return ParameterDialog(self, dataset_params=inferred)
    def save_state(self):
        band_enabled_states = {}
        for key, cb in self.band_enabled.items():
            try:
                band_enabled_states[key] = cb.isChecked()
            except Exception:
                band_enabled_states[key] = True # Default if error
        # Sanitize viewer_states to JSON-serializable primitives
        safe_viewer_states = {}
        try:
            for k, v in (self.viewer_states or {}).items():
                if not isinstance(v, dict):
                    continue
                safe = {}
                if 'zoom' in v:
                    try:
                        safe['zoom'] = float(v.get('zoom', 1.0))
                    except Exception:
                        safe['zoom'] = 1.0
                if 'rotation' in v:
                    try:
                        safe['rotation'] = float(v.get('rotation', 0.0))
                    except Exception:
                        safe['rotation'] = 0.0
                if 'scroll_x' in v:
                    try:
                        safe['scroll_x'] = int(v.get('scroll_x', 0))
                    except Exception:
                        safe['scroll_x'] = 0
                if 'scroll_y' in v:
                    try:
                        safe['scroll_y'] = int(v.get('scroll_y', 0))
                    except Exception:
                        safe['scroll_y'] = 0
                if safe:
                    safe_viewer_states[k] = safe
        except Exception:
            safe_viewer_states = {}

        return {
            'folder': self.current_folder,
            'bitdepth': self.bitdepth,
            'width': self.width_entry.text(),
            'height': self.height_entry.text(),
            'raw_height': int(getattr(self, 'raw_height', 384) or 384),
            'tdi_stage': int(getattr(self, 'tdi_stage', 0) or 0),
            'band_offsets': self.band_offsets,
            'rgb_bands': self.rgb_bands,
            'gap': self.gap_var.value(),
            'contrast_enhance': self.contrast_enhance_var.isChecked(),
            'contrast_min': self.contrast_min_var.value(),
            'contrast_max': self.contrast_max_var.value(),
            'frame_mode': self.frame_mode_var.checkedId(),
            'rgb_frame_mode': self.rgb_frame_mode_var.checkedId(),
            'fit_mode': self.fit_mode_var.checkedId(),
            'start_frame': self.start_frame_entry.value(),
            'end_frame': self.end_frame_entry.value(),
            'current_frame': self.current_frame_index,
            'matrix_size': getattr(self, 'matrix_size_var', None) and int(self.matrix_size_var.value()),
            'viewer_states': safe_viewer_states,
            'flip_flags': self.flip_flags,
            'bands_info': getattr(self, 'bands_info', {}),
            'band_enabled': band_enabled_states,
            # Add more states as needed (e.g., viewer_states if persistent)
        }
    def load_state(self, data):
        try:
            self.current_folder = data.get('folder')
            saved_band_enabled = data.get('band_enabled', {})
            # store for use when async loader finishes
            try:
                self._saved_band_enabled = dict(saved_band_enabled)
            except Exception:
                self._saved_band_enabled = {}
            if self.current_folder:
                self.base_name = os.path.basename(self.current_folder)
                if hasattr(self, 'folder_label'):
                    self.folder_label.setText(self.base_name)
                if self.main_app:
                    self.main_app.update_tab_name(self, self.base_name)
                
                # Notify Iris of the restored folder
                self._notify_iris_folder_loaded(self.current_folder)

                # Wrap folder loading in try-except to isolate errors
                try:
                    # FIXED: Use correct method name (load_folder_data, not load_bands_from_folder)
                    self.load_folder_data()
                except AttributeError as ae:
                    print(f"AttributeError in load_folder_data: {ae}. Skipping band load for safety.")
        except Exception as e:
            print(f"Error loading folder in load_state: {e}")
        # Safely set params with hasattr checks
        self.bitdepth = data.get('bitdepth', 10)
        saved_width = data.get('width', '8448')
        saved_tdi_stage = int(data.get('tdi_stage', 0) or 0)
        saved_height = data.get('height', '384')
        saved_raw_height = data.get('raw_height')
        if saved_raw_height is None:
            try:
                saved_raw_height = int(saved_height) * saved_tdi_stage if saved_tdi_stage > 0 else int(saved_height)
            except Exception:
                saved_raw_height = 384
        self._apply_image_params(saved_width, saved_raw_height, self.bitdepth, saved_tdi_stage)
        self.band_offsets = data.get('band_offsets', {f"b{i}": {"x": 0, "y": 0} for i in range(7)})
        self.rgb_bands = data.get('rgb_bands', {"R": "b0", "G": "b1", "B": "b2"})
        # Safe RGB band UI updates (after load_folder_data populates avail_keys)
        if hasattr(self, 'red_band_var') and self.red_band_var.count() > 0:
            self.red_band_var.setCurrentText(self.rgb_bands["R"])
        if hasattr(self, 'green_band_var') and self.green_band_var.count() > 0:
            self.green_band_var.setCurrentText(self.rgb_bands["G"])
        if hasattr(self, 'blue_band_var') and self.blue_band_var.count() > 0:
            self.blue_band_var.setCurrentText(self.rgb_bands["B"])
        if hasattr(self, 'gap_var'):
            self.gap_var.setValue(data.get('gap', 0))
        if hasattr(self, 'contrast_enhance_var'):
            self.contrast_enhance_var.setChecked(data.get('contrast_enhance', False))
        if hasattr(self, 'contrast_min_var'):
            self.contrast_min_var.setValue(data.get('contrast_min', 0))
        if hasattr(self, 'contrast_max_var'):
            self.contrast_max_var.setValue(data.get('contrast_max', self._current_max_dn()))
        self._normalize_legacy_contrast_limits()
        # Safe frame mode updates
        frame_mode_id = data.get('frame_mode', 0)
        if hasattr(self, 'frame_mode_var') and self.frame_mode_var.button(frame_mode_id):
            self.frame_mode_var.button(frame_mode_id).setChecked(True)
        rgb_frame_mode_id = data.get('rgb_frame_mode', 0)
        if hasattr(self, 'rgb_frame_mode_var') and self.rgb_frame_mode_var.button(rgb_frame_mode_id):
            self.rgb_frame_mode_var.button(rgb_frame_mode_id).setChecked(True)
        fit_mode_id = data.get('fit_mode', 1)
        if hasattr(self, 'fit_mode_var') and self.fit_mode_var.button(fit_mode_id):
            self.fit_mode_var.button(fit_mode_id).setChecked(True)
        if hasattr(self, 'start_frame_entry'):
            self.start_frame_entry.setValue(data.get('start_frame', 1))
        if hasattr(self, 'end_frame_entry'):
            self.end_frame_entry.setValue(data.get('end_frame', 1))
        # matrix size (pixel info matrix)
        if hasattr(self, 'matrix_size_var') and 'matrix_size' in data:
            try:
                self.matrix_size_var.setValue(int(data.get('matrix_size', self.matrix_size_var.value())))
            except Exception:
                pass
        # Defer setting current_frame and frame range until after views are updated to ensure sliders/ranges are set
        deferred_current_frame = data.get('current_frame', None)
        try:
            self._deferred_start_frame = int(data.get('start_frame')) if 'start_frame' in data else None
        except Exception:
            self._deferred_start_frame = None
        try:
            self._deferred_end_frame = int(data.get('end_frame')) if 'end_frame' in data else None
        except Exception:
            self._deferred_end_frame = None
        self.flip_flags = data.get('flip_flags', {})
        self.bands_info = data.get('bands_info', {})
        # Clear and delete old checkboxes from UI layout (if any)
        if hasattr(self, 'band_checkbox_layout'):
            while self.band_checkbox_layout.count():
                item = self.band_checkbox_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        # Clear the band_enabled dict to release references to old checkboxes
        self.band_enabled = {}
        # Create new checkboxes for loaded bands (enabled by saved state or default True)
        if hasattr(self, 'band_frames') and self.band_frames:
            self.band_enabled = {key: QCheckBox() for key in self.band_frames}
            for key, cb in sorted(self.band_enabled.items(), key=lambda x: x[0]):
                cb.setChecked(saved_band_enabled.get(key, True))
                cb.setText(key)
                cb.stateChanged.connect(lambda state, k=key: self.toggle_band(k, state))
                if hasattr(self, 'band_checkbox_layout'):
                    self.band_checkbox_layout.addWidget(cb)
        # Safe refresh: update frame controls first, then views (which will populate individual tabs based on enabled)
        try:
            self.update_frame_controls()
        except AttributeError as ae:
            print(f"AttributeError in update_frame_controls: {ae}. Skipping frame controls update.")
        except Exception as e:
            print(f"Error in update_frame_controls: {e}")
        try:
            self.update_views(full_refresh=True)
        except AttributeError as ae:
            print(f"AttributeError in update_views: {ae}. Skipping view refresh.")
        except Exception as e:
            print(f"Error in update_views: {e}")
        # restore viewer_states if present
        try:
            vs = data.get('viewer_states') or {}
            if isinstance(vs, dict):
                # Coerce values to safe primitives
                coerced = {}
                for k, v in vs.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        coerced[k] = {
                            'zoom': float(v.get('zoom', 1.0)),
                            'rotation': float(v.get('rotation', 0.0)),
                            'scroll_x': int(v.get('scroll_x', 0)),
                            'scroll_y': int(v.get('scroll_y', 0)),
                        }
                    except Exception:
                        continue
                self.viewer_states = coerced
        except Exception:
            pass
        # Now set the current frame if provided
        try:
            if deferred_current_frame is not None:
                self.current_frame_index = int(deferred_current_frame)
                if hasattr(self, 'frame_slider'):
                    try:
                        self.frame_slider.setValue(self.current_frame_index)
                    except Exception:
                        pass
        except Exception:
            pass
        # Apply fit mode if needed
        try:
            if self.fit_mode_var.checkedId() == 0:
                self.fit_to_screen()
        except Exception:
            pass
        print("load_state completed successfully (with safeguards).")
        #QMessageBox.information(self, "Session Restored", "Last session restored successfully.")
    def clear_state(self):
        self.band_frames = {}
        self.current_folder = None
        self.base_name = ""
        self.current_frame_index = 0
        self.flip_flags = {}
        self.playing = False
        self.playback_mode = False
        self.last_play_frame_time = None
        self.view_cache.clear()
        self.loaded_band_memories.clear()
        self.param_hash = self._compute_param_hash()
        self.viewer_states.clear()
        # Clear band checkboxes
        while self.band_checkbox_layout.count():
            item = self.band_checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.band_enabled = {}
        # Clear individual bands notebook tabs
        while self.individual_bands_notebook.count():
            widget = self.individual_bands_notebook.widget(0)
            self.individual_bands_notebook.removeTab(0)
            if widget:
                widget.deleteLater()
        # Reset frame controls
        self.frame_slider.setRange(0, 0)
        self.frame_label.setText("0/0")
        self.start_frame_entry.setValue(1)
        self.end_frame_entry.setValue(1)
        # Reset play button
        try:
            self.play_btn.setText("▶ Play")
        except Exception:
            pass
        # Clear viewers
        try:
            self.all_bands_viewer.show_image(None)
        except Exception:
            pass
        try:
            self.rgb_preview_viewer.show_image(None)
        except Exception:
            pass
        try:
            self.histogram_viewer.clear()
        except Exception:
            pass
        # Reset RGB selectors
        self.red_band_var.clear()
        self.green_band_var.clear()
        self.blue_band_var.clear()
        # Reset other params if needed
        self.bitdepth_var.setCurrentIndex(1)
        self._sync_contrast_range_to_bitdepth(clamp_only=False)
        self.gap_var.setValue(0)
        self.contrast_enhance_var.setChecked(False)
        self.contrast_min_var.setValue(0)
        self.contrast_max_var.setValue(self._current_max_dn())
        print("State cleared successfully.")
    def _get_viewer_for_key(self, key):
        # Helper: Find GraphicsImageViewer for a band key (search notebook)
        for i in range(self.individual_bands_notebook.count()):
            widget = self.individual_bands_notebook.widget(i)
            if hasattr(widget, 'key') and widget.key == key:
                return widget.findChild(GraphicsImageViewer)
        return None
    def _flip_viewer_image(self, key, vertical, horizontal=False):
        # Apply flip to specific viewer's image (call apply_flip on it)
        viewer = self._get_viewer_for_key(key)
        if viewer:
            viewer.apply_flip(vertical=vertical, all=False, click_pos=None) # Or adapt as needed
    def _height_from_key(self, k, default=None, bin_factor=1):
        try:
            frames = self.band_frames.get(k)
            if frames is None:
                return int(max(1, round(float(default if default is not None else 384) / float(bin_factor))))
            if isinstance(frames, list) and len(frames) > 0:
                h = int(getattr(frames[0], "shape", (None,))[0] or frames[0].shape[0])
            elif hasattr(frames, "h"):
                h = int(getattr(frames, "h"))
            else:
                h = int(round(float(default if default is not None else 384)))
            return int(max(1, h))
        except Exception:
            return int(max(1, round(float(default if default is not None else 384) / float(bin_factor))))
    def _compute_param_hash(self):
        # Serialize checkbox states instead of objects
        enabled_states = {k: v.isChecked() for k, v in self.band_enabled.items()}
        offset_states = {k: {'x': v['x'], 'y': v['y']} for k, v in self.band_offsets.items()}
       
        param_str = (
            f"{json.dumps(enabled_states)}" \
            f"{json.dumps(offset_states)}" \
            f"{self.gap_var.value()}" \
            f"{self.contrast_enhance_var.isChecked()}" \
            f"{getattr(self, 'ENABLE_1_TO_4_LAYOUT', False)}" \
            f"{self.start_frame_entry.value()}" \
            f"{self.end_frame_entry.value()}" \
            f"{self.frame_mode_var.checkedId()}" \
            f"{self.rgb_frame_mode_var.checkedId()}" \
            f"{self.rgb_bands}" \
            f"{self.fit_mode_var.checkedId()}"
        )
        # Note: Excluding current_frame_index from hash prevents rebuilding ALL sub-tabs
        # every time the frame changes. Sub-tabs handle their own frame updates via lazy loading.
        return hashlib.sha256(param_str.encode()).hexdigest()
    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(main_layout)
       
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)
        self.left_panel.setLayout(left_layout)
        self.left_scroll.setWidget(self.left_panel)
        self.left_scroll.setMinimumWidth(420)
        self.left_scroll.setMaximumWidth(560)
        self.left_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(self.left_scroll)
        views_container = QWidget()
        views_layout = QVBoxLayout()
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(4)
        views_container.setLayout(views_layout)
        self.views_toggle = QToolButton()
        self.views_toggle.setText("Display Modes")
        self.views_toggle.setToolTip("Show/hide display modes")
        self.views_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.views_toggle.setArrowType(Qt.DownArrow)
        self.views_toggle.setCheckable(True)
        self.views_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.views_toggle.setMinimumHeight(24)
        self.views_toggle.setStyleSheet("QToolButton { text-align: left; padding-left: 8px; font-weight: 600; }")
        views_layout.addWidget(self.views_toggle)
        # Content area to show/hide (start collapsed)
        self.views_content = QWidget()
        self.views_content_layout = QVBoxLayout()
        self.views_content_layout.setContentsMargins(8, 4, 4, 4)
        self.views_content_layout.setSpacing(6)
        self.views_content.setLayout(self.views_content_layout)
        views_layout.addWidget(self.views_content)
        self.views_content.setVisible(True)
        def _on_views_toggled(checked):
            self.views_content.setVisible(checked)
            self.views_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.views_toggle.toggled.connect(_on_views_toggled)
        # Helper: mapping name -> attribute that holds the widget we want to (re)use
        self._view_widget_map = {
            "All Bands": "all_bands_viewer",
            "Individual Bands": "individual_bands_tab", # top-level tab widget for individual bands
            "RGB Fusion": "fusion_tab",
            "Histogram": "histogram_tab"
        }
        # Create the checkboxes (DEFAULT: OFF)
        self.view_checkboxes = {}
        for name in ["All Bands", "Individual Bands", "RGB Fusion", "Histogram"]:
            cb = QCheckBox(name)
            cb.setChecked(False) # <-- default OFF
            cb.stateChanged.connect(lambda s, n=name: self._on_view_checkbox_toggled(n, s == Qt.Checked))
            self.views_content_layout.addWidget(cb)
            self.view_checkboxes[name] = cb
        self.views_content_layout.addStretch()
        # Try to insert above Frame Control if that container exists; otherwise append to left_layout
        inserted = False
        try:
            if hasattr(self, 'frame_control_container'):
                # search left_layout for the frame_control_container index and insert before it
                for i in range(left_layout.count()):
                    w = left_layout.itemAt(i).widget()
                    if w is self.frame_control_container:
                        left_layout.insertWidget(i, views_container)
                        inserted = True
                        break
        except Exception:
            inserted = False
        if not inserted:
            left_layout.addWidget(views_container)
       
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        left_layout.addWidget(self.folder_label)
       
        hb = QHBoxLayout()
        select_btn = QPushButton("Select Folder & Stitch")
        select_btn.setToolTip("Load and stitch folder")
        select_btn.clicked.connect(self.select_folder)
        hb.addWidget(select_btn)
        self.load_menu_btn = QToolButton()
        self.load_menu_btn.setArrowType(Qt.DownArrow)
        self.load_menu_btn.setMaximumWidth(22)
        self.load_menu_btn.setToolTip("Open recent band folders")
        self.load_menu_btn.clicked.connect(self._show_recent_menu)
        hb.addWidget(self.load_menu_btn)
        left_layout.addLayout(hb)
       
        frame_group = QGroupBox("Frame Controls")
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(8, 20, 8, 8) # (left, top, right, bottom)
        frame_group.setLayout(frame_layout)
        left_layout.addWidget(frame_group)
       
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        frame_layout.addWidget(self.frame_slider)
       
        self.frame_label = QLabel("0/0")
        frame_layout.addWidget(self.frame_label)
       
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setToolTip("Play or pause frames")
        self.play_btn.clicked.connect(self.toggle_play)
        playback_layout.addWidget(self.play_btn)
       
        prev_btn = QPushButton("◀")
        prev_btn.setToolTip("Previous frame")
        prev_btn.clicked.connect(lambda: self.change_frame(-1))
        playback_layout.addWidget(prev_btn)
       
        next_btn = QPushButton("▶")
        next_btn.setToolTip("Next frame")
        next_btn.clicked.connect(lambda: self.change_frame(1))
        playback_layout.addWidget(next_btn)
        add_video_btn = QPushButton("Video Mode")
        add_video_btn.setToolTip("Open Video Mode tab")
        add_video_btn.clicked.connect(lambda: self.main_app.add_video_tab(
            folder=self.current_folder,
            width=int(self.width_entry.text()) if self.width_entry.text().isdigit() else None,
            height=int(self.height_entry.text()) if self.height_entry.text().isdigit() else None,
            bitdepth=int(self.bitdepth_var.currentText()) if self.bitdepth_var.currentText().isdigit() else None
        ) if self.main_app else None)
        playback_layout.addWidget(add_video_btn)
               
        frame_layout.addLayout(playback_layout)
       
        frame_row_layout = QHBoxLayout()
        frame_row_layout.setContentsMargins(0, 0, 0, 0)
        frame_row_layout.setSpacing(10)
        frame_mode_layout = QHBoxLayout()
        frame_mode_layout.addWidget(self.frame_mode_single)
        frame_mode_layout.addWidget(self.frame_mode_range)
        frame_layout.addLayout(frame_mode_layout)
       
        frame_range_layout = QHBoxLayout()
        frame_range_layout.addWidget(QLabel("Start:"))
        frame_range_layout.addWidget(self.start_frame_entry)
        frame_range_layout.addWidget(QLabel("End:"))
        frame_range_layout.addWidget(self.end_frame_entry)
        frame_layout.addLayout(frame_range_layout)
       
        # ---- Collapsible "Band Offsets" section (triangle dropdown) ----
        offset_container = QWidget()
        offset_container_layout = QVBoxLayout()
        offset_container_layout.setContentsMargins(0, 0, 0, 0)
        offset_container_layout.setSpacing(4)
        offset_container.setLayout(offset_container_layout)
        left_layout.addWidget(offset_container)
        self.offset_toggle = QToolButton()
        self.offset_toggle.setText("Band Offsets")
        self.offset_toggle.setToolTip("Show/hide band offsets")
        self.offset_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.offset_toggle.setArrowType(Qt.RightArrow) # Down when expanded
        self.offset_toggle.setCheckable(True)
        self.offset_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.offset_toggle.setMinimumHeight(24) # tweak as needed
        self.offset_toggle.setStyleSheet(
            "QToolButton { text-align: left; padding-left: 8px; font-weight: 600; }"
        )
        offset_container_layout.addWidget(self.offset_toggle)
        # Content area that will be shown/hide
        self.offset_content = QWidget()
        self.offset_content_layout = QVBoxLayout()
        self.offset_content_layout.setContentsMargins(8, 4, 4, 4)
        self.offset_content_layout.setSpacing(6)
        self.offset_content.setLayout(self.offset_content_layout)
        offset_container_layout.addWidget(self.offset_content)
        # Wire toggle to show/hide content and switch arrow direction
        def _on_offset_toggled(checked):
            self.offset_content.setVisible(checked)
            self.offset_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.offset_toggle.toggled.connect(_on_offset_toggled)
        # Build the per-band rows inside offset_content_layout (same controls as before)
        self.offset_spins = {}
        # ensure band_offsets exists
        if not hasattr(self, 'band_offsets') or self.band_offsets is None:
            self.band_offsets = {f"b{i}": {"x": 0, "y": 0} for i in range(7)}
        for i in range(7):
            band_key = f"b{i}"
            band_layout = QHBoxLayout()
            band_layout.setSpacing(6)
            band_label = QLabel(f"Band {i}")
            band_label.setFixedWidth(60)
            band_layout.addWidget(band_label)
            x_label = QLabel("X:")
            band_layout.addWidget(x_label)
            spin_x = QSpinBox()
            spin_x.setRange(-10000, 10000)
            spin_x.setValue(self.band_offsets.get(band_key, {}).get("x", 0))
            spin_x.valueChanged.connect(lambda v, idx=i: self.update_offset_value(idx, 'x', v))
            self.offset_spins[f"{band_key}_x"] = spin_x
            band_layout.addWidget(spin_x)
            y_label = QLabel("Y:")
            band_layout.addWidget(y_label)
            spin_y = QSpinBox()
            spin_y.setRange(-10000, 10000)
            spin_y.setValue(self.band_offsets.get(band_key, {}).get("y", 0))
            spin_y.valueChanged.connect(lambda v, idx=i: self.update_offset_value(idx, 'y', v))
            self.offset_spins[f"{band_key}_y"] = spin_y
            band_layout.addWidget(spin_y)
            band_layout.addStretch()
            self.offset_content_layout.addLayout(band_layout)
        self.offset_content.setVisible(self.offset_toggle.isChecked())
       
        param_layout = QVBoxLayout()
        left_layout.addLayout(param_layout)
       
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Band Gap:"))
        gap_layout.addWidget(self.gap_var)
        param_layout.addLayout(gap_layout)

        self.enable_1_to_4_layout_cb = QCheckBox("1:4 Layout")
        self.enable_1_to_4_layout_cb.setChecked(False)
        self.enable_1_to_4_layout_cb.stateChanged.connect(self._on_1_to_4_layout_toggled)
        param_layout.addWidget(self.enable_1_to_4_layout_cb)
       
               
        fit_layout = QHBoxLayout()
        fit_layout.addWidget(self.fit_mode_screen)
        fit_layout.addWidget(self.fit_mode_actual)
        param_layout.addLayout(fit_layout)
       
        contrast_layout = QVBoxLayout()
        contrast_top = QHBoxLayout()
        contrast_top.addWidget(self.contrast_enhance_var)
        contrast_top.addStretch()
        contrast_layout.addLayout(contrast_top)
        
        contrast_bottom = QHBoxLayout()
        contrast_bottom.setSpacing(4)
        contrast_bottom.addWidget(QLabel("Min:"))
        contrast_bottom.addWidget(self.contrast_min_var)
        contrast_bottom.addWidget(QLabel("Max:"))
        contrast_bottom.addWidget(self.contrast_max_var)
        auto_btn = QPushButton("Auto")
        auto_btn.setMaximumWidth(60)
        auto_btn.setToolTip("Auto contrast")
        auto_btn.clicked.connect(self.set_auto_contrast)
        contrast_bottom.addWidget(auto_btn)
        contrast_layout.addLayout(contrast_bottom)
        param_layout.addLayout(contrast_layout)
        # Measure checkbox moved to viewer bottom bar (per-view control)
       
        row1_layout = QHBoxLayout()
        save_btn = QPushButton("Save Progress")
        save_btn.setToolTip("Save current dataset settings")
        save_btn.clicked.connect(self.save_parameters)
        row1_layout.addWidget(save_btn)
        show_params_btn = QPushButton("Change Params")
        show_params_btn.setToolTip("Edit width/height/bit depth")
        show_params_btn.clicked.connect(self.show_params_popup)
        row1_layout.addWidget(show_params_btn)
        export_btn = QPushButton("Export Image")
        export_btn.setToolTip("Export currently displayed image")
        export_btn.clicked.connect(self.export_current_image)
        row1_layout.addWidget(export_btn)
        # Second row: Reload | Refresh
        row2_layout = QHBoxLayout()
        reload_btn = QPushButton("Reload")
        reload_btn.setToolTip("Reload folder data")
        reload_btn.clicked.connect(self.reload_folder_data)
        row2_layout.addWidget(reload_btn)
        refresh_tab_btn = QPushButton("Refresh")
        refresh_tab_btn.setToolTip("Refresh current tab")
        refresh_tab_btn.clicked.connect(self.refresh_current_tab)
        row2_layout.addWidget(refresh_tab_btn)
        param_layout.addLayout(row1_layout)
        param_layout.addLayout(row2_layout)
       
        left_layout.addStretch()
       
        self.display_frame = QWidget()
        display_layout = QVBoxLayout()
        self.display_frame.setLayout(display_layout)
        main_layout.addWidget(self.display_frame)
       
        self.view_tabs = QTabWidget()
        self.view_tabs.setTabBar(CustomTabBar()) # Use custom tab bar (disables default closable)
        self.view_tabs.setMovable(True)

        self.display_splitter = QSplitter(Qt.Vertical)
        self.display_splitter.setChildrenCollapsible(False)
        display_layout.addWidget(self.display_splitter, 1)

        self.display_splitter.addWidget(self.view_tabs)

        self.terminal_panel = QWidget(self)
        terminal_panel_layout = QVBoxLayout()
        terminal_panel_layout.setContentsMargins(0, 0, 0, 0)
        terminal_panel_layout.setSpacing(2)
        self.terminal_panel.setLayout(terminal_panel_layout)

        self.terminal_btn = QPushButton("Terminal ↑")
        self.terminal_btn.setToolTip("Show or hide terminal panel")
        self.terminal_btn.clicked.connect(self.toggle_terminal)
        terminal_panel_layout.addWidget(self.terminal_btn)

        self.terminal_widget = TerminalWidget(self)
        self.terminal_widget.setMaximumHeight(16777215)
        self.terminal_widget.hide()
        terminal_panel_layout.addWidget(self.terminal_widget, 1)
        self.display_splitter.addWidget(self.terminal_panel)
        self._terminal_button_height = max(28, self.terminal_btn.sizeHint().height())
        self.display_splitter.setSizes([1000, self._terminal_button_height])
        self.display_splitter.handle(1).setEnabled(False)
       
        progress_container = QWidget()
        progress_layout = QHBoxLayout()
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(8)
        progress_container.setLayout(progress_layout)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        progress_style = """
        QProgressBar {
            border: none;
            border-radius: 2px;
            height: 4px;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #00c6ff, stop:0.5 #0072ff, stop:1 #5856eb);
            border-radius: 2px;
        }
        """
        self.progress_bar.setStyleSheet(progress_style)
        progress_layout.addWidget(self.progress_bar)
        self.percent_label = QLabel("0%")
        self.percent_label.setFixedWidth(40)
        self.percent_label.setAlignment(Qt.AlignCenter)
        self.percent_label.setStyleSheet("""
            QLabel {
                color: #00d4ff;
                font-weight: bold;
                font-size: 8px;
                background-color: rgba(0, 0, 0, 0.7);
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 2px;
                min-width: 30px;
            }
        """)
        self.percent_label.hide()
        progress_layout.addWidget(self.percent_label)
        progress_container.setStyleSheet("background: transparent;")
        display_layout.addWidget(progress_container)
       
        self.init_view_tabs()
        self.setup_tab_connections()
        self.memory_monitor.start()
        try:
            for name, cb in getattr(self, 'view_checkboxes', {}).items():
                found = False
                for i in range(self.view_tabs.count()):
                    if self.view_tabs.tabText(i) == name:
                        found = True
                        break
                cb.blockSignals(True)
                cb.setChecked(found)
                cb.blockSignals(False)
        except Exception:
            pass
        self.pixel_info_box = PixelInfoBox(matrix_size_var=self.matrix_size_var)
        left_layout.addWidget(self.pixel_info_box)
        self.start_frame_entry.valueChanged.connect(self.validate_frame_range)
        self.end_frame_entry.valueChanged.connect(self.validate_frame_range)
       
        self.view_tabs.currentChanged.connect(self.on_tab_changed)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        self.frame_mode_var.buttonClicked.connect(self.update_views)
        self.rgb_frame_mode_var.buttonClicked.connect(self.preview_rgb_fusion)
   
    def init_view_tabs(self):
        # All Bands tab
        self.all_bands_tab = QWidget()
        all_bands_layout = QVBoxLayout()
        self.all_bands_tab.setLayout(all_bands_layout)
        self.all_bands_viewer = GraphicsImageViewer(
            parent=self,
            pixel_info_callback=self.update_pixel_info,
            matrix_size_var=self.matrix_size_var
        )
        all_bands_layout.addWidget(self.all_bands_viewer)
        #self.view_tabs.addTab(self.all_bands_tab, "All Bands")
        self._set_custom_close_button(self.view_tabs.count() - 1)
        # Individual Bands tab
        self.individual_bands_tab = QWidget()
        individual_bands_layout = QVBoxLayout()
        self.individual_bands_tab.setLayout(individual_bands_layout)
        self.band_checkbox_container = QWidget()
        self.band_checkbox_layout = QGridLayout(self.band_checkbox_container)
        self.band_checkbox_layout.setContentsMargins(6, 6, 6, 6)
        self.band_checkbox_layout.setHorizontalSpacing(2)
        self.band_checkbox_layout.setVerticalSpacing(2)
        individual_bands_layout.addWidget(self.band_checkbox_container)
        self.individual_bands_notebook = QTabWidget()
        # Consistently use setup_tab_connections for connecting up the notebook logic
        individual_bands_layout.addWidget(self.individual_bands_notebook)
        #self.view_tabs.addTab(self.individual_bands_tab, "Individual Bands")
        self._set_custom_close_button(self.view_tabs.count() - 1)
        # Histogram tab
        self.histogram_tab = QWidget()
        histogram_layout = QVBoxLayout()
        self.histogram_tab.setLayout(histogram_layout)
        self.histogram_viewer = HistogramViewer()
        # Ensure histogram knows the current bit depth for axis labeling
        try:
            self.histogram_viewer.set_bitdepth(self.bitdepth)
        except Exception:
            pass
        # If user toggles histogram mode, refresh views/histogram
        try:
            self.histogram_viewer.mode_changed.connect(self.update_views)
        except Exception:
            pass
        histogram_layout.addWidget(self.histogram_viewer)
        #self.view_tabs.addTab(self.histogram_tab, "Histogram")
        self._set_custom_close_button(self.view_tabs.count() - 1)
        # RGB Fusion tab: preview on top, channel mapping and offsets at bottom
        self.fusion_tab = QWidget()
        fusion_layout = QVBoxLayout()
        fusion_layout.setContentsMargins(6, 6, 6, 6)
        fusion_layout.setSpacing(8)
        self.fusion_tab.setLayout(fusion_layout)
        # Top: preview viewer (expands)
        self.rgb_preview_viewer = GraphicsImageViewer(
            parent=self,
            pixel_info_callback=self.update_pixel_info,
            matrix_size_var=self.matrix_size_var
        )
        self.preview_frame = QGroupBox("RGB Fusion Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(6, 6, 6, 6)
        self.preview_frame.setLayout(preview_layout)
        preview_layout.addWidget(self.rgb_preview_viewer)
        fusion_layout.addWidget(self.preview_frame, 1) # Stretch to expand
        # Bottom: controls panel (fixed height)
        controls_frame = QGroupBox("RGB Channel Configuration")
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(8, 6, 8, 6)
        controls_layout.setSpacing(12)
        controls_frame.setLayout(controls_layout)
        # Channel mapping and offsets (vertical rows now)
        map_widget = QWidget()
        map_layout = QVBoxLayout() # Changed to vertical for rows
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(6) # Reduced spacing for compactness
        map_widget.setLayout(map_layout)
        # Red row
        red_row = QHBoxLayout()
        red_row.addWidget(QLabel("Red:"))
        self.red_band_var = QComboBox()
        self.red_band_var.addItems([f"b{i}" for i in range(7)])
        self.red_band_var.setCurrentText("b0")
        red_row.addWidget(self.red_band_var)
        red_row.addWidget(QLabel("X offset:"))
        self.rgb_offset_r_x = QSpinBox()
        self.rgb_offset_r_x.setRange(-10000, 10000)
        self.rgb_offset_r_x.setValue(0)
        red_row.addWidget(self.rgb_offset_r_x)
        red_row.addWidget(QLabel("Y offset:"))
        self.rgb_offset_r_y = QSpinBox()
        self.rgb_offset_r_y.setRange(-10000, 10000)
        self.rgb_offset_r_y.setValue(0)
        red_row.addWidget(self.rgb_offset_r_y)
        map_layout.addLayout(red_row)
        # Green row
        green_row = QHBoxLayout()
        green_row.addWidget(QLabel("Green:"))
        self.green_band_var = QComboBox()
        self.green_band_var.addItems([f"b{i}" for i in range(7)])
        self.green_band_var.setCurrentText("b1")
        green_row.addWidget(self.green_band_var)
        green_row.addWidget(QLabel("X offset:"))
        self.rgb_offset_g_x = QSpinBox()
        self.rgb_offset_g_x.setRange(-10000, 10000)
        self.rgb_offset_g_x.setValue(0)
        green_row.addWidget(self.rgb_offset_g_x)
        green_row.addWidget(QLabel("Y offset:"))
        self.rgb_offset_g_y = QSpinBox()
        self.rgb_offset_g_y.setRange(-10000, 10000)
        self.rgb_offset_g_y.setValue(0)
        green_row.addWidget(self.rgb_offset_g_y)
        map_layout.addLayout(green_row)
        # Blue row
        blue_row = QHBoxLayout()
        blue_row.addWidget(QLabel("Blue:"))
        self.blue_band_var = QComboBox()
        self.blue_band_var.addItems([f"b{i}" for i in range(7)])
        self.blue_band_var.setCurrentText("b2")
        blue_row.addWidget(self.blue_band_var)
        blue_row.addWidget(QLabel("X offset:"))
        self.rgb_offset_b_x = QSpinBox()
        self.rgb_offset_b_x.setRange(-10000, 10000)
        self.rgb_offset_b_x.setValue(0)
        blue_row.addWidget(self.rgb_offset_b_x)
        blue_row.addWidget(QLabel("Y offset:"))
        self.rgb_offset_b_y = QSpinBox()
        self.rgb_offset_b_y.setRange(-10000, 10000)
        self.rgb_offset_b_y.setValue(0)
        blue_row.addWidget(self.rgb_offset_b_y)
        map_layout.addLayout(blue_row)
        controls_layout.addWidget(map_widget, 0)
        # Frame selection mode
        mode_widget = QWidget()
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(4)
        mode_widget.setLayout(mode_layout)
        mode_layout.addWidget(self.rgb_frame_mode_single)
        mode_layout.addWidget(self.rgb_frame_mode_all)
        controls_layout.addWidget(mode_widget, 0)
        # Spacer
        controls_layout.addStretch(1)
        # Preview button & auto-preview checkbox
        actions_widget = QWidget()
        actions_layout = QVBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(4)
        actions_widget.setLayout(actions_layout)
        preview_btn = QPushButton("Preview RGB Fusion")
        preview_btn.setToolTip("Preview RGB fusion")
        preview_btn.clicked.connect(self.preview_rgb_fusion)
        actions_layout.addWidget(preview_btn)
        self.auto_preview_rgb = QCheckBox("Auto preview")
        self.auto_preview_rgb.setChecked(True)
        actions_layout.addWidget(self.auto_preview_rgb)
        controls_layout.addWidget(actions_widget, 0)
        fusion_layout.addWidget(controls_frame, 0) # Controls at bottom (no stretch)
        #self.view_tabs.addTab(self.fusion_tab, "RGB Fusion")
        self._set_custom_close_button(self.view_tabs.count() - 1)
        # Auto-preview connections
        def _auto_preview_trigger():
            if getattr(self, "auto_preview_rgb", None) and self.auto_preview_rgb.isChecked():
                # Use QTimer to debounce rapid changes
                if hasattr(self, '_preview_timer'):
                    self._preview_timer.stop()
                self._preview_timer = QTimer()
                self._preview_timer.setSingleShot(True)
                self._preview_timer.timeout.connect(self.preview_rgb_fusion)
                self._preview_timer.start(300) # 300ms debounce
        # Connect channel selectors and offsets
        self.red_band_var.currentTextChanged.connect(_auto_preview_trigger)
        self.green_band_var.currentTextChanged.connect(_auto_preview_trigger)
        self.blue_band_var.currentTextChanged.connect(_auto_preview_trigger)
        self.rgb_frame_mode_single.toggled.connect(_auto_preview_trigger)
        self.rgb_offset_r_x.valueChanged.connect(_auto_preview_trigger)
        self.rgb_offset_r_y.valueChanged.connect(_auto_preview_trigger)
        self.rgb_offset_g_x.valueChanged.connect(_auto_preview_trigger)
        self.rgb_offset_g_y.valueChanged.connect(_auto_preview_trigger)
        self.rgb_offset_b_x.valueChanged.connect(_auto_preview_trigger)
        self.rgb_offset_b_y.valueChanged.connect(_auto_preview_trigger)
        # Help tab
        try:
            self.help_tab = create_help_tab(main_app=self, mode="band")
            self.view_tabs.addTab(self.help_tab, "Help")
            self._set_custom_close_button(self.view_tabs.count() - 1)
        except Exception as e:
            print(f"Failed to create help tab: {e}")
        self.view_tabs.setTabsClosable(True)
        self.view_tabs.tabCloseRequested.connect(self._on_view_tab_close)
  
    def cycle_view_tabs_forward(self):
        current = self.view_tabs.currentIndex()
        next_index = (current + 1) % self.view_tabs.count()
        self.view_tabs.setCurrentIndex(next_index)
    def cycle_view_tabs_backward(self):
        current = self.view_tabs.currentIndex()
        next_index = (current - 1) % self.view_tabs.count()
        self.view_tabs.setCurrentIndex(next_index)
    def _on_view_tab_close(self, index: int):
        try:
            name = self.view_tabs.tabText(index)
        except Exception:
            return
        self._remove_view_tab(name)
    def zoom_in(self):
        factor = 1.25
        self._zoom_current_viewer(factor)
    def zoom_out(self):
        factor = 0.8
        self._zoom_current_viewer(factor)
    def _zoom_current_viewer(self, factor):
        current_tab = self.view_tabs.currentIndex()
        tab_name = self.view_tabs.tabText(current_tab) if current_tab >= 0 else ""
        if tab_name == "All Bands":
            viewer = self.all_bands_viewer
        elif tab_name == "RGB Fusion":
            viewer = self.rgb_preview_viewer
        elif tab_name == "Individual Bands":
            current_band_tab = self.individual_bands_notebook.currentWidget()
            if current_band_tab:
                viewer = current_band_tab.findChild(GraphicsImageViewer)
            else:
                return
        else:
            return
        if viewer:
            old_zoom = viewer.zoom
            new_zoom = old_zoom * factor
            scroll_area = viewer.scroll_area
            viewport_center = scroll_area.viewport().rect().center()
            mouse_pos = viewport_center
            scroll_x = scroll_area.horizontalScrollBar().value()
            scroll_y = scroll_area.verticalScrollBar().value()
            mouse_image_x = (mouse_pos.x() + scroll_x) / old_zoom
            mouse_image_y = (mouse_pos.y() + scroll_y) / old_zoom
            new_scroll_x = mouse_image_x * new_zoom - mouse_pos.x()
            new_scroll_y = mouse_image_y * new_zoom - mouse_pos.y()
            viewer.zoom = new_zoom
            viewer.show_image(viewer.current_pil_image, fit_to_screen=False)
            scroll_area.horizontalScrollBar().setValue(int(new_scroll_x))
            scroll_area.verticalScrollBar().setValue(int(new_scroll_y))
   
    def toggle_measure_mode(self, state):
        enabled = state == Qt.Checked
        viewers = self.get_all_viewers()
        for viewer in viewers:
            # Set the underlying behaviour
            try:
                viewer.graphics_view.measure_enabled = enabled
            except Exception:
                pass

            # Update per-viewer UI checkbox if present (block signals to avoid recursion)
            try:
                if hasattr(viewer, 'graphics_view') and hasattr(viewer.graphics_view, 'set_interaction_mode'):
                    viewer.graphics_view.set_interaction_mode("measure" if enabled else "off")
                if hasattr(viewer, 'measure_mode_btn'):
                    viewer.measure_mode_btn.setText("Mode: Measure" if enabled else "Mode: Off")
                    viewer.measure_mode_btn.setStyleSheet(
                        "background-color: #4CAF50; color: white;" if enabled else "background-color: #E57373; color: white;"
                    )
            except Exception:
                pass

            try:
                parent = viewer
                # climb to widget that may own the overlay
                while parent is not None and not hasattr(parent, 'pixel_info_box_overlay'):
                    parent = parent.parent()
                if parent is not None and hasattr(parent, 'pixel_info_box_overlay') and parent.pixel_info_box_overlay is not None:
                    parent.pixel_info_box_overlay.set_interaction_mode("measure" if enabled else "off")
                    if not enabled:
                        parent.pixel_info_box_overlay.update_measurements(0, 0, 0)
            except Exception:
                pass

            # Clear any transient measurement points when disabling
            try:
                if not enabled:
                    viewer.graphics_view.measure_points = []
                    viewer.graphics_view.viewport().update()
            except Exception:
                pass

        # Update central pixel info box (app-level) as well
        try:
            if hasattr(self, 'pixel_info_box'):
                self.pixel_info_box.set_interaction_mode("measure" if enabled else "off")
                if not enabled:
                    self.pixel_info_box.update_measurements(0, 0, 0)
        except Exception:
            pass
   
    def get_all_viewers(self):
        viewers = []
        if hasattr(self, 'all_bands_viewer'):
            viewers.append(self.all_bands_viewer)
        if hasattr(self, 'rgb_preview_viewer'):
            viewers.append(self.rgb_preview_viewer)
        if hasattr(self, 'individual_bands_notebook'):
            for i in range(self.individual_bands_notebook.count()):
                w = self.individual_bands_notebook.widget(i)
                if w:
                    viewer = w.findChild(GraphicsImageViewer)
                    if viewer:
                        viewers.append(viewer)
        return viewers
    def toggle_contrast(self):
        self.contrast_enhance_var.setChecked(not self.contrast_enhance_var.isChecked())
        self.update_views()
    def _on_view_checkbox_toggled(self, name, checked):
        if checked:
            self._add_view_tab(name)
            # NEW: Handle if previously unloaded
            tab_key = name.lower().replace(' ', '_')
            if tab_key in self.unloaded_keys:
                self.unloaded_keys.discard(tab_key)
                QTimer.singleShot(0, lambda: self._reload_tab_data(tab_key))
        else:
            idx = -1
            for i in range(self.view_tabs.count()):
                if self.view_tabs.tabText(i) == name:
                    idx = i
                    break
            if idx >= 0:
                widget = self.view_tabs.widget(idx)
                self.unload_view_widget(widget, name)
                self.view_tabs.removeTab(idx)
                widget.deleteLater()
                gc.collect()
    def _add_view_tab(self, name):
        for i in range(self.view_tabs.count()):
            if self.view_tabs.tabText(i) == name:
                return
        attr = self._view_widget_map.get(name)
        widget = None
        if attr and hasattr(self, attr):
            widget = getattr(self, attr)
        if widget is not None:
            try:
                parent = widget.parent()
                if parent is not None:
                    widget.setParent(None)
            except RuntimeError:
                widget = None
        if widget is None:
            widget = QWidget()
            l = QVBoxLayout(widget)
            l.addWidget(QLabel(f"{name} (placeholder)"))
        try:
            idx = self.view_tabs.count()
            self.view_tabs.addTab(widget, name)
        except RuntimeError as e:
            print(f"[DEBUG] Failed to add tab '{name}': {e} - adding a fresh placeholder instead")
            widget = QWidget()
            l = QVBoxLayout(widget)
            l.addWidget(QLabel(f"{name} (placeholder - fallback)"))
            idx = self.view_tabs.count()
            self.view_tabs.addTab(widget, name)
        try:
            self._set_custom_close_button(idx)
        except Exception as e:
            print(f"[DEBUG] _set_custom_close_button failed for tab '{name}': {e}")
    def _remove_view_tab(self, name):
        for i in range(self.view_tabs.count()):
            if self.view_tabs.tabText(i) == name:
                widget = self.view_tabs.widget(i)
                try:
                    # Stop RGB preview triggers/workers before removing the tab.
                    if name == "RGB Fusion":
                        if hasattr(self, '_preview_timer') and self._preview_timer is not None:
                            try:
                                self._preview_timer.stop()
                            except Exception:
                                pass
                        if hasattr(self, '_rgb_worker') and self._rgb_worker and self._rgb_worker.isRunning():
                            try:
                                self._rgb_worker.requestInterruption()
                                self._rgb_worker.quit()
                                if not self._rgb_worker.wait(1000):
                                    self._rgb_worker.terminate()
                                    self._rgb_worker.wait(1000)
                            except Exception:
                                pass
                    # Always run mode-specific unload logic first to release buffers.
                    self.unload_view_widget(widget, name)
                except Exception:
                    pass
                try:
                    if name == "Individual Bands":
                        # clear per-band viewers if any
                        if hasattr(self, 'individual_bands_notebook'):
                            for j in range(self.individual_bands_notebook.count()):
                                w = self.individual_bands_notebook.widget(j)
                                if w and w.layout() and w.layout().count() > 0:
                                    child = w.layout().itemAt(0).widget()
                                    if hasattr(child, 'original_image_data'):
                                        child.original_image_data = None
                    elif name == "RGB Fusion":
                        if hasattr(self, 'rgb_preview_viewer') and hasattr(self, 'rgb_preview_viewer', 'original_image_data'):
                            self.rgb_preview_viewer.original_image_data = None
                    elif name == "All Bands":
                        if hasattr(self, 'all_bands_viewer') and hasattr(self, 'all_bands_viewer', 'original_image_data'):
                            self.all_bands_viewer.original_image_data = None
                except Exception:
                    pass
                # remove tab
                self.view_tabs.removeTab(i)
                # make sure Qt processes deletes and run GC
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
                try:
                    gc.collect()
                except Exception:
                    pass
                # --- NEW: keep checkbox state in sync ---
                try:
                    cb = self.view_checkboxes.get(name) if hasattr(self, 'view_checkboxes') else None
                    if cb:
                        # block signals so the checkbox toggle doesn't try to re-add the tab
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                except Exception:
                    pass
                return
   
    def update_offset_value(self, idx=None, axis=None, value=None):
        if idx is not None and axis is not None and value is not None:
            band_key = f"b{idx}"
            self.band_offsets[band_key][axis] = value
        else:
            for band_key in self.band_offsets:
                self.offset_spins.get(f"{band_key}_x", QSpinBox()).value()
                self.offset_spins.get(f"{band_key}_y", QSpinBox()).value()
        self._invalidate_cache()
        if self.view_tabs.currentIndex() in [0, 3]: # Only update views if in All Bands or RGB Fusion tab
            self.update_views()
       
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with .bandXX files")
        if not folder:
            return
       
        self.current_folder = folder
        self.folder_label.setText(os.path.basename(folder))
       
        if not self.load_parameters(folder):
            dialog = self._build_parameter_dialog(folder)
            if dialog.exec_() == QDialog.Accepted:
                params = dialog.get_parameters()
                self._apply_image_params(params["width"], params["raw_height"], params["bit_depth"], params["tdi_stage"])
                # Persist folder default silently
                try:
                    save_params_for_path(folder, {
                        'width': params["width"],
                        'height': params["height"],
                        'raw_height': int(params["raw_height"]),
                        'bit_depth': int(params["bit_depth"]),
                        'tdi_stage': int(params["tdi_stage"]),
                    }, as_default=True)
                except Exception:
                    pass
            else:
                return
       
        # Add to recent history
        try:
            add_recent(folder, 'band', {
                'width': int(self.width_entry.text()),
                'height': int(self.height_entry.text()),
                'raw_height': int(getattr(self, 'raw_height', self.height_entry.text()) or self.height_entry.text()),
                'bit_depth': int(self.bitdepth_var.currentText()),
                'tdi_stage': int(getattr(self, 'tdi_stage', 0) or 0),
            })
        except Exception:
            pass
        self._notify_iris_folder_loaded(folder)
        self.load_folder_data()

    def _notify_iris_folder_loaded(self, folder: str):
        """Tell Iris which acquisition folder is now loaded so it can read the right .log file."""
        try:
            main_app = getattr(self, 'main_app', None)
            if main_app is None:
                return
            iris = getattr(main_app, 'iris', None)
            if iris is None:
                return
            iris._last_loaded_folder = folder
            tab_widget = getattr(main_app, 'tab_widget', None)
            tab_index = tab_widget.indexOf(self) if tab_widget is not None else -1
            if tab_index >= 0:
                try:
                    iris.notify_tab_activated(tab_index, self, "band")
                except Exception:
                    pass
                try:
                    frame_count = 0
                    band_count = 0
                    if getattr(self, "band_frames", None):
                        try:
                            first_key = sorted(self.band_frames.keys())[0]
                            frame_count = len(self.band_frames.get(first_key) or [])
                        except Exception:
                            frame_count = 0
                        band_count = len(getattr(self, "bands_info", {}) or {}) or len(self.band_frames)
                    iris.notify_dataset_loaded(
                        tab_index,
                        folder,
                        frame_count,
                        band_count,
                        getattr(self, "bands_info", {}) or {},
                        self,
                    )
                except Exception:
                    pass
            # Also trigger the full analysis in background
            if hasattr(iris, 'analyze_folder_on_load'):
                iris.analyze_folder_on_load(folder)
        except Exception as e:
            print(f"[Iris] Folder notify error: {e}")
    def _show_recent_menu(self):
        try:
            menu = QMenu(self)
            recs = get_recents_for_mode('band', limit=7)
            if not recs:
                a = menu.addAction("No recent folders")
                a.setEnabled(False)
            else:
                for r in recs:
                    ts = r.get('last_opened', '')
                    display = f"{os.path.basename(r.get('path',''))} — {ts[:19]}"
                    act = menu.addAction(display)
                    act.setToolTip(r.get('path'))
                    path = r.get('path')
                    act.triggered.connect(lambda checked, p=path: self._open_recent_folder(p))
            all_recs = get_recents_for_mode('band')
            if len(all_recs) > 7:
                menu.addSeparator()
                vm = menu.addAction("View more...")
                vm.triggered.connect(lambda: self._open_full_history('band'))
            pos = self.load_menu_btn.mapToGlobal(self.load_menu_btn.rect().bottomLeft())
            menu.exec_(pos)
        except Exception:
            pass

    def _open_recent_folder(self, folder):
        # mimic select_folder semantics but without dialog
        if not folder:
            return
        self.current_folder = folder
        self.folder_label.setText(os.path.basename(folder))
        
        if not self.load_parameters(folder):
            # no cached parameters → use dialog to ask user
            dialog = self._build_parameter_dialog(folder)
            if dialog.exec_() == QDialog.Accepted:
                params = dialog.get_parameters()
                self._apply_image_params(params["width"], params["raw_height"], params["bit_depth"], params["tdi_stage"])
                try:
                    save_params_for_path(folder, {
                        'width': params["width"],
                        'height': params["height"],
                        'raw_height': int(params["raw_height"]),
                        'bit_depth': int(params["bit_depth"]),
                        'tdi_stage': int(params["tdi_stage"]),
                    }, as_default=True)
                except Exception:
                    pass
            else:
                return
        try:
            add_recent(folder, 'band', {
                'width': int(self.width_entry.text()),
                'height': int(self.height_entry.text()),
                'raw_height': int(getattr(self, 'raw_height', self.height_entry.text()) or self.height_entry.text()),
                'bit_depth': int(self.bitdepth_var.currentText()),
                'tdi_stage': int(getattr(self, 'tdi_stage', 0) or 0),
            })
        except Exception:
            pass
        self._notify_iris_folder_loaded(folder)
        self.load_folder_data()

    def _open_full_history(self, mode):
        try:
            sel = select_from_history(self, mode=mode)
            if sel:
                self._open_recent_folder(sel)
        except Exception:
            pass

    def update_pixel_info(self, x, y, values, is_rgb=False):
        latlon = None
        try:
            if getattr(self, 'geo_info', None) is not None:
                bands_info = getattr(self, 'bands_info', None)
                # gap
                gap = 0
                if getattr(self, 'gap_var', None):
                    try:
                        gap = int(self.gap_var.value())
                    except Exception:
                        gap = 0
                # original per-band height (prefer UI entry)
                orig_band_h = None
                try:
                    orig_band_h = int(self.height_entry.text())
                except Exception:
                    try:
                        orig_band_h = int(self.geo_info[3])
                    except Exception:
                        orig_band_h = None
                try:
                    merge_lr = False
                    x_mapped, y_mapped = x, y
                    if hasattr(self, "_map_display_coords_for_geo"):
                        x_mapped, y_mapped, merge_lr = self._map_display_coords_for_geo(x, y)
                    lat, lon, band_idx = image_coords_to_latlon(
                        x_mapped, y_mapped,
                        self.geo_info,
                        bands_info=bands_info,
                        gap=gap,
                        orig_band_h=orig_band_h,
                        merge_lr=merge_lr
                    )
                    latlon = (lat, lon, band_idx)
                except Exception as ex:
                    print("DEBUG: update_pixel_info geolocate failed:", ex)
                    latlon = None
        except Exception:
            latlon = None
        # --- Scale pixel values back to full DN range for high bit-depth sources ---
        scaled_values = values
        try:
            bitdepth = int(getattr(self, "bitdepth", 8) or 8)
        except Exception:
            bitdepth = 8
        try:
            if isinstance(values, np.ndarray) and bitdepth > 8:
                # If values already look like raw DN (uint16 or >8-bit range), don't rescale.
                if values.dtype != np.uint8:
                    scaled_values = values
                elif values.size > 0 and np.max(values) > 255:
                    scaled_values = values
                else:
                    max_dn = (1 << bitdepth) - 1
                    if max_dn > 0:
                        scale = float(max_dn) / 255.0
                        # Work in float to preserve as much information as possible,
                        # then round to nearest integer DN.
                        scaled_values = np.rint(values.astype(np.float32) * scale).astype(int)
        except Exception:
            # On any failure, fall back to original 0–255 values
            scaled_values = values
        # Pass latlon tuple (or None) and scaled_values to PixelInfoBox
        self.pixel_info_box.update_info(x, y, scaled_values, is_rgb=is_rgb, dn_value=latlon)
    def update_pixel_info_rgb(self, x, y, values, is_rgb=True):
        latlon = None
        try:
            if getattr(self, 'geo_info', None) is not None:
                bands_info = getattr(self, 'bands_info', None)
                gap = 0
                if getattr(self, 'gap_var', None):
                    try:
                        gap = int(self.gap_var.value())
                    except Exception:
                        gap = 0
                orig_band_h = None
                try:
                    orig_band_h = int(self.height_entry.text())
                except Exception:
                    try:
                        orig_band_h = int(self.geo_info[3])
                    except Exception:
                        orig_band_h = None
                try:
                    merge_lr = False
                    x_mapped, y_mapped = x, y
                    if hasattr(self, "_map_display_coords_for_geo"):
                        x_mapped, y_mapped, merge_lr = self._map_display_coords_for_geo(x, y)
                    lat, lon, band_idx = image_coords_to_latlon(
                        x_mapped, y_mapped,
                        self.geo_info,
                        bands_info=bands_info,
                        gap=gap,
                        orig_band_h=orig_band_h,
                        merge_lr=merge_lr
                    )
                    latlon = (lat, lon, band_idx)
                except Exception as ex:
                    print("DEBUG: update_pixel_info_rgb geolocate failed:", ex)
                    latlon = None
        except Exception:
            latlon = None
        # For RGB fusion / colour views, also scale values back to full DN range
        scaled_values = values
        try:
            bitdepth = int(getattr(self, "bitdepth", 8) or 8)
        except Exception:
            bitdepth = 8
        try:
            if isinstance(values, np.ndarray) and bitdepth > 8:
                if values.dtype != np.uint8:
                    scaled_values = values
                elif values.size > 0 and np.max(values) > 255:
                    scaled_values = values
                else:
                    max_dn = (1 << bitdepth) - 1
                    if max_dn > 0:
                        scale = float(max_dn) / 255.0
                        scaled_values = np.rint(values.astype(np.float32) * scale).astype(int)
        except Exception:
            scaled_values = values
        self.pixel_info_box.update_info(x, y, scaled_values, is_rgb=is_rgb, dn_value=latlon)
           
    def on_tab_changed(self, index):
        if index < 0 or getattr(self, '_is_closing', False):
            return
        if not self._qt_alive(getattr(self, 'view_tabs', None)):
            return
        # Free/unload previous tab's large data if applicable
        if hasattr(self, '_last_tab_index') and self._last_tab_index >= 0 and self._last_tab_index != index:
            prev_tab_name = self.view_tabs.tabText(self._last_tab_index)
            prev_tab_key = prev_tab_name.lower().replace(' ', '_')
            # Save individual sub-tab viewer state before unloading
            if prev_tab_key == 'individual_bands' and hasattr(self, 'individual_bands_notebook'):
                current_sub = self.individual_bands_notebook.currentIndex()
                if current_sub >= 0:
                    self._save_individual_viewer_state(current_sub)
            elif prev_tab_key in self.access_times:  # Only unload non-individual tabs
                self._unload_data_only(prev_tab_key)
                if prev_tab_key not in self.unloaded_keys:
                    self.unloaded_keys.add(prev_tab_key)
        tab_name = self.view_tabs.tabText(index)
        tab_key = tab_name.lower().replace(' ', '_')
        if tab_key in ['all_bands', 'rgb_fusion', 'histogram']:
            self.access_times[tab_key] = time.time()
            if tab_key in self.unloaded_keys:
                self._reload_tab_data(tab_key)
                self.unloaded_keys.discard(tab_key)
        elif tab_key == 'individual_bands' and hasattr(self, 'individual_bands_notebook'):
            sub_index = self.individual_bands_notebook.currentIndex()
            if sub_index >= 0:
                # Trigger the consolidated sub-tab handler to restore state/load data
                self.on_individual_tab_changed(sub_index)
        self._last_tab_index = index
       
        current_hash = self._compute_param_hash()
        if tab_name == "All Bands":
            self._update_cached_view('all_bands', self.update_all_bands_view)
        elif tab_name == "Individual Bands":
            # Only do a full rebuild if no sub-tabs exist yet or params changed.
            # Individual band tabs persist across main-tab switches so we avoid
            # the destructive update_individual_bands_view when they are still loaded.
            needs_rebuild = (self.individual_bands_notebook.count() == 0
                             or self._individual_bands_built_hash != current_hash)
            if needs_rebuild:
                self._update_cached_view('individual_bands', self.update_individual_bands_view)
                self._individual_bands_built_hash = current_hash
        elif tab_name == "Histogram":
            self.update_histogram_view() # Histogram is cheap, no cache needed
        elif tab_name == "RGB Fusion":
            self.preview_rgb_fusion()
        #self.refresh()

    def _keep_main_window_on_screen(self):
        """Clamp the top-level window geometry to the current screen."""
        try:
            win = self.window()
            if win is None:
                return
            if win.isFullScreen() or win.isMaximized():
                return
            # Use the screen at window center to avoid jumping across displays.
            center = win.frameGeometry().center() if hasattr(win, "frameGeometry") else None
            app = QApplication.instance()
            screen = app.screenAt(center) if (app is not None and center is not None and hasattr(app, "screenAt")) else None
            if screen is None and hasattr(win, "windowHandle") and win.windowHandle() is not None:
                screen = win.windowHandle().screen()
            if screen is None and hasattr(win, "screen"):
                screen = win.screen()
            if screen is None and app is not None:
                screen = app.primaryScreen()
            if screen is None:
                return

            avail = screen.availableGeometry()
            geo = win.geometry()
            margin = 12

            max_w = max(320, avail.width() - margin * 2)
            max_h = max(240, avail.height() - margin * 2)
            new_w = min(geo.width(), max_w)
            new_h = min(geo.height(), max_h)

            min_x = avail.left() + margin
            min_y = avail.top() + margin
            max_x = avail.right() - margin - new_w + 1
            max_y = avail.bottom() - margin - new_h + 1
            new_x = min(max(geo.x(), min_x), max_x)
            new_y = min(max(geo.y(), min_y), max_y)

            if (new_x, new_y, new_w, new_h) != (geo.x(), geo.y(), geo.width(), geo.height()):
                win.setGeometry(new_x, new_y, new_w, new_h)
        except Exception:
            pass
    def on_sub_tab_changed(self, sub_index):
        if sub_index < 0:
            return
        widget = self.individual_bands_notebook.widget(sub_index)
        if hasattr(widget, 'key'):
            key = widget.key
            self.access_times[key] = time.time()
            # Prevent reentrant reload loops and segfaults from repeated tab-change cycles.
            if key in self.unloaded_keys and not getattr(self, '_is_reloading_band', False):
                self._is_reloading_band = True
                try:
                    self._reload_band_data(key)
                    self.unloaded_keys.discard(key)
                finally:
                    self._is_reloading_band = False
    def _reload_tab_data(self, tab_key):
        if tab_key == 'all_bands':
            self.update_all_bands_view()
        elif tab_key == 'rgb_fusion':
            self.preview_rgb_fusion()
        elif tab_key == 'histogram':
            self.update_histogram_view()
    def _reload_band_data(self, key):
        """Force reload: unload if needed, reset to placeholder, then load with UI."""
        print(f"[DEBUG] _reload_band_data called for {key}")
        for i in range(self.individual_bands_notebook.count()):
            widget = self.individual_bands_notebook.widget(i)
            if (hasattr(widget, 'key') and widget.key == key) or (key == 'pan' and getattr(widget, 'key', None) == 'pan'):
                print(f"[DEBUG] Found {key} at {i}, current objectName: {widget.objectName()}")
                # Unload if loaded
                if widget.objectName() == "loaded":
                    self.unload_individual_subtab(widget)
                # Ensure placeholder state for loading UI
                widget.setObjectName("placeholder")
                print(f"[DEBUG] Set to placeholder, now calling lazy_load_individual_tab({i})")
                # NEW: Delay to allow placeholder to be recognized and UI to update
                QTimer.singleShot(50, lambda idx=i: self.lazy_load_individual_tab(idx)) # 50ms delay
                # Avoid self-triggering on_sub_tab_changed (that's already running for this index)
                # self.individual_bands_notebook.setCurrentIndex(i)
                break
        else:
            print(f"[DEBUG] No tab for {key}")
    def _load_pan_data(self):
        # Extracted from toggle_band pan check branch
        unbinned_keys = [k for k, cb in self.band_enabled.items() if cb.isChecked() and k.endswith(('_left', '_right'))]
        if not unbinned_keys:
            return
        # Find existing pan tab (placeholder or loaded)
        for i in range(self.individual_bands_notebook.count()):
            widget = self.individual_bands_notebook.widget(i)
            if widget.objectName() in ["pan_placeholder", "loaded"] and getattr(widget, 'key', None) == 'pan':
                self.individual_bands_notebook.setCurrentIndex(i) # Triggers lazy_load_individual_tab
                return
        # If no tab, add placeholder (as in toggle_band)
        placeholder = QWidget()
        placeholder.setObjectName("pan_placeholder")
        placeholder.key = 'pan'
        placeholder.unbinned_keys = unbinned_keys
        placeholder.original_tab_text = "Pan"
        self.individual_bands_notebook.addTab(placeholder, "Pan (*)")
        self.individual_bands_notebook.setCurrentIndex(self.individual_bands_notebook.count() - 1) # Trigger load
    def _load_individual_band_data(self, key):
        # Extracted from toggle_band normal check branch
        # Find existing tab
        for i in range(self.individual_bands_notebook.count()):
            widget = self.individual_bands_notebook.widget(i)
            if hasattr(widget, 'key') and widget.key == key:
                self.individual_bands_notebook.setCurrentIndex(i) # Triggers lazy_load_individual_tab
                return
        # If no tab, add placeholder
        placeholder = QWidget()
        placeholder.setObjectName("placeholder")
        placeholder.key = key
        base_key = key.rsplit('_', 1)[0] if '_' in key else key
        side = key.split('_')[-1] if '_' in key else ''
        tab_text = f"Band {base_key[1:]} {side}".strip()
        placeholder.original_tab_text = tab_text
        self.individual_bands_notebook.addTab(placeholder, f"{tab_text} (*)")
        self.individual_bands_notebook.setCurrentIndex(self.individual_bands_notebook.count() - 1) # Trigger load
    def unload_individual_subtab(self, sub_widget):
        key = getattr(sub_widget, 'key', None)
        print(f"[DEBUG] unload_individual_subtab called for key: {key}")
        if not key:
            print(f"[DEBUG] No key found, skipping unload")
            return
        if hasattr(sub_widget, 'worker') and sub_widget.worker and sub_widget.worker.isRunning():
            sub_widget.worker.requestInterruption()
            sub_widget.worker.wait(2000)
            if sub_widget.worker.isRunning():
                sub_widget.worker.terminate()
                sub_widget.worker.wait(1000)
            print(f"[DEBUG] Worker interrupted for {key}")
        if hasattr(sub_widget, 'loading_timer') and sub_widget.loading_timer:
            sub_widget.loading_timer.stop()
            print(f"[DEBUG] Loading timer stopped for {key}")
        viewer = sub_widget.findChild(GraphicsImageViewer)
        if viewer:
            viewer.show_image(None)
            try:
                if hasattr(viewer, 'current_pil_image') and viewer.current_pil_image:
                    del viewer.current_pil_image
                if hasattr(viewer, 'raw_pil_image') and viewer.raw_pil_image:
                    del viewer.raw_pil_image
            except Exception as e:
                print(f"[DEBUG] Error deleting images for {key}: {e}")
            viewer.setParent(None)
            viewer.deleteLater()
            print(f"[DEBUG] Viewer cleared and deleted for {key}")
        layout = sub_widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()
            print(f"[DEBUG] Layout cleared for {key}")
        if key in self.loaded_band_memories:
            del self.loaded_band_memories[key]
            print(f"[DEBUG] Removed '{key}' from loaded_band_memories: {self.loaded_band_memories}")
        # Set placeholder state so later reload will consistently start from placeholder.
        sub_widget.setObjectName("placeholder")
        gc.collect()
        print(f"[DEBUG] Unload complete for {key} (layout cleared, now placeholder)")
    def _unload_data_only(self, key):
        """Unload data for a specific loaded band/Pan without removing the tab from the notebook."""
        print(f"[DEBUG] _unload_data_only called for key: {key}")
        notebook = getattr(self, 'individual_bands_notebook', None)
        if not key or not self._qt_alive(notebook):
            print(f"[DEBUG] No key provided, skipping unload")
            return
        # Find the widget by key
        widget = None
        for i in range(notebook.count()):
            w = notebook.widget(i)
            if (hasattr(w, 'key') and w.key == key) or \
            (key == 'pan' and (getattr(w, 'key', None) == 'pan' or w.objectName() == "pan_placeholder")):
                widget = w
                break
        if not widget:
            print(f"[DEBUG] No widget found for key {key}, skipping unload")
            return
        # NEW: Force-clear any existing layout/content to ensure clean placeholder state
        layout = widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()
            print(f"[DEBUG] Cleared existing layout for {key}")
        # Perform unload (existing code)
        if hasattr(widget, 'worker') and widget.worker and widget.worker.isRunning():
            widget.worker.requestInterruption()
            widget.worker.quit() # NEW: Polite quit
            if not widget.worker.wait(2000): # Wait up to 2s
                widget.worker.terminate() # Force if needed
                widget.worker.wait(1000) # Brief wait after terminate
            print(f"[DEBUG] Worker interrupted/quit for {key}")
        if hasattr(widget, 'loading_timer') and widget.loading_timer:
            widget.loading_timer.stop()
            print(f"[DEBUG] Loading timer stopped for {key}")
        viewer = widget.findChild(GraphicsImageViewer)
        if viewer:
            viewer.show_image(None)
            try:
                if hasattr(viewer, 'current_pil_image') and viewer.current_pil_image:
                    del viewer.current_pil_image
                if hasattr(viewer, 'raw_pil_image') and viewer.raw_pil_image:
                    del viewer.raw_pil_image
            except Exception as e:
                print(f"[DEBUG] Error deleting images for {key}: {e}")
            viewer.setParent(None)
            viewer.deleteLater()
            print(f"[DEBUG] Viewer cleared and deleted for {key}")
        # Reset to placeholder (existing)
        widget.setObjectName("placeholder")
        widget.key = key # Ensure key is set
        # Clean up memory trackers (existing)
        if key in self.loaded_band_memories:
            del self.loaded_band_memories[key]
            print(f"[DEBUG] Removed '{key}' from loaded_band_memories: {self.loaded_band_memories}")
        # Do NOT pop from viewer_states here; it is needed for restoration.
        self.unloaded_keys.add(key)
        print(f"[DEBUG] Reset widget to 'placeholder' for {key}, added to unloaded_keys: {self.unloaded_keys}")
        # NEW: Update tab text immediately and force UI repaint for visibility
        tab_idx = notebook.indexOf(widget)
        if tab_idx >= 0 and hasattr(widget, 'original_tab_text'):
            notebook.setTabText(tab_idx, f"{widget.original_tab_text} (*)")
        # NEW: Force Qt to process events (ensures placeholder state is visible before next load)
        QApplication.processEvents()
        gc.collect()
        print(f"[DEBUG] GC collected after _unload_data_only for {key}")
    def handle_memory_pressure(self):
        usage = psutil.virtual_memory().percent
        # Only start unloading once we are above the 90% threshold
        if usage < 90.0:
            return
        loaded_keys = [k for k in self.access_times if k not in self.unloaded_keys]
        if not loaded_keys:
            gc.collect()
            return
        lru_key = min(loaded_keys, key=lambda k: self.access_times[k])
        print(f"Unloaded data for {lru_key} to free memory") # Existing log
        # NEW: Emit signal for main-thread unload (instead of direct call)
        self.memory_monitor.unload_request.emit(lru_key)
        gc.collect()
        if psutil.virtual_memory().percent >= 87.0:
            QTimer.singleShot(500, self.handle_memory_pressure)
    def _update_cached_view(self, cache_key, update_func):
        cache = self.view_cache.get(cache_key, {})
        current_frame = self.current_frame_index
        current_hash = self._compute_param_hash()
        cached_frame = cache.get('frame_index', -1)
        cached_hash = cache.get('hash', '')
        if cached_frame == current_frame and cached_hash == current_hash:
            # Use cache
            pil_image = cache.get('pil_image')
            raw_data = cache.get('raw_data')
            if raw_data is None:
                # Backward compatibility with older cache entries.
                raw_data = cache.get('original_data')
            self._apply_cached_image(cache_key, pil_image, raw_data)
            print(f"Used cache for {cache_key} frame {current_frame}")
        else:
            # Update and cache
            try:
                update_func()
            except RuntimeError as e:
                if "has been deleted" in str(e):
                    print(f"[DEBUG] Skipping {cache_key} update: stale Qt object ({e})")
                    return
                raise
            viewer = self._get_viewer_for_key(cache_key)
            if viewer:
                pil_image = viewer.current_pil_image
                # Keep full-bit raw data when available for accurate pixel-info values.
                raw_data = viewer.original_raw_data if getattr(viewer, 'original_raw_data', None) is not None else viewer.original_image_data
                self.view_cache[cache_key] = {
                    'frame_index': current_frame,
                    'pil_image': pil_image.copy() if pil_image else None,
                    'raw_data': raw_data.copy() if raw_data is not None else None,
                    'hash': current_hash
                }
    def _get_viewer_for_key(self, key):
        try:
            if key == 'all_bands':
                return getattr(self, 'all_bands_viewer', None)
            if key in ('rgb_fusion', 'rgb'):
                return getattr(self, 'rgb_preview_viewer', None)
            if key == 'individual_bands':
                # Return the currently visible per-band viewer (if any)
                notebook = getattr(self, 'individual_bands_notebook', None)
                if notebook:
                    widget = notebook.currentWidget()
                    if widget:
                        # find the first GraphicsImageViewer child in that tab
                        try:
                            viewer = widget.findChild(type(self.all_bands_viewer)) if getattr(self, 'all_bands_viewer', None) else None
                        except Exception:
                            viewer = None
                        # fallback: generic findChild for GraphicsImageViewer by name/class
                        if viewer is None:
                            try:
                                # find by class name (defensive)
                                for child in widget.findChildren(QWidget):
                                    if getattr(child, '__class__', None) and child.__class__.__name__ == 'GraphicsImageViewer':
                                        return child
                            except Exception:
                                pass
                        return viewer
                return None
        except Exception:
            return None
        return None
    def _apply_cached_image(self, key, pil_image, raw_data):
        try:
            viewer = self._get_viewer_for_key(key)
            if not viewer:
                return
            # Ensure we pass a copy so the viewer owns its image state
            try:
                pil_to_show = pil_image.copy() if hasattr(pil_image, 'copy') else pil_image
            except Exception:
                pil_to_show = pil_image
            try:
                # fit_to_screen decision matches existing use elsewhere
                fit = (self.fit_mode_var.checkedId() == 0) if getattr(self, 'fit_mode_var', None) else False
                # Use display image for UI paint path, then restore raw array directly.
                viewer.show_image(pil_to_show, fit_to_screen=fit, raw_pil=pil_to_show)
                if raw_data is not None:
                    viewer.original_raw_data = raw_data.copy() if hasattr(raw_data, 'copy') else raw_data
            except Exception:
                # fallback: set fields directly
                try:
                    viewer.current_pil_image = pil_to_show
                except Exception:
                    pass
                try:
                    copied = raw_data.copy() if hasattr(raw_data, 'copy') else raw_data
                    viewer.original_image_data = copied
                    viewer.original_raw_data = copied
                except Exception:
                    viewer.original_image_data = raw_data
                    viewer.original_raw_data = raw_data
        except Exception as e:
            # don't raise from UI helper
            print(f"_apply_cached_image error for key={key}: {e}")
            return
    def save_parameters(self):
        params = {
            "width": self.width_entry.text(),
            "height": self.height_entry.text(),
            "raw_height": int(getattr(self, 'raw_height', self.height_entry.text()) or self.height_entry.text()),
            "bit_depth": int(self.bitdepth_var.currentText()),
            "tdi_stage": int(getattr(self, 'tdi_stage', 0) or 0),
            "band_gap": self.gap_var.value(),
            "matrix_size": self.matrix_size_var.value(),
            "contrast_enhance": self.contrast_enhance_var.isChecked(),
            "contrast_min": self.contrast_min_var.value(),
            "contrast_max": self.contrast_max_var.value(),
            "band_enabled": {key: cb.isChecked() for key, cb in self.band_enabled.items()},
            "rgb_channels": {
                "red": self.red_band_var.currentText(),
                "green": self.green_band_var.currentText(),
                "blue": self.blue_band_var.currentText()
            },
            "band_offsets": self.band_offsets
        }
       
        try:
            from utils import save_params_for_path
            save_params_for_path(self.current_folder, params, as_default=True)
            QMessageBox.information(self, "Success", "Parameters saved successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {e}")
   
    def _invalidate_cache(self):
        self.view_cache.clear()
        self.param_hash = self._compute_param_hash()
        self.update_views()

    def _on_1_to_4_layout_toggled(self, state):
        self.ENABLE_1_TO_4_LAYOUT = (state == Qt.Checked)
        # Layout changes affect both all-bands and individual tabs.
        self._invalidate_cache()
        try:
            self.update_individual_bands_view()
        except Exception:
            pass
       
    def load_parameters(self, folder):
        try:
            params = get_saved_params_for_file(folder)
            if not params:
                data = load_folder_params(folder)
                if isinstance(data, dict):
                    params = data.get("default") if isinstance(data.get("default"), dict) else data
            if not params:
                return False
           
            width = params.get("width", "8448")
            tdi_stage = normalize_tdi_stage(params.get("tdi_stage", 0))
            raw_height = params.get("raw_height")
            if raw_height is None:
                try:
                    raw_height = int(params.get("height", 384)) * tdi_stage if tdi_stage > 0 else int(params.get("height", 384))
                except Exception:
                    raw_height = 384
            self._apply_image_params(width, raw_height, params.get("bit_depth", 10), tdi_stage)
            self.gap_var.setValue(params.get("band_gap", 0))
            self.matrix_size_var.setValue(params.get("matrix_size", 3))
            self.contrast_enhance_var.setChecked(params.get("contrast_enhance", False))
            self.contrast_min_var.setValue(params.get("contrast_min", 0.0))
            self.contrast_max_var.setValue(params.get("contrast_max", float(self._current_max_dn())))
            self._normalize_legacy_contrast_limits()
           
           
            band_enabled = params.get("band_enabled", {})
            for key in band_enabled:
                if key in self.band_enabled:
                    self.band_enabled[key].setChecked(band_enabled[key])
       
           
            rgb_channels = params.get("rgb_channels", {})
            self.red_band_var.setCurrentText(rgb_channels.get("red", "b0"))
            self.green_band_var.setCurrentText(rgb_channels.get("green", "b1"))
            self.blue_band_var.setCurrentText(rgb_channels.get("blue", "b2"))
           
            band_offsets = params.get("band_offsets", {})
            for i in range(7):
                band_key = f"b{i}"
                offsets = band_offsets.get(band_key, {"x": 0, "y": 0})
                # Only update if not already set by user
                if self.offset_spins[f"{band_key}_x"].value() == 0:
                    self.offset_spins[f"{band_key}_x"].setValue(offsets["x"])
                if self.offset_spins[f"{band_key}_y"].value() == 0:
                    self.offset_spins[f"{band_key}_y"].setValue(offsets["y"])
                self.band_offsets[band_key]["x"] = self.offset_spins[f"{band_key}_x"].value()
                self.band_offsets[band_key]["y"] = self.offset_spins[f"{band_key}_y"].value()
            return True
        except Exception as e:
            print(f"Failed to load parameters: {e}")
            return False
   
    def refresh(self):
        self.update_views()
        if self.fit_mode_var.checkedId() == 0:
            self.fit_to_screen()
        print("refresh")
    def _safe_restore_individual_index(self, idx):
        try:
            if hasattr(self, 'individual_bands_notebook') and self.individual_bands_notebook is not None:
                if 0 <= idx < self.individual_bands_notebook.count():
                    self.individual_bands_notebook.setCurrentIndex(idx)
        except Exception:
            pass
    def refresh_current_tab(self):
        try:
            # Save current individual sub-tab index (if the notebook exists)
            saved_individual_index = -1
            try:
                if hasattr(self, 'individual_bands_notebook') and self.individual_bands_notebook is not None:
                    saved_individual_index = self.individual_bands_notebook.currentIndex()
            except Exception:
                saved_individual_index = -1
            try:
                if hasattr(self, 'update_views'):
                    self.update_views()
            except Exception as e:
                print("refresh_current_tab: refresh call failed:", e)
            # Restore the saved individual sub-tab index after Qt finishes its event work.
            if saved_individual_index >= 0:
                QTimer.singleShot(0, lambda idx=saved_individual_index: self._safe_restore_individual_index(idx))
        except Exception as e:
            print("refresh_current_tab outer error:", e)
    def reload_folder_data(self):
        try:
            # call the existing loader (which validates width/height/bitdepth)
            self.load_folder_data()
            # update UI after load
            self.update_views()
            if self.fit_mode_var.checkedId() == 0:
                self.fit_to_screen()
        except Exception as e:
            QMessageBox.critical(self, "Reload Error", f"Failed to reload folder: {e}")
            print("reload_folder_data error:", e)
    def show_params_popup(self):
        try:
            dialog = self._build_parameter_dialog(self.current_folder)
            if dialog.exec_() == QDialog.Accepted:
                params = dialog.get_parameters()
                self._apply_image_params(params["width"], params["raw_height"], params["bit_depth"], params["tdi_stage"])
                if self.current_folder:
                    try:
                        save_params_for_path(self.current_folder, {
                            "width": self.width_entry.text(),
                            "height": self.height_entry.text(),
                            "raw_height": int(getattr(self, 'raw_height', self.height_entry.text()) or self.height_entry.text()),
                            "bit_depth": int(self.bitdepth_var.currentText()),
                            "tdi_stage": int(getattr(self, 'tdi_stage', 0) or 0),
                        }, as_default=True)
                    except Exception:
                        pass
                    try:
                        add_recent(self.current_folder, 'band', {
                            "width": int(self.width_entry.text()),
                            "height": int(self.height_entry.text()),
                            "raw_height": int(getattr(self, 'raw_height', self.height_entry.text()) or self.height_entry.text()),
                            "bit_depth": int(self.bitdepth_var.currentText()),
                            "tdi_stage": int(getattr(self, 'tdi_stage', 0) or 0),
                        })
                    except Exception:
                        pass
            else:
                return
            self.load_folder_data()
        except Exception as e:
            print("show_params_popup error:", e)
            QMessageBox.information(self, "Parameters", "Could not read parameters.")
   
    def _set_custom_close_button(self, index):
        close_btn = QToolButton()
        close_btn.setText("×") # Trendy Unicode cross
        close_btn.setFixedSize(18, 18) # Compact size for modern look
        close_btn.setProperty("class", "tab-close") # For stylesheet targeting
        close_btn.setToolTip("Close this tab")
        close_btn.clicked.connect(lambda: self._on_view_tab_close(index))
        self.view_tabs.tabBar().setTabButton(index, QTabBar.RightSide, close_btn)
    def _save_individual_viewer_state(self, tab_index):
        """Save the zoom/scroll/rotation state of the individual band viewer at tab_index."""
        if tab_index < 0:
            return
        try:
            widget = self.individual_bands_notebook.widget(tab_index)
            if not widget or widget.objectName() != "loaded":
                return
            key = getattr(widget, 'key', None)
            if not key:
                return
            viewer = widget.findChild(GraphicsImageViewer)
            if not viewer:
                return
            gv = viewer.graphics_view
            self.viewer_states[key] = {
                'zoom': viewer.zoom,
                'rotation': getattr(viewer, 'rotation', 0.0),
                'scroll_x': gv.horizontalScrollBar().value(),
                'scroll_y': gv.verticalScrollBar().value(),
            }
        except (RuntimeError, AttributeError):
            pass

    def _restore_individual_viewer_state(self, tab_index):
        """Restore saved zoom/scroll/rotation state for the individual band viewer at tab_index.
        Returns True if state was restored, False otherwise."""
        if tab_index < 0:
            return False
        try:
            widget = self.individual_bands_notebook.widget(tab_index)
            if not widget or widget.objectName() != "loaded":
                return False
            key = getattr(widget, 'key', None)
            if not key:
                return False
            state = self.viewer_states.get(key)
            if not state:
                return False
            viewer = widget.findChild(GraphicsImageViewer)
            if not viewer:
                return False
            gv = viewer.graphics_view
            zoom = state.get('zoom', 1.0)
            rotation = state.get('rotation', 0.0)
            viewer.graphics_view.setTransform(QTransform().rotate(state.get('rotation', 0.0)).scale(state.get('zoom', 1.0), state.get('zoom', 1.0)))
            viewer.zoom = zoom
            viewer.rotation = rotation
            gv.horizontalScrollBar().setValue(state.get('scroll_x', 0))
            gv.verticalScrollBar().setValue(state.get('scroll_y', 0))
            return True
        except (RuntimeError, AttributeError):
            return False

    def on_individual_tab_changed(self, index):
        notebook = getattr(self, 'individual_bands_notebook', None)
        if index < 0 or getattr(self, '_is_closing', False) or not self._qt_alive(notebook):
            return

        # 1. Save state of previous sub-tab
        prev = getattr(self, '_last_individual_tab_index', -1)
        if prev >= 0 and prev != index:
            self._save_individual_viewer_state(prev)
        self._last_individual_tab_index = index

        widget = notebook.widget(index)
        if not widget:
            return

        # 2. Track access time and handle unloads/reloads (former on_sub_tab_changed)
        key = getattr(widget, 'key', None)
        if key:
            self.access_times[key] = time.time()
            if key in self.unloaded_keys:
                self._reload_band_data(key)
                self.unloaded_keys.discard(key)

        # 3. Restore state OR trigger fit-to-screen (if loaded)
        # Note: If it's a placeholder, restoration happens later in _on_individual_band_loaded
        if widget.objectName() == "loaded":
            restored = self._restore_individual_viewer_state(index)
            if not restored:
                if self.fit_mode_var.checkedId() == 0:
                    self.fit_to_screen()
                else:
                    self.actual_size()
        elif widget.objectName() in ["placeholder", "pan_placeholder"]:
            # Trigger lazy load - this matches the check in band_views._setup_individual_tab_loading_signal
            try:
                self.lazy_load_individual_tab(index)
            except Exception as e:
                print(f"Error triggering lazy load for index {index}: {e}")

    def _current_max_dn(self):
        try:
            bd = int(getattr(self, 'bitdepth_var', None).currentText())
        except Exception:
            try:
                bd = int(getattr(self, 'bitdepth', 10))
            except Exception:
                bd = 8
        bd = max(8, bd)
        return (1 << bd) - 1 if bd > 8 else 255

    def _sync_contrast_range_to_bitdepth(self, clamp_only=False):
        max_dn = float(self._current_max_dn())
        try:
            old_min = float(self.contrast_min_var.value())
            old_max = float(self.contrast_max_var.value())
        except Exception:
            old_min, old_max = 0.0, max_dn
        self.contrast_min_var.setRange(0.0, max_dn)
        self.contrast_max_var.setRange(0.0, max_dn)
        if clamp_only:
            self.contrast_min_var.setValue(min(max(old_min, 0.0), max_dn))
            self.contrast_max_var.setValue(min(max(old_max, 0.0), max_dn))
        else:
            self.contrast_min_var.setValue(0.0)
            self.contrast_max_var.setValue(max_dn)
        # Keep range hint without adding a width-heavy inline label.
        try:
            tip = f"DN range: 0-{int(max_dn)}"
            self.contrast_min_var.setToolTip(tip)
            self.contrast_max_var.setToolTip(tip)
        except Exception:
            pass
        self._normalize_legacy_contrast_limits()
        # Keep histogram limits aligned with current DN domain unless data refresh sets them.
        try:
            self.histogram_viewer.min_val = int(self.contrast_min_var.value())
            self.histogram_viewer.max_val = int(self.contrast_max_var.value())
        except Exception:
            pass

    def _normalize_legacy_contrast_limits(self):
        """Expand old 8-bit defaults (0..255) when current bit depth is >8-bit."""
        try:
            max_dn = float(self._current_max_dn())
            if max_dn <= 255.0:
                return
            mn = float(self.contrast_min_var.value())
            mx = float(self.contrast_max_var.value())
            # Legacy sessions often persist 0/255 even for 10/12/16-bit data.
            if mx <= 255.0 and mn <= 1.0:
                self.contrast_min_var.setValue(0.0)
                self.contrast_max_var.setValue(max_dn)
        except Exception:
            pass
   
    def set_auto_contrast(self):
        if not self.band_frames:
            QMessageBox.warning(self, "Warning", "No data loaded, cannot set auto-contrast.")
            return
        # Compute min/max across all loaded bands for the current frame
        min_val = float('inf')
        max_val = float('-inf')
        keys = sorted(self.band_frames.keys()) # Use all loaded keys, ignore enabled checkboxes
        for key in keys:
            frames = self.band_frames.get(key)
            if frames and self.current_frame_index < len(frames):
                frame = frames[self.current_frame_index]
                min_val = min(min_val, float(np.min(frame)))
                max_val = max(max_val, float(np.max(frame)))
        if min_val == float('inf') or max_val == float('-inf'):
            min_val = 0.0
            max_val = float(self._current_max_dn())
        self.contrast_min_var.setValue(min_val)
        self.contrast_max_var.setValue(max_val)
        self.histogram_viewer.min_val = min_val
        self.histogram_viewer.max_val = max_val
        # self.refresh()
       
    def load_folder_data(self):
        if getattr(self, '_is_closing', False):
            return
        try:
            width = int(self.width_entry.text())
            height = int(self.height_entry.text())
            self.bitdepth = int(self.bitdepth_var.currentText())
            self.tdi_stage = int(getattr(self, 'tdi_stage', 0) or 0)
            self.raw_height = height * self.tdi_stage if self.tdi_stage > 0 else height
            self._sync_contrast_range_to_bitdepth(clamp_only=True)
            # Propagate updated bit depth to histogram viewer for correct DN axis labels
            try:
                if hasattr(self, "histogram_viewer"):
                    self.histogram_viewer.set_bitdepth(self.bitdepth)
            except Exception:
                pass
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive integers")
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self._load_generation += 1
        try:
            self._stop_thread(getattr(self, 'worker', None))
        except Exception:
            pass
        try:
            self._stop_thread(getattr(self, 'view_worker', None))
        except Exception:
            pass
        self.worker = LoadWorker(self.current_folder, width, height, self.bitdepth, self)
        self.worker._load_generation = self._load_generation
        self.worker.finished.connect(self.on_load_finished)
        self.worker.error.connect(self.on_load_error)
        # wire progress to both widgets
        self.worker.progress.connect(self.update_progress) # <- persistent bar
        # optionally update a label status
        #worker.progress.connect(lambda v: self._set_status(f"Loading... {v}%"))
        self._start_progress_session()
        self.worker.start()
    def _start_progress_session(self):
        self._progress_session_active = True
        self._progress_last_value = 0
        try:
            self.progress_bar.setValue(0)
            self.percent_label.setText("0%")
            self.percent_label.show()
        except Exception:
            pass
    def _finish_progress_session(self):
        self._progress_session_active = False
        self._progress_last_value = 0
        try:
            self.progress_bar.setValue(0)
            self.percent_label.hide()
        except Exception:
            pass
    def _set_status(self, text):
        if hasattr(self, 'worker') and self.worker.isRunning():
            try:
                self.worker.requestInterruption()
                self.worker.wait(1000)
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(2000)
            except Exception:
                pass
        try:
            self._finish_progress_session()
        except Exception:
            pass
    def _cancel_loading(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            try:
                # ask the worker to stop cooperatively
                self.worker.requestInterruption()
                # wait a short time for it to stop
                self.worker.wait(1000) # wait 1 second
                if self.worker.isRunning():
                    # last resort
                    self.worker.terminate()
                    self.worker.wait(2000)
            except Exception:
                try:
                    self.worker.terminate()
                except Exception:
                    pass
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.close()
        try:
            self._finish_progress_session()
        except Exception:
            pass
    def closeEvent(self, ev):
        # Prevent double-close from recursion during event loop teardown
        if getattr(self, '_is_closing', False):
            return
        self._is_closing = True

        # Stop memory monitor first
        if self.memory_monitor:
            try:
                self.memory_monitor.stop()
                self.memory_monitor.wait(2000)
            except Exception:
                pass

        # Stop RGB fusion worker cleanly to avoid QThread being destroyed
        try:
            if hasattr(self, '_rgb_worker') and self._rgb_worker:
                if self._rgb_worker.isRunning():
                    self._rgb_worker.requestInterruption()
                    self._rgb_worker.quit()
                    if not self._rgb_worker.wait(2000):
                        self._rgb_worker.terminate()
                        self._rgb_worker.wait(1000)
        except Exception as e:
            print(f"[DEBUG] Error stopping RGB worker: {e}")
        
        # Stop individual band workers
        try:
            if hasattr(self, 'individual_bands_notebook'):
                for i in range(self.individual_bands_notebook.count()):
                    w = self.individual_bands_notebook.widget(i)
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
        except Exception as e:
            print(f"[DEBUG] Error stopping individual band workers: {e}")
        
        try:
            if hasattr(self, 'individual_bands_notebook'):
                for i in range(self.individual_bands_notebook.count()):
                    w = self.individual_bands_notebook.widget(i)
                    if hasattr(w, 'worker') and w.worker and w.worker.isRunning():
                        w.worker.requestInterruption()
                        w.worker.quit()
                        w.worker.wait(2000)  # Wait longer for clean exit
                        if w.worker.isRunning():
                            w.worker.terminate()
                            w.worker.wait(1000)
        except Exception as e:
            print(f"Error stopping individual workers: {e}")
        
        # NEW: Stop view_worker if running
        try:
            if hasattr(self, 'view_worker') and self.view_worker and self.view_worker.isRunning():
                self.view_worker.requestInterruption()
                self.view_worker.quit()
                self.view_worker.wait(2000)
                if self.view_worker.isRunning():
                    self.view_worker.terminate()
                    self.view_worker.wait(1000)
        except Exception as e:
            print(f"Error stopping view_worker: {e}")
        
        # NEW: Stop main load worker
        try:
            if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
                self.worker.requestInterruption()
                self.worker.quit()
                self.worker.wait(2000)
                if self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)
        except Exception as e:
            print(f"Error stopping main worker: {e}")
        
        super().closeEvent(ev)

    def on_load_finished(self, result):
        if getattr(self, '_is_closing', False):
            return
        sender = self.sender()
        if sender is not None and sender is not getattr(self, 'worker', None):
            return
        if sender is not None and getattr(sender, '_load_generation', None) != getattr(self, '_load_generation', None):
            return
        required = (
            self,
            getattr(self, 'band_checkbox_layout', None),
            getattr(self, 'red_band_var', None),
            getattr(self, 'green_band_var', None),
            getattr(self, 'blue_band_var', None),
        )
        if not all(self._qt_alive(obj) for obj in required):
            return
        # Assign results (lightweight, safe on main thread)
        self.band_frames = result['band_frames']
        self.geo_info = result['geo_info']
        self.bands_info = result['bands_info']
        self.base_name = result['base_name']
        # Update tab name if main_app exists
        if self.main_app:
            self.main_app.update_tab_name(self, os.path.basename(self.current_folder))
        # Set number of bands
        self.num_bands = len(self.bands_info) if self.bands_info else 1
        # Debug prints
        print("DEBUG: geo_info (center_lat, center_lon, band_w, band_h, pixel_m) =", self.geo_info)
        print("DEBUG: self.band_frames keys (count) =", len(self.band_frames))
        print("DEBUG: bands_info keys:", self.bands_info)
        print("DEBUG: width_entry, height_entry:", self.width_entry.text(), self.height_entry.text())
        try:
            # Clear old checkboxes
            while self.band_checkbox_layout.count():
                item = self.band_checkbox_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.band_enabled = {}
            # Create new checkboxes for loaded bands
            self.band_enabled = {key: QCheckBox() for key in self.band_frames}
            for key, cb in sorted(self.band_enabled.items(), key=lambda x: x[0]):
                # default unchecked; will apply saved state if available below
                cb.setChecked(False)
                cb.setText(key)
                cb.stateChanged.connect(lambda state, k=key: self.toggle_band(k, state))
                self.band_checkbox_layout.addWidget(cb)
                QApplication.processEvents() # Keep UI responsive during loop
        except RuntimeError:
            return
        # Apply saved enabled states if present (this will trigger toggle_band via stateChanged)
        try:
            saved = getattr(self, '_saved_band_enabled', None)
            if saved and isinstance(saved, dict):
                for k, checked in saved.items():
                    if k in self.band_enabled:
                        try:
                            self.band_enabled[k].setChecked(bool(checked))
                        except Exception:
                            pass
        except Exception:
            pass
        # Update RGB selectors
        avail_keys = sorted(self.band_frames.keys())
        try:
            self.red_band_var.clear()
            self.red_band_var.addItems(avail_keys)
            self.green_band_var.clear()
            self.green_band_var.addItems(avail_keys)
            self.blue_band_var.clear()
            self.blue_band_var.addItems(avail_keys)
        except RuntimeError:
            return
        QApplication.processEvents() # Ensure UI updates
        self.view_worker = ViewUpdateWorker(self, full_refresh=True)
        self.view_worker.finished.connect(self._post_view_update)
        self.view_worker.error.connect(lambda err: QMessageBox.critical(self, "View Update Error", f"Failed to update views: {err}"))
        self.view_worker.progress.connect(self.update_progress)
        self.view_worker.images_ready.connect(self._update_viewers_with_images)
        self.view_worker.start()
        # Lightweight UI updates (safe on main thread)
        try:
            self.update_frame_controls()
            QApplication.processEvents() # Keep UI responsive
        except Exception as e:
            print(f"update_frame_controls() error: {e}")
        # apply fit to screen if requested
        try:
            if self.fit_mode_var.checkedId() == 0:
                self.fit_to_screen()
        except Exception:
            pass
        # set contrast autoscaling if available
        try:
            self.set_auto_contrast()
        except Exception:
            pass
        try:
            if self.current_folder:
                self._notify_iris_folder_loaded(self.current_folder)
        except Exception:
            pass
    def _post_view_update(self):
        """Handle lightweight UI tasks after view update completes."""
        if getattr(self, '_is_closing', False) or not self._qt_alive(self):
            return
        try:
            if self.fit_mode_var.checkedId() == 0:
                self.fit_to_screen()
            self.set_auto_contrast()
            QApplication.processEvents() # Ensure UI refreshes
            self.update_individual_bands_view() # Lightweight: placeholders
            # Apply deferred frame range/current frame if provided by load_state
            try:
                if getattr(self, '_deferred_start_frame', None) is not None and hasattr(self, 'start_frame_entry'):
                    self.start_frame_entry.setValue(int(self._deferred_start_frame))
                if getattr(self, '_deferred_end_frame', None) is not None and hasattr(self, 'end_frame_entry'):
                    self.end_frame_entry.setValue(int(self._deferred_end_frame))
                # After setting range, update frame controls and views
                try:
                    self.update_frame_controls()
                except Exception:
                    pass
                # Set deferred current frame if specified
                if getattr(self, '_deferred_current_frame', None) is not None:
                    try:
                        self.current_frame_index = int(self._deferred_current_frame)
                        if hasattr(self, 'frame_slider'):
                            self.frame_slider.setValue(self.current_frame_index)
                    except Exception:
                        pass
            except Exception:
                pass
            self.update_histogram_view() # Lightweight
        except Exception as e:
            print(f"Post-view update error: {e}")
        gc.collect() # Clean up memory
   
        # Reset progress bar to 0 only after everything is done
        self._finish_progress_session()
        print("DEBUG: View update complete—no hangs!")
    def update_progress(self, value):
        """Unified progress update with monotonic values for a single visible load session."""
        try:
            ivalue = int(value)
        except Exception:
            return
        ivalue = max(0, min(100, ivalue))
        if not self._progress_session_active and ivalue > 0:
            self._start_progress_session()
        # Keep the bar monotonic within a session to avoid "loading twice" perception.
        if self._progress_session_active and ivalue < self._progress_last_value:
            ivalue = self._progress_last_value
        self._progress_last_value = ivalue
        self.progress_bar.setValue(ivalue)
        self.percent_label.setText(f"{ivalue}%")
        if self._progress_session_active:
            self.percent_label.show()
        else:
            self.percent_label.hide()
    def _update_viewers_with_images(self, images_dict):
        """Receive prepared images from worker and update UI on main thread."""
        try:
            if 'all_bands' in images_dict and self.all_bands_viewer:
                pil_display = images_dict['all_bands']['display']
                pil_raw = images_dict['all_bands'].get('raw')
                self.all_bands_viewer.geo_info = self.geo_info
                self.all_bands_viewer.show_image(pil_display, fit_to_screen=(self.fit_mode_var.checkedId() == 0), raw_pil=pil_raw)
                raw_array = images_dict['all_bands'].get('raw_array')
                if raw_array is not None:
                    self.all_bands_viewer.original_raw_data = raw_array
            if 'rgb' in images_dict and self.rgb_preview_viewer:
                pil_display = images_dict['rgb']['display']
                pil_raw = images_dict['rgb'].get('raw')
                self.rgb_preview_viewer.show_image(pil_display, fit_to_screen=(self.fit_mode_var.checkedId() == 0), raw_pil=pil_raw)
                raw_array = images_dict['rgb'].get('raw_array')
                if raw_array is not None:
                    self.rgb_preview_viewer.original_raw_data = raw_array
        except Exception as e:
            print(f"Error updating viewers: {e}")
    def _prepare_all_bands_images(self):
        if not self.band_frames:
            return {'display': None, 'raw': None}
        keys = [k for k in sorted(self.band_frames.keys()) if self.band_frames[k] is not None]
        if not keys:
            return {'display': None, 'raw': None}

        frame_lengths = [len(self.band_frames[k]) for k in keys if hasattr(self.band_frames[k], '__len__')]
        if not frame_lengths:
            return {'display': None, 'raw': None}
        min_frames = min(frame_lengths)
        if self.current_frame_index >= min_frames:
            self.current_frame_index = 0

        stitch_sequence = self.build_stitch_sequence()
        merge_lr = self._is_1_to_4_enabled()
        # Helper to get width of a band entry safely
        def _get_frame_width_for_key(k):
            try:
                return int(self.band_frames[k].w)
            except Exception:
                try:
                    arr = self.band_frames[k]
                    if arr and hasattr(arr[0], 'shape'):
                        return int(arr[0].shape[1])
                except Exception:
                    return 0
        max_width = 0
        for entry in stitch_sequence:
            kind = entry.get('kind')
            if kind in ('full_binned', 'full_unbinned'):
                key = entry.get('key')
                if key in self.band_frames:
                    w = _get_frame_width_for_key(key)
                    if kind == 'full_binned':
                        factor = self._binned_upsample_factor(entry.get('bin_factor', 1))
                        if merge_lr:
                            w *= factor
                    max_width = max(max_width, w)
            elif kind in ('half_left', 'half_right'):
                key = entry.get('key')
                if key in self.band_frames:
                    w = _get_frame_width_for_key(key)
                    max_width = max(max_width, w)
            elif kind == 'paired_unbinned':
                left_k = entry.get('left_key')
                right_k = entry.get('right_key')
                w_left = _get_frame_width_for_key(left_k) if left_k in self.band_frames else 0
                w_right = _get_frame_width_for_key(right_k) if right_k in self.band_frames else 0
                max_width = max(max_width, w_left + w_right)
        enhance = self.contrast_enhance_var.isChecked()
        gap_value = self.gap_var.value()
        parts = []
        raw_parts = []
        processed_bases = set()
        for entry in stitch_sequence:
            base = entry.get('base')
            kind = entry.get('kind')
            if base in processed_bases and kind not in ('half_left', 'half_right'):
                continue
            if kind in ('full_binned', 'full_unbinned'):
                key = entry.get('key')
                if key not in self.band_frames:
                    processed_bases.add(base)
                    continue
                frame = self._safe_get_band_frame(key, self.current_frame_index)
                if frame is None:
                    processed_bases.add(base)
                    continue
                frame = self.apply_offset(frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                raw_frame = self._get_raw_frame(key, self.current_frame_index)
                if raw_frame is None:
                    raw_frame = frame.copy()
                else:
                    raw_frame = self.apply_offset(raw_frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                if kind == 'full_binned':
                    factor = self._binned_upsample_factor(entry.get('bin_factor', 1))
                    if merge_lr:
                        frame = self._upsample_frame(frame, factor)
                        raw_frame = self._upsample_frame(raw_frame, factor)
                display_frame = self.apply_contrast_enhancement(frame) if enhance else frame.copy()
                if display_frame.shape[1] < max_width:
                    pad_w = max_width - display_frame.shape[1]
                    padding = np.zeros((display_frame.shape[0], pad_w), dtype=display_frame.dtype)
                    display_frame = np.hstack([display_frame, padding])
                    raw_padding = np.zeros((raw_frame.shape[0], pad_w), dtype=raw_frame.dtype)
                    raw_frame = np.hstack([raw_frame, raw_padding])
                parts.append(display_frame)
                raw_parts.append(raw_frame)
                continue
            if kind in ('half_left', 'half_right'):
                key = entry.get('key')
                if key not in self.band_frames:
                    processed_bases.add(base)
                    continue
                frame = self._safe_get_band_frame(key, self.current_frame_index)
                if frame is None:
                    processed_bases.add(base)
                    continue
                frame = self.apply_offset(frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                display_frame = self.apply_contrast_enhancement(frame) if enhance else frame.copy()
                raw_frame = self._get_raw_frame(key, self.current_frame_index)
                if raw_frame is None:
                    raw_frame = frame.copy()
                else:
                    raw_frame = self.apply_offset(raw_frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                if display_frame.shape[1] < max_width:
                    pad_w = max_width - display_frame.shape[1]
                    padding = np.zeros((display_frame.shape[0], pad_w), dtype=display_frame.dtype)
                    display_frame = np.hstack([display_frame, padding])
                    raw_padding = np.zeros((raw_frame.shape[0], pad_w), dtype=raw_frame.dtype)
                    raw_frame = np.hstack([raw_frame, raw_padding])
                parts.append(display_frame)
                raw_parts.append(raw_frame)
                continue
            if kind == 'paired_unbinned':
                left_k = entry.get('left_key')
                right_k = entry.get('right_key')
                if left_k not in self.band_frames or right_k not in self.band_frames:
                    processed_bases.add(base)
                    continue
                left_frame = self._safe_get_band_frame(left_k, self.current_frame_index)
                right_frame = self._safe_get_band_frame(right_k, self.current_frame_index)
                if left_frame is None or right_frame is None:
                    processed_bases.add(base)
                    continue
                left_frame = self.apply_offset(left_frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                right_frame = self.apply_offset(right_frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                display_left = self.apply_contrast_enhancement(left_frame) if enhance else left_frame.copy()
                display_right = self.apply_contrast_enhancement(right_frame) if enhance else right_frame.copy()
                raw_left = self._get_raw_frame(left_k, self.current_frame_index)
                if raw_left is None:
                    raw_left = left_frame.copy()
                else:
                    raw_left = self.apply_offset(raw_left, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                raw_right = self._get_raw_frame(right_k, self.current_frame_index)
                if raw_right is None:
                    raw_right = right_frame.copy()
                else:
                    raw_right = self.apply_offset(raw_right, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)

                max_h = max(display_left.shape[0], display_right.shape[0])
                display_left = self._pad_to_height(display_left, max_h)
                display_right = self._pad_to_height(display_right, max_h)
                raw_left = self._pad_to_height(raw_left, max_h)
                raw_right = self._pad_to_height(raw_right, max_h)

                display_full = np.hstack([display_left, display_right])
                raw_full = np.hstack([raw_left, raw_right])
                if display_full.shape[1] < max_width:
                    pad_w = max_width - display_full.shape[1]
                    padding = np.zeros((display_full.shape[0], pad_w), dtype=display_full.dtype)
                    display_full = np.hstack([display_full, padding])
                    raw_padding = np.zeros((raw_full.shape[0], pad_w), dtype=raw_full.dtype)
                    raw_full = np.hstack([raw_full, raw_padding])
                parts.append(display_full)
                raw_parts.append(raw_full)
                processed_bases.add(base)
                continue
            key = entry.get('key')
            if key in self.band_frames:
                frame = self._safe_get_band_frame(key, self.current_frame_index)
                if frame is None:
                    processed_bases.add(base)
                    continue
                frame = self.apply_offset(frame, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                display_frame = self.apply_contrast_enhancement(frame) if enhance else frame.copy()
                if display_frame.shape[1] < max_width:
                    pad_w = max_width - display_frame.shape[1]
                    padding = np.zeros((display_frame.shape[0], pad_w), dtype=display_frame.dtype)
                    display_frame = np.hstack([display_frame, padding])
                parts.append(display_frame)
                raw_fallback = self._get_raw_frame(key, self.current_frame_index)
                if raw_fallback is None:
                    raw_fallback = frame.copy()
                else:
                    raw_fallback = self.apply_offset(raw_fallback, self.band_offsets.get(base, {'x':0, 'y':0})['x'], self.band_offsets.get(base, {'x':0, 'y':0})['y'], crop_y=True)
                raw_parts.append(raw_fallback)
            processed_bases.add(base)
        for i in range(len(parts) - 1):
            gap_arr = np.zeros((gap_value, parts[i].shape[1]), dtype=parts[i].dtype)
            parts[i] = np.vstack([parts[i], gap_arr])
            raw_gap = np.zeros((gap_value, raw_parts[i].shape[1]), dtype=raw_parts[i].dtype)
            raw_parts[i] = np.vstack([raw_parts[i], raw_gap])
        if parts:
            full_display = np.vstack(parts)
            full_raw = np.vstack(raw_parts)
            pil_display = Image.fromarray(full_display)
            pil_raw = Image.fromarray(full_raw)
            del full_display, parts, raw_parts
            gc.collect()
            return {'display': pil_display, 'raw': pil_raw, 'raw_array': full_raw}
        return {'display': None, 'raw': None}
    def _prepare_rgb_images(self):
        if not self.band_frames:
            return {'display': None, 'raw': None}
        self.rgb_bands["R"] = self.red_band_var.currentText()
        self.rgb_bands["G"] = self.green_band_var.currentText()
        self.rgb_bands["B"] = self.blue_band_var.currentText()
        keys = sorted(self.band_frames.keys())
        frame_lengths = [len(self.band_frames[k]) for k in keys if self.band_frames.get(k) is not None and hasattr(self.band_frames[k], '__len__')]
        if not frame_lengths:
            self.current_frame_index = 0
        else:
            min_frames = min(frame_lengths)
            if self.current_frame_index >= min_frames:
                self.current_frame_index = 0
        rgb_mode = "Single" if self.rgb_frame_mode_single.isChecked() else "All"
        enhance = self.contrast_enhance_var.isChecked()
        offset_r_x = self.rgb_offset_r_x.value()
        offset_r_y = self.rgb_offset_r_y.value()
        offset_g_x = self.rgb_offset_g_x.value()
        offset_g_y = self.rgb_offset_g_y.value()
        offset_b_x = self.rgb_offset_b_x.value()
        offset_b_y = self.rgb_offset_b_y.value()
        try:
            def build_channel_stack(channel):
                band_key = self.rgb_bands[channel]
                base_key = band_key.rsplit('_', 1)[0] if '_' in band_key else band_key
                if band_key not in self.band_frames or not self.band_frames[band_key]:
                    return None, None
                h, w = self.band_frames[band_key][0].shape if len(self.band_frames[band_key]) > 0 else (0, 0)
                gap = self.gap_var.value() if rgb_mode == "All" else 0
                if rgb_mode == "Single":
                    if self.current_frame_index >= len(self.band_frames[band_key]):
                        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
                    frame = self.band_frames[band_key][self.current_frame_index]
                    frame = self.apply_offset(frame, self.band_offsets.get(base_key, {'x':0, 'y':0})['x'],
                                             self.band_offsets.get(base_key, {'x':0, 'y':0})['y'], crop_y=True)
                    display_frame = self.apply_contrast_enhancement(frame) if enhance else frame.copy()
                    raw_frame = self._get_raw_frame(band_key, self.current_frame_index)
                    if raw_frame is None:
                        raw_frame = frame.copy()
                    else:
                        raw_frame = self.apply_offset(raw_frame, self.band_offsets.get(base_key, {'x':0, 'y':0})['x'],
                                                     self.band_offsets.get(base_key, {'x':0, 'y':0})['y'], crop_y=True)
                    return display_frame, raw_frame
                start_frame = self.start_frame_entry.value() - 1
                end_frame = self.end_frame_entry.value() - 1
                if start_frame < 0 or end_frame >= len(self.band_frames[band_key]) or start_frame > end_frame:
                    return None, None
                parts_display = []
                parts_raw = []
                for i in range(start_frame, end_frame + 1):
                    frame = self.band_frames[band_key][i]
                    frame = self.apply_offset(frame, self.band_offsets.get(base_key, {'x':0, 'y':0})['x'],
                                             self.band_offsets.get(base_key, {'x':0, 'y':0})['y'], crop_y=True)
                    display_frame = self.apply_contrast_enhancement(frame) if enhance else frame.copy()
                    raw_frame = self._get_raw_frame(band_key, i)
                    if raw_frame is None:
                        raw_frame = frame.copy()
                    else:
                        raw_frame = self.apply_offset(raw_frame, self.band_offsets.get(base_key, {'x':0, 'y':0})['x'],
                                                     self.band_offsets.get(base_key, {'x':0, 'y':0})['y'], crop_y=True)
                    if i < end_frame:
                        gap_arr = np.zeros((gap, display_frame.shape[1]), dtype=np.uint8)
                        display_frame = np.vstack([display_frame, gap_arr])
                        raw_gap = np.zeros((gap, raw_frame.shape[1]), dtype=raw_frame.dtype)
                        raw_frame = np.vstack([raw_frame, raw_gap])
                    parts_display.append(display_frame)
                    parts_raw.append(raw_frame)
                if parts_display:
                    full_display = np.vstack(parts_display)
                    full_raw = np.vstack(parts_raw)
                    return full_display, full_raw
                return None, None
            r_display, r_raw = build_channel_stack("R")
            g_display, g_raw = build_channel_stack("G")
            b_display, b_raw = build_channel_stack("B")
            if r_display is None and g_display is None and b_display is None:
                return {'display': None, 'raw': None}
            h, w = r_display.shape if r_display is not None else (g_display.shape if g_display is not None else b_display.shape)
            if r_display is None:
                r_display = np.zeros((h, w), dtype=np.uint8)
                r_raw = np.zeros((h, w), dtype=(g_raw.dtype if g_raw is not None else (b_raw.dtype if b_raw is not None else np.uint8)))
            if g_display is None:
                g_display = np.zeros((h, w), dtype=np.uint8)
                g_raw = np.zeros((h, w), dtype=(r_raw.dtype if r_raw is not None else (b_raw.dtype if b_raw is not None else np.uint8)))
            if b_display is None:
                b_display = np.zeros((h, w), dtype=np.uint8)
                b_raw = np.zeros((h, w), dtype=(r_raw.dtype if r_raw is not None else (g_raw.dtype if g_raw is not None else np.uint8)))
            offsets_x = [offset_r_x, offset_g_x, offset_b_x]
            offsets_y = [offset_r_y, offset_g_y, offset_b_y]
            min_x = min(0, *offsets_x)
            min_y = min(0, *offsets_y)
            max_x = max(0, *offsets_x) + w
            max_y = max(0, *offsets_y) + h
            total_w = max_x - min_x
            total_h = max_y - min_y
            def pad_channel(channel_arr, off_x, off_y):
                pad_left = off_x - min_x
                pad_right = total_w - (pad_left + w)
                pad_top = off_y - min_y
                pad_bottom = total_h - (pad_top + h)
                return np.pad(channel_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            r_padded_display = pad_channel(r_display, offset_r_x, offset_r_y)
            g_padded_display = pad_channel(g_display, offset_g_x, offset_g_y)
            b_padded_display = pad_channel(b_display, offset_b_x, offset_b_y)
            r_padded_raw = pad_channel(r_raw, offset_r_x, offset_r_y)
            g_padded_raw = pad_channel(g_raw, offset_g_x, offset_g_y)
            b_padded_raw = pad_channel(b_raw, offset_b_x, offset_b_y)
            rgb_array_display = np.stack([r_padded_display, g_padded_display, b_padded_display], axis=-1)
            rgb_array_raw = np.stack([r_padded_raw, g_padded_raw, b_padded_raw], axis=-1)
            pil_display = Image.fromarray(rgb_array_display)
            # Keep high-bit raw RGB as ndarray for pixel-info math; PIL preview stays 8-bit.
            pil_raw = pil_display
            raw_array = rgb_array_raw
            del r_padded_display, g_padded_display, b_padded_display, rgb_array_display
            del r_padded_raw, g_padded_raw, b_padded_raw
            gc.collect()
            return {'display': pil_display, 'raw': pil_raw, 'raw_array': raw_array}
        except Exception as e:
            print(f"Error in _prepare_rgb_images: {e}")
            return {'display': None, 'raw': None}
    def on_load_error(self, err):
        if getattr(self, '_is_closing', False):
            return
        sender = self.sender()
        if sender is not None and sender is not getattr(self, 'worker', None):
            return
        self._finish_progress_session()
        QMessageBox.critical(self, "Error", err)
        placeholder = Image.fromarray(np.zeros((int(self.height_entry.text()), int(self.width_entry.text()) // 2), dtype=np.uint8))
        try:
            self.all_bands_viewer.show_image(placeholder, fit_to_screen=(self.fit_mode_var.checkedId() == 0))
        except Exception:
            pass
    def toggle_terminal(self):
        try:
            expanded = bool(getattr(self, "_terminal_expanded", False))
            total = max(240, self.display_splitter.height())
            if expanded:
                self.terminal_widget.hide()
                btn_h = int(getattr(self, "_terminal_button_height", 30))
                self.display_splitter.setSizes([max(120, total - btn_h), btn_h])
                self.display_splitter.handle(1).setEnabled(False)
                self.terminal_btn.setText("Terminal ↑")
                self._terminal_expanded = False
            else:
                self.terminal_widget.show()
                bottom = max(160, int(total * 0.30))
                top = max(120, total - bottom)
                self.display_splitter.setSizes([top, bottom])
                self.display_splitter.handle(1).setEnabled(True)
                self.terminal_btn.setText("Terminal ↓")
                self._terminal_expanded = True
                try:
                    self.terminal_widget.focus_input()
                except Exception:
                    pass
        except Exception as e:
            print(f"Terminal toggle error: {e}")
    def export_current_image(self):
        if not self.current_folder or not self.base_name:
            QMessageBox.critical(self, "Error", "No image loaded. Please select a folder first.")
            return
        image = None
        tab_suffix = None
        current_tab = self.view_tabs.currentIndex()
        if current_tab < 0:
            QMessageBox.critical(self, "Error", "No tab selected.")
            return
        tab_name = self.view_tabs.tabText(current_tab)
        # All-bands tab
        if tab_name == "All Bands":
            image = getattr(self.all_bands_viewer, 'current_pil_image', None)
            tab_suffix = "all_bands"
            if image is None:
                QMessageBox.critical(self, "Error", "No image available in All Bands tab")
                return
        # Individual bands tab
        elif tab_name == "Individual Bands":
            if self.individual_bands_notebook.count() == 0:
                QMessageBox.critical(self, "Error", "No individual bands enabled or loaded.")
                return
            current_band_tab = self.individual_bands_notebook.currentIndex()
            if current_band_tab < 0:
                QMessageBox.critical(self, "Error", "No individual band selected.")
                return
            band_widget = self.individual_bands_notebook.widget(current_band_tab)
            if band_widget is None:
                QMessageBox.critical(self, "Error", "No band widget found.")
                return
            viewer = band_widget.findChild(GraphicsImageViewer)
            if viewer is None:
                QMessageBox.critical(self, "Error", "No image viewer found in band tab.")
                return
            image = viewer.current_pil_image
            if image is None:
                QMessageBox.critical(self, "Error", "No image available in the current band.")
                return
            try:
                tab_text = self.individual_bands_notebook.tabText(current_band_tab)
                band_key = f"b{tab_text.split()[1]}"
            except Exception:
                band_key = "unknown"
            tab_suffix = f"individual_bands_{band_key}"
        # RGB preview tab
        elif tab_name == "RGB Fusion":
            image = getattr(self.rgb_preview_viewer, 'current_pil_image', None)
            tab_suffix = "rgb"
            if image is None:
                QMessageBox.critical(self, "Error", "No image available in RGB Fusion tab")
                return
        # Histogram tab: not supported
        elif tab_name == "Histogram":
            QMessageBox.critical(self, "Error", "Histogram export is not supported")
            return
        # Help tab or others: not supported
        else:
            QMessageBox.critical(self, "Error", f"Export not supported for tab: {tab_name}")
            return
        # Save with separate filter options for dropdown
        default_filename = f"{self.base_name}_{tab_suffix}.png"
        filters = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff);;All Files (*)"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Current Image As", default_filename, filters
        )
        if filename:
            try:
                # Ensure proper PIL image based on data
                if not hasattr(image, 'save'):
                    # Fallback: convert numpy array to PIL
                    if len(image.shape) == 2:
                        # Grayscale
                        pil_image = Image.fromarray(image, mode='L')
                    elif len(image.shape) == 3 and image.shape[2] == 3:
                        # RGB
                        pil_image = Image.fromarray(image)
                    else:
                        # Default to RGB if unsure
                        pil_image = Image.fromarray(image)
                    image = pil_image
                # Save with format based on extension
                image.save(filename)
                QMessageBox.information(self, "Success", f"Image exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export image: {e}")
    def update_frame_controls(self):
        if not self.band_frames:
            self.frame_slider.setRange(0, 0)
            self.frame_label.setText("0/0")
            self.start_frame_entry.setValue(1)
            self.end_frame_entry.setValue(1)
            return
       
        keys = sorted(k for k in self.band_frames.keys() if self.band_frames[k] is not None)
        if not keys:
            return

        # Use minimum frame count among all bands to prevent out-of-range index selections.
        frame_counts = [len(self.band_frames[k]) for k in keys if hasattr(self.band_frames[k], '__len__')]
        if not frame_counts:
            return

        max_frames = min(frame_counts)
        if self.current_frame_index >= max_frames:
            self.current_frame_index = 0

        self.frame_slider.setRange(0, max_frames-1)
        self.frame_label.setText(f"{self.current_frame_index+1}/{max_frames}")
        self.start_frame_entry.setRange(1, max_frames)
        self.start_frame_entry.setValue(1)
        self.end_frame_entry.setRange(1, max_frames)
        self.end_frame_entry.setValue(max_frames)
   
    def on_frame_slider_changed(self, value):
        self.current_frame_index = value
        self.update_views()
        self.frame_label.setText(f"{self.current_frame_index+1}/{self.frame_slider.maximum()+1}")
        if self.fit_mode_var.checkedId() == 0:
            self.fit_to_screen()
        try:
            main_app = getattr(self, 'main_app', None)
            iris = getattr(main_app, 'iris', None) if main_app is not None else None
            tab_widget = getattr(main_app, 'tab_widget', None) if main_app is not None else None
            tab_index = tab_widget.indexOf(self) if tab_widget is not None else -1
            if iris is not None and tab_index >= 0:
                iris.notify_frame_changed(tab_index, value)
        except Exception:
            pass
   
    def change_frame(self, delta):
        new_index = self.current_frame_index + delta
        max_frames = self.frame_slider.maximum() + 1 if self.band_frames else 0
       
        if max_frames > 0:
            if new_index < 0:
                new_index = 0
            elif new_index >= max_frames:
                new_index = max_frames - 1
           
            self.current_frame_index = new_index
            self.frame_slider.setValue(new_index)
           
            # Call optimized update during playback
            self.update_views(full_refresh=not self.playback_mode)
            if self.fit_mode_var.checkedId() == 0 and not self.playback_mode:
                self.fit_to_screen()
   
    def toggle_play(self):
        if not self.band_frames:
            return
       
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("⏸ Pause")
            # Start periodic playback
            try:
                self.play_timer.start(self.play_delay)
            except Exception:
                # Fallback: single-step if timer fails
                self.play_next_frame()
        else:
            self.play_btn.setText("▶ Play")
            try:
                self.play_timer.stop()
            except Exception:
                pass
   
    def play_next_frame(self):
        if not self.playing or not self.band_frames:
            self.play_btn.setText("▶ Play")
            self.playing = False
            return
       
        max_frames = self.frame_slider.maximum() + 1
        if self.current_frame_index < max_frames - 1:
            self.change_frame(1)
        else:
            self.current_frame_index = 0
            self.frame_slider.setValue(0)
            self.update_views()
    def validate_frame_range(self):
        if not self.band_frames:
            return
        # If check passes, update views
        self.update_views(full_refresh=True)
