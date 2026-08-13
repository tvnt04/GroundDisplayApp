import os
import numpy as np
from PIL import Image
import gc
import psutil
import tempfile
# import cv2  # pip install opencv-python
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QSpinBox, QLineEdit, QComboBox, QFileDialog, QMessageBox, QLabel, QProgressBar, 
    QSizePolicy, QSlider, QCheckBox, QGroupBox, QFormLayout, QToolButton, QMenu, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QUrl
# from PyQt5.QtMultimediaWidgets import QVideoWidget
# from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from PyQt5.QtGui import QPainter, QImage, QPixmap
from image_viewer import GraphicsImageViewer
from utils import LazyFrames, load_folder_params, save_params_for_path, add_recent, get_recents_for_mode, select_from_history
from help_tab import create_help_tab
import sys
import json
import re

# Constants (default values)
DEFAULT_WIDTH = 8448
DEFAULT_HEIGHT = 384
DEFAULT_BIT_DEPTH = 10
GAP = 0  # Pixels between bands

def unpack_by_bitdepth(data, w, h, bitdepth):
    try:
        if bitdepth == 8:
            return np.frombuffer(data, dtype=np.uint8).reshape((-1, h, w))
        elif bitdepth == 32:
            total_pixels = w * h
            bytes_per_frame = total_pixels * 4
            num_frames = len(data) // bytes_per_frame
            frames = []
            for i in range(num_frames):
                start = i * bytes_per_frame
                chunk = data[start:start + bytes_per_frame]
                raw = np.frombuffer(chunk, dtype='<u4', count=total_pixels)
                if raw.size < total_pixels:
                    continue
                raw = raw.reshape((h, w))
                scaled = np.clip(
                    raw.astype(np.float64) * (255.0 / 4294967295.0),
                    0, 255
                ).astype(np.uint8)
                frames.append(scaled)
            return frames
        elif bitdepth == 16:
            total_pixels = w * h
            bytes_per_frame = total_pixels * 2
            num_frames = len(data) // bytes_per_frame
            frames = []
            for i in range(num_frames):
                start = i * bytes_per_frame
                chunk = data[start:start + bytes_per_frame]
                raw = np.frombuffer(chunk, dtype='<u2', count=total_pixels)
                if raw.size < total_pixels:
                    continue
                raw = raw.reshape((h, w))
                valid = raw[raw > 0]
                if valid.size == 0:
                    frames.append(np.zeros((h, w), dtype=np.uint8))
                    continue
                low = np.percentile(valid, 2)
                high = np.percentile(valid, 98)
                if high <= low:
                    high = low + 1
                scaled = np.clip(
                    (raw.astype(np.float32) - low) * (255.0 / (high - low)),
                    0, 255
                ).astype(np.uint8)
                frames.append(scaled)
            return frames
        elif bitdepth == 10:
            total_pixels = w * h
            bytes_per_frame = (total_pixels * 10) // 8
            num_frames = len(data) // bytes_per_frame
            frames = []
            for i in range(num_frames):
                start = i * bytes_per_frame
                packed_data = data[start : start + bytes_per_frame]
                num_full_groups = len(packed_data) // 5
                d = np.frombuffer(packed_data[:num_full_groups*5], dtype=np.uint8).reshape(-1,5)
                expanded_data = np.zeros(d.shape[0], dtype=np.uint64)
                for j in range(5):
                    expanded_data += d[:,j].astype(np.uint64) << (8 * j)
                unpacked = np.zeros(num_full_groups * 4, dtype=np.uint16)
                for j in range(4):
                    unpacked[j::4] = (expanded_data >> (10 * j)) & 0x3FF
                remaining_bytes = len(packed_data) % 5
                if remaining_bytes:
                    last_bits = int.from_bytes(packed_data[-remaining_bytes:], 'little')
                    extra_pixels = np.array([(last_bits >> (10 * k)) & 0x3FF for k in range((remaining_bytes * 8) // 10)])
                    unpacked = np.concatenate([unpacked, extra_pixels[:total_pixels - len(unpacked)]])
                if len(unpacked) < total_pixels:
                    unpacked = np.pad(unpacked, (0, total_pixels - len(unpacked)), 'constant')
                scaled = np.clip((unpacked * 255.0 / 1023), 0, 255).astype(np.uint8)
                frames.append(scaled.reshape(h, w))
            return frames
        elif bitdepth == 12:
            total_pixels = w * h
            frame_size = (total_pixels * 12) // 8
            num_frames = len(data) // frame_size
            frames = []
            for i in range(num_frames):
                chunk = data[i * frame_size:(i + 1) * frame_size]
                d = np.frombuffer(chunk, dtype=np.uint8)
                num_groups = len(d) // 3
                d = d[:num_groups*3].reshape(-1, 3)
                pixels = np.zeros(num_groups * 2, dtype=np.uint16)
                for j in range(num_groups):
                    pixels[j*2] = ((d[j,0] | (d[j,1] & 0x0F) << 8)) & 0xFFF
                    pixels[j*2 + 1] = ((d[j,1] >> 4) | (d[j,2] << 4)) & 0xFFF
                if len(pixels) < total_pixels:
                    pixels = np.pad(pixels, (0, total_pixels - len(pixels)), 'constant')
                scaled = np.clip((pixels * 255.0 / 4095), 0, 255).astype(np.uint8)
                frames.append(scaled.reshape(h, w))
            return frames
        else:
            raise ValueError(f"Unsupported bit depth: {bitdepth}")
    except Exception as e:
        print(f"Error in unpack_by_bitdepth: {e}")
        return []


class FrameSource:
    """Lazy frame source for a single band file. Supports len() and indexing."""
    def __init__(self, path, w, h, bitdepth):
        self.path = path
        self.w = int(w)
        self.h = int(h)
        self.bitdepth = int(bitdepth)
        total_pixels = self.w * self.h
        if self.bitdepth == 8:
            self.frame_size = total_pixels
        elif self.bitdepth == 16:
            self.frame_size = total_pixels * 2
        elif self.bitdepth == 32:
            self.frame_size = total_pixels * 4
        elif self.bitdepth == 10:
            self.frame_size = (total_pixels * 10) // 8
        elif self.bitdepth == 12:
            self.frame_size = (total_pixels * 12) // 8
        else:
            raise ValueError("Unsupported bitdepth for FrameSource")
        try:
            self.file_size = os.path.getsize(self.path)
            self.num_frames = max(0, self.file_size // self.frame_size)
        except Exception:
            self.file_size = 0
            self.num_frames = 0

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.num_frames + idx
        if idx < 0 or idx >= self.num_frames:
            raise IndexError("frame index out of range")
        with open(self.path, 'rb') as fh:
            fh.seek(idx * self.frame_size)
            data = fh.read(self.frame_size)
        frames = unpack_by_bitdepth(data, self.w, self.h, self.bitdepth)
        return frames[0] if frames else np.zeros((self.h, self.w), dtype=np.uint8)

def check_memory_requirement(expected_bytes, parent=None):
    avail = psutil.virtual_memory().available
    if avail < expected_bytes * 1.5:  # 1.5x buffer
        QMessageBox.warning(parent, "Memory Warning", f"Required: {expected_bytes/1e9:.2f} GB\nAvailable: {avail/1e9:.2f} GB\nTry smaller dataset.")
        return False
    return True

class LoadingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, folder, width, height, bitdepth):
        super().__init__()
        self.folder = folder
        self.width = width
        self.height = height
        self.bitdepth = bitdepth

    def run(self):
        try:
            # Find ProcMode and base_name
            proc_mode, base_name = None, None
            for f in os.listdir(self.folder):
                if f.endswith(".json") and f != "parameters.json":
                    with open(os.path.join(self.folder, f), "r") as jf:
                        try:
                            config = json.load(jf)
                        except Exception:
                            config = {}
                        proc_mode = config.get("ProcMode")
                        base_name = os.path.splitext(f)[0]
                    if proc_mode:
                        break
            if not proc_mode:
                for f in os.listdir(self.folder):
                    if f.endswith(".log"):
                        with open(os.path.join(self.folder, f), "r") as lf:
                            for line in lf:
                                if "Arguments received from parameter file" in line:
                                    proc_mode = line.split("file")[-1].strip(": \n")
                                    base_name = os.path.splitext(f)[0]
                                    break
                        if proc_mode:
                            break
            if not proc_mode:
                self.error.emit("Could not extract ProcMode from JSON or LOG")
                return

            # Parse band_selection and binning
            band_selection = None
            binning = None
            parts = proc_mode.strip().split()
            if len(parts) > 12:
                band_selection = int(parts[6])
                binning = int(parts[12])
            else:
                nums = [int(tok) for tok in parts if tok.isdigit()]
                if len(nums) >= 2:
                    band_selection, binning = nums[0], nums[1]
            if band_selection is None or binning is None:
                self.error.emit("Invalid ProcMode format")
                return

            # Load band files
            band_frames = {}
            total_bands = 7
            loaded_bands = 0
            files_checked = []
            for i in range(total_bands):
                self.progress.emit(int((i / total_bands) * 50))  # 50% for metadata/bands
                if not ((band_selection >> i) & 1):
                    continue
                is_binned = (binning >> i) & 1
                try:
                    if is_binned:
                        fname = os.path.join(self.folder, f"{base_name}.band{i}2")
                        files_checked.append(fname)
                        if os.path.exists(fname) and os.path.getsize(fname) > 0:
                            # create lazy LazyFrames for binned file (memmap-backed)
                            src = LazyFrames(fname, self.width // 2, self.height // 2, self.bitdepth)
                            if len(src) > 0:
                                band_frames[f"b{i}_binned"] = src
                                loaded_bands += 1
                    else:
                        lfile = os.path.join(self.folder, f"{base_name}.band{i}0")
                        rfile = os.path.join(self.folder, f"{base_name}.band{i}1")
                        files_checked.extend([lfile, rfile])
                        if os.path.exists(lfile) and os.path.getsize(lfile) > 0:
                            src = LazyFrames(lfile, self.width // 2, self.height, self.bitdepth)
                            if len(src) > 0:
                                band_frames[f"b{i}_left"] = src
                                loaded_bands += 1
                        if os.path.exists(rfile) and os.path.getsize(rfile) > 0:
                            src = LazyFrames(rfile, self.width // 2, self.height, self.bitdepth)
                            if len(src) > 0:
                                band_frames[f"b{i}_right"] = src
                                loaded_bands += 1
                except Exception as e:
                    print(f"Band {i} error: {e}")
                    continue

            # Build bands_info (50-100% progress)
            if band_frames:
                base_keys = []
                for k in band_frames.keys():
                    base = k.split('_')[0]
                    if base not in base_keys:
                        base_keys.append(base)
                num_bands = len(base_keys) if base_keys else 1
                bands_info = {}
                for idx, base in enumerate(base_keys):
                    variants = [k for k in band_frames.keys() if k.startswith(base)]
                    has_left = any('left' in v.lower() for v in variants)
                    has_right = any('right' in v.lower() for v in variants)
                    is_split = has_left and has_right
                    bin_factor = 2 if any('binned' in v.lower() for v in variants) else 1
                    bands_info[base] = {
                        'index': idx,
                        'variants': variants,
                        'binned': bin_factor > 1,
                        'split': is_split,
                        'bin_factor': bin_factor
                    }
                self.progress.emit(100)
                self.finished.emit({
                    'band_frames': band_frames,
                    'bands_info': bands_info,
                    'max_frames': max(len(v) for v in band_frames.values()) if band_frames else 0,
                    'files_checked': files_checked
                })
            else:
                self.error.emit(f"No valid band frames loaded. Checked: {', '.join(files_checked)}")
        except Exception as e:
            self.error.emit(str(e))

class VideoGenerationThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)  # Path to generated video
    error = pyqtSignal(str)

    def __init__(self, band_frames, stitch_sequence, max_frames, max_width, fps, total_h, is_rgb, rgb_bands, rgb_offsets, start_frame, end_frame, parent=None):
        super().__init__(parent)
        self.band_frames = band_frames
        self.stitch_sequence = stitch_sequence
        self.max_frames = max_frames
        self.max_width = max_width
        self.fps = fps
        self.total_h = total_h
        self.is_rgb = is_rgb
        self.rgb_bands = rgb_bands
        self.rgb_offsets = rgb_offsets
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)

    def run(self):
        try:
            import cv2  # Import here to avoid segfault on module load
            parent = self.parent()
            # use provided frame range
            if self.is_rgb:
                # Only consider the selected RGB band sources for frame count
                keys = [self.rgb_bands.get('R'), self.rgb_bands.get('G'), self.rgb_bands.get('B')]
                srcs = [self.band_frames[k] for k in keys if k in self.band_frames]
                if not srcs:
                    self.error.emit("Selected RGB bands are not available for generation")
                    return
                min_frames = min(len(v) for v in srcs)
            else:
                min_frames = min(len(v) for v in self.band_frames.values()) if self.band_frames else 0
            start = max(0, min(self.start_frame, min_frames - 1))
            end = max(0, min(self.end_frame, min_frames - 1))
            if end < start:
                start, end = end, start
            frame_count = end - start + 1
            if self.max_width <= 0 or self.total_h <= 0:
                self.error.emit("Invalid video dimensions: width={}, height={}".format(self.max_width, self.total_h))
                return
            _, temp_path = tempfile.mkstemp(suffix='.mp4')
            # Always write color (BGR) frames for compatibility; grayscale will be converted to 3-channel
            writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.max_width, self.total_h), isColor=True)
            for frame_offset, frame_idx in enumerate(range(start, end + 1)):
                if self.is_rgb:
                    # Debug/log current selection once
                    if frame_offset == 0:
                        try:
                            print(f"VideoGeneration: RGB mode={self.is_rgb}, R={self.rgb_bands.get('R')}, G={self.rgb_bands.get('G')}, B={self.rgb_bands.get('B')}, offsets={self.rgb_offsets}, frames={start}-{end}")
                        except Exception:
                            pass
                    # RGB fusion mode - assemble into a 3-channel frame efficiently
                    r_key = self.rgb_bands["R"]
                    g_key = self.rgb_bands["G"]
                    b_key = self.rgb_bands["B"]
                    r_offset = self.rgb_offsets["R"]
                    g_offset = self.rgb_offsets["G"]
                    b_offset = self.rgb_offsets["B"]

                    r_frame = self.band_frames[r_key][frame_idx] if r_key in self.band_frames else None
                    g_frame = self.band_frames[g_key][frame_idx] if g_key in self.band_frames else None
                    b_frame = self.band_frames[b_key][frame_idx] if b_key in self.band_frames else None

                    if r_frame is None and g_frame is None and b_frame is None:
                        out_frame = np.zeros((self.total_h, self.max_width, 3), dtype=np.uint8)
                    else:
                        # Apply offsets and pad to output size
                        r_frame = self.apply_offset_simple(r_frame, r_offset['x'], r_offset['y']) if r_frame is not None else None
                        g_frame = self.apply_offset_simple(g_frame, g_offset['x'], g_offset['y']) if g_frame is not None else None
                        b_frame = self.apply_offset_simple(b_frame, b_offset['x'], b_offset['y']) if b_frame is not None else None

                        # Determine target dims
                        max_h = self.total_h
                        max_w = self.max_width
                        # Create output and fill channels directly (R,G,B -> 0,1,2)
                        out_frame = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                        if r_frame is not None:
                            r_pad = self.pad_frame(r_frame, max_h, max_w)
                            out_frame[..., 0] = r_pad
                        if g_frame is not None:
                            g_pad = self.pad_frame(g_frame, max_h, max_w)
                            out_frame[..., 1] = g_pad
                        if b_frame is not None:
                            b_pad = self.pad_frame(b_frame, max_h, max_w)
                            out_frame[..., 2] = b_pad

                    # write as BGR by reversing channels without costly color conversion
                    writer.write(out_frame[..., ::-1])
                else:
                    # Original grayscale stitching — follow Band Mode logic using parent's helpers
                    parts = []
                    processed_bases = set()
                    for entry in self.stitch_sequence:
                        base = entry.get('base')
                        kind = entry.get('kind')
                        if base in processed_bases:
                            continue
                        if kind in ('full_binned', 'full_unbinned'):
                            key = entry.get('key')
                            if key not in self.band_frames:
                                continue
                            frame = self.band_frames[key][frame_idx]
                            # use parent's apply_offset with crop_y to match band mode
                            try:
                                frame = parent.apply_offset(frame, 0, 0, crop_y=True) if hasattr(parent, 'apply_offset') else frame
                            except Exception:
                                pass
                            # pad width if needed
                            if frame.shape[1] < self.max_width:
                                pad_w = self.max_width - frame.shape[1]
                                padding = np.zeros((frame.shape[0], pad_w), dtype=frame.dtype)
                                frame = np.hstack([frame, padding])
                            parts.append(frame)
                            processed_bases.add(base)
                        elif kind == 'split_unbinned':
                            left_k = entry.get('left_key')
                            right_k = entry.get('right_key')
                            if left_k not in self.band_frames or right_k not in self.band_frames:
                                continue
                            left_frame = self.band_frames[left_k][frame_idx]
                            right_frame = self.band_frames[right_k][frame_idx]
                            try:
                                left_frame = parent.apply_offset(left_frame, 0, 0, crop_y=True) if hasattr(parent, 'apply_offset') else left_frame
                                right_frame = parent.apply_offset(right_frame, 0, 0, crop_y=True) if hasattr(parent, 'apply_offset') else right_frame
                            except Exception:
                                pass
                            # hstack left and right
                            display_full = np.hstack([left_frame, right_frame])
                            # pad width if needed
                            if display_full.shape[1] < self.max_width:
                                pad_w = self.max_width - display_full.shape[1]
                                padding = np.zeros((display_full.shape[0], pad_w), dtype=display_full.dtype)
                                display_full = np.hstack([display_full, padding])
                            parts.append(display_full)
                            processed_bases.add(base)

                    # Build full_display by vertical stacking with GAP rows
                    if parts:
                        full_display = parts[0]
                        for p in parts[1:]:
                            gap_arr = np.zeros((GAP, full_display.shape[1]), dtype=np.uint8) if GAP > 0 else None
                            if gap_arr is not None:
                                full_display = np.vstack([full_display, gap_arr])
                            full_display = np.vstack([full_display, p])
                    else:
                        full_display = np.zeros((self.total_h, self.max_width), dtype=np.uint8)

                    # Convert grayscale to BGR for writer
                    try:
                        bgr = cv2.cvtColor(full_display, cv2.COLOR_GRAY2BGR)
                    except Exception:
                        bgr = np.stack([full_display, full_display, full_display], axis=-1)
                    writer.write(bgr)
                self.progress.emit(int((frame_offset + 1) / frame_count * 100))

            writer.release()
            gc.collect()
            self.finished.emit(temp_path)
        except Exception as e:
            self.error.emit(str(e))

    def apply_offset_simple(self, frame, offset_x, offset_y):
        # Simple offset by rolling
        frame = np.roll(frame, offset_x, axis=1)
        frame = np.roll(frame, offset_y, axis=0)
        return frame

    def pad_frame(self, frame, target_h, target_w):
        h, w = frame.shape
        if h < target_h or w < target_w:
            padded = np.zeros((target_h, target_w), dtype=frame.dtype)
            padded[:h, :w] = frame
            return padded
        return frame

class PlaybackApp(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.band_frames = {}
        self.bands_info = {}
        self.max_frames = 0
        self.min_frames = 0
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.bitdepth = DEFAULT_BIT_DEPTH
        self.fps = 30
        self.current_folder = None
        self.video_path = None
        self.stitch_sequence = None
        self.max_width = 0
        self.total_h = 0
        self.rgb_bands = {"R": None, "G": None, "B": None}
        self.rgb_offsets = {"R": {"x": 0, "y": 0}, "G": {"x": 0, "y": 0}, "B": {"x": 0, "y": 0}}
        self.is_rgb_mode = False
        self.init_ui()
        # self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.player.positionChanged.connect(self.update_slider)
        # self.player.durationChanged.connect(self.set_slider_range)
        # self.player.stateChanged.connect(self.handle_state_changed)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.next_frame)
        self.current_frame_index = 0

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

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel, stretch=1)

        hb = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.setToolTip("Select folder containing band files")
        self.select_folder_btn.clicked.connect(self.select_folder)
        hb.addWidget(self.select_folder_btn)
        self.load_menu_btn = QToolButton()
        self.load_menu_btn.setArrowType(Qt.DownArrow)
        self.load_menu_btn.setMaximumWidth(22)
        self.load_menu_btn.setToolTip("Open recent folders")
        self.load_menu_btn.clicked.connect(self._show_recent_menu)
        hb.addWidget(self.load_menu_btn)
        left_layout.addLayout(hb)

        left_layout.addWidget(QLabel("Height:"))
        self.height_entry = QLineEdit(str(self.height))
        left_layout.addWidget(self.height_entry)
        self.height_entry.editingFinished.connect(self.validate_height)

        left_layout.addWidget(QLabel("Width:"))
        self.width_entry = QLineEdit(str(self.width))
        left_layout.addWidget(self.width_entry)
        self.width_entry.editingFinished.connect(self.validate_width)

        left_layout.addWidget(QLabel("Bit Depth:"))
        self.bitdepth_var = QComboBox()
        self.bitdepth_var.addItems(["8", "10", "12", "16", "32"])
        self.bitdepth_var.setCurrentText(str(self.bitdepth))
        left_layout.addWidget(self.bitdepth_var)
        self.bitdepth_var.currentIndexChanged.connect(self.invalidate_video)

        left_layout.addWidget(QLabel("FPS:"))
        self.fps_var = QSpinBox()
        self.fps_var.setRange(1, 60)
        self.fps_var.setValue(self.fps)
        left_layout.addWidget(self.fps_var)
        self.fps_var.valueChanged.connect(self.invalidate_video)

        # Frame range for generation (1-based)
        left_layout.addWidget(QLabel("Start Frame:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(1, 1000000000)
        self.start_frame_spin.setValue(1)
        left_layout.addWidget(self.start_frame_spin)

        left_layout.addWidget(QLabel("End Frame:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(1, 1000000000)
        self.end_frame_spin.setValue(1)
        left_layout.addWidget(self.end_frame_spin)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        left_layout.addWidget(QLabel("Playback Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        self.speed_combo.setEnabled(False)
        left_layout.addWidget(self.speed_combo)

        self.loop_checkbox = QCheckBox("Loop Video")
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.stateChanged.connect(self.toggle_loop)
        left_layout.addWidget(self.loop_checkbox)

        # RGB Fusion controls
        rgb_group = QGroupBox("RGB Fusion")
        rgb_layout = QVBoxLayout()
        rgb_group.setLayout(rgb_layout)
        left_layout.addWidget(rgb_group)

        self.rgb_mode_checkbox = QCheckBox("Enable RGB Fusion")
        self.rgb_mode_checkbox.stateChanged.connect(self.toggle_rgb_mode)
        rgb_layout.addWidget(self.rgb_mode_checkbox)

        band_layout = QFormLayout()
        rgb_layout.addLayout(band_layout)

        self.red_band_combo = QComboBox()
        self.red_band_combo.setEnabled(False)
        band_layout.addRow("Red Band:", self.red_band_combo)

        self.green_band_combo = QComboBox()
        self.green_band_combo.setEnabled(False)
        band_layout.addRow("Green Band:", self.green_band_combo)

        self.blue_band_combo = QComboBox()
        self.blue_band_combo.setEnabled(False)
        band_layout.addRow("Blue Band:", self.blue_band_combo)

        offset_layout = QFormLayout()
        rgb_layout.addLayout(offset_layout)

        self.red_offset_x = QSpinBox()
        self.red_offset_x.setRange(-1000, 1000)
        self.red_offset_x.setValue(0)
        self.red_offset_x.setEnabled(False)
        offset_layout.addRow("R Offset X:", self.red_offset_x)

        self.red_offset_y = QSpinBox()
        self.red_offset_y.setRange(-1000, 1000)
        self.red_offset_y.setValue(0)
        self.red_offset_y.setEnabled(False)
        offset_layout.addRow("R Offset Y:", self.red_offset_y)

        self.green_offset_x = QSpinBox()
        self.green_offset_x.setRange(-1000, 1000)
        self.green_offset_x.setValue(0)
        self.green_offset_x.setEnabled(False)
        offset_layout.addRow("G Offset X:", self.green_offset_x)

        self.green_offset_y = QSpinBox()
        self.green_offset_y.setRange(-1000, 1000)
        self.green_offset_y.setValue(0)
        self.green_offset_y.setEnabled(False)
        offset_layout.addRow("G Offset Y:", self.green_offset_y)

        self.blue_offset_x = QSpinBox()
        self.blue_offset_x.setRange(-1000, 1000)
        self.blue_offset_x.setValue(0)
        self.blue_offset_x.setEnabled(False)
        offset_layout.addRow("B Offset X:", self.blue_offset_x)

        self.blue_offset_y = QSpinBox()
        self.blue_offset_y.setRange(-1000, 1000)
        self.blue_offset_y.setValue(0)
        self.blue_offset_y.setEnabled(False)
        offset_layout.addRow("B Offset Y:", self.blue_offset_y)
        
        # Preview button for RGB fusion (lightweight)
        self.preview_rgb_btn = QPushButton("Preview RGB")
        self.preview_rgb_btn.setToolTip("Preview RGB fusion for the selected frame")
        self.preview_rgb_btn.setEnabled(False)
        self.preview_rgb_btn.clicked.connect(self.preview_rgb)
        rgb_layout.addWidget(self.preview_rgb_btn)
        # Invalidate existing video on band/offset changes
        for cb in (self.red_band_combo, self.green_band_combo, self.blue_band_combo):
            cb.currentIndexChanged.connect(self.invalidate_video)
        for sb in (self.red_offset_x, self.red_offset_y, self.green_offset_x, self.green_offset_y, self.blue_offset_x, self.blue_offset_y):
            sb.valueChanged.connect(self.invalidate_video)

        # store quick mapping of discovered band files (key -> filepath)
        self.quick_band_files = {}

        # Generate button placed at the bottom for user flow
        self.process_btn = QPushButton("Generate Video")
        self.process_btn.setToolTip("Generate video using current settings")
        self.process_btn.clicked.connect(self.generate_video)
        self.process_btn.setEnabled(False)
        left_layout.addWidget(self.process_btn)

        left_layout.addStretch()

        # Right panel (display + help tabs)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, stretch=4)

        self.right_tabs = QTabWidget()
        right_layout.addWidget(self.right_tabs, 1)
        display_tab = QWidget()
        display_layout = QVBoxLayout()
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_tab.setLayout(display_layout)
        self.right_tabs.addTab(display_tab, "Display")

        # pixel info matrix size control (used by GraphicsImageViewer)
        self.matrix_size_var = QSpinBox()
        self.matrix_size_var.setRange(3, 9)
        self.matrix_size_var.setSingleStep(2)
        self.matrix_size_var.setValue(3)
        self.image_viewer = GraphicsImageViewer(pixel_info_callback=self.on_pixel_info, matrix_size_var=self.matrix_size_var)
        display_layout.addWidget(self.image_viewer)

        try:
            self.help_tab = create_help_tab(main_app=getattr(self, "_main_app", None), mode="video")
            self.right_tabs.addTab(self.help_tab, "Help")
        except Exception as e:
            print(f"Video help tab unavailable: {e}")

        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setToolTip("Play or pause video preview")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.backward_btn = QPushButton("◀◀ 10s")
        self.backward_btn.setToolTip("Jump backward by 10 seconds")
        self.backward_btn.clicked.connect(self.skip_backward)
        self.backward_btn.setEnabled(False)
        controls_layout.addWidget(self.backward_btn)

        self.current_time_label = QLabel("00:00")
        controls_layout.addWidget(self.current_time_label)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.frame_slider, stretch=1)

        self.total_time_label = QLabel("/ 00:00")
        controls_layout.addWidget(self.total_time_label)

        self.forward_btn = QPushButton("▶▶ 10s")
        self.forward_btn.setToolTip("Jump forward by 10 seconds")
        self.forward_btn.clicked.connect(self.skip_forward)
        self.forward_btn.setEnabled(False)
        controls_layout.addWidget(self.forward_btn)

        # bottom playback controls intentionally not added for Video Mode
        # (kept as attributes for compatibility but not shown)

    def update_rgb_bands(self):
        if not self.band_frames:
            return
        band_keys = sorted(self.band_frames.keys())
        for combo in [self.red_band_combo, self.green_band_combo, self.blue_band_combo]:
            combo.clear()
            combo.addItems(band_keys)
        # Set defaults
        if len(band_keys) >= 3:
            self.red_band_combo.setCurrentText(band_keys[0])
            self.green_band_combo.setCurrentText(band_keys[1])
            self.blue_band_combo.setCurrentText(band_keys[2])
        elif len(band_keys) >= 1:
            for combo in [self.red_band_combo, self.green_band_combo, self.blue_band_combo]:
                combo.setCurrentText(band_keys[0])

    def preview_rgb(self):
        # Lightweight preview: read first frame from selected band files and show fused RGB
        try:
            if not self.current_folder:
                QMessageBox.warning(self, "No folder", "Select a folder first.")
                return
            r_key = self.red_band_combo.currentText()
            g_key = self.green_band_combo.currentText()
            b_key = self.blue_band_combo.currentText()
            if not (r_key and g_key and b_key):
                QMessageBox.warning(self, "Select bands", "Choose R, G and B bands first.")
                return

            def load_first_frame_for_key(key):
                # Prefer already-loaded lazy source
                try:
                    if key in self.band_frames:
                        src = self.band_frames[key]
                        # if it's a FrameSource or list-like, get current frame
                        idx = getattr(self, 'current_frame_index', 0)
                        idx = max(0, idx)
                        frame = src[idx] if len(src) > idx else None
                        if frame is not None:
                            # apply per-channel offsets
                            ox = 0
                            oy = 0
                            if 'r' in key.lower():
                                ox = self.red_offset_x.value()
                                oy = self.red_offset_y.value()
                            elif 'g' in key.lower():
                                ox = self.green_offset_x.value()
                                oy = self.green_offset_y.value()
                            elif 'b' in key.lower():
                                ox = self.blue_offset_x.value()
                                oy = self.blue_offset_y.value()
                            return self.apply_offset_simple(frame, ox, oy)
                except Exception:
                    pass
                # If no source, fallback to quick mapping file read (first frame)
                path = self.quick_band_files.get(key)
                if not path:
                    for f in os.listdir(self.current_folder):
                        if key in f:
                            path = os.path.join(self.current_folder, f)
                            break
                if not path or not os.path.exists(path):
                    return None
                with open(path, 'rb') as fh:
                    data = fh.read()
                frames = unpack_by_bitdepth(data, self.width // (2 if 'binned' in key else 1), self.height // (2 if 'binned' in key else 1), self.bitdepth)
                if not frames:
                    return None
                frame = frames[0]
                # apply offsets from spinboxes
                if key == self.red_band_combo.currentText():
                    frame = self.apply_offset_simple(frame, self.red_offset_x.value(), self.red_offset_y.value())
                elif key == self.green_band_combo.currentText():
                    frame = self.apply_offset_simple(frame, self.green_offset_x.value(), self.green_offset_y.value())
                elif key == self.blue_band_combo.currentText():
                    frame = self.apply_offset_simple(frame, self.blue_offset_x.value(), self.blue_offset_y.value())
                return frame

            r = load_first_frame_for_key(r_key)
            g = load_first_frame_for_key(g_key)
            b = load_first_frame_for_key(b_key)
            if r is None and g is None and b is None:
                QMessageBox.warning(self, "Preview failed", "Could not load preview frames for selected bands.")
                return

            # Replace missing channels with zeros sized to the available ones
            shapes = [arr.shape for arr in (r, g, b) if arr is not None]
            if not shapes:
                QMessageBox.warning(self, "Preview failed", "No frames available.")
                return
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)

            def ensure(arr):
                if arr is None:
                    return np.zeros((max_h, max_w), dtype=np.uint8)
                h, w = arr.shape
                if h != max_h or w != max_w:
                    padded = np.zeros((max_h, max_w), dtype=arr.dtype)
                    padded[:h, :w] = arr
                    return padded
                return arr

            r = ensure(r)
            g = ensure(g)
            b = ensure(b)

            rgb = np.stack([r, g, b], axis=-1)
            pil = Image.fromarray(rgb)
            self.image_viewer.show_image(pil, fit_to_screen=True)
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", str(e))

    def toggle_rgb_mode(self, state):
        self.is_rgb_mode = state == Qt.Checked
        enabled = self.is_rgb_mode
        self.red_band_combo.setEnabled(enabled)
        self.green_band_combo.setEnabled(enabled)
        self.blue_band_combo.setEnabled(enabled)
        self.red_offset_x.setEnabled(enabled)
        self.red_offset_y.setEnabled(enabled)
        self.green_offset_x.setEnabled(enabled)
        self.green_offset_y.setEnabled(enabled)
        self.blue_offset_x.setEnabled(enabled)
        self.blue_offset_y.setEnabled(enabled)
        if self.band_frames:
            self.update_rgb_bands()
        # Invalidate any previously generated video
        self.invalidate_video()

    def invalidate_video(self):
        # If a video was generated, mark it stale and require regeneration
        if getattr(self, 'video_path', None):
            self.video_path = None
            self.play_btn.setEnabled(False)
            self.backward_btn.setEnabled(False)
            self.forward_btn.setEnabled(False)
            self.speed_combo.setEnabled(False)
            QMessageBox.information(self, "Regenerate Required", "Parameters changed — click Generate Video to create an updated video.")

    def validate_height(self):
        try:
            h = int(self.height_entry.text())
            if h <= 0:
                raise ValueError
            self.height = h
            self.invalidate_video()
        except ValueError:
            QMessageBox.warning(self, "Invalid Height", "Height must be a positive integer.")
            self.height_entry.setText(str(self.height))

    def validate_width(self):
        try:
            w = int(self.width_entry.text())
            if w <= 0:
                raise ValueError
            self.width = w
            self.invalidate_video()
        except ValueError:
            QMessageBox.warning(self, "Invalid Width", "Width must be a positive integer.")
            self.width_entry.setText(str(self.width))

    def format_time(self, ms):
        seconds = ms // 1000
        minutes = seconds // 60
        seconds %= 60
        return f"{minutes:02d}:{seconds:02d}"

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Frame Folder")
        if folder:
            return self.open_folder(folder)

    def open_folder(self, folder):
        self.current_folder = folder
        self._update_host_tab_name(folder)
        # Auto-apply folder-level defaults if present
        try:
            fdata = load_folder_params(folder)
            if fdata and fdata.get('default'):
                d = fdata.get('default')
                try:
                    if 'height' in d: self.height_entry.setText(str(d.get('height')))
                    if 'width' in d: self.width_entry.setText(str(d.get('width')))
                    if 'bit_depth' in d: self.bitdepth_var.setCurrentText(str(d.get('bit_depth')))
                except Exception:
                    pass
        except Exception:
            pass

        # Record recent
        try:
            params = {'width': int(self.width_entry.text()), 'height': int(self.height_entry.text()), 'bit_depth': int(self.bitdepth_var.currentText())}
            add_recent(folder, 'video', params)
        except Exception:
            pass

        # Invalidate any previously generated video when switching folder
        self.invalidate_video()
        # quick scan for available band files to populate RGB combos without full load
        self.quick_band_files = {}
        keys = set()
        for f in os.listdir(folder):
            m = re.search(r"\.band(\d+)([012])$", f)
            if not m:
                continue
            idx = m.group(1)
            typ = m.group(2)
            if typ == '2':
                key = f"b{idx}_binned"
            elif typ == '0':
                key = f"b{idx}_left"
            else:
                key = f"b{idx}_right"
            keys.add(key)
            self.quick_band_files[key] = os.path.join(folder, f)

        key_list = sorted(keys)
        if key_list:
            for combo in [self.red_band_combo, self.green_band_combo, self.blue_band_combo]:
                combo.clear()
                combo.addItems(key_list)
            # auto-preview when selection changes
            self.red_band_combo.currentIndexChanged.connect(lambda: self.preview_rgb() if self.rgb_mode_checkbox.isChecked() else None)
            self.green_band_combo.currentIndexChanged.connect(lambda: self.preview_rgb() if self.rgb_mode_checkbox.isChecked() else None)
            self.blue_band_combo.currentIndexChanged.connect(lambda: self.preview_rgb() if self.rgb_mode_checkbox.isChecked() else None)
            # preview when enabling RGB mode
            self.rgb_mode_checkbox.toggled.connect(lambda checked: self.preview_rgb() if checked else None)
            # enable rgb controls so user can select bands before full load
            self.rgb_mode_checkbox.setEnabled(True)
            self.preview_rgb_btn.setEnabled(True)
            self.process_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, "No band files", "No band files detected in selected folder.")

    def _show_recent_menu(self):
        try:
            menu = QMenu(self)
            recs = get_recents_for_mode('video', limit=7)
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
                    act.triggered.connect(lambda checked, p=path: self.open_folder(p))
            all_recs = get_recents_for_mode('video')
            if len(all_recs) > 7:
                menu.addSeparator()
                vm = menu.addAction("View more...")
                vm.triggered.connect(lambda: self._open_full_history('video'))
            pos = self.load_menu_btn.mapToGlobal(self.load_menu_btn.rect().bottomLeft())
            menu.exec_(pos)
        except Exception:
            pass

    def _open_full_history(self, mode):
        try:
            sel = select_from_history(self, mode=mode)
            if sel:
                if mode == 'video':
                    self.open_folder(sel)
        except Exception:
            pass

    def process_data(self):
        if not self.current_folder:
            QMessageBox.critical(self, "Error", "No folder selected. Please select a folder first.")
            return
        try:
            self.height = int(self.height_entry.text())
            self.width = int(self.width_entry.text())
            self.bitdepth = int(self.bitdepth_var.currentText())
        except ValueError:
            QMessageBox.critical(self, "Error", "Width, Height, and Bit Depth must be valid integers.")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.loading_thread = LoadingThread(self.current_folder, self.width, self.height, self.bitdepth)
        self.loading_thread.progress.connect(self.progress_bar.setValue)
        self.loading_thread.finished.connect(self.on_loading_finished)
        self.loading_thread.error.connect(self.on_loading_error)
        self.loading_thread.start()

    def on_loading_finished(self, data):
        self.band_frames = data['band_frames']
        self.bands_info = data['bands_info']
        self.max_frames = data['max_frames']
        self.min_frames = min(len(v) for v in self.band_frames.values()) if self.band_frames else 0
        self.stitch_sequence = self.build_stitch_sequence()
        # Calc max_width and total_h once
        self.max_width = 0
        self.total_h = 0
        for entry in self.stitch_sequence:
            if entry['kind'] in ('full_binned', 'full_unbinned'):
                key = entry.get('key')
                if key in self.band_frames:
                    w = self.band_frames[key][0].shape[1]
                    if entry['kind'] == 'full_binned':
                        factor = max(1, int(entry.get('bin_factor', 1)) or 1)
                        w *= factor
                    self.max_width = max(self.max_width, w)
                self.total_h += entry['per_block_h']
            elif entry['kind'] == 'paired_unbinned':
                left_k = entry.get('left_key')
                right_k = entry.get('right_key')
                w_left = self.band_frames[left_k][0].shape[1] if left_k in self.band_frames else 0
                w_right = self.band_frames[right_k][0].shape[1] if right_k in self.band_frames else 0
                self.max_width = max(self.max_width, w_left + w_right)
                self.total_h += entry['per_block_h']
            self.total_h += GAP
        self.total_h -= GAP  # Remove last gap
        if self.max_width <= 0 or self.total_h <= 0:
            QMessageBox.critical(self, "Error", "Invalid dimensions calculated from frames. Check width/height/bitdepth settings.")
            self.progress_bar.setVisible(False)
            return
        self.update_rgb_bands()
        self.current_frame_index = 0
        self.show_current_frame()
        self.progress_bar.setVisible(False)
        self.generate_video()

    def save_state(self):
        return {
            'current_folder': getattr(self, 'current_folder', None),
            'width': int(self.width_entry.text()) if hasattr(self, 'width_entry') else self.width,
            'height': int(self.height_entry.text()) if hasattr(self, 'height_entry') else self.height,
            'bitdepth': int(self.bitdepth_var.currentText()) if hasattr(self, 'bitdepth_var') else self.bitdepth,
            'fps': int(self.fps_var.value()) if hasattr(self, 'fps_var') else self.fps,
            'start_frame': int(self.start_frame_spin.value()) if hasattr(self, 'start_frame_spin') else 1,
            'end_frame': int(self.end_frame_spin.value()) if hasattr(self, 'end_frame_spin') else 1,
            'current_frame': int(getattr(self, 'current_frame_index', 0)),
            'is_rgb_mode': bool(self.rgb_mode_checkbox.isChecked()) if hasattr(self, 'rgb_mode_checkbox') else False,
            'rgb_bands': {'R': self.red_band_combo.currentText() if hasattr(self, 'red_band_combo') else None,
                          'G': self.green_band_combo.currentText() if hasattr(self, 'green_band_combo') else None,
                          'B': self.blue_band_combo.currentText() if hasattr(self, 'blue_band_combo') else None},
            'rgb_offsets': {
                'R': {'x': self.red_offset_x.value(), 'y': self.red_offset_y.value()} if hasattr(self, 'red_offset_x') else {'x':0,'y':0},
                'G': {'x': self.green_offset_x.value(), 'y': self.green_offset_y.value()} if hasattr(self, 'green_offset_x') else {'x':0,'y':0},
                'B': {'x': self.blue_offset_x.value(), 'y': self.blue_offset_y.value()} if hasattr(self, 'blue_offset_x') else {'x':0,'y':0},
            },
            'loop': bool(self.loop_checkbox.isChecked()) if hasattr(self, 'loop_checkbox') else True,
            'speed': self.speed_combo.currentText() if hasattr(self, 'speed_combo') else '1.0x',
            'video_path': getattr(self, 'video_path', None)
        }

    def load_state(self, data):
        try:
            cf = data.get('current_folder')
            if cf:
                self.current_folder = cf
                self._update_host_tab_name(cf)
                # restore simple UI fields
                try:
                    if 'width' in data and hasattr(self, 'width_entry'):
                        self.width_entry.setText(str(data.get('width')))
                    if 'height' in data and hasattr(self, 'height_entry'):
                        self.height_entry.setText(str(data.get('height')))
                    if 'bitdepth' in data and hasattr(self, 'bitdepth_var'):
                        self.bitdepth_var.setCurrentText(str(data.get('bitdepth')))
                    if 'fps' in data and hasattr(self, 'fps_var'):
                        self.fps_var.setValue(int(data.get('fps', self.fps_var.value())))
                    if 'start_frame' in data and hasattr(self, 'start_frame_spin'):
                        self.start_frame_spin.setValue(int(data.get('start_frame', 1)))
                    if 'end_frame' in data and hasattr(self, 'end_frame_spin'):
                        self.end_frame_spin.setValue(int(data.get('end_frame', 1)))
                except Exception:
                    pass
                # process data to populate band_frames
                try:
                    self.process_data()
                except Exception as e:
                    print(f"PlaybackApp.load_state: process_data failed: {e}")
            # restore playback-specific states (after loading)
            try:
                if 'is_rgb_mode' in data and hasattr(self, 'rgb_mode_checkbox'):
                    self.rgb_mode_checkbox.setChecked(bool(data.get('is_rgb_mode', False)))
                rb = data.get('rgb_bands', {})
                if rb and hasattr(self, 'red_band_combo'):
                    try:
                        if rb.get('R'):
                            self.red_band_combo.setCurrentText(rb.get('R'))
                        if rb.get('G'):
                            self.green_band_combo.setCurrentText(rb.get('G'))
                        if rb.get('B'):
                            self.blue_band_combo.setCurrentText(rb.get('B'))
                    except Exception:
                        pass
                ro = data.get('rgb_offsets', {})
                if ro and hasattr(self, 'red_offset_x'):
                    try:
                        r = ro.get('R', {})
                        self.red_offset_x.setValue(int(r.get('x', 0)))
                        self.red_offset_y.setValue(int(r.get('y', 0)))
                        g = ro.get('G', {})
                        self.green_offset_x.setValue(int(g.get('x', 0)))
                        self.green_offset_y.setValue(int(g.get('y', 0)))
                        b = ro.get('B', {})
                        self.blue_offset_x.setValue(int(b.get('x', 0)))
                        self.blue_offset_y.setValue(int(b.get('y', 0)))
                    except Exception:
                        pass
                if 'loop' in data and hasattr(self, 'loop_checkbox'):
                    try:
                        self.loop_checkbox.setChecked(bool(data.get('loop', True)))
                    except Exception:
                        pass
                if 'speed' in data and hasattr(self, 'speed_combo'):
                    try:
                        self.speed_combo.setCurrentText(data.get('speed', self.speed_combo.currentText()))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            print(f"PlaybackApp.load_state error: {e}")

    def on_loading_error(self, error_msg):
        QMessageBox.critical(self, "Loading Error", error_msg)
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

    def build_stitch_sequence(self):
        stitch_sequence = []
        keys = list(self.band_frames.keys())
        binned_keys = [k for k in keys if 'binned' in k.lower()]
        left_keys = [k for k in keys if k.lower().endswith('_left')]
        right_keys = [k for k in keys if k.lower().endswith('_right')]
        full_unbinned_keys = [k for k in keys if k not in binned_keys and k not in left_keys and k not in right_keys]
        for k in binned_keys:
            base = k.split('_', 1)[0]
            per_h = self.height
            stitch_sequence.append({
                'base': base, 'kind': 'full_binned', 'per_block_h': per_h, 'bin_factor': 2,
                'is_split': False, 'side': None, 'key': k
            })
        unbinned_split_bases = set(k.rsplit('_', 1)[0] for k in left_keys + right_keys)
        for base in sorted(unbinned_split_bases):
            left_k = f"{base}_left"
            right_k = f"{base}_right"
            if left_k in self.band_frames and right_k in self.band_frames:
                per_h = self.height
                stitch_sequence.append({
                    'base': base, 'kind': 'paired_unbinned', 'per_block_h': per_h, 'bin_factor': 1,
                    'is_split': False, 'side': None, 'left_key': left_k, 'right_key': right_k
                })
        for k in full_unbinned_keys:
            base = k.split('_', 1)[0]
            per_h = self.height
            if not any(e['base'] == base for e in stitch_sequence):
                stitch_sequence.append({
                    'base': base, 'kind': 'full_unbinned', 'per_block_h': per_h, 'bin_factor': 1,
                    'is_split': False, 'side': None, 'key': k
                })
        cur_y = 0
        for e in stitch_sequence:
            e['start_y'] = cur_y
            cur_y += int(e['per_block_h']) + GAP
        return stitch_sequence

    def generate_video(self):
        self.fps = self.fps_var.value()
        self.rgb_bands["R"] = self.red_band_combo.currentText()
        self.rgb_bands["G"] = self.green_band_combo.currentText()
        self.rgb_bands["B"] = self.blue_band_combo.currentText()
        self.rgb_offsets["R"] = {"x": self.red_offset_x.value(), "y": self.red_offset_y.value()}
        self.rgb_offsets["G"] = {"x": self.green_offset_x.value(), "y": self.green_offset_y.value()}
        self.rgb_offsets["B"] = {"x": self.blue_offset_x.value(), "y": self.blue_offset_y.value()}
        # If band_frames not populated (we did quick scan only), create lazy FrameSource objects
        if not self.band_frames and hasattr(self, 'quick_band_files') and self.quick_band_files:
            self.band_frames = {}
            for key, path in self.quick_band_files.items():
                # infer size from key name
                if 'binned' in key:
                    w = self.width // 2
                    h = self.height // 2
                else:
                    w = self.width // 2
                    h = self.height
                try:
                    self.band_frames[key] = LazyFrames(path, w, h, self.bitdepth)
                except Exception:
                    continue
        # If RGB mode, validate selected bands exist and compute output size from them
        if self.is_rgb_mode:
            r_key = self.rgb_bands["R"]
            g_key = self.rgb_bands["G"]
            b_key = self.rgb_bands["B"]
            if not (r_key in self.band_frames and g_key in self.band_frames and b_key in self.band_frames):
                QMessageBox.critical(self, "Error", "Selected RGB bands are not all available. Please select valid bands.")
                return
            # derive output dimensions from selected band sources
            try:
                src_r = self.band_frames[r_key]
                src_g = self.band_frames[g_key]
                src_b = self.band_frames[b_key]
                # FrameSource uses attributes w,h; other types may use shape
                def src_shape(src):
                    if hasattr(src, 'h') and hasattr(src, 'w'):
                        return (src.h, src.w)
                    else:
                        # try first frame
                        f = src[0]
                        return f.shape
                r_h, r_w = src_shape(src_r)
                g_h, g_w = src_shape(src_g)
                b_h, b_w = src_shape(src_b)
                max_h = max(r_h, g_h, b_h)
                max_w = max(r_w, g_w, b_w)
            except Exception:
                QMessageBox.critical(self, "Error", "Could not determine RGB band dimensions.")
                return
            channels = 3
            estimated_bytes_per_frame = max_h * max_w * channels
            min_frames = min(len(src_r), len(src_g), len(src_b))
            total_estimated = estimated_bytes_per_frame * min_frames
        else:
            channels = 1
            # determine output dims from available stitch sequence if not already set
            calc_max_w = self.max_width
            calc_total_h = self.total_h
            if (not calc_max_w or not calc_total_h) and self.band_frames:
                # recompute from stitch_sequence or build one
                if not self.stitch_sequence:
                    self.stitch_sequence = self.build_stitch_sequence()
                calc_max_w = 0
                calc_total_h = 0
                for entry in self.stitch_sequence:
                    if entry['kind'] in ('full_binned', 'full_unbinned'):
                        key = entry.get('key')
                        if key in self.band_frames:
                            try:
                                w = self.band_frames[key][0].shape[1]
                                calc_max_w = max(calc_max_w, w)
                            except Exception:
                                pass
                        calc_total_h += entry['per_block_h']
                    elif entry['kind'] == 'split_unbinned':
                        left_k = entry.get('left_key')
                        right_k = entry.get('right_key')
                        if left_k in self.band_frames:
                            try:
                                calc_max_w = max(calc_max_w, self.band_frames[left_k][0].shape[1])
                            except Exception:
                                pass
                        if right_k in self.band_frames:
                            try:
                                calc_max_w = max(calc_max_w, self.band_frames[right_k][0].shape[1])
                            except Exception:
                                pass
                        calc_total_h += entry['per_block_h'] * 2
                    calc_total_h += GAP
                calc_total_h = max(0, calc_total_h - GAP)
            estimated_bytes_per_frame = (calc_total_h or self.total_h) * (calc_max_w or self.max_width) * channels  # uint8
            min_frames = min(len(v) for v in self.band_frames.values()) if self.band_frames else 0
            total_estimated = estimated_bytes_per_frame * min_frames

        if min_frames == 0:
            QMessageBox.critical(self, "Error", "No frames found to generate video.")
            return
        if not check_memory_requirement(total_estimated, self):
            self.process_btn.setEnabled(True)
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        # frame range (convert to 0-based)
        start = max(0, self.start_frame_spin.value() - 1)
        end = max(0, self.end_frame_spin.value() - 1)
        # persist min_frames and output dims so playback uses same range
        self.min_frames = min_frames
        if self.is_rgb_mode:
            # set display dims to selected RGB output dims
            self.max_width = max_w
            self.total_h = max_h
        else:
            # if we computed calc_max_w/calc_total_h above, persist them
            if 'calc_max_w' in locals() and calc_max_w:
                self.max_width = calc_max_w
            if 'calc_total_h' in locals() and calc_total_h:
                self.total_h = calc_total_h
        # pick dimensions for writer
        out_w = max_w if self.is_rgb_mode else self.max_width
        out_h = max_h if self.is_rgb_mode else self.total_h
        self.video_gen_thread = VideoGenerationThread(self.band_frames, self.stitch_sequence, self.max_frames, out_w, self.fps, out_h, self.is_rgb_mode, self.rgb_bands, self.rgb_offsets, start, end, self)
        self.video_gen_thread.progress.connect(self.progress_bar.setValue)
        self.video_gen_thread.finished.connect(self.on_video_generated)
        self.video_gen_thread.error.connect(self.on_video_error)
        self.video_gen_thread.start()

    def on_video_generated(self, video_path):
        self.video_path = video_path
        self.current_frame_index = 0
        self.play_btn.setEnabled(True)
        self.backward_btn.setEnabled(True)
        self.forward_btn.setEnabled(True)
        self.speed_combo.setEnabled(True)
        self.frame_slider.setRange(0, self.min_frames - 1)
        self.frame_slider.setValue(0)
        total_ms = int(self.min_frames / self.fps * 1000)
        self.set_slider_range(total_ms)
        self.show_current_frame()
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "Video generated! Ready to play.")

    def on_video_error(self, error_msg):
        QMessageBox.critical(self, "Video Error", error_msg)
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

    def toggle_play(self):
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            interval = int(1000 / self.fps / float(self.speed_combo.currentText()[:-1]))
            self.playback_timer.setInterval(interval)
            self.playback_timer.start()
            self.play_btn.setText("❚❚ Pause")

    def skip_backward(self):
        self.current_frame_index = max(0, self.current_frame_index - int(self.fps))
        self.show_current_frame()
        self.update_slider_from_frame()

    def skip_forward(self):
        self.current_frame_index = min(self.min_frames - 1, self.current_frame_index + int(self.fps))
        self.show_current_frame()
        self.update_slider_from_frame()

    def update_slider(self, position):
        self.frame_slider.setValue(self.current_frame_index)
        current_seconds = self.current_frame_index / self.fps
        self.current_time_label.setText(self.format_time(int(current_seconds * 1000)))

    def set_slider_range(self, duration):
        self.frame_slider.setRange(0, self.min_frames - 1)
        total_seconds = self.min_frames / self.fps
        self.total_time_label.setText("/ " + self.format_time(int(total_seconds * 1000)))

    def set_position(self, position):
        # position is frame index
        self.current_frame_index = position
        self.show_current_frame()
        self.update_slider(0)  # Dummy, will update correctly

    def update_slider_from_frame(self):
        position = int(self.current_frame_index * 1000 / self.fps)
        self.update_slider(position)

    def next_frame(self):
        self.current_frame_index += 1
        if self.current_frame_index >= self.min_frames:
            if self.loop_checkbox.isChecked():
                self.current_frame_index = 0
            else:
                self.playback_timer.stop()
                self.play_btn.setText("▶ Play")
                return
        self.show_current_frame()
        self.update_slider_from_frame()

    def change_speed(self, text):
        rate = float(text[:-1])
        if self.playback_timer.isActive():
            interval = int(1000 / self.fps / rate)
            self.playback_timer.setInterval(interval)

    def toggle_loop(self, state):
        # Loop is handled in next_frame
        pass

    def handle_state_changed(self, state):
        # Not used since we're using timer for playback
        pass

    def show_current_frame(self):
        frame_idx = self.current_frame_index

        if self.is_rgb_mode:
            # RGB fusion mode
            r_key = self.rgb_bands["R"]
            g_key = self.rgb_bands["G"]
            b_key = self.rgb_bands["B"]
            r_offset = self.rgb_offsets["R"]
            g_offset = self.rgb_offsets["G"]
            b_offset = self.rgb_offsets["B"]

            r_frame = self.band_frames[r_key][frame_idx] if r_key in self.band_frames else np.zeros((self.total_h, self.max_width), dtype=np.uint8)
            g_frame = self.band_frames[g_key][frame_idx] if g_key in self.band_frames else np.zeros((self.total_h, self.max_width), dtype=np.uint8)
            b_frame = self.band_frames[b_key][frame_idx] if b_key in self.band_frames else np.zeros((self.total_h, self.max_width), dtype=np.uint8)

            # Apply offsets
            r_frame = self.apply_offset_simple(r_frame, r_offset['x'], r_offset['y'])
            g_frame = self.apply_offset_simple(g_frame, g_offset['x'], g_offset['y'])
            b_frame = self.apply_offset_simple(b_frame, b_offset['x'], b_offset['y'])

            # Ensure same size
            max_h = max(r_frame.shape[0], g_frame.shape[0], b_frame.shape[0])
            max_w = max(r_frame.shape[1], g_frame.shape[1], b_frame.shape[1])
            r_frame = self.pad_frame(r_frame, max_h, max_w)
            g_frame = self.pad_frame(g_frame, max_h, max_w)
            b_frame = self.pad_frame(b_frame, max_h, max_w)

            rgb_array = np.stack([r_frame, g_frame, b_frame], axis=-1)
            pil_image = Image.fromarray(rgb_array)
        else:
            # Original grayscale stitching
            parts = []
            processed_bases = set()
            for entry in self.stitch_sequence:
                base = entry.get('base')
                kind = entry.get('kind')
                if base in processed_bases:
                    continue
                if kind in ('full_binned', 'full_unbinned'):
                    key = entry.get('key')
                    if key not in self.band_frames:
                        continue
                    frame = self.band_frames[key][frame_idx]
                    if frame.shape[1] < self.max_width:
                        pad_w = self.max_width - frame.shape[1]
                        padding = np.zeros((frame.shape[0], pad_w), dtype=frame.dtype)
                        frame = np.hstack([frame, padding])
                    parts.append(frame)
                    processed_bases.add(base)
                elif kind == 'split_unbinned':
                    left_k = entry.get('left_key')
                    right_k = entry.get('right_key')
                    if left_k not in self.band_frames or right_k not in self.band_frames:
                        continue
                    left_frame = self.band_frames[left_k][frame_idx]
                    right_frame = self.band_frames[right_k][frame_idx]
                    if left_frame.shape[1] < self.max_width:
                        pad_w = self.max_width - left_frame.shape[1]
                        padding = np.zeros((left_frame.shape[0], pad_w), dtype=left_frame.dtype)
                        left_frame = np.hstack([left_frame, padding])
                    if right_frame.shape[1] < self.max_width:
                        pad_w = self.max_width - right_frame.shape[1]
                        padding = np.zeros((right_frame.shape[0], pad_w), dtype=right_frame.dtype)
                        right_frame = np.hstack([right_frame, padding])
                    parts.append(left_frame)
                    parts.append(right_frame)
                    processed_bases.add(base)

            full_display = np.zeros((self.total_h, self.max_width), dtype=np.uint8)
            cur_y = 0
            for part in parts:
                h_part = part.shape[0]
                full_display[cur_y:cur_y + h_part, :part.shape[1]] = part
                cur_y += h_part + GAP

            pil_image = Image.fromarray(full_display)

        self.image_viewer.show_image(pil_image, fit_to_screen=True)

    def apply_offset_simple(self, frame, offset_x, offset_y):
        # Simple offset by rolling
        frame = np.roll(frame, offset_x, axis=1)
        frame = np.roll(frame, offset_y, axis=0)
        return frame

    def pad_frame(self, frame, target_h, target_w):
        h, w = frame.shape
        if h < target_h or w < target_w:
            padded = np.zeros((target_h, target_w), dtype=frame.dtype)
            padded[:h, :w] = frame
            return padded
        return frame

    def on_pixel_info(self, x, y, values, is_rgb=False):
        # Placeholder for pixel info callback
        pass

    def closeEvent(self, event):
        if self.video_path and os.path.exists(self.video_path):
            os.remove(self.video_path)  # Clean up temp file
        super().closeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal Frame Playback")
        self.resize(1280, 720)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.app = PlaybackApp()
        layout.addWidget(self.app)

if __name__ == "__main__":
    print("This module is not meant to be run directly. Please run main.py to start the application and access video mode.")
    sys.exit(0)
