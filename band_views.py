# band_views.py (modified)
import numpy as np
from PIL import Image
import gc
import re
import psutil
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QCheckBox, QLabel, QProgressBar, QApplication
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from image_viewer import GraphicsImageViewer
from utils import check_memory_requirement, image_coords_to_latlon
from PyQt5.QtGui import QTransform

def _safe_show_image(viewer, pil_image, **kwargs):
    """Best-effort show_image that ignores stale/deleted Qt wrappers."""
    if viewer is None:
        return False
    try:
        viewer.show_image(pil_image, **kwargs)
        return True
    except RuntimeError as e:
        if "has been deleted" in str(e):
            return False
        raise
    except Exception:
        return False

class IndividualBandWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, frames, enhance, offset_x, offset_y, start_frame, end_frame, gap, parent=None):
        super().__init__(parent)
        self.frames = frames
        self.enhance = enhance
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.gap = gap

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            self.progress.emit(0)
            parent = self.parent()
            if self.start_frame == self.end_frame:
                self.progress.emit(50)
                frame = self.frames[self.start_frame]
                frame = parent.apply_offset(frame, self.offset_x, self.offset_y, crop_y=True)
                display_frame = parent.apply_contrast_enhancement(frame) if self.enhance else frame.copy()
                if hasattr(self.frames, 'get_raw'):
                    raw_frame = self.frames.get_raw(self.start_frame)
                    raw_frame = parent.apply_offset(raw_frame, self.offset_x, self.offset_y, crop_y=True)
                    raw_frame_unscaled = raw_frame.copy()
                    # Scale raw frame to full 16-bit range for proper display brightness
                    if hasattr(self.frames, 'bitdepth'):
                        bd = self.frames.bitdepth
                        if bd == 8:
                            scale = 65535.0 / 255.0
                        elif bd == 10:
                            scale = 65535.0 / 1023.0
                        elif bd == 12:
                            scale = 65535.0 / 4095.0
                        else:
                            scale = 1.0
                        if scale != 1.0:
                            raw_frame = (raw_frame.astype(np.float32) * scale).astype(np.uint16)
                    pil_raw = Image.fromarray(raw_frame)
                    raw_array = raw_frame_unscaled
                else:
                    raw_frame = frame.copy()
                    pil_raw = Image.fromarray(raw_frame)
                    raw_array = raw_frame
                pil_display = Image.fromarray(display_frame)
                self.progress.emit(100)
            else:
                num_frames = self.end_frame - self.start_frame + 1
                parts_display = []
                parts_raw = []
                parts_raw_unscaled = []
                for i_idx, i in enumerate(range(self.start_frame, self.end_frame + 1)):
                    if self.isInterruptionRequested():
                        return
                    self.progress.emit(int((i_idx + 1) / num_frames * 100))
                    frame = self.frames[i]
                    frame = parent.apply_offset(frame, self.offset_x, self.offset_y, crop_y=True)
                    display_frame = parent.apply_contrast_enhancement(frame) if self.enhance else frame.copy()
                    if hasattr(self.frames, 'get_raw'):
                        raw_frame = self.frames.get_raw(i)
                        raw_frame = parent.apply_offset(raw_frame, self.offset_x, self.offset_y, crop_y=True)
                        raw_frame_unscaled = raw_frame.copy()
                        # Scale raw frame to full 16-bit range for proper display brightness
                        if hasattr(self.frames, 'bitdepth'):
                            bd = self.frames.bitdepth
                            if bd == 8:
                                scale = 65535.0 / 255.0
                            elif bd == 10:
                                scale = 65535.0 / 1023.0
                            elif bd == 12:
                                scale = 65535.0 / 4095.0
                            else:
                                scale = 1.0
                            if scale != 1.0:
                                raw_frame = (raw_frame.astype(np.float32) * scale).astype(np.uint16)
                        parts_raw.append(raw_frame)
                        parts_raw_unscaled.append(raw_frame_unscaled)
                    else:
                        raw_frame = frame.copy()
                        parts_raw.append(raw_frame)
                        parts_raw_unscaled.append(raw_frame)
                    if i < self.end_frame:
                        gap_arr = np.zeros((self.gap, display_frame.shape[1]), dtype=np.uint8)
                        display_frame = np.vstack([display_frame, gap_arr])
                        raw_frame = np.vstack([raw_frame, gap_arr.copy()])
                    parts_display.append(display_frame)
                full_display = np.vstack(parts_display)
                full_raw = np.vstack(parts_raw)
                full_raw_unscaled = np.vstack(parts_raw_unscaled)
                pil_display = Image.fromarray(full_display)
                # PIL cannot represent multi-channel uint16 as standard RGB.
                # Keep high-bit raw data in ndarray (`original_raw_data`) and pass
                # an 8-bit preview PIL for viewer rendering.
                pil_raw = pil_display
            self.finished.emit({'display': pil_display, 'raw': pil_raw, 'raw_array': full_raw_unscaled if 'full_raw_unscaled' in locals() else raw_array})
        except Exception as e:
            self.error.emit(str(e))

class MergedBandWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, band_frames, left_key, right_key, enhance, offset_x, offset_y, start_frame, end_frame, gap, parent=None):
        super().__init__(parent)
        self.band_frames = band_frames
        self.left_key = left_key
        self.right_key = right_key
        self.enhance = enhance
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.gap = gap

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            self.progress.emit(0)
            parent = self.parent()
            if self.start_frame == self.end_frame:
                self.progress.emit(50)
                left_frame = self.band_frames[self.left_key][self.start_frame]
                right_frame = self.band_frames[self.right_key][self.start_frame]
                left_frame = parent.apply_offset(left_frame, self.offset_x, self.offset_y, crop_y=True)
                right_frame = parent.apply_offset(right_frame, self.offset_x, self.offset_y, crop_y=True)
                display_left = parent.apply_contrast_enhancement(left_frame) if self.enhance else left_frame.copy()
                display_right = parent.apply_contrast_enhancement(right_frame) if self.enhance else right_frame.copy()
                if hasattr(self.band_frames[self.left_key], 'get_raw'):
                    raw_left = self.band_frames[self.left_key].get_raw(self.start_frame)
                    raw_left = parent.apply_offset(raw_left, self.offset_x, self.offset_y, crop_y=True)
                    raw_left_unscaled = raw_left.copy()
                    # Scale raw frame to full 16-bit range for proper display brightness
                    if hasattr(self.band_frames[self.left_key], 'bitdepth'):
                        bd = self.band_frames[self.left_key].bitdepth
                        if bd == 8:
                            scale = 65535.0 / 255.0
                        elif bd == 10:
                            scale = 65535.0 / 1023.0
                        elif bd == 12:
                            scale = 65535.0 / 4095.0
                        else:
                            scale = 1.0
                        if scale != 1.0:
                            raw_left = (raw_left.astype(np.float32) * scale).astype(np.uint16)
                    raw_left_array = raw_left_unscaled
                else:
                    raw_left = left_frame.copy()
                    raw_left_array = raw_left
                if hasattr(self.band_frames[self.right_key], 'get_raw'):
                    raw_right = self.band_frames[self.right_key].get_raw(self.start_frame)
                    raw_right = parent.apply_offset(raw_right, self.offset_x, self.offset_y, crop_y=True)
                    raw_right_unscaled = raw_right.copy()
                    # Scale raw frame to full 16-bit range for proper display brightness
                    if hasattr(self.band_frames[self.right_key], 'bitdepth'):
                        bd = self.band_frames[self.right_key].bitdepth
                        if bd == 8:
                            scale = 65535.0 / 255.0
                        elif bd == 10:
                            scale = 65535.0 / 1023.0
                        elif bd == 12:
                            scale = 65535.0 / 4095.0
                        else:
                            scale = 1.0
                        if scale != 1.0:
                            raw_right = (raw_right.astype(np.float32) * scale).astype(np.uint16)
                    raw_right_array = raw_right_unscaled
                else:
                    raw_right = right_frame.copy()
                    raw_right_array = raw_right
                display_full = np.hstack([display_left, display_right])
                raw_full = np.hstack([raw_left, raw_right])
                raw_array = np.hstack([raw_left_array, raw_right_array])
                pil_display = Image.fromarray(display_full)
                pil_raw = Image.fromarray(raw_full)
                self.progress.emit(100)
                self.finished.emit({'display': pil_display, 'raw': pil_raw, 'raw_array': raw_array})
            else:
                num_frames = self.end_frame - self.start_frame + 1
                parts_display = []
                parts_raw = []
                parts_raw_unscaled = []
                parts_raw_unscaled_left = []
                parts_raw_unscaled_right = []
                for i_idx, i in enumerate(range(self.start_frame, self.end_frame + 1)):
                    if self.isInterruptionRequested():
                        return
                    self.progress.emit(int((i_idx + 1) / num_frames * 100))
                    left_frame = self.band_frames[self.left_key][i]
                    right_frame = self.band_frames[self.right_key][i]
                    left_frame = parent.apply_offset(left_frame, self.offset_x, self.offset_y, crop_y=True)
                    right_frame = parent.apply_offset(right_frame, self.offset_x, self.offset_y, crop_y=True)
                    display_left = parent.apply_contrast_enhancement(left_frame) if self.enhance else left_frame.copy()
                    display_right = parent.apply_contrast_enhancement(right_frame) if self.enhance else right_frame.copy()
                    if hasattr(self.band_frames[self.left_key], 'get_raw'):
                        raw_left = self.band_frames[self.left_key].get_raw(i)
                        raw_left = parent.apply_offset(raw_left, self.offset_x, self.offset_y, crop_y=True)
                        raw_left_unscaled = raw_left.copy()
                        # Scale raw frame to full 16-bit range for proper display brightness
                        if hasattr(self.band_frames[self.left_key], 'bitdepth'):
                            bd = self.band_frames[self.left_key].bitdepth
                            if bd == 8:
                                scale = 65535.0 / 255.0
                            elif bd == 10:
                                scale = 65535.0 / 1023.0
                            elif bd == 12:
                                scale = 65535.0 / 4095.0
                            else:
                                scale = 1.0
                            if scale != 1.0:
                                raw_left = (raw_left.astype(np.float32) * scale).astype(np.uint16)
                        parts_raw_unscaled_left.append(raw_left_unscaled)
                    else:
                        raw_left = left_frame.copy()
                        parts_raw_unscaled_left.append(raw_left)
                    if hasattr(self.band_frames[self.right_key], 'get_raw'):
                        raw_right = self.band_frames[self.right_key].get_raw(i)
                        raw_right = parent.apply_offset(raw_right, self.offset_x, self.offset_y, crop_y=True)
                        raw_right_unscaled = raw_right.copy()
                        # Scale raw frame to full 16-bit range for proper display brightness
                        if hasattr(self.band_frames[self.right_key], 'bitdepth'):
                            bd = self.band_frames[self.right_key].bitdepth
                            if bd == 8:
                                scale = 65535.0 / 255.0
                            elif bd == 10:
                                scale = 65535.0 / 1023.0
                            elif bd == 12:
                                scale = 65535.0 / 4095.0
                            else:
                                scale = 1.0
                            if scale != 1.0:
                                raw_right = (raw_right.astype(np.float32) * scale).astype(np.uint16)
                        parts_raw_unscaled_right.append(raw_right_unscaled)
                    else:
                        raw_right = right_frame.copy()
                        parts_raw_unscaled_right.append(raw_right)
                    display_full = np.hstack([display_left, display_right])
                    raw_full = np.hstack([raw_left, raw_right])
                    raw_full_unscaled = np.hstack([raw_left_unscaled, raw_right_unscaled])
                    if i < self.end_frame:
                        gap_arr = np.zeros((self.gap, display_full.shape[1]), dtype=np.uint8)
                        display_full = np.vstack([display_full, gap_arr])
                        raw_full = np.vstack([raw_full, gap_arr.copy()])
                        gap_arr_unscaled = np.zeros((self.gap, raw_full_unscaled.shape[1]), dtype=np.uint16)
                        raw_full_unscaled = np.vstack([raw_full_unscaled, gap_arr_unscaled])
                    parts_display.append(display_full)
                    parts_raw.append(raw_full)
                    parts_raw_unscaled.append(raw_full_unscaled)
                full_display = np.vstack(parts_display)
                full_raw = np.vstack(parts_raw)
                full_raw_unscaled = np.vstack(parts_raw_unscaled)
                pil_display = Image.fromarray(full_display)
                # PIL does not support 3-channel uint16 RGB consistently.
                # Keep raw high-bit data in `original_raw_data` instead.
                pil_raw = pil_display
            self.finished.emit({'display': pil_display, 'raw': pil_raw, 'raw_array': full_raw_unscaled})
        except Exception as e:
            self.error.emit(str(e))

class PanBandWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, band_frames, unbinned_keys, enhance, offsets, start_frame, end_frame, gap, parent=None):
        super().__init__(parent)
        self.band_frames = band_frames
        self.unbinned_keys = unbinned_keys
        self.enhance = enhance
        self.offsets = offsets
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.gap = gap

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            self.progress.emit(0)
            parent = self.parent()
            base_keys = sorted(set(k.rsplit('_', 1)[0] for k in self.unbinned_keys))
            if self.start_frame == self.end_frame:
                num_bases = len(base_keys)
                parts_display = []
                parts_raw = []
                for j, base_key in enumerate(base_keys):
                    if self.isInterruptionRequested():
                        return
                    self.progress.emit(int((j + 1) / num_bases * 100))
                    left_key = f"{base_key}_left"
                    right_key = f"{base_key}_right"
                    offset_x = self.offsets.get(base_key, {'x':0})['x']
                    offset_y = self.offsets.get(base_key, {'y':0})['y']
                    left_frame = self.band_frames[left_key][self.start_frame]
                    right_frame = self.band_frames[right_key][self.start_frame]
                    left_frame = parent.apply_offset(left_frame, offset_x, offset_y, crop_y=True)
                    right_frame = parent.apply_offset(right_frame, offset_x, offset_y, crop_y=True)
                    display_left = parent.apply_contrast_enhancement(left_frame) if self.enhance else left_frame.copy()
                    display_right = parent.apply_contrast_enhancement(right_frame) if self.enhance else right_frame.copy()
                    if hasattr(self.band_frames[left_key], 'get_raw'):
                        raw_left = self.band_frames[left_key].get_raw(self.start_frame)
                        raw_left = parent.apply_offset(raw_left, offset_x, offset_y, crop_y=True)
                    else:
                        raw_left = left_frame.copy()
                    if hasattr(self.band_frames[right_key], 'get_raw'):
                        raw_right = self.band_frames[right_key].get_raw(self.start_frame)
                        raw_right = parent.apply_offset(raw_right, offset_x, offset_y, crop_y=True)
                    else:
                        raw_right = right_frame.copy()
                    display_full = np.hstack([display_left, display_right])
                    raw_full = np.hstack([raw_left, raw_right])
                    parts_display.append(display_full)
                    parts_raw.append(raw_full)
                full_display = np.vstack(parts_display)
                full_raw = np.vstack(parts_raw)
                pil_display = Image.fromarray(full_display)
                # For RGB fusion, `full_raw` can be uint16 RGB, which PIL can't build as RGB.
                # Keep high-bit data in `original_raw_data`; pass 8-bit preview PIL here.
                pil_raw = pil_display
            else:
                num_frames = self.end_frame - self.start_frame + 1
                parts_display = []
                parts_raw = []
                for i_idx, i in enumerate(range(self.start_frame, self.end_frame + 1)):
                    if self.isInterruptionRequested():
                        return
                    frame_progress = int((i_idx + 1) / num_frames * 100)
                    self.progress.emit(frame_progress)
                    frame_parts_display = []
                    frame_parts_raw = []
                    num_bases = len(base_keys)
                    for j, base_key in enumerate(base_keys):
                        sub_progress = int(frame_progress + (j / num_bases) * (100 / num_frames))
                        self.progress.emit(sub_progress)
                        left_key = f"{base_key}_left"
                        right_key = f"{base_key}_right"
                        offset_x = self.offsets.get(base_key, {'x':0})['x']
                        offset_y = self.offsets.get(base_key, {'y':0})['y']
                        left_frame = self.band_frames[left_key][i]
                        right_frame = self.band_frames[right_key][i]
                        left_frame = parent.apply_offset(left_frame, offset_x, offset_y, crop_y=True)
                        right_frame = parent.apply_offset(right_frame, offset_x, offset_y, crop_y=True)
                        display_left = parent.apply_contrast_enhancement(left_frame) if self.enhance else left_frame.copy()
                        display_right = parent.apply_contrast_enhancement(right_frame) if self.enhance else right_frame.copy()
                        if hasattr(self.band_frames[left_key], 'get_raw'):
                            raw_left = self.band_frames[left_key].get_raw(i)
                            raw_left = parent.apply_offset(raw_left, offset_x, offset_y, crop_y=True)
                        else:
                            raw_left = left_frame.copy()
                        if hasattr(self.band_frames[right_key], 'get_raw'):
                            raw_right = self.band_frames[right_key].get_raw(i)
                            raw_right = parent.apply_offset(raw_right, offset_x, offset_y, crop_y=True)
                        else:
                            raw_right = right_frame.copy()
                        display_full = np.hstack([display_left, display_right])
                        raw_full = np.hstack([raw_left, raw_right])
                        if i < self.end_frame:
                            gap_arr = np.zeros((self.gap, display_full.shape[1]), dtype=np.uint8)
                            display_full = np.vstack([display_full, gap_arr])
                            raw_full = np.vstack([raw_full, gap_arr.copy()])
                        frame_parts_display.append(display_full)
                        frame_parts_raw.append(raw_full)
                    frame_full_display = np.vstack(frame_parts_display)
                    frame_full_raw = np.vstack(frame_parts_raw)
                    parts_display.append(frame_full_display)
                    parts_raw.append(frame_full_raw)
                full_display = np.vstack(parts_display)
                full_raw = np.vstack(parts_raw)
                pil_display = Image.fromarray(full_display)
                pil_raw = Image.fromarray(full_raw)
            self.progress.emit(100)
            self.finished.emit({'display': pil_display, 'raw': pil_raw, 'raw_array': full_raw if 'full_raw' in locals() else frame_full_raw})
        except Exception as e:
            self.error.emit(str(e))

class BandViewsMixin:     
    def _safe_get_band_frame(self, key, idx):
        """Safely return a specific frame for a band key, None if not available."""
        if key not in self.band_frames:
            return None
        src = self.band_frames.get(key)
        if src is None:
            return None
        try:
            # Use len() where possible to avoid out-of-range attempt.
            if hasattr(src, '__len__') and len(src) <= idx:
                return None
            if idx < 0:
                return None
            frame = src[idx]
            return frame.copy() if hasattr(frame, 'copy') else frame
        except Exception:
            return None

    def _get_raw_frame(self, key, idx):
        """Return raw (full bit-depth) frame if available, else fallback to display frame."""
        try:
            src = self.band_frames.get(key)
        except Exception:
            return None
        if src is None:
            return None
        try:
            if hasattr(src, 'get_raw'):
                if hasattr(src, '__len__') and len(src) <= idx:
                    return None
                if idx < 0:
                    return None
                return src.get_raw(idx)
        except Exception:
            pass
        try:
            frame = src[idx]
            return frame.copy() if hasattr(frame, 'copy') else frame
        except Exception:
            return None

    def _is_1_to_4_enabled(self):
        return bool(getattr(self, "ENABLE_1_TO_4_LAYOUT", False))

    def _binned_upsample_factor(self, bin_factor):
        # Use metadata-driven factor when 1:4 is enabled; default 2 for typical binning.
        if not self._is_1_to_4_enabled():
            return 1
        try:
            factor = int(bin_factor) if int(bin_factor) > 1 else 2
        except Exception:
            factor = 2
        return max(1, factor)

    def _get_unbinned_pair_keys(self, base):
        """Return (left_key, right_key) for an unbinned split base if present, else (None, None)."""
        info = getattr(self, 'bands_info', {}).get(base, {}) if hasattr(self, 'bands_info') else {}
        if info.get('binned', False):
            return None, None
        if 'split' in info and not info.get('split', False):
            return None, None
        variants = info.get('variants', []) if isinstance(info, dict) else []
        # Prefer explicit variant keys if provided.
        for lk, rk in [
            (f"{base}_left", f"{base}_right"),
            (f"{base}0", f"{base}1"),
        ]:
            if lk in self.band_frames and rk in self.band_frames:
                return lk, rk
        for lk in variants:
            for rk in variants:
                if lk == rk:
                    continue
                if (lk.endswith("_left") and rk.endswith("_right")) or (lk.endswith("0") and rk.endswith("1")):
                    if lk in self.band_frames and rk in self.band_frames:
                        return lk, rk
        return None, None

    def _unbinned_base_for_key(self, key):
        """Return base if key is part of an unbinned left/right pair, else None."""
        if key.endswith(("_left", "_right")):
            base = key.rsplit("_", 1)[0]
            lk, rk = self._get_unbinned_pair_keys(base)
            return base if lk and rk else None
        if key.endswith(("0", "1")):
            base = key[:-1]
            # Only treat 0/1 suffixes as a split pair when the base itself is a recognized band entry.
            if base not in getattr(self, 'bands_info', {}):
                return None
            lk, rk = self._get_unbinned_pair_keys(base)
            return base if lk and rk else None
        return None

    def build_stitch_sequence(self, for_display=True):
        bands_info_local = getattr(self, 'bands_info', {}) or {}
        band_frames = getattr(self, 'band_frames', {}) or {}
        try:
            orig_band_h = int(self.height_entry.text())
        except Exception:
            orig_band_h = 384
        gap = int(self.gap_var.value()) if hasattr(self, 'gap_var') else 0
        stitch_sequence = []
        merge_lr = self._is_1_to_4_enabled()
        keys = list(band_frames.keys())
        if keys:
            if merge_lr:
                # In 1:4 layout keep true band order (by index), do not force unbinned to the bottom.
                binned_by_base = {}
                full_unbinned_by_base = {}
                split_bases = set()

                for k in keys:
                    kl = str(k).lower()
                    base = k.split('_', 1)[0]
                    if 'binned' in kl:
                        binned_by_base.setdefault(base, k)
                    elif kl.endswith('_left') or kl.endswith('_right'):
                        split_bases.add(k.rsplit('_', 1)[0])
                    else:
                        full_unbinned_by_base.setdefault(base, k)

                bases_all = set(binned_by_base.keys()) | set(full_unbinned_by_base.keys()) | set(split_bases)
                ordered_info_bases = [b for b in sorted(bands_info_local.keys(), key=lambda k: bands_info_local[k]['index']) if b in bases_all]
                remaining_bases = sorted([b for b in bases_all if b not in ordered_info_bases])
                ordered_bases = ordered_info_bases + remaining_bases

                for base in ordered_bases:
                    info = bands_info_local.get(base, {})
                    bin_factor = int(info.get('bin_factor', 1)) or 1
                    offset_y = int(self.band_offsets.get(base, {'y': 0})['y'])

                    if base in binned_by_base:
                        k = binned_by_base[base]
                        orig_per_h = self._height_from_key(k, default=orig_band_h, bin_factor=bin_factor)
                        per_h = max(1, orig_per_h - abs(offset_y))
                        if for_display and self._is_1_to_4_enabled():
                            per_h = int(per_h * self._binned_upsample_factor(bin_factor))
                        stitch_sequence.append({
                            'base': base, 'kind': 'full_binned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None, 'key': k
                        })
                        continue

                    if base in split_bases:
                        left_k, right_k = self._get_unbinned_pair_keys(base)
                        if left_k in band_frames and right_k in band_frames:
                            orig_per_h = self._height_from_key(left_k, default=orig_band_h, bin_factor=bin_factor)
                            per_h = max(1, orig_per_h - abs(offset_y))
                            stitch_sequence.append({
                                'base': base, 'kind': 'paired_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                                'is_split': False, 'side': None, 'left_key': left_k, 'right_key': right_k
                            })
                        continue

                    k = full_unbinned_by_base.get(base)
                    if k is not None:
                        orig_per_h = self._height_from_key(k, default=orig_band_h, bin_factor=bin_factor)
                        per_h = max(1, orig_per_h - abs(offset_y))
                        stitch_sequence.append({
                            'base': base, 'kind': 'full_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None, 'key': k
                        })
            else:
                binned_keys = [k for k in keys if 'binned' in k.lower()]
                left_keys = [k for k in keys if k.lower().endswith('_left')]
                right_keys = [k for k in keys if k.lower().endswith('_right')]
                full_unbinned_keys = [k for k in keys if k not in binned_keys and k not in left_keys and k not in right_keys]
                for k in binned_keys:
                    base = k.split('_', 1)[0]
                    info = bands_info_local.get(base, {})
                    bin_factor = int(info.get('bin_factor', 1)) or 1
                    orig_per_h = self._height_from_key(k, default=orig_band_h, bin_factor=bin_factor)
                    offset_y = int(self.band_offsets.get(base, {'y': 0})['y'])
                    per_h = max(1, orig_per_h - abs(offset_y))
                    if for_display and self._is_1_to_4_enabled():
                        per_h = int(per_h * self._binned_upsample_factor(bin_factor))
                    stitch_sequence.append({
                        'base': base, 'kind': 'full_binned', 'per_block_h': per_h, 'bin_factor':bin_factor,
                        'is_split': False, 'side': None, 'key': k
                    })
                unbinned_split_bases = set(k.rsplit('_', 1)[0] for k in left_keys + right_keys)
                for base in sorted(unbinned_split_bases):
                    left_k = f"{base}_left"
                    right_k = f"{base}_right"
                    if left_k in band_frames and right_k in band_frames:
                        info = bands_info_local.get(base, {})
                        bin_factor = int(info.get('bin_factor', 1)) or 1
                        offset_y = int(self.band_offsets.get(base, {'y': 0})['y'])
                        if merge_lr:
                            orig_per_h = self._height_from_key(left_k, default=orig_band_h, bin_factor=bin_factor)
                            per_h = max(1, orig_per_h - abs(offset_y))
                            stitch_sequence.append({
                                'base': base, 'kind': 'paired_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                                'is_split': False, 'side': None, 'left_key': left_k, 'right_key': right_k
                            })
                        else:
                            orig_left_h = self._height_from_key(left_k, default=orig_band_h, bin_factor=bin_factor)
                            orig_right_h = self._height_from_key(right_k, default=orig_band_h, bin_factor=bin_factor)
                            per_h_left = max(1, orig_left_h - abs(offset_y))
                            per_h_right = max(1, orig_right_h - abs(offset_y))
                            stitch_sequence.append({
                                'base': base, 'kind': 'half_left', 'per_block_h': per_h_left, 'bin_factor': bin_factor,
                                'is_split': True, 'side': 'left', 'key': left_k
                            })
                            stitch_sequence.append({
                                'base': base, 'kind': 'half_right', 'per_block_h': per_h_right, 'bin_factor': bin_factor,
                                'is_split': True, 'side': 'right', 'key': right_k
                            })
                for k in full_unbinned_keys:
                    base = k.split('_', 1)[0]
                    info = bands_info_local.get(base, {})
                    bin_factor = int(info.get('bin_factor', 1)) or 1
                    orig_per_h = self._height_from_key(k, default=orig_band_h, bin_factor=bin_factor)
                    offset_y = int(self.band_offsets.get(base, {'y': 0})['y'])
                    per_h = max(1, orig_per_h - abs(offset_y))
                    already = any(e['base'] == base for e in stitch_sequence)
                    if not already:
                        stitch_sequence.append({
                            'base': base, 'kind': 'full_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None, 'key': k
                        })
        else:
            ordered_bases = sorted(bands_info_local.keys(), key=lambda k: bands_info_local[k]['index'])
            if merge_lr:
                for b in ordered_bases:
                    info = bands_info_local[b]
                    bin_factor = int(info.get('bin_factor', 1)) or 1
                    orig_per_h = max(1, int(round(float(orig_band_h) / float(bin_factor))))
                    offset_y = int(self.band_offsets.get(b, {'y': 0})['y'])
                    per_h = max(1, orig_per_h - abs(offset_y))
                    if info.get('binned', False):
                        if for_display and self._is_1_to_4_enabled():
                            per_h = int(per_h * self._binned_upsample_factor(bin_factor))
                        stitch_sequence.append({
                            'base': b, 'kind': 'full_binned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None
                        })
                    elif info.get('split', False):
                        stitch_sequence.append({
                            'base': b, 'kind': 'paired_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None
                        })
                    else:
                        stitch_sequence.append({
                            'base': b, 'kind': 'full_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None
                        })
            else:
                binned_bases = [b for b in ordered_bases if bands_info_local[b].get('binned', False)]
                unbinned_bases = [b for b in ordered_bases if not bands_info_local[b].get('binned', False)]
                for b in binned_bases:
                    info = bands_info_local[b]
                    bin_factor = int(info.get('bin_factor', 1)) or 1
                    orig_per_h = max(1, int(round(float(orig_band_h) / float(bin_factor))))
                    offset_y = int(self.band_offsets.get(b, {'y': 0})['y'])
                    per_h = max(1, orig_per_h - abs(offset_y))
                    if for_display and self._is_1_to_4_enabled():
                        per_h = int(per_h * self._binned_upsample_factor(bin_factor))
                    stitch_sequence.append({
                        'base': b, 'kind': 'full_binned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                        'is_split': False, 'side': None
                    })
                for b in unbinned_bases:
                    info = bands_info_local[b]
                    bin_factor = int(info.get('bin_factor', 1)) or 1
                    orig_per_h = max(1, int(round(float(orig_band_h) / float(bin_factor))))
                    offset_y = int(self.band_offsets.get(b, {'y': 0})['y'])
                    per_h = max(1, orig_per_h - abs(offset_y))
                    if info.get('split', False):
                        if merge_lr:
                            stitch_sequence.append({
                                'base': b, 'kind': 'paired_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                                'is_split': False, 'side': None
                            })
                        else:
                            stitch_sequence.append({
                                'base': b, 'kind': 'half_left', 'per_block_h': per_h, 'bin_factor': bin_factor,
                                'is_split': True, 'side': 'left'
                            })
                            stitch_sequence.append({
                                'base': b, 'kind': 'half_right', 'per_block_h': per_h, 'bin_factor': bin_factor,
                                'is_split': True, 'side': 'right'
                            })
                    else:
                        stitch_sequence.append({
                            'base': b, 'kind': 'full_unbinned', 'per_block_h': per_h, 'bin_factor': bin_factor,
                            'is_split': False, 'side': None
                        })
        cur_y = 0
        for e in stitch_sequence:
            e['start_y'] = cur_y
            cur_y += int(e['per_block_h']) + gap
        return stitch_sequence

    def _map_display_coords_for_geo(self, x, y):
        """Map display coords to raw-layout coords when 1:4 layout is enabled."""
        if not self._is_1_to_4_enabled():
            return x, y, False
        try:
            display_seq = self.build_stitch_sequence(for_display=True)
            raw_seq = self.build_stitch_sequence(for_display=False)
            if not display_seq or not raw_seq or len(display_seq) != len(raw_seq):
                return x, y, True
            for idx, disp_entry in enumerate(display_seq):
                start_y = disp_entry.get('start_y', 0)
                h = disp_entry.get('per_block_h', 0)
                if y >= start_y and y < (start_y + h):
                    raw_entry = raw_seq[idx]
                    offset = y - start_y
                    if disp_entry.get('kind') == 'full_binned':
                        factor = self._binned_upsample_factor(disp_entry.get('bin_factor', 1))
                        if factor > 1:
                            offset = int(offset / factor)
                    y_raw = int(raw_entry.get('start_y', 0) + offset)
                    return x, y_raw, True
        except Exception:
            pass
        return x, y, True

    def apply_offset(self, frame, offset_x=0, offset_y=0, crop=False, crop_y=False):
        if offset_x == 0 and offset_y == 0:
            return frame

        h, w = frame.shape
        result = np.zeros_like(frame)

        dst_x0 = max(0, offset_x)
        dst_x1 = min(w, w + offset_x)
        dst_y0 = max(0, offset_y)
        dst_y1 = min(h, h + offset_y)

        src_x0 = max(0, -offset_x)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y0 = max(0, -offset_y)
        src_y1 = src_y0 + (dst_y1 - dst_y0)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0 and src_x1 > src_x0 and src_y1 > src_y0:
            result[dst_y0:dst_y1, dst_x0:dst_x1] = frame[src_y0:src_y1, src_x0:src_x1]

        if crop:
            # Existing full crop (both X and Y) - unchanged
            non_zero_y = np.any(result != 0, axis=1)
            non_zero_x = np.any(result != 0, axis=0)
            if np.any(non_zero_y) and np.any(non_zero_x):
                y_min, y_max = np.where(non_zero_y)[0][[0, -1]]
                x_min, x_max = np.where(non_zero_x)[0][[0, -1]]
                return result[y_min:y_max+1, x_min:x_max+1]
            else:
                return np.zeros((0, 0), dtype=frame.dtype)
        elif crop_y:
            # New: Crop only Y, keep full width (pad X if needed)
            non_zero_y = np.any(result != 0, axis=1)
            if np.any(non_zero_y):
                y_min, y_max = np.where(non_zero_y)[0][[0, -1]]
                return result[y_min:y_max+1, :]
            else:
                return np.zeros((0, w), dtype=frame.dtype)  # Keep original width, height 0
        return result

    def _upsample_frame(self, frame, factor):
        if factor <= 1:
            return frame
        return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)

    def _pad_to_height(self, frame, target_h):
        if frame.shape[0] >= target_h:
            return frame
        pad_h = target_h - frame.shape[0]
        return np.pad(frame, ((0, pad_h), (0, 0)), mode='constant', constant_values=0)

    def apply_contrast_enhancement(self, frame):
        if not self.contrast_enhance_var.isChecked():
            return frame
        
        min_val = self.histogram_viewer.min_val
        max_val = self.histogram_viewer.max_val
        
        if max_val <= min_val:
            min_val = float(np.min(frame))
            max_val = float(np.max(frame))
        
        if self.contrast_min_var.value() != 0.0:
            min_val = self.contrast_min_var.value()
        try:
            default_max = float(self._current_max_dn()) if hasattr(self, '_current_max_dn') else 255.0
        except Exception:
            default_max = 255.0
        if self.contrast_max_var.value() != default_max:
            max_val = self.contrast_max_var.value()
        
        if max_val == min_val:
            return frame
        
        enhanced = ((frame.astype(np.float32) - min_val) / (max_val - min_val) * 255)
        return np.clip(enhanced, 0, 255).astype(np.uint8)


    def unload_view_widget(self, widget, name):
        if name == "All Bands":
            _safe_show_image(getattr(self, 'all_bands_viewer', None), None)
            try:
                if hasattr(self.all_bands_viewer, 'current_pil_image') and self.all_bands_viewer.current_pil_image:
                    del self.all_bands_viewer.current_pil_image
                if hasattr(self.all_bands_viewer, 'raw_pil_image') and self.all_bands_viewer.raw_pil_image:
                    del self.all_bands_viewer.raw_pil_image
            except:
                pass
        elif name == "Individual Bands":
            while self.individual_bands_notebook.count():
                sub_widget = self.individual_bands_notebook.widget(0)
                self.unload_individual_subtab(sub_widget)
                self.individual_bands_notebook.removeTab(0)
                sub_widget.deleteLater()
            self.individual_band_keys = []
            self.band_enabled = {}
            while self.band_checkbox_layout.count():
                item = self.band_checkbox_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        elif name == "RGB Fusion":
            _safe_show_image(getattr(self, 'rgb_preview_viewer', None), None)
            try:
                if hasattr(self.rgb_preview_viewer, 'current_pil_image') and self.rgb_preview_viewer.current_pil_image:
                    del self.rgb_preview_viewer.current_pil_image
                if hasattr(self.rgb_preview_viewer, 'raw_pil_image') and self.rgb_preview_viewer.raw_pil_image:
                    del self.rgb_preview_viewer.raw_pil_image
            except:
                pass
        elif name == "Histogram":
            self.histogram_viewer.clear()

    def _unload_data_only(self, key):
        """Unload data for a specific loaded band/Pan without removing the tab from the notebook."""
        print(f"[DEBUG] _unload_data_only called for key: {key}")
        if not key:
            print(f"[DEBUG] No key provided, skipping unload")
            return

        # Find the widget by key
        widget = None
        for i in range(self.individual_bands_notebook.count()):
            w = self.individual_bands_notebook.widget(i)
            if (hasattr(w, 'key') and w.key == key) or \
            (key == 'pan' and (getattr(w, 'key', None) == 'pan' or w.objectName() == "pan_placeholder")):
                widget = w
                break

        if not widget:
            print(f"[DEBUG] No widget found for key {key}, skipping unload")
            return

        # Perform unload similar to unload_individual_subtab
        if hasattr(widget, 'worker') and widget.worker and widget.worker.isRunning():
            widget.worker.requestInterruption()
            widget.worker.wait(2000)
            if widget.worker.isRunning():
                widget.worker.terminate()
                widget.worker.wait(1000)
            print(f"[DEBUG] Worker interrupted for {key}")
        if hasattr(widget, 'loading_timer') and widget.loading_timer:
            widget.loading_timer.stop()
            print(f"[DEBUG] Loading timer stopped for {key}")

        viewer = widget.findChild(GraphicsImageViewer)
        if viewer:
            _safe_show_image(viewer, None)
            try:
                if hasattr(viewer, 'current_pil_image') and viewer.current_pil_image:
                    del viewer.current_pil_image
                if hasattr(viewer, 'raw_pil_image') and viewer.raw_pil_image:
                    del viewer.raw_pil_image
            except Exception as e:
                print(f"[DEBUG] Error deleting images for {key}: {e}")
            # Detach and delete viewer
            viewer.setParent(None)
            viewer.deleteLater()
            print(f"[DEBUG] Viewer cleared and deleted for {key}")

        # Clear the layout completely
        layout = widget.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
                    item.widget().deleteLater()
            print(f"[DEBUG] Layout cleared for {key}")

        # Reset widget to placeholder state (preserves key for identification)
        widget.setObjectName("placeholder")
        widget.key = key  # Ensure key is set

        # Clean up memory trackers
        if key in self.loaded_band_memories:
            del self.loaded_band_memories[key]
            print(f"[DEBUG] Removed '{key}' from loaded_band_memories: {self.loaded_band_memories}")
        self.viewer_states.pop(key, None)

        # Flag as unloaded
        self.unloaded_keys.add(key)
        print(f"[DEBUG] Reset widget to 'placeholder' for {key}, added to unloaded_keys: {self.unloaded_keys}")

        # Optional: Update tab text to indicate unloaded (e.g., add (*))
        tab_idx = self.individual_bands_notebook.indexOf(widget)
        if tab_idx >= 0 and hasattr(widget, 'original_tab_text'):
            self.individual_bands_notebook.setTabText(tab_idx, f"{widget.original_tab_text} (*)")

        gc.collect()
        print(f"[DEBUG] GC collected after _unload_data_only for {key}")


    def update_views(self, full_refresh=False):
        current_tab = self.view_tabs.currentIndex()
        tab_name = self.view_tabs.tabText(current_tab) if current_tab >= 0 else ""
        
        # During playback, refresh only visible tab minimally (skip heavy ops)
        if self.playback_mode and not full_refresh:
            print("Optimized playback update: Refreshing only visible tab...")
            if "All Bands" in tab_name:
                # Quick refresh: Assume self.current_pil_image is already built in change_frame
                if hasattr(self, 'current_pil_image'):
                    self.all_bands_viewer.show_image(self.current_pil_image, fit_to_screen=False)
                else:
                    self.update_all_bands_view()  # Fallback to full if no cached image
            elif "Individual Bands" in tab_name:
                # Quick individual: Refresh current tab only
                if hasattr(self, 'individual_bands_notebook'):
                    current_i = self.individual_bands_notebook.currentIndex()
                    if current_i >= 0:
                        widget = self.individual_bands_notebook.widget(current_i)
                        if widget and hasattr(widget, 'key'):
                            key = widget.key
                            # Assume get_frame_for_band exists; else adapt
                            frame = self.get_frame_for_band(key, self.current_frame_index) if hasattr(self, 'get_frame_for_band') else None
                            if frame is not None:
                                pil = Image.fromarray(frame)
                                viewer = widget.findChild(GraphicsImageViewer)
                                if viewer:
                                    viewer.show_image(pil, fit_to_screen=False)
            elif "RGB Fusion" in tab_name:
                # Quick RGB: Single frame only, no multi-frame
                self._quick_single_rgb()  # Add this helper below
            # Skip histogram during playback
            return  # Early exit
        
        # Full update mode (non-playback or explicit refresh) - existing logic
        if "All Bands" in tab_name:
            self.update_all_bands_view()
        if "Individual Bands" in tab_name:
            self.update_individual_bands_view()
        if "RGB Fusion" in tab_name:
            self.preview_rgb_fusion()
        
        # Update histogram only if visible or full refresh (existing)
        if full_refresh or "Histogram" in tab_name:
            self.update_histogram_view()

    def _quick_single_rgb(self):
        if not self.band_frames:
            return
        # Simplified: Single frame RGB stack (adapt bands from your rgb_bands dict)
        keys = list(self.band_frames.keys())
        frames = self.band_frames.get(keys[0]) if keys else None
        if frames is None or self.current_frame_index >= len(frames):
            return
        h, w = frames.h, frames.w if hasattr(frames, 'h') else (frames[0].shape[0], frames[0].shape[1])
        rgb_display = []
        for channel in ["R", "G", "B"]:  # Assuming self.rgb_bands exists
            band_key = getattr(self, 'rgb_bands', {}).get(channel, keys[0] if keys else None)
            if band_key in self.band_frames and self.current_frame_index < len(self.band_frames[band_key]):
                frame = self.band_frames[band_key][self.current_frame_index]
                # Skip contrast/offset for speed
                rgb_display.append(frame)
            else:
                rgb_display.append(np.zeros((h, w), dtype=np.uint8))
        if len(rgb_display) == 3:
            rgb_array = np.stack(rgb_display, axis=-1)
            pil = Image.fromarray(rgb_array)
            self.rgb_preview_viewer.show_image(pil, fit_to_screen=False)
        
    def update_all_bands_view(self):
        if not self.band_frames:
            _safe_show_image(getattr(self, 'all_bands_viewer', None), None)
            return

        keys = [k for k in sorted(self.band_frames.keys()) if self.band_frames[k] is not None]
        if not keys:
            _safe_show_image(getattr(self, 'all_bands_viewer', None), None)
            return

        valid_lengths = [len(self.band_frames[k]) for k in keys if hasattr(self.band_frames[k], '__len__')]
        if not valid_lengths:
            _safe_show_image(getattr(self, 'all_bands_viewer', None), None)
            return

        # Ensure current frame index is within available range of all bands.
        min_frames = min(valid_lengths)
        if self.current_frame_index >= min_frames:
            self.current_frame_index = 0

        stitch_sequence = self.build_stitch_sequence()
        merge_lr = self._is_1_to_4_enabled()

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
                    max_width = max(max_width, _get_frame_width_for_key(key))
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

            # skip entries for bases we've already composed (prevents duplicates)
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

                # pad to max_width so vertical stacking aligns
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

                # align heights before horizontal stacking
                max_h = max(display_left.shape[0], display_right.shape[0])
                display_left = self._pad_to_height(display_left, max_h)
                display_right = self._pad_to_height(display_right, max_h)
                raw_left = self._pad_to_height(raw_left, max_h)
                raw_right = self._pad_to_height(raw_right, max_h)

                display_full = np.hstack([display_left, display_right])
                raw_full = np.hstack([raw_left, raw_right])

                # pad to max_width so vertical stacking keeps consistent width
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

            # fallback handling for any unknown kinds (process once)
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

        # add vertical gaps between parts
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
            self.all_bands_viewer.geo_info = self.geo_info
            _safe_show_image(
                self.all_bands_viewer,
                pil_display,
                fit_to_screen=(self.fit_mode_var.checkedId() == 0),
                raw_pil=pil_raw
            )
            try:
                self.all_bands_viewer.original_raw_data = full_raw
            except RuntimeError:
                pass

            # free memory
            del full_display, full_raw, pil_display, pil_raw, parts, raw_parts
        else:
            _safe_show_image(getattr(self, 'all_bands_viewer', None), None)

        gc.collect()

    def _update_dots(self, widget):
        if not hasattr(widget, 'dot_count'):
            widget.dot_count = 0
        widget.dot_count = (widget.dot_count % 4) + 1
        widget.loading_label.setText(f"Loading{'.' * widget.dot_count}")

    def lazy_load_individual_tab(self, idx):
        if idx < 0:
            return
        print(f"[DEBUG] lazy_load_individual_tab called for idx {idx}")
        widget = self.individual_bands_notebook.widget(idx)
        if not widget:
            print(f"[DEBUG] Invalid widget state for idx {idx}")
            return
        obj_name = widget.objectName()
        print(f"[DEBUG] lazy_load_individual_tab: widget objectName = {obj_name}")
        if obj_name not in ["placeholder", "pan_placeholder"]:
            print(f"[DEBUG] Invalid widget state (not placeholder) for idx {idx}, obj_name={obj_name}")
            return
        is_pan = widget.objectName() == "pan_placeholder"
        is_placeholder = widget.objectName() == "placeholder"
        if not (is_placeholder or is_pan):
            return  # Already loaded or not a placeholder

        # Replace placeholder with loading layout
        loading_layout = QVBoxLayout()
        widget.setLayout(loading_layout)

        widget.loading_label = QLabel("Loading")
        widget.loading_label.setAlignment(Qt.AlignCenter)
        loading_layout.addWidget(widget.loading_label)

        widget.loading_progress = QProgressBar()
        widget.loading_progress.setRange(0, 100)
        widget.loading_progress.setValue(0)
        widget.loading_progress.setTextVisible(False)
        widget.loading_progress.setFixedHeight(4)
        loading_layout.addWidget(widget.loading_progress)

        widget.setObjectName("loading")

        # Start blinking timer
        widget.loading_timer = QTimer()
        widget.loading_timer.timeout.connect(lambda: self._update_dots(widget))
        widget.loading_timer.start(500)
        widget.dot_count = 0
        key_or_pan = getattr(widget, 'key', 'pan')
        print(f"[DEBUG] Starting worker for {getattr(widget, 'key', 'pan')}")
        # Start worker for computing image
        enhance = self.contrast_enhance_var.isChecked()
        is_range = self.frame_mode_var.checkedId() != 0
        start_frame = self.start_frame_entry.value() - 1 if is_range else self.current_frame_index
        end_frame = self.end_frame_entry.value() - 1 if is_range else self.current_frame_index
        gap = self.gap_var.value()

        if is_pan:
            base_keys = sorted(set(k.rsplit('_', 1)[0] for k in widget.unbinned_keys))
            offsets = {base: self.band_offsets.get(base, {'x':0, 'y':0}) for base in base_keys}
            worker = PanBandWorker(self.band_frames, widget.unbinned_keys, enhance, offsets, start_frame, end_frame, gap, self)
            worker.finished.connect(lambda images: self._on_pan_band_loaded(widget, images))
            worker.error.connect(lambda err: self._on_pan_band_error(widget, err))
            worker.progress.connect(widget.loading_progress.setValue)
        else:
            key = widget.key
            if key.endswith("_merged"):
                base_key = key.rsplit('_', 1)[0]
                left_key, right_key = self._get_unbinned_pair_keys(base_key)
                if not left_key or not right_key:
                    left_key = f"{base_key}_left"
                    right_key = f"{base_key}_right"
                offset_x = self.band_offsets.get(base_key, {'x':0, 'y':0})['x']
                offset_y = self.band_offsets.get(base_key, {'x':0, 'y':0})['y']
                worker = MergedBandWorker(self.band_frames, left_key, right_key, enhance, offset_x, offset_y, start_frame, end_frame, gap, self)
                worker.finished.connect(lambda images, k=key: self._on_individual_band_loaded(widget, images, k))
                worker.error.connect(lambda err: self._on_individual_band_error(widget, err))
                worker.progress.connect(widget.loading_progress.setValue)
            else:
                base_key = key.rsplit('_', 1)[0] if '_' in key else key
                offset_x = self.band_offsets.get(base_key, {'x':0, 'y':0})['x']
                offset_y = self.band_offsets.get(base_key, {'x':0, 'y':0})['y']
                worker = IndividualBandWorker(self.band_frames[key], enhance, offset_x, offset_y, start_frame, end_frame, gap, self)
                worker.finished.connect(lambda images, k=key: self._on_individual_band_loaded(widget, images, k))
                worker.error.connect(lambda err: self._on_individual_band_error(widget, err))
                worker.progress.connect(widget.loading_progress.setValue)
        if key in self.unloaded_keys:  # If previously unloaded, mark as loading from unloaded
            self.unloaded_keys.discard(key)
        worker.start()
        print(f"[DEBUG] Worker started for {key_or_pan}")
        widget.worker = worker  # Keep reference to worker

    def _on_individual_band_loaded(self, widget, images, key):
        if not widget or widget.objectName() != "loading":
            return  # Tab changed or removed, ignore
        # Stop timer
        if hasattr(widget, 'loading_timer') and widget.loading_timer:
            widget.loading_timer.stop()
        # Clear loading layout
        while widget.layout().count():
            item = widget.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        viewer = GraphicsImageViewer(
            parent=self,
            pixel_info_callback=self.update_pixel_info,
            matrix_size_var=self.matrix_size_var
        )
        widget.layout().addWidget(viewer)
        widget.setObjectName("loaded")
        if key in self.unloaded_keys:
            self.unloaded_keys.discard(key)
        widget.key = key

        # Restore shared and per-band states
        viewer.graphics_view.mouse_zoom_enabled = self.shared_mouse_zoom_enabled
        viewer.graphics_view.toggle_magnifier(self.shared_magnifier_enabled)
        viewer.graphics_view.set_magnifier_zoom(int(self.shared_magnifier_zoom * 10))
        viewer.graphics_view.magnifier_radius = self.shared_magnifier_radius
        viewer.graphics_view.torch_enabled = self.shared_magnifier_torch
        viewer.flip_mode = 0

        state = self.viewer_states.get(key, {})
        if state:
            viewer.zoom = state.get('zoom', 1.0)
            viewer.rotation = state.get('rotation', 0.0)
            viewer.graphics_view.setTransform(QTransform().rotate(state.get('rotation', 0.0)).scale(state.get('zoom', 1.0), state.get('zoom', 1.0)))
            h_scroll = viewer.graphics_view.horizontalScrollBar()
            v_scroll = viewer.graphics_view.verticalScrollBar()
            if h_scroll: h_scroll.setValue(state.get('scroll_x', 0))
            if v_scroll: v_scroll.setValue(state.get('scroll_y', 0))

        # Show image
        pil_display = images['display']
        pil_raw = images['raw']
        viewer.geo_info = self.geo_info
        viewer.show_image(pil_display, fit_to_screen=(self.fit_mode_var.checkedId() == 0), raw_pil=pil_raw)
        if isinstance(images, dict) and images.get('raw_array') is not None:
            viewer.original_raw_data = images.get('raw_array')
        # Update tab text
        tab_idx = self.individual_bands_notebook.indexOf(widget)
        if hasattr(widget, 'original_tab_text'):
            self.individual_bands_notebook.setTabText(tab_idx, widget.original_tab_text)
        
        del images  # Free worker data
        gc.collect()

    def _on_pan_band_loaded(self, widget, images):
        if not widget or widget.objectName() != "loading":
            return
        # Stop timer
        if hasattr(widget, 'loading_timer') and widget.loading_timer:
            widget.loading_timer.stop()
        # Clear loading
        while widget.layout().count():
            item = widget.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        viewer = GraphicsImageViewer(
            parent=self,
            pixel_info_callback=self.update_pixel_info,
            matrix_size_var=self.matrix_size_var
        )
        widget.layout().addWidget(viewer)
        widget.setObjectName("loaded")
        if 'pan' in self.unloaded_keys:
            self.unloaded_keys.discard('pan')

        # Restore shared and per-band (use 'pan' key)
        viewer.graphics_view.mouse_zoom_enabled = self.shared_mouse_zoom_enabled
        viewer.graphics_view.toggle_magnifier(self.shared_magnifier_enabled)
        viewer.graphics_view.set_magnifier_zoom(int(self.shared_magnifier_zoom * 10))
        viewer.graphics_view.magnifier_radius = self.shared_magnifier_radius
        viewer.graphics_view.torch_enabled = self.shared_magnifier_torch
        viewer.flip_mode = 0

        pan_key = 'pan'
        state = self.viewer_states.get(pan_key, {})
        if state:
            viewer.zoom = state.get('zoom', 1.0)
            viewer.rotation = state.get('rotation', 0.0)
            viewer.graphics_view.setTransform(QTransform().rotate(state.get('rotation', 0.0)).scale(state.get('zoom', 1.0)))
            h_scroll = viewer.graphics_view.horizontalScrollBar()
            v_scroll = viewer.graphics_view.verticalScrollBar()
            if h_scroll: h_scroll.setValue(state.get('scroll_x', 0))
            if v_scroll: v_scroll.setValue(state.get('scroll_y', 0))

        # Show
        pil_display = images['display']
        pil_raw = images['raw']
        viewer.geo_info = self.geo_info
        viewer.show_image(pil_display, fit_to_screen=(self.fit_mode_var.checkedId() == 0), raw_pil=pil_raw)
        if isinstance(images, dict) and images.get('raw_array') is not None:
            viewer.original_raw_data = images.get('raw_array')
        # Update tab text
        tab_idx = self.individual_bands_notebook.indexOf(widget)
        if hasattr(widget, 'original_tab_text'):
            self.individual_bands_notebook.setTabText(tab_idx, widget.original_tab_text)
        
        del images  # Free worker data
        gc.collect()

    def _on_pan_band_error(self, widget, err):
        if not widget or widget.objectName() != "loading":
            return
        print(f"[ERROR] Pan band loading error: {err}")
        # Stop timer
        if hasattr(widget, 'loading_timer') and widget.loading_timer:
            widget.loading_timer.stop()
        # Clear loading layout
        while widget.layout().count():
            item = widget.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Add error message
        error_label = QLabel(f"Error loading pan band: {err}")
        error_label.setStyleSheet("color: red; font-weight: bold;")
        widget.layout().addWidget(error_label)
        widget.setObjectName("error")

    def _on_individual_band_error(self, widget, err):
        if not widget or widget.objectName() != "loading":
            return
        print(f"[ERROR] Individual band loading error: {err}")
        # Stop timer
        if hasattr(widget, 'loading_timer') and widget.loading_timer:
            widget.loading_timer.stop()
        # Clear loading layout
        while widget.layout().count():
            item = widget.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Add error message
        error_label = QLabel(f"Error loading band: {err}")
        error_label.setStyleSheet("color: red; font-weight: bold;")
        widget.layout().addWidget(error_label)
        widget.setObjectName("error")

    def _setup_individual_tab_loading_signal(self):
        if hasattr(self, '_individual_tab_connected') and self._individual_tab_connected:
            return
        def on_tab_changed(idx):
            print(f"[DEBUG] Tab changed to index {idx}")
            if idx < 0:
                return
            widget = self.individual_bands_notebook.widget(idx)
            obj_name = widget.objectName() if widget else 'None'
            w_key = getattr(widget, 'key', 'None')
            print(f"[DEBUG] Widget objectName: {obj_name}, key: {w_key}")
            if widget and obj_name in ["placeholder", "pan_placeholder"]:
                print(f"[DEBUG] Detected placeholder/pan_placeholder at {idx}, calling lazy_load_individual_tab")
                self.lazy_load_individual_tab(idx)
            else:
                print(f"[DEBUG] Widget not a placeholder (obj_name='{obj_name}'), skipping load")
        self.individual_bands_notebook.currentChanged.connect(on_tab_changed)
        self._individual_tab_connected = True
        print("[DEBUG] Individual tab loading signal connected")

    def update_individual_bands_view(self):
        # Save checked states before clearing
        enabled_states = {
            k: cb.isChecked()
            for k, cb in self.band_enabled.items()
            if cb and cb.parent() is not None
        }
        # If left/right were previously enabled, carry that state into merged checkbox.
        for k, enabled in list(enabled_states.items()):
            if not enabled:
                continue
            base = self._unbinned_base_for_key(k)
            if base:
                enabled_states[f"{base}_merged"] = True

        # Clear existing tabs/widgets
        while self.individual_bands_notebook.count():
            widget = self.individual_bands_notebook.widget(0)
            self.unload_individual_subtab(widget)
            self.individual_bands_notebook.removeTab(0)
            widget.deleteLater()

        # Clear band checkbox layout
        while self.band_checkbox_layout.count():
            item = self.band_checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.band_frames:
            return

        # Clear old references
        self.band_enabled = {}


        container_width = max(1, self.band_checkbox_container.width())
        checkbox_width = 130  # tuned to match your UI
        columns = max(1, container_width // checkbox_width)

        row = 0
        col = 0

        # Re-create checkboxes for loaded bands.
        # Always merge unbinned left/right into a single checkbox in Individual view.
        checkbox_entries = []
        bands_info_local = getattr(self, 'bands_info', {}) or {}
        if bands_info_local:
            ordered_bases = sorted(bands_info_local.keys(), key=lambda k: bands_info_local[k]['index'])
            for base in ordered_bases:
                info = bands_info_local.get(base, {})
                if info.get('split', False) and not info.get('binned', False):
                    left_key, right_key = self._get_unbinned_pair_keys(base)
                    if left_key and right_key:
                        checkbox_entries.append((f"{base}_merged", f"Band {base[1:]} (Merged)"))
                        continue
                # Fallback to a concrete key if present
                for k in info.get('variants', []):
                    if k in self.band_frames:
                        checkbox_entries.append((k, f"Band {k[1:]}"))
                        break
        else:
            merged_bases = set()
            for k in sorted(self.band_frames.keys()):
                base = self._unbinned_base_for_key(k)
                if base:
                    if base in merged_bases:
                        continue
                    checkbox_entries.append((f"{base}_merged", f"Band {base[1:]} (Merged)"))
                    merged_bases.add(base)
                else:
                    checkbox_entries.append((k, f"Band {k[1:]}"))

        for key, label in checkbox_entries:
            cb = QCheckBox(label)
            cb.blockSignals(True)
            cb.setChecked(enabled_states.get(key, False))
            cb.blockSignals(False)
            cb.stateChanged.connect(lambda state, k=key: self.toggle_band(k, state))

            self.band_checkbox_layout.addWidget(cb, row, col)
            self.band_enabled[key] = cb

            col += 1
            if col >= columns:
                col = 0
                row += 1

        # Pan is intentionally disabled for merged unbinned bands in Individual view.

        # Get checked keys
        keys = []
        merged_bases = set()
        for k in sorted(self.band_enabled.keys()):
            cb = self.band_enabled.get(k)
            if not (cb and cb.isChecked()):
                continue
            base = self._unbinned_base_for_key(k)
            if base:
                if base in merged_bases:
                    continue
                keys.append(f"{base}_merged")
                merged_bases.add(base)
            else:
                keys.append(k)

        self.individual_band_keys = keys  # Save for lazy loading

        # Add placeholder tabs
        for key in keys:
            placeholder = QWidget()
            placeholder.setObjectName("placeholder")
            placeholder.key = key

            if key.endswith("_merged"):
                base_key = key.rsplit('_', 1)[0]
                tab_text = f"Band {base_key[1:]}".strip()
            else:
                base_key = key.rsplit('_', 1)[0] if '_' in key else key
                side = key.split('_')[-1] if '_' in key else ''
                tab_text = f"Band {base_key[1:]} {side}".strip()

            placeholder.original_tab_text = tab_text
            self.individual_bands_notebook.addTab(
                placeholder, f"{tab_text} (*)"
            )

        # Pan tab intentionally omitted for merged unbinned bands.

        self._setup_individual_tab_loading_signal()

        gc.collect()

    def toggle_band(self, key, state):
        if key == "pan":
            cb = self.band_enabled.get("pan")
            if cb:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            return

        # Always merge left/right for unbinned bands in Individual view.
        base = self._unbinned_base_for_key(key)
        if base:
            merged_key = f"{base}_merged"
            if state == Qt.Checked:
                key = merged_key
            else:
                other_key = None
                if key.endswith('_left'):
                    other_key = f"{base}_right"
                elif key.endswith('_right'):
                    other_key = f"{base}_left"
                elif key.endswith('0'):
                    other_key = f"{base}1"
                elif key.endswith('1'):
                    other_key = f"{base}0"
                other_cb = self.band_enabled.get(other_key) if other_key else None
                if other_cb and other_cb.isChecked():
                    return
                key = merged_key

        if key == "pan":
            # Handle pan uncheck: remove if exists
            tab_idx = -1
            for i in range(self.individual_bands_notebook.count()):
                widget = self.individual_bands_notebook.widget(i)
                if getattr(widget, 'key', None) == 'pan' or widget.objectName() == "pan_placeholder":
                    tab_idx = i
                    break
            if tab_idx >= 0 and state == Qt.Unchecked:
                widget = self.individual_bands_notebook.widget(tab_idx)
                self.unload_individual_subtab(widget)
                self.individual_bands_notebook.removeTab(tab_idx)
                if hasattr(self, 'loaded_band_memories') and 'pan' in self.loaded_band_memories:
                    del self.loaded_band_memories['pan']
                    print(f"Removed 'pan' from loaded_band_memories: {self.loaded_band_memories}")
                # Track as unloaded to trigger reload on re-check
                self.unloaded_keys.add('pan')
                print(f"[DEBUG] Added 'pan' to unloaded_keys: {self.unloaded_keys}")
                return
            # Handle pan check: add if not exists
            if state == Qt.Checked:
                unbinned_keys = [k for k, cb in self.band_enabled.items() if cb.isChecked() and k.endswith(('_left', '_right'))]
                if not unbinned_keys:
                    cb = self.band_enabled.get("pan")
                    if cb:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                    QMessageBox.warning(self, "No Unbinned Bands", "No unbinned bands (left/right) selected for Pan tab.")
                    return
                # Predict memory
                predicted = self._predict_band_memory(None, True, unbinned_keys)
                if not hasattr(self, 'loaded_band_memories'):
                    self.loaded_band_memories = {}
                current_used = sum(self.loaded_band_memories.values())

                # Use available memory and keep 1GB headroom
                vm = psutil.virtual_memory()
                available = vm.available
                headroom = 1 * 1024**3  # 1 GB
                max_allowed = max(0, available - headroom)
                print(f"[MEM] available={available}, headroom={headroom}, current_used={current_used}, predicted={predicted}")

                if current_used + predicted > max_allowed:
                    cb = self.band_enabled.get("pan")
                    if cb:
                        cb.blockSignals(True)
                        cb.setChecked(False)
                        cb.blockSignals(False)
                    QMessageBox.warning(
                        self,
                        "Memory Limit Reached",
                        f"Adding Pan tab would need {(predicted / 1e9):.2f} GB but only {(max_allowed - current_used) / 1e9:.2f} GB is safely available (keeping 1GB headroom)."
                    )
                    return
                # Add placeholder (no immediate load)
                self.loaded_band_memories['pan'] = predicted
                print(f"Reserved 'pan' => {predicted}, loaded_band_memories: {self.loaded_band_memories}")

                # Now add the lightweight placeholder (no heavy allocs yet)
                placeholder = QWidget()
                placeholder.setObjectName("pan_placeholder")
                placeholder.key = 'pan'
                placeholder.unbinned_keys = unbinned_keys
                placeholder.original_tab_text = "Pan"
                pan_idx = self.individual_bands_notebook.addTab(placeholder, "Pan (*)")
                self._setup_individual_tab_loading_signal()
                # Handle if previously unloaded: trigger immediate reload
                if 'pan' in self.unloaded_keys:
                    self.unloaded_keys.discard('pan')
                    print(f"[DEBUG] Discarded 'pan' from unloaded_keys, triggering _reload_band_data")
                    QTimer.singleShot(0, lambda: self._reload_band_data('pan'))
                else:
                    print(f"[DEBUG] 'pan' not in unloaded_keys, added as new placeholder (lazy load on visit)")
                return

        # Handle normal band (non-pan)
        if state == Qt.Checked:
            # Check if already exists (loaded or placeholder)
            existing_idx = -1
            for i in range(self.individual_bands_notebook.count()):
                widget = self.individual_bands_notebook.widget(i)
                if hasattr(widget, 'key') and widget.key == key:
                    existing_idx = i
                    break
            if existing_idx >= 0:
                # If it's a placeholder, switch to it to trigger load
                self.individual_bands_notebook.setCurrentIndex(existing_idx)
                print(f"[DEBUG] Switched to existing tab for {key} (index {existing_idx}) to trigger load")
                return
            # Add placeholder (no immediate load)
            placeholder = QWidget()
            placeholder.setObjectName("placeholder")
            placeholder.key = key
            if key.endswith("_merged"):
                base_key = key.rsplit('_', 1)[0]
                tab_text = f"Band {base_key[1:]}".strip()
            else:
                base_key = key.rsplit('_', 1)[0] if '_' in key else key
                side = key.split('_')[-1] if '_' in key else ''
                tab_text = f"Band {base_key[1:]} {side}".strip()
            placeholder.original_tab_text = tab_text
            tab_idx = self.individual_bands_notebook.addTab(placeholder, f"{tab_text} (*)")
            self.individual_band_keys.append(key)
            self._setup_individual_tab_loading_signal()
            print(f"[DEBUG] Added new placeholder for {key} at index {tab_idx}")
            # Handle if previously unloaded: trigger immediate reload
            if key in self.unloaded_keys:
                self.unloaded_keys.discard(key)
                print(f"[DEBUG] Discarded {key} from unloaded_keys, triggering _reload_band_data")
                QTimer.singleShot(0, lambda: self._reload_band_data(key))
            else:
                print(f"[DEBUG] {key} not in unloaded_keys, added as new placeholder (lazy load on visit)")
        else:
            # Remove specific tab
            tab_idx = -1
            for i in range(self.individual_bands_notebook.count()):
                widget = self.individual_bands_notebook.widget(i)
                if hasattr(widget, 'key') and widget.key == key:
                    tab_idx = i
                    break
            if tab_idx >= 0:
                widget = self.individual_bands_notebook.widget(tab_idx)
                self.unload_individual_subtab(widget)
                self.individual_bands_notebook.removeTab(tab_idx)
                self.viewer_states.pop(key, None)
                if key in self.individual_band_keys:
                    self.individual_band_keys.remove(key)
                if hasattr(self, 'loaded_band_memories') and key in self.loaded_band_memories:
                    del self.loaded_band_memories[key]
                    print(f"Removed '{key}' from loaded_band_memories: {self.loaded_band_memories}")
                # Track as unloaded to trigger reload on re-check
                self.unloaded_keys.add(key)
                print(f"[DEBUG] Added {key} to unloaded_keys: {self.unloaded_keys}")
            else:
                print(f"[DEBUG] No tab found to unload for {key}")
        # Update sender reference (if needed for other logic)
        if hasattr(self, 'band_enabled') and key in self.band_enabled:
            self.band_enabled[key] = self.sender()

    def update_histogram_view(self):
        if not self.band_frames:
            return
        
        keys = sorted(self.band_frames.keys())
        frames = self.band_frames.get(keys[0]) if keys else None
        if frames is None or self.current_frame_index >= len(frames):
            self.current_frame_index = 0

        folder = getattr(self, "folder", "")
        if self.histogram_viewer.single_frame_radio.isChecked():
            self.histogram_viewer.update_histogram(
                {k: self.band_frames[k] for k in keys if self.band_frames[k] is not None},
                self.current_frame_index,
                "Single",
                folder=folder,
            )
        else:
            try:
                start_frame = self.start_frame_entry.value() - 1
                end_frame = self.end_frame_entry.value() - 1
                if start_frame < 0 or end_frame >= len(frames) or start_frame > end_frame:
                    return

                self.histogram_viewer.update_histogram(
                    {k: self.band_frames[k] for k in keys if self.band_frames[k] is not None},
                    self.current_frame_index,
                    "Range",
                    start_frame,
                    end_frame,
                    folder=folder,
                )
            except ValueError:
                return
        gc.collect()

    def preview_rgb_fusion(self):
        if not self.band_frames:
            self.rgb_preview_viewer.show_image(None)
            return

        self.rgb_bands["R"] = self.red_band_var.currentText()
        self.rgb_bands["G"] = self.green_band_var.currentText()
        self.rgb_bands["B"] = self.blue_band_var.currentText()

        keys = sorted(self.band_frames.keys())
        frames = self.band_frames.get(keys[0]) if keys else None
        if frames is None or self.current_frame_index >= len(frames):
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
            def get_frame(channel, i):
                band_key = self.rgb_bands[channel]
                base_key = band_key.rsplit('_', 1)[0] if '_' in band_key else band_key
                if band_key not in self.band_frames or i >= len(self.band_frames[band_key]):
                    return None, None
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
                return display_frame, raw_frame

            parts_display = []
            parts_raw = []

            if rgb_mode == "Single":
                start_frame = self.current_frame_index
                end_frame = self.current_frame_index
                gap = 0
            else:
                start_frame = self.start_frame_entry.value() - 1
                end_frame = self.end_frame_entry.value() - 1
                gap = self.gap_var.value()

            frame_indices = list(range(start_frame, end_frame + 1))
            num_frames = len(frame_indices)
            if num_frames <= 0:
                self.rgb_preview_viewer.show_image(None)
                return

            # RAM guard for "All Frames" RGB fusion. If estimated memory is high,
            # decimate frame indices so the preview remains responsive and avoids OOM kill.
            if rgb_mode != "Single":
                sample_idx = frame_indices[0]
                r_display_s, r_raw_s = get_frame("R", sample_idx)
                g_display_s, g_raw_s = get_frame("G", sample_idx)
                b_display_s, b_raw_s = get_frame("B", sample_idx)
                if any(v is not None for v in (r_display_s, g_display_s, b_display_s)):
                    h_s = max(
                        r_display_s.shape[0] if r_display_s is not None else 0,
                        g_display_s.shape[0] if g_display_s is not None else 0,
                        b_display_s.shape[0] if b_display_s is not None else 0
                    )
                    w_s = 0
                    if r_display_s is not None:
                        w_s = r_display_s.shape[1]
                    elif g_display_s is not None:
                        w_s = g_display_s.shape[1]
                    elif b_display_s is not None:
                        w_s = b_display_s.shape[1]
                    offsets_x = [offset_r_x, offset_g_x, offset_b_x]
                    offsets_y = [offset_r_y, offset_g_y, offset_b_y]
                    min_x = min(0, *offsets_x)
                    min_y = min(0, *offsets_y)
                    max_x = max(0, *offsets_x) + w_s
                    max_y = max(0, *offsets_y) + h_s
                    total_w_s = max_x - min_x
                    total_h_s = max_y - min_y
                    raw_dtype = None
                    for rr in (r_raw_s, g_raw_s, b_raw_s):
                        if rr is not None:
                            raw_dtype = rr.dtype
                            break
                    raw_itemsize = int(np.dtype(raw_dtype).itemsize) if raw_dtype is not None else 1
                    # Conservative estimate includes display+raw and intermediate copies.
                    bytes_per_frame = int(total_h_s) * int(total_w_s) * 3 * (1 + raw_itemsize)
                    expected_bytes = int(bytes_per_frame * num_frames * 3)
                    safe_bytes = int(psutil.virtual_memory().available * 0.35)
                    if safe_bytes > 0 and expected_bytes > safe_bytes:
                        step = max(2, (expected_bytes + safe_bytes - 1) // safe_bytes)
                        frame_indices = frame_indices[::step]
                        num_frames = len(frame_indices)
                        try:
                            QMessageBox.information(
                                self,
                                "RGB Fusion Preview",
                                f"Frame range is large for available RAM.\n"
                                f"Previewing every {step}th frame ({num_frames} frames) to avoid memory kill."
                            )
                        except Exception:
                            pass

            frame_h = 0  # To be set from first frame

            keep_highbit_raw = (rgb_mode == "Single")

            for out_idx, i in enumerate(frame_indices):
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
                r_display, r_raw = get_frame("R", i)
                g_display, g_raw = get_frame("G", i)
                b_display, b_raw = get_frame("B", i)

                if r_display is None and g_display is None and b_display is None:
                    continue

                h_r = r_display.shape[0] if r_display is not None else 0
                h_g = g_display.shape[0] if g_display is not None else 0
                h_b = b_display.shape[0] if b_display is not None else 0
                max_h = max(h_r, h_g, h_b)

                if r_display is None:
                    width = r_raw.shape[1] if r_raw is not None else g_display.shape[1]
                    r_display = np.zeros((max_h, width), dtype=np.uint8)
                    r_raw = np.zeros((max_h, width), dtype=(r_raw.dtype if r_raw is not None else np.uint8))
                elif h_r < max_h:
                    pad_top = (max_h - h_r) // 2
                    pad_bottom = max_h - h_r - pad_top
                    r_display = np.pad(r_display, ((pad_top, pad_bottom), (0, 0)), 'constant')
                    r_raw = np.pad(r_raw, ((pad_top, pad_bottom), (0, 0)), 'constant')

                if g_display is None:
                    width = g_raw.shape[1] if g_raw is not None else r_display.shape[1]
                    g_display = np.zeros((max_h, width), dtype=np.uint8)
                    g_raw = np.zeros((max_h, width), dtype=(g_raw.dtype if g_raw is not None else np.uint8))
                elif h_g < max_h:
                    pad_top = (max_h - h_g) // 2
                    pad_bottom = max_h - h_g - pad_top
                    g_display = np.pad(g_display, ((pad_top, pad_bottom), (0, 0)), 'constant')
                    g_raw = np.pad(g_raw, ((pad_top, pad_bottom), (0, 0)), 'constant')

                if b_display is None:
                    width = b_raw.shape[1] if b_raw is not None else r_display.shape[1]
                    b_display = np.zeros((max_h, width), dtype=np.uint8)
                    b_raw = np.zeros((max_h, width), dtype=(b_raw.dtype if b_raw is not None else np.uint8))
                elif h_b < max_h:
                    pad_top = (max_h - h_b) // 2
                    pad_bottom = max_h - h_b - pad_top
                    b_display = np.pad(b_display, ((pad_top, pad_bottom), (0, 0)), 'constant')
                    b_raw = np.pad(b_raw, ((pad_top, pad_bottom), (0, 0)), 'constant')

                if not (r_display.shape == g_display.shape == b_display.shape):
                    QMessageBox.warning(self, "Warning", "Channels have different dimensions after padding, RGB fusion may be incorrect.")
                    continue

                offsets_x = [offset_r_x, offset_g_x, offset_b_x]
                offsets_y = [offset_r_y, offset_g_y, offset_b_y]
                min_x = min(0, *offsets_x)
                min_y = min(0, *offsets_y)
                max_x = max(0, *offsets_x) + r_display.shape[1]
                max_y = max(0, *offsets_y) + max_h
                total_w = max_x - min_x
                total_h_frame = max_y - min_y

                if frame_h == 0:
                    frame_h = total_h_frame  # Set from first frame

                def pad_channel(channel_arr, off_x, off_y):
                    pad_left = off_x - min_x
                    pad_right = total_w - (pad_left + channel_arr.shape[1])
                    pad_top = off_y - min_y
                    pad_bottom = total_h_frame - (pad_top + channel_arr.shape[0])
                    return np.pad(channel_arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

                r_padded_display = pad_channel(r_display, offset_r_x, offset_r_y)
                g_padded_display = pad_channel(g_display, offset_g_x, offset_g_y)
                b_padded_display = pad_channel(b_display, offset_b_x, offset_b_y)

                if keep_highbit_raw:
                    r_padded_raw = pad_channel(r_raw, offset_r_x, offset_r_y)
                    g_padded_raw = pad_channel(g_raw, offset_g_x, offset_g_y)
                    b_padded_raw = pad_channel(b_raw, offset_b_x, offset_b_y)
                else:
                    # For range preview, keep raw stack lightweight to avoid OOM.
                    r_padded_raw = r_padded_display
                    g_padded_raw = g_padded_display
                    b_padded_raw = b_padded_display

                rgb_frame_display = np.stack([r_padded_display, g_padded_display, b_padded_display], axis=-1)
                rgb_frame_raw = np.stack([r_padded_raw, g_padded_raw, b_padded_raw], axis=-1)

                parts_display.append(rgb_frame_display)
                parts_raw.append(rgb_frame_raw)

                if out_idx < num_frames - 1:
                    gap_rgb = np.zeros((gap, total_w, 3), dtype=np.uint8)
                    parts_display.append(gap_rgb)
                    gap_rgb_raw = np.zeros((gap, total_w, 3), dtype=(rgb_frame_raw.dtype if keep_highbit_raw else np.uint8))
                    parts_raw.append(gap_rgb_raw)

            if parts_display:
                full_display = np.vstack(parts_display)
                full_raw = np.vstack(parts_raw)

                pil_display = Image.fromarray(full_display)
                # For RGB fusion, `full_raw` can be uint16 RGB; PIL does not support that well.
                # We keep raw DN data in `original_raw_data` (set below) and pass preview PIL.
                pil_raw = pil_display

                # Set properties for rotation handling
                self.rgb_preview_viewer.is_individual = True
                self.rgb_preview_viewer.is_rgb_fusion = True  # NEW: Flag for RGB-specific handling
                self.rgb_preview_viewer.frame_h = frame_h
                self.rgb_preview_viewer.per_block_h = frame_h  # Use actual fused height
                self.rgb_preview_viewer.gap = gap if rgb_mode == "All" else 0
                self.rgb_preview_viewer.is_frame_stack = (rgb_mode == "All")

                self.rgb_preview_viewer.show_image(pil_display, fit_to_screen=(self.fit_mode_var.checkedId() == 0), raw_pil=pil_raw)
                # Keep full DN matrix only for single-frame mode; for range mode this is too heavy.
                # Pixel info still works in range mode using displayed 8-bit data.
                if rgb_mode == "Single":
                    self.rgb_preview_viewer.original_raw_data = full_raw
                else:
                    self.rgb_preview_viewer.original_raw_data = full_display

                del full_display, full_raw, parts_display, parts_raw
                gc.collect()
            else:
                self.rgb_preview_viewer.show_image(None)

        except Exception as e:
            print(f"Error in preview_rgb_fusion: {e}")
            self.rgb_preview_viewer.show_image(None)
        finally:
            pass
        gc.collect()

    def fit_to_screen(self):
        current_tab = self.view_tabs.currentIndex()
        tab_name = self.view_tabs.tabText(current_tab) if current_tab >= 0 else ""
        if tab_name == "All Bands":
            self.all_bands_viewer.fit_to_screen()
        elif tab_name == "RGB Fusion":
            self.rgb_preview_viewer.fit_to_screen()
        elif tab_name == "Individual Bands":
            current_i = self.individual_bands_notebook.currentIndex()
            if current_i >= 0:
                widget = self.individual_bands_notebook.widget(current_i)
                if widget:
                    viewer = widget.findChild(GraphicsImageViewer)
                    if viewer:
                        viewer.fit_to_screen()

    def actual_size(self):
        current_tab = self.view_tabs.currentIndex()
        tab_name = self.view_tabs.tabText(current_tab) if current_tab >= 0 else ""
        if tab_name == "All Bands":
            self.all_bands_viewer.actual_size()
        elif tab_name == "RGB Fusion":
            self.rgb_preview_viewer.actual_size()
        elif tab_name == "Individual Bands":
            current_i = self.individual_bands_notebook.currentIndex()
            if current_i >= 0:
                widget = self.individual_bands_notebook.widget(current_i)
                if widget:
                    viewer = widget.findChild(GraphicsImageViewer)
                    if viewer:
                        viewer.actual_size()
