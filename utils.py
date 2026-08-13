import os
import sys
import glob, shutil, getpass, socket, platform
import json
import sqlite3
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox, QDialog, QListWidget, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal, QEvent, Qt, QProcess, QThread
from PyQt5.QtGui import QTextCursor, QPalette, QColor
import hashlib
import math
import gc
import shlex
import psutil
from app_paths import get_app_data_path, migrate_legacy_file
import re
import time
import threading
try:
    import cv2
    # OpenCV wheels can export QT_* paths to cv2/qt/plugins, which conflicts
    # with PyQt5 platform plugins (xcb load failure on Linux).
    _cv2_qt_marker = f"{os.sep}cv2{os.sep}qt{os.sep}plugins"
    _qt_platform_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if _cv2_qt_marker in _qt_platform_path:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    _qt_plugin_path = os.environ.get("QT_PLUGIN_PATH", "")
    if _cv2_qt_marker in _qt_plugin_path:
        os.environ.pop("QT_PLUGIN_PATH", None)
except ImportError:
    cv2 = None
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def _process_frame_array_to_hist(frame_like, ignore_extremes):
    # Special handling for LazyFrames - get raw bitdepth data
    if isinstance(frame_like, LazyFrames):
        # This shouldn't happen directly, but handle it just in case
        frame_arr = frame_like.get_raw(0) if hasattr(frame_like, 'get_raw') else np.asarray(frame_like)
    else:
        frame_arr = np.asarray(frame_like)
    
    original_dtype = frame_arr.dtype
    
    # For uint16 or higher, keep original dtype; for others convert to uint8
    if original_dtype in (np.uint16, np.uint32, np.uint64, np.int16, np.int32, np.int64):
        a = frame_arr.ravel()
        num_bins = 256
        if original_dtype == np.uint16:
            num_bins = min(65536, int(frame_arr.max()) + 1)  # <-- CHANGED: Cap at actual max+1
        elif original_dtype in (np.uint32, np.int32):
            num_bins = min(65536, int(frame_arr.max()) + 1)  # Cap at 65536 for memory
        elif original_dtype in (np.uint64, np.int64):
            num_bins = min(65536, int(frame_arr.max()) + 1)  # Cap at 65536 for memory
        hist_vals = np.clip(a, 0, num_bins - 1).astype(np.int64, copy=False)
    else:
        a = np.asarray(frame_like, dtype=np.uint8).ravel()
        num_bins = 256
        hist_vals = a
    
    if a.size == 0:
        return np.zeros(num_bins, dtype=np.int64), 0.0, 0.0, 0, 255, 0

    # mean / variance accumulators
    sum_val = float(a.sum())
    sum_sq = float((a.astype(np.float64) ** 2).sum())
    count = int(a.size)
    min_val = int(a.min())
    max_val = int(a.max())

    if ignore_extremes:
        if num_bins == 256:
            mask = (a > 0) & (a < 255)
        else:
            # For high bitdepth, ignore 0 and max values
            mask = (a > 0) & (a < (1 << int(np.log2(num_bins))))
        # keep masked only if enough pixels remain otherwise use all
        if mask.sum() > max(100, a.size * 0.01):
            a_used = hist_vals[mask]
        else:
            a_used = hist_vals
    else:
        a_used = hist_vals

    # very fast histogram in C
    hist = np.bincount(a_used, minlength=num_bins).astype(np.int64)
    return hist, sum_val, sum_sq, count, min_val, max_val

def _compute_hist_for_key(args):
    key, frames, frame_mode, current_frame_index, start_frame, end_frame, ignore_extremes = args

    # accumulators - initialize with None to determine size from first frame
    hist_acc = None
    hist_size = 256  # default
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0
    gmin = None
    gmax = None

    if frame_mode == "Single":
        idxs = [current_frame_index]
    else:
        idxs = list(range(start_frame, end_frame + 1))

    for idx in idxs:
        try:
            if idx < 0 or idx >= len(frames):
                continue
            
            # Get frame - for LazyFrames, use get_raw() to get original bitdepth
            if isinstance(frames, LazyFrames):
                frame = frames.get_raw(idx)
            else:
                frame = frames[idx]
            
            hist, s, s2, cnt, mn, mx = _process_frame_array_to_hist(frame, ignore_extremes)
            
            # Initialize hist_acc with the size from first non-empty histogram
            if hist_acc is None:
                hist_size = len(hist)
                hist_acc = np.zeros(hist_size, dtype=np.int64)
            
            # Accumulate histogram - pad/truncate to match hist_size if needed
            if len(hist) > hist_size:
                hist_acc += hist[:hist_size]
            elif len(hist) < hist_size:
                hist_padded = np.pad(hist, (0, hist_size - len(hist)), mode='constant', constant_values=0)
                hist_acc += hist_padded
            else:
                hist_acc += hist
            
            total_sum += s
            total_sum_sq += s2
            total_count += cnt
            if gmin is None or mn < gmin:
                gmin = int(mn)
            if gmax is None or mx > gmax:
                gmax = int(mx)
        except Exception as e:
            # ignore single-frame read errors but continue
            print(f"Error processing frame {idx}: {e}")
            continue

    # Ensure hist_acc is initialized even if no frames were processed
    if hist_acc is None:
        hist_acc = np.zeros(256, dtype=np.int64)

    # return the key too (keeps ordering explicit)
    return (key, hist_acc, gmin if total_count > 0 else 0, gmax if total_count > 0 else 0,
            total_count, total_sum, total_sum_sq)

def meters_per_degree(lat_deg):
    phi = math.radians(lat_deg)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2*phi) + 1.175 * math.cos(4*phi) - 0.0023 * math.cos(6*phi)
    m_per_deg_lon = (111412.84 * math.cos(phi) - 93.5 * math.cos(3*phi) + 0.118 * math.cos(5*phi))
    if m_per_deg_lat == 0: m_per_deg_lat = 111132.0
    if m_per_deg_lon == 0: m_per_deg_lon = 111320.0 * math.cos(phi)
    return m_per_deg_lat, m_per_deg_lon

def image_coords_to_latlon(x_img, y_img, geo_info, bands_info=None, gap=0, orig_band_h=None, merge_lr=False):
    center_lat, center_lon, geo_w, geo_h, pixel_size_m = geo_info
    gap = int(gap or 0)

    if orig_band_h is not None:
        per_band_orig = int(orig_band_h)
    else:
        per_band_orig = int(geo_h)

    if not bands_info:
        per_block_h = per_band_orig
        block_index = int(y_img) // max(1, per_block_h + gap)
        block_index = max(0, block_index)
        band_index = int(block_index)
        center_x = float(geo_w) / 2.0
        center_y_of_block = block_index * (per_block_h + gap) + (per_block_h / 2.0)
        eff_pixel_m = float(pixel_size_m)
        x_offset = 0
        bin_factor = 1
    else:
        ordered_bases = sorted(bands_info.keys(), key=lambda k: bands_info[k]['index'])
        binned_bases = [b for b in ordered_bases if bands_info[b].get('binned', False)]
        unbinned_bases = [b for b in ordered_bases if not bands_info[b].get('binned', False)]
        stitch_sequence = []
        for base in binned_bases:
            info = bands_info[base]
            bin_factor = int(info.get('bin_factor', 1)) or 1
            per_block_h = max(1, int(round(float(orig_band_h) / float(bin_factor))))
            stitch_sequence.append({
                'base': base, 'kind': 'full_binned', 'per_block_h': per_block_h, 'bin_factor': bin_factor,
                'is_split': False, 'side': None
            })
        for base in unbinned_bases:
            info = bands_info[base]
            bin_factor = int(info.get('bin_factor', 1)) or 1
            per_block_h = int(orig_band_h)
            if info.get('split', False):
                if merge_lr:
                    stitch_sequence.append({
                        'base': base, 'kind': 'full_unbinned', 'per_block_h': per_block_h, 'bin_factor': bin_factor,
                        'is_split': True, 'side': None
                    })
                else:
                    stitch_sequence.append({
                        'base': base, 'kind': 'half_left', 'per_block_h': per_block_h, 'bin_factor': bin_factor,
                        'is_split': True, 'side': 'left'
                    })
                    stitch_sequence.append({
                        'base': base, 'kind': 'half_right', 'per_block_h': per_block_h, 'bin_factor': bin_factor,
                        'is_split': True, 'side': 'right'
                    })
            else:
                stitch_sequence.append({
                    'base': base, 'kind': 'full_unbinned', 'per_block_h': per_block_h, 'bin_factor': bin_factor,
                    'is_split': False, 'side': None
                })
        starts = []
        cur_y = 0
        for entry in stitch_sequence:
            starts.append((entry, cur_y))
            cur_y += int(entry['per_block_h']) + gap

        # find which block contains y_img
        found = None
        for block_idx, (entry, start_y) in enumerate(starts):
            h_eff = entry['per_block_h']
            if y_img >= start_y and y_img < (start_y + h_eff):
                found = (block_idx, entry, start_y, h_eff)
                break
        if found is None:
            # clamp to last block
            block_idx = max(0, len(starts) - 1)
            entry, start_y = starts[block_idx]
            h_eff = entry['per_block_h']
            found = (block_idx, entry, start_y, h_eff)

        block_index, entry, start_y, per_block_h_eff = found

        base_name = entry['base']
        band_index = int(bands_info.get(base_name, {}).get('index', block_index))

        if entry['is_split']:
            if merge_lr:
                x_offset = int(geo_w // 2) if float(x_img) >= (float(geo_w) / 2.0) else 0
            else:
                x_offset = int(geo_w // 2) if entry['side'] == 'right' else 0
        else:
            x_offset = 0

        bin_factor = int(entry.get('bin_factor', 1)) or 1
        eff_pixel_m = float(pixel_size_m) * max(1, bin_factor)

        center_x = float(geo_w) / 2.0

        center_y_of_block = float(start_y) + (float(per_block_h_eff) / 2.0)

    global_x = float(x_img) + float(x_offset)

    dx_m = (global_x - center_x) * float(eff_pixel_m)
    dy_m = (center_y_of_block - float(y_img)) * float(eff_pixel_m)

    phi = math.radians(center_lat)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2*phi) + 1.175 * math.cos(4*phi) - 0.0023 * math.cos(6*phi)
    m_per_deg_lon = (111412.84 * math.cos(phi) - 93.5 * math.cos(3*phi) + 0.118 * math.cos(5*phi))
    if m_per_deg_lat == 0:
        m_per_deg_lat = 111132.0
    if m_per_deg_lon == 0:
        m_per_deg_lon = 111320.0 * math.cos(phi)

    lat = float(center_lat) + (dy_m / m_per_deg_lat)
    lon = float(center_lon) + (dx_m / m_per_deg_lon)

    return float(lat), float(lon), int(band_index)

def check_memory_requirement(expected_bytes, parent=None):
    avail = psutil.virtual_memory().available
    if avail < expected_bytes:
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Memory Warning")
        msg.setText("Insufficient RAM available for this operation.")
        msg.setInformativeText(
            f"Required: {expected_bytes/1e9:.2f} GB\n"
            f"Available: {avail/1e9:.2f} GB\n\n"
            "Suggestion: Reduce frame range or close other programs."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return False
    return True

# ---------------- Parameter persistence helpers ----------------
PARAM_FILENAME = "parameters.json"
PARAM_DB_PATH = migrate_legacy_file(
    get_app_data_path("params.db"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.db")
)

def _init_db():
    conn = sqlite3.connect(PARAM_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS folder_settings (
                folder_path TEXT PRIMARY KEY,
                params_json TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()

_init_db()

def _get_db_conn():
    return sqlite3.connect(PARAM_DB_PATH)

import tempfile, fnmatch
from datetime import datetime

# Recent history storage
RECENT_FILE = migrate_legacy_file(
    get_app_data_path('recent.json'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recent.json')
)
_RECENT_LIMIT = 1000
VALID_TDI_STAGES = {0, 2, 4, 8, 16, 32}


def _param_hash(params):
    try:
        s = json.dumps(params, sort_keys=True)
        return hashlib.md5(s.encode('utf-8')).hexdigest()
    except Exception:
        return ""


def normalize_tdi_stage(value):
    try:
        stage = int(value)
    except Exception:
        return 0
    return stage if stage in VALID_TDI_STAGES else 0


def _bitdepth_bytes_per_frame(total_pixels, bit_depth):
    total_pixels = int(total_pixels)
    if total_pixels <= 0:
        return 0
    if bit_depth == 8:
        return total_pixels
    if bit_depth == 10:
        return (total_pixels * 10) // 8
    if bit_depth == 12:
        return (total_pixels * 12) // 8
    if bit_depth == 16:
        return total_pixels * 2
    if bit_depth == 32:
        return total_pixels * 4
    return 0


def _infer_frame_count_from_logs(log_text):
    if not log_text:
        return None
    patterns = [
        r"Expected:\s*(\d+)\s*\|\s*Captured:\s*(\d+)",
        r"No frame drops\s*[^\d]*(\d+)\s*/\s*(\d+)\s*captured",
        r"\bCaptured:\s*(\d+)\b",
        r"\bFrames:\s*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text, re.IGNORECASE)
        if not match:
            continue
        nums = [int(g) for g in match.groups() if g is not None]
        if not nums:
            continue
        if len(nums) >= 2:
            # Prefer the captured count when both expected and captured are present.
            return max(nums[-1], 0) or None
        return max(nums[0], 0) or None
    return None


def _infer_bit_depth_from_band_files(folder, width, effective_height, frame_count=None):
    try:
        width_i = int(width)
        height_i = int(effective_height)
    except Exception:
        return None
    if width_i <= 0 or height_i <= 0:
        return None

    candidate_scores = {8: 0, 10: 0, 12: 0, 16: 0}
    candidate_details = {8: None, 10: None, 12: None, 16: None}
    band_pattern = re.compile(r"\.band(\d)(\d?)$", re.IGNORECASE)

    try:
        names = sorted(os.listdir(folder))
    except Exception:
        return None

    for name in names:
        match = band_pattern.search(name)
        if not match:
            continue

        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            file_size = os.path.getsize(path)
        except Exception:
            continue
        if file_size <= 0:
            continue

        variant = match.group(2)
        frame_w = width_i
        frame_h = height_i
        if variant == "2":
            frame_w = max(1, width_i // 2)
            frame_h = max(1, height_i // 2)
        elif variant in {"0", "1"}:
            frame_w = max(1, width_i // 2)

        total_pixels = frame_w * frame_h
        if total_pixels <= 0:
            continue

        for bit_depth in (8, 10, 12, 16):
            bytes_per_frame = _bitdepth_bytes_per_frame(total_pixels, bit_depth)
            if bytes_per_frame <= 0:
                continue
            if frame_count:
                if file_size == bytes_per_frame * int(frame_count):
                    candidate_scores[bit_depth] += 3
                    if candidate_details[bit_depth] is None:
                        candidate_details[bit_depth] = {
                            "file": name,
                            "file_size": file_size,
                            "frame_w": frame_w,
                            "frame_h": frame_h,
                            "bytes_per_frame": bytes_per_frame,
                            "frame_count": int(frame_count),
                            "variant": variant or "full",
                        }
            elif file_size % bytes_per_frame == 0:
                candidate_scores[bit_depth] += 1
                if candidate_details[bit_depth] is None:
                    candidate_details[bit_depth] = {
                        "file": name,
                        "file_size": file_size,
                        "frame_w": frame_w,
                        "frame_h": frame_h,
                        "bytes_per_frame": bytes_per_frame,
                        "frame_count": file_size // bytes_per_frame,
                        "variant": variant or "full",
                    }

    best_bit_depth = None
    best_score = 0
    for bit_depth in (16, 12, 10, 8):
        score = candidate_scores.get(bit_depth, 0)
        if score > best_score:
            best_score = score
            best_bit_depth = bit_depth
    if best_score > 0 and best_bit_depth is not None:
        detail = candidate_details.get(best_bit_depth) or {}
        print(
            "[AutoFill] Detected "
            f"{best_bit_depth}-bit depth from {detail.get('file', 'unknown file')}: "
            f"file_size={detail.get('file_size', 0)} bytes, "
            f"calculation=({detail.get('frame_h', 0)} x {detail.get('frame_w', 0)} x {best_bit_depth}/8) "
            f"x {detail.get('frame_count', 0)} frames = "
            f"{detail.get('bytes_per_frame', 0)} x {detail.get('frame_count', 0)} = "
            f"{detail.get('bytes_per_frame', 0) * detail.get('frame_count', 0)} bytes"
        )
    return best_bit_depth if best_score > 0 else None


def infer_dataset_image_params(folder):
    """
    Infer width, RegionHeight, and TDI stages from dataset JSON/log files.
    """
    folder = os.path.abspath(folder)
    inferred = {}
    log_text_parts = []

    def _set_if_valid(key, value):
        try:
            ivalue = int(float(value))
        except Exception:
            return
        if ivalue > 0:
            inferred[key] = ivalue

    def _consume_mapping(obj):
        if not isinstance(obj, dict):
            return
        if "Width" in obj:
            _set_if_valid("width", obj.get("Width"))
        if "RegionHeight" in obj:
            _set_if_valid("raw_height", obj.get("RegionHeight"))
        if "TDIStages" in obj:
            inferred["tdi_stage"] = normalize_tdi_stage(obj.get("TDIStages"))
        elif "TDI_Stages" in obj:
            inferred["tdi_stage"] = normalize_tdi_stage(obj.get("TDI_Stages"))
        if "BandHeight" in obj:
            _set_if_valid("effective_height", obj.get("BandHeight"))

    try:
        names = os.listdir(folder)
    except Exception:
        return {}

    for name in sorted(names):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        lower_name = name.lower()
        if lower_name.endswith(".json") and lower_name != "parameters.json":
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    payload = json.load(f)
                _consume_mapping(payload)
            except Exception:
                pass
        elif lower_name.endswith(".log"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    log_text_parts.append(f.read())
            except Exception:
                pass

    log_text = "\n".join(log_text_parts)
    frame_count = _infer_frame_count_from_logs(log_text)
    if log_text:
        patterns = [
            ("width", r"Applied Width\s*=\s*(\d+)"),
            ("raw_height", r"Applied RegionHeight\s*=\s*(\d+)"),
            ("tdi_stage", r"Applied TDI_Stages\s*=\s*(\d+)"),
            ("effective_height", r"Applied Height\s*=\s*(\d+)"),
            ("effective_height", r"Computed BAND_HEIGHT\s*=\s*(\d+)"),
        ]
        for key, pattern in patterns:
            match = re.search(pattern, log_text, re.IGNORECASE)
            if not match:
                continue
            if key == "tdi_stage":
                inferred[key] = normalize_tdi_stage(match.group(1))
            else:
                _set_if_valid(key, match.group(1))

    raw_height = inferred.get("raw_height")
    tdi_stage = normalize_tdi_stage(inferred.get("tdi_stage", 0))
    if raw_height:
        inferred["tdi_stage"] = tdi_stage
        inferred["effective_height"] = raw_height if tdi_stage == 0 else max(1, raw_height // tdi_stage)
        bit_depth = _infer_bit_depth_from_band_files(
            folder,
            inferred.get("width", 0),
            inferred.get("effective_height", 0),
            frame_count=frame_count,
        )
        if bit_depth:
            inferred["bit_depth"] = bit_depth

    return inferred


def load_recents():
    try:
        if os.path.exists(RECENT_FILE):
            with open(RECENT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        # non-fatal
        print(f"Warning: could not load recent file: {e}")
    return []


def save_recents(lst):
    try:
        with open(RECENT_FILE, 'w', encoding='utf-8') as f:
            json.dump(lst, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save recent file: {e}")


def add_recent(path, mode, params=None):
    """Add or update a recent entry (most-recent first)."""
    try:
        lst = load_recents()
        # remove duplicates
        lst = [r for r in lst if not (r.get('path') == path and r.get('mode') == mode)]
        entry = {
            'path': path,
            'mode': mode,
            'last_opened': datetime.utcnow().isoformat(),
            'meta': {
                'param_hash': _param_hash(params or {}),
                'params': params or {}
            }
        }
        lst.insert(0, entry)
        if len(lst) > _RECENT_LIMIT:
            lst = lst[:_RECENT_LIMIT]
        save_recents(lst)
    except Exception as e:
        print(f"Warning: add_recent failed: {e}")


def get_recents_for_mode(mode, limit=None):
    try:
        lst = load_recents()
        filtered = [r for r in lst if r.get('mode') == mode]
        if limit is not None:
            return filtered[:limit]
        return filtered
    except Exception:
        return []


def remove_recent(path, mode=None):
    """Remove a recent entry by path, optionally constrained to mode."""
    try:
        lst = load_recents()
        filtered = []
        for r in lst:
            same_path = (r.get('path') == path)
            same_mode = (mode is None or r.get('mode') == mode)
            if same_path and same_mode:
                continue
            filtered.append(r)
        if len(filtered) != len(lst):
            save_recents(filtered)
            return True
    except Exception as e:
        print(f"Warning: remove_recent failed: {e}")
    return False


def select_from_history(parent, mode=None):
    """Show a modal dialog allowing the user to pick a recent entry. Returns the selected path or None."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("History")
    v = QVBoxLayout(dlg)
    lbl = QLabel(f"Recent items{' ('+mode+')' if mode else ''}")
    v.addWidget(lbl)
    listw = QListWidget()
    recs = load_recents()
    if mode:
        recs = [r for r in recs if r.get('mode') == mode]

    def refresh_list(select_row=0):
        listw.clear()
        for r in recs:
            ts = r.get('last_opened', '')
            display = f"{os.path.basename(r.get('path',''))} — {ts}"
            listw.addItem(display)
        if recs:
            select_row = max(0, min(select_row, len(recs) - 1))
            listw.setCurrentRow(select_row)

    refresh_list()
    v.addWidget(listw)
    h = QHBoxLayout()
    delete_btn = QPushButton("×")
    delete_btn.setToolTip("Delete selected item from recent history")
    delete_btn.setFixedWidth(32)
    open_btn = QPushButton("Open")
    open_btn.setToolTip("Open selected item")
    cancel_btn = QPushButton("Cancel")
    cancel_btn.setToolTip("Close history")
    h.addWidget(delete_btn)
    h.addStretch()
    h.addWidget(open_btn)
    h.addWidget(cancel_btn)
    v.addLayout(h)

    def on_open():
        idx = listw.currentRow()
        if idx >= 0 and idx < len(recs):
            dlg.done(1)
        else:
            dlg.done(0)

    def on_delete():
        nonlocal recs
        idx = listw.currentRow()
        if idx < 0 or idx >= len(recs):
            return
        rec = recs[idx]
        if remove_recent(rec.get('path'), rec.get('mode')):
            recs.pop(idx)
            refresh_list(select_row=idx)

    open_btn.clicked.connect(on_open)
    delete_btn.clicked.connect(on_delete)
    cancel_btn.clicked.connect(lambda: dlg.done(0))
    listw.itemDoubleClicked.connect(lambda *_: on_open())

    res = dlg.exec_()
    if res == 1:
        idx = listw.currentRow()
        if idx >= 0 and idx < len(recs):
            return recs[idx]['path']
    return None



def atomic_write_json(path, data):
    dirp = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(dir=dirp)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def load_folder_params(folder):
    # normalize folder path
    folder = os.path.abspath(folder).replace('\\', '/')
    
    # 1. Check for legacy parameters.json
    p = os.path.join(folder, PARAM_FILENAME)
    legacy_data = None
    if os.path.isfile(p):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                legacy_data = json.load(f)
        except Exception:
            pass

    # 2. Fetch from SQLite
    db_data = None
    try:
        with _get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT params_json FROM folder_settings WHERE folder_path = ?", (folder,))
            row = cur.fetchone()
            if row:
                db_data = json.loads(row[0])
    except Exception as e:
        print(f"Warning: load_folder_params DB error: {e}")

    # 3. Migration logic
    if legacy_data:
        # If we have legacy data and no DB data or they differ, migrate to DB
        if not db_data or _param_hash(legacy_data) != _param_hash(db_data):
            save_params_for_path(folder, legacy_data, is_migration=True)
            db_data = legacy_data
        
        # Always delete the old JSON once seen
        try:
            os.remove(p)
            print(f"Migrated and deleted legacy {PARAM_FILENAME} for {folder}")
        except Exception as e:
            print(f"Warning: could not delete legacy {PARAM_FILENAME}: {e}")

    return db_data


def get_saved_params_for_file(file_path):
    """Return saved params for a file or folder default if present."""
    # If folder path passed, return folder default
    if os.path.isdir(file_path):
        data = load_folder_params(file_path)
        if data:
            return data.get('default')
        return None

    folder = os.path.dirname(os.path.abspath(file_path))
    data = load_folder_params(folder)
    if not data:
        return None
    rel = os.path.relpath(file_path, folder).replace('\\', '/')
    files = data.get('files', {})

    # Exact filename match
    if rel in files:
        return files[rel]

    # Pattern matches (most specific first)
    keys = sorted(files.keys(), key=lambda k: (-len(k), k))
    for k in keys:
        if any(c in k for c in '*?[]'):
            if fnmatch.fnmatch(rel, k):
                return files[k]

    # Fallback to folder default
    return data.get('default')


def save_params_for_path(path, params, as_default=False, pattern=None, is_migration=False):
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    folder = os.path.abspath(folder).replace('\\', '/')
    
    # Load existing data from DB
    data = {}
    try:
        with _get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT params_json FROM folder_settings WHERE folder_path = ?", (folder,))
            row = cur.fetchone()
            if row:
                data = json.loads(row[0])
    except Exception:
        pass

    if is_migration:
        data = params
    elif as_default:
        data['default'] = params
    else:
        files = data.setdefault('files', {})
        if pattern:
            files[pattern] = params
        else:
            if os.path.isdir(path):
                data['default'] = params
            else:
                rel = os.path.relpath(path, folder).replace('\\', '/')
                files[rel] = params
    
    # Save to SQLite
    try:
        with _get_db_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO folder_settings (folder_path, params_json) VALUES (?, ?)",
                (folder, json.dumps(data))
            )
            conn.commit()
    except Exception as e:
        print(f"Warning: Failed to save parameters to central DB for {folder}: {e}")


# ---------- Bit Depth Handling ----------
def unpack_8bit(data, w, h):
    try:
        # Support bytes or numpy uint8 view
        if isinstance(data, (bytes, bytearray)):
            arr = np.frombuffer(data, dtype=np.uint8)
        else:
            arr = np.asarray(data, dtype=np.uint8).ravel()
        total = w * h
        if arr.size % total != 0:
            # Truncate incomplete tail
            arr = arr[:(arr.size // total) * total]
        if arr.size == 0:
            return []
        return arr.reshape((-1, h, w))
    except Exception as e:
        print(f"Error in unpack_8bit: {e}")
        return []

def _unpack_10bit_raw(data, w, h):
    """Unpack 10-bit data and return raw uint16 values (0-1023)."""
    total_pixels = w * h
    unpacked = np.zeros(total_pixels, dtype=np.uint16)
    
    num_full_groups = len(data) // 5
    d = np.frombuffer(data[:num_full_groups*5], dtype=np.uint8).reshape(-1,5)
    expanded_data = np.zeros(d.shape[0], dtype=np.uint64)
    for j in range(5):
        expanded_data += d[:,j].astype(np.uint64) << (8 * j)
    for j in range(4):
        unpacked[j::4] = (expanded_data >> (10 * j)) & 0x3FF
    
    remaining_bytes = len(data) % 5
    if remaining_bytes:
        last_bits = int.from_bytes(data[-remaining_bytes:], 'little')
        extra_pixels = np.array([(last_bits >> (10 * k)) & 0x3FF for k in range((remaining_bytes * 8) // 10)], dtype=np.uint16)
        unpacked[num_full_groups*4:num_full_groups*4+len(extra_pixels)] = extra_pixels[:total_pixels - num_full_groups*4]
    
    return unpacked.reshape((h, w))


def unpack_10bit(packed_bytes, w, h):
    """Unpack a single-frame 10-bit packed buffer into a uint16 array shaped (h, w)."""
    # Accept numpy array or bytes
    if not isinstance(packed_bytes, (bytes, bytearray)):
        try:
            packed_bytes = packed_bytes.tobytes()
        except Exception:
            packed_bytes = bytes(packed_bytes)
    return _unpack_10bit_raw(packed_bytes, w, h)

def _unpack_12bit_raw(data, w, h):
    """Unpack 12-bit data and return raw uint16 values (0-4095)."""
    total_pixels = w * h
    d = np.frombuffer(data, dtype=np.uint8)
    d = d[:(len(d) // 3) * 3].reshape(-1, 3)
    px0 = d[:, 0].astype(np.uint16) + ((d[:, 1].astype(np.uint16) & 0x0F) << 8)
    px1 = ((d[:, 1].astype(np.uint16) >> 4) & 0x0F) + (d[:, 2].astype(np.uint16) << 4)
    frame = np.empty((total_pixels,), dtype=np.uint16)
    frame[0::2] = px0
    frame[1::2] = px1
    return frame[:total_pixels].reshape((h, w))


def unpack_12bit(packed_bytes, w, h):
    """Unpack a single-frame 12-bit packed buffer into a uint16 array shaped (h, w)."""
    if not isinstance(packed_bytes, (bytes, bytearray)):
        try:
            packed_bytes = packed_bytes.tobytes()
        except Exception:
            packed_bytes = bytes(packed_bytes)
    return _unpack_12bit_raw(packed_bytes, w, h)


def unpack_by_bitdepth(data, w, h, bitdepth, return_raw=False):
    """Unpack data by bitdepth. If return_raw=True, return raw bitdepth data instead of linearly mapped uint8."""
    if bitdepth == 32:
        total_pixels = w * h
        if len(data) < total_pixels * 4:
            return []

        arr = np.frombuffer(data, dtype='<u4', count=total_pixels).reshape((h, w))
        if return_raw:
            return [arr]

        scaled = np.clip(
            arr.astype(np.float64) * (255.0 / 4294967295.0),
            0, 255
        ).astype(np.uint8)
        return [scaled]

    if bitdepth == 16:
        total_pixels = w * h
        if len(data) < total_pixels * 2:
            return []

        # RAW16 is little-endian
        arr = np.frombuffer(data, dtype='<u2', count=total_pixels)
        arr = arr.reshape((h, w))

        if return_raw:
            return [arr]

        # Detect actual bit depth for proper scaling
        max_val = arr.max()
        if max_val <= 1023:
            scale_factor = 255.0 / 1023.0
        elif max_val <= 4095:
            scale_factor = 255.0 / 4095.0
        else:
            scale_factor = 255.0 / 65535.0
        scaled = np.clip((arr.astype(np.float32) * scale_factor), 0, 255).astype(np.uint8)
        return [scaled]

    if bitdepth == 8:
        return unpack_8bit(data, w, h)
    elif bitdepth == 10:
        total_pixels = w * h
        bytes_per_frame = (total_pixels * 10) // 8
        num_frames = len(data) // bytes_per_frame
        frames = []
        for i in range(num_frames):
            start = i * bytes_per_frame
            packed_data = data[start : start + bytes_per_frame]
            raw = _unpack_10bit_raw(packed_data if isinstance(packed_data, (bytes, bytearray)) else packed_data.tobytes(), w, h)
            if return_raw:
                frames.append(raw)
            else:
                scaled = np.clip((raw.astype(np.float32) * 255.0 / 1023), 0, 255).astype(np.uint8)
                frames.append(scaled)
        return frames
    elif bitdepth == 12:
        total_pixels = w * h
        frame_size = (total_pixels * 12) // 8
        num_frames = len(data) // frame_size
        frames = []
        for i in range(num_frames):
            chunk = data[i * frame_size:(i + 1) * frame_size]
            raw = _unpack_12bit_raw(chunk if isinstance(chunk, (bytes, bytearray)) else chunk.tobytes(), w, h)
            if return_raw:
                frames.append(raw)
            else:
                scaled = np.clip((raw.astype(np.float32) * 255.0 / 4095.0), 0, 255).astype(np.uint8)
                frames.append(scaled)
        return frames
    else:
        raise ValueError(f"Unsupported bit depth: {bitdepth}")
    

class LazyFrames:
    def __init__(self, file_path, w, h, bitdepth, use_memmap=False):
        self.file_path = file_path
        self.w = w
        self.h = h
        self.bitdepth = bitdepth
        total_pixels = w * h
        if bitdepth == 8:
            self.bytes_per_frame = total_pixels
        elif bitdepth == 10:
            self.bytes_per_frame = (total_pixels * 10) // 8
        elif bitdepth == 12:
            self.bytes_per_frame = (total_pixels * 12) // 8
        elif bitdepth == 16:
            self.bytes_per_frame = total_pixels * 2
        elif bitdepth == 32:
            self.bytes_per_frame = total_pixels * 4
        else:
            raise ValueError(f"Unsupported bit depth: {bitdepth}")
        # Use numpy memmap for lazy access only if explicitly requested
        self.mem = None
        if use_memmap:
            try:
                self.mem = np.memmap(file_path, dtype=np.uint8, mode='r')
                file_size = self.mem.size
            except Exception:
                # Fallback to os.path.getsize if memmap fails
                file_size = os.path.getsize(file_path)
        else:
            file_size = os.path.getsize(file_path)

        self.num_frames = file_size // self.bytes_per_frame if self.bytes_per_frame > 0 else 0

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_frames:
            raise IndexError("Frame index out of range")
        start = idx * self.bytes_per_frame
        # Read only the frame slice from memmap or file
        if self.mem is not None:
            # memmap slice returns numpy uint8 view; convert to bytes for unpackers
            slice_view = self.mem[start:start + self.bytes_per_frame]
            chunk = slice_view.tobytes()
        else:
            with open(self.file_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(self.bytes_per_frame)
            if len(chunk) < self.bytes_per_frame:
                chunk += bytes(self.bytes_per_frame - len(chunk))

        frames = unpack_by_bitdepth(chunk, self.w, self.h, self.bitdepth, return_raw=False)
        if frames is None:
            return np.zeros((self.h, self.w), dtype=np.uint8)
        if isinstance(frames, np.ndarray):
            if frames.size == 0:
                return np.zeros((self.h, self.w), dtype=np.uint8)
            if frames.ndim == 2:
                return frames
            return frames[0]
        if len(frames) == 0:
            return np.zeros((self.h, self.w), dtype=np.uint8)
        return frames[0]
    
    def get_raw(self, idx):
        """Get raw bitdepth frame for histogram computation."""
        if idx < 0 or idx >= self.num_frames:
            raise IndexError("Frame index out of range")
        start = idx * self.bytes_per_frame
        if self.mem is not None:
            slice_view = self.mem[start:start + self.bytes_per_frame]
            chunk = slice_view.tobytes()
        else:
            with open(self.file_path, 'rb') as f:
                f.seek(start)
                chunk = f.read(self.bytes_per_frame)
            if len(chunk) < self.bytes_per_frame:
                chunk += bytes(self.bytes_per_frame - len(chunk))

        frames = unpack_by_bitdepth(chunk, self.w, self.h, self.bitdepth, return_raw=True)
        if frames is None:
            return np.zeros((self.h, self.w), dtype=np.uint16)
        if isinstance(frames, np.ndarray):
            if frames.size == 0:
                return np.zeros((self.h, self.w), dtype=np.uint16)
            if frames.ndim == 2:
                return frames
            return frames[0]
        if len(frames) == 0:
            return np.zeros((self.h, self.w), dtype=np.uint16)
        return frames[0]
        
class TerminalWidget(QWidget):
    output_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Prompt identity
        self.user = getpass.getuser()
        self.host = socket.gethostname().split('.')[0]
        self.cwd = os.getcwd()

        # Layout
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Single QTextEdit for output and input
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(False)  # Allow editing
        self.terminal.setAcceptRichText(False)
        try:
            f = self.terminal.font()
            f.setFamily("Courier New")
            f.setPointSize(10)
            self.terminal.setFont(f)
        except Exception:
            pass
        root.addWidget(self.terminal, 1)

        # Install event filter for key handling
        self.terminal.installEventFilter(self)

        # History
        self._history = []
        self._hist_idx = None
        self.history_file = get_app_data_path('terminal_history.json')
        self.load_history()
        self._history_limit = 500
        self._current_input = ""  # Buffer for current line input

        # Default shell
        sys_platform = platform.system().lower()
        if 'windows' in sys_platform:
            self._default_shell = shutil.which("powershell.exe") or shutil.which("cmd.exe") or "cmd.exe"
            self._use_cmd = True
        else:
            self._default_shell = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
            self._use_cmd = False

        # Start closed
        self.setMaximumHeight(0)
        self.hide()

        self.output_signal.connect(self._append_text)
        self._append_text(f"[Terminal started: shell={self._default_shell} cwd={self.cwd}]\n")
        self._insert_prompt()

    def _insert_prompt(self):
        prompt = self._build_prompt() + " "
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(prompt)
        self.terminal.moveCursor(QTextCursor.End)

    def _build_prompt(self):
        home = os.path.expanduser("~")
        if self.cwd == home:
            cwd_display = "~"
        elif self.cwd.startswith(home + os.sep):
            cwd_display = "~" + self.cwd[len(home):]
        else:
            cwd_display = self.cwd
        return f"{self.user}@{self.host}:{cwd_display}$"

    def _append_text(self, txt):
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(txt)
        self.terminal.moveCursor(QTextCursor.End)

    def eventFilter(self, obj, event):
        if obj == self.terminal and event.type() == QEvent.KeyPress:
            key = event.key()
            mod = event.modifiers()

            if key == Qt.Key_Return or key == Qt.Key_Enter:
                # Get the current line after prompt
                cursor = self.terminal.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.select(QTextCursor.LineUnderCursor)
                line = cursor.selectedText()
                prompt_len = len(self._build_prompt() + " ")
                cmd = line[prompt_len:]  # No strip() to preserve trailing spaces if needed

                # Add newline after command
                self._append_text("\n")

                # Update history
                cmd_stripped = cmd.strip()
                if cmd_stripped:
                    self.add_to_history(cmd_stripped)
                    if not (self._history and self._history[-1] == cmd_stripped):
                        self._history.append(cmd_stripped)
                        if len(self._history) > self._history_limit:
                            self._history.pop(0)
                self._hist_idx = None
                self._current_input = ""

                # Builtins
                parts = shlex.split(cmd)
                if parts and parts[0] == "cd":
                    if len(parts) == 1 or parts[1] == "~":
                        target = os.path.expanduser("~")
                    else:
                        target = os.path.expanduser(parts[1])
                        if not os.path.isabs(target):
                            target = os.path.normpath(os.path.join(self.cwd, target))
                    try:
                        os.chdir(target)
                        self.cwd = os.getcwd()
                    except Exception as e:
                        self._append_text(f"cd: {e}\n")
                    self._insert_prompt()
                    return True

                if cmd_stripped in ("clear", "cls"):
                    self.terminal.clear()
                    self._insert_prompt()
                    return True
                    
                # Run external command (async, prompt after finish)
                self._run_command(cmd)
                return True

            elif key == Qt.Key_Up:
                if self._hist_idx is None:
                    self._hist_idx = len(self._history)
                if self._hist_idx > 0:
                    self._hist_idx -= 1
                    self._replace_current_input(self._history[self._hist_idx])
                return True

            elif key == Qt.Key_Down:
                if self._hist_idx is not None and self._hist_idx < len(self._history) - 1:
                    self._hist_idx += 1
                    self._replace_current_input(self._history[self._hist_idx])
                elif self._hist_idx is not None:
                    self._hist_idx = None
                    self._replace_current_input("")
                return True
                
            elif key == Qt.Key_Tab:
                self._do_completion()
                return True
                
            elif key == Qt.Key_L and mod == Qt.ControlModifier:
                self.terminal.clear()
                self._insert_prompt()
                return True

            elif key == Qt.Key_Backspace:
                # Prevent deleting before prompt
                cursor = self.terminal.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.select(QTextCursor.LineUnderCursor)
                line = cursor.selectedText()
                prompt_len = len(self._build_prompt() + " ")
                if len(line) <= prompt_len:
                    return True  # Block backspace

            # Allow other keys to type normally
            return False  # Explicitly return False for unhandled key events

        # Handle all other events by passing to parent class, ensuring boolean return
        return super().eventFilter(obj, event)
            
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self._history = json.load(f)
            except PermissionError:
                print(f"Permission denied loading history: {self.history_file}")
            except Exception as e:
                print(f"Error loading history: {e}")

    def add_to_history(self, cmd):
        try:
            self._history.append(cmd)
            with open(self.history_file, 'w') as f:
                json.dump(self._history, f)
        except PermissionError:
            print(f"Permission denied saving history: {self.history_file}")
        except Exception as e:
            print(f"Error saving history: {e}")

    def _do_completion(self):
        cursor = self.terminal.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.LineUnderCursor)
        line = cursor.selectedText()
        prompt_len = len(self._build_prompt() + " ")
        text = line[prompt_len:]

        if not text.strip():
            return

        parts = text.split()
        prefix = parts[-1]

        # If it's the first word, complete commands
        if len(parts) == 1:
            paths = []
            for p in os.environ.get("PATH", "").split(os.pathsep):
                if os.path.isdir(p):
                    paths.extend(glob.glob(os.path.join(p, prefix) + "*"))
            matches = [os.path.basename(m) for m in paths if os.access(m, os.X_OK)]
        else:
            # File/directory completion
            matches = glob.glob(prefix + "*")

        if len(matches) == 1:
            parts[-1] = matches[0]
            new_line = " ".join(parts)
            self._replace_current_input(new_line)
        elif len(matches) > 1:
            self._append_text("\n" + "  ".join(matches) + "\n")
            self._insert_prompt()
            self._replace_current_input(text)

    def _replace_current_input(self, text):
        cursor = self.terminal.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.LineUnderCursor)
        line = cursor.selectedText()
        prompt = line[:len(self._build_prompt() + " ")]
        cursor.removeSelectedText()
        cursor.insertText(prompt + text)
        cursor.movePosition(QTextCursor.End)
        self.terminal.setTextCursor(cursor)
        self._current_input = text

    def _run_command(self, cmd):
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setWorkingDirectory(self.cwd)

        def on_ready_read():
            try:
                data = proc.readAll().data().decode(errors="ignore")
            except Exception:
                data = ""
            if data:
                self.output_signal.emit(data)

        def on_finished(exit_code, exit_status):
            # Read any remaining data that might not have triggered readyRead
            try:
                remaining_data = proc.readAll().data().decode(errors="ignore")
            except Exception:
                remaining_data = ""
            if remaining_data:
                self.output_signal.emit(remaining_data)
            
            try:
                proc.readyRead.disconnect(on_ready_read)
            except Exception:
                pass
            try:
                proc.finished.disconnect(on_finished)
            except Exception:
                pass
            proc.deleteLater()
            self._insert_prompt()

        proc.readyRead.connect(on_ready_read)
        proc.finished.connect(on_finished)

        if isinstance(self._default_shell, str) and 'cmd.exe' in self._default_shell.lower():
            proc.start(self._default_shell, ["/C", cmd])
        else:
            proc.start(self._default_shell, ["-c", cmd])

    def focus_input(self):
        self.terminal.setFocus()
        self.terminal.moveCursor(QTextCursor.End)
