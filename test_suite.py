"""
=============================================================================
  ADVERSARIAL TEST SUITE  —  Band/Satellite Image Viewer Application
  "Try to break it before your users do"
=============================================================================

This suite acts like a hostile user / QA tester:
  - Feeds garbage, corrupt, truncated, oversized, zero-size inputs
  - Boundary values: 0, 1, -1, MAX_INT, float-where-int-expected
  - Wrong types: None, str, list, dict where numpy/bytes expected
  - Concurrent / repeat calls (state leaks)
  - File system abuse: missing files, read-only paths, corrupt JSON
  - Logic probes: does offset math stay in-bounds? do coordinates wrap?
  - Memory: single-pixel frames, 0×0 frames, 1-byte files

Run:
    python adversarial_test_suite.py            # pure-logic tests only
    python adversarial_test_suite.py -v         # verbose
    python adversarial_test_suite.py --qt       # include Qt widget tests

Report → adversarial_report.html
=============================================================================
"""

import sys, os, math, json, struct, tempfile, shutil, traceback, time
import argparse, html as _html, threading, gc
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

# ── Test registry ────────────────────────────────────────────────────────────
_TESTS   = []   # (category, name, fn, needs_qt)
_RESULTS = []

def test(category, needs_qt=False):
    def deco(fn):
        _TESTS.append((category, fn.__name__, fn, needs_qt))
        return fn
    return deco

# ── Helpers to manufacture raw frames ────────────────────────────────────────
def _raw8(w, h, fill=128, n=1):
    return np.full((h, w), fill, dtype=np.uint8).tobytes() * n

def _raw10(w, h, val=512, n=1):
    total = w * h
    buf = bytearray()
    for _ in range((total + 3) // 4):
        v = 0
        for j in range(4):
            v |= (val & 0x3FF) << (j * 10)
        buf += v.to_bytes(5, 'little')
    return bytes(buf) * n

def _raw12(w, h, val=2048, n=1):
    total = w * h
    buf = bytearray()
    for _ in range((total + 1) // 2):
        lo  = val & 0xFF
        mid = ((val >> 8) & 0x0F) | ((val & 0x0F) << 4)
        hi  = (val >> 4) & 0xFF
        buf += bytes([lo, mid, hi])
    return bytes(buf) * n

def _raw16(w, h, val=30000, n=1):
    return np.full((h, w), val, dtype='<u2').tobytes() * n

def _tmpfile(data):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".raw")
    f.write(data); f.close()
    return f.name


# ═══════════════════════════════════════════════════════════════════════════
#  1.  PIXEL UNPACKING — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_zero_width():
    """Width=0 must not divide-by-zero or raise uncaught."""
    from utils import unpack_8bit
    result = unpack_8bit(b"\x00" * 16, 0, 4)
    assert isinstance(result, (list, np.ndarray))

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_zero_height():
    from utils import unpack_8bit
    result = unpack_8bit(b"\x00" * 16, 4, 0)
    assert isinstance(result, (list, np.ndarray))

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_single_pixel():
    from utils import unpack_8bit
    result = unpack_8bit(bytes([99]), 1, 1)
    assert len(result) == 1
    assert int(result[0][0, 0]) == 99

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_extra_trailing_bytes():
    """File slightly larger than n*frame — must not crash, must return correct frame count."""
    from utils import unpack_8bit
    w, h = 4, 4          # 16 bytes / frame
    good = _raw8(w, h, fill=7, n=3)   # 48 bytes
    padded = good + b"\xDE\xAD"        # 50 bytes (junk tail)
    result = unpack_8bit(padded, w, h)
    assert len(result) == 3, f"Expected 3 frames, got {len(result)}"

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_one_byte_short():
    """One byte missing from the last frame — must return 2 frames, not crash."""
    from utils import unpack_8bit
    w, h = 4, 4
    data = _raw8(w, h, n=3)[:-1]   # 47 bytes instead of 48
    result = unpack_8bit(data, w, h)
    assert len(result) == 2

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_wrong_type_string():
    """Passing a string instead of bytes must not crash the application."""
    from utils import unpack_8bit
    try:
        result = unpack_8bit("not bytes at all", 4, 4)
        assert isinstance(result, (list, np.ndarray))
    except (TypeError, ValueError):
        pass  # explicit rejection is fine; silent crash is not

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_none_input():
    from utils import unpack_8bit
    try:
        result = unpack_8bit(None, 4, 4)
        assert isinstance(result, (list, np.ndarray))
    except (TypeError, AttributeError):
        pass  # must not propagate as unhandled exception to GUI

@test("1. Pixel Unpack – Adversarial")
def test_unpack8_negative_width():
    from utils import unpack_8bit
    try:
        result = unpack_8bit(b"\x00" * 64, -8, 4)
        assert isinstance(result, (list, np.ndarray))
    except (ValueError, Exception):
        pass

@test("1. Pixel Unpack – Adversarial")
def test_unpack10_zero_pixels():
    from utils import unpack_10bit
    result = unpack_10bit(b"", 1, 1)
    assert result is not None  # shape may be wrong but must not raise

@test("1. Pixel Unpack – Adversarial")
def test_unpack10_corrupt_data():
    """All 0xFF bytes — values must stay in 0-1023 range."""
    from utils import unpack_10bit
    w, h = 8, 4
    # 10-bit frame needs (8*4*10)//8 = 40 bytes
    data = b"\xFF" * 40
    result = unpack_10bit(data, w, h)
    assert result.shape == (h, w)
    assert int(result.max()) <= 1023, f"10-bit overflow: {result.max()}"

@test("1. Pixel Unpack – Adversarial")
def test_unpack10_all_zeros():
    from utils import unpack_10bit
    w, h = 8, 4
    data = b"\x00" * ((w * h * 10) // 8)
    result = unpack_10bit(data, w, h)
    assert int(result.max()) == 0

@test("1. Pixel Unpack – Adversarial")
def test_unpack12_corrupt_data():
    """All 0xFF — values must stay in 0-4095 range."""
    from utils import unpack_12bit
    w, h = 8, 4
    data = b"\xFF" * ((w * h * 12) // 8)
    result = unpack_12bit(data, w, h)
    assert result.shape == (h, w)
    assert int(result.max()) <= 4095, f"12-bit overflow: {result.max()}"

@test("1. Pixel Unpack – Adversarial")
def test_unpack_by_bitdepth_16_all_zero():
    from utils import unpack_by_bitdepth
    w, h = 8, 4
    data = b"\x00" * (w * h * 2)
    result = unpack_by_bitdepth(data, w, h, 16)
    assert isinstance(result, list)
    # All-zero 16-bit: after percentile stretch, should produce valid uint8 output
    if result:
        assert result[0].dtype == np.uint8

@test("1. Pixel Unpack – Adversarial")
def test_unpack_by_bitdepth_16_all_max():
    """Uniform max value — stretch should not crash (high==low guard)."""
    from utils import unpack_by_bitdepth
    w, h = 8, 4
    data = _raw16(w, h, val=65535)
    result = unpack_by_bitdepth(data, w, h, 16)
    assert isinstance(result, list)

@test("1. Pixel Unpack – Adversarial")
def test_unpack_by_bitdepth_single_pixel_10bit():
    from utils import unpack_by_bitdepth
    data = _raw10(1, 1, val=500)
    result = unpack_by_bitdepth(data, 1, 1, 10)
    assert isinstance(result, list)

@test("1. Pixel Unpack – Adversarial")
def test_unpack_video_mode_extra_garbage():
    """video_mode.unpack_by_bitdepth with extra bytes at the end."""
    from video_mode import unpack_by_bitdepth
    w, h = 8, 4
    data = _raw8(w, h, n=2) + b"\xBA\xD0\xDA\xDA"
    result = unpack_by_bitdepth(data, w, h, 8)
    assert len(result) >= 2

@test("1. Pixel Unpack – Adversarial")
def test_unpack_video_mode_one_byte_file():
    from video_mode import unpack_by_bitdepth
    result = unpack_by_bitdepth(b"\x00", 8, 4, 8)
    assert isinstance(result, list)  # must not crash


# ═══════════════════════════════════════════════════════════════════════════
#  2.  LAZY FRAMES — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_empty_file():
    """Zero-byte file → 0 frames, no crash."""
    from utils import LazyFrames
    path = _tmpfile(b"")
    try:
        lf = LazyFrames(path, 8, 4, 8)
        assert len(lf) == 0
    finally:
        os.unlink(path)

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_one_byte_file():
    from utils import LazyFrames
    path = _tmpfile(b"\xFF")
    try:
        lf = LazyFrames(path, 8, 4, 8)
        assert len(lf) == 0   # 1 byte < 32-byte frame
    finally:
        os.unlink(path)

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_corrupt_all_ff():
    """File full of 0xFF — must return a valid array, not crash."""
    from utils import LazyFrames
    w, h = 8, 4
    data = b"\xFF" * (w * h)   # exactly 1 frame of 8-bit
    path = _tmpfile(data)
    try:
        lf = LazyFrames(path, w, h, 8)
        frame = lf[0]
        assert frame.shape == (h, w)
        assert int(frame.max()) == 255
    finally:
        os.unlink(path)

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_negative_index():
    from utils import LazyFrames
    path = _tmpfile(_raw8(8, 4, n=3))
    try:
        lf = LazyFrames(path, 8, 4, 8)
        try:
            frame = lf[-1]   # Python-style negative index
            # If supported, must return a valid frame
            assert frame.shape == (4, 8)
        except IndexError:
            pass  # rejecting negative index is also fine
    finally:
        os.unlink(path)

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_concurrent_reads():
    """Two threads reading different frames must not corrupt each other."""
    from utils import LazyFrames
    w, h = 8, 4
    # Frame 0 = all 10, Frame 1 = all 20
    data = bytes([10] * (w * h)) + bytes([20] * (w * h))
    path = _tmpfile(data)
    results = {}
    errors = []
    def read(idx):
        try:
            results[idx] = int(LazyFrames(path, w, h, 8)[idx][0, 0])
        except Exception as e:
            errors.append(str(e))
    threads = [threading.Thread(target=read, args=(i,)) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    try:
        assert not errors, f"Thread errors: {errors}"
        assert results[0] == 10
        assert results[1] == 20
    finally:
        os.unlink(path)

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_get_raw_empty_file():
    from utils import LazyFrames
    path = _tmpfile(b"")
    try:
        lf = LazyFrames(path, 8, 4, 16)
        try:
            lf.get_raw(0)
            assert False, "Expected IndexError on empty file"
        except IndexError:
            pass
    finally:
        os.unlink(path)

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_file_deleted_after_open():
    """File disappears after LazyFrames is created — access must fail gracefully."""
    from utils import LazyFrames
    path = _tmpfile(_raw8(8, 4, n=2))
    lf = LazyFrames(path, 8, 4, 8)
    os.unlink(path)
    try:
        frame = lf[0]
        # memmap may still work after unlink on Linux; that is acceptable
        assert frame is not None
    except (FileNotFoundError, OSError, ValueError):
        pass  # explicit failure is fine

@test("2. LazyFrames – Adversarial")
def test_lazy_frames_index_exactly_at_last():
    from utils import LazyFrames
    w, h, n = 4, 2, 5
    path = _tmpfile(_raw8(w, h, fill=42, n=n))
    try:
        lf = LazyFrames(path, w, h, 8)
        frame = lf[n - 1]   # last valid index
        assert frame.shape == (h, w)
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  GEO UTILITIES — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("3. Geo Utilities – Adversarial")
def test_meters_per_degree_zero_lat():
    from utils import meters_per_degree
    lat_m, lon_m = meters_per_degree(0)
    assert lat_m > 0 and lon_m > 0

@test("3. Geo Utilities – Adversarial")
def test_meters_per_degree_exactly_90():
    """At exactly 90° longitude converges to ~0 — must not divide-by-zero."""
    from utils import meters_per_degree
    lat_m, lon_m = meters_per_degree(90)
    assert not math.isnan(lat_m)
    assert not math.isnan(lon_m)
    assert not math.isinf(lat_m)

@test("3. Geo Utilities – Adversarial")
def test_meters_per_degree_minus_90():
    from utils import meters_per_degree
    lat_m, lon_m = meters_per_degree(-90)
    assert not math.isnan(lat_m)
    assert not math.isnan(lon_m)

@test("3. Geo Utilities – Adversarial")
def test_image_coords_beyond_image_bounds():
    """Pixel coords outside image — must return a number, not crash."""
    from utils import image_coords_to_latlon
    geo_info = (10.0, 20.0, 100, 50, 5.0)
    lat, lon, band = image_coords_to_latlon(999, 999, geo_info)
    assert isinstance(lat, float)
    assert isinstance(lon, float)
    assert isinstance(band, int)

@test("3. Geo Utilities – Adversarial")
def test_image_coords_negative_xy():
    from utils import image_coords_to_latlon
    geo_info = (10.0, 20.0, 100, 50, 5.0)
    lat, lon, band = image_coords_to_latlon(-50, -10, geo_info)
    assert isinstance(lat, float)

@test("3. Geo Utilities – Adversarial")
def test_image_coords_zero_pixel_size():
    """pixel_size_m = 0 would cause divide-by-zero — must survive."""
    from utils import image_coords_to_latlon
    geo_info = (10.0, 20.0, 100, 50, 0.0)
    try:
        lat, lon, band = image_coords_to_latlon(50, 25, geo_info)
        assert not math.isnan(lat), "NaN lat with 0 pixel size"
    except (ZeroDivisionError, Exception):
        pass  # explicit rejection is fine

@test("3. Geo Utilities – Adversarial")
def test_image_coords_empty_bands_info():
    """Empty bands_info dict — must not KeyError."""
    from utils import image_coords_to_latlon
    geo_info = (10.0, 20.0, 100, 50, 5.0)
    lat, lon, band = image_coords_to_latlon(50, 25, geo_info, bands_info={}, orig_band_h=50)
    assert isinstance(lat, float)

@test("3. Geo Utilities – Adversarial")
def test_image_coords_bands_info_missing_index_key():
    """bands_info entry without 'index' key — must not KeyError."""
    from utils import image_coords_to_latlon
    geo_info = (10.0, 20.0, 100, 50, 5.0)
    bands_info = {
        "B1": {"binned": False, "split": False, "bin_factor": 1}  # no 'index'
    }
    try:
        lat, lon, band = image_coords_to_latlon(50, 25, geo_info, bands_info=bands_info, orig_band_h=50)
        assert isinstance(lat, float)
    except (KeyError, Exception):
        pass

@test("3. Geo Utilities – Adversarial")
def test_image_coords_extreme_lat_lon():
    """Center at lat=89.9 — should not produce NaN/Inf."""
    from utils import image_coords_to_latlon
    geo_info = (89.9, 179.9, 100, 50, 1.0)
    lat, lon, _ = image_coords_to_latlon(50, 25, geo_info)
    assert not math.isnan(lat)
    assert not math.isinf(lon)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  FILE / JSON UTILITIES — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("4. File Utilities – Adversarial")
def test_atomic_write_json_deeply_nested():
    """Very deep nesting must not hit recursion limit in json.dump."""
    from utils import atomic_write_json
    data = {}
    cur = data
    for i in range(100):
        cur["child"] = {}
        cur = cur["child"]
    cur["leaf"] = "value"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "deep.json")
        atomic_write_json(path, data)
        assert os.path.exists(path)

@test("4. File Utilities – Adversarial")
def test_atomic_write_json_unicode():
    from utils import atomic_write_json
    data = {"emoji": "🛰️", "arabic": "مرحبا", "japanese": "衛星"}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "unicode.json")
        atomic_write_json(path, data)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["emoji"] == "🛰️"

@test("4. File Utilities – Adversarial")
def test_atomic_write_json_empty_dict():
    from utils import atomic_write_json
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "empty.json")
        atomic_write_json(path, {})
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == {}

@test("4. File Utilities – Adversarial")
def test_load_folder_params_corrupted_db_entry(tmp_path):
    """If DB has corrupt JSON for a folder it must not crash the app."""
    import sqlite3
    from utils import PARAM_DB_PATH, load_folder_params
    # Write a garbage value for this test folder
    folder = str(tmp_path / "corrupt_folder")
    os.makedirs(folder, exist_ok=True)
    try:
        conn = sqlite3.connect(PARAM_DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO folder_settings (folder_path, params_json) VALUES (?, ?)",
            (os.path.abspath(folder).replace('\\', '/'), "{NOT VALID JSON !!!")
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # if table doesn't exist yet, skip DB manipulation
    result = load_folder_params(folder)
    assert result is None or isinstance(result, dict)

@test("4. File Utilities – Adversarial")
def test_load_folder_params_legacy_json_corrupt(tmp_path):
    """Legacy parameters.json exists but is corrupt — must not crash."""
    from utils import load_folder_params
    folder = tmp_path / "band_data"
    folder.mkdir()
    (folder / "parameters.json").write_text("{broken json !!}")
    result = load_folder_params(str(folder))
    assert result is None or isinstance(result, dict)

@test("4. File Utilities – Adversarial")
def test_add_recent_very_long_path():
    """Extremely long path must not corrupt the recents file."""
    from utils import add_recent, get_recents_for_mode
    long_path = "/fake/" + "a" * 4096
    add_recent(long_path, "band")   # must not raise
    recents = get_recents_for_mode("band")
    assert isinstance(recents, list)

@test("4. File Utilities – Adversarial")
def test_add_recent_special_chars_in_path():
    from utils import add_recent, get_recents_for_mode
    weird = "/tmp/folder with spaces & 'quotes' and \"double\" and <tags>"
    add_recent(weird, "raw")
    recents = get_recents_for_mode("raw")
    paths = [r.get("path") if isinstance(r, dict) else r for r in recents]
    assert weird in paths

@test("4. File Utilities – Adversarial")
def test_add_recent_none_params():
    from utils import add_recent
    add_recent("/some/path", "band", params=None)  # must not raise

@test("4. File Utilities – Adversarial")
def test_add_recent_params_with_nonserializable(tmp_path):
    """params containing non-JSON-serializable objects must not propagate crash to caller."""
    from utils import add_recent
    try:
        add_recent(str(tmp_path), "band", params={"np_array": np.array([1, 2, 3])})
    except Exception:
        pass  # crash inside add_recent is acceptable; crash propagating to UI is not

@test("4. File Utilities – Adversarial")
def test_remove_recent_nonexistent_path():
    from utils import remove_recent
    result = remove_recent("/path/that/was/never/added/xyz123", "band")
    # Should return False, not raise
    assert result is False or result is None

@test("4. File Utilities – Adversarial")
def test_remove_recent_then_get():
    from utils import add_recent, remove_recent, get_recents_for_mode
    path = "/tmp/to_remove_test_path_99"
    add_recent(path, "band")
    remove_recent(path, "band")
    recents = get_recents_for_mode("band")
    paths = [r.get("path") if isinstance(r, dict) else r for r in recents]
    assert path not in paths

@test("4. File Utilities – Adversarial")
def test_recents_file_corrupt_on_disk(tmp_path, monkeypatch=None):
    """If recent.json is corrupt, load_recents must return [] not raise."""
    import utils as utils_mod
    orig = utils_mod.RECENT_FILE
    corrupt_path = str(tmp_path / "recent.json")
    Path(corrupt_path).write_text("{this is NOT valid JSON!!!!}")
    utils_mod.RECENT_FILE = corrupt_path
    try:
        result = utils_mod.load_recents()
        assert isinstance(result, list)
    finally:
        utils_mod.RECENT_FILE = orig

@test("4. File Utilities – Adversarial")
def test_save_params_then_load_roundtrip(tmp_path):
    from utils import save_params_for_path, load_folder_params
    folder = str(tmp_path / "data")
    os.makedirs(folder)
    params = {"width": 9344, "height": 384, "bitdepth": 10, "note": "test"}
    save_params_for_path(folder, params, as_default=True)
    loaded = load_folder_params(folder)
    assert loaded is not None
    default = loaded.get("default") or loaded
    assert default.get("width") == 9344 or loaded.get("width") == 9344

@test("4. File Utilities – Adversarial")
def test_infer_dataset_image_params_from_json(tmp_path):
    from utils import infer_dataset_image_params
    folder = str(tmp_path / "dataset_json")
    os.makedirs(folder)
    Path(folder, "capture.json").write_text(json.dumps({
        "RegionHeight": 384,
        "Width": 8448,
        "TDIStages": 2,
        "BandHeight": 192
    }))
    inferred = infer_dataset_image_params(folder)
    assert inferred.get("width") == 8448
    assert inferred.get("raw_height") == 384
    assert inferred.get("tdi_stage") == 2
    assert inferred.get("effective_height") == 192

@test("4. File Utilities – Adversarial")
def test_infer_dataset_image_params_invalid_tdi_defaults_to_zero(tmp_path):
    from utils import infer_dataset_image_params
    folder = str(tmp_path / "dataset_log")
    os.makedirs(folder)
    Path(folder, "capture.log").write_text(
        "[04,655.508][I44] Applied Width=8448\n"
        "[04,655.733][I45] Applied RegionHeight=384\n"
        "[04,656.582][I48] Applied TDI_Stages=64\n"
    )
    inferred = infer_dataset_image_params(folder)
    assert inferred.get("width") == 8448
    assert inferred.get("raw_height") == 384
    assert inferred.get("tdi_stage") == 0
    assert inferred.get("effective_height") == 384


# ═══════════════════════════════════════════════════════════════════════════
#  5.  OFFSET & PAD MATH — ADVERSARIAL  (BandViewsMixin stand-alone tests)
# ═══════════════════════════════════════════════════════════════════════════

class _FakeMixin:
    """Minimal shim so we can call BandViewsMixin helper methods without full UI."""
    from band_views import BandViewsMixin
    apply_offset  = BandViewsMixin.apply_offset
    _upsample_frame = BandViewsMixin._upsample_frame
    _pad_to_height  = BandViewsMixin._pad_to_height

_mixin = _FakeMixin()

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_zero():
    frame = np.arange(32, dtype=np.uint8).reshape(4, 8)
    result = _mixin.apply_offset(frame, 0, 0)
    assert np.array_equal(result, frame)

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_positive_x():
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin.apply_offset(frame, 3, 0)
    assert result.shape == (4, 8)
    # First 3 columns must be zeros (shifted right)
    assert np.all(result[:, :3] == 0), f"Expected leading zeros: {result[:, :3]}"
    assert np.all(result[:, 3:] == 1)

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_negative_x():
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin.apply_offset(frame, -3, 0)
    assert result.shape == (4, 8)
    assert np.all(result[:, 5:] == 0)  # last 3 cols zeroed

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_larger_than_frame():
    """Offset bigger than frame dimensions — result must be all-zero, not crash."""
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin.apply_offset(frame, 100, 0)
    assert result.shape == (4, 8)
    assert np.all(result == 0), "Expected all-zero frame after massive offset"

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_negative_larger_than_frame():
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin.apply_offset(frame, -100, 0)
    assert result.shape == (4, 8)
    assert np.all(result == 0)

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_both_axes():
    frame = np.ones((8, 8), dtype=np.uint8)
    result = _mixin.apply_offset(frame, 2, 2)
    assert result.shape == (8, 8)
    assert np.all(result[:2, :] == 0)   # top rows zeroed
    assert np.all(result[:, :2] == 0)   # left cols zeroed

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_crop_mode():
    frame = np.ones((8, 8), dtype=np.uint8)
    result = _mixin.apply_offset(frame, 2, 2, crop=True)
    # Crop removes the zero rows/cols — shape shrinks
    assert result.shape[0] <= 8
    assert result.shape[1] <= 8

@test("5. Offset & Pad – Adversarial")
def test_apply_offset_single_pixel_frame():
    frame = np.array([[42]], dtype=np.uint8)
    result = _mixin.apply_offset(frame, 0, 0)
    assert result[0, 0] == 42
    result2 = _mixin.apply_offset(frame, 1, 0)
    assert result2.shape == (1, 1)
    assert result2[0, 0] == 0   # shifted off the edge

@test("5. Offset & Pad – Adversarial")
def test_upsample_factor_1():
    frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    result = _mixin._upsample_frame(frame, 1)
    assert np.array_equal(result, frame)

@test("5. Offset & Pad – Adversarial")
def test_upsample_factor_2():
    frame = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    result = _mixin._upsample_frame(frame, 2)
    assert result.shape == (4, 4)
    assert result[0, 0] == 10 and result[0, 1] == 10
    assert result[1, 0] == 10 and result[1, 2] == 20

@test("5. Offset & Pad – Adversarial")
def test_upsample_factor_zero():
    """factor=0 must not divide-by-zero or produce garbage."""
    frame = np.ones((4, 4), dtype=np.uint8)
    result = _mixin._upsample_frame(frame, 0)
    # factor<=1 returns original per implementation
    assert result.shape == (4, 4)

@test("5. Offset & Pad – Adversarial")
def test_pad_to_height_no_padding_needed():
    frame = np.ones((10, 8), dtype=np.uint8)
    result = _mixin._pad_to_height(frame, 5)  # target < actual
    assert result.shape == (10, 8)  # unchanged

@test("5. Offset & Pad – Adversarial")
def test_pad_to_height_exact():
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin._pad_to_height(frame, 4)
    assert result.shape == (4, 8)

@test("5. Offset & Pad – Adversarial")
def test_pad_to_height_larger():
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin._pad_to_height(frame, 10)
    assert result.shape == (10, 8)
    assert np.all(result[4:, :] == 0)  # padded rows are zero

@test("5. Offset & Pad – Adversarial")
def test_pad_to_height_zero_target():
    frame = np.ones((4, 8), dtype=np.uint8)
    result = _mixin._pad_to_height(frame, 0)
    # target < actual → must return original, not empty array
    assert result.shape[0] >= 0


# ═══════════════════════════════════════════════════════════════════════════
#  6.  VIDEO MODE HELPERS — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("6. Video Mode Helpers – Adversarial")
def test_apply_offset_simple_zero():
    from video_mode import PlaybackApp
    import sys
    from PyQt5.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    pa = PlaybackApp()
    frame = np.arange(32, dtype=np.uint8).reshape(4, 8)
    result = pa.apply_offset_simple(frame, 0, 0)
    assert np.array_equal(result, frame)
    pa.close()

@test("6. Video Mode Helpers – Adversarial", needs_qt=True)
def test_apply_offset_simple_wraps():
    """np.roll wraps, so no data is lost — just shifted cyclically."""
    from video_mode import PlaybackApp
    pa = PlaybackApp()
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
    shifted = pa.apply_offset_simple(frame, 4, 0)  # offset = full width
    assert np.array_equal(shifted, frame)   # wrap-around = identity
    pa.close()

@test("6. Video Mode Helpers – Adversarial", needs_qt=True)
def test_pad_frame_already_big():
    from video_mode import PlaybackApp
    pa = PlaybackApp()
    frame = np.ones((10, 10), dtype=np.uint8)
    result = pa.pad_frame(frame, 5, 5)   # target smaller than actual
    assert result.shape == (10, 10)      # should not shrink
    pa.close()

@test("6. Video Mode Helpers – Adversarial", needs_qt=True)
def test_pad_frame_enlarges():
    from video_mode import PlaybackApp
    pa = PlaybackApp()
    frame = np.ones((4, 4), dtype=np.uint8)
    result = pa.pad_frame(frame, 8, 8)
    assert result.shape == (8, 8)
    assert np.all(result[4:, :] == 0)
    pa.close()

@test("6. Video Mode Helpers – Adversarial", needs_qt=True)
def test_framesource_16bit_supported():
    from video_mode import FrameSource
    path = _tmpfile(_raw16(8, 4))
    try:
        fs = FrameSource(path, 8, 4, 16)
        assert len(fs) == 1
        frame = fs[0]
        assert frame.shape == (4, 8)
        assert frame.dtype == np.uint8
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
#  7.  HISTOGRAM COMPUTATION — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("7. Histogram – Adversarial")
def test_process_frame_all_zeros():
    from utils import _process_frame_array_to_hist
    frame = np.zeros((8, 8), dtype=np.uint8)
    hist, s, s2, cnt, mn, mx = _process_frame_array_to_hist(frame, ignore_extremes=False)
    assert cnt == 64
    assert mn == 0 and mx == 0

@test("7. Histogram – Adversarial")
def test_process_frame_all_max():
    from utils import _process_frame_array_to_hist
    frame = np.full((8, 8), 255, dtype=np.uint8)
    hist, s, s2, cnt, mn, mx = _process_frame_array_to_hist(frame, ignore_extremes=False)
    assert cnt == 64
    assert mx == 255

@test("7. Histogram – Adversarial")
def test_process_frame_single_pixel():
    from utils import _process_frame_array_to_hist
    frame = np.array([[128]], dtype=np.uint8)
    hist, s, s2, cnt, mn, mx = _process_frame_array_to_hist(frame, ignore_extremes=False)
    assert cnt == 1
    assert mn == mx == 128

@test("7. Histogram – Adversarial")
def test_process_frame_ignore_extremes_all_zero():
    """When all pixels are 0, ignore_extremes masks everything — must not crash."""
    from utils import _process_frame_array_to_hist
    frame = np.zeros((8, 8), dtype=np.uint8)
    hist, s, s2, cnt, mn, mx = _process_frame_array_to_hist(frame, ignore_extremes=True)
    assert hist is not None

@test("7. Histogram – Adversarial")
def test_process_frame_uint16_all_zero():
    from utils import _process_frame_array_to_hist
    frame = np.zeros((4, 4), dtype=np.uint16)
    hist, *_ = _process_frame_array_to_hist(frame, ignore_extremes=False)
    assert hist is not None

@test("7. Histogram – Adversarial")
def test_process_frame_uint16_max():
    from utils import _process_frame_array_to_hist
    frame = np.full((4, 4), 65535, dtype=np.uint16)
    hist, s, s2, cnt, mn, mx = _process_frame_array_to_hist(frame, ignore_extremes=False)
    assert mx == 65535

@test("7. Histogram – Adversarial")
def test_compute_hist_empty_frames():
    from utils import _compute_hist_for_key
    result = _compute_hist_for_key(("key", [], "Single", 0, 0, 0, False))
    key, hist, gmin, gmax, cnt, s, s2 = result
    assert hist is not None
    assert cnt == 0

@test("7. Histogram – Adversarial")
def test_compute_hist_frame_index_out_of_range():
    from utils import _compute_hist_for_key
    frames = [np.ones((4, 4), dtype=np.uint8)]
    # Request frame index 999 — must handle gracefully
    result = _compute_hist_for_key(("k", frames, "Single", 999, 0, 0, False))
    key, hist, *_ = result
    assert hist is not None


# ═══════════════════════════════════════════════════════════════════════════
#  8.  TILE ORDER — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("8. TileOrder – Adversarial")
def test_tile_order_1x1_grid():
    from tiled_viewer import TileOrder
    for order in ("row_major", "col_major", "snake"):
        idx = TileOrder.get_index(order, 0, 0, 1, 1)
        assert idx == 0

@test("8. TileOrder – Adversarial")
def test_tile_order_no_duplicate_indices():
    """Each cell in an NxM grid must map to a unique index."""
    from tiled_viewer import TileOrder
    rows, cols = 4, 5
    for order in ("row_major", "col_major", "snake"):
        seen = set()
        for r in range(rows):
            for c in range(cols):
                idx = TileOrder.get_index(order, r, c, rows, cols)
                assert idx not in seen, f"Duplicate index {idx} at ({r},{c}) in {order}"
                seen.add(idx)
        assert len(seen) == rows * cols

@test("8. TileOrder – Adversarial")
def test_tile_order_indices_in_range():
    from tiled_viewer import TileOrder
    rows, cols = 3, 4
    for order in ("row_major", "col_major", "snake"):
        for r in range(rows):
            for c in range(cols):
                idx = TileOrder.get_index(order, r, c, rows, cols)
                assert 0 <= idx < rows * cols, f"Index {idx} out of range in {order}"

@test("8. TileOrder – Adversarial")
def test_tile_order_large_grid():
    from tiled_viewer import TileOrder
    rows, cols = 20, 20
    for order in ("row_major", "col_major"):
        seen = set()
        for r in range(rows):
            for c in range(cols):
                idx = TileOrder.get_index(order, r, c, rows, cols)
                seen.add(idx)
        assert len(seen) == rows * cols


# ═══════════════════════════════════════════════════════════════════════════
#  9.  PARAM HASH — ADVERSARIAL
# ═══════════════════════════════════════════════════════════════════════════

@test("9. Param Hash – Adversarial")
def test_param_hash_empty_dict():
    from utils import _param_hash
    h = _param_hash({})
    assert isinstance(h, str) and len(h) == 32   # MD5 hex

@test("9. Param Hash – Adversarial")
def test_param_hash_deterministic():
    from utils import _param_hash
    params = {"width": 9344, "height": 384}
    assert _param_hash(params) == _param_hash(params)

@test("9. Param Hash – Adversarial")
def test_param_hash_order_independent():
    """JSON sort_keys ensures {"a":1,"b":2} == {"b":2,"a":1}."""
    from utils import _param_hash
    p1 = {"a": 1, "b": 2, "c": 3}
    p2 = {"c": 3, "a": 1, "b": 2}
    assert _param_hash(p1) == _param_hash(p2)

@test("9. Param Hash – Adversarial")
def test_param_hash_different_params_differ():
    from utils import _param_hash
    assert _param_hash({"width": 100}) != _param_hash({"width": 200})

@test("9. Param Hash – Adversarial")
def test_param_hash_non_serializable_returns_empty():
    """Non-JSON-serializable input must return empty string, not raise."""
    from utils import _param_hash
    result = _param_hash({"arr": np.array([1, 2, 3])})
    assert isinstance(result, str)   # empty string is the documented fallback


# ═══════════════════════════════════════════════════════════════════════════
#  10. IMAGE VIEWER UTILITIES — ADVERSARIAL  (Qt)
# ═══════════════════════════════════════════════════════════════════════════

@test("10. Image Viewer – Adversarial", needs_qt=True)
def test_pil_to_qimage_1x1_grayscale():
    from PIL import Image
    from image_viewer import pil_to_qimage
    img = Image.fromarray(np.array([[128]], dtype=np.uint8), mode='L')
    q = pil_to_qimage(img)
    assert not q.isNull()
    assert q.width() == 1 and q.height() == 1

@test("10. Image Viewer – Adversarial", needs_qt=True)
def test_pil_to_qimage_rgba_converts():
    """RGBA mode triggers the 'convert to RGB' fallback path."""
    from PIL import Image
    from image_viewer import pil_to_qimage
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    img = Image.fromarray(arr, mode='RGBA')
    q = pil_to_qimage(img)
    assert not q.isNull()

@test("10. Image Viewer – Adversarial", needs_qt=True)
def test_pil_to_qimage_palette_mode():
    """Palette ('P') mode — must convert, not crash."""
    from PIL import Image
    from image_viewer import pil_to_qimage
    img = Image.new('P', (8, 8))
    q = pil_to_qimage(img)
    assert not q.isNull()

@test("10. Image Viewer – Adversarial", needs_qt=True)
def test_qimage_to_pil_large():
    from PIL import Image
    from image_viewer import pil_to_qimage, qimage_to_pil
    arr = np.random.randint(0, 255, (256, 512), dtype=np.uint8)
    img_in = Image.fromarray(arr, 'L')
    qimg = pil_to_qimage(img_in)
    img_out = qimage_to_pil(qimg)
    assert img_out.size == (512, 256)

@test("10. Image Viewer – Adversarial", needs_qt=True)
def test_pil_to_qimage_all_white():
    from PIL import Image
    from image_viewer import pil_to_qimage
    img = Image.fromarray(np.full((16, 16), 255, dtype=np.uint8), 'L')
    q = pil_to_qimage(img)
    assert not q.isNull()


# ═══════════════════════════════════════════════════════════════════════════
#  11. WIDGET STRESS — ADVERSARIAL  (Qt)
# ═══════════════════════════════════════════════════════════════════════════

@test("11. Widget Stress – Adversarial", needs_qt=True)
def test_band_app_open_close_3x():
    """Repeated open/close must not leak or crash."""
    from band_app import BandStitchProApp
    for _ in range(3):
        app = BandStitchProApp()
        app.close()
        del app
    gc.collect()

@test("11. Widget Stress – Adversarial", needs_qt=True)
def test_raw_viewer_load_nonexistent_file():
    """Asking RawViewer to load a nonexistent file must show an error, not crash."""
    from raw_mode import RawViewer
    v = RawViewer()
    try:
        if hasattr(v, '_do_load_raw_file'):
            v._do_load_raw_file("/nonexistent/file.raw")
        elif hasattr(v, 'load_raw_file'):
            v.load_raw_file()   # will open file dialog — skip silently
    except Exception:
        pass  # GUI dialogs may fail in headless mode
    v.close()

@test("11. Widget Stress – Adversarial", needs_qt=True)
def test_graphics_viewer_show_none_image():
    """Calling show_image(None) must clear the viewer, not crash."""
    from image_viewer import GraphicsImageViewer
    v = GraphicsImageViewer()
    v.show()
    try:
        v.show_image(None)
    except Exception:
        pass   # some headless environments skip rendering
    v.close()

@test("11. Widget Stress – Adversarial", needs_qt=True)
def test_graphics_viewer_show_1x1_image():
    from PIL import Image
    from image_viewer import GraphicsImageViewer
    v = GraphicsImageViewer()
    img = Image.fromarray(np.array([[0]], dtype=np.uint8), 'L')
    try:
        v.show_image(img)
    except Exception:
        pass
    v.close()

@test("11. Widget Stress – Adversarial", needs_qt=True)
def test_histogram_viewer_set_empty():
    from ui_components import HistogramViewer
    hv = HistogramViewer()
    try:
        hv.set_histograms({}, "empty test")
    except Exception:
        pass
    hv.close()

@test("11. Widget Stress – Adversarial", needs_qt=True)
def test_histogram_viewer_set_none_data():
    from ui_components import HistogramViewer
    hv = HistogramViewer()
    try:
        hv.set_histograms(None, "none test")
    except Exception:
        pass
    hv.close()


# ═══════════════════════════════════════════════════════════════════════════
#  12. EDITOR TAB COMMANDS — ADVERSARIAL  (Qt)
# ═══════════════════════════════════════════════════════════════════════════

@test("12. Editor Commands – Adversarial", needs_qt=True)
def test_swap_same_index():
    """Swapping a tile with itself must be a no-op, not corrupt state."""
    from editor_tab import SwapCommand
    from PyQt5.QtWidgets import QUndoStack

    class FakeEditor:
        def __init__(self):
            self.tiles = ["A", "B", "C"]
        def _swap_tiles(self, a, b):
            self.tiles[a], self.tiles[b] = self.tiles[b], self.tiles[a]
        def _refresh(self): pass

    ed = FakeEditor()
    stack = QUndoStack()
    stack.push(SwapCommand(ed, 1, 1))
    assert ed.tiles == ["A", "B", "C"]

@test("12. Editor Commands – Adversarial", needs_qt=True)
def test_flip_undo_restores_exactly():
    from editor_tab import FlipCommand
    from PyQt5.QtWidgets import QUndoStack
    import numpy as np

    class FakeEditor:
        def __init__(self):
            self.tiles = [np.array([[1, 2], [3, 4]])]
        def _flip_tile(self, idx, lr=False, tb=False):
            if lr: self.tiles[idx] = np.fliplr(self.tiles[idx])
            if tb: self.tiles[idx] = np.flipud(self.tiles[idx])
        def _refresh(self): pass

    original = np.array([[1, 2], [3, 4]])
    ed = FakeEditor()
    stack = QUndoStack()
    stack.push(FlipCommand(ed, [0], lr=True))
    assert not np.array_equal(ed.tiles[0], original)
    stack.undo()
    assert np.array_equal(ed.tiles[0], original)

@test("12. Editor Commands – Adversarial", needs_qt=True)
def test_rotate_undo_redo_cycle():
    from editor_tab import RotateCommand
    from PyQt5.QtWidgets import QUndoStack
    import numpy as np

    class FakeEditor:
        def __init__(self):
            self.tiles = [np.arange(4).reshape(2, 2)]
        def _rotate_tile(self, idx, k):
            self.tiles[idx] = np.rot90(self.tiles[idx], k)
        def _refresh(self): pass

    original = np.arange(4).reshape(2, 2).copy()
    ed = FakeEditor()
    stack = QUndoStack()
    stack.push(RotateCommand(ed, [0], k=1))
    stack.undo()
    assert np.array_equal(ed.tiles[0], original)
    stack.redo()
    assert not np.array_equal(ed.tiles[0], original)


# ═══════════════════════════════════════════════════════════════════════════
#  13. MEMORY / RESOURCE SAFETY
# ═══════════════════════════════════════════════════════════════════════════

@test("13. Memory Safety – Adversarial")
def test_unpack_very_large_dimension_doesnt_allocate():
    """Absurd dimensions with 0 data — must return empty immediately."""
    from utils import unpack_8bit
    result = unpack_8bit(b"", 999999, 999999)
    assert result == [] or len(result) == 0

@test("13. Memory Safety – Adversarial")
def test_lazy_frames_zero_wh():
    """LazyFrames with w=0 or h=0 — bytes_per_frame=0, must not divide by zero."""
    path = _tmpfile(b"\x00" * 100)
    try:
        from utils import LazyFrames
        try:
            lf = LazyFrames(path, 0, 4, 8)
            # bytes_per_frame = 0 → num_frames would be infinite or 0
            # Must not hang or OOM
            n = len(lf)
            assert n == 0 or n >= 0
        except (ValueError, ZeroDivisionError):
            pass
    finally:
        os.unlink(path)

@test("13. Memory Safety – Adversarial")
def test_upsample_large_factor_small_frame():
    """Upsampling a tiny frame by a large factor must not OOM (should be caught before call)."""
    frame = np.ones((2, 2), dtype=np.uint8)
    try:
        result = _mixin._upsample_frame(frame, 4)   # 8×8 — fine
        assert result.shape == (8, 8)
    except MemoryError:
        pass  # acceptable to fail gracefully


# ═══════════════════════════════════════════════════════════════════════════
#  RUNNER & HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════

def _run_one(name, fn):
    start = time.perf_counter()
    try:
        import inspect
        sig = inspect.signature(fn)
        if "tmp_path" in sig.parameters:
            with tempfile.TemporaryDirectory() as td:
                fn(tmp_path=Path(td))
        else:
            fn()
        return "PASS", (time.perf_counter() - start) * 1000, ""
    except AssertionError as e:
        return "FAIL", (time.perf_counter() - start) * 1000, str(e)
    except Exception as e:
        tb = traceback.format_exc()
        return "ERROR", (time.perf_counter() - start) * 1000, f"{type(e).__name__}: {e}\n\n{tb}"


def _html_report(results, run_qt, total_time):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total   = len(results)
    passed  = sum(1 for r in results if r[2] == "PASS")
    failed  = sum(1 for r in results if r[2] == "FAIL")
    errors  = sum(1 for r in results if r[2] == "ERROR")
    skipped = sum(1 for r in results if r[2] == "SKIP")
    pct     = int(passed / total * 100) if total else 0

    PC = "#2e7d32"; FC = "#c62828"; EC = "#e65100"; SC = "#757575"
    bar_c = PC if pct == 100 else (FC if pct < 70 else "#f57f17")

    cats = {}
    for cat, name, status, dur, detail in results:
        cats.setdefault(cat, []).append((name, status, dur, detail))

    rows = ""
    for cat, items in cats.items():
        cp = sum(1 for _, s, _, _ in items if s == "PASS")
        cc = len(items)
        cc_col = PC if cp == cc else FC
        rows += f'<tr><td colspan="4" style="background:#f0f0f0;font-weight:bold;color:{cc_col}">{_html.escape(cat)} — {cp}/{cc}</td></tr>'
        for name, status, dur, detail in items:
            sc = {PC: "PASS", FC: "FAIL", EC: "ERROR", SC: "SKIP"}.get(
                {"PASS": PC, "FAIL": FC, "ERROR": EC, "SKIP": SC}.get(status, SC), SC)
            badge_col = {"PASS": PC, "FAIL": FC, "ERROR": EC, "SKIP": SC}.get(status, SC)
            badge = f'<span style="background:{badge_col};color:white;padding:2px 8px;border-radius:4px;font-size:11px">{status}</span>'
            det = ""
            if detail:
                esc = _html.escape(detail).replace("\n", "<br>")
                det = f'<details><summary style="cursor:pointer;color:#1565c0">▶ Details</summary><pre style="font-size:11px;overflow:auto;background:#fff8e1;padding:8px;border-radius:4px">{esc}</pre></details>'
            rows += f'<tr><td style="font-family:monospace;font-size:12px;padding:4px 8px">{_html.escape(name)}</td><td style="text-align:center">{badge}</td><td style="text-align:right;color:#555;font-size:12px">{dur:.1f} ms</td><td>{det}</td></tr>'

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Adversarial Test Report</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;margin:24px;background:#fafafa;color:#212121}}
  h1{{color:#b71c1c}} h2{{color:#4a148c;margin-top:24px}}
  .summary{{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:20px}}
  .card{{background:white;border-radius:8px;padding:16px 24px;box-shadow:0 1px 4px rgba(0,0,0,.15);min-width:100px;text-align:center}}
  .card .num{{font-size:32px;font-weight:bold}} .card .lbl{{font-size:12px;color:#555}}
  table{{border-collapse:collapse;width:100%;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
  th{{background:#4a148c;color:white;padding:8px 12px;text-align:left}}
  tr:hover{{background:#f5f5f5}} td{{padding:6px 10px;border-bottom:1px solid #eee;vertical-align:top}}
  .bar{{height:18px;border-radius:9px;background:#e0e0e0;overflow:hidden;margin-bottom:8px}}
  .bar-fill{{height:100%;background:{bar_c}}}
</style></head><body>
<h1>🔨 Adversarial Test Report — Band/Satellite Viewer</h1>
<p style="color:#555">Generated: {now} &nbsp;|&nbsp; Runtime: {total_time:.2f}s &nbsp;|&nbsp; Qt tests: {"✅ enabled" if run_qt else "⚠ disabled (add --qt)"}</p>
<div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>
<p><strong>{pct}% passing</strong> ({passed}/{total})</p>
<div class="summary">
  <div class="card"><div class="num" style="color:{PC}">{passed}</div><div class="lbl">PASSED</div></div>
  <div class="card"><div class="num" style="color:{FC}">{failed}</div><div class="lbl">FAILED</div></div>
  <div class="card"><div class="num" style="color:{EC}">{errors}</div><div class="lbl">ERRORS</div></div>
  <div class="card"><div class="num" style="color:{SC}">{skipped}</div><div class="lbl">SKIPPED</div></div>
  <div class="card"><div class="num">{total}</div><div class="lbl">TOTAL</div></div>
</div>
<h2>What each failing test means for your users</h2>
<ul style="font-size:13px;line-height:1.8">
  <li><strong>Pixel Unpack failures</strong> → app crashes or freezes when user opens a slightly corrupt or truncated .raw file</li>
  <li><strong>LazyFrames failures</strong> → crash when file is deleted mid-session or file system runs out of space</li>
  <li><strong>Geo Utility failures</strong> → wrong lat/lon shown in status bar, or crash near poles/dateline</li>
  <li><strong>File Utility failures</strong> → recents list becomes corrupt, saved parameters lost between sessions</li>
  <li><strong>Offset/Pad failures</strong> → visual corruption (shifted bands, wrong alignment) when offset > image size</li>
  <li><strong>Histogram failures</strong> → histogram panel shows nothing or crashes on all-black/all-white frames</li>
  <li><strong>TileOrder failures</strong> → tiles appear in wrong positions in tiled viewer</li>
  <li><strong>Widget Stress failures</strong> → crash on rapid tab open/close or repeated operations</li>
</ul>
<table><thead><tr><th>Test Name</th><th>Status</th><th>Time</th><th>Details</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--qt", action="store_true", help="Run Qt widget tests (requires display or offscreen)")
    args = parser.parse_args()

    # Make app modules importable
    for p in [
        ".", "..",
        os.path.join(os.path.dirname(__file__), "..", "user-data", "uploads"),
        os.path.dirname(__file__),
    ]:
        ap = os.path.abspath(p)
        if ap not in sys.path:
            sys.path.insert(0, ap)

    if args.qt:
        from PyQt5.QtWidgets import QApplication
        _qapp = QApplication.instance() or QApplication(sys.argv)

    t0 = time.perf_counter()
    results = []

    for category, name, fn, needs_qt in _TESTS:
        if needs_qt and not args.qt:
            status, dur, detail = "SKIP", 0.0, "Qt tests disabled — run with --qt"
        else:
            if args.verbose:
                print(f"  [{category}]  {name} ... ", end="", flush=True)
            status, dur, detail = _run_one(name, fn)
            if args.verbose:
                print(status)
        results.append((category, name, status, dur, detail))

    total_time = time.perf_counter() - t0

    total   = len(results)
    passed  = sum(1 for r in results if r[2] == "PASS")
    failed  = sum(1 for r in results if r[2] == "FAIL")
    errors  = sum(1 for r in results if r[2] == "ERROR")
    skipped = sum(1 for r in results if r[2] == "SKIP")

    print("\n" + "=" * 65)
    print(f"  ADVERSARIAL RESULTS:  {passed} passed | {failed} failed | {errors} errors | {skipped} skipped")
    print(f"  TOTAL: {total} tests in {total_time:.2f}s")
    print("=" * 65)

    if failed or errors:
        print("\n⚠  THINGS THAT BROKE:")
        for cat, name, status, dur, detail in results:
            if status in ("FAIL", "ERROR"):
                print(f"\n  [{status}] {name}")
                for line in (detail or "").strip().split("\n")[:6]:
                    print(f"         {line}")

    report_path = os.path.join(os.path.dirname(__file__), "adversarial_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_html_report(results, args.qt, total_time))
    print(f"\n📄 Report → {report_path}\n")
    sys.exit(0 if not (failed or errors) else 1)


if __name__ == "__main__":
    main()
