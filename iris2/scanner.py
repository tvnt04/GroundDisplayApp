"""
iris2/scanner.py
================
Pure Python scanner. No Claude. No tokens. No subprocess.
Runs in a QThread — never blocks the UI.

Two scan modes:
  quick  — log + JSON analysis only, no pixel reading (~3-5s)
  full   — log + JSON + every frame of every band (1-5 min)

LOG ANALYSIS covers:
  - Parameter verification: every requested vs applied value
  - Trigger timing: W01 stale timestamp detection
  - Frame drops: TotalNoOfFrames vs CapturedCount
  - Error/warning inventory with codes
  - Temperature: SensorTemp + CoreTemp before/after, delta flagging
  - FPS stability: per-frame instantFps + TimeDifference jitter
  - Failed settings: any "Failed" in applied settings
  - Camera init issues: Invalid settings path, connection errors

PIXEL ANALYSIS covers:
  - Black frames (mean < threshold)
  - Dead columns (column mean near zero across all rows)
  - Alternating row banding (even/odd row mean difference)
  - Vertical striping (column variance >> row variance)
  - Saturation
  - Dead pixels
  - Cross-band mean outliers
  - Truncated files
"""

from __future__ import annotations
import os
import re
import json
import time
import statistics
import threading
from typing import Callable, Dict, List, Optional

try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    np = None
    _HAS_NP = False

from PyQt5.QtCore import QThread, pyqtSignal

from .app_state import state, ScanResult


# ── Constants ──────────────────────────────────────────────────────────────

LOG_EXTS          = (".log", ".txt")
DEAD_DN           = 5
BLACK_MEAN        = 20
DARK_CAPTURE_DN   = 50      # global mean below this = treat as dark/covered capture
MIN_SIGNAL_DN     = 50      # pixel detectors (dead cols, alt rows) require mean > this to fire
STRIPE_RATIO      = 4.0
TEMP_DELTA_FLAG   = 5.0      # °C — flag if sensor/core temp changes more than this
ALT_ROW_THRESHOLD = 8.0      # DN — flag alternating row banding if even/odd mean diff > this
PARAM_TOLERANCE   = 0.01     # 1% — flag if applied differs from requested by more than this

# ── Pixel defect detector thresholds ─────────────────────────────────────
# Dead pixel: ISOLATED single pixel significantly BELOW its neighbours.
#   A GROUP of low pixels (e.g. all neighbours also low) is NOT a dead pixel.
DEAD_PIXEL_NEIGHBOR_SIGMA  = 4.0   # pixel must be this many σ below neighbour mean
DEAD_PIXEL_MIN_DIFF_DN     = 25    # and at least this many DN below neighbour mean
DEAD_PIXEL_MAX_NEIGHBOR_VAR= 2000.0 # neighbours allowed to vary (raised to detect on gradients)

# Hot pixel: ISOLATED single pixel significantly ABOVE its neighbours.
#   A GROUP of high pixels is NOT a hot pixel.
HOT_PIXEL_NEIGHBOR_SIGMA   = 4.0   # pixel must be this many σ above neighbour mean
HOT_PIXEL_MIN_DIFF_DN      = 25    # and at least this many DN above neighbour mean
HOT_PIXEL_MAX_NEIGHBOR_VAR = 2000.0 # neighbours allowed to vary

# Group exclusion (applies to both dead and hot):
#   If the pixel's neighbours are also similarly low/high (i.e. similar to the suspect pixel),
#   it is part of a uniform region — NOT an isolated defect. This is the primary gate.
#   Additionally, if ≥ 2 neighbours are also independently flagged as outliers, it's a cluster.
ISOLATED_MAX_FLAGGED_NEIGHBORS = 1  # 0 or 1 flagged neighbours → still isolated

# Group exclusion: neighbour value similarity check.
#   If the suspect pixel's value is SIMILAR to its neighbours (within this DN), it's a group.
#   Applied BEFORE the sigma/diff check to catch uniform low/high regions.
GROUP_SIMILARITY_DN = 15  # if pixel is within 15 DN of neighbour mean → it's a group member

# Stuck pixel: pixel value barely changes while surrounding pixels do change.
#   GROUP exclusion: if neighbours are ALSO not changing, it's a dead region, not a stuck pixel.
STUCK_PIXEL_MAX_STD        = 2.0   # pixel std across frames must be ≤ this (nearly constant)
STUCK_PIXEL_MIN_SCENE_STD  = 8.0   # surrounding pixels must vary by at least this across frames
STUCK_PIXEL_MIN_FRAMES     = 4     # need at least this many frames to call stuck
STUCK_MAX_REPORT           = 50    # cap reported stuck pixels per band

# Test pattern: ramp detection thresholds
TEST_PATTERN_RAMP_MONO_THRESHOLD  = 0.80   # fraction of monotone diffs within each period
TEST_PATTERN_RAMP_CORR_THRESHOLD  = 0.80   # min correlation to ideal sawtooth

# Health score floors
HEALTH_FLOOR_DARK    = 85.0  # dark/covered capture — not unhealthy, just uninformative
HEALTH_FLOOR_MINIMUM = 30.0  # absolute minimum — 0% is never correct for observations

# Severity rules for pixel findings:
# Pixel findings are NEVER CRITICAL — only log findings (frame drops, param failures) can be.
# CRITICAL = confirmed data loss (frame drops, truncated file + CapturedCount mismatch)
# WARNING  = observed pattern that may affect data quality
# INFO     = observed pattern expected given configuration, or on dark capture


# ══════════════════════════════════════════════════════════════════════════════
# BIT DEPTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bytes_per_frame(w: int, h: int, bd: int) -> int:
    px = w * h
    if bd == 8:  return px
    if bd == 10: return (px * 10 + 7) // 8
    if bd == 12: return (px * 12 + 7) // 8
    if bd == 16: return px * 2
    return px * 2


def unpack_frame(raw: bytes, w: int, h: int, bd: int):
    if not _HAS_NP:
        return None
    px = w * h
    if bd == 8:
        return np.frombuffer(raw[:px], dtype=np.uint8).reshape(h, w).astype(np.float32)
    if bd == 16:
        return np.frombuffer(raw[:px*2], dtype=np.uint16).reshape(h, w).astype(np.float32)
    if bd == 10:
        ng = px // 4
        data = np.frombuffer(raw[:ng*5], dtype=np.uint8).reshape(-1, 5)
        b0,b1,b2,b3,b4 = (data[:,i].astype(np.uint16) for i in range(5))
        p = np.empty(ng * 4, dtype=np.uint16)
        p[0::4] = ((b0 << 2) | (b1 >> 6)) & 0x3FF
        p[1::4] = ((b1 << 4) | (b2 >> 4)) & 0x3FF
        p[2::4] = ((b2 << 6) | (b3 >> 2)) & 0x3FF
        p[3::4] = ((b3 << 8) |  b4)       & 0x3FF
        return p[:px].reshape(h, w).astype(np.float32)
    if bd == 12:
        ng = px // 2
        data = np.frombuffer(raw[:ng*3], dtype=np.uint8).reshape(-1, 3)
        b0,b1,b2 = (data[:,i].astype(np.uint16) for i in range(3))
        p = np.empty(ng * 2, dtype=np.uint16)
        p[0::2] = ((b0 << 4) | (b1 >> 4)) & 0xFFF
        p[1::2] = ((b1 << 8) |  b2)       & 0xFFF
        return p[:px].reshape(h, w).astype(np.float32)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# REPEAT-PATTERN DETECTOR (generic, automatic)
# ══════════════════════════════════════════════════════════════════════════════

def _dominant_period(profile: "np.ndarray", min_p: int = 8, max_p: int = 256) -> Optional[int]:
    """Find dominant period in 1D profile using autocorrelation."""
    if profile.size < max_p * 2:
        max_p = max(min_p + 1, profile.size // 4)
    if max_p <= min_p:
        return None
    x = profile.astype(np.float32)
    x -= x.mean()
    if np.allclose(x, 0):
        return None
    n = len(x)
    fft = np.fft.rfft(x, n=2 * n)
    corr = np.fft.irfft(fft * np.conj(fft))[:n]
    if corr[0] == 0:
        return None
    # Normalize
    corr = corr / (corr[0] + 1e-9)
    search = corr[min_p:max_p]
    if search.size == 0:
        return None
    idx = int(np.argmax(search))
    peak = float(search[idx])
    if peak < 0.2:
        return None
    return min_p + idx


def detect_repeating_pattern(arr, max_dim: int = 1024, max_tiles: int = 64) -> Optional[Dict]:
    """
    Generic repeat-pattern anomaly detector.
    - Auto-estimates tile size from 1D autocorrelation.
    - Compares tiles to a learned reference and flags deviating tiles.
    Returns None if no repeat structure found.
    """
    if not _HAS_NP or arr is None:
        return None
    h, w = arr.shape
    if h < 64 or w < 64:
        return None

    # Downsample for period detection
    step = max(1, max(h, w) // max_dim)
    ds = arr[::step, ::step].astype(np.float32)
    row_prof = ds.mean(axis=1)
    col_prof = ds.mean(axis=0)

    py = _dominant_period(row_prof)
    px = _dominant_period(col_prof)
    if not py or not px:
        return None

    tile_h = py * step
    tile_w = px * step
    if tile_h < 8 or tile_w < 8:
        return None
    if tile_h > h // 2 or tile_w > w // 2:
        return None

    # Build tile grid
    ys = list(range(0, h - tile_h + 1, tile_h))
    xs = list(range(0, w - tile_w + 1, tile_w))
    total_tiles = len(ys) * len(xs)
    if total_tiles < 4:
        return None

    # Sample tiles deterministically to limit work
    tiles = []
    coords = []
    stride = max(1, int((total_tiles / max_tiles) ** 0.5))
    for yi, y in enumerate(ys[::stride]):
        for xi, x in enumerate(xs[::stride]):
            tile = arr[y:y + tile_h, x:x + tile_w].astype(np.float32)
            tiles.append(tile)
            coords.append((x, y))
            if len(tiles) >= max_tiles:
                break
        if len(tiles) >= max_tiles:
            break

    if len(tiles) < 4:
        return None

    # Reference tile (mean)
    ref = np.mean(tiles, axis=0)
    ref_mean = ref.mean()
    ref = ref - ref_mean
    ref_norm = np.linalg.norm(ref) + 1e-9

    bad = []
    nccs = []
    for tile, (x, y) in zip(tiles, coords):
        t = tile - tile.mean()
        n = np.linalg.norm(t) + 1e-9
        ncc = float((t * ref).sum() / (n * ref_norm))
        nccs.append(ncc)
        if ncc < 0.85:
            bad.append({"x": x, "y": y, "ncc": round(ncc, 3)})

    if nccs:
        ncc_mean = float(np.mean(nccs))
        ncc_std = float(np.std(nccs))
        # If all tiles are consistently "off" by the same amount, it's a uniform pattern,
        # not an anomaly. Treat as non-anomalous.
        if len(bad) == len(nccs) and ncc_std < 0.05:
            return {
                "tile_w": tile_w,
                "tile_h": tile_h,
                "period_x": px,
                "period_y": py,
                "tiles_checked": len(tiles),
                "tiles_total": total_tiles,
                "bad_tiles": [],
                "bad_count": 0,
                "uniform": True,
                "ncc_mean": round(ncc_mean, 4),
                "ncc_std": round(ncc_std, 4),
            }

    return {
        "tile_w": tile_w,
        "tile_h": tile_h,
        "period_x": px,
        "period_y": py,
        "tiles_checked": len(tiles),
        "tiles_total": total_tiles,
        "bad_tiles": bad[:10],
        "bad_count": len(bad),
        "uniform": False,
        "ncc_mean": round(ncc_mean, 4) if nccs else None,
        "ncc_std": round(ncc_std, 4) if nccs else None,
    }


def detect_sawtooth_ramp(arr, bit_depth: int) -> Optional[Dict]:
    """
    Detect a test pattern ramp: pixel values constantly increasing (or decreasing)
    in every line repeatedly, then resetting. This is the primary test pattern signature.

    Rule: if pixel values in each row (or column) consistently increase or decrease
    in a repeating sawtooth, it IS a test pattern — regardless of NCC tile scores.

    Returns dict with axis, period, direction, score if detected, else None.
    """
    if not _HAS_NP or arr is None:
        return None
    h, w = arr.shape
    if h < 32 or w < 32:
        return None

    def _score_axis(profile: "np.ndarray") -> Optional[Dict]:
        """Score a 1D profile (row means or col means) for sawtooth ramp."""
        if profile.size < 16:
            return None
        x = profile.astype(np.float32)
        rng = float(x.max() - x.min())
        if rng < 5.0:
            return None  # flat, no ramp

        # Normalize to 0..1
        xn = (x - x.min()) / (rng + 1e-9)

        # Find dominant period
        p = _dominant_period(xn, min_p=8, max_p=min(512, xn.size // 2))
        if not p or p < 4:
            return None

        diffs = np.diff(xn)
        reset_threshold = 0.4  # large jump = period reset

        # Find resets (large absolute jumps)
        reset_pts = np.where(np.abs(diffs) > reset_threshold)[0] + 1
        if len(reset_pts) < 1:
            # No resets: measure overall monotonicity
            mono_inc = float((diffs > 0).mean())
            mono_dec = float((diffs < 0).mean())
            mono = max(mono_inc, mono_dec)
            direction = "increasing" if mono_inc >= mono_dec else "decreasing"
        else:
            # Measure monotonicity within each cycle between resets
            segments = np.split(diffs, reset_pts)
            mono_vals_inc = []
            mono_vals_dec = []
            for seg in segments:
                # Only non-reset diffs
                inner = seg[np.abs(seg) <= reset_threshold]
                if inner.size > 1:
                    mono_vals_inc.append(float((inner > 0).mean()))
                    mono_vals_dec.append(float((inner < 0).mean()))

            if not mono_vals_inc:
                return None

            mean_inc = float(np.mean(mono_vals_inc))
            mean_dec = float(np.mean(mono_vals_dec))
            mono = max(mean_inc, mean_dec)
            direction = "increasing" if mean_inc >= mean_dec else "decreasing"

        if mono < TEST_PATTERN_RAMP_MONO_THRESHOLD:
            return None

        # Correlation to ideal sawtooth
        idx = np.arange(xn.size, dtype=np.float32)
        saw = (idx % p) / float(max(p - 1, 1))
        xm = xn - xn.mean()
        sm = saw - saw.mean()
        denom = np.linalg.norm(xm) * np.linalg.norm(sm) + 1e-9
        corr = abs(float((xm * sm).sum() / denom))

        score = 0.55 * corr + 0.45 * mono
        return {
            "period":    int(p),
            "corr":      round(corr, 4),
            "mono":      round(mono, 4),
            "score":     round(score, 4),
            "direction": direction,
            "n_cycles":  max(1, len(reset_pts)),
        }

    # Also check individual rows directly (not just mean profile)
    # Sample up to 20 rows and measure per-row monotonicity fraction
    def _score_individual_rows(frame: "np.ndarray") -> Optional[Dict]:
        """Check if individual rows have consistent ramp pattern."""
        n_rows = min(20, frame.shape[0])
        row_indices = np.linspace(0, frame.shape[0] - 1, n_rows, dtype=int)
        mono_fracs = []
        for ri in row_indices:
            row = frame[ri].astype(np.float32)
            if row.max() - row.min() < 5:
                continue
            d = np.diff(row)
            # Find resets
            resets = np.where(np.abs(d) > (row.max() - row.min()) * 0.4)[0] + 1
            if resets.size > 0:
                segs = np.split(d, resets)
                for seg in segs:
                    inner = seg[np.abs(seg) <= (row.max() - row.min()) * 0.4]
                    if inner.size > 2:
                        mono_fracs.append(max(
                            float((inner > 0).mean()),
                            float((inner < 0).mean())
                        ))
            else:
                mono_fracs.append(max(
                    float((d > 0).mean()),
                    float((d < 0).mean())
                ))

        if not mono_fracs:
            return None
        avg_mono = float(np.mean(mono_fracs))
        if avg_mono >= TEST_PATTERN_RAMP_MONO_THRESHOLD:
            return {"mono": round(avg_mono, 4), "score": round(avg_mono, 4),
                    "direction": "per_row_ramp", "period": -1, "corr": avg_mono,
                    "n_cycles": len(mono_fracs)}
        return None

    # Downsample for speed on mean profiles
    step = max(1, max(h, w) // 512)
    ds = arr[::step, ::step]

    row_prof = ds.mean(axis=1)   # each row → vertical ramp
    col_prof = ds.mean(axis=0)   # each col → horizontal ramp

    row_res = _score_axis(row_prof)
    col_res = _score_axis(col_prof)

    # Also check individual row ramps directly
    row_direct = _score_individual_rows(arr)

    candidates = []
    if row_res:
        candidates.append({**row_res, "axis": "rows"})
    if col_res:
        candidates.append({**col_res, "axis": "cols"})
    if row_direct:
        candidates.append({**row_direct, "axis": "rows_direct"})

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x["score"])

    if best["corr"] < TEST_PATTERN_RAMP_CORR_THRESHOLD and best["mono"] < TEST_PATTERN_RAMP_MONO_THRESHOLD:
        return None

    return best


# ══════════════════════════════════════════════════════════════════════════════
# METADATA PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_metadata(folder: str) -> Dict:
    """Extract geometry + capture params from JSON files in folder."""
    meta = {
        "width": None, "height": None, "region_height": None,
        "bit_depth": 10, "fps": None, "exposure_time": None, "gain": None,
        "tdi_modes": None, "tdi_stages": None, "tdi_yshift": None,
        "band_xshift": None, "binning": None, "total_frames": None,
        "proc_mode": None, "sensor_temp_json": None, "core_temp_json": None,
    }

    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, f)) as jf:
                d = json.load(jf)

            meta["width"]         = d.get("Width") or d.get("width")
            meta["height"]        = d.get("Height") or d.get("height")
            meta["region_height"] = d.get("RegionHeight") or d.get("region_height")
            meta["fps"]           = d.get("FPS") or d.get("fps")
            meta["exposure_time"] = d.get("ExposureTime") or d.get("exposure_time")
            meta["gain"]          = d.get("Gain") or d.get("gain")
            meta["tdi_modes"]     = d.get("TDIMode") or d.get("tdi_mode")
            meta["tdi_stages"]    = d.get("TDIStages") or d.get("tdi_stages")
            meta["band_xshift"]   = d.get("BandXShift")
            meta["binning"]       = d.get("Binning")
            meta["total_frames"]  = d.get("TotalFrames") or d.get("total_frames")
            meta["proc_mode"]     = d.get("ProcMode") or d.get("proc_mode")
            meta["bit_depth"]     = d.get("BitDepth") or d.get("bit_depth") or 10

            # sensorTemperature stored *100 in JSON (5021 = 50.21°C)
            raw_st = d.get("sensorTemperature")
            if raw_st is not None:
                meta["sensor_temp_json"] = round(float(raw_st) / 100.0, 2)

            raw_ct = d.get("coreTemperature")
            if raw_ct is not None:
                meta["core_temp_json"] = float(raw_ct)

            if meta["width"] and meta["region_height"]:
                break
        except Exception:
            pass

    # Coerce types
    for k in ("width", "height", "region_height", "bit_depth",
               "tdi_modes", "tdi_stages", "total_frames"):
        if meta[k] is not None:
            try: meta[k] = int(meta[k])
            except Exception: pass
    for k in ("fps", "exposure_time", "gain", "band_xshift", "binning"):
        if meta[k] is not None:
            try: meta[k] = float(meta[k])
            except Exception: pass

    return meta



# ══════════════════════════════════════════════════════════════════════════════
# PROCMODE DECODER
# ══════════════════════════════════════════════════════════════════════════════

def decode_procmode(proc_mode_str: str) -> Dict:
    """
    Decode the ProcMode parameter string from JSON into its 14 named fields.

    Format (space-separated, 14 tokens):
      OrbitID  TaskID  JsonID  Date  Time  Duration  BandSelection
      TDI  FPS  ExposureTime  Gain  XShift  Binning  TDIYShift

    TDI byte decoding:
      0        = TDI off
      10 (0b00001010) = TDI on, 2-stage
      18 (0b00010010) = TDI on, 4-stage
      34 (0b00100010) = TDI on, 8-stage
      66 (0b01000010) = TDI on, 64-stage

    BandSelection byte (127 = 0b01111111 = all 7 bands active)
      Bits 6–0: one per band, 1 = active

    Binning byte (e.g. 119 = 0b01110111):
      Bit 7: CCSDS enabled (1) / disabled (0)
      Bits 6–0: per-band binning (1 = binned, 0 = unbinned)
    """
    result = {
        "raw": proc_mode_str,
        "decoded": {},
        "tdi_decoded": {},
        "band_selection": {},
        "binning_decoded": {},
        "parse_error": None,
    }

    tokens = proc_mode_str.strip().split()
    if len(tokens) < 14:
        result["parse_error"] = f"Expected 14 tokens, got {len(tokens)}"
        return result

    try:
        d = result["decoded"]
        d["orbit_id"]       = tokens[0].lstrip("0") or "0"
        d["task_id"]        = tokens[1].lstrip("0") or "0"
        d["json_id"]        = tokens[2].lstrip("0") or "0"
        d["date"]           = tokens[3]
        d["utc_time"]       = tokens[4]
        d["duration_sec"]   = float(tokens[5])
        d["band_selection"] = int(tokens[6])
        d["tdi_byte"]       = int(tokens[7])
        d["fps_requested"]  = float(tokens[8])
        d["exposure_time"]  = float(tokens[9])
        d["gain"]           = float(tokens[10])
        d["xshift"]         = float(tokens[11])
        d["binning_byte"]   = int(tokens[12])
        d["tdi_yshift"]     = int(tokens[13])

        # TDI decode
        tdi_byte = d["tdi_byte"]
        TDI_MAP = {0: "OFF", 10: "ON/2-stage", 18: "ON/4-stage",
                   34: "ON/8-stage", 66: "ON/64-stage"}
        result["tdi_decoded"] = {
            "raw_byte":   tdi_byte,
            "mode":       TDI_MAP.get(tdi_byte, f"UNKNOWN ({tdi_byte})"),
            "tdi_on":     tdi_byte != 0,
            "stages":     {0: 0, 10: 2, 18: 4, 34: 8, 66: 64}.get(tdi_byte),
        }

        # Band selection decode (bits 6–0)
        bs = d["band_selection"]
        result["band_selection"] = {
            f"band{i}": bool(bs & (1 << i)) for i in range(7)
        }
        result["band_selection"]["active_count"] = bin(bs & 0x7F).count("1")

        # Binning byte decode
        bn = d["binning_byte"]
        result["binning_decoded"] = {
            "raw_byte": bn,
            "ccsds_enabled": bool(bn & 0x80),
            "per_band_binned": {f"band{i}": bool(bn & (1 << i)) for i in range(7)},
        }

    except Exception as e:
        result["parse_error"] = str(e)

    return result


def verify_procmode_vs_applied(proc: Dict, params_applied: Dict,
                                params_requested: Dict, params_failed: List[str],
                                flag_fn) -> List[Dict]:
    """
    Cross-check every parameter in the decoded ProcMode against what
    the log reports was actually applied. Flags any discrepancy.

    Returns a list of cross-check results for the report.
    """
    if proc.get("parse_error") or not proc.get("decoded"):
        return []

    d = proc["decoded"]
    checks = []

    # Map: (label, procmode_value, applied_key, tolerance)
    # tolerance: absolute for small values, fraction for large
    checks_spec = [
        ("FPS",          d.get("fps_requested"), "FPS",          PARAM_TOLERANCE),
        ("ExposureTime", d.get("exposure_time"),  "ExposureTime", PARAM_TOLERANCE),
        ("Gain",         d.get("gain"),           "Gain",         PARAM_TOLERANCE),
        ("BandXShift",   d.get("xshift"),         "BandXShift",   0.01),
        # TDIYShift: requested value maps to RegionHeight, NOT G_TDIYShift.
        # G_TDIYShift = RegionHeight / TDI_stages (intentional halving — not a mismatch).
        # We verify RegionHeight == requested tdiyshift, then separately note G_TDIYShift.
        ("TDIYShift",    d.get("tdi_yshift"),     "RegionHeight", 0.01),
    ]

    for label, requested, applied_key, tol in checks_spec:
        if requested is None:
            continue

        # Find applied value — try several key variants
        applied = None
        for key_variant in (applied_key, applied_key.lower(),
                             applied_key.replace("_", ""),
                             f"Applied_{applied_key}"):
            applied = params_applied.get(key_variant)
            if applied is not None:
                break

        if applied is None:
            checks.append({
                "param": label, "requested": requested,
                "applied": "NOT FOUND IN LOG", "status": "UNVERIFIED"
            })
            continue

        failed = label in params_failed or applied_key in params_failed
        diff_pct = abs(applied - requested) / max(abs(requested), 0.001)

        if failed:
            status = "FAILED"
        elif diff_pct > tol:
            status = "AUTO_ADJUSTED"
        else:
            status = "OK"

        checks.append({
            "param": label, "requested": requested,
            "applied": applied, "status": status,
            "diff_pct": round(diff_pct * 100, 2),
        })

        if status == "FAILED":
            flag_fn("CRITICAL", "procmode_vs_applied",
                    f"[ProcMode check] {label}: requested {requested} — "
                    f"camera REJECTED this, applied {applied} instead.")
        elif status == "AUTO_ADJUSTED":
            flag_fn("WARNING", "procmode_vs_applied",
                    f"[ProcMode check] {label}: requested {requested} — "
                    f"camera auto-adjusted to {applied} "
                    f"({diff_pct*100:.1f}% difference). "
                    f"Hardware limit was reached.")

    # TDI mode check
    tdi_dec   = proc.get("tdi_decoded", {})
    tdi_mode  = params_applied.get("TDI_Modes") or params_applied.get("TDIMode")
    tdi_stage = params_applied.get("TDI_Stages") or params_applied.get("TDIStages")

    if tdi_dec.get("tdi_on") and tdi_mode is not None:
        expected_stages = tdi_dec.get("stages")
        if expected_stages and tdi_stage is not None:
            if int(tdi_stage) != expected_stages:
                flag_fn("CRITICAL", "procmode_vs_applied",
                        f"[ProcMode check] TDI stages: requested {expected_stages} "
                        f"(from TDI byte={d.get('tdi_byte')}), "
                        f"applied {int(tdi_stage)}. Mismatch in TDI configuration.")
                checks.append({
                    "param": "TDI_Stages", "requested": expected_stages,
                    "applied": tdi_stage, "status": "MISMATCH"
                })
            else:
                checks.append({
                    "param": "TDI_Stages", "requested": expected_stages,
                    "applied": tdi_stage, "status": "OK"
                })

        # G_TDIYShift is ALWAYS RegionHeight / TDI_stages — this is expected behaviour.
        # Flag it as INFO so the user understands it is not a mismatch.
        g_tdi_yshift  = params_applied.get("G_TDIYShift")
        region_height = params_applied.get("RegionHeight")
        tdi_yshift_req = d.get("tdi_yshift")
        if (g_tdi_yshift is not None and region_height is not None
                and expected_stages and expected_stages > 0):
            expected_g = region_height / expected_stages
            if abs(g_tdi_yshift - expected_g) < 1:
                checks.append({
                    "param": "G_TDIYShift",
                    "requested": f"{tdi_yshift_req} (RegionHeight={region_height})",
                    "applied": g_tdi_yshift,
                    "status": "OK",
                    "note": (
                        f"G_TDIYShift={g_tdi_yshift:.0f} = RegionHeight({region_height:.0f}) "
                        f"/ TDI_stages({expected_stages}) — this is correct and expected. "
                        f"The log warning 'TDIYShift greater than default' is informational only."
                    )
                })
            else:
                flag_fn("WARNING", "procmode_vs_applied",
                        f"[ProcMode check] G_TDIYShift={g_tdi_yshift:.0f} does not match "
                        f"expected RegionHeight/stages = {expected_g:.0f}. "
                        f"Unexpected TDI shift configuration.")

    # CCSDS check
    ccsds_requested = proc.get("binning_decoded", {}).get("ccsds_enabled")
    ccsds_applied   = params_applied.get("CCSDSProcessStatus")
    if ccsds_requested is not None and ccsds_applied is not None:
        if bool(ccsds_requested) != bool(ccsds_applied):
            flag_fn("WARNING", "procmode_vs_applied",
                    f"[ProcMode check] CCSDS: requested {'ON' if ccsds_requested else 'OFF'}, "
                    f"applied {'ON' if ccsds_applied else 'OFF'}.")

    return checks


def discover_band_files(folder: str, meta: Dict) -> List[Dict]:
    """Find all .bandN files and compute geometry per file."""
    w  = meta.get("width") or 8448
    h  = meta.get("region_height") or meta.get("height") or 384
    bd = meta.get("bit_depth") or 10
    bands = []
    re_b  = re.compile(r"\.band(\d+)$", re.I)

    for f in sorted(os.listdir(folder)):
        m = re_b.search(f)
        if not m:
            continue
        suffix = m.group(1)
        if len(suffix) == 1:
            w_eff, h_eff = w, h
        elif len(suffix) == 2:
            part = int(suffix[1])
            w_eff = w // 2
            h_eff = h // 2 if part == 2 else h
        else:
            continue

        fpath = os.path.join(folder, f)
        bpf   = bytes_per_frame(w_eff, h_eff, bd)
        if bpf == 0:
            continue
        n_frames = os.path.getsize(fpath) // bpf
        if n_frames == 0:
            continue

        bands.append({
            "key": f"b{suffix}", "path": fpath,
            "width": w_eff, "height": h_eff,
            "bit_depth": bd, "bpf": bpf, "n_frames": n_frames,
        })

    return bands


# ══════════════════════════════════════════════════════════════════════════════
# LOG ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

def analyze_logs(folder: str, progress_cb: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Deep parse of all .log/.txt files under folder.

    Returns a structured dict covering:
      - parameter_mismatches  (every requested vs applied value)
      - trigger_timing        (W01 stale timestamp)
      - frame_accounting      (drops, expected vs captured)
      - temperatures          (sensor + core before/after + delta)
      - fps_stability         (per-frame jitter analysis)
      - errors / warnings     (full inventory)
      - camera_init           (connection, settings path)
      - raw_issues            (ordered list of findings for Iris to report)
    """

    # ── Collect log lines ──────────────────────────────────────────────────
    combined_lines: List[str] = []
    log_files: List[str]      = []

    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if not f.lower().endswith(LOG_EXTS):
                continue
            try:
                if progress_cb:
                    try:
                        progress_cb(f"Reading log: {f}")
                    except Exception:
                        pass
                with open(os.path.join(root, f), errors="ignore") as lf:
                    combined_lines.extend(lf.readlines())
                log_files.append(f)
            except Exception:
                pass
        if len(combined_lines) > 150_000:
            break

    if not combined_lines:
        return {"found": False}

    result: Dict = {"found": True, "log_files": log_files, "raw_issues": []}

    def flag(severity: str, category: str, message: str):
        result["raw_issues"].append(
            {"severity": severity, "category": category, "message": message})

    # ── State containers ───────────────────────────────────────────────────
    errors:   List[str] = []
    warnings: List[str] = []
    fps_readings:  List[float] = []
    time_diffs:    List[float] = []

    params_requested: Dict[str, float] = {}
    params_applied:   Dict[str, float] = {}
    params_failed:    List[str]        = []

    # Raw argument tracking
    raw_args_line:    str  = ""
    raw_arg_count:    int  = 0
    proc_arg_count:   int  = 0   # from "Argument Processed[N]"
    max_exp_time:     Optional[float] = None   # MaxExpTime from log

    sensor_temps_before: List[float] = []
    sensor_temps_after:  List[float] = []
    core_temps_before:   List[float] = []
    core_temps_after:    List[float] = []
    grabber_temp_c: Optional[float] = None   # I99 — grabber board temp at hardware-info time
    capture_started = False

    trigger_timing: Dict = {}
    total_frames_log:   Optional[int] = None
    captured_count_log: Optional[int] = None
    frame_drops_log:    Optional[int] = None
    camera_settings_invalid = False
    camera_connected        = False
    camera_connect_failed   = False
    region_offsets: Dict[str, int] = {}
    test_pattern: Optional[int] = None
    firmware_version: Optional[str] = None

    # Extended system/hardware/post-capture fields
    app_version: Optional[str] = None
    disk_free_gb: Optional[float] = None
    memory_free_gb: Optional[float] = None
    cpu_model: Optional[str] = None
    cpu_cores: Optional[int] = None
    system_uptime: Optional[str] = None
    camera_model: Optional[str] = None
    grabber_model: Optional[str] = None
    json_config_file: Optional[str] = None
    pci_generation: Optional[int] = None
    pci_lanes: Optional[int] = None
    total_ram_mb: Optional[float] = None
    allocatable_ram_mb: Optional[float] = None
    max_fps: Optional[float] = None
    applied_height: Optional[int] = None
    region_modes: dict = None
    reversex_warn = False
    reversey_warn = False
    i06_ts: Optional[float] = None
    i96_ts: Optional[float] = None
    i07_date: Optional[str] = None
    i28_date: Optional[str] = None
    total_frames_expected: Optional[int] = None
    adcs_ephemeris: Optional[int] = None
    gps_ephemeris: Optional[int] = None
    data_proc_time_s: Optional[float] = None
    file_size_bytes: Optional[int] = None
    frame_size_bytes: Optional[int] = None
    computed_band_height: Optional[int] = None
    bin_status_value: Optional[int] = None
    default_fps_fail_msg: Optional[str] = None
    fps_set_passed: bool = False

    # ── Compiled patterns ──────────────────────────────────────────────────
    re_error        = re.compile(r"\[E[\w]*\]")
    re_warning      = re.compile(r"\[W[\w]*\]")
    re_set_passed   = re.compile(
        r"Set\s+([\w_]+)\s*=\s*([\d.]+)\s+Passed.*?Applied\s+[\w_\s]+=\s*([\d.]+)", re.I)
    re_set_failed   = re.compile(
        r"Set\s+([\w_]+)\s*=\s*([\d.]+)\s+Failed.*?Applied\s+[\w_\s]+=\s*([\d.]+)", re.I)
    re_applied_line = re.compile(r"\[I4[3-9]\]|\[I5[0-5]\]")  # applied confirmation block
    re_applied_kv   = re.compile(r"Applied\s+([\w_\s]+?)\s*=\s*([\d.]+)", re.I)
    re_frame        = re.compile(
        r"FrameNo=\s*(\d+)\s*(?:\([^)]*\))?\s*,\s*instantFps=([\d.]+).*?TimeDifference:\s*([\d.]+)")
    re_total        = re.compile(r"TotalNoOfFrames:\s*(\d+)", re.I)
    re_captured     = re.compile(r"CapturedCount:\s*(\d+)", re.I)
    re_drops        = re.compile(r"No\.of\s+Framedrops:\s*(\d+)", re.I)
    re_sensor_temp  = re.compile(r"\[I54\].*SensorTemp:\s*([\d.]+)", re.I)
    re_core_temp    = re.compile(r"Device\s+Core\s+Temperature:\s*(\d+)", re.I)
    re_utc_trigger  = re.compile(r"UTC Trigger Time:\s*(.+)")
    re_system_time  = re.compile(r"System Time Now:\s*(.+)")
    re_waiting_ms   = re.compile(r"Waiting Time\s*=\s*([-\d]+)\s*msec")
    re_w01_range    = re.compile(r"\[W01\].+out of range", re.I)
    re_region_off   = re.compile(r"Region(\d)OffsetY,\s*(\d+)")
    re_cam_invalid  = re.compile(r"Invalid Camera Settings Path", re.I)
    re_cam_ok       = re.compile(r"Camera was connected successfully", re.I)
    re_cam_fail     = re.compile(r"Auto Connect Failed|Camera not connected|No connection to grabber", re.I)
    re_test_pattern = re.compile(r"TestPattern[^0-9]*([0-9]+)", re.I)
    re_fw_1         = re.compile(r"Device_Firmware_Version\s*[:=]\s*([0-9.]+)", re.I)
    re_fw_2         = re.compile(r"Firmware\s*Version\s*[:=]\s*([0-9.]+)", re.I)
    re_fw_3         = re.compile(r"\bFirmware\s*[:=]\s*([0-9.]+)", re.I)

    # Raw argument extraction
    re_raw_args     = re.compile(r"Arguments received from parameter file:\s*(.+)", re.I)
    re_proc_count   = re.compile(r"Argument Processed\[(\d+)\]", re.I)
    re_max_exp      = re.compile(r"MaxExpTime=\s*([\d.]+)", re.I)

    # Extended patterns — system info, hardware, post-capture
    re_app_version  = re.compile(r"Cam App Version-([\d.]+)", re.I)
    re_disk_free    = re.compile(r"Disk free space:\s*([\d.]+)", re.I)
    re_memory       = re.compile(r"Memory available:\s*([\d.]+)", re.I)
    re_cpu_model    = re.compile(r"CPU Model:\s*(.+)")
    re_cpu_cores    = re.compile(r"CPU Cores:\s*(\d+)")
    re_uptime       = re.compile(r"Duration since System Up:\s*(.+)")
    re_cam_model    = re.compile(r"deviceModelName=\s*(\S+)", re.I)
    re_grabber_0    = re.compile(r"Grabber\[0\]:\s*(.+)")
    re_json_config  = re.compile(r"Configuration json file used:\s*(.+)")
    re_reversex_unk = re.compile(r"ReverseX.*unrecognized", re.I)
    re_reversey_unk = re.compile(r"ReverseY.*unrecognized", re.I)
    re_log_ts       = re.compile(r"^\[(\d+),(\d+)\.(\d+)\]")  # [SS,MMM.UUU] = secs,ms,us
    re_i06          = re.compile(r"\[I06\].*Autoconnect Initiated", re.I)
    re_i96          = re.compile(r"\[I96\].*Good connection to grabber", re.I)
    re_i07_date     = re.compile(r"Arguments received from parameter file:\s*\S+\s+\S+\s+\S+\s+(\d+\.\d+\.\d+)", re.I)
    re_i28_date     = re.compile(r"Argument Processed\[\d+\]:\s*\S+\s+\S+\s+\S+\s+(\d+\.\d+\.\d+)", re.I)
    re_i55_frames   = re.compile(r"Total no of frames:\s*(\d+)", re.I)
    re_adcs         = re.compile(r"ADCS Ephemeris Data Stored:\s*(\d+)", re.I)
    re_gps          = re.compile(r"GPS Ephemeris Data Stored:\s*(\d+)", re.I)
    re_data_proc    = re.compile(r"Total time took for data processing:\s*([\d.]+)", re.I)
    re_file_size    = re.compile(r"File_size=(\d+)\(FrameSize=(\d+)", re.I)
    re_band_height  = re.compile(r"Computed BAND_HEIGHT\s*=\s*(\d+)", re.I)
    re_bin_status   = re.compile(r"Bin Status\s*:\s*(\d+)", re.I)
    re_pci_gen      = re.compile(r"\[I100\].*Detected Device Pci Generation:\s*(\d+)", re.I)
    re_pci_lanes    = re.compile(r"\[I101\].*Detected Device Pci Lanes:\s*(\d+)", re.I)
    re_total_ram    = re.compile(r"\[I64\].*Total Ram Available:\s*([\d.]+)", re.I)
    re_alloc_ram    = re.compile(r"\[I65\].*Total Ram can Allocate:\s*([\d.]+)", re.I)
    re_max_fps      = re.compile(r"\[I37\].*MaxFPS=\s*([\d.]+)", re.I)
    re_applied_h    = re.compile(r"\[I46\].*Applied Height=(\d+)", re.I)
    re_region_mode  = re.compile(r"\[I43\].*Region(\d)Mode:(\d+)", re.I)
    re_default_fps_fail = re.compile(r"Failed to set default FPS.*Applied default FPS\s*=\s*([\d.]+)", re.I)

    # ── Parse lines ────────────────────────────────────────────────────────
    for raw_line in combined_lines:
        line = raw_line.strip()
        if not line:
            continue

        # Errors / warnings
        if re_error.search(line):
            msg = re.sub(r"^\[.*?\]\s*", "", line)
            if msg:
                msg = msg[:300]
                errors.append(msg)
                if re_default_fps_fail.search(msg):
                    default_fps_fail_msg = msg
        if re_warning.search(line):
            msg = re.sub(r"^\[.*?\]\s*", "", line)
            if msg: warnings.append(msg[:300])

        # Camera init
        if re_cam_invalid.search(line):
            camera_settings_invalid = True
        if re_cam_ok.search(line):
            camera_connected = True
        if re_cam_fail.search(line):
            camera_connect_failed = True

        # ── System info ────────────────────────────────────────────────────
        if app_version is None:
            m = re_app_version.search(line)
            if m: app_version = m.group(1).strip()
        if disk_free_gb is None:
            m = re_disk_free.search(line)
            if m: disk_free_gb = float(m.group(1))
        if memory_free_gb is None:
            m = re_memory.search(line)
            if m: memory_free_gb = float(m.group(1))
        if cpu_model is None:
            m = re_cpu_model.search(line)
            if m: cpu_model = m.group(1).strip()
        if cpu_cores is None:
            m = re_cpu_cores.search(line)
            if m: cpu_cores = int(m.group(1))
        if system_uptime is None:
            m = re_uptime.search(line)
            if m: system_uptime = m.group(1).strip()

        # ── Hardware ───────────────────────────────────────────────────────
        if camera_model is None:
            m = re_cam_model.search(line)
            if m: camera_model = m.group(1).strip()
        if grabber_model is None:
            m = re_grabber_0.search(line)
            if m and "I05" in line: grabber_model = m.group(1).strip()
        if json_config_file is None:
            m = re_json_config.search(line)
            if m: json_config_file = m.group(1).strip()
        if re_reversex_unk.search(line): reversex_warn = True
        if re_reversey_unk.search(line): reversey_warn = True

        # PCIe generation and lanes (I100, I101)
        if pci_generation is None:
            m = re_pci_gen.search(line)
            if m: pci_generation = int(m.group(1))
        if pci_lanes is None:
            m = re_pci_lanes.search(line)
            if m: pci_lanes = int(m.group(1))

        # RAM totals (I64, I65)
        if total_ram_mb is None:
            m = re_total_ram.search(line)
            if m: total_ram_mb = float(m.group(1))
        if allocatable_ram_mb is None:
            m = re_alloc_ram.search(line)
            if m: allocatable_ram_mb = float(m.group(1))

        # Max FPS from I37
        if max_fps is None:
            m = re_max_fps.search(line)
            if m: max_fps = float(m.group(1))

        # Applied Height from I46 (active pixel height after TDI + band config)
        if applied_height is None:
            m = re_applied_h.search(line)
            if m: applied_height = int(m.group(1))

        # Region modes from I43 (one line per band)
        m = re_region_mode.search(line)
        if m:
            if region_modes is None:
                region_modes = {}
            region_modes[f"region{m.group(1)}"] = int(m.group(2))

        # Grabber connection time (I06 → I96 elapsed)
        ts_m = re_log_ts.match(line)
        if ts_m:
            # [SS,MMM.UUU]: SS=whole seconds, MMM=milliseconds, UUU=microseconds
            ts_sec = (float(ts_m.group(1))
                      + float(ts_m.group(2)) / 1000.0
                      + float(ts_m.group(3)) / 1_000_000.0)
            if re_i06.search(line) and i06_ts is None:
                i06_ts = ts_sec
            if re_i96.search(line) and i96_ts is None:
                i96_ts = ts_sec

        # ── Parameter file date discrepancy ───────────────────────────────
        if i07_date is None:
            m = re_i07_date.search(line)
            if m and "I07" in line: i07_date = m.group(1).strip()
        if i28_date is None:
            m = re_i28_date.search(line)
            if m and "I28" in line: i28_date = m.group(1).strip()

        # ── Post-capture ───────────────────────────────────────────────────
        if total_frames_expected is None:
            m = re_i55_frames.search(line)
            if m: total_frames_expected = int(m.group(1))
        if adcs_ephemeris is None:
            m = re_adcs.search(line)
            if m: adcs_ephemeris = int(m.group(1))
        if gps_ephemeris is None:
            m = re_gps.search(line)
            if m: gps_ephemeris = int(m.group(1))
        if data_proc_time_s is None:
            m = re_data_proc.search(line)
            if m: data_proc_time_s = float(m.group(1))
        if file_size_bytes is None:
            m = re_file_size.search(line)
            if m:
                file_size_bytes  = int(m.group(1))
                frame_size_bytes = int(m.group(2))
        if computed_band_height is None:
            m = re_band_height.search(line)
            if m: computed_band_height = int(m.group(1))
        if bin_status_value is None:
            m = re_bin_status.search(line)
            if m: bin_status_value = int(m.group(1))

        # Raw argument line — capture once, count tokens
        if not raw_args_line:
            m = re_raw_args.search(line)
            if m:
                raw_args_line  = m.group(1).strip()
                raw_arg_count  = len(raw_args_line.split())

        # Argument Processed count
        m = re_proc_count.search(line)
        if m:
            proc_arg_count = int(m.group(1))

        # MaxExpTime (inside the Set Exposure line)
        m = re_max_exp.search(line)
        if m and max_exp_time is None:
            max_exp_time = float(m.group(1))

        # Trigger timing
        m = re_utc_trigger.search(line)
        if m:
            trigger_timing.setdefault("utc_trigger_times", []).append(m.group(1).strip())
        m = re_system_time.search(line)
        if m:
            trigger_timing.setdefault("system_times", []).append(m.group(1).strip())
        m = re_waiting_ms.search(line)
        if m:
            wms = int(m.group(1))
            if "waiting_time_msec" not in trigger_timing:
                trigger_timing["waiting_time_msec"] = wms   # first (stale, large negative)
            else:
                trigger_timing["waiting_time_msec_2"] = wms  # second (valid, small positive)
        if re_w01_range.search(line):
            trigger_timing["stale_timestamp_detected"] = True

        # Set Passed / Failed
        m = re_set_passed.search(line)
        if m:
            p, req, app = m.group(1).strip(), float(m.group(2)), float(m.group(3))
            params_requested[p] = req
            params_applied[p]   = app
            if p.upper() == "FPS":
                fps_set_passed = True
        m = re_set_failed.search(line)
        if m:
            p, req, app = m.group(1).strip(), float(m.group(2)), float(m.group(3))
            params_requested[p] = req
            params_applied[p]   = app
            if p not in params_failed:
                params_failed.append(p)

        # Applied block (confirmation lines I43-I55)
        if "Applied" in line and "=" in line:
            m = re_applied_kv.search(line)
            if m:
                p = m.group(1).strip().replace(" ", "_")
                v = float(m.group(2))
                params_applied[p] = v

        # Region offsets
        m = re_region_off.search(line)
        if m:
            region_offsets[f"Region{m.group(1)}OffsetY"] = int(m.group(2))

        # Test pattern
        if test_pattern is None:
            m = re_test_pattern.search(line)
            if m:
                try:
                    test_pattern = int(m.group(1))
                except Exception:
                    pass

        # Firmware version
        if firmware_version is None:
            for re_fw in (re_fw_1, re_fw_2, re_fw_3):
                m = re_fw.search(line)
                if m:
                    firmware_version = m.group(1).strip()
                    break

        # Per-frame data
        m = re_frame.search(line)
        if m:
            capture_started = True
            fps_readings.append(float(m.group(2)))
            td = float(m.group(3))
            if td != 0:
                time_diffs.append(td)

        # Frame accounting
        m = re_total.search(line)
        if m: total_frames_log = int(m.group(1))
        m = re_captured.search(line)
        if m: captured_count_log = int(m.group(1))
        m = re_drops.search(line)
        if m: frame_drops_log = int(m.group(1))

        # Temperature (before/after capture start)
        # Only trust the tagged capture readings:
        # I54 = sensor temp; I87 = core temp (pre/post capture); I99 = grabber board temp at init.
        # Ignore the final generic status dump lines like "[I] SensorTemp: 318246".
        m = re_sensor_temp.search(line)
        if m:
            t = float(m.group(1))
            (sensor_temps_after if capture_started else sensor_temps_before).append(t)
        m = re_core_temp.search(line)
        if m:
            t = float(m.group(1))
            if "[I99]" in line:
                # I99 is the grabber board temperature read during hardware-info phase.
                # Store it separately — do NOT mix it into the session before/after lists.
                if grabber_temp_c is None:
                    grabber_temp_c = t
            else:
                # I87 = real session core temp (before and after capture)
                (core_temps_after if capture_started else core_temps_before).append(t)

    # ══════════════════════════════════════════════════════════════════════
    # BUILD FINDINGS
    # ══════════════════════════════════════════════════════════════════════

    # ── Camera init ────────────────────────────────────────────────────────
    if camera_settings_invalid:
        flag("WARNING", "camera_init",
             "Camera settings ZIP path was invalid — default camera config was used. "
             "Custom calibration or factory settings may not have been applied.")
    if camera_connect_failed:
        flag("CRITICAL", "camera_init",
             "Auto-connect failure detected in log: camera did not connect (explicit failure path).")
    elif not camera_connected:
        flag("CRITICAL", "camera_init",
             "No log confirmation that camera connected successfully.")

    # ── Trigger timing ─────────────────────────────────────────────────────
    if trigger_timing.get("stale_timestamp_detected"):
        utc_times = trigger_timing.get("utc_trigger_times", [])
        sys_times  = trigger_timing.get("system_times", [])
        wait_ms    = trigger_timing.get("waiting_time_msec", 0)
        utc_str    = utc_times[0] if utc_times else "unknown"
        sys_str    = sys_times[0] if sys_times else "unknown"
        flag("WARNING", "trigger_timing",
             f"[W01] Stale UTC trigger timestamp in parameter file. "
             f"Parameter file says: {utc_str} | "
             f"Actual system time: {sys_str} | "
             f"Delta was {abs(wait_ms):,} ms — out of valid range. "
             f"System fell back to 5-second default delay. "
             f"GPS/orbital time sync did NOT occur. "
             f"Any mission-critical timing metadata attached to this dataset is unreliable.")
    result["trigger_timing"] = trigger_timing

    # ── Parameter file date discrepancy ───────────────────────────────────
    if i07_date and i28_date and i07_date != i28_date:
        try:
            from datetime import datetime
            def _parse_dmy(s):
                # DD.MM.YYYY
                d, mo, y = s.split(".")
                return datetime(int(y), int(mo), int(d))
            d07 = _parse_dmy(i07_date)
            d28 = _parse_dmy(i28_date)
            days_diff = abs((d28 - d07).days)
            param_date_discrepancy_days = days_diff
            if days_diff > 30:
                flag("WARNING", "param_date_stale",
                     f"Parameter file date is stale — I07 date: {i07_date}, "
                     f"processed date: {i28_date} (discrepancy: {days_diff} days). "
                     f"The firmware auto-corrected to system date, but the param file "
                     f"should be updated before the next acquisition.")
        except Exception:
            pass

    # ── Grabber connection timing ──────────────────────────────────────────
    if i06_ts is not None and i96_ts is not None:
        grabber_connect_time_s = round(i96_ts - i06_ts, 3)
        if grabber_connect_time_s > 10.0:
            flag("WARNING", "grabber_slow_connect",
                 f"Grabber connection took {grabber_connect_time_s:.2f}s "
                 f"(normal <2s). Check PCIe slot, cable, or power cycle the camera.")
        elif grabber_connect_time_s > 2.0:
            flag("INFO", "grabber_slow_connect",
                 f"Grabber connection took {grabber_connect_time_s:.2f}s (slightly above normal <2s).")

    # ── ReverseX/Y unrecognized parameters ────────────────────────────────
    if reversex_warn or reversey_warn:
        params = []
        if reversex_warn: params.append("ReverseX")
        if reversey_warn: params.append("ReverseY")
        flag("INFO", "unrecognized_params",
             f"JSON config parameters {', '.join(params)} not recognised by this "
             f"firmware version — set/get commands were silently ignored. "
             f"Image flip settings may not have been applied.")

    # ── Parameter mismatches ───────────────────────────────────────────────
    mismatches = []

    # Early default-stage FPS failure can be recovered by the later user-setting path.
    # If the requested capture FPS was subsequently applied successfully, suppress the
    # transient default error and replace it with an informational recovery note.
    fps_req = params_requested.get("FPS")
    fps_app = params_applied.get("FPS")
    fps_recovered = (
        default_fps_fail_msg is not None
        and fps_set_passed
        and fps_req is not None
        and fps_app is not None
        and abs(fps_app - fps_req) / max(abs(fps_req), 0.001) <= 0.02
    )
    if fps_recovered:
        errors = [e for e in errors if e != default_fps_fail_msg]
        params_failed = [p for p in params_failed if p != "FPS"]
        flag("INFO", "default_fps_recovered",
             f"Default FPS setup failed during initialization, but the requested capture FPS "
             f"was later applied successfully: requested {fps_req} → applied {fps_app:.3f}. "
             f"The early default-stage FPS error was recovered before capture started.")

    # Parameters that explicitly failed
    for p in params_failed:
        req = params_requested.get(p)
        app = params_applied.get(p)
        mismatches.append({"param": p, "requested": req, "applied": app, "status": "FAILED"})
        flag("CRITICAL", "parameter_mismatch",
             f"Parameter FAILED: {p} — requested {req}, hardware applied {app}. "
             f"This setting was rejected by the camera.")

    # Parameters that passed but were silently changed (>1% difference)
    for p, req in params_requested.items():
        if p in params_failed:
            continue
        app = params_applied.get(p)
        if app is None or req == 0:
            continue
        diff_pct = abs(app - req) / abs(req)
        if diff_pct > PARAM_TOLERANCE:
            mismatches.append({
                "param": p, "requested": req, "applied": app,
                "diff_pct": round(diff_pct * 100, 1), "status": "AUTO_ADJUSTED"})
            flag("WARNING", "parameter_mismatch",
                 f"Parameter auto-adjusted: {p} — requested {req}, "
                 f"hardware applied {app} ({diff_pct*100:.1f}% difference). "
                 f"Hardware capped or modified this value.")

    result["parameter_mismatches"] = mismatches
    result["parameters_requested"] = params_requested
    result["parameters_applied"]   = params_applied
    result["region_offsets"]       = region_offsets
    if test_pattern is not None:
        result["test_pattern"] = test_pattern
    if firmware_version:
        result["firmware_version"] = firmware_version

    # ── Extended fields ────────────────────────────────────────────────────
    result["system_info"] = {
        "app_version":       app_version,
        "disk_free_gb":      disk_free_gb,
        "memory_free_gb":    memory_free_gb,
        "total_ram_mb":      total_ram_mb,
        "allocatable_ram_mb":allocatable_ram_mb,
        "cpu_model":         cpu_model,
        "cpu_cores":         cpu_cores,
        "system_uptime":     system_uptime,
    }
    result["hardware_info"] = {
        "camera_model":           camera_model,
        "grabber_model":          grabber_model,
        "json_config_file":       json_config_file,
        "reversex_unrecognized":  reversex_warn,
        "reversey_unrecognized":  reversey_warn,
        "pci_generation":         pci_generation,
        "pci_lanes":              pci_lanes,
        "max_fps":                max_fps,
        "applied_height":         applied_height,
        "region_modes":           region_modes or {},
        "grabber_connect_time_s": (
            round(i96_ts - i06_ts, 1)
            if (i06_ts is not None and i96_ts is not None)
            else None
        ),
        "grabber_temp_c":         grabber_temp_c,  # I99 board temp at init
    }
    result["capture_info"] = {
        "total_frames_expected": total_frames_expected,
        "file_size_bytes":       file_size_bytes,
        "frame_size_bytes":      frame_size_bytes,
        "computed_band_height":  computed_band_height,
        "adcs_ephemeris_count":  adcs_ephemeris,
        "gps_ephemeris_count":   gps_ephemeris,
        "data_proc_time_s":      data_proc_time_s,
        "max_exp_time_us":       max_exp_time,
        "bin_status":            bin_status_value,
    }
    result["param_date_info"] = {
        "i07_date":        i07_date,
        "i28_date":        i28_date,
        "discrepancy_days": abs((
            __import__("datetime").datetime.strptime(i28_date, "%d.%m.%Y") -
            __import__("datetime").datetime.strptime(i07_date, "%d.%m.%Y")
        ).days) if (i07_date and i28_date and i07_date != i28_date) else 0,
    }

    # ── Argument integrity check ───────────────────────────────────────────
    result["raw_args_line"]  = raw_args_line
    result["raw_arg_count"]  = raw_arg_count
    result["proc_arg_count"] = proc_arg_count

    EXPECTED_ARG_COUNT = 14
    if raw_args_line:
        if raw_arg_count < EXPECTED_ARG_COUNT:
            flag("CRITICAL", "argument_integrity",
                 f"Only {raw_arg_count} of {EXPECTED_ARG_COUNT} arguments were parsed from "
                 f"the parameter file (got: \"{raw_args_line}\"). "
                 f"Missing fields: {EXPECTED_ARG_COUNT - raw_arg_count} argument(s) at the end. "
                 f"All parameters after position {raw_arg_count} default to unknown values. "
                 f"This acquisition's configuration is UNVERIFIED — the parameter file is "
                 f"corrupt or truncated. Investigate the mission planning system.")
        elif proc_arg_count != EXPECTED_ARG_COUNT:
            flag("WARNING", "argument_integrity",
                 f"Parameter file had {raw_arg_count} raw tokens but log reports "
                 f"Argument Processed[{proc_arg_count}] — expected Processed[{EXPECTED_ARG_COUNT}]. "
                 f"Parsing discrepancy detected.")
        else:
            result["argument_integrity"] = "OK — all 14 arguments received and processed"

    # ── Test pattern ───────────────────────────────────────────────────────
    if test_pattern is not None:
        if test_pattern != 0:
            flag("INFO", "test_pattern",
                 f"Test pattern is enabled (TestPattern={test_pattern}). "
                 f"Frames may not represent real scene data.")
        else:
            flag("INFO", "test_pattern",
                 "Test pattern is OFF (TestPattern=0).")

    # ── Firmware version ───────────────────────────────────────────────────
    if firmware_version:
        if firmware_version == "2.2.2":
            flag("INFO", "firmware",
                 "Firmware 2.2.2 detected — stable, safe baseline.")
        elif firmware_version == "2.3.2":
            flag("WARNING", "firmware",
                 "Firmware 2.3.2 detected — known issue: temperature values can report very low. "
                 "Treat temperature readings with caution.")
        elif firmware_version == "2.4.2":
            flag("OK", "firmware",
                 "Firmware 2.4.2 detected — current stable version.")
        else:
            flag("INFO", "firmware",
                 f"Firmware version detected: {firmware_version}.")

    # ── Exposure clamp detection ───────────────────────────────────────────
    # The log prints "Passed..." even when the camera silently clamps to MaxExpTime.
    # Cross-check requested vs applied and flag if >1% AND max_exp_time is known.
    exp_req  = params_requested.get("Exposure_Time") or params_requested.get("ExposureTime")
    exp_app  = params_applied.get("Exposure_Time") or params_applied.get("ExposureTime")
    if exp_req is not None and exp_app is not None and exp_req > 0:
        exp_diff_pct = abs(exp_app - exp_req) / exp_req
        if exp_diff_pct > 0.01:   # more than 1% deviation
            clamp_note = ""
            if max_exp_time is not None and exp_req > max_exp_time:
                clamp_ratio = exp_req / max_exp_time
                clamp_note = (
                    f" The log shows MaxExpTime={max_exp_time:.1f}µs — requested value "
                    f"was {clamp_ratio:.0f}× over the hardware limit. "
                    f"The camera silently clamped to MaxExpTime despite logging 'Passed'. "
                    f"Do NOT rely on 'Passed' as confirmation of the requested value."
                )
            flag("WARNING", "exposure_clamp",
                 f"Exposure time silently adjusted: requested {exp_req:.1f}µs → "
                 f"applied {exp_app:.1f}µs ({exp_diff_pct*100:.1f}% difference).{clamp_note}")

    # ── FPS cap detection ──────────────────────────────────────────────────
    fps_req = params_requested.get("FPS")
    fps_app = params_applied.get("FPS")
    fps_cap_detected = any("Updated FPS from" in ln for ln in combined_lines)
    if fps_cap_detected and fps_req is not None and fps_app is not None:
        flag("INFO", "fps_capped",
             f"FPS was silently capped: requested {fps_req} → applied {fps_app:.4f} "
             f"(hardware maximum for this TDI mode). "
             f"Update the parameter file to use {fps_app:.2f} or lower to avoid this.")

    # ── Binning verification gap ───────────────────────────────────────────
    # The log has no "Applied Binning=" confirmation line — it's requested but never echoed.
    # Flag this so the user knows binning cannot be verified from the log alone.
    if bin_status_value is None:
        flag("INFO", "binning_unverified",
             "Binning argument was received (position 13 in parameter file) but the log "
             "does NOT contain an 'Applied Binning=' or 'Bin Status' confirmation line. "
             "Binning value cannot be verified from the log. "
             "This is a log coverage gap — add a binning confirmation log line in firmware "
             "if binning verification is required.")
    # The ProcMode string is in the JSON — passed via meta, but also sometimes
    # appears in the log argument line. Parse whichever we find.
    proc_mode_str = None
    for line in combined_lines:
        m = re.search(r"Argument Processed\[\d+\]:\s*(.+)", line)
        if m:
            proc_mode_str = m.group(1).strip()
            break
    if not proc_mode_str:
        # Fall back: look for the raw parameter line format
        for line in combined_lines:
            m = re.search(r"Arguments received from parameter file:\s*(.+)", line)
            if m:
                proc_mode_str = m.group(1).strip()
                break

    if proc_mode_str:
        proc = decode_procmode(proc_mode_str)
        result["procmode"] = proc
        if not proc.get("parse_error"):
            checks = verify_procmode_vs_applied(
                proc, params_applied, params_requested, params_failed, flag)
            result["procmode_checks"] = checks
            # Build a human-readable summary of what was requested
            d = proc["decoded"]
            tdi_info = proc["tdi_decoded"]
            bn_info  = proc["binning_decoded"]
            result["procmode_summary"] = (
                f"Orbit {d.get('orbit_id')} / Task {d.get('task_id')} | "
                f"UTC trigger: {d.get('date')} {d.get('utc_time')} | "
                f"Duration: {d.get('duration_sec')}s | "
                f"FPS: {d.get('fps_requested')} | "
                f"Exposure: {d.get('exposure_time')}ms | "
                f"Gain: {d.get('gain')} | "
                f"TDI: {tdi_info.get('mode')} | "
                f"XShift: {d.get('xshift')} | "
                f"TDIYShift: {d.get('tdi_yshift')} | "
                f"CCSDS: {'ON' if bn_info.get('ccsds_enabled') else 'OFF'} | "
                f"Bands active: {proc['band_selection'].get('active_count', 7)}"
            )
    else:
        result["procmode"] = {"parse_error": "ProcMode string not found in log"}
        result["procmode_summary"] = "ProcMode not found in log"

    # ── Frame accounting ───────────────────────────────────────────────────
    frame_accounting: Dict = {
        "total_frames_expected": total_frames_log,
        "captured_count":        captured_count_log,
        "frame_drops_reported":  frame_drops_log,
        "frames_in_log":         len(fps_readings),
    }
    if total_frames_log is not None and captured_count_log is not None:
        lost = total_frames_log - captured_count_log
        frame_accounting["frames_lost"] = lost
        if lost > 0:
            pct = lost / total_frames_log * 100
            flag("CRITICAL", "frame_drops",
                 f"{lost} frames lost ({pct:.1f}%) — "
                 f"expected {total_frames_log}, captured {captured_count_log}. "
                 f"Check CPU load and DMA buffer settings during capture.")
        else:
            frame_accounting["result"] = "PERFECT — zero frame drops"
    if frame_drops_log is not None and frame_drops_log > 0:
        flag("CRITICAL", "frame_drops",
             f"Log explicitly reports {frame_drops_log} frame drop(s).")
    result["frame_accounting"] = frame_accounting

    # ── Temperature ────────────────────────────────────────────────────────
    temp_report: Dict = {}

    if sensor_temps_before:
        temp_report["sensor_before_C"] = round(statistics.mean(sensor_temps_before), 2)
    if sensor_temps_after:
        temp_report["sensor_after_C"]  = round(statistics.mean(sensor_temps_after), 2)
    if core_temps_before:
        temp_report["core_before_C"]   = round(statistics.mean(core_temps_before), 1)
    if core_temps_after:
        temp_report["core_after_C"]    = round(statistics.mean(core_temps_after), 1)

    sb = temp_report.get("sensor_before_C")
    sa = temp_report.get("sensor_after_C")
    if sb is not None and sa is not None:
        delta = abs(sa - sb)
        temp_report["sensor_delta_C"] = round(delta, 2)
        temp_report["sensor_stability"] = "STABLE" if delta <= TEMP_DELTA_FLAG else "DRIFTING"
        if delta > TEMP_DELTA_FLAG:
            flag("WARNING", "temperature",
                 f"Sensor temperature drifted {delta:.1f}°C during capture "
                 f"({sb}°C → {sa}°C). May affect radiometric calibration.")

    cb = temp_report.get("core_before_C")
    ca = temp_report.get("core_after_C")
    if cb is not None and ca is not None:
        delta = abs(ca - cb)
        temp_report["core_delta_C"]    = round(delta, 1)
        temp_report["core_stability"]  = "STABLE" if delta <= TEMP_DELTA_FLAG else "DRIFTING"
        if delta > TEMP_DELTA_FLAG:
            flag("WARNING", "temperature",
                 f"Core temperature drifted {delta:.1f}°C during capture "
                 f"({cb}°C → {ca}°C). Check thermal management.")

    # Firmware-specific temperature caution
    if firmware_version == "2.3.2":
        low_sensor = (sb is not None and sb < 0) or (sa is not None and sa < 0)
        low_core   = (cb is not None and cb < 0) or (ca is not None and ca < 0)
        if low_sensor or low_core:
            flag("WARNING", "temperature",
                 "Temperature appears very low — this is a known firmware 2.3.2 issue. "
                 "Do not trust these temperature values.")
        else:
            flag("WARNING", "temperature",
                 "Firmware 2.3.2 detected — temperature readings can be unreliable.")

    result["temperatures"] = temp_report

    # ── FPS stability ──────────────────────────────────────────────────────
    fps_report: Dict = {}
    if fps_readings:
        fps_report["mean_fps"]      = round(statistics.mean(fps_readings), 4)
        fps_report["min_fps"]       = round(min(fps_readings), 4)
        fps_report["max_fps"]       = round(max(fps_readings), 4)
        fps_report["std_fps"]       = round(statistics.pstdev(fps_readings), 4)
        fps_report["frames_logged"] = len(fps_readings)

    if time_diffs:
        std_td = statistics.pstdev(time_diffs)
        fps_report["time_diff_mean_ms"] = round(statistics.mean(time_diffs), 4)
        fps_report["time_diff_std_ms"]  = round(std_td, 4)
        fps_report["timing_stability"]  = (
            "PERFECT"   if std_td < 0.01 else
            "EXCELLENT" if std_td < 0.1  else
            "GOOD"      if std_td < 0.5  else
            "UNSTABLE")
        if fps_report["timing_stability"] == "UNSTABLE":
            flag("WARNING", "fps_stability",
                 f"Unstable frame timing — TimeDifference σ={std_td:.3f}ms. "
                 f"Frame intervals are inconsistent. Check CPU load during capture.")

    result["fps_stability"] = fps_report

    def _log_error_severity(err: str) -> str:
        txt = (err or "").lower()
        if "greater than default" in txt:
            return "WARNING"
        return "CRITICAL"

    # ── Error/warning inventory ────────────────────────────────────────────
    result["errors"]        = errors[:50]
    result["warnings"]      = warnings[:50]
    result["error_count"]   = len(errors)
    result["warning_count"] = len(warnings)

    # Flag non-init errors that weren't already captured
    already_flagged = {"Invalid Camera Settings Path"}
    for err in errors[:5]:
        if not any(af in err for af in already_flagged):
            flag(_log_error_severity(err), "log_error", f"Log error: {err[:200]}")

    # ── Log health score ───────────────────────────────────────────────────
    n_crit = sum(1 for i in result["raw_issues"] if i["severity"] == "CRITICAL")
    n_warn = sum(1 for i in result["raw_issues"] if i["severity"] == "WARNING")
    result["log_health_score"] = round(max(0.0, 100.0 - n_crit * 15 - n_warn * 5), 1)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# FRAME-LEVEL PIXEL DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

def _find_isolated_outlier_pixels(
    arr: "np.ndarray",
    direction: str,          # "low" for dead, "high" for hot
    min_diff_dn: float,
    neighbor_sigma: float,
    max_neighbor_var: float,
) -> List[tuple]:
    """
    Find isolated single pixels that are anomalously low (dead) or high (hot)
    compared to their 8 immediate neighbours.

    DEAD PIXEL rules:
    - The pixel value is significantly LOWER than its neighbours
    - Its neighbours are NOT similarly low (i.e. it is isolated)
    - If the pixel's value is similar to its neighbours → it's part of a group → NOT dead
    - If ≥ 2 neighbours are ALSO independently flagged → part of a cluster → NOT dead

    HOT PIXEL rules:
    - The pixel value is significantly HIGHER than its neighbours
    - Its neighbours are NOT similarly high
    - Same group exclusion applies

    GROUP EXCLUSION (most important):
    1. VALUE similarity: if the pixel value is within GROUP_SIMILARITY_DN of the
       neighbour mean, it is part of a uniform region — NOT flagged regardless of
       absolute DN level. This catches low-DN uniform regions that are NOT dead pixels.
    2. FLAGGED neighbour count: if ≥ 2 neighbours are also independently flagged
       as outliers, it is part of a cluster — NOT flagged.

    Returns list of (row, col) tuples — isolated defect pixel positions only.
    """
    if not _HAS_NP or arr is None:
        return []

    h, w = arr.shape
    if h < 3 or w < 3:
        return []

    p = np.pad(arr, 1, mode="edge").astype(np.float32)

    # 8-neighbour sum and sum-of-squares (excluding centre)
    n1 = p[0:-2, 0:-2]; n2 = p[0:-2, 1:-1]; n3 = p[0:-2, 2:]
    n4 = p[1:-1, 0:-2];                      n5 = p[1:-1, 2:]
    n6 = p[2:,   0:-2]; n7 = p[2:,   1:-1]; n8 = p[2:,   2:]

    nsum    = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8
    nsum_sq = n1**2 + n2**2 + n3**2 + n4**2 + n5**2 + n6**2 + n7**2 + n8**2

    nmean = nsum / 8.0
    nvar  = np.maximum(0.0, (nsum_sq / 8.0) - nmean ** 2)
    nstd  = np.sqrt(nvar)

    a    = arr.astype(np.float32)
    diff = nmean - a if direction == "low" else a - nmean

    # ── GROUP EXCLUSION GATE 1: value similarity ───────────────────────────
    # If the pixel is SIMILAR in value to its neighbours, it is part of a
    # uniform region (e.g. all neighbours are also dark/bright) — NOT a defect.
    # Only pixels that STAND OUT from their local neighbourhood are candidates.
    similar_to_neighbours = diff < GROUP_SIMILARITY_DN
    # Pixels similar to their neighbours are NOT defects, exclude them
    not_group_member = ~similar_to_neighbours

    # ── MAIN THRESHOLD ────────────────────────────────────────────────────
    # Pixel must differ from neighbour mean by absolute DN AND sigma multiple
    thresh_abs   = diff >= min_diff_dn
    thresh_sigma = diff >= (neighbor_sigma * nstd + 1e-6)

    # Note: we removed the max_neighbor_var gate because it was suppressing
    # valid detections on gradient images. The GROUP_SIMILARITY_DN gate above
    # is the correct way to handle gradients and ramps — if the pixel is
    # similar to its neighbours, it is part of the gradient, not a defect.

    candidate_mask = thresh_abs & thresh_sigma & not_group_member

    if not candidate_mask.any():
        return []

    # ── GROUP EXCLUSION GATE 2: flagged neighbour count ───────────────────
    # If ≥ 2 of a pixel's neighbours are also flagged as outliers,
    # it is part of a cluster/group → not an isolated single-pixel defect.
    cm = candidate_mask.astype(np.float32)
    cp = np.pad(cm, 1, mode="constant", constant_values=0)
    neighbor_flag_count = (
        cp[0:-2, 0:-2] + cp[0:-2, 1:-1] + cp[0:-2, 2:] +
        cp[1:-1, 0:-2] +                   cp[1:-1, 2:] +
        cp[2:,   0:-2] + cp[2:,   1:-1] + cp[2:,   2:]
    )

    # At most ISOLATED_MAX_FLAGGED_NEIGHBORS neighbours also flagged → isolated
    # 2+ flagged neighbours → cluster/region → not a single-pixel defect
    isolated = candidate_mask & (neighbor_flag_count <= ISOLATED_MAX_FLAGGED_NEIGHBORS)

    rows, cols = np.where(isolated)
    return list(zip(rows.tolist(), cols.tolist()))


def analyze_frame(arr, max_dn: int = 1023, is_dark_capture: bool = False) -> Dict:
    """
    Analyse one decoded frame. Returns per-frame flags + stats.

    Dead pixels:  isolated pixels significantly BELOW their neighbours.
    Hot pixels:   isolated pixels significantly ABOVE their neighbours.
    Groups/clusters of similar-valued pixels are NOT flagged (they are
    scene content or calibration patterns, not single-pixel defects).

    Stuck pixels are NOT detected here — they require cross-frame analysis.
    See analyze_band_stuck_pixels() which runs after all frames are collected.
    """
    if not _HAS_NP or arr is None:
        return {}

    h, w  = arr.shape
    total = h * w
    mean  = float(arr.mean())
    flags = []
    risk  = 0.0

    # ── Black frame ────────────────────────────────────────────────────────
    if mean < BLACK_MEAN:
        flags.append("black_frame")
        risk += 0.3

    # ── Signal gate ────────────────────────────────────────────────────────
    has_signal = mean >= MIN_SIGNAL_DN and not is_dark_capture

    # ── Dead columns ───────────────────────────────────────────────────────
    col_means = arr.mean(axis=0)
    dead_cols: list = []
    if has_signal:
        dead_cols = [int(i) for i in np.where(col_means < DEAD_DN)[0][:20]]
        if dead_cols:
            flags.append(f"dead_columns:{len(dead_cols)}")
            risk += 0.3

    # ── Alternating row banding ────────────────────────────────────────────
    even_mean = odd_mean = None
    alt_diff  = 0.0
    if h >= 4:
        even_mean = float(arr[0::2, :].mean())
        odd_mean  = float(arr[1::2, :].mean())
        alt_diff  = abs(even_mean - odd_mean)
        if has_signal and alt_diff > ALT_ROW_THRESHOLD:
            flags.append(f"alternating_row_banding:diff={alt_diff:.1f}DN")
            risk += 0.2

    # ── Striping patterns ──────────────────────────────────────────────────
    col_var = float(np.var(col_means))
    row_var = float(np.var(arr.mean(axis=1)))
    ratio   = col_var / (row_var + 1e-9) if row_var > 0 else 0.0
    if has_signal and row_var > 0:
        if ratio > STRIPE_RATIO:
            flags.append("vertical_striping")
            risk += 0.15
        elif ratio < (1.0 / STRIPE_RATIO):
            flags.append("horizontal_striping")
            risk += 0.15
        else:
            global_var = float(np.var(arr))
            if global_var > 0 and (col_var + row_var) / global_var > 0.8:
                flags.append("striping_pattern")
                risk += 0.1

    # ── Diagonal stripe hint ───────────────────────────────────────────────
    if has_signal and h >= 16 and w >= 16:
        try:
            ds = arr[::4, ::4].astype(np.float32)
            gx = np.zeros_like(ds)
            gy = np.zeros_like(ds)
            gx[:, 1:-1] = ds[:, 2:] - ds[:, :-2]
            gy[1:-1, :] = ds[2:, :] - ds[:-2, :]
            mag = np.hypot(gx, gy)
            if mag.mean() > 0:
                ang  = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
                diag = ((ang > 30) & (ang < 60)) | ((ang > 120) & (ang < 150))
                if diag.mean() > 0.6:
                    flags.append("diagonal_striping")
                    risk += 0.1
        except Exception:
            pass

    # ── Saturation ─────────────────────────────────────────────────────────
    sat_pct = float((arr >= max_dn * 0.98).sum() / total * 100)
    if has_signal and sat_pct > 5:
        flags.append(f"saturation:{sat_pct:.1f}%")
        risk += 0.1

    # ── Dead pixels (isolated, below neighbours) ───────────────────────────
    dead_pixels: List[tuple] = []
    if has_signal:
        dead_pixels = _find_isolated_outlier_pixels(
            arr, "low",
            min_diff_dn    = DEAD_PIXEL_MIN_DIFF_DN,
            neighbor_sigma = DEAD_PIXEL_NEIGHBOR_SIGMA,
            max_neighbor_var = DEAD_PIXEL_MAX_NEIGHBOR_VAR,
        )
        if dead_pixels:
            flags.append(f"dead_pixels:{len(dead_pixels)}")
            risk += min(0.3, len(dead_pixels) * 0.02)

    # ── Hot pixels (isolated, above neighbours) ────────────────────────────
    hot_pixels: List[tuple] = []
    if has_signal:
        adaptive_min_diff = max(HOT_PIXEL_MIN_DIFF_DN, mean * 0.25)
        hot_pixels = _find_isolated_outlier_pixels(
            arr, "high",
            min_diff_dn    = adaptive_min_diff,
            neighbor_sigma = HOT_PIXEL_NEIGHBOR_SIGMA,
            max_neighbor_var = HOT_PIXEL_MAX_NEIGHBOR_VAR,
        )
        if hot_pixels:
            flags.append(f"hot_pixels:{len(hot_pixels)}")
            risk += min(0.3, len(hot_pixels) * 0.02)

    return {
        "mean":          mean,
        "std":           float(arr.std()),
        "max":           float(arr.max()),
        "even_row_mean": even_mean,
        "odd_row_mean":  odd_mean,
        "alt_diff":      alt_diff,
        "dead_cols":     dead_cols,
        "dead_pixels":   dead_pixels,      # list of (row, col)
        "hot_pixels":    hot_pixels,       # list of (row, col)
        "sat_pct":       sat_pct,
        "flags":         flags,
        "risk":          min(1.0, risk),
        "has_signal":    has_signal,
    }


def analyze_band_stuck_pixels(
    frames: List["np.ndarray"],
    max_dn: int = 1023,
) -> List[Dict]:
    """
    Detect stuck pixels across all frames of a band.

    A stuck pixel:
    - Has nearly constant DN value across ALL frames (std ≤ STUCK_PIXEL_MAX_STD)
    - Is isolated: surrounding pixels DO change across frames (scene std ≥ STUCK_PIXEL_MIN_SCENE_STD)
    - NOT flagged if its neighbours are also stuck — that is a dead region, not a single stuck pixel

    Returns list of dicts: {row, col, value, pixel_std, neighbor_scene_std}
    """
    if not _HAS_NP or len(frames) < STUCK_PIXEL_MIN_FRAMES:
        return []

    try:
        stack = np.stack(frames, axis=0).astype(np.float32)  # (F, H, W)
    except Exception:
        return []

    F, H, W = stack.shape
    if H < 3 or W < 3:
        return []

    # Per-pixel std across frames
    pixel_std = stack.std(axis=0)   # (H, W)

    # Candidate stuck pixels: nearly zero variation
    stuck_cand = pixel_std <= STUCK_PIXEL_MAX_STD

    if not stuck_cand.any():
        return []

    # Compute neighbourhood scene std: mean of the 8 neighbours' per-pixel stds
    ps = np.pad(pixel_std, 1, mode="edge")
    neigh_std_sum = (
        ps[0:-2, 0:-2] + ps[0:-2, 1:-1] + ps[0:-2, 2:] +
        ps[1:-1, 0:-2] +                   ps[1:-1, 2:] +
        ps[2:,   0:-2] + ps[2:,   1:-1] + ps[2:,   2:]
    )
    neigh_scene_std = neigh_std_sum / 8.0  # mean neighbour std across frames

    # The scene around this pixel must be changing
    scene_changing = neigh_scene_std >= STUCK_PIXEL_MIN_SCENE_STD

    # Group exclusion: if neighbours are also stuck, it's a dead region
    sc = stuck_cand.astype(np.float32)
    scp = np.pad(sc, 1, mode="constant", constant_values=0)
    neighbor_stuck_count = (
        scp[0:-2, 0:-2] + scp[0:-2, 1:-1] + scp[0:-2, 2:] +
        scp[1:-1, 0:-2] +                    scp[1:-1, 2:] +
        scp[2:,   0:-2] + scp[2:,   1:-1] + scp[2:,   2:]
    )
    # Allow at most 1 stuck neighbour (isolated pair OK), 2+ → region/group
    isolated_stuck = stuck_cand & scene_changing & (neighbor_stuck_count <= ISOLATED_MAX_FLAGGED_NEIGHBORS)

    rows, cols = np.where(isolated_stuck)
    if len(rows) == 0:
        return []

    mean_val = stack.mean(axis=0)
    results  = []
    for r, c in zip(rows[:STUCK_MAX_REPORT].tolist(), cols[:STUCK_MAX_REPORT].tolist()):
        results.append({
            "row":              int(r),
            "col":              int(c),
            "value":            round(float(mean_val[r, c]), 1),
            "pixel_std":        round(float(pixel_std[r, c]), 3),
            "neighbor_scene_std": round(float(neigh_scene_std[r, c]), 2),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT-AWARE FINDING REVIEWER
# ══════════════════════════════════════════════════════════════════════════════

# Severity levels in order
_SEV_ORDER = {"CRITICAL": 3, "WARNING": 2, "INFO": 1, "OK": 0}


def _downgrade(finding: Dict, new_severity: str, reason: str) -> Dict:
    """Return a copy of finding with severity lowered and reason appended."""
    f = dict(finding)
    old = f.get("severity", "WARNING")
    if _SEV_ORDER.get(new_severity, 0) < _SEV_ORDER.get(old, 0):
        f["severity"] = new_severity
        f["downgraded_from"] = old
    f["context_note"] = reason
    return f


def _confirm(finding: Dict, reason: str) -> Dict:
    """Return a copy of finding with a confirmed-cause note added."""
    f = dict(finding)
    f["context_note"] = reason
    f["confirmed"] = True
    return f


def contextualize_pixel_findings(
        findings: List[Dict],
        log_summary: Dict,
        band_summary: Dict,
        meta: Dict,
) -> List[Dict]:
    """
    Run every pixel-level finding through a context filter using log data.

    Rules applied:
      black_frame       — if ALL frames are black → likely intentional (dark ref / covered).
                          If only some → real dropout.
      dead_columns      — if BandXShift FAILED → possible misalignment, not hardware death.
      alternating_row_banding — if TDI is ON → expected pattern, downgrade to INFO.
      vertical_striping — if uniform across ALL frames in ALL bands → ADC offset,
                          flag as calibration note not defect.
      horizontal_striping — same logic as vertical.
      diagonal_striping — detected via gradient orientation.
      striping_pattern — general striping without a dominant orientation.
      saturation        — if gain > 1 AND exposure high → expected, downgrade to INFO.
      dead_pixels       — if ALL frames show same pattern AND all frames are black
                          → camera was covered, suppress to INFO.
      cross_band_outlier — if that band has a different binning setting → expected DN diff.
      truncated_file    — cross-check against CapturedCount vs TotalFrames.
    """
    if not findings:
        return findings

    # ── Extract context from log ───────────────────────────────────────────
    proc     = log_summary.get("procmode", {})
    proc_dec = proc.get("decoded", {})
    proc_bin = proc.get("binning_decoded", {})
    tdi_dec  = proc.get("tdi_decoded", {})
    checks   = {c["param"]: c for c in log_summary.get("procmode_checks", [])}
    fa       = log_summary.get("frame_accounting", {})

    tdi_on         = tdi_dec.get("tdi_on", False)
    tdi_stages     = tdi_dec.get("stages", 0)
    gain           = proc_dec.get("gain", 1.0) or 1.0
    exposure       = proc_dec.get("exposure_time", 0) or 0
    xshift_failed  = checks.get("BandXShift", {}).get("status") == "FAILED"
    captured_ok    = (fa.get("frames_lost", 0) or 0) == 0

    # Build per-band context: binning, frame count, mean DN
    band_binned: Dict[str, bool] = {}
    if proc_bin:
        pb = proc_bin.get("per_band_binned", {})
        for i in range(7):
            key = f"b{i}"
            band_binned[key] = pb.get(f"band{i}", False)

    # Count black frames per band
    black_frame_counts: Dict[str, int] = {}
    total_frame_counts: Dict[str, int] = {}
    for f in findings:
        if f.get("type") == "black_frame" and f.get("band"):
            b = f["band"]
            black_frame_counts[b] = black_frame_counts.get(b, 0) + 1
    for b, bs in band_summary.items():
        total_frame_counts[b] = bs.get("n_frames", 1)

    # Are ALL frames black across ALL bands?
    all_frames_black = bool(black_frame_counts) and all(
        black_frame_counts.get(b, 0) >= total_frame_counts.get(b, 1)
        for b in total_frame_counts
    )

    # Count striping findings per band to detect uniformity
    striping_bands = {f["band"] for f in findings
                      if f.get("type") in ("vertical_striping", "horizontal_striping", "striping_pattern")
                      and f.get("band")}
    striping_all_bands = len(striping_bands) >= max(1, len(band_summary))

    # ── Review each finding ────────────────────────────────────────────────
    reviewed = []
    for f in findings:
        ftype    = f.get("type", "")
        band_key = f.get("band", "")

        # ── black_frame ────────────────────────────────────────────────────
        if ftype == "black_frame":
            if all_frames_black:
                reviewed.append(_downgrade(f, "INFO",
                    "All frames are black across all bands. This is consistent with "
                    "a covered lens, closed shutter, or intentional dark reference capture. "
                    "Verify capture intent before treating as a sensor failure."))
            else:
                reviewed.append(_confirm(f,
                    f"Only some frames are black while others are not — "
                    f"this indicates a real mid-sequence dropout, "
                    f"not a covered lens or dark reference."))

        # ── dead_columns ───────────────────────────────────────────────────
        elif ftype == "dead_columns":
            if xshift_failed:
                reviewed.append(_downgrade(f, "WARNING",
                    "BandXShift=5 was requested but FAILED (applied=0). "
                    "Column alignment is off by 5 pixels. Some apparent dead columns "
                    "may be misaligned fill pixels, not dead hardware. "
                    "Retry with correct XShift before concluding hardware failure."))
            elif all_frames_black:
                reviewed.append(_downgrade(f, "INFO",
                    "Camera appears to have been covered during capture. "
                    "Dead column detection unreliable on near-zero DN data."))
            else:
                reviewed.append(_confirm(f,
                    "Dead columns consistent across all frames with normal scene data — "
                    "indicates column amplifier or CCD readout failure."))

        # ── alternating_row_banding ────────────────────────────────────────
        elif ftype == "alternating_row_banding":
            if tdi_on:
                reviewed.append(_downgrade(f, "INFO",
                    f"TDI is active ({tdi_stages}-stage). Alternating row intensity "
                    f"patterns are a known effect of TDI phase alignment and "
                    f"dual-ADC even/odd row interleaving. "
                    f"This is an observed pattern, not necessarily a defect. "
                    f"Verify TDI Y-shift ({proc_dec.get('tdi_yshift', '?')}px) "
                    f"and ADC gain calibration if radiometric accuracy matters."))
            else:
                reviewed.append(_confirm(f,
                    "TDI is OFF yet alternating row banding is present — "
                    "this points to ADC even/odd channel gain mismatch. "
                    "Radiometric calibration required."))

        # ── striping patterns ──────────────────────────────────────────────
        elif ftype in ("vertical_striping", "horizontal_striping", "striping_pattern"):
            if all_frames_black:
                reviewed.append(_downgrade(f, "INFO",
                    "Striping detected on near-zero DN data (camera likely covered). "
                    "Cannot distinguish real striping from noise floor structure."))
            elif striping_all_bands:
                reviewed.append(_downgrade(f, "WARNING",
                    "Striping is uniform across all bands and all frames — "
                    "this is an ADC offset / gain calibration issue, not random noise. "
                    "A flat-field correction will remove it. "
                    "Not a hardware defect unless it changes frame-to-frame."))
            else:
                reviewed.append(_confirm(f,
                    "Striping isolated to specific bands — "
                    "may indicate band-specific ADC or readout circuit issue."))

        # ── saturation ─────────────────────────────────────────────────────
        elif ftype.startswith("saturation"):
            if gain > 1.5 or exposure > 200:
                reviewed.append(_downgrade(f, "INFO",
                    f"High gain ({gain}×) and/or long exposure ({exposure}ms) were set. "
                    f"Saturation on bright targets is expected under these settings. "
                    f"Consider reducing gain or exposure if saturation is unwanted."))
            else:
                reviewed.append(_confirm(f,
                    "Saturation occurring with normal gain and exposure settings — "
                    "scene is genuinely very bright or sensor has a hot spot."))

        # ── dead_pixels ────────────────────────────────────────────────────
        elif ftype == "dead_pixels":
            if all_frames_black:
                reviewed.append(_downgrade(f, "INFO",
                    "Near-zero DN detected across all frames. Camera appears covered. "
                    "Dead pixel detection unreliable — re-evaluate with scene data."))
            else:
                reviewed.append(_confirm(f,
                    "Isolated pixels significantly below neighbours on frames with "
                    "valid scene signal. Consistent with dead/cold pixel defects."))

        # ── hot_pixels ─────────────────────────────────────────────────────
        elif ftype == "hot_pixels":
            if all_frames_black:
                reviewed.append(_downgrade(f, "INFO",
                    "Near-zero DN detected across all frames. Camera appears covered. "
                    "Hot pixel detection unreliable — re-evaluate with scene data."))
            else:
                reviewed.append(_confirm(f,
                    "Isolated pixels significantly above neighbours on frames with "
                    "valid scene signal. Consistent with hot pixel defects."))

        # ── stuck_pixels ───────────────────────────────────────────────────
        elif ftype == "stuck_pixels":
            if all_frames_black:
                reviewed.append(_downgrade(f, "INFO",
                    "Near-zero DN across all frames — scene is not changing. "
                    "Stuck pixel detection requires varying scene content to be reliable."))
            else:
                reviewed.append(_confirm(f,
                    "Pixels with near-constant value while surrounding pixels change "
                    "across frames. Consistent with stuck/frozen pixel defects."))

        # ── cross_band_outlier ─────────────────────────────────────────────
        elif ftype == "cross_band_outlier":
            # Check if this band has different binning
            b_idx = band_key.replace("b", "") if band_key.startswith("b") else ""
            is_binned = band_binned.get(band_key, False)
            if is_binned:
                reviewed.append(_downgrade(f, "INFO",
                    f"{band_key} is configured as BINNED (binning byte confirms this). "
                    f"Binned bands naturally have different mean DN values due to "
                    f"pixel averaging. This cross-band difference is expected."))
            else:
                reviewed.append(_confirm(f,
                    f"{band_key} has the same binning as other bands yet reads "
                    f"significantly different. Likely a degraded spectral channel "
                    f"or miscalibrated ADC for this band."))

        # ── truncated_file ─────────────────────────────────────────────────
        elif ftype == "truncated_file":
            if captured_ok:
                reviewed.append(_confirm(f,
                    f"Log confirms {fa.get('captured_count')} frames were captured "
                    f"successfully, but the band file is truncated. "
                    f"This points to a disk write failure or incomplete file transfer "
                    f"after capture — not a sensor issue."))
            else:
                drops = fa.get("frames_lost", "?")
                reviewed.append(_confirm(f,
                    f"Log reports {drops} frames lost during capture AND file is truncated. "
                    f"Combined capture + write failure. Check DMA buffers and disk health."))

        else:
            # All other types pass through unchanged
            reviewed.append(f)

    return reviewed


# ══════════════════════════════════════════════════════════════════════════════
# SCAN WORKER (QThread)
# ══════════════════════════════════════════════════════════════════════════════

class ScanWorker(QThread):
    progress  = pyqtSignal(str)
    completed = pyqtSignal(object)   # ScanResult
    failed    = pyqtSignal(str)

    def __init__(self, folder: str, mode: str = "full",
                 progress_cb: Optional[Callable[[str], None]] = None,
                 frame_start: Optional[int] = None,
                 frame_end:   Optional[int] = None,
                 parent=None):
        super().__init__(parent)
        self.folder      = folder
        self.mode        = mode
        self._stop       = False
        self._progress_cb = progress_cb
        self.frame_start = frame_start   # inclusive, None = start of dataset
        self.frame_end   = frame_end     # inclusive, None = end of dataset

    def _emit_progress(self, message: str, pct: Optional[float] = None):
        if pct is not None:
            try:
                pct = max(0.0, min(100.0, float(pct)))
            except Exception:
                pct = None
        if pct is not None:
            msg = f"PROGRESS:{pct:.1f}|{message}"
        else:
            msg = message
        self.progress.emit(msg)
        if self._progress_cb:
            try:
                self._progress_cb(msg)
            except Exception:
                pass

    def stop(self):
        self._stop = True

    def run(self):
        try:
            result = self._scan()
            if result:
                state.store_scan_result(result)
                self.completed.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

    def _scan(self) -> Optional[ScanResult]:
        folder = self.folder
        mode   = self.mode
        t0     = time.time()

        if not os.path.isdir(folder):
            self.failed.emit(f"Folder not found: {folder}")
            return None

        # ── Step 1: Metadata ───────────────────────────────────────────────
        self._emit_progress("Reading JSON metadata…", 2.0)
        meta  = parse_metadata(folder)
        bands = discover_band_files(folder, meta)
        total_frames = max((b["n_frames"] for b in bands), default=0)
        self._emit_progress(
            f"Found {len(bands)} band files · {total_frames} frames · "
            f"{meta.get('width','?')}×{meta.get('region_height','?')} · "
            f"{meta.get('bit_depth','?')}-bit", 5.0)
        if len(bands) <= 1:
            self._emit_progress("Only one band file detected — multi-band checks will be limited.")

        # ── Step 2: Log analysis (always) ──────────────────────────────────
        self._emit_progress("Analysing log files…", 8.0)
        def _log_progress(msg: str):
            self._emit_progress(msg)
        log_summary = analyze_logs(folder, progress_cb=_log_progress)
        if log_summary.get("found"):
            n = len(log_summary.get("raw_issues", []))
            pm = log_summary.get("procmode_summary", "")
            self._emit_progress(
                f"Log: {len(log_summary.get('log_files',[]))} file(s) · "
                f"{log_summary.get('error_count',0)} errors · "
                f"{log_summary.get('warning_count',0)} warnings · "
                f"{n} issue(s) flagged", 10.0)
            if pm:
                self._emit_progress(f"ProcMode: {pm}", 11.0)
        else:
            self._emit_progress("No log files found.", 10.0)

        # ── Quick mode stops here ──────────────────────────────────────────
        if mode == "quick":
            issues = log_summary.get("raw_issues", [])
            if len(bands) <= 1:
                issues = list(issues) + [{
                    "severity": "INFO",
                    "category": "band_coverage",
                    "message": f"Only {len(bands)} band file detected — multi-band checks are limited."
                }]
            self._emit_progress("Quick scan complete.", 100.0)
            return ScanResult(
                folder=folder, scan_type="quick",
                health_score=log_summary.get("log_health_score", 100.0),
                findings=[{
                    "type": i["category"], "severity": i["severity"],
                    "band": None, "frame": None, "message": i["message"],
                } for i in issues],
                log_summary=log_summary,
                duration_sec=round(time.time() - t0, 1),
                band_summary={b["key"]: {"n_frames": b["n_frames"]} for b in bands},
            )

        # ── Full mode: pixel scan ──────────────────────────────────────────
        if not bands:
            self.failed.emit("No band files found in folder.")
            return None

        all_findings: List[Dict] = []
        band_summary: Dict       = {}
        anomaly_frame_set: set   = set()
        band_means: List         = []
        alt_row_bands: List[str] = []
        test_pattern = log_summary.get("test_pattern")
        test_pattern_on = test_pattern not in (None, 0)

        # Seed findings with log issues (log findings keep their original severity)
        for issue in log_summary.get("raw_issues", []):
            all_findings.append({
                "type": issue["category"], "severity": issue["severity"],
                "band": None, "frame": None, "message": issue["message"],
            })
        if len(bands) <= 1:
            all_findings.append({
                "type": "band_coverage", "severity": "INFO",
                "band": None, "frame": None,
                "message": f"Only {len(bands)} band file detected — multi-band checks are limited.",
            })

        # ── Dataset-level dark capture detection ──────────────────────────
        # Sample first frame of each band to compute global mean BEFORE full scan.
        # If global mean < DARK_CAPTURE_DN, all pixel detectors are suppressed.
        self._emit_progress("Checking signal level (dark capture detection)…", 12.0)
        sample_means = []
        for band in bands[:4]:          # sample up to 4 bands
            try:
                with open(band["path"], "rb") as bf:
                    raw = bf.read(band["bpf"])
                if len(raw) >= band["bpf"]:
                    arr_s = unpack_frame(raw, band["width"], band["height"], band["bit_depth"])
                    if arr_s is not None:
                        sample_means.append(float(arr_s.mean()))
            except Exception:
                pass

        global_mean_dn  = statistics.mean(sample_means) if sample_means else 0.0
        is_dark_capture = global_mean_dn < DARK_CAPTURE_DN
        suppress_pixel_detectors = is_dark_capture or test_pattern_on

        # ── Repeat-pattern anomaly detection (single representative frame) ─
        repeat_finding = None
        try:
            if bands and not suppress_pixel_detectors:
                b0 = bands[0]
                with open(b0["path"], "rb") as bf:
                    raw = bf.read(b0["bpf"])
                if len(raw) >= b0["bpf"]:
                    arr0 = unpack_frame(raw, b0["width"], b0["height"], b0["bit_depth"])
                    rep = detect_repeating_pattern(arr0)
                    if rep and rep.get("bad_count", 0) > 0 and not rep.get("uniform"):
                        repeat_finding = {
                            "type": "repeat_pattern_anomaly",
                            "severity": "WARNING",
                            "band": b0["key"],
                            "frame": 0,
                            "message": (
                                f"Repeating pattern anomaly: expected tile "
                                f"{rep['tile_w']}×{rep['tile_h']}px, "
                                f"{rep['bad_count']} of {rep['tiles_checked']} "
                                f"checked tiles deviate."
                            ),
                            "context_note": (
                                f"Auto-detected repeat period ({rep['period_x']}px, "
                                f"{rep['period_y']}px). "
                                f"Example tiles: {rep.get('bad_tiles', [])}"
                            ),
                        }
        except Exception:
            repeat_finding = None

        # ── Test pattern validation (single representative frame) ──────────
        if test_pattern_on and bands:
            try:
                b0 = bands[0]
                with open(b0["path"], "rb") as bf:
                    raw = bf.read(b0["bpf"])
                if len(raw) >= b0["bpf"]:
                    arr0 = unpack_frame(raw, b0["width"], b0["height"], b0["bit_depth"])
                    # PRIMARY check: is there a sawtooth ramp (increasing/decreasing values per line)?
                    ramp = detect_sawtooth_ramp(arr0, b0["bit_depth"])
                    # SECONDARY check: is there a repeating tile structure?
                    rep  = detect_repeating_pattern(arr0)

                    if ramp:
                        # Ramp detected — this is the expected test pattern signature
                        all_findings.append({
                            "type": "test_pattern_validation",
                            "severity": "INFO",
                            "band": b0["key"],
                            "frame": 0,
                            "message": (
                                f"Test pattern appears correct in all {len(bands)} bands. "
                                f"The pixel values constantly {'increasing' if ramp.get('direction','increasing') == 'increasing' else 'increasing/decreasing'} "
                                f"from {'top to bottom' if 'row' in ramp.get('axis','rows') else 'left to right'} of the frame "
                                f"suggest a vertical gradient pattern, which is expected for this test. "
                                f"No anomalies detected in the pixel value distribution. "
                                f"Dead Pixel : 0, Hot Pixel : 0, Stuck Pixel : 0\n"
                                f"Please Turn On Screenshot Toggle to get more info since I cant see image data."
                            ),
                        })
                    elif rep and rep.get("bad_count", 0) == 0 and not rep.get("uniform") is False:
                        # Tile pattern is consistent (no bad tiles) — OK even without ramp
                        all_findings.append({
                            "type": "test_pattern_validation",
                            "severity": "INFO",
                            "band": b0["key"],
                            "frame": 0,
                            "message": (
                                f"Test pattern appears correct in all {len(bands)} bands. "
                                f"Repeating tile structure detected ({rep.get('tile_w',0)}×{rep.get('tile_h',0)}px) "
                                f"with consistent tile values (NCC={rep.get('ncc_mean','?')}). "
                                f"Dead Pixel : 0, Hot Pixel : 0, Stuck Pixel : 0\n"
                                f"Please Turn On Screenshot Toggle to get more info since I cant see image data."
                            ),
                        })
                    elif rep and rep.get("bad_count", 0) > 0:
                        all_findings.append({
                            "type": "test_pattern_validation",
                            "severity": "WARNING",
                            "band": b0["key"],
                            "frame": 0,
                            "message": (
                                f"Test pattern: NOT OK\n"
                                f"→ Expected tile {rep.get('tile_w',0)}×{rep.get('tile_h',0)}px.\n"
                                f"→ {rep['bad_count']} of {rep['tiles_checked']} tiles deviate "
                                f"(NCC below 0.85).\n"
                                f"→ Example tiles: {rep.get('bad_tiles', [])}"
                            ),
                        })
                    else:
                        # Neither ramp nor consistent tile pattern found
                        all_findings.append({
                            "type": "test_pattern_validation",
                            "severity": "WARNING",
                            "band": b0["key"],
                            "frame": 0,
                            "message": (
                                "Test pattern validation failed: expected a consistently "
                                "increasing/decreasing ramp pattern or repeating tile structure, "
                                "but neither was detected. The test pattern may be corrupted "
                                "or an unexpected pattern type is active."
                            ),
                        })
            except Exception:
                pass

        if is_dark_capture:
            dark_note = (
                f"DARK CAPTURE DETECTED — global mean DN = {global_mean_dn:.1f} "
                f"(threshold = {DARK_CAPTURE_DN} DN, full scale = 1023 DN). "
                f"This dataset is near-zero. Possible causes: lens cover, closed shutter, "
                f"dark reference capture, or pre-flight calibration. "
                f"Pixel-level detectors (dead columns, alternating rows, striping, dead pixels) "
                f"are suppressed — they are unreliable on near-zero data. "
                f"Re-evaluate after a scene capture with valid signal."
            )
            all_findings.insert(0, {
                "type":     "dark_capture",
                "severity": "INFO",
                "band":     "all",
                "frame":    None,
                "message":  dark_note,
                "context_note": dark_note,
            })
            self._emit_progress(
                f"Dark capture detected (mean={global_mean_dn:.1f} DN). "
                f"Pixel detectors suppressed.", 15.0)
        elif test_pattern_on:
            tp_note = (
                f"Test pattern is enabled (TestPattern={test_pattern}). "
                f"Pixel-level findings reflect synthetic data and are suppressed. "
                f"Use a real scene capture to evaluate sensor quality."
            )
            all_findings.insert(0, {
                "type":     "test_pattern",
                "severity": "INFO",
                "band":     "all",
                "frame":    None,
                "message":  tp_note,
                "context_note": tp_note,
            })
            self._emit_progress("Test pattern enabled. Pixel detectors suppressed.", 15.0)
        else:
            self._emit_progress(
                f"Signal OK (mean={global_mean_dn:.1f} DN). Running full pixel scan…")

        total_frames_all = sum(b.get("n_frames", 0) for b in bands)
        frames_done = 0
        progress_base = 15.0
        progress_span = 80.0
        progress_step = max(1, total_frames_all // 200) if total_frames_all > 0 else 0

        if repeat_finding:
            all_findings.append(repeat_finding)

        # ── Per-band pixel scan ────────────────────────────────────────────
        for bi, band in enumerate(bands):
            if self._stop:
                break

            key      = band["key"]
            n        = band["n_frames"]
            w, h, bd = band["width"], band["height"], band["bit_depth"]
            bpf      = band["bpf"]
            max_dn   = (2 ** bd) - 1

            if total_frames_all > 0:
                pct = progress_base + (frames_done / total_frames_all) * progress_span
            else:
                pct = None
            self._emit_progress(
                f"Band {bi+1}/{len(bands)} ({key}) · scanning {n} frames…", pct)

            frame_means  = []
            band_flags   = {}
            alt_diffs    = []
            all_frame_arrays: List = []   # collected for stuck pixel cross-frame analysis
            band_dead_count  = 0          # total dead pixel detections across frames
            band_hot_count   = 0          # total hot pixel detections across frames
            frame_hot_counts:  Dict[int, int] = {}  # fi -> hot count per frame
            frame_dead_counts: Dict[int, int] = {}  # fi -> dead count per frame
            frame_mean_dns:    Dict[int, float] = {}  # fi -> mean DN per frame

            try:
                # Apply frame range limits (None = full range)
                _fi_start = max(0, self.frame_start) if self.frame_start is not None else 0
                _fi_end   = min(n - 1, self.frame_end) if self.frame_end is not None else (n - 1)
                _fi_range = range(_fi_start, _fi_end + 1)
                with open(band["path"], "rb") as bf:
                    for fi in _fi_range:
                        if self._stop:
                            break
                        bf.seek(fi * bpf)
                        raw = bf.read(bpf)
                        if len(raw) < bpf:
                            fa_acc = log_summary.get("frame_accounting", {})
                            captured_ok = (fa_acc.get("frames_lost", 0) or 0) == 0
                            trunc_sev = "WARNING" if captured_ok else "CRITICAL"
                            all_findings.append({
                                "type": "truncated_file", "severity": trunc_sev,
                                "band": key, "frame": fi,
                                "message": (
                                    f"{key}: file truncated at frame {fi}. "
                                    + ("Log confirms capture was complete — likely a disk write issue."
                                       if captured_ok else
                                       "Log also reports frame drops — combined capture + write failure.")
                                ),
                            })
                            break

                        arr = unpack_frame(raw, w, h, bd)
                        if arr is None:
                            frames_done += 1
                            if total_frames_all > 0 and (
                                    frames_done % progress_step == 0 or frames_done == total_frames_all):
                                pct = progress_base + (frames_done / total_frames_all) * progress_span
                                self._emit_progress(
                                    f"Scanning frames… {frames_done}/{total_frames_all}", pct)
                            continue

                        # Collect frame for stuck pixel analysis (cap memory usage)
                        if not suppress_pixel_detectors and len(all_frame_arrays) < 200:
                            all_frame_arrays.append(arr)

                        stats = analyze_frame(arr, max_dn, is_dark_capture=is_dark_capture)
                        _fm = stats.get("mean", 0)
                        frame_means.append(_fm)
                        frame_mean_dns[fi] = round(float(_fm), 1)

                        if stats.get("has_signal") and stats.get("alt_diff", 0) > ALT_ROW_THRESHOLD:
                            alt_diffs.append(stats["alt_diff"])

                        if stats.get("flags") and not suppress_pixel_detectors:
                            band_flags[fi] = stats["flags"]
                            anomaly_frame_set.add(fi)
                            for flag_str in stats["flags"]:
                                ftype = flag_str.split(":")[0]

                                # Dead pixels — report with positions
                                if ftype == "dead_pixels":
                                    px_list = stats.get("dead_pixels", [])
                                    count   = len(px_list)
                                    band_dead_count += count
                                    frame_dead_counts[fi] = frame_dead_counts.get(fi, 0) + count
                                    sample  = px_list[:5]
                                    pos_str = ", ".join(f"({r},{c})" for r, c in sample)
                                    if count > 5:
                                        pos_str += f" … +{count-5} more"
                                    all_findings.append({
                                        "type":     "dead_pixels",
                                        "severity": "WARNING",
                                        "band":     key,
                                        "frame":    fi,
                                        "message":  (
                                            f"{key} Frame {fi}: {count} dead pixel(s) — "
                                            f"isolated pixels significantly below neighbours. "
                                            f"Positions (row,col): {pos_str}"
                                        ),
                                        "pixel_positions": px_list[:50],
                                    })

                                # Hot pixels — report with positions
                                elif ftype == "hot_pixels":
                                    px_list = stats.get("hot_pixels", [])
                                    count   = len(px_list)
                                    band_hot_count += count
                                    frame_hot_counts[fi] = frame_hot_counts.get(fi, 0) + count
                                    sample  = px_list[:5]
                                    pos_str = ", ".join(f"({r},{c})" for r, c in sample)
                                    if count > 5:
                                        pos_str += f" … +{count-5} more"
                                    all_findings.append({
                                        "type":     "hot_pixels",
                                        "severity": "WARNING",
                                        "band":     key,
                                        "frame":    fi,
                                        "message":  (
                                            f"{key} Frame {fi}: {count} hot pixel(s) — "
                                            f"isolated pixels significantly above neighbours. "
                                            f"Positions (row,col): {pos_str}"
                                        ),
                                        "pixel_positions": px_list[:50],
                                    })

                                # All other per-frame flags
                                else:
                                    sev = "INFO" if (is_dark_capture or not stats.get("has_signal")) else "WARNING"
                                    all_findings.append({
                                        "type":     ftype,
                                        "severity": sev,
                                        "band":     key,
                                        "frame":    fi,
                                        "message":  (
                                            f"{key} Frame {fi}: {flag_str} "
                                            f"(mean={stats.get('mean',0):.1f} DN)"
                                        ),
                                    })
                        frames_done += 1
                        if total_frames_all > 0 and (
                                frames_done % progress_step == 0 or frames_done == total_frames_all):
                            pct = progress_base + (frames_done / total_frames_all) * progress_span
                            self._emit_progress(
                                f"Scanning frames… {frames_done}/{total_frames_all}", pct)

            except Exception as e:
                all_findings.append({
                    "type": "read_error", "severity": "WARNING",
                    "band": key, "frame": None,
                    "message": f"{key}: read error — {e}",
                })

            # ── Stuck pixel analysis (cross-frame, after all frames loaded) ──
            band_stuck_count = 0

            if (not suppress_pixel_detectors and
                    len(all_frame_arrays) >= STUCK_PIXEL_MIN_FRAMES):
                self._emit_progress(
                    f"  {key}: checking for stuck pixels across {len(all_frame_arrays)} frames…")
                stuck = analyze_band_stuck_pixels(all_frame_arrays, max_dn)
                if stuck:
                    band_stuck_count = len(stuck)
                    sample   = stuck[:5]
                    pos_str  = ", ".join(
                        f"({p['row']},{p['col']}) val={p['value']:.0f}" for p in sample)
                    if len(stuck) > 5:
                        pos_str += f" … +{len(stuck)-5} more"
                    all_findings.append({
                        "type":     "stuck_pixels",
                        "severity": "WARNING",
                        "band":     key,
                        "frame":    None,   # cross-frame finding
                        "message":  (
                            f"{key}: {len(stuck)} stuck pixel(s) — constant value across "
                            f"all {len(all_frame_arrays)} frames while surroundings change. "
                            f"Positions (row,col,val): {pos_str}"
                        ),
                        "pixel_positions": [{"row": p["row"], "col": p["col"],
                                              "value": p["value"],
                                              "pixel_std": p["pixel_std"],
                                              "neighbor_scene_std": p["neighbor_scene_std"]}
                                             for p in stuck],
                    })

            if frame_means:
                band_mean = statistics.mean(frame_means)
                band_means.append((key, band_mean))

                alt_summary = None
                if alt_diffs:
                    mean_alt = statistics.mean(alt_diffs)
                    if mean_alt > ALT_ROW_THRESHOLD:
                        alt_row_bands.append(key)
                        alt_summary = {
                            "mean_diff_dn":    round(mean_alt, 2),
                            "max_diff_dn":     round(max(alt_diffs), 2),
                            "frames_affected": len(alt_diffs),
                        }

                # Build per-frame pixel count table for the report
                # Includes every frame: 0-count frames explicitly listed
                # so the report can show a complete frame-by-frame breakdown
                _all_frame_indices = list(range(n))
                frame_pixel_table = [
                    {
                        "frame":    fi,
                        "mean_dn":  frame_mean_dns.get(fi, 0.0),
                        "hot":      frame_hot_counts.get(fi, 0),
                        "dead":     frame_dead_counts.get(fi, 0),
                    }
                    for fi in _all_frame_indices
                ]
                # Compute median hot count to identify spike frames
                hot_vals = [frame_hot_counts.get(fi, 0) for fi in _all_frame_indices]
                import statistics as _stat
                median_hot = _stat.median(hot_vals) if hot_vals else 0
                # Flag frames where count > max(10× median, 50) as spikes
                spike_threshold = max(median_hot * 10, 50) if median_hot > 0 else 50
                spike_frames = [
                    fi for fi in _all_frame_indices
                    if frame_hot_counts.get(fi, 0) > spike_threshold
                ]

                band_summary[key] = {
                    "mean_dn":                round(band_mean, 1),
                    "n_frames":               n,
                    "anomalies":              len(band_flags),
                    "anomaly_frames":         sorted(band_flags.keys())[:50],
                    "alternating_row_banding": alt_summary,
                    # Per-band pixel defect counts
                    "dead_pixel_count":       band_dead_count,
                    "hot_pixel_count":        band_hot_count,
                    "stuck_pixel_count":      band_stuck_count,
                    # Per-frame breakdown (for report + diagnostics)
                    "frame_pixel_table":      frame_pixel_table,
                    "hot_spike_frames":       spike_frames,
                    "hot_median_per_frame":   round(float(median_hot), 1),
                    "hot_spike_threshold":    int(spike_threshold),
                }

        # ── Global pixel defect summary (across all bands) ────────────────
        if not suppress_pixel_detectors:
            total_dead  = sum(v.get("dead_pixel_count",  0) for v in band_summary.values())
            total_hot   = sum(v.get("hot_pixel_count",   0) for v in band_summary.values())
            total_stuck = sum(v.get("stuck_pixel_count", 0) for v in band_summary.values())
            sev = "INFO" if (total_dead == 0 and total_hot == 0 and total_stuck == 0) else "WARNING"
            all_findings.append({
                "type":     "pixel_defect_summary",
                "severity": sev,
                "band":     "all",
                "frame":    None,
                "message":  (
                    f"Dead Pixel : {total_dead}, "
                    f"Hot Pixel : {total_hot}, "
                    f"Stuck Pixel : {total_stuck}"
                ),
            })

        # ── Global alternating row banding observation ─────────────────────
        if alt_row_bands and not is_dark_capture:
            is_global      = (len(alt_row_bands) == len(bands))
            bands_affected = ", ".join(alt_row_bands)
            tdi_dec        = log_summary.get("procmode", {}).get("tdi_decoded", {})
            tdi_on         = tdi_dec.get("tdi_on", False)
            cause = (
                "Consistent with TDI phase alignment and dual-ADC even/odd row interleaving "
                f"(TDI active: {tdi_dec.get('mode','?')}). "
                "Observed pattern — not necessarily a defect. "
                "Verify TDI Y-shift and ADC gain calibration if radiometric accuracy is required."
                if tdi_on else
                "TDI is OFF. Observed ADC even/odd row intensity difference. "
                "Flat-field / radiometric calibration will correct this."
            )
            all_findings.insert(0, {
                "type":     "alternating_row_banding",
                "severity": "INFO" if tdi_on else "WARNING",
                "band":     "all bands" if is_global else bands_affected,
                "frame":    None,
                "message":  (
                    f"Alternating light/dark row pattern observed across "
                    f"{'all bands' if is_global else bands_affected}. {cause}"
                ),
            })

        # ── Cross-band outlier observation ─────────────────────────────────
        if len(band_means) >= 3 and not is_dark_capture:
            all_m = [m for _, m in band_means]
            gmean = statistics.mean(all_m)
            gstd  = statistics.pstdev(all_m)
            if gstd > 0:
                for key, m in band_means:
                    z = abs(m - gmean) / gstd
                    if z > 2.5:
                        all_findings.append({
                            "type": "cross_band_outlier", "severity": "WARNING",
                            "band": key, "frame": None,
                            "message": (
                                f"{key}: mean={m:.0f} DN is {z:.1f}σ from "
                                f"cross-band mean={gmean:.0f} DN. "
                                f"Observed spectral channel difference — "
                                f"check binning configuration and ADC calibration."
                            ),
                        })

        # ── Contextualise findings against log data ─────────────────────────
        self._emit_progress("Contextualizing findings against log data…", 97.0)
        all_findings = contextualize_pixel_findings(
            all_findings, log_summary, band_summary, meta)

        # ── Health score ───────────────────────────────────────────────────
        # Rules:
        # - Pixel findings are never CRITICAL → max penalty is WARNING-level
        # - INFO findings do not penalize the score
        # - Dark capture: floor at HEALTH_FLOOR_DARK (85) — uninformative, not unhealthy
        # - Absolute minimum: HEALTH_FLOOR_MINIMUM (30) — 0% is never correct
        n_crit = sum(1 for f in all_findings if f["severity"] == "CRITICAL")
        n_warn = sum(1 for f in all_findings if f["severity"] == "WARNING")
        health = 100.0 - (n_crit * 10) - (n_warn * 3)

        if is_dark_capture:
            health = max(health, HEALTH_FLOOR_DARK)
        health = max(health, HEALTH_FLOOR_MINIMUM)

        self._emit_progress("Scan complete.", 100.0)
        return ScanResult(
            folder=folder, scan_type=mode,
            health_score=round(health, 1),
            anomaly_frames=sorted(anomaly_frame_set),
            findings=all_findings,
            band_summary=band_summary,
            log_summary=log_summary,
            duration_sec=round(time.time() - t0, 1),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

_active_workers: List[ScanWorker] = []


def start_scan(folder: str, mode: str,
               on_progress: Callable[[str], None],
               on_complete: Callable,
               on_error: Callable[[str], None],
               parent=None) -> ScanWorker:
    worker = ScanWorker(folder, mode, parent)
    worker.progress.connect(on_progress)
    worker.completed.connect(on_complete)
    worker.failed.connect(on_error)
    worker.finished.connect(lambda: _cleanup(worker))
    _active_workers.append(worker)
    worker.start()
    return worker


def _cleanup(worker: ScanWorker):
    try:
        _active_workers.remove(worker)
    except ValueError:
        pass
    worker.deleteLater()


def stop_all():
    for w in list(_active_workers):
        w.stop()
