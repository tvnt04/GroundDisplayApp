from __future__ import annotations

import os
import re
import json
import math
import time
import statistics
import threading
from pathlib import Path
from app_paths import get_app_data_path, migrate_legacy_file
from typing import Dict, List, Optional, Tuple, Any


# ══════════════════════════════════════════════════════════════════════════════
# .META FILE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_meta_file(folder: str) -> Dict:
    """
    Parse the *.meta text file in a dataset folder.
    Returns a summary dict with satellite identity, ground track,
    band configuration, frame count, and timestamp span.
    """
    meta_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".meta") or f.lower().endswith(".txt")
        and "meta" in f.lower()
    ]
    if not meta_files:
        # Also check plain .txt files that are actually meta files
        meta_files = [f for f in os.listdir(folder)
                      if f.lower().endswith(".txt") and not f.lower().endswith("_ephemeris.txt")]

    result: Dict = {
        "found": False,
        "file": None,
        "sat_id": None,
        "orbit_number": None,
        "task_id": None,
        "image_start_time": None,
        "imaging_duration": None,
        "total_frames": 0,
        "lat_start": None,
        "lat_end": None,
        "lon_start": None,
        "lon_end": None,
        "lat_range": None,
        "lon_range": None,
        "bands_used": None,
        "active_band_indices": [],
        "inactive_band_indices": [],
        "time_counter_first": None,
        "time_counter_last": None,
        "time_counter_span_ns": None,
        "config_and_tdi_file": None,
        "sensor_temperature_raw": None,
        "sensor_temperature_c": None,
        "core_temperature_c": None,
        "width": None,
        "height": None,
        "total_frames_meta": None,
        "binning": None,
        "tdi_mode": None,
        "tdi_stages": None,
        "band_height": None,
        "proc_mode_string": None,
        "pps_ref_values": [],
        "time_ref_values": [],
        "ppssref_nonzero": False,
        "timeref_nonzero": False,
    }

    # Find the right .meta file (prefer one with dataset name, not ephemeris)
    target = None
    for mf in sorted(meta_files):
        if "_ephemeris" not in mf.lower():
            target = os.path.join(folder, mf)
            result["file"] = mf
            break
    if not target:
        return result

    try:
        content = Path(target).read_text(errors="ignore")
    except Exception:
        return result

    result["found"] = True

    # Frame-level fields
    lats         = [float(x) for x in re.findall(r'Latitude\s*:\s*([\d.]+)', content)]
    lons         = [float(x) for x in re.findall(r'Longitude\s*:\s*([\d.-]+)', content)]
    time_ctrs    = [int(x)   for x in re.findall(r'TimeCounter\s*:\s*(\d+)', content)]
    pps_refs     = [int(x)   for x in re.findall(r'PPSRef\s*:\s*(\d+)', content)]
    time_refs    = [int(x)   for x in re.findall(r'TimeRef\s*:\s*(\d+)', content)]
    frame_nums   = [int(x)   for x in re.findall(r'Meta Chunks Frame\s+(\d+)', content)]
    bands_used   = [int(x)   for x in re.findall(r'bandsUsed\s*:\s*(\d+)', content)]

    # Satellite identity (from first frame)
    m = re.search(r'SAT_ID\s*:\s*(\S+)', content)
    if m: result["sat_id"] = m.group(1)

    m = re.search(r'OrbitNumber\s*:\s*(\d+)', content)
    if m: result["orbit_number"] = int(m.group(1))

    m = re.search(r'Task_ID\s*:\s*(\d+)', content)
    if m: result["task_id"] = int(m.group(1))

    m = re.search(r'ImageStartTime\s*:\s*(\d+)', content)
    if m: result["image_start_time"] = int(m.group(1))

    m = re.search(r'ImagingDuration\s*:\s*(\d+)', content)
    if m: result["imaging_duration"] = int(m.group(1))

    m = re.search(r'ConfigAndTDIFile\s*:\s*(\d+)', content)
    if m: result["config_and_tdi_file"] = int(m.group(1))

    # Ground track
    if lats:
        result["lat_start"] = lats[0]
        result["lat_end"]   = lats[-1]
        result["lat_range"] = round(max(lats) - min(lats), 4)
    if lons:
        result["lon_start"] = lons[0]
        result["lon_end"]   = lons[-1]
        result["lon_range"] = round(max(lons) - min(lons), 4)

    # Frame count
    if frame_nums:
        result["total_frames"] = max(frame_nums) + 1

    # Timestamps
    if time_ctrs:
        result["time_counter_first"]   = time_ctrs[0]
        result["time_counter_last"]    = time_ctrs[-1]
        result["time_counter_span_ns"] = time_ctrs[-1] - time_ctrs[0]

    # PPS / TimeRef anomaly detection
    result["pps_ref_values"]    = sorted(set(pps_refs))
    result["time_ref_values"]   = sorted(set(time_refs))
    result["ppssref_nonzero"]   = any(v != 0 for v in pps_refs)
    result["timeref_nonzero"]   = any(v != 0 for v in time_refs)

    # Bands configuration
    if bands_used:
        result["bands_used"] = max(set(bands_used), key=bands_used.count)  # mode

    # Per-band active/inactive: band1Active..band7Active
    # Values: 1-7 = active band index, 255 = inactive sentinel
    active, inactive = [], []
    for i in range(1, 8):
        vals = re.findall(rf'band{i}Active\s*:\s*(\d+)', content)
        if vals:
            v = int(vals[0])
            if v == 255:
                inactive.append(i)
            else:
                active.append(i)
    result["active_band_indices"]   = active
    result["inactive_band_indices"] = inactive

    # Params Chunks section (at end of file)
    params_m = re.search(r'Params Chunks\s*={3,}(.+?)={3,}', content, re.DOTALL)
    if params_m:
        pc = params_m.group(1)
        def _pv(key, cast=str):
            mm = re.search(rf'{key}\s*:\s*(\S+)', pc)
            return cast(mm.group(1)) if mm else None
        result["width"]             = _pv("width", int)
        result["height"]            = _pv("Height", int)
        result["total_frames_meta"] = _pv("totalFrames", int)
        result["binning"]           = _pv("binning", int)
        result["tdi_mode"]          = _pv("TDIMode", int)
        result["tdi_stages"]        = _pv("TDIStages", int)
        result["band_height"]       = _pv("bandHeight", int)
        ct = _pv("coreTemperature", int)
        if ct is not None: result["core_temperature_c"] = float(ct)
        st = _pv("sensorTemperature", int)
        if st is not None:
            result["sensor_temperature_raw"] = st
            result["sensor_temperature_c"]   = round(st / 100.0, 2)
        pm = re.search(r'mode\s*:\s*(.+)', pc)
        if pm: result["proc_mode_string"] = pm.group(1).strip()

    return result


# ══════════════════════════════════════════════════════════════════════════════
# EPHEMERIS FILE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_ephemeris_file(folder: str) -> Dict:
    """
    Parse the *_ephemeris.txt file in a dataset folder.
    Returns orbital parameter statistics across all records.
    """
    eph_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith("_ephemeris.txt")
    ]
    result: Dict = {
        "found": False,
        "file": None,
        "total_records": 0,
        "orbit_time_start": None,
        "orbit_time_end": None,
        "alt_min_km": None,
        "alt_max_km": None,
        "alt_mean_km": None,
        "lat_min": None,
        "lat_max": None,
        "lon_min": None,
        "lon_max": None,
        "beta_angle_min": None,
        "beta_angle_max": None,
        "beta_angle_mean": None,
        "velocity_z_mean": None,
        "all_valid": False,
        "validity_flags": [],
        "ground_speed_kmps": None,
        "gsd_along_track_m": None,
        "attitude_available": False,
    }

    if not eph_files:
        return result

    eph_path = os.path.join(folder, eph_files[0])
    result["file"] = eph_files[0]

    try:
        content = Path(eph_path).read_text(errors="ignore")
    except Exception:
        return result

    result["found"] = True

    # Parse all numeric fields
    altitudes   = [float(x) for x in re.findall(r'altitude\s*:\s*([\d.]+)', content)]
    lats        = [float(x) for x in re.findall(r'\blat(?:itude)?\s*:\s*([\d.]+)', content)
                   if 'nadir' not in content[max(0, content.find(x)-50):content.find(x)+50]]
    lons        = [float(x) for x in re.findall(r'\blon(?:gitude)?\s*:\s*([\d.-]+)', content)]
    betas       = [float(x) for x in re.findall(r'beta_angle\s*:\s*([\d.]+)', content)]
    orbit_times = [float(x) for x in re.findall(r'orbit_time\s*:\s*([\d.]+)', content)]
    eci_vz      = [float(x) for x in re.findall(r'eci_velocity_z\s*:\s*([\d.]+)', content)]
    validity    = [int(x)   for x in re.findall(r'validity_flags\s*:\s*(\d+)', content)]
    parts       = re.findall(r'Ephemeris Part (\d+)', content)

    result["total_records"] = len(parts)

    if orbit_times:
        result["orbit_time_start"] = orbit_times[0]
        result["orbit_time_end"]   = orbit_times[-1]

    if altitudes:
        result["alt_min_km"]  = round(min(altitudes), 2)
        result["alt_max_km"]  = round(max(altitudes), 2)
        result["alt_mean_km"] = round(statistics.mean(altitudes), 2)

    if lats:
        result["lat_min"] = round(min(lats), 4)
        result["lat_max"] = round(max(lats), 4)
    if lons:
        result["lon_min"] = round(min(lons), 4)
        result["lon_max"] = round(max(lons), 4)

    if betas:
        result["beta_angle_min"]  = round(min(betas), 2)
        result["beta_angle_max"]  = round(max(betas), 2)
        result["beta_angle_mean"] = round(statistics.mean(betas), 2)

    if eci_vz:
        vz = statistics.mean(eci_vz)
        result["velocity_z_mean"]    = round(vz, 3)
        # Ground speed heuristic: LEO ~7.5 km/s, use vz as proxy
        result["ground_speed_kmps"]  = round(abs(vz), 3)

    if validity:
        result["validity_flags"] = sorted(set(validity))
        result["all_valid"]      = all(v == 65535 for v in validity)

    # Attitude quaternion presence
    result["attitude_available"] = bool(re.search(r'att_quat_1', content))

    # GSD estimate using mean altitude and ground speed
    alt_m = (result["alt_mean_km"] or 500.0) * 1000.0
    gs    = (result["ground_speed_kmps"] or 7.5) * 1000.0  # m/s
    # FPS from log not available here — placeholder, caller should fill
    result["gsd_along_track_m"] = None  # filled in merge_summaries()

    return result


def merge_summaries(log_summary: Dict, meta_sum: Dict,
                    ephem_sum: Dict, fps: float = None) -> Dict:
    """
    Merge log, meta, and ephemeris summaries into a single enriched dict.
    Also computes GSD using FPS from log and altitude from ephemeris.
    """
    merged = {
        "log": log_summary,
        "meta": meta_sum,
        "ephem": ephem_sum,
    }

    # GSD: need fps (from log) + altitude (from ephemeris)
    if fps is None and log_summary:
        fps = (log_summary.get("parameters_applied", {}).get("FPS") or
               log_summary.get("procmode", {}).get("decoded", {}).get("fps_requested"))
    alt_km = ephem_sum.get("alt_mean_km") if ephem_sum else None
    gs_kms = ephem_sum.get("ground_speed_kmps") if ephem_sum else None

    if fps and alt_km and gs_kms:
        gsd_m = (gs_kms * 1000.0) / fps
        merged["gsd_along_track_m"] = round(gsd_m, 1)
        if ephem_sum:
            ephem_sum["gsd_along_track_m"] = round(gsd_m, 1)

    # Cross-validation findings
    findings = []

    if meta_sum and meta_sum.get("found"):
        # Sensor temperature cross-check
        sensor_c_meta = meta_sum.get("sensor_temperature_c")
        temps_log     = (log_summary or {}).get("temperatures", {})
        sensor_c_log  = temps_log.get("sensor_before_C") or temps_log.get("sensor_after_C")
        if sensor_c_meta and sensor_c_log:
            delta = abs(sensor_c_meta - sensor_c_log)
            if delta > 2.0:
                findings.append({
                    "source": "cross_validation",
                    "severity": "WARNING",
                    "message": (
                        f"Sensor temperature mismatch: meta={sensor_c_meta:.1f}°C "
                        f"vs log={sensor_c_log:.1f}°C (Δ{delta:.1f}°C). "
                        f"Check which source is authoritative."
                    )
                })

        # Frame count cross-check
        fc_meta = meta_sum.get("total_frames") or meta_sum.get("total_frames_meta")
        fa_log  = (log_summary or {}).get("frame_accounting", {})
        fc_log  = fa_log.get("captured_count") or fa_log.get("total_frames_expected")
        if fc_meta and fc_log and fc_meta != fc_log:
            findings.append({
                "source": "cross_validation",
                "severity": "WARNING",
                "message": (
                    f"Frame count mismatch: meta has {fc_meta} frames, "
                    f"log reports {fc_log}. Dataset may be incomplete."
                )
            })

        # Band count cross-check
        bands_used_meta = meta_sum.get("bands_used")
        proc = (log_summary or {}).get("procmode", {})
        bs_log = proc.get("band_selection", {}).get("active_count")
        if bands_used_meta and bs_log and int(bands_used_meta) != int(bs_log):
            findings.append({
                "source": "cross_validation",
                "severity": "INFO",
                "message": (
                    f"Band count note: meta reports bandsUsed={bands_used_meta}, "
                    f"log BandSelection shows {bs_log} active. "
                    f"This can occur with split-region configurations."
                )
            })

    merged["cross_validation_findings"] = findings
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# MISSION TYPE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

# Mission type signatures — ordered by specificity (most specific first)
_MISSION_SIGNATURES = [
    # (mission_type, confidence_threshold, list_of_(regex, weight) pairs)
    ("test_pattern_calibration", 0.7, [
        (r'TestPattern\s*[=:]\s*[1-9]', 10),
        (r'\[I31\].*TestPattern', 5),
        (r'test.*pattern.*on', 3),
    ]),
    ("dark_calibration", 0.6, [
        (r'dark.*cap|covered.*lens|lens.*cap|shutter.*closed', 6),
        (r'exposure.*\b[5-9]\d{3,}', 3),  # very long exposure
        (r'TDI.*OFF|TDI_Modes=0', 2),
    ]),
    ("tdi_high_altitude", 0.6, [
        (r'TDI_Stages.*64|TDI.*64.stage', 8),
        (r'altitude.*[5-9]\d{2}|[56]\d{2}.*km', 4),
        (r'TDI_Modes=4|tdi.*byte.*66', 4),
    ]),
    ("tdi_standard", 0.5, [
        (r'TDI_Stages.*8|TDI.*8.stage', 7),
        (r'TDI_Modes=2|tdi.*byte.*34', 5),
        (r'RegionHeight.*384', 2),
    ]),
    ("tdi_low", 0.5, [
        (r'TDI_Stages.*[24]|TDI.*(2|4).stage', 7),
        (r'TDI_Modes=1|tdi.*byte.*(10|18)', 5),
    ]),
    ("no_tdi_full_frame", 0.5, [
        (r'TDI_Modes=0|TDI.*OFF|tdi.*byte.*0\b', 8),
        (r'BandSelection.*127|all.*7.*bands', 3),
        (r'FPS.*1[0-9]\.|fps.*1[0-9]\.', 2),
    ]),
    ("partial_band_capture", 0.5, [
        (r'Region\dMode:0', 4),
        (r'band.*OFF|bandsUsed.*[1-6]\b', 3),
        (r'BandSelection.*(?:30|60|14|62)\b', 3),
    ]),
    ("high_fps_pushbroom", 0.5, [
        (r'FPS.*[5-9]\d\.|fps.*[5-9]\d\.', 7),
        (r'MaxFPS.*\d{3}', 3),
    ]),
    ("low_fps_integration", 0.5, [
        (r'FPS.*[1-9]\.|fps.*[1-9]\.', 5),
        (r'ExposureTime.*[2-9]\d{4}', 4),  # long exposure
    ]),
    ("unknown", 0.0, []),
]


def detect_mission_type(log_content: str, log_summary: Dict = None,
                         meta_sum: Dict = None) -> Dict:
    text = (log_content or "").lower()

    # Extract key parameters for context
    params = {}
    if log_summary:
        proc = log_summary.get("procmode", {}).get("decoded", {})
        params["fps"]          = proc.get("fps_requested")
        params["tdi_byte"]     = proc.get("tdi_byte")
        params["band_sel"]     = proc.get("band_selection")
        params["exposure"]     = proc.get("exposure_time")
        params["gain"]         = proc.get("gain")
        params["duration"]     = proc.get("duration_sec")
        params["tdi_stages"]   = (log_summary.get("parameters_applied", {})
                                  .get("TDI_Stages") or
                                  log_summary.get("parameters_applied", {})
                                  .get("TDIStages"))
        params["test_pattern"] = log_summary.get("test_pattern", 0)
        params["firmware"]     = log_summary.get("firmware_version", "")

    if meta_sum and meta_sum.get("found"):
        params["alt_km"]       = None  # from ephem
        params["bands_used"]   = meta_sum.get("bands_used")
        params["tdi_mode_meta"]= meta_sum.get("tdi_mode")
        params["tdi_stages_meta"]= meta_sum.get("tdi_stages")

    # Score each mission type
    scores  = {}
    evidence= {}
    for mtype, threshold, sigs in _MISSION_SIGNATURES:
        score = 0
        evs   = []
        for pattern, weight in sigs:
            if re.search(pattern, text, re.I):
                score += weight
                evs.append(pattern)
        scores[mtype]   = score
        evidence[mtype] = evs

    # Normalise: best possible score = sum of all weights for that type
    def _max_score(mtype):
        for mt, _, sigs in _MISSION_SIGNATURES:
            if mt == mtype:
                return sum(w for _, w in sigs) or 1
        return 1

    norm_scores = {mt: s / _max_score(mt) for mt, s in scores.items()}

    # Parameter overrides (direct evidence overrides pattern matching)
    if params.get("test_pattern") and params["test_pattern"] != 0:
        norm_scores["test_pattern_calibration"] = 1.0

    # Pick best
    best_type  = max(norm_scores, key=norm_scores.get)
    confidence = norm_scores[best_type]

    # Fallback
    if confidence < 0.2:
        best_type  = "unknown"
        confidence = 0.0

    return {
        "mission_type": best_type,
        "confidence":   round(confidence, 3),
        "scores":       {k: round(v, 3) for k, v in norm_scores.items() if v > 0},
        "evidence":     evidence.get(best_type, []),
        "parameters":   params,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE LEARNER
# ══════════════════════════════════════════════════════════════════════════════

_LEARNER_FILE = migrate_legacy_file(
    get_app_data_path(".iris_templates.json"),
    os.path.join(os.path.dirname(__file__), ".iris_templates.json")
)
_learner_lock = threading.Lock()


def _load_templates() -> Dict:
    try:
        if os.path.exists(_LEARNER_FILE):
            with open(_LEARNER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_templates(templates: Dict):
    try:
        with open(_LEARNER_FILE, "w", encoding="utf-8") as f:
            json.dump(templates, f, indent=2)
    except Exception as e:
        print(f"[TemplateLearner] Save failed: {e}")


def learn_from_scan(log_summary: Dict, meta_sum: Dict = None,
                     mission_type: str = "unknown") -> Dict:
    """
    Update the template baseline for a given mission type using
    observed parameter values from this scan.

    Tracked fields per mission type:
      sensor_temp, core_temp, fps, exposure, gain, health_score,
      frames_captured, bands_used

    Uses Welford's online algorithm for running mean + variance.
    """
    if not log_summary:
        return {"learned": False, "reason": "No log summary"}

    # Extract observable values
    temps   = log_summary.get("temperatures", {})
    fa      = log_summary.get("frame_accounting", {})
    proc    = log_summary.get("procmode", {}).get("decoded", {})
    p_app   = log_summary.get("parameters_applied", {})

    obs = {
        "sensor_temp":      temps.get("sensor_before_C"),
        "core_temp":        temps.get("core_before_C"),
        "fps":              p_app.get("FPS") or proc.get("fps_requested"),
        "exposure":         p_app.get("ExposureTime") or proc.get("exposure_time"),
        "gain":             p_app.get("Gain") or proc.get("gain"),
        "frames_captured":  fa.get("captured_count"),
        "bands_used":       (log_summary.get("procmode", {})
                             .get("band_selection", {}).get("active_count")),
    }
    if meta_sum and meta_sum.get("found"):
        obs["sensor_temp_meta"] = meta_sum.get("sensor_temperature_c")
        obs["core_temp_meta"]   = meta_sum.get("core_temperature_c")

    with _learner_lock:
        templates = _load_templates()
        profile   = templates.setdefault(mission_type, {
            "mission_type": mission_type,
            "sample_count": 0,
            "fields": {}
        })

        n = profile["sample_count"] + 1
        profile["sample_count"] = n

        updated = []
        for field, value in obs.items():
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            f = profile["fields"].setdefault(field, {
                "n": 0, "mean": value, "M2": 0.0,
                "min": value, "max": value
            })

            # Welford online update
            old_mean = f["mean"]
            f["n"] += 1
            delta  = value - old_mean
            f["mean"] += delta / f["n"]
            delta2 = value - f["mean"]
            f["M2"]  += delta * delta2
            f["min"]  = min(f["min"], value)
            f["max"]  = max(f["max"], value)
            # Variance = M2 / (n-1) if n > 1 else 0
            f["std"] = math.sqrt(f["M2"] / max(f["n"] - 1, 1))
            updated.append(field)

        profile["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_templates(templates)

    return {
        "learned":      True,
        "mission_type": mission_type,
        "sample_count": n,
        "fields_updated": updated,
    }


def flag_template_deviations(log_summary: Dict, meta_sum: Dict = None,
                               mission_type: str = "unknown",
                               sigma_threshold: float = 2.5) -> List[Dict]:
    """
    Compare observed values against the learned baseline for this mission type.
    Flags anomalies where |observed - mean| > sigma_threshold * std.

    Requires at least 3 samples before flagging (avoids noise on fresh templates).

    Returns list of deviation dicts: {field, observed, mean, std, sigma, severity, message}
    """
    if not log_summary:
        return []

    with _learner_lock:
        templates = _load_templates()
    profile = templates.get(mission_type)
    if not profile or profile.get("sample_count", 0) < 3:
        return []

    temps  = log_summary.get("temperatures", {})
    fa     = log_summary.get("frame_accounting", {})
    proc   = log_summary.get("procmode", {}).get("decoded", {})
    p_app  = log_summary.get("parameters_applied", {})

    obs = {
        "sensor_temp":     temps.get("sensor_before_C"),
        "core_temp":       temps.get("core_before_C"),
        "fps":             p_app.get("FPS") or proc.get("fps_requested"),
        "exposure":        p_app.get("ExposureTime") or proc.get("exposure_time"),
        "gain":            p_app.get("Gain") or proc.get("gain"),
        "frames_captured": fa.get("captured_count"),
    }
    if meta_sum and meta_sum.get("found"):
        obs["sensor_temp_meta"] = meta_sum.get("sensor_temperature_c")

    deviations = []
    for field, value in obs.items():
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        baseline = profile["fields"].get(field)
        if not baseline or baseline.get("n", 0) < 3:
            continue
        if baseline.get("std", 0) < 0.001:
            continue  # zero-variance field — skip

        mean  = baseline["mean"]
        std   = baseline["std"]
        sigma = abs(value - mean) / std

        if sigma < sigma_threshold:
            continue

        # Severity: >4σ = WARNING, >2.5σ = INFO
        severity = "WARNING" if sigma > 4.0 else "INFO"

        direction = "above" if value > mean else "below"
        deviations.append({
            "field":    field,
            "observed": round(value, 3),
            "mean":     round(mean, 3),
            "std":      round(std, 3),
            "sigma":    round(sigma, 2),
            "severity": severity,
            "message": (
                f"[Template] {field} = {value:.2f} is {sigma:.1f}σ {direction} "
                f"the baseline mean of {mean:.2f} ± {std:.2f} "
                f"(from {baseline['n']} {mission_type} sessions). "
                f"{'Investigate — significant deviation.' if severity == 'WARNING' else 'Noted.'}"
            ),
        })

    return deviations


def get_template_summary() -> Dict:
    """Return all learned templates with their baselines — for display/debug."""
    with _learner_lock:
        templates = _load_templates()
    result = {}
    for mtype, profile in templates.items():
        fields_summary = {}
        for field, f in profile.get("fields", {}).items():
            fields_summary[field] = {
                "n":    f["n"],
                "mean": round(f["mean"], 3),
                "std":  round(f.get("std", 0), 3),
                "min":  round(f["min"], 3),
                "max":  round(f["max"], 3),
            }
        result[mtype] = {
            "sample_count": profile.get("sample_count", 0),
            "last_updated": profile.get("last_updated", ""),
            "fields":       fields_summary,
        }
    return result


def reset_template(mission_type: str = None) -> Dict:
    """Reset one template (or all if mission_type is None)."""
    with _learner_lock:
        templates = _load_templates()
        if mission_type:
            removed = mission_type if mission_type in templates else None
            templates.pop(mission_type, None)
        else:
            removed = list(templates.keys())
            templates = {}
        _save_templates(templates)
    return {"reset": removed or "none"}


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_full_report(scan_result, folder: str,
                          mission_type: str = "unknown",
                          template_deviations: List[Dict] = None,
                          kb_excerpts: Dict[str, str] = None,
                          enable_template_comparison: bool = False) -> str:
    """
    Generate the complete structured Iris report from a ScanResult.
    Merges log findings with .meta and ephemeris data.
    Adds template-learning deviation flags when enable_template_comparison=True.
    Adds KB context where available.

    This replaces tool_generate_report() in tools.py — same output format,
    extended with 3-file data and template flags.
    """
    from .app_state import ScanResult  # local import to avoid circular
    sc    = scan_result
    log   = sc.log_summary or {}
    meta  = sc.meta_summary or {}
    ephem = sc.ephem_summary or {}
    name  = os.path.basename(folder)
    devs  = template_deviations or []
    kb    = kb_excerpts or {}

    # Auto-compute template deviations if requested and none provided
    if enable_template_comparison and not devs:
        devs = flag_template_deviations(log, meta, mission_type)
    
    # Clear deviations if template comparison is disabled
    if not enable_template_comparison:
        devs = []

    lines = []

    def section(title):
        lines.append("")
        lines.append(f"{'─'*62}")
        lines.append(f"  {title}")
        lines.append(f"{'─'*62}")

    def finding(severity, text, note=None):
        icon = {"CRITICAL":"🔴","WARNING":"🟡","INFO":"⚪","OK":"✅"}.get(severity, "•")
        lines.append(f"  {icon} [{severity}] {text}")
        if note:
            for ln in note.split(". "):
                ln = ln.strip()
                if ln: lines.append(f"      → {ln}.")

    # ── Header ────────────────────────────────────────────────────────────
    lines.append(f"╔══════════════════════════════════════════════════════════════════╗")
    lines.append(f"  IRIS ACQUISITION REPORT")
    lines.append(f"  Dataset : {name}")
    src_tag = "log"
    if meta.get("found"): src_tag += " + .meta"
    if ephem.get("found"): src_tag += " + ephemeris"
    lines.append(f"  Sources : {src_tag}")
    if mission_type != "unknown":
        lines.append(f"  Mission : {mission_type.replace('_',' ').title()}")
    lines.append(f"  Scan    : {sc.scan_type}  |  "
                 f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sc.scanned_at))}")
    lines.append(f"  Duration: {sc.duration_sec}s")
    lines.append(f"╚══════════════════════════════════════════════════════════════════╝")

    critical = [f for f in sc.findings if f["severity"] == "CRITICAL"]
    warnings = [f for f in sc.findings if f["severity"] == "WARNING"]
    infos    = [f for f in sc.findings if f["severity"] == "INFO"]

    # Add template deviations to counts
    t_warn = [d for d in devs if d["severity"] == "WARNING"]
    t_info = [d for d in devs if d["severity"] == "INFO"]

    verdict = ("✅ OK" if not critical and len(warnings) + len(t_warn) <= 1 else
               "🟡 PATTERNS NOTED" if not critical else "🔴 PROBLEMS FOUND")
    lines.append(f"")
    lines.append(f"  HEALTH: {sc.health_score:.0f}/100  {verdict}")
    lines.append(f"  Summary: {len(critical)} confirmed · {len(warnings)+len(t_warn)} warnings "
                 f"· {len(infos)+len(t_info)} info")
    visible_summary = []
    for f in critical[:2]:
        visible_summary.append(("🔴", f.get("message", "")))
    for f in warnings[:3]:
        visible_summary.append(("🟡", f.get("message", "")))
    for f in infos[:2]:
        visible_summary.append(("⚪", f.get("message", "")))
    if visible_summary:
        lines.append("  Top findings:")
        for icon, msg in visible_summary:
            msg = (msg or "").replace("\n", " ").strip()
            lines.append(f"    {icon} {msg}")
    if devs:
        lines.append(f"  Template flags: {len(devs)} deviation(s) vs {mission_type} baseline")

    # ── Section 0: Mission Identity ───────────────────────────────────────
    section("0. MISSION IDENTITY")
    sysinfo = log.get("system_info", {})
    hwinfo  = log.get("hardware_info", {})

    if meta.get("sat_id"):
        lines.append(f"    Satellite    : {meta['sat_id']}")
    if meta.get("orbit_number"):
        lines.append(f"    Orbit Number : {meta['orbit_number']}")
    if meta.get("task_id"):
        lines.append(f"    Task ID      : {meta['task_id']}")
    if sysinfo.get("app_version"):
        lines.append(f"    App          : Xdlinx Cam App v{sysinfo['app_version']}")
    fw = log.get("firmware_version", "")
    if fw:
        lines.append(f"    Firmware     : {fw}")
    if hwinfo.get("camera_model"):
        lines.append(f"    Camera       : {hwinfo['camera_model']}")
    if hwinfo.get("grabber_model"):
        lines.append(f"    Grabber      : {hwinfo['grabber_model']}")
    if sysinfo.get("cpu_model"):
        lines.append(f"    CPU          : {sysinfo['cpu_model']} ({sysinfo.get('cpu_cores','?')} cores)")
    if sysinfo.get("disk_free_gb"):
        lines.append(f"    Disk free    : {sysinfo['disk_free_gb']:.2f} GB")
    if sysinfo.get("memory_free_gb"):
        lines.append(f"    RAM free     : {sysinfo['memory_free_gb']:.2f} GB")
    pci_gen   = hwinfo.get("pci_generation")
    pci_lanes = hwinfo.get("pci_lanes")
    if pci_gen is not None and pci_lanes is not None:
        lines.append(f"    PCIe         : Gen {pci_gen} × {pci_lanes} lanes")

    # ── Section 1: Parameter File ─────────────────────────────────────────
    section("1. PARAMETER FILE CHECK (14 items)")
    arg_count  = log.get("raw_arg_count", 0)
    proc_count = log.get("proc_arg_count", 0)
    raw_line   = log.get("raw_args_line", "")
    if arg_count == 14 and proc_count == 14:
        finding("OK", "All 14 parameters found and processed.")
    elif arg_count == 14 and proc_count != 14:
        finding("WARNING",
                f"All 14 raw parameters were found, but log reports Argument Processed[{proc_count}] instead of 14.",
                "Parsing discrepancy detected between the raw parameter line and the processed argument count")
    elif arg_count > 0:
        finding("CRITICAL", f"Only {arg_count}/14 parameters found.",
                f"Raw: {raw_line}. Missing values default to unknown.")
    else:
        finding("INFO", "No raw argument line found in log.")

    pdi = log.get("param_date_info", {})
    if pdi.get("discrepancy_days", 0) > 0:
        finding("WARNING",
                f"Parameter file date stale by {pdi['discrepancy_days']} days "
                f"— I07: {pdi.get('i07_date','?')}, corrected: {pdi.get('i28_date','?')}.",
                "Update parameter file date before next mission.")

    proc = log.get("procmode", {})
    d    = proc.get("decoded", {})
    if d:
        tdi_dec = proc.get("tdi_decoded", {})
        bn_dec  = proc.get("binning_decoded", {})
        bs_dec  = proc.get("band_selection", {})
        lines.append("")
        lines.append("  Parameter values:")
        lines.append(f"    [1]  OrbitID        = {d.get('orbit_id')}")
        lines.append(f"    [2]  TaskID         = {d.get('task_id')}")
        lines.append(f"    [3]  JsonID         = {d.get('json_id')}")
        lines.append(f"    [4]  Date           = {d.get('date')}")
        lines.append(f"    [5]  UTC Time       = {d.get('utc_time')}")
        lines.append(f"    [6]  Duration       = {d.get('duration_sec')}s")
        lines.append(f"    [7]  BandSelection  = {d.get('band_selection')} "
                     f"({bs_dec.get('active_count','?')} bands active)")
        lines.append(f"    [8]  TDI byte       = {d.get('tdi_byte')} → {tdi_dec.get('mode')}")
        lines.append(f"    [9]  FPS            = {d.get('fps_requested')}")
        lines.append(f"    [10] ExposureTime   = {d.get('exposure_time')}µs")
        lines.append(f"    [11] Gain           = {d.get('gain')}×")
        lines.append(f"    [12] XShift         = {d.get('xshift')}")
        lines.append(f"    [13] Binning byte   = {d.get('binning_byte')} "
                     f"(CCSDS: {'ON' if bn_dec.get('ccsds_enabled') else 'OFF'})")
        lines.append(f"    [14] TDIYShift      = {d.get('tdi_yshift')}")

    # Binning verification gap — scanner always emits this; surface it in the report
    binning_issue = next(
        (i for i in log.get("raw_issues", []) if i.get("category") == "binning_unverified"),
        None
    )
    if binning_issue:
        finding("INFO", binning_issue["message"])
    else:
        bin_status = (log.get("capture_info") or {}).get("bin_status")
        if bin_status is not None:
            finding("OK", f"Binning confirmation found in log: Bin Status = {bin_status}")

    # JSON config file used (I32)
    json_cfg = log.get("hardware_info", {}).get("json_config_file")
    if json_cfg:
        lines.append(f"    JSON config      : {json_cfg}")

    # ── Section 2: Requested vs Applied ──────────────────────────────────
    section("2. REQUESTED VS APPLIED")
    checks = log.get("procmode_checks", [])
    p_app  = log.get("parameters_applied", {})
    if checks:
        for chk in checks:
            status = chk.get("status", "?")
            param  = chk.get("param")
            req    = chk.get("requested")
            app    = chk.get("applied")
            note   = chk.get("note")
            diff   = chk.get("diff_pct")
            sev = ("OK" if status == "OK" else
                   "CRITICAL" if status in ("FAILED","MISMATCH") else
                   "WARNING" if status == "AUTO_ADJUSTED" else "INFO")
            msg = f"{param}: {req} → {app}"
            if diff:
                msg += f" ({diff:.1f}% off)"
            finding(sev, msg, note if note else None)
    else:
        finding("INFO", "No cross-check data found in logs.")

    # Hardware limits — show MaxFPS and MaxExpTime so operator knows headroom
    hwinfo2     = log.get("hardware_info", {})
    capinfo2    = log.get("capture_info", {})
    max_fps_val = hwinfo2.get("max_fps")
    max_exp_val = capinfo2.get("max_exp_time_us")
    proc_dec2   = log.get("procmode", {}).get("decoded", {})
    fps_req_val = proc_dec2.get("fps_requested")
    exp_req_val = proc_dec2.get("exposure_time")
    if max_fps_val and fps_req_val:
        pct_fps = round(float(fps_req_val) / max_fps_val * 100, 1)
        lines.append(f"    Hardware MaxFPS      : {max_fps_val:.3f}  "
                     f"(requested {fps_req_val} — {pct_fps}% of hardware limit)")
    if max_exp_val and exp_req_val:
        pct_exp = round(float(exp_req_val) / max_exp_val * 100, 1)
        lines.append(f"    Hardware MaxExpTime  : {max_exp_val:.1f} µs  "
                     f"(requested {exp_req_val:.1f} µs — {pct_exp}% of hardware limit)")

    # Template deviation flags (parameter-related)
    param_devs = [d for d in devs if d["field"] in ("fps","exposure","gain")]
    for dev in param_devs:
        finding(dev["severity"], dev["message"])

    # ── Section 3: Frame Accounting ───────────────────────────────────────
    section("3. FRAME ACCOUNTING")
    fa = log.get("frame_accounting", {})
    if fa:
        total    = fa.get("total_frames_expected", "?")
        captured = fa.get("captured_count", "?")
        lost     = fa.get("frames_lost", 0) or 0
        drops    = fa.get("frame_drops_reported", 0) or 0
        if lost == 0 and drops == 0:
            finding("OK", f"No frame drops — {captured}/{total} captured")
        elif lost > 0:
            pct = (lost/total*100) if isinstance(total,int) and total > 0 else 0
            finding("CRITICAL", f"{lost} frames lost ({pct:.1f}%)")
        if drops > 0:
            finding("CRITICAL", f"Log reports {drops} frame drop(s).")
        lines.append(f"    Expected: {total}  |  Captured: {captured}  |  Lost: {lost}")

        # Cross-check with meta
        if meta.get("found"):
            fc_meta = meta.get("total_frames") or meta.get("total_frames_meta")
            if fc_meta and fc_meta != captured:
                finding("WARNING",
                        f"Meta frame count ({fc_meta}) differs from log ({captured}).")
    else:
        finding("INFO", "No frame accounting data in log.")

    # ── Section 4: Temperature ────────────────────────────────────────────
    section("4. TEMPERATURE")
    temps  = log.get("temperatures", {})
    hwinfo = log.get("hardware_info", {})
    if temps:
        sb = temps.get("sensor_before_C", "?")
        sa = temps.get("sensor_after_C",  "?")
        cb = temps.get("core_before_C",   "?")
        ca = temps.get("core_after_C",    "?")
        sd = temps.get("sensor_delta_C",  0) or 0
        cd = temps.get("core_delta_C",    0) or 0
        ss = temps.get("sensor_stability","?")

        # Grabber board temp (I99 — hardware-info phase, before camera connect)
        gtc = hwinfo.get("grabber_temp_c")
        if gtc is not None:
            lines.append(f"    Grabber board  : {gtc}°C  (I99 — at hardware init)")

        # Session sensor temp: I54 before capture → I54 after capture
        sb_txt = f"{sb:.2f}" if isinstance(sb, float) else sb
        sa_txt = f"{sa:.2f}" if isinstance(sa, float) else sa
        lines.append(f"    Sensor (log)   : {sb_txt}°C → {sa_txt}°C  (Δ {sd:.2f}°C)  [{ss}]")

        # Session core temp: I87 before capture → I87 after capture
        cb_txt = f"{cb:.1f}" if isinstance(cb, float) else cb
        ca_txt = f"{ca:.1f}" if isinstance(ca, float) else ca
        lines.append(f"    Core   (log)   : {cb_txt}°C → {ca_txt}°C  (Δ {cd:.1f}°C)")

        # Meta cross-check (sensorTemperature field in Params Chunks)
        if meta.get("found") and meta.get("sensor_temperature_c"):
            mc = meta["sensor_temperature_c"]
            lines.append(f"    Sensor (meta)  : {mc:.2f}°C  (raw {meta.get('sensor_temperature_raw','')})")
            if isinstance(sb, float) and abs(mc - sb) > 2.0:
                finding("WARNING",
                    f"Sensor temperature mismatch: log={sb:.2f}°C vs meta={mc:.2f}°C "
                    f"(Δ{abs(mc-sb):.2f}°C). Check which source is authoritative.")
        if meta.get("found") and meta.get("core_temperature_c"):
            lines.append(f"    Core   (meta)  : {meta['core_temperature_c']:.1f}°C")

        # Stability verdict
        if ss == "DRIFTING":
            finding("WARNING", f"Sensor drifted {sd:.2f}°C during capture.",
                    "May affect radiometric calibration.")
        else:
            finding("OK", "Temperatures stable.")
    else:
        finding("INFO", "No temperature data found in log.")

    # Template deviations for temperature
    temp_devs = [d for d in devs if "temp" in d["field"]]
    for dev in temp_devs:
        finding(dev["severity"], dev["message"])

    # ── Section 5: Trigger Timing ─────────────────────────────────────────
    section("5. TRIGGER TIMING")
    tt = log.get("trigger_timing", {})
    utc_times = tt.get("utc_trigger_times", [])
    sys_times = tt.get("system_times", [])
    if tt.get("stale_timestamp_detected"):
        stale_utc = utc_times[0] if utc_times else "?"
        stale_sys = sys_times[0] if sys_times else "?"
        wait_ms   = tt.get("waiting_time_msec", 0)
        finding("WARNING",
                f"[W01] Stale trigger: file={stale_utc} | system={stale_sys} | "
                f"delta={abs(wait_ms):,} ms → fell back to 5s default.",
                "GPS/orbital sync did NOT occur from file timestamp. "
                "Firmware corrected via I28. Update parameter file date/time.")
        if len(utc_times) >= 2:
            valid_wait = tt.get("waiting_time_msec_2")
            if valid_wait and 0 < valid_wait < 60000:
                finding("OK", f"Corrected trigger (I28): {utc_times[1]} — "
                         f"wait={valid_wait} ms ✅ valid.")
    elif utc_times:
        finding("OK", f"Trigger timing nominal — UTC: {utc_times[0]}")
    else:
        finding("INFO", "No trigger timing data found in log.")

    # ── Section 6: FPS Stability ──────────────────────────────────────────
    section("6. FPS STABILITY")
    timing = log.get("fps_stability", {}) or log.get("timing", {})
    fps_mean = timing.get("mean_fps") or timing.get("fps_mean")
    fps_std  = timing.get("std_fps") or timing.get("fps_std") or 0.0
    td_mean  = timing.get("time_diff_mean_ms")
    td_std   = timing.get("time_diff_std_ms") or timing.get("timing_std_ms") or 0.0
    n_frames = timing.get("frames_logged") or timing.get("n_frames")
    fps_status = timing.get("timing_stability") or timing.get("fps_status", "?")
    if fps_mean:
        td_mean_txt = f"{td_mean:.4f} ms" if isinstance(td_mean, (int, float)) else "?"
        lines.append(
            f"    Applied FPS: {fps_mean:.4f}  |  std: {fps_std:.4f}  "
            f"|  TimeDiff mean: {td_mean_txt}  |  Frames: {n_frames}"
        )
        if fps_status == "PERFECT":
            finding("OK", f"FPS stability: PERFECT — no jitter detected.")
        elif fps_status in ("STABLE", "EXCELLENT", "GOOD"):
            finding("OK", f"FPS stability: STABLE — minor jitter within tolerance.")
        else:
            finding("WARNING", f"FPS instability detected — std={fps_std:.4f}.")
    else:
        # Explain why there's no FPS data — log format mismatch vs genuinely absent
        n_log_frames = log.get("frame_accounting", {}).get("frames_in_log", 0)
        fps_req = log.get("procmode", {}).get("decoded", {}).get("fps_requested")
        if n_log_frames == 0:
            finding("INFO",
                "Per-frame FPS lines not found in log — timing jitter cannot be assessed. "
                "Log does not contain FrameNo/instantFps/TimeDifference entries for this capture. "
                "This is normal when the log was generated by a firmware build that omits "
                "per-frame callback logging.")
        else:
            finding("INFO",
                f"FPS stability data unavailable ({n_log_frames} frames logged, "
                "but per-frame timing lines were not parsed). "
                "Check log format against expected FrameNo/instantFps pattern.")
        if fps_req:
            lines.append(f"    Requested FPS    : {fps_req}")

    # ── Section 7: Camera Setup ───────────────────────────────────────────
    section("7. CAMERA SETUP")
    hwinfo  = log.get("hardware_info", {})
    sysinfo = log.get("system_info", {})
    init_issues = [i for i in log.get("raw_issues", []) if i.get("category") == "camera_init"]
    if not init_issues:
        finding("OK", "Camera connected. No setup issues.")
    for iss in init_issues:
        finding(iss["severity"], iss["message"])
    gct = hwinfo.get("grabber_connect_time_s")
    if gct is not None:
        if gct > 120:
            finding("WARNING", f"Grabber connection: {gct:.0f}s ({gct/60:.1f} min) — unusually slow.",
                    "Normal <30s. Check PCIe slot or camera power sequence.")
        elif gct > 10:
            finding("INFO", f"Grabber connected in {gct:.0f}s (slightly above normal <10s).")
        else:
            finding("OK", f"Grabber connected in {gct:.0f}s.")

    # PCIe connection info (I100/I101)
    pci_gen   = hwinfo.get("pci_generation")
    pci_lanes = hwinfo.get("pci_lanes")
    if pci_gen is not None and pci_lanes is not None:
        expected_lanes = 2
        if pci_lanes < expected_lanes:
            finding("WARNING",
                f"PCIe: Gen {pci_gen} × {pci_lanes} lanes — expected ×{expected_lanes} lanes. "
                "Bandwidth may be insufficient for full-speed capture. "
                "Check BIOS PCIe lane configuration or physical adapter isolation.")
        else:
            finding("OK", f"PCIe: Gen {pci_gen} × {pci_lanes} lanes — correct.")

    # Applied pixel height (I46) — cross-check against band count × band height
    app_h = hwinfo.get("applied_height")
    if app_h is not None:
        lines.append(f"    Applied pixel height : {app_h} px  "
                     f"(= active bands × RegionHeight / TDI_stages + 1 metadata line)")

    # Region mode summary (I43 block)
    rmodes = hwinfo.get("region_modes", {})
    if rmodes:
        on_regions  = [k for k, v in sorted(rmodes.items()) if v == 1]
        off_regions = [k for k, v in sorted(rmodes.items()) if v == 0]
        lines.append(f"    Regions ON   : {', '.join(on_regions) or 'none'}")
        lines.append(f"    Regions OFF  : {', '.join(off_regions) or 'none'}")

    # ReverseX/Y unrecognized — firmware does not support these params
    rx = hwinfo.get("reversex_unrecognized", False)
    ry = hwinfo.get("reversey_unrecognized", False)
    if rx or ry:
        params = ", ".join(p for p, flag in [("ReverseX", rx), ("ReverseY", ry)] if flag)
        finding("INFO",
            f"JSON config parameter(s) {params} not recognised by firmware {fw or '?'} "
            "— commands were silently ignored. Image flip settings were NOT applied. "
            "Remove these from the JSON config or upgrade firmware to avoid confusion.")

    # System resources
    mem = sysinfo.get("memory_free_gb")
    if mem is not None:
        lines.append(f"    Memory free      : {mem:.2f} GB")
    uptime = sysinfo.get("system_uptime")
    if uptime:
        lines.append(f"    System uptime    : {uptime}")

    if fw:
        finding("INFO", f"Firmware: {fw}")

    # ── Section 8: Orbital & Geolocation (meta + ephemeris) ─────────────
    if meta.get("found") or ephem.get("found"):
        section("8. ORBITAL & GEOLOCATION")
        if meta.get("sat_id"):
            lines.append(f"    SAT_ID       : {meta['sat_id']}")
        if meta.get("image_start_time"):
            import datetime
            try:
                dt = datetime.datetime.utcfromtimestamp(meta["image_start_time"])
                lines.append(f"    Imaging start: {meta['image_start_time']}  ({dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
            except Exception:
                lines.append(f"    Imaging start: {meta['image_start_time']}")
        if meta.get("lat_start") is not None:
            lines.append(f"    Ground track : {meta['lat_start']:.4f}°N,{meta['lon_start']:.4f}°E "
                         f"→ {meta['lat_end']:.4f}°N,{meta['lon_end']:.4f}°E")
            lines.append(f"    Coverage     : Δlat={meta.get('lat_range','?'):.2f}°  "
                         f"Δlon={meta.get('lon_range','?'):.2f}°")

        if ephem.get("found"):
            alt_min = ephem.get("alt_min_km")
            alt_max = ephem.get("alt_max_km")
            alt_mean = ephem.get("alt_mean_km")
            if alt_min is not None:
                lines.append(f"    Altitude     : {alt_min:.1f}–{alt_max:.1f} km  "
                             f"(mean {alt_mean:.0f} km)")
            beta_min = ephem.get("beta_angle_min")
            beta_max = ephem.get("beta_angle_max")
            if beta_min is not None:
                lines.append(f"    Beta angle   : {beta_min:.1f}–{beta_max:.1f}°")
            lines.append(f"    Validity     : "
                         f"{'✅ All valid (65535)' if ephem.get('all_valid') else '⚠ Some records invalid'}")

            # GSD — use pre-computed value from merge_summaries if available,
            # otherwise compute inline from ground speed + applied FPS
            gsd = (sc.merged_summary or {}).get("gsd_along_track_m") or                   ephem.get("gsd_along_track_m")
            if not gsd:
                fps_app = (log.get("parameters_applied", {}).get("FPS") or
                           log.get("procmode", {}).get("decoded", {}).get("fps_requested"))
                gs_kms  = ephem.get("ground_speed_kmps")
                if fps_app and gs_kms and float(fps_app) > 0:
                    gsd = round((gs_kms * 1000.0) / float(fps_app), 1)
            if gsd:
                lines.append(f"    GSD estimate : ~{gsd:.1f} m along-track")

            # Ground speed
            gs = ephem.get("ground_speed_kmps")
            if gs:
                lines.append(f"    Ground speed : {gs:.3f} km/s")

            # Attitude quaternions
            if ephem.get("attitude_available"):
                finding("OK",
                    "Attitude quaternions present — precise georeferencing supported.")
            else:
                finding("INFO",
                    "Attitude quaternions not found in ephemeris — "
                    "georeferencing accuracy may be reduced.")

            lines.append(f"    Ephem records: {ephem.get('total_records', '?')}")

        # ADCS + GPS ephemeris counts
        capinfo = log.get("capture_info", {})
        adcs = capinfo.get("adcs_ephemeris_count")
        gps_eph = capinfo.get("gps_ephemeris_count")
        if adcs is not None:
            if adcs > 0:
                finding("OK", f"ADCS ephemeris: {adcs} records stored per frame.")
            else:
                finding("WARNING", "No ADCS ephemeris stored — geolocation data unavailable.")
        if gps_eph is not None:
            if gps_eph > 0:
                finding("OK", f"GPS ephemeris: {gps_eph} records stored.")
            else:
                finding("INFO", "No GPS ephemeris records stored.")

        # PPS / TimeRef zero check (from meta)
        if meta.get("found"):
            pps_nonzero  = meta.get("ppssref_nonzero", False)
            time_nonzero = meta.get("timeref_nonzero", False)
            if not pps_nonzero and not time_nonzero:
                finding("INFO",
                    "PPSRef and TimeRef are zero in all meta frames — "
                    "PPS sync was not active. Sub-second timing precision relies "
                    "on TimeCounter (8 ns resolution) only.")
            elif pps_nonzero:
                finding("OK", "PPSRef non-zero — PPS synchronisation was active.")
            tc_span_ns = meta.get("time_counter_span_ns")
            if tc_span_ns is not None:
                tc_span_ms = tc_span_ns / 1_000_000
                lines.append(f"    TimeCounter span : {tc_span_ms:.1f} ms "
                             f"({tc_span_ns:,} ns)"  )

        # Cross-validation findings from merge_summaries
        for cv in (sc.merged_summary or {}).get("cross_validation_findings", []):
            finding(cv.get("severity", "INFO"), cv.get("message", ""))

    # ── Section 9: Band Configuration (meta cross-check) ────────────────
    if meta.get("found") and (meta.get("bands_used") or meta.get("active_band_indices")):
        section("9. BAND CONFIGURATION")
        lines.append(f"    bandsUsed (meta) : {meta.get('bands_used','?')}")
        if meta.get("active_band_indices"):
            lines.append(f"    Active bands     : {meta['active_band_indices']}")
        if meta.get("inactive_band_indices"):
            lines.append(f"    Inactive bands   : {meta['inactive_band_indices']}")
        if meta.get("band_height"):
            lines.append(f"    Band height      : {meta['band_height']} px")
        if meta.get("tdi_mode") is not None:
            lines.append(f"    TDI mode (meta)  : {meta['tdi_mode']}")
        if meta.get("tdi_stages") is not None:
            lines.append(f"    TDI stages (meta): {meta['tdi_stages']}")

        # Sensor width — flag if not the expected 8448 px full-width
        w_meta = meta.get("width")
        EXPECTED_WIDTH = 8448
        if w_meta is not None:
            if w_meta == EXPECTED_WIDTH:
                lines.append(f"    Width (meta)     : {w_meta} px ✅ (full sensor width)")
            elif w_meta == EXPECTED_WIDTH // 2:
                finding("INFO",
                    f"Sensor width = {w_meta} px — this is half the full sensor width "
                    f"({EXPECTED_WIDTH} px). Expected for a split-region or unbinned band "
                    "capture where each half is stored separately (.bandX0 / .bandX1). "
                    "Verify band file naming matches the capture configuration.")
            else:
                finding("WARNING",
                    f"Sensor width = {w_meta} px — unexpected value "
                    f"(standard full width: {EXPECTED_WIDTH} px, half-width: {EXPECTED_WIDTH//2} px). "
                    "Verify capture geometry and band file sizing.")

        # Frame count cross-validation (meta vs log)
        fc_meta = meta.get("total_frames") or meta.get("total_frames_meta")
        fa_log  = log.get("frame_accounting", {})
        fc_log  = fa_log.get("captured_count") or fa_log.get("total_frames_expected")
        if fc_meta and fc_log:
            if fc_meta == fc_log:
                finding("OK",
                    f"Frame count consistent: meta={fc_meta}, log={fc_log}.")
            else:
                finding("WARNING",
                    f"Frame count mismatch: meta reports {fc_meta} frames, "
                    f"log reports {fc_log}. Dataset may be incomplete or meta was "
                    "written before capture finished.")
        elif fc_meta:
            lines.append(f"    Total frames (meta): {fc_meta}")

    # ── Section 10: Template Learning Deviations ──────────────────────────
    if devs:
        section(f"10. TEMPLATE DEVIATIONS  (vs '{mission_type}' baseline)")
        lines.append(f"  Compared against {len(devs)} baseline field(s):")
        for dev in devs:
            finding(dev["severity"], dev["message"])
    else:
        section("10. TEMPLATE STATUS")
        with _learner_lock:
            tpls = _load_templates()
        profile = tpls.get(mission_type, {})
        n = profile.get("sample_count", 0)
        if n >= 3:
            finding("OK", f"Within normal range for '{mission_type}' ({n} reference sessions).")
        elif n > 0:
            finding("INFO", f"Template for '{mission_type}' has {n} sample(s) — need 3+ to flag deviations.")
        else:
            finding("INFO", f"No template yet for '{mission_type}' — this scan will seed it.")

    # ── Section 11: POST-CAPTURE ──────────────────────────────────────────
    section("11. POST-CAPTURE")
    capinfo = log.get("capture_info", {})
    if capinfo.get("file_size_bytes"):
        fsz = capinfo["file_size_bytes"] / 1e9
        frm = capinfo.get("frame_size_bytes", 0) / 1e6
        exp = capinfo.get("total_frames_expected", "?")
        finding("OK", f"Output: {fsz:.2f} GB  ({frm:.1f} MB/frame × {exp} frames)")
    dpt = capinfo.get("data_proc_time_s")
    if dpt is not None:
        label = "OK" if dpt < 5.0 else "INFO"
        finding(label, f"Data processing: {dpt:.3f}s")
    # Computed band height from log (I88 line)
    cbh = capinfo.get("computed_band_height")
    if cbh is not None:
        lines.append(f"    Computed band height : {cbh} px")
    if log.get("firmware_version"):
        fw_post = log["firmware_version"]
        known = {
            "2.3.2": "Known issue: sensor temperature values report abnormally low.",
            "2.2.2": "Stable firmware — temperature readings are credible.",
            "2.4.2": "Stable firmware — temperature readings are credible.",
        }
        note = known.get(fw_post.split()[0], "")
        finding("INFO", f"Firmware: {fw_post}", note if note else None)

    log_errors = log.get("errors", []) or []
    log_warnings = log.get("warnings", []) or []
    if log_errors or log_warnings:
        section("12. LOG ERROR SUMMARY")
        if log_errors:
            lines.append(f"    Error count   : {len(log_errors)}")
            for err in log_errors[:8]:
                sev = "WARNING" if "greater than default" in err.lower() else "CRITICAL"
                finding(sev, f"Log error: {err}")
        if log_warnings:
            lines.append(f"    Warning count : {len(log_warnings)}")
            for warn in log_warnings[:8]:
                finding("WARNING", f"Log warning: {warn}")

    # ── Footer ────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"{'═'*64}")
    lines.append(f"  END — {name}  |  Health: {sc.health_score:.0f}/100  |  Mission: {mission_type}")
    lines.append(f"{'═'*64}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BULK FILE SCANNER
# ══════════════════════════════════════════════════════════════════════════════

def bulk_scan_files(root_folder: str,
                    file_types: List[str] = None,
                    progress_cb=None) -> Dict:
    """
    Recursively walk root_folder and parse all .log + .meta + ephemeris files.
    file_types: list of extensions to include, e.g. [".log", ".meta"]
                Default: all three types.

    Returns structured dict with per-file summaries and cross-file analytics.
    """
    if file_types is None:
        file_types = [".log", ".meta", "_ephemeris.txt"]

    if not root_folder or not os.path.isdir(root_folder):
        return {"error": f"Folder not found: {root_folder}"}

    from .scanner import analyze_logs  # local import

    sessions: List[Dict] = []
    errors:   List[Dict] = []

    # Discover dataset folders (any folder with .log or .meta)
    candidate_folders = set()
    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            ext = fn.lower()
            if ext.endswith(".log") or ext.endswith(".meta") or ext.endswith("_ephemeris.txt"):
                candidate_folders.add(dirpath)

    total = len(candidate_folders)
    for i, folder in enumerate(sorted(candidate_folders)):
        if progress_cb:
            try:
                progress_cb(f"Scanning {os.path.basename(folder)} ({i+1}/{total})…")
            except Exception:
                pass
        try:
            # Always parse log
            log_sum  = {}
            log_files = [f for f in os.listdir(folder) if f.lower().endswith(".log")]
            if log_files and ".log" in " ".join(file_types):
                log_sum = analyze_logs(folder)

            # Parse meta if requested
            meta_sum = {}
            if any(".meta" in t for t in file_types):
                meta_sum = parse_meta_file(folder)

            # Parse ephemeris if requested
            ephem_sum = {}
            if any("ephemeris" in t for t in file_types):
                ephem_sum = parse_ephemeris_file(folder)

            # Mission detection from log content
            mission_type = "unknown"
            if log_sum.get("found", True):
                lc = ""
                for lf in log_files:
                    try:
                        lc += Path(os.path.join(folder, lf)).read_text(errors="ignore")
                    except Exception:
                        pass
                m_det = detect_mission_type(lc, log_sum, meta_sum)
                mission_type = m_det["mission_type"]

            # Extract key scalars for analytics
            temps = log_sum.get("temperatures", {})
            fa    = log_sum.get("frame_accounting", {})
            proc  = log_sum.get("procmode", {}).get("decoded", {})
            p_app = log_sum.get("parameters_applied", {})

            sessions.append({
                "name":           os.path.basename(folder),
                "folder":         folder,
                "mission_type":   mission_type,
                # Temperature
                "sensor_temp":    temps.get("sensor_before_C"),
                "sensor_temp_end":temps.get("sensor_after_C"),
                "sensor_delta":   temps.get("sensor_delta_C"),
                "core_temp":      temps.get("core_before_C"),
                "core_temp_meta": meta_sum.get("core_temperature_c") if meta_sum else None,
                "sensor_temp_meta":meta_sum.get("sensor_temperature_c") if meta_sum else None,
                # Capture params
                "fps":            p_app.get("FPS") or proc.get("fps_requested"),
                "exposure":       p_app.get("ExposureTime") or proc.get("exposure_time"),
                "gain":           p_app.get("Gain") or proc.get("gain"),
                "tdi_byte":       proc.get("tdi_byte"),
                "bands_active":   (log_sum.get("procmode",{})
                                   .get("band_selection",{}).get("active_count")),
                "firmware":       log_sum.get("firmware_version","?"),
                # Frame accounting
                "total_frames":   fa.get("total_frames_expected"),
                "captured":       fa.get("captured_count"),
                "drops":          fa.get("frames_lost", 0) or 0,
                # Issues
                "n_critical":     sum(1 for i in log_sum.get("raw_issues",[])
                                      if i.get("severity") == "CRITICAL"),
                "n_warnings":     sum(1 for i in log_sum.get("raw_issues",[])
                                      if i.get("severity") == "WARNING"),
                "stale_trigger":  log_sum.get("trigger_timing",{}).get("stale_timestamp_detected",False),
                "arg_count":      log_sum.get("raw_arg_count",0),
                # Geo
                "lat_start":      meta_sum.get("lat_start") if meta_sum else None,
                "lat_end":        meta_sum.get("lat_end") if meta_sum else None,
                "lon_start":      meta_sum.get("lon_start") if meta_sum else None,
                "alt_km":         ephem_sum.get("alt_mean_km") if ephem_sum else None,
                "sat_id":         meta_sum.get("sat_id") if meta_sum else None,
                "orbit_number":   meta_sum.get("orbit_number") if meta_sum else None,
                "has_meta":       bool(meta_sum and meta_sum.get("found")),
                "has_ephem":      bool(ephem_sum and ephem_sum.get("found")),
            })
        except Exception as e:
            errors.append({"folder": folder, "error": str(e)})

    if not sessions:
        return {
            "root": root_folder,
            "sessions": 0,
            "error": "No parseable files found.",
            "errors": errors,
        }

    # ── Cross-session analytics ──────────────────────────────────────────
    analytics = compute_cross_session_analytics(sessions)

    return {
        "root":      root_folder,
        "sessions":  len(sessions),
        "sessions_data": sessions,
        "analytics": analytics,
        "errors":    errors,
        "error_count": len(errors),
    }


def compute_cross_session_analytics(sessions: List[Dict]) -> Dict:
    """
    Compute aggregate statistics across multiple session records.
    Called by bulk_scan_files() and exposed as a standalone tool
    so the agent can run analytics on already-parsed data.

    Returns:
      - temperature stats (mean, min, max, std per field)
      - fps stats
      - frame drop summary
      - firmware distribution
      - mission type distribution
      - flagged outliers per field
    """

    def _stats(values: List[float]) -> Dict:
        vals = [v for v in values if v is not None]
        if not vals:
            return {"n": 0}
        n = len(vals)
        mn = statistics.mean(vals)
        return {
            "n":    n,
            "mean": round(mn, 3),
            "std":  round(statistics.pstdev(vals), 3) if n > 1 else 0.0,
            "min":  round(min(vals), 3),
            "max":  round(max(vals), 3),
            "median": round(statistics.median(vals), 3),
        }

    def _outliers(field, values_with_names, sigma=2.5):
        vals = [(n, v) for n, v in values_with_names if v is not None]
        if len(vals) < 3:
            return []
        raw = [v for _, v in vals]
        mn  = statistics.mean(raw)
        sd  = statistics.pstdev(raw)
        if sd < 0.001:
            return []
        return [
            {"name": nm, "value": round(v, 3), "sigma": round(abs(v-mn)/sd, 2)}
            for nm, v in vals if abs(v - mn) / sd > sigma
        ]

    # Field vectors
    def _vec(field):
        return [(s["name"], s.get(field)) for s in sessions]

    sensor_temps  = [s.get("sensor_temp")   for s in sessions]
    sensor_temps2 = [s.get("sensor_temp_meta") for s in sessions]
    core_temps    = [s.get("core_temp")     for s in sessions]
    fps_vals      = [s.get("fps")           for s in sessions]
    exposure_vals = [s.get("exposure")      for s in sessions]
    gain_vals     = [s.get("gain")          for s in sessions]
    drop_vals     = [s.get("drops",0)       for s in sessions]
    capture_vals  = [s.get("captured")      for s in sessions]

    # Firmware distribution
    fw_dist: Dict[str, int] = {}
    for s in sessions:
        fw = (s.get("firmware") or "unknown").split()[0]
        fw_dist[fw] = fw_dist.get(fw, 0) + 1

    # Mission type distribution
    mt_dist: Dict[str, int] = {}
    for s in sessions:
        mt = s.get("mission_type","unknown")
        mt_dist[mt] = mt_dist.get(mt, 0) + 1

    # Stale trigger count
    stale_count = sum(1 for s in sessions if s.get("stale_trigger"))
    drop_count  = sum(1 for s in sessions if (s.get("drops") or 0) > 0)
    arg_fail    = [s for s in sessions if s.get("arg_count",14) < 14]

    return {
        "sensor_temperature": {
            "log":  _stats(sensor_temps),
            "meta": _stats([v for v in sensor_temps2 if v]),
            "outliers": _outliers("sensor_temp", _vec("sensor_temp")),
        },
        "core_temperature":   _stats(core_temps),
        "fps":                _stats(fps_vals),
        "exposure":           _stats(exposure_vals),
        "gain":               _stats(gain_vals),
        "frames_captured":    _stats(capture_vals),
        "frame_drops": {
            "total_sessions_with_drops": drop_count,
            "total_frames_lost":         sum(d for d in drop_vals if d),
        },
        "firmware_distribution":      fw_dist,
        "mission_type_distribution":  mt_dist,
        "stale_trigger_count":        stale_count,
        "arg_integrity_failures":     len(arg_fail),
        "arg_fail_sessions":          [s["name"] for s in arg_fail],
        "total_sessions":             len(sessions),
    }


# ══════════════════════════════════════════════════════════════════════════════
# QUICK REPORT GENERATION FOR BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def quick_report_from_folder(folder: str, scan_result=None) -> str:
    """
    Generate a quick report from dataset folder without scanning.
    Useful for batch processing where scan_result is already available.
    
    If scan_result is None, returns a minimal dataset info string.
    """
    dataset_name = os.path.basename(folder.rstrip('/'))
    
    if scan_result is None:
        # Minimal fallback — just dataset info
        return f"Dataset: {dataset_name}\n(Re-run scanner to generate full report)\n"
    
    # Use existing generate_full_report with the scan_result
    return generate_full_report(scan_result, folder, enable_template_comparison=False)


def batch_reports(capture_folder: str, pattern: str = "Acq*", 
                  return_dict: bool = False, enable_template_comparison: bool = False) -> dict:
    """
    Generate reports for multiple datasets.
    
    Args:
        capture_folder: Root folder containing dataset folders
        pattern: Glob pattern for dataset names (default: Acq*)
        return_dict: If True, return dict of {name: report_text}.
                    If False, return list of (name, report_text) tuples.
        enable_template_comparison: If True, compare against learned templates and show deviations.
                                   If False, show template status only (default: False).
    
    Returns:
        Dict or list of reports, or errors if any.
    
    Example:
        reports = batch_reports('/home/xd/Capture')
        for name, report in reports.items():
            print(f"=== {name} ===")
            print(report)
    """
    from glob import glob
    from .app_state import ScanResult
    
    search_path = os.path.join(capture_folder, pattern)
    datasets = sorted([d for d in glob(search_path) if os.path.isdir(d)])
    
    reports = {} if return_dict else []
    
    for dataset in datasets:
        name = os.path.basename(dataset)
        try:
            # Quick scan of the dataset
            from .tools import tool_run_scan
            scan_summary = tool_run_scan(dataset, mode="quick")
            if scan_summary and "error" not in scan_summary:
                # Get the scan result from state
                from .app_state import state
                scan_obj = state.get_scan_result(dataset)
            else:
                scan_obj = None
            
            if scan_obj:
                report_text = generate_full_report(scan_obj, dataset, enable_template_comparison=enable_template_comparison)
            else:
                error_msg = scan_summary.get("error", "Scan failed") if scan_summary else "Scan failed"
                report_text = f"[{name}] {error_msg}\n"
            
            if return_dict:
                reports[name] = report_text
            else:
                reports.append((name, report_text))
        
        except Exception as e:
            error_msg = f"[{name}] Error: {str(e)}\n"
            if return_dict:
                reports[name] = error_msg
            else:
                reports.append((name, error_msg))
    
    return reports
