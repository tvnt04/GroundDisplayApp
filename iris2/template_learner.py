"""
iris/template_learner.py

Adaptive mission template learning for Iris.

Two problems this solves:

1. FIELD-LEVEL LEARNING (parametric anomaly detection)
   "Temperature is always 46°C. This session shows 8°C → flag it."
   Uses Welford online algorithm — no raw data stored, just running
   mean + variance per field per mission type.

2. STRUCTURAL LEARNING (log format / field discovery)
   "This new mission has a field called 'LineScanRate' I've never seen.
    Learn what it means and flag it when it deviates."
   The learner discovers all key=value pairs in any log, adds unknown
   fields to the template automatically, and starts tracking them.
   It does NOT need hard-coded field lists.

MISSION TYPES are detected automatically from log content — the learner
does NOT need to know the mission type in advance. When a completely new
log structure appears, it gets classified as a new mission type and builds
its own baseline from scratch.

Storage: iris/.iris_templates.json (human-readable, editable)

Usage:
    from .template_learner import TemplateLearner
    tl = TemplateLearner()
    tl.ingest(log_summary, meta_summary, log_text)
    deviations = tl.flag_deviations(log_summary, meta_summary, log_text)
    report     = tl.status_report()
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Storage ────────────────────────────────────────────────────────────────────

_DEFAULT_PATH   = os.path.join(os.path.dirname(__file__), ".iris_templates.json")
_LOCK           = threading.Lock()
_MIN_SAMPLES    = 3       # samples needed before flagging
_SIGMA_INFO     = 2.5     # σ threshold for INFO flag
_SIGMA_WARN     = 4.0     # σ threshold for WARNING flag
_MIN_STD        = 0.01    # ignore near-zero variance fields


# ══════════════════════════════════════════════════════════════════════════════
# FIELD EXTRACTOR — discovers ALL numeric key=value pairs in a log
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that extract (label, value) pairs from log lines.
# The extractor is purely regex-based — no hard-coded field names.
# New fields in new log formats are discovered automatically.

_RE_KV_FLOAT = re.compile(
    r'(?:Applied|Set|Device|Sensor|Computed|Total|Captured|Allocated|Max|Min)\s+'
    r'([A-Za-z_][A-Za-z0-9_]*)'           # field name
    r'\s*[=:]\s*'
    r'([-+]?\d+(?:\.\d+)?)',               # numeric value
    re.I
)

# Also capture "X: value" style (I-code lines)
_RE_COLON_KV = re.compile(
    r'\[I\d+\]\s+([A-Za-z][A-Za-z0-9_ ]{2,30}):\s*([-+]?\d+(?:\.\d+)?)\s*$'
)

# Frame stats from FrameNo lines (instantFps, TimeDifference)
_RE_FRAME_LINE = re.compile(
    r'instantFps=([\d.]+).*TimeDifference:\s*([\d.]+)'
)

# Applied block (I43–I55 style): "Applied FPS = 55.999"
_RE_APPLIED = re.compile(
    r'Applied\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([-+]?\d+(?:\.\d+)?)',
    re.I
)

# SensorTemp / CoreTemp / Device Core Temperature
_RE_SENSOR_TEMP  = re.compile(r'SensorTemp:\s*([\d.]+)', re.I)
_RE_CORE_TEMP    = re.compile(r'Device Core Temperature:\s*(\d+)', re.I)

# File size, frame size
_RE_FILE_SIZE    = re.compile(r'File_size=(\d+)\(FrameSize=(\d+)', re.I)

# Disk / RAM
_RE_DISK         = re.compile(r'Disk free space:\s*([\d.]+)', re.I)
_RE_RAM          = re.compile(r'RAM available:\s*([\d.]+)', re.I)

# Grabber connect timestamp (corrected format)
_RE_TS           = re.compile(r'^\[(\d+),(\d+)\.(\d+)\]')


def _parse_ts(line: str) -> Optional[float]:
    """[SS,MMM.UUU] → total seconds. See timestamp.py for format docs."""
    m = _RE_TS.match(line.strip())
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2))/1000.0 + int(m.group(3))/1_000_000.0


def extract_all_fields(log_text: str,
                        log_summary: Dict = None,
                        meta_summary: Dict = None) -> Dict[str, float]:
    """
    Extract ALL discoverable numeric fields from a log.
    Returns flat dict {field_name: float_value}.

    This is the core of structural learning — it finds every numeric
    value in the log without needing a hard-coded field list.
    New missions with new fields are handled automatically.
    """
    fields: Dict[str, float] = {}
    lines  = log_text.splitlines()

    i06_ts = None
    i96_ts = None
    fps_readings: List[float] = []
    td_readings:  List[float] = []
    sensor_temps_before: List[float] = []
    core_temps_before:   List[float] = []
    capture_started = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Capture start marker
        if 'FrameNo=' in stripped:
            capture_started = True

        # Grabber timestamps (corrected: [SS,MMM.UUU] format)
        ts = _parse_ts(stripped)
        if ts is not None:
            if '[I06]' in stripped and 'Autoconnect' in stripped and i06_ts is None:
                i06_ts = ts
            if '[I96]' in stripped and 'Good connection' in stripped and i96_ts is None:
                i96_ts = ts

        # Frame-level FPS + TimeDifference
        m = _RE_FRAME_LINE.search(stripped)
        if m:
            fps_v = float(m.group(1))
            td_v  = float(m.group(2))
            if fps_v > 0:
                fps_readings.append(fps_v)
            if td_v > 0:
                td_readings.append(td_v)
            continue

        # Temperature (before capture only → baseline)
        if not capture_started:
            m = _RE_SENSOR_TEMP.search(stripped)
            if m:
                sensor_temps_before.append(float(m.group(1)))
            m = _RE_CORE_TEMP.search(stripped)
            if m:
                core_temps_before.append(float(m.group(1)))

        # Applied block (most authoritative applied values)
        m = _RE_APPLIED.search(stripped)
        if m:
            key = _normalise_key(m.group(1))
            val = float(m.group(2))
            fields[f"applied_{key}"] = val
            continue

        # Generic key=value from info lines
        for m in _RE_KV_FLOAT.finditer(stripped):
            key = _normalise_key(m.group(1))
            val = float(m.group(2))
            if key not in fields:  # first occurrence wins for most fields
                fields[key] = val

        # I-code colon-style "Key: value"
        m = _RE_COLON_KV.match(stripped)
        if m:
            key = _normalise_key(m.group(1))
            val = float(m.group(2))
            if key not in fields:
                fields[key] = val

        # Disk / RAM
        m = _RE_DISK.search(stripped)
        if m: fields["disk_free_gb"] = float(m.group(1))
        m = _RE_RAM.search(stripped)
        if m: fields["ram_available_gb"] = float(m.group(1))

        # File size
        m = _RE_FILE_SIZE.search(stripped)
        if m:
            fields["file_size_bytes"]  = float(m.group(1))
            fields["frame_size_bytes"] = float(m.group(2))

    # Derived / aggregated fields
    if sensor_temps_before:
        fields["sensor_temp_before"] = sensor_temps_before[0]
        fields["sensor_temp_after"]  = sensor_temps_before[-1]
        if len(sensor_temps_before) > 1:
            fields["sensor_temp_delta"] = abs(sensor_temps_before[-1] - sensor_temps_before[0])
    if core_temps_before:
        fields["core_temp_before"] = core_temps_before[0]

    if fps_readings:
        import statistics as _s
        fields["fps_mean"]   = round(_s.mean(fps_readings), 4)
        fields["fps_std"]    = round(_s.pstdev(fps_readings), 4) if len(fps_readings) > 1 else 0.0
    if td_readings:
        import statistics as _s
        fields["timediff_mean_ms"] = round(_s.mean(td_readings), 4)
        fields["timediff_std_ms"]  = round(_s.pstdev(td_readings), 4) if len(td_readings) > 1 else 0.0

    if i06_ts is not None and i96_ts is not None:
        fields["grabber_connect_s"] = round(i96_ts - i06_ts, 6)

    # From structured log_summary (already parsed by scanner)
    if log_summary:
        fa   = log_summary.get("frame_accounting", {})
        proc = log_summary.get("procmode", {}).get("decoded", {})
        p_app= log_summary.get("parameters_applied", {})
        temps= log_summary.get("temperatures", {})

        _safe_add(fields, "frames_expected",    fa.get("total_frames_expected"))
        _safe_add(fields, "frames_captured",    fa.get("captured_count"))
        _safe_add(fields, "frame_drops",        fa.get("frames_lost", 0))
        _safe_add(fields, "fps_requested",      proc.get("fps_requested"))
        _safe_add(fields, "exposure_requested", proc.get("exposure_time"))
        _safe_add(fields, "gain_requested",     proc.get("gain"))
        _safe_add(fields, "duration_sec",       proc.get("duration_sec"))
        _safe_add(fields, "tdi_byte",           proc.get("tdi_byte"))
        _safe_add(fields, "band_selection",     proc.get("band_selection"))
        _safe_add(fields, "tdi_yshift",         proc.get("tdi_yshift"))
        _safe_add(fields, "sensor_temp_before", temps.get("sensor_before_C"))
        _safe_add(fields, "sensor_temp_after",  temps.get("sensor_after_C"))
        _safe_add(fields, "sensor_temp_delta",  temps.get("sensor_delta_C"))
        _safe_add(fields, "core_temp_before",   temps.get("core_before_C"))
        _safe_add(fields, "core_temp_delta",    temps.get("core_delta_C"))
        _safe_add(fields, "health_score",       getattr(log_summary, "health_score", None))

    # From meta_summary
    if meta_summary and meta_summary.get("found"):
        _safe_add(fields, "meta_bands_used",      meta_summary.get("bands_used"))
        _safe_add(fields, "meta_sensor_temp_c",   meta_summary.get("sensor_temperature_c"))
        _safe_add(fields, "meta_core_temp_c",     meta_summary.get("core_temperature_c"))
        _safe_add(fields, "meta_tdi_mode",        meta_summary.get("tdi_mode"))
        _safe_add(fields, "meta_tdi_stages",      meta_summary.get("tdi_stages"))
        _safe_add(fields, "meta_total_frames",    meta_summary.get("total_frames"))
        _safe_add(fields, "meta_lat_range",       meta_summary.get("lat_range"))
        _safe_add(fields, "meta_lon_range",       meta_summary.get("lon_range"))

    # Remove fields with nonsensical values
    fields = {k: v for k, v in fields.items()
              if v is not None and not math.isnan(v) and not math.isinf(v)
              and abs(v) < 1e12}

    return fields


def _safe_add(d: dict, key: str, value):
    if value is not None:
        try:
            d[key] = float(value)
        except (TypeError, ValueError):
            pass


def _normalise_key(raw: str) -> str:
    """Lower-case, replace spaces with underscores, strip trailing units."""
    k = raw.strip().lower()
    k = re.sub(r'\s+', '_', k)
    k = re.sub(r'[^a-z0-9_]', '', k)
    k = re.sub(r'_+', '_', k).strip('_')
    return k[:40]  # cap length


# ══════════════════════════════════════════════════════════════════════════════
# MISSION TYPE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

# Signatures are (mission_type, [(regex, weight), ...])
# Evaluated against lowercased log text. Highest score wins.
# New missions that don't match anything score 0 → "unknown_{hash}"

_SIGNATURES: List[Tuple[str, List[Tuple[str, float]]]] = [
    ("test_pattern_calibration", [
        (r'testpattern\s*[=:]\s*[1-9]', 10.0),
        (r'test.*pattern.*on',            4.0),
        (r'i31.*testpattern',             5.0),
    ]),
    ("dark_calibration", [
        (r'dark.*cap|covered.*lens|lens.*cap|shutter.*closed', 8.0),
        (r'exposure.*[5-9]\d{4,}',        4.0),
    ]),
    ("tdi_64_high_altitude", [
        (r'tdi_stages.*64|tdi.*64.stage', 10.0),
        (r'tdi_modes.*4|tdi.*byte.*66',   5.0),
    ]),
    ("tdi_8_standard", [
        (r'tdi_stages.*8\b|tdi.*8.stage', 8.0),
        (r'tdi_modes.*2|tdi.*byte.*34',   6.0),
        (r'regionheight.*384',            2.0),
    ]),
    ("tdi_4_low", [
        (r'tdi_stages.*4\b|tdi.*4.stage', 8.0),
        (r'tdi_modes.*1|tdi.*byte.*18',   5.0),
    ]),
    ("tdi_2_low", [
        (r'tdi_stages.*2\b|tdi.*2.stage', 8.0),
        (r'tdi_modes.*1|tdi.*byte.*10',   5.0),
    ]),
    ("no_tdi_full_frame", [
        (r'tdi_modes.*0|tdi.*off|tdi.*byte.*\b0\b', 8.0),
        (r'bandselection.*127',            4.0),
        (r'fps.*1[0-9]\.',                 2.0),
    ]),
    ("partial_band_capture", [
        (r'region\dmode:0',               5.0),
        (r'bandsused.*[1-6]\b',           4.0),
        (r'bandselection.*(?:30|60|14|62|15)\b', 4.0),
    ]),
    ("high_fps_pushbroom", [
        (r'fps.*[5-9]\d\.',               8.0),
        (r'maxfps.*\d{3}',                3.0),
    ]),
    ("low_fps_deep_integration", [
        (r'fps.*[1-9]\.',                 5.0),
        (r'exposuretime.*[3-9]\d{4}',     5.0),
    ]),
]


def detect_mission_type(log_text: str,
                         log_summary: Dict = None,
                         meta_summary: Dict = None,
                         fields: Dict = None) -> Dict:
    """
    Identify mission type from log content + optional parsed data.
    Works log-only — no band files required.

    Returns:
        mission_type: str
        confidence:   float (0–1)
        scores:       {type: score}
        evidence:     [patterns that fired]
        is_new_mission: bool (True if no signature matched well)
        structural_fingerprint: str (hash of discovered field names)
    """
    text  = (log_text or "").lower()

    # Direct override: test pattern detected in parsed summary
    if log_summary and log_summary.get("test_pattern") not in (None, 0):
        return {
            "mission_type":           "test_pattern_calibration",
            "confidence":             1.0,
            "scores":                 {"test_pattern_calibration": 1.0},
            "evidence":               ["test_pattern != 0 in log_summary"],
            "is_new_mission":         False,
            "structural_fingerprint": _field_fingerprint(fields or {}),
        }

    # Score each type
    raw_scores: Dict[str, float] = {}
    evidence:   Dict[str, List]  = {}

    for mtype, sigs in _SIGNATURES:
        score = 0.0
        evs   = []
        max_s = sum(w for _, w in sigs)
        for pattern, weight in sigs:
            if re.search(pattern, text, re.I):
                score += weight
                evs.append(pattern)
        raw_scores[mtype]  = score / max_s if max_s > 0 else 0.0
        evidence[mtype]    = evs

    best_type  = max(raw_scores, key=raw_scores.get) if raw_scores else "unknown"
    confidence = raw_scores.get(best_type, 0.0)

    is_new = confidence < 0.25

    # If it's new and completely unrecognised, name it by its structural fingerprint
    fp = _field_fingerprint(fields or {})
    if is_new:
        best_type = f"unknown_{fp[:8]}"
        confidence = 0.0

    return {
        "mission_type":           best_type,
        "confidence":             round(confidence, 3),
        "scores":                 {k: round(v, 3) for k, v in raw_scores.items() if v > 0},
        "evidence":               evidence.get(best_type, []),
        "is_new_mission":         is_new,
        "structural_fingerprint": fp,
    }


def _field_fingerprint(fields: Dict) -> str:
    """Short hash of the set of field names — identifies log structure."""
    import hashlib
    key_str = ",".join(sorted(fields.keys()))
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE LEARNER  (Welford online algorithm, fully adaptive)
# ══════════════════════════════════════════════════════════════════════════════

class TemplateLearner:
    """
    Adaptive per-mission-type baseline learner.

    Key behaviours:
    - Discovers ALL numeric fields in any log format automatically
    - Learns mean + variance for each field using Welford's algorithm
      (no raw data stored — memory O(1) per field)
    - Handles new mission types by creating a new profile automatically
    - Flags deviations once a field has ≥3 samples
    - Persists to JSON — human-readable and editable

    One global instance is created at module level: `learner`.
    """

    def __init__(self, storage_path: str = _DEFAULT_PATH):
        self._path    = storage_path
        self._lock    = threading.Lock()
        self._cache: Optional[Dict] = None

    # ── Storage ────────────────────────────────────────────────────────────

    def _load(self) -> Dict:
        if self._cache is not None:
            return self._cache
        try:
            if os.path.exists(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                    return self._cache
        except Exception:
            pass
        self._cache = {}
        return self._cache

    def _save(self, data: Dict):
        self._cache = data
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"[TemplateLearner] Save failed: {e}")

    # ── Core Welford update ────────────────────────────────────────────────

    @staticmethod
    def _welford_update(stat: Dict, value: float) -> Dict:
        """
        Welford online algorithm for running mean + variance.
        stat dict: {n, mean, M2, min, max}
        Returns updated stat dict.
        """
        n    = stat["n"] + 1
        delta  = value - stat["mean"]
        mean   = stat["mean"] + delta / n
        delta2 = value - mean
        M2     = stat["M2"] + delta * delta2
        return {
            "n":    n,
            "mean": mean,
            "M2":   M2,
            "std":  math.sqrt(M2 / max(n - 1, 1)),
            "min":  min(stat["min"], value),
            "max":  max(stat["max"], value),
        }

    @staticmethod
    def _new_stat(value: float) -> Dict:
        return {"n": 1, "mean": value, "M2": 0.0, "std": 0.0,
                "min": value, "max": value}

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(self,
               log_summary: Dict,
               meta_summary: Dict = None,
               log_text: str = "",
               mission_type: str = None) -> Dict:
        """
        Learn from one scan. Call this after every successful scan.

        - Extracts ALL numeric fields from log_text + log_summary + meta_summary
        - Detects mission type if not provided
        - Updates Welford statistics for every field in this mission's profile
        - Adds new fields to the profile automatically (structural adaptation)
        - Stores the set of field names seen in this log (structural fingerprint)

        Returns summary of what was learned.
        """
        # Extract all fields
        fields = extract_all_fields(log_text, log_summary, meta_summary)

        # Detect mission type if not given
        if mission_type is None:
            det = detect_mission_type(log_text, log_summary, meta_summary, fields)
            mission_type = det["mission_type"]

        with self._lock:
            data = self._load()

            profile = data.setdefault(mission_type, {
                "mission_type":    mission_type,
                "sample_count":    0,
                "first_seen":      _now(),
                "last_seen":       _now(),
                "fields":          {},
                "known_field_keys": [],   # all field names ever seen
                "structural_versions": [], # list of fingerprints seen
            })

            n = profile["sample_count"] + 1
            profile["sample_count"] = n
            profile["last_seen"]    = _now()

            fp = _field_fingerprint(fields)
            if fp not in profile["structural_versions"]:
                profile["structural_versions"].append(fp)
                # New log structure for this mission type — note it
                if len(profile["structural_versions"]) > 1:
                    profile["structure_evolved"] = True

            updated_fields = []
            new_fields     = []

            for key, value in fields.items():
                is_new = key not in profile["fields"]
                if is_new:
                    profile["fields"][key] = self._new_stat(value)
                    new_fields.append(key)
                    if key not in profile["known_field_keys"]:
                        profile["known_field_keys"].append(key)
                else:
                    profile["fields"][key] = self._welford_update(
                        profile["fields"][key], value)
                    updated_fields.append(key)

            self._save(data)

        return {
            "mission_type":     mission_type,
            "sample_count":     n,
            "fields_updated":   len(updated_fields),
            "new_fields_found": new_fields,  # fields seen for first time in this log
            "total_fields":     len(profile["fields"]),
            "structural_fingerprint": fp,
        }

    # ── Deviation detection ───────────────────────────────────────────────

    def flag_deviations(self,
                         log_summary: Dict,
                         meta_summary: Dict = None,
                         log_text: str = "",
                         mission_type: str = None,
                         sigma_info: float = _SIGMA_INFO,
                         sigma_warn: float = _SIGMA_WARN) -> List[Dict]:
        """
        Compare observed values against learned baselines.
        Returns list of deviation dicts for report injection.

        Fields with fewer than _MIN_SAMPLES observations are skipped.
        Fields with near-zero variance are skipped.
        """
        fields = extract_all_fields(log_text, log_summary, meta_summary)

        if mission_type is None:
            det = detect_mission_type(log_text, log_summary, meta_summary, fields)
            mission_type = det["mission_type"]

        with self._lock:
            data    = self._load()
        profile = data.get(mission_type)

        if not profile or profile.get("sample_count", 0) < _MIN_SAMPLES:
            return []

        deviations = []

        for key, value in fields.items():
            baseline = profile["fields"].get(key)
            if not baseline or baseline.get("n", 0) < _MIN_SAMPLES:
                continue
            std = baseline.get("std", 0.0)
            if std < _MIN_STD:
                continue

            mean  = baseline["mean"]
            sigma = abs(value - mean) / std

            if sigma < sigma_info:
                continue

            severity  = "WARNING" if sigma >= sigma_warn else "INFO"
            direction = "above" if value > mean else "below"
            n_samples = baseline["n"]

            # Human-friendly field label
            label = _pretty_label(key)

            deviations.append({
                "field":     key,
                "label":     label,
                "observed":  round(value, 4),
                "mean":      round(mean, 4),
                "std":       round(std, 4),
                "sigma":     round(sigma, 2),
                "min_seen":  round(baseline["min"], 4),
                "max_seen":  round(baseline["max"], 4),
                "n_samples": n_samples,
                "severity":  severity,
                "mission_type": mission_type,
                "message": (
                    f"[Template:{mission_type}] {label} = {_fmt(value)} is "
                    f"{sigma:.1f}σ {direction} normal "
                    f"(baseline: {_fmt(mean)} ± {_fmt(std)}, "
                    f"range {_fmt(baseline['min'])}–{_fmt(baseline['max'])}, "
                    f"n={n_samples} sessions). "
                    + ("⚠ Significant deviation — investigate."
                       if severity == "WARNING"
                       else "Noted.")
                ),
            })

        # Sort: warnings first, then by sigma descending
        deviations.sort(key=lambda d: (-{"WARNING":1,"INFO":0}[d["severity"]], -d["sigma"]))
        return deviations

    # ── New field notification ─────────────────────────────────────────────

    def new_fields_in(self,
                       log_text: str,
                       log_summary: Dict = None,
                       meta_summary: Dict = None,
                       mission_type: str = None) -> List[str]:
        """
        Return field names that appear in this log but were never seen
        before in this mission type's history.
        Useful for alerting the user that the log format has evolved.
        """
        fields = extract_all_fields(log_text, log_summary, meta_summary)

        if mission_type is None:
            det = detect_mission_type(log_text, log_summary, meta_summary, fields)
            mission_type = det["mission_type"]

        with self._lock:
            data    = self._load()
        profile = data.get(mission_type)
        if not profile:
            return list(fields.keys())  # everything is new

        known = set(profile.get("known_field_keys", []))
        return [k for k in fields if k not in known]

    # ── Status & management ────────────────────────────────────────────────

    def status(self) -> Dict:
        """
        Return all learned templates with their field baselines.
        """
        with self._lock:
            data = self._load()

        result = {}
        for mtype, profile in data.items():
            fields_summary = {}
            for fname, stat in profile.get("fields", {}).items():
                if stat.get("n", 0) < 1:
                    continue
                fields_summary[fname] = {
                    "n":    stat["n"],
                    "mean": round(stat["mean"], 4),
                    "std":  round(stat.get("std", 0), 4),
                    "min":  round(stat["min"], 4),
                    "max":  round(stat["max"], 4),
                    "label": _pretty_label(fname),
                }
            result[mtype] = {
                "sample_count":       profile.get("sample_count", 0),
                "first_seen":         profile.get("first_seen", ""),
                "last_seen":          profile.get("last_seen", ""),
                "total_fields":       len(fields_summary),
                "structure_evolved":  profile.get("structure_evolved", False),
                "structural_versions":len(profile.get("structural_versions", [])),
                "fields":             fields_summary,
            }
        return result

    def status_report(self) -> str:
        """Human-readable summary of all learned templates."""
        data = self.status()
        if not data:
            return "No templates learned yet. Run scans to build baselines."

        lines = ["═" * 60, "  IRIS TEMPLATE BASELINES", "═" * 60]
        for mtype, info in sorted(data.items()):
            n = info["sample_count"]
            ev = " [structure evolved]" if info.get("structure_evolved") else ""
            lines.append(f"\n  {mtype}  ({n} sessions){ev}")
            lines.append(f"  First: {info['first_seen']}  Last: {info['last_seen']}")
            lines.append(f"  {info['total_fields']} tracked fields:")
            for fname, stat in sorted(info["fields"].items()):
                if stat["n"] < 2:
                    continue
                flag = "  ← <3 samples" if stat["n"] < _MIN_SAMPLES else ""
                lines.append(
                    f"    {stat['label']:35s}  "
                    f"n={stat['n']:3d}  "
                    f"mean={stat['mean']:>10.3f}  "
                    f"std={stat['std']:>8.3f}  "
                    f"range [{stat['min']:.3f}–{stat['max']:.3f}]{flag}"
                )
        lines.append("\n" + "═" * 60)
        return "\n".join(lines)

    def reset(self, mission_type: str = None) -> Dict:
        """
        Delete one mission profile (or all if mission_type is None).
        """
        with self._lock:
            data = self._load()
            if mission_type:
                removed = [mission_type] if mission_type in data else []
                data.pop(mission_type, None)
            else:
                removed = list(data.keys())
                data    = {}
            self._cache = None  # invalidate cache
            self._save(data)
        return {"reset": removed}

    def merge_profile(self, other_json_path: str) -> Dict:
        """
        Merge templates from another iris_templates.json into this one.
        Useful when multiple machines each have their own baselines and
        you want to combine them.
        Combines Welford statistics exactly (no data needed).
        """
        try:
            with open(other_json_path, "r", encoding="utf-8") as f:
                other = json.load(f)
        except Exception as e:
            return {"error": str(e)}

        with self._lock:
            data    = self._load()
            merged  = 0
            added   = 0

            for mtype, op in other.items():
                if mtype not in data:
                    data[mtype] = op
                    added += 1
                    continue

                # Merge two Welford streams: Chan's parallel algorithm
                lp = data[mtype]
                for fname, os_stat in op.get("fields", {}).items():
                    if fname not in lp["fields"]:
                        lp["fields"][fname] = os_stat
                    else:
                        ls = lp["fields"][fname]
                        na, nb = ls["n"], os_stat["n"]
                        if nb == 0:
                            continue
                        n_total = na + nb
                        delta   = os_stat["mean"] - ls["mean"]
                        mean_c  = ls["mean"] + delta * nb / n_total
                        M2_c    = (ls["M2"] + os_stat["M2"] +
                                   delta**2 * na * nb / n_total)
                        lp["fields"][fname] = {
                            "n":    n_total,
                            "mean": mean_c,
                            "M2":   M2_c,
                            "std":  math.sqrt(M2_c / max(n_total - 1, 1)),
                            "min":  min(ls["min"], os_stat["min"]),
                            "max":  max(ls["max"], os_stat["max"]),
                        }
                lp["sample_count"] = lp.get("sample_count",0) + op.get("sample_count",0)
                lp["last_seen"]    = _now()
                for k in op.get("known_field_keys", []):
                    if k not in lp.get("known_field_keys", []):
                        lp.setdefault("known_field_keys", []).append(k)
                merged += 1

            self._save(data)

        return {"merged": merged, "added_new": added, "total_types": len(data)}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _fmt(v: float) -> str:
    """Format a float nicely — fewer decimals for large values."""
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.4f}"


_LABEL_MAP = {
    "sensor_temp_before":   "Sensor Temp (start, °C)",
    "sensor_temp_after":    "Sensor Temp (end, °C)",
    "sensor_temp_delta":    "Sensor Temp Δ (°C)",
    "core_temp_before":     "Core Temp (start, °C)",
    "core_temp_delta":      "Core Temp Δ (°C)",
    "fps_mean":             "FPS (mean, measured)",
    "fps_std":              "FPS jitter (std)",
    "fps_requested":        "FPS (requested)",
    "timediff_mean_ms":     "Frame interval (ms, mean)",
    "timediff_std_ms":      "Frame interval jitter (ms)",
    "exposure_requested":   "Exposure (µs, requested)",
    "applied_exposuretime": "Exposure (µs, applied)",
    "applied_fps":          "FPS (applied)",
    "applied_gain":         "Gain (applied)",
    "frames_captured":      "Frames captured",
    "frames_expected":      "Frames expected",
    "frame_drops":          "Frame drops",
    "grabber_connect_s":    "Grabber connect time (s)",
    "disk_free_gb":         "Disk free (GB)",
    "ram_available_gb":     "RAM available (GB)",
    "file_size_bytes":      "Output file size (bytes)",
    "meta_sensor_temp_c":   "Sensor Temp (meta, °C)",
    "meta_core_temp_c":     "Core Temp (meta, °C)",
    "meta_bands_used":      "Bands used (meta)",
    "meta_tdi_stages":      "TDI stages (meta)",
    "health_score":         "Health score",
}


def _pretty_label(key: str) -> str:
    if key in _LABEL_MAP:
        return _LABEL_MAP[key]
    # Generic: replace underscores, title-case
    return key.replace("_", " ").title()


# ── Global singleton ──────────────────────────────────────────────────────────

learner = TemplateLearner()


# ── Convenience wrappers (drop-in replacements for meta_parser functions) ─────

def learn_from_scan(log_summary: Dict,
                     meta_summary: Dict = None,
                     mission_type: str = "unknown",
                     log_text: str = "") -> Dict:
    """Wrapper for tools.py compatibility."""
    return learner.ingest(log_summary, meta_summary, log_text, mission_type)


def flag_template_deviations(log_summary: Dict,
                               meta_summary: Dict = None,
                               mission_type: str = "unknown",
                               log_text: str = "",
                               sigma_threshold: float = _SIGMA_INFO) -> List[Dict]:
    """Wrapper for tools.py compatibility."""
    return learner.flag_deviations(log_summary, meta_summary, log_text,
                                    mission_type, sigma_info=sigma_threshold)


def get_template_summary() -> Dict:
    return learner.status()


def reset_template(mission_type: str = None) -> Dict:
    return learner.reset(mission_type)
