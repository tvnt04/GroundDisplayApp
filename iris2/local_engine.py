from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Any


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze(folder: str = "", question: str = "") -> Dict:
    from .app_state import state
    from .tools import tool_get_app_state, tool_get_scan_results

    # Resolve folder
    if not folder:
        folder = state.active_folder or ""

    if not folder:
        return {
            "has_data":    False,
            "report_text": "No dataset is open. Open a dataset folder first.",
            "findings":    [],
            "answer":      "",
            "suggestions": ["Open a dataset folder to begin analysis."],
        }

    # Get scan results
    scan = state.get_scan_result(folder)

    if not scan:
        return {
            "has_data":    False,
            "report_text": (
                f"No scan results for {os.path.basename(folder)}. "
                "Run a scan first: type 'scan' or press the scan button."
            ),
            "findings":    [],
            "answer":      "",
            "suggestions": [
                "Run 'scan' for a fast metadata/log check.",
                "Run 'full scan' only when you need frame-level or pixel-level analysis.",
            ],
        }

    report_text = _build_report(scan, folder)
    answer      = _answer_question(question, scan) if question else ""
    suggestions = _suggest_actions(scan)

    return {
        "has_data":    True,
        "report_text": report_text,
        "findings":    scan.findings,
        "answer":      answer,
        "suggestions": suggestions,
        "health_score": scan.health_score,
        "scan_type":   scan.scan_type,
        "folder":      folder,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def _build_report(scan, folder: str) -> str:
    """
    Build the structured "fill-in-the-blanks" report from log/meta/ephem.
    Falls back to the legacy local report if generation fails.
    """
    try:
        from .tools import tool_generate_report
        rep = tool_generate_report(folder, enable_template_comparison=False)
        if rep and rep.get("report"):
            return rep["report"]
    except Exception as e:
        print(f"[LocalEngine] Structured report failed, using legacy format: {e}")
    return _build_report_legacy(scan, folder)


def _build_report_legacy(scan, folder: str) -> str:
    """Legacy local report kept as fallback."""
    name      = os.path.basename(folder)
    scan_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(scan.scanned_at))
    width     = 62

    lines = []

    # Header
    lines.append("╔" + "═" * width + "╗")
    lines.append(f"  IRIS LOCAL ANALYSIS: {name}")
    lines.append(f"  Scan: {scan.scan_type} | Time: {scan_time} | Duration: {scan.duration_sec:.1f}s")
    lines.append("╚" + "═" * width + "╝")
    lines.append("")

    # Health score
    health = scan.health_score
    if health >= 90:
        icon, status = "✅", "HEALTHY"
    elif health >= 75:
        icon, status = "🟡", "PATTERNS NOTED"
    elif health >= 50:
        icon, status = "🟠", "ISSUES FOUND"
    else:
        icon, status = "🔴", "CRITICAL ISSUES"

    critical = sum(1 for f in scan.findings if f.get("severity") == "CRITICAL")
    warning  = sum(1 for f in scan.findings if f.get("severity") == "WARNING")
    info     = sum(1 for f in scan.findings if f.get("severity") == "INFO")

    lines.append(f"  HEALTH: {health:.0f}/100  {icon} {status}")
    lines.append(f"  Findings: {critical} critical · {warning} warnings · {info} info")
    lines.append(f"  Anomaly frames: {len(scan.anomaly_frames)}")
    lines.append("")

    # Findings by severity
    for severity, label in [("CRITICAL", "CRITICAL ISSUES"), ("WARNING", "WARNINGS"), ("INFO", "INFO")]:
        findings = [f for f in scan.findings if f.get("severity") == severity]
        if not findings:
            continue

        lines.append("─" * width)
        lines.append(f"  {label} ({len(findings)})")
        lines.append("─" * width)

        for i, finding in enumerate(findings, 1):
            msg    = finding.get("message", "")
            detail = finding.get("detail", "")

            icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "⚪"}.get(severity, "•")
            lines.append(f"  {icon} [{i}] {msg}")
            if detail:
                # Wrap detail text
                for dline in _wrap(detail, width - 6):
                    lines.append(f"        {dline}")
            lines.append("")

    # Anomaly frames
    if scan.anomaly_frames:
        lines.append("─" * width)
        lines.append(f"  ANOMALY FRAMES ({len(scan.anomaly_frames)} total)")
        lines.append("─" * width)
        shown = scan.anomaly_frames[:20]
        lines.append("  " + ", ".join(str(f) for f in shown))
        if len(scan.anomaly_frames) > 20:
            lines.append(f"  ... and {len(scan.anomaly_frames) - 20} more")
        lines.append("")

    # Band summary
    if scan.band_summary:
        lines.append("─" * width)
        lines.append("  BAND SUMMARY")
        lines.append("─" * width)
        for band_key, bdata in sorted(scan.band_summary.items()):
            if not isinstance(bdata, dict):
                continue
            n = bdata.get("n_frames", "?")
            lines.append(f"  {band_key}: {n} frames")
        lines.append("")

    # Log summary
    if scan.log_summary:
        lines.append("─" * width)
        lines.append("  LOG SUMMARY")
        lines.append("─" * width)
        ls = scan.log_summary
        for key, val in ls.items():
            if val and key not in ("raw", "full_text"):
                lines.append(f"  {key}: {val}")
        lines.append("")

    # Action items
    actions = _suggest_actions(scan)
    if actions:
        lines.append("─" * width)
        lines.append("  SUGGESTED ACTIONS")
        lines.append("─" * width)
        for i, action in enumerate(actions, 1):
            lines.append(f"  {i}. {action}")
        lines.append("")

    lines.append("─" * width)
    lines.append("  [Local Analysis — Iris rule engine]")

    return "\n".join(lines)


# ── Question answering ────────────────────────────────────────────────────────

def _answer_question(question: str, scan) -> str:
    """
    Answer a specific question from scan data using rule-based logic.
    Returns a direct answer string, or "" if the question can't be answered
    from scan data alone (signal to escalate to Ollama/API).
    """
    q = question.lower().strip()

    # Worst frames
    if any(k in q for k in ("worst frame", "worst frames", "bad frame", "most anomalous")):
        if not scan.anomaly_frames:
            return "No anomaly frames detected in this dataset."
        shown = scan.anomaly_frames[:10]
        return (
            f"Worst frames (most anomalous): {', '.join(str(f) for f in shown)}"
            + (f" (+ {len(scan.anomaly_frames)-10} more)" if len(scan.anomaly_frames) > 10 else "")
        )

    # Health score
    if any(k in q for k in ("health", "score", "overall")):
        return f"Health score: {scan.health_score:.0f}/100. " + _health_description(scan.health_score)

    # Critical issues
    if any(k in q for k in ("critical", "urgent", "serious", "most important")):
        crits = [f for f in scan.findings if f.get("severity") == "CRITICAL"]
        if not crits:
            return "No critical issues found in this dataset."
        return "Critical issues:\n" + "\n".join(
            f"  • {f.get('message','')}" for f in crits
        )

    # Warnings
    if "warning" in q:
        warns = [f for f in scan.findings if f.get("severity") == "WARNING"]
        if not warns:
            return "No warnings found."
        return f"{len(warns)} warning(s):\n" + "\n".join(
            f"  • {f.get('message','')}" for f in warns
        )

    # Exposure
    if any(k in q for k in ("exposure", "exp")):
        matches = [f for f in scan.findings if "exp" in f.get("type","").lower()
                   or "exposure" in f.get("message","").lower()]
        if matches:
            return "\n".join(f"  • {f.get('message','')}" for f in matches)
        return "No exposure-related findings in scan results."

    # FPS
    if "fps" in q or "frame rate" in q:
        matches = [f for f in scan.findings if "fps" in f.get("type","").lower()
                   or "fps" in f.get("message","").lower()]
        if matches:
            return "\n".join(f"  • {f.get('message','')}" for f in matches)
        return "No FPS-related findings in scan results."

    # Frame count
    if any(k in q for k in ("how many frames", "frame count", "total frames")):
        bs = scan.band_summary
        if bs:
            counts = [v.get("n_frames", 0) for v in bs.values() if isinstance(v, dict)]
            if counts:
                return f"Frame count: {max(counts)} frames across {len(bs)} band(s)."
        return "Frame count not available in scan summary."

    # Anomalies
    if any(k in q for k in ("anomal", "anomaly", "how many bad")):
        n = len(scan.anomaly_frames)
        if n == 0:
            return "No anomaly frames detected."
        return f"{n} anomaly frame(s) detected: {scan.anomaly_frames[:10]}"

    # Temperature
    if any(k in q for k in ("temp", "temperature", "hot", "thermal")):
        matches = [f for f in scan.findings if "temp" in f.get("type","").lower()
                   or "temp" in f.get("message","").lower()]
        if matches:
            return "\n".join(f"  • {f.get('message','')}" for f in matches)
        log = scan.log_summary or {}
        for k, v in log.items():
            if "temp" in k.lower():
                return f"{k}: {v}"
        return "No temperature data in scan results."

    # Can't answer from rules alone — return empty to signal Ollama escalation
    return ""


def _health_description(score: float) -> str:
    if score >= 90:
        return "Sensor is operating normally. No significant issues detected."
    elif score >= 75:
        return "Minor patterns noted. Review warnings before next mission."
    elif score >= 50:
        return "Notable issues found. Investigate before next capture."
    else:
        return "Critical issues present. Do not use for mission without investigation."


# ── Action suggestions ────────────────────────────────────────────────────────

def _suggest_actions(scan) -> List[str]:
    """Generate context-aware action suggestions from scan findings."""
    actions = []
    finding_types = {f.get("type", "") for f in scan.findings}
    severities    = {f.get("severity", "") for f in scan.findings}

    if "CRITICAL" in severities:
        actions.append("Investigate critical issues before next mission capture.")

    if any("exp" in t for t in finding_types):
        actions.append("Check exposure settings — clamping detected. Adjust ExposureTime parameter.")

    if any("fps" in t for t in finding_types):
        actions.append("Review FPS settings — rate was capped. Check MaxFPS constraint.")

    if any("trigger" in t for t in finding_types):
        actions.append("Fix UTC trigger timestamp sync before next acquisition.")

    if any("zip" in t or "settings" in t for t in finding_types):
        actions.append("Resolve camera settings ZIP path — custom calibration not applied.")

    if any("frame" in t and "lost" in t for t in finding_types):
        actions.append("Investigate frame loss — check storage bandwidth and DMA settings.")

    if scan.anomaly_frames:
        worst = scan.anomaly_frames[:3]
        actions.append(f"Review anomaly frames: {', '.join(str(f) for f in worst)}")

    if not actions:
        actions.append("Dataset looks healthy. No immediate actions required.")

    return actions


# ── Context builder for Ollama/API ────────────────────────────────────────────

def build_context_for_llm(folder: str = "", question: str = "") -> str:
    """
    Build a minimal context string for Ollama — only what is relevant
    to the question. Keeps token count low so the model focuses.
    """
    from .app_state import state

    if not folder:
        folder = state.active_folder or ""
    if not folder:
        return "No dataset open."

    scan = state.get_scan_result(folder)
    if not scan:
        return f"No scan results for {os.path.basename(folder)}. Run a scan first."

    name = os.path.basename(folder)
    q    = question.lower()

    # Always include: dataset name + health
    lines = [
        f"Dataset: {name}  |  Health: {scan.health_score:.0f}/100  |  "
        f"Scan: {scan.scan_type}",
    ]

    # Include critical findings always — they are always relevant
    critical = [f for f in scan.findings if f.get("severity") == "CRITICAL"]
    if critical:
        lines.append("Confirmed issues:")
        for f in critical:
            lines.append(f"  {f.get('message','')[:120]}")

    # Include log data only if question is about log topics
    log_topics = ("fps","exposure","temperature","temp","trigger","frame drop",
                  "parameter","gain","tdi","log","timing","firmware","grabber")
    if any(t in q for t in log_topics) or not q:
        ls = scan.log_summary or {}
        fa = ls.get("frame_accounting", {})
        tt = ls.get("trigger_timing", {})
        fps = ls.get("fps_stability", {})
        temps = ls.get("temperatures", {})
        if fa.get("frames_lost", 0):
            lines.append(f"Frame drops: {fa['frames_lost']} lost")
        if tt.get("stale_timestamp_detected"):
            lines.append("Trigger: stale timestamp — GPS sync did NOT occur")
        if fps.get("timing_stability"):
            lines.append(f"FPS: {fps.get('mean_fps','?')} mean, stability={fps['timing_stability']}")
        if temps.get("sensor_before_C"):
            lines.append(f"Temperature: sensor {temps['sensor_before_C']}°C, core {temps.get('core_before_C','?')}°C")
        proc = ls.get("procmode", {}).get("decoded", {})
        if proc:
            lines.append(f"Params: FPS={proc.get('fps_requested')} "
                         f"Exp={proc.get('exposure_time')}µs "
                         f"TDI={ls.get('procmode',{}).get('tdi_decoded',{}).get('mode','?')} "
                         f"Gain={proc.get('gain')}")

    # Include pixel data only if question is about pixel topics
    pixel_topics = ("hot","dead","pixel","stripe","band","pattern","banding",
                    "column","saturation","image","quality","anomaly")
    if any(t in q for t in pixel_topics) or not q:
        warnings = [f for f in scan.findings
                    if f.get("severity") == "WARNING"
                    and f.get("type") not in ("vertical_striping",)]  # skip noise-heavy types
        if warnings:
            lines.append(f"Observed patterns ({len(warnings)} warnings, showing top 4):")
            for f in warnings[:4]:
                lines.append(f"  {f.get('message','')[:100]}")
        if scan.anomaly_frames:
            lines.append(f"Anomaly frames: {len(scan.anomaly_frames)} — first few: {scan.anomaly_frames[:6]}")

    return "\n".join(lines)


# ── Text utilities ────────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> List[str]:
    """Simple word wrapper."""
    words  = text.split()
    lines  = []
    current = []
    length  = 0
    for word in words:
        if length + len(word) + 1 > width:
            lines.append(" ".join(current))
            current = [word]
            length  = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return lines
