from __future__ import annotations
import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Callable

from PyQt5.QtCore import QTimer

from .app_state import state, ScanResult
from .scanner import start_scan, analyze_logs, parse_metadata, discover_band_files, detect_repeating_pattern, unpack_frame
from .event_bus import bus, AppEvent, EventType
from .retrieval import (
    search_knowledge, knowledge_base_status,
    save_memory, recall_memory, memory_summary,
)
from .indexer import run_indexing, delete_from_index
from .meta_parser import (
    parse_meta_file, parse_ephemeris_file, merge_summaries,
    detect_mission_type, generate_full_report,
)
from .template_learner import learner as _learner


def _resolve_folder(folder: str = "") -> str:
    """
    Resolve dataset folder with full fallback chain. Never fails when a
    dataset is actually loaded — even if the event bus didn't fire.
      1. Explicit folder arg (if a real directory)
      2. state.active_folder
      3. Any open tab
      4. Most recent session folder
      5. Most recently scanned folder from SQLite cache
    """
    if folder and os.path.isdir(folder):
        return folder
    f = state.active_folder
    if f and os.path.isdir(f):
        return f
    for tab in state.all_tabs():
        if tab.folder and os.path.isdir(tab.folder):
            return tab.folder
    for f in state.session_folders():
        if f and os.path.isdir(f):
            return f
    try:
        for row in (state.list_cached_scans() or [])[:5]:
            f = row.get("folder", "")
            if f and os.path.isdir(f):
                return f
    except Exception:
        pass
    return ""


def _looks_like_dataset_folder(folder: str) -> bool:
    """
    Heuristic for a Band Mode dataset folder:
    it must directly contain .bandXX files.
    """
    if not folder or not os.path.isdir(folder):
        return False
    try:
        names = [n.lower() for n in os.listdir(folder)]
    except Exception:
        return False

    if any(re.search(r"\.band\d+$", n) for n in names):
        return True
    return False

# Knowledge folder — set IRIS_KNOWLEDGE_FOLDER env var or defaults to iris/knowledge/
_KNOWLEDGE_FOLDER = os.environ.get(
    "IRIS_KNOWLEDGE_FOLDER",
    os.path.join(os.path.dirname(__file__), "knowledge")
)


# ─────────────────────────────────────────────────────────────────────────────
# READ TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def tool_get_app_state() -> Dict:
    """Return a full snapshot of what's currently open in the application."""
    tabs = []
    for tab in state.all_tabs():
        tabs.append({
            "tab_index":     tab.tab_index,
            "mode":          tab.mode,
            "dataset_name":  tab.dataset_name,
            "folder":        tab.folder,
            "frame_count":   tab.frame_count,
            "current_frame": tab.current_frame,
            "band_count":    tab.band_count,
            "is_active":     tab.is_active,
        })

    result = {
        "open_tabs":      tabs,
        "active_folder":  state.active_folder,
        "active_frame":   state.active_frame,
        "session_folders": state.session_folders()[:10],
    }

    scan = state.get_scan_result(state.active_folder) if state.active_folder else None
    if scan:
        result["scan_available"] = True
        result["scan_type"]      = scan.scan_type
        result["health_score"]   = scan.health_score
        result["anomaly_count"]  = len(scan.anomaly_frames)
        result["anomaly_frames"] = scan.anomaly_frames[:30]
    else:
        result["scan_available"] = False

    return result


def tool_list_folder(folder: str) -> Dict:
    """List and categorize files in a folder."""
    if not os.path.isdir(folder):
        return {"error": f"Not a directory: {folder}"}
    try:
        files = os.listdir(folder)
    except PermissionError:
        return {"error": f"Permission denied: {folder}"}

    band_files = sorted([f for f in files if re.search(r"\.band\d+$", f, re.I)])
    log_files  = [f for f in files if f.lower().endswith((".log", ".txt"))]
    json_files = [f for f in files if f.lower().endswith(".json")]
    img_files  = [f for f in files if f.lower().endswith((".png",".jpg",".tif",".tiff"))]

    meta = parse_metadata(folder)
    bands = discover_band_files(folder, meta)

    return {
        "folder":        folder,
        "total_files":   len(files),
        "band_files":    band_files,
        "band_count":    len({re.search(r"\.band(\d)",f,re.I).group(1) for f in band_files if re.search(r"\.band(\d)",f,re.I)}),
        "log_files":     log_files,
        "json_files":    json_files,
        "image_files":   img_files[:10],
        "metadata":      meta,
        "band_details":  [{"key":b["key"],"frames":b["n_frames"],"size":b["width"]} for b in bands],
    }


def tool_read_logs(folder: str) -> Dict:
    """
    Read and parse all log/txt files in folder (and subfolders).
    Extracts: FPS, frame counts, errors, warnings, timing stability.
    """
    if not os.path.isdir(folder):
        return {"error": f"Not a directory: {folder}"}
    return analyze_logs(folder)


def tool_get_scan_results(folder: str) -> Dict:
    """Return existing scan results for a folder without running a new scan."""
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}

    result = state.get_scan_result(folder)
    if not result:
        return {
            "status":  "NO_RESULTS",
            "message": "No scan results found. Use run_scan to scan this folder first.",
            "folder":  folder,
        }

    return {
        "folder":        result.folder,
        "scan_type":     result.scan_type,
        "health_score":  result.health_score,
        "anomaly_frames": result.anomaly_frames,
        "findings":      result.findings[:50],
        "band_summary":  result.band_summary,
        "log_summary":   {
            "error_count":     result.log_summary.get("error_count", 0),
            "warning_count":   result.log_summary.get("warning_count", 0),
            "frame_loss_pct":  result.log_summary.get("frame_loss_pct", 0),
            "timing":          result.log_summary.get("timing", {}),
        } if result.log_summary else {},
        "duration_sec":  result.duration_sec,
        "scanned_at":    time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.localtime(result.scanned_at)),
    }


def tool_get_frame_info(folder: str, frame_index: int) -> Dict:
    """
    Get all known information about a specific frame:
    any scan findings, position in anomaly list, what to look for.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}

    findings_for_frame = []
    scan = state.get_scan_result(folder)
    if scan:
        findings_for_frame = [f for f in scan.findings
                               if f.get("frame") == frame_index]

    return {
        "folder":       folder,
        "frame_index":  frame_index,
        "findings":     findings_for_frame,
        "is_anomaly":   frame_index in (scan.anomaly_frames if scan else []),
        "total_frames": state.active_tab.frame_count if state.active_tab else "?",
    }


def tool_find_anomaly_frames(folder: str, anomaly_type: str = None) -> Dict:
    """
    Return the list of frames with anomalies, optionally filtered by type.
    anomaly_type options: black_frame, dead_columns, vertical_striping,
                          saturation, dead_pixels, cross_band_outlier, etc.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}

    scan = state.get_scan_result(folder)
    if not scan:
        return {"error": "No scan results. Run run_scan first.", "folder": folder}

    if anomaly_type:
        filtered = [f for f in scan.findings
                    if anomaly_type.lower() in f.get("type","").lower()]
        frames = sorted(set(f["frame"] for f in filtered if f.get("frame") is not None))
    else:
        frames = scan.anomaly_frames

    return {
        "folder":        folder,
        "anomaly_type":  anomaly_type or "all",
        "frames":        frames,
        "count":         len(frames),
        "total_frames":  max((b.get("n_frames",0) for b in scan.band_summary.values()), default=0),
        "health_score":  scan.health_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ACT TOOLS
# ─────────────────────────────────────────────────────────────────────────────

def tool_run_scan(
    folder: str,
    mode: str = "quick",
    _force: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Launch a dataset scan on a folder.
    mode: "quick" (~3s, logs/meta/ephemeris only), "sample" (~20s, sparse frame sampling),
          "full" (all frames, 1-5 min, explicit frame/pixel analysis).
    This is blocking — the agent waits for it to finish.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}
    if not os.path.isdir(folder):
        return {"error": f"Folder not found: {folder}"}
    if mode not in ("quick", "sample", "full"):
        mode = "quick"
    if _force:
        try:
            state.invalidate_scan(folder, reason="Force scan requested.")
        except Exception:
            pass

    # Run synchronously in the current thread (agent worker thread)
    # We don't use QThread here because we're already in a worker thread
    from .scanner import ScanWorker
    import threading

    result_holder = {}
    error_holder  = {}
    done_event    = threading.Event()

    class _InlineWorker(ScanWorker):
        def run(self):
            try:
                r = self._scan()
                if r:
                    # Store immediately before signalling done so any caller
                    # that queries state after done_event.wait() sees the result.
                    state.store_scan_result(r)
                    result_holder["result"] = r
                else:
                    error_holder["error"] = "Scan returned no result"
            except Exception as e:
                error_holder["error"] = str(e)
            finally:
                done_event.set()

    worker = _InlineWorker(folder, mode, progress_cb=progress_cb)

    # Run in a thread so we can join it
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()

    # Wait up to 10 minutes
    done_event.wait(timeout=600)

    if "error" in error_holder:
        return {"error": error_holder["error"]}

    r = result_holder.get("result")
    if not r:
        return {"error": "Scan timed out or returned no result"}

    # Note: state.store_scan_result(r) was already called inside _InlineWorker.run().

    return {
        "status":        "COMPLETE",
        "folder":        folder,
        "scan_type":     r.scan_type,
        "health_score":  r.health_score,
        "anomaly_frames": r.anomaly_frames,
        "anomaly_count": len(r.anomaly_frames),
        "critical_count": sum(1 for f in r.findings if f["severity"] == "CRITICAL"),
        "warning_count":  sum(1 for f in r.findings if f["severity"] == "WARNING"),
        "top_findings":  r.findings[:10],
        "band_summary":  r.band_summary,
        "duration_sec":  r.duration_sec,
    }


def tool_navigate_to_frame(frame_index: int, folder: str = None) -> Dict:
    """
    Move the application frame slider to frame_index.
    The frame viewer will update immediately.
    Supports natural user input (1-based frame numbers) for compatibility.
    """
    # Find the right tab
    tab = None
    if folder:
        tab = state.tab_by_folder(folder)
    if not tab:
        tab = state.active_tab

    # Fall back to first available tab if dataset is in process of opening
    if not tab:
        tabs = state.all_tabs()
        if tabs:
            tab = tabs[0]

    if not tab:
        return {"error": "No dataset is currently open."}

    widget = tab.widget_ref or state.active_widget
    if not widget:
        return {"error": "Widget reference lost. Please reload the dataset."}

    # acceptance: if user asks for 1..N, map to 0..N-1 when frame_count exists.
    if tab.frame_count > 0 and 1 <= frame_index <= tab.frame_count:
        frame_index -= 1
    if frame_index < 0:
        frame_index = 0
    if tab.frame_count > 0 and frame_index >= tab.frame_count:
        frame_index = tab.frame_count - 1

    final_index = frame_index

    def _do():
        try:
            if hasattr(widget, "frame_slider") and widget.frame_slider is not None:
                widget.frame_slider.setValue(final_index)
            elif hasattr(tab, "widget_ref") and hasattr(tab.widget_ref, "frame_slider"):
                tab.widget_ref.frame_slider.setValue(final_index)
            else:
                print("[Iris] navigate_to_frame: target widget has no frame_slider")
        except Exception as e:
            print(f"[Iris] navigate_to_frame error: {e}")

    QTimer.singleShot(0, _do)

    return {
        "status":      "OK",
        "frame_index": final_index,
        "dataset":     tab.dataset_name,
        "message":     f"Navigated to frame {final_index} in {tab.dataset_name}",
    }


def tool_open_dataset(folder_path: str) -> Dict:
    """Open a dataset folder in a new Band Mode tab."""
    if not os.path.isdir(folder_path):
        return {"error": f"Folder not found: {folder_path}"}
    # bus.emit() is thread-safe — no QTimer or sleep needed.
    # The OPEN_DATASET subscriber in __init__.py uses QTimer.singleShot itself,
    # so the actual tab open happens on the Qt main thread asynchronously.
    try:
        bus.emit(AppEvent(EventType.OPEN_DATASET, {"folder": folder_path}, source="iris"))
    except Exception as e:
        return {"error": f"Failed to emit open event: {e}"}
    return {
        "status":  "OK",
        "folder":  folder_path,
        "message": f"Opened {os.path.basename(folder_path)}",
    }


def tool_open_last_session() -> Dict:
    """
    Open the most recent dataset from memory/session files.
    Resolution order:
    1) iris_memory.json last_active_dataset.path
    2) iris_memory.json recent_datasets[]
    3) last_session.json band mode folder
    4) recent.json entries
    """
    candidates = []

    # 1) Iris memory
    mem_path = os.path.join(os.path.dirname(__file__), "iris_memory.json")
    try:
        if os.path.exists(mem_path):
            mem = json.loads(open(mem_path, "r", encoding="utf-8").read())
            p = ((mem.get("last_active_dataset") or {}).get("path") or "").strip()
            if p:
                candidates.append(("iris_memory", p))
            for item in (mem.get("recent_datasets") or []):
                p = (item.get("path") or "").strip()
                if p:
                    candidates.append(("iris_recent", p))
    except Exception:
        pass

    # 2) last_session.json
    try:
        if os.path.exists("last_session.json"):
            data = json.loads(open("last_session.json", "r", encoding="utf-8").read())
            band = (data.get("modes", {}).get("band") or [])
            if band and isinstance(band, list):
                st = band[0].get("state", {}) if band[0] else {}
                p = (st.get("folder") or "").strip()
                if p:
                    candidates.append(("last_session", p))
    except Exception:
        pass

    # 4) recent.json
    try:
        if os.path.exists("recent.json"):
            rec = json.loads(open("recent.json", "r", encoding="utf-8").read())
            if isinstance(rec, list) and rec:
                for item in rec:
                    p = (item.get("path") or "").strip()
                    if p:
                        candidates.append(("recent", p))
    except Exception:
        pass

    seen = set()
    for source, p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.isdir(p) and _looks_like_dataset_folder(p):
            r = tool_open_dataset(p)
            if r.get("error"):
                continue
            r["source"] = source
            r["folder"] = p
            return r

    return {"error": "No valid last session dataset found."}


def tool_set_zoom(level: float) -> Dict:
    """
    Set the zoom level on the active frame viewer.
    level: e.g. 1.0 = fit to window, 2.0 = 2x, 0.5 = 0.5x
    """
    tab = state.active_tab
    if not tab or not tab.widget_ref:
        return {"error": "No active viewer."}

    widget = tab.widget_ref

    def _do():
        try:
            # Try common zoom method names
            for method in ("set_zoom", "setZoom", "zoom_to"):
                if hasattr(widget, method):
                    getattr(widget, method)(level)
                    return
            # Try the image viewer directly
            for attr in ("image_viewer", "viewer", "band_viewer"):
                v = getattr(widget, attr, None)
                if v and hasattr(v, "set_zoom"):
                    v.set_zoom(level)
                    return
        except Exception as e:
            print(f"[Iris] set_zoom error: {e}")

    QTimer.singleShot(0, _do)
    return {"status": "OK", "zoom_level": level}


def tool_close_tab(tab_index: int) -> Dict:
    """Close a specific tab by index."""
    bus.emit(AppEvent(EventType.CLOSE_TAB, {"tab_index": tab_index}, source="iris"))
    return {"status": "OK", "message": f"Closed tab {tab_index}"}


def tool_control_app(
    action: str,
    tab_index: int = -1,
    mode: str = "",
    value: str = "",
    folder: str = "",
    dataset_name: str = "",
    x: Optional[float] = None,
    y: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> Dict:
    """
    General Iris control surface for application actions that are not analysis tools.
    """
    action = (action or "").strip().lower()
    if not action:
        return {"error": "Missing action."}

    allowed = {
        "switch_tab",
        "add_tab",
        "set_theme",
        "toggle_theme",
        "play_pause",
        "play",
        "pause",
        "next_frame",
        "prev_frame",
        "refresh",
        "reload",
        "save_progress",
        "export_image",
        "fit_to_screen",
        "actual_size",
        "auto_contrast",
        "open_terminal",
        "close_terminal",
        "toggle_terminal",
        "close_tab",
        "open_view",
        "close_view",
        "set_band_gap",
        "set_histogram_bands",
        "open_magnifier",
        "close_magnifier",
        "set_magnifier_center",
        "set_magnifier_zoom",
        "set_contrast",
    }
    if action not in allowed:
        return {"error": f"Unsupported action: {action}"}

    payload = {
        "action": action,
        "tab_index": int(tab_index),
        "mode": mode,
        "value": value,
        "folder": folder,
        "dataset_name": dataset_name,
        "x": x,
        "y": y,
        "min": min_value,
        "max": max_value,
        "enabled": enabled,
    }
    try:
        bus.emit(AppEvent(EventType.CONTROL_APP, payload, source="iris"))
    except Exception as e:
        return {"error": f"Failed to emit control event: {e}"}

    msg = f"Executed action: {action}"
    if action == "switch_tab":
        if tab_index >= 0:
            msg = f"Switched to tab {tab_index}"
        elif dataset_name:
            msg = f"Switched to dataset tab: {dataset_name}"
        elif folder:
            msg = f"Switched to dataset tab for: {folder}"
    elif action == "add_tab":
        msg = f"Opened new {mode or 'band'} tab"
    elif action == "set_theme":
        msg = f"Theme set to {value or 'requested mode'}"
    elif action == "close_tab":
        msg = f"Closed tab {tab_index if tab_index >= 0 else 'current'}"
    return {"status": "OK", "action": action, "message": msg}


def tool_generate_report(folder: str, include_examples: bool = False, 
                        enable_template_comparison: bool = False) -> Dict:
    """
    Generate the full structured report from all available sources:
    .log (always) + .meta + *_ephemeris.txt (if present).
    Runs mission detection, template learning, and deviation flagging automatically.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}

    cached = state.get_report(folder)
    if cached:
        return {"report": cached, "folder": folder, "cached": True}

    scan = state.get_scan_result(folder)
    if not scan:
        return {"error": "No scan results available. Run run_scan first."}

    # Parse companion files if not already done
    if not scan.meta_summary:
        scan.meta_summary  = parse_meta_file(folder)
    if not scan.ephem_summary:
        scan.ephem_summary = parse_ephemeris_file(folder)

    log     = scan.log_summary or {}
    log_text = ""
    for fn in (os.listdir(folder) if os.path.isdir(folder) else []):
        if fn.lower().endswith(".log"):
            try:
                with open(os.path.join(folder, fn), errors="ignore") as f:
                    log_text += f.read()
            except Exception:
                pass

    # Detect mission type
    m_det        = detect_mission_type(log_text, log, scan.meta_summary)
    mission_type = m_det["mission_type"]

    # Learn from this scan (updates template baseline)
    _learner.ingest(log, scan.meta_summary, log_text, mission_type)

    # Flag deviations vs baseline (only if requested)
    devs = []
    if enable_template_comparison:
        devs = _learner.flag_deviations(log, scan.meta_summary, log_text, mission_type)

    # ── Merge summaries (computes GSD + cross-validation) ────────────────
    fps_applied = log.get("parameters_applied", {}).get("FPS") or \
                  log.get("procmode", {}).get("decoded", {}).get("fps_requested")
    merged = merge_summaries(log, scan.meta_summary, scan.ephem_summary, fps_applied)
    scan.merged_summary = merged
    state.store_scan_result(scan)   # persist merged_summary to SQLite cache

    # ── KB context — search local KB first, fall back to Supabase ────────
    # We use local_rag (offline, instant) and only hit Supabase for topics
    # not covered locally.  Both return the same interface.
    from .local_rag import search_local_kb

    def _kb_search(query: str, top_k: int = 2) -> str:
        """Search local KB; fall back to Supabase if local returns nothing."""
        local = search_local_kb(query, top_k=top_k)
        if local.get("total_found", 0) > 0:
            return local["context_text"]
        remote = search_knowledge(query, top_k=top_k)
        if remote.get("total_found", 0) > 0:
            return remote["context_text"]
        return ""

    kb_excerpts: dict = {}

    # Always-present queries — these cover the most common report sections
    kb_excerpts["tdi_yshift"] = _kb_search(
        "TDIYShift RegionHeight divided TDI stages expected behaviour")
    kb_excerpts["binning"] = _kb_search(
        "binning byte CCSDS parameter log confirmation verification")
    kb_excerpts["trigger_w01"] = _kb_search(
        "W01 stale UTC trigger timestamp parameter file date fallback")
    kb_excerpts["width_geometry"] = _kb_search(
        "sensor width 8448 4224 region split unbinned band half width")
    kb_excerpts["gsd"] = _kb_search(
        "GSD ground sample distance along track FPS altitude calculation")

    # Conditional queries based on what was actually found
    tdi_on = log.get("procmode", {}).get("tdi_decoded", {}).get("tdi_on", False)
    if tdi_on:
        kb_excerpts["tdi_operation"] = _kb_search(
            "TDI mode operation stages RegionHeight frame overlap")

    fw = log.get("firmware_version", "")
    if fw:
        kb_excerpts["firmware_temp"] = _kb_search(
            f"firmware {fw} temperature sensor credibility DeviceTemperature")

    for f in scan.findings:
        t = f.get("type", "")
        if t in ("dead_pixels", "hot_pixels") and "pixel_defect" not in kb_excerpts:
            kb_excerpts["pixel_defect"] = _kb_search(
                "dead hot pixel defect correction sensor calibration")
        if t == "vertical_striping" and "striping" not in kb_excerpts:
            kb_excerpts["striping"] = _kb_search(
                "vertical striping fixed pattern noise ADC flat field correction")
        if t == "cross_band_outlier" and "cross_band" not in kb_excerpts:
            kb_excerpts["cross_band"] = _kb_search(
                "cross band spectral outlier binning ADC calibration")
        if t == "alternating_row_banding" and "alt_row" not in kb_excerpts:
            kb_excerpts["alt_row"] = _kb_search(
                "alternating row banding TDI dual ADC even odd interleaving")

    for dev in devs:
        if "temp" in dev["field"] and "temp_range" not in kb_excerpts:
            kb_excerpts["temp_range"] = _kb_search(
                "sensor temperature operating range DeviceTemperature max threshold")
        if "fps" in dev["field"] and "fps_limit" not in kb_excerpts:
            kb_excerpts["fps_limit"] = _kb_search(
                "FPS cap hardware maximum TDI mode AcquisitionFrameRateMax")

    # Strip empty strings so report assembly has clean data
    kb_excerpts = {k: v for k, v in kb_excerpts.items() if v}

    report = generate_full_report(
        scan_result=scan, folder=folder,
        mission_type=mission_type,
        template_deviations=devs,
        kb_excerpts=kb_excerpts,
        enable_template_comparison=enable_template_comparison,
    )

    state.store_report(folder, report)
    return {
        "report":              report,
        "folder":              folder,
        "cached":              False,
        "mission_type":        mission_type,
        "mission_confidence":  m_det["confidence"],
        "template_deviations": len(devs),
        "kb_excerpts_used":    len(kb_excerpts),
    }


def _sample_log_for_llm(folder: str, max_chars: int = 120000) -> str:
    """
    Build a compact log sample for LLM review.
    Keeps head + tail and prioritizes error/warn lines in the middle.
    """
    if not folder or not os.path.isdir(folder):
        return ""

    log_files = [
        os.path.join(folder, fn)
        for fn in os.listdir(folder)
        if fn.lower().endswith(".log")
    ]
    if not log_files:
        return ""

    # Read all logs (usually one). If huge, keep head + tail + notable lines.
    chunks: List[str] = []
    remaining = max_chars

    for path in log_files:
        try:
            with open(path, errors="ignore") as f:
                data = f.read()
        except Exception:
            continue

        if len(data) <= remaining:
            chunks.append(data)
            remaining -= len(data)
            continue

        # Too big: keep head + tail and any error-ish lines
        head_len = max_chars // 3
        tail_len = max_chars // 3
        head = data[:head_len]
        tail = data[-tail_len:]

        # Extract notable lines with line numbers from the full log
        mid_budget = max_chars - len(head) - len(tail)
        notable: List[str] = []
        if mid_budget > 2000:
            for i, line in enumerate(data.splitlines(), start=1):
                l = line.lower()
                if any(k in l for k in ("error", "warn", "fail", "timeout", "invalid", "e09", "w01")):
                    notable.append(f"{i:06d}: {line}")
                    if sum(len(x) + 1 for x in notable) > mid_budget:
                        break

        block = "\n".join([
            "[LOG_HEAD]",
            head,
            "\n[LOG_NOTES]",
            "\n".join(notable),
            "\n[LOG_TAIL]",
            tail,
        ])
        chunks.append(block)
        remaining = 0
        break

    return "\n".join(chunks).strip()


def tool_llm_log_audit(folder: str, max_chars: int = 120000) -> Dict:
    """
    Ask Ollama to audit raw logs for suspicious gaps or anomalies
    that may be missed by rules.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}

    log_sample = _sample_log_for_llm(folder, max_chars=max_chars)
    if not log_sample:
        return {"error": "No .log files found for LLM audit."}

    try:
        from .ollama import ask_iris, available_models
    except Exception as e:
        return {"error": f"Ollama not available: {e}"}

    if not available_models():
        return {"error": "Ollama has no models available. Run: ollama pull gemma3:4b"}

    prompt = (
        "You are a strict log auditor for IRIS camera acquisitions. "
        "Review the provided log excerpt and identify any suspicious gaps, "
        "missing sequences, resets, unexplained delays, or anomalies that "
        "might be missed by rule-based checks.\n\n"
        "Rules:\n"
        "- Do NOT invent errors. If nothing stands out, say 'No additional concerns found.'\n"
        "- When you flag something, quote the relevant line (or line number) from the excerpt.\n"
        "- Keep output short: 3-8 bullet points max.\n\n"
        "Log excerpt:\n"
        f"{log_sample}"
    )

    try:
        audit = ask_iris(question=prompt, context="", action="question")
    except Exception as e:
        return {"error": f"Ollama audit failed: {e}"}

    return {
        "folder": folder,
        "audit": audit.strip(),
    }


def tool_detect_repeating_pattern(folder: str, band: str = "b0", frame_index: int = 0) -> Dict:
    """
    On-demand repeating pattern anomaly detector for a specific band/frame.
    Auto-detects repeating tile size and flags deviating tiles.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}
    if not os.path.isdir(folder):
        return {"error": f"Folder not found: {folder}"}

    meta = parse_metadata(folder)
    bands = discover_band_files(folder, meta)
    if not bands:
        return {"error": "No band files found."}

    band_map = {b["key"]: b for b in bands}
    b = band_map.get(band, bands[0])
    bpf = b["bpf"]
    if frame_index < 0:
        frame_index = 0
    if b["n_frames"] > 0 and frame_index >= b["n_frames"]:
        frame_index = b["n_frames"] - 1

    try:
        with open(b["path"], "rb") as bf:
            bf.seek(frame_index * bpf)
            raw = bf.read(bpf)
        if len(raw) < bpf:
            return {"error": "Frame data truncated."}
        arr = unpack_frame(raw, b["width"], b["height"], b["bit_depth"])
        rep = detect_repeating_pattern(arr)
        if not rep:
            return {"status": "NO_PATTERN", "message": "No repeating pattern detected.", "band": b["key"], "frame": frame_index}
        return {"status": "OK", "band": b["key"], "frame": frame_index, "repeat": rep}
    except Exception as e:
        return {"error": str(e)}


def tool_compare_datasets(folder_a: str, folder_b: str) -> Dict:
    """
    Compare two datasets side by side.
    If folder_a or folder_b is blank, uses the two most recently active tabs.
    Returns a structured diff: health, anomaly overlap, band differences,
    log differences, and which issues appear in only one dataset.
    """
    # Resolve blank folders from open tabs
    tabs = state.all_tabs()
    tab_folders = [t.folder for t in tabs if t.folder]

    if not folder_a:
        folder_a = tab_folders[0] if len(tab_folders) >= 1 else ""
    if not folder_b:
        folder_b = tab_folders[1] if len(tab_folders) >= 2 else ""

    if not folder_a or not folder_b:
        return {
            "error": "Need two datasets to compare. "
                     "Open two datasets in separate tabs, or provide folder paths. "
                     f"Currently open: {tab_folders}"
        }
    if folder_a == folder_b:
        return {"error": "Both folders are the same. Provide two different datasets."}

    result_a = state.get_scan_result(folder_a)
    result_b = state.get_scan_result(folder_b)

    missing = []
    if not result_a:
        missing.append(f"{os.path.basename(folder_a)} (not scanned)")
    if not result_b:
        missing.append(f"{os.path.basename(folder_b)} (not scanned)")
    if missing:
        return {
            "error": f"Missing scan results for: {', '.join(missing)}. "
                     "Scan both datasets first.",
            "suggestion": "Call run_scan on each folder then retry compare_datasets."
        }

    name_a = os.path.basename(folder_a)
    name_b = os.path.basename(folder_b)

    # Anomaly frame sets
    set_a = set(result_a.anomaly_frames)
    set_b = set(result_b.anomaly_frames)
    shared = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    # Finding type counts
    def type_counts(findings):
        counts = {}
        for f in findings:
            t = f.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    types_a = type_counts(result_a.findings)
    types_b = type_counts(result_b.findings)
    all_types = set(types_a) | set(types_b)

    type_diff = {}
    for t in sorted(all_types):
        ca, cb = types_a.get(t, 0), types_b.get(t, 0)
        if ca != cb:
            type_diff[t] = {"in_a": ca, "in_b": cb}

    # Band mean comparison
    band_diff = {}
    all_band_keys = set(result_a.band_summary) | set(result_b.band_summary)
    for k in sorted(all_band_keys):
        ma = result_a.band_summary.get(k, {}).get("mean_dn")
        mb = result_b.band_summary.get(k, {}).get("mean_dn")
        if ma is not None and mb is not None:
            diff = round(mb - ma, 1)
            if abs(diff) > 5:  # only report meaningful differences
                band_diff[k] = {"a": ma, "b": mb, "delta": diff}

    return {
        "dataset_a": {
            "name":           name_a,
            "folder":         folder_a,
            "health":         result_a.health_score,
            "anomaly_count":  len(result_a.anomaly_frames),
            "critical":       sum(1 for f in result_a.findings if f["severity"] == "CRITICAL"),
            "warnings":       sum(1 for f in result_a.findings if f["severity"] == "WARNING"),
            "scan_type":      result_a.scan_type,
        },
        "dataset_b": {
            "name":           name_b,
            "folder":         folder_b,
            "health":         result_b.health_score,
            "anomaly_count":  len(result_b.anomaly_frames),
            "critical":       sum(1 for f in result_b.findings if f["severity"] == "CRITICAL"),
            "warnings":       sum(1 for f in result_b.findings if f["severity"] == "WARNING"),
            "scan_type":      result_b.scan_type,
        },
        "comparison": {
            "health_delta":          round(result_b.health_score - result_a.health_score, 1),
            "shared_anomaly_frames": shared[:30],
            "only_in_a":             only_a[:30],
            "only_in_b":             only_b[:30],
            "shared_count":          len(shared),
            "only_in_a_count":       len(only_a),
            "only_in_b_count":       len(only_b),
            "finding_type_diff":     type_diff,
            "band_mean_diff":        band_diff,
        },
        "verdict": (
            f"{name_a} is healthier by {abs(result_b.health_score - result_a.health_score):.0f} points."
            if result_a.health_score > result_b.health_score else
            f"{name_b} is healthier by {abs(result_b.health_score - result_a.health_score):.0f} points."
            if result_b.health_score > result_a.health_score else
            "Both datasets have the same health score."
        ),
    }




# ─────────────────────────────────────────────────────────────────────────────
# NEW TOOLS: Refresh, Tree Discovery, Dataset Listing
# ─────────────────────────────────────────────────────────────────────────────

def tool_extract_all_log_parameters(root_folder: str = "") -> Dict:
    """
    Extract ALL numeric parameters from all .log files under root_folder.
    
    Builds a comprehensive parameter database by:
    1. Scanning every log file for numeric parameters (FPS, exposure, gain, TDI, temps, etc.)
    2. Grouping by parameter name across all sessions
    3. Computing statistics: count, mean, median, min, max, stdev
    
    Returns:
    - parameter_stats: {param_name: {count, mean, median, min, max, stdev, unit}, ...}
    - session_parameters: [{session, fps, exposure_time, gain, temps, ...}, ...]
    - Can answer arbitrary questions without code changes:
      * "lowest fps recorded"
      * "average exposure time"
      * "highest gain" 
      * "fps outliers"
      * "compare fps vs temperature"
      etc.
    """
    import statistics as _stats
    
    if not root_folder:
        root_folder = state.active_folder
    if not root_folder or not os.path.isdir(root_folder):
        return {"error": f"Folder not found: {root_folder}"}
    
    # Collect all log files
    log_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in sorted(filenames):
            if f.lower().endswith(".log"):
                log_files.append(os.path.join(dirpath, f))
    
    if not log_files:
        return {"error": f"No .log files found under {root_folder}"}
    
    # Dictionary to accumulate all parameter values: {param_name: [values]}
    param_values: Dict[str, List[float]] = {}
    session_parameters = []
    
    for lf in log_files:
        try:
            with open(lf, errors="ignore") as fh:
                content = fh.read()
        except Exception as e:
            continue
        
        acq_name = os.path.basename(lf).replace(".log", "")
        session_data = {"session": acq_name}
        
        # ── Temperature parameters ────────────────────────────────────────
        dev_temps = [int(x) for x in re.findall(r"Device Core Temperature:\s*(\d+)", content)]
        sens_temps = [float(x) for x in re.findall(r"\[I54\].*SensorTemp:\s*([\d.]+)", content)]
        
        if sens_temps:
            session_data["sensor_temp_start_c"] = sens_temps[0]
            session_data["sensor_temp_end_c"] = sens_temps[-1]
            session_data["sensor_temp_mean_c"] = round(_stats.mean(sens_temps), 2)
            param_values.setdefault("sensor_temp_c", []).extend(sens_temps)
        
        if dev_temps:
            session_data["core_temp_start_c"] = dev_temps[0]
            session_data["core_temp_end_c"] = dev_temps[-1]
            session_data["core_temp_mean_c"] = round(_stats.mean(dev_temps), 2)
            param_values.setdefault("core_temp_c", []).extend(dev_temps)
        
        # ── FPS ────────────────────────────────────────────────────────────
        m_fps_req = re.search(r"FPS requested:\s*([\d.]+)", content)
        m_fps_app = re.search(r"FPS applied:\s*([\d.]+)", content)
        m_fps_cap = re.search(r"max FPS:\s*([\d.]+)", content)
        
        if m_fps_req:
            fps_req = float(m_fps_req.group(1))
            session_data["fps_requested"] = fps_req
            param_values.setdefault("fps_requested", []).append(fps_req)
        
        if m_fps_app:
            fps_app = float(m_fps_app.group(1))
            session_data["fps_applied"] = fps_app
            param_values.setdefault("fps_applied", []).append(fps_app)
        
        if m_fps_cap:
            fps_cap = float(m_fps_cap.group(1))
            session_data["fps_capped"] = fps_cap
            param_values.setdefault("fps_capped", []).append(fps_cap)
        
        # ── Exposure time ──────────────────────────────────────────────────
        m_exp_req = re.search(r"Exposure.*requested:\s*([\d.]+)", content, re.I)
        m_exp_app = re.search(r"Set Exposure Time=\s*([\d.]+)", content)
        m_max_exp = re.search(r"MaxExpTime=\s*([\d.]+)", content)
        
        if m_exp_req:
            exp_req = float(m_exp_req.group(1))
            session_data["exposure_requested_us"] = exp_req
            param_values.setdefault("exposure_requested_us", []).append(exp_req)
        
        if m_exp_app:
            exp_app = float(m_exp_app.group(1))
            session_data["exposure_applied_us"] = exp_app
            param_values.setdefault("exposure_applied_us", []).append(exp_app)
        
        if m_max_exp:
            max_exp = float(m_max_exp.group(1))
            session_data["max_exposure_us"] = max_exp
            param_values.setdefault("max_exposure_us", []).append(max_exp)
        
        # ── Gain ────────────────────────────────────────────────────────
        m_gain = re.search(r"Gain\s*[:=]\s*([\d.]+)", content, re.I)
        if m_gain:
            gain = float(m_gain.group(1))
            session_data["gain_db"] = gain
            param_values.setdefault("gain_db", []).append(gain)
        
        # ── Frame accounting ────────────────────────────────────────────────
        m_total_f = re.search(r"Total Frames expected:\s*(\d+)", content)
        m_capt_f = re.search(r"Captured:\s*(\d+)", content)
        
        if m_total_f:
            total_f = int(m_total_f.group(1))
            session_data["total_frames"] = total_f
            param_values.setdefault("total_frames", []).append(total_f)
        
        if m_capt_f:
            capt_f = int(m_capt_f.group(1))
            session_data["captured_frames"] = capt_f
            param_values.setdefault("captured_frames", []).append(capt_f)
        
        # ── TDI parameters ─────────────────────────────────────────────────
        m_tdi_stages = re.search(r"TDI.*stages[:\s]*(\d+)", content, re.I)
        m_region_h = re.search(r"Region Height:\s*(\d+)", content)
        
        if m_tdi_stages:
            tdi_st = int(m_tdi_stages.group(1))
            session_data["tdi_stages"] = tdi_st
            param_values.setdefault("tdi_stages", []).append(tdi_st)
        
        if m_region_h:
            rh = int(m_region_h.group(1))
            session_data["region_height"] = rh
            param_values.setdefault("region_height", []).append(rh)
        
        # ── Binning ─────────────────────────────────────────────────────
        m_binning = re.search(r"Binning\s*[:=]\s*(\d+)", content, re.I)
        if m_binning:
            binning = int(m_binning.group(1))
            session_data["binning"] = binning
            param_values.setdefault("binning", []).append(binning)
        
        session_parameters.append(session_data)
    
    if not session_parameters:
        return {"error": "No parameters found in any log."}
    
    # Compute statistics for each parameter
    def compute_stats(values):
        if not values:
            return {}
        return {
            "count": len(values),
            "mean": round(_stats.mean(values), 2),
            "median": round(_stats.median(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "stdev": round(_stats.stdev(values), 2) if len(values) > 1 else 0.0,
        }
    
    parameter_stats = {param: compute_stats(vals) for param, vals in param_values.items()}
    
    return {
        "folder": root_folder,
        "num_sessions": len(session_parameters),
        "parameter_stats": parameter_stats,
        "session_parameters": session_parameters,
        "available_parameters": sorted(parameter_stats.keys()),
        "note": (
            "This extraction includes ALL numeric parameters from logs. "
            "Ask ANY question: 'lowest fps', 'average exposure', 'gain outliers', "
            "'fps vs temperature correlation', 'highest frame count', etc. "
            "No code changes needed—Iris will analyze on the fly."
        ),
    }


def tool_extract_sensor_data(root_folder: str = "") -> Dict:
    """
    [DEPRECATED: Use tool_extract_all_log_parameters instead]
    Extract raw sensor data only from all .log files under root_folder.
    """
    result = tool_extract_all_log_parameters(root_folder)
    if result.get("error"):
        return result
    
    # Return sensor-only view for backward compatibility
    return {
        "folder": result.get("folder"),
        "num_sessions": result.get("num_sessions"),
        "summary": {k: v for k, v in result.get("parameter_stats", {}).items() if "temp" in k},
        "note": "Use tool_extract_all_log_parameters for full parameter extraction."
    }


def tool_session_report(root_folder: str = "") -> Dict:
    """
    Scan every .log file under root_folder (no band files needed) and produce
    a findings-only report — one flag per observed issue, grouped by category.
    No tables. Every finding names the affected session(s) and explains what
    was observed, what was expected, and what to do about it.
    """
    import statistics as _stats

    if not root_folder:
        root_folder = state.active_folder
    if not root_folder or not os.path.isdir(root_folder):
        return {"error": f"Folder not found: {root_folder}"}

    # ── Collect all log files ──────────────────────────────────────────────
    log_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in sorted(filenames):
            if f.lower().endswith(".log"):
                log_files.append(os.path.join(dirpath, f))
    log_files = sorted(log_files)

    if not log_files:
        return {"error": f"No .log files found under {root_folder}"}

    # ── Parse each log ─────────────────────────────────────────────────────
    from .scanner import analyze_logs, decode_procmode

    sessions = []
    for lf in log_files:
        log_dir = os.path.dirname(lf)
        # analyze_logs walks a folder — point it at the log's parent dir
        # but if multiple logs are in the same folder, parse individually
        try:
            with open(lf, errors="ignore") as fh:
                content = fh.read()
        except Exception as e:
            sessions.append({"name": os.path.basename(lf), "error": str(e)})
            continue

        summary = analyze_logs(log_dir)

        # Extract acquisition name from filename or from log header
        acq_name = os.path.basename(lf).replace(".log", "")
        m_started = re.search(r"Program started at:\s*(.+)", content)
        started = m_started.group(1).strip() if m_started else "?"

        # Firmware
        m_fw = re.search(r"Device Firmware Version:\s*(.+)", content)
        firmware = m_fw.group(1).strip() if m_fw else "?"

        # Disk free
        m_disk = re.search(r"Disk free space:\s*([\d.]+)", content)
        disk_gb = float(m_disk.group(1)) if m_disk else None

        # Raw args
        m_raw = re.search(r"Arguments received from parameter file:\s*(.+)", content)
        raw_args = m_raw.group(1).strip() if m_raw else ""
        arg_count = len(raw_args.split()) if raw_args else 0

        m_proc = re.search(r"Argument Processed\[(\d+)\]", content)
        proc_arg_count = int(m_proc.group(1)) if m_proc else 0

        # Decode procmode from the Argument Processed line (formatted version)
        m_proc_line = re.search(r"Argument Processed\[\d+\]:\s*(.+)", content)
        proc_line = m_proc_line.group(1).strip() if m_proc_line else raw_args
        proc = decode_procmode(proc_line) if proc_line else {}
        d = proc.get("decoded", {})
        tdi_dec = proc.get("tdi_decoded", {})

        # Applied values
        p_app = summary.get("parameters_applied", {})
        fps_applied  = p_app.get("FPS")
        exp_applied  = p_app.get("ExposureTime") or p_app.get("Exposure_Time")
        gain_applied = p_app.get("Gain")
        xsh_applied  = p_app.get("BandXShift")
        tdi_applied  = p_app.get("TDI_Modes") or p_app.get("TDIMode")
        rh_applied   = p_app.get("RegionHeight")

        # MaxExpTime
        m_maxexp = re.search(r"MaxExpTime=\s*([\d.]+)", content)
        max_exp = float(m_maxexp.group(1)) if m_maxexp else None

        # FPS cap
        fps_capped = bool(re.search(r"Updated FPS from", content))
        m_fps_cap = re.search(r"max FPS:\s*([\d.]+)", content)
        fps_cap_val = float(m_fps_cap.group(1)) if m_fps_cap else None

        # Exposure clamp
        exp_req = d.get("exposure_time")
        exp_clamped = False
        exp_clamp_note = ""
        if exp_req and exp_applied and exp_req > 0:
            diff_pct = abs(exp_applied - exp_req) / exp_req
            if diff_pct > 0.01:
                exp_clamped = True
                if max_exp and exp_req > max_exp:
                    exp_clamp_note = (
                        f"requested {exp_req:.0f}µs, MaxExpTime={max_exp:.0f}µs, "
                        f"applied {exp_applied:.1f}µs "
                        f"({exp_req/max_exp:.0f}× over limit)"
                    )
                else:
                    exp_clamp_note = (
                        f"requested {exp_req:.0f}µs → applied {exp_applied:.1f}µs "
                        f"({diff_pct*100:.1f}% difference)"
                    )

        # Temperatures — parse directly from content for start/end
        dev_temps   = [int(x) for x in re.findall(r"Device Core Temperature:\s*(\d+)", content)]
        sens_temps  = [float(x) for x in re.findall(r"\[I54\].*SensorTemp:\s*([\d.]+)", content)]
        dev_start   = dev_temps[0]  if dev_temps  else None
        dev_end     = dev_temps[-1] if dev_temps  else None
        sens_start  = sens_temps[0]  if sens_temps else None
        sens_end    = sens_temps[-1] if sens_temps else None
        sens_delta  = round(abs(sens_end - sens_start), 2) if (sens_start and sens_end) else None

        # Frame accounting
        fa = summary.get("frame_accounting", {})
        total_f    = fa.get("total_frames_expected")
        captured_f = fa.get("captured_count")
        drops      = fa.get("frames_lost", 0) or 0

        # Trigger timing
        tt = summary.get("trigger_timing", {})
        stale_trigger = tt.get("stale_timestamp_detected", False)
        utc_times = tt.get("utc_trigger_times", [])
        wait_ms   = tt.get("waiting_time_msec", 0)

        # TDIYShift
        tdiy_warn = bool(re.search(r"TDIYShift.*greater than default", content))
        g_tdi     = p_app.get("G_TDIYShift")

        # Binning arg
        binning_arg = d.get("binning_byte")

        sessions.append({
            "name":         acq_name,
            "started":      started,
            "firmware":     firmware,
            "disk_gb":      disk_gb,
            "arg_count":    arg_count,
            "proc_count":   proc_arg_count,
            "args_ok":      (arg_count == 14 and proc_arg_count == 14),
            # decoded args
            "orbit_id":     d.get("orbit_id"),
            "task_id":      d.get("task_id"),
            "json_id":      d.get("json_id"),
            "date_arg":     d.get("date"),
            "time_arg":     d.get("utc_time"),
            "duration":     d.get("duration_sec"),
            "band_sel":     d.get("band_selection"),
            "tdi_byte":     d.get("tdi_byte"),
            "tdi_mode_str": tdi_dec.get("mode", "?"),
            "tdi_stages":   tdi_dec.get("stages"),
            "fps_req":      d.get("fps_requested"),
            "fps_applied":  fps_applied,
            "fps_capped":   fps_capped,
            "fps_cap_val":  fps_cap_val,
            "exp_req":      exp_req,
            "exp_applied":  exp_applied,
            "exp_clamped":  exp_clamped,
            "exp_clamp_note": exp_clamp_note,
            "max_exp":      max_exp,
            "gain_req":     d.get("gain"),
            "gain_applied": gain_applied,
            "xshift_req":   d.get("xshift"),
            "xshift_applied": xsh_applied,
            "binning_arg":  binning_arg,
            "tdi_applied":  tdi_applied,
            "rh_applied":   rh_applied,
            "g_tdi_yshift": g_tdi,
            "tdiy_warn":    tdiy_warn,
            "tdiy_req":     d.get("tdi_yshift"),
            # temperature
            "dev_start":    dev_start,
            "dev_end":      dev_end,
            "sens_start":   sens_start,
            "sens_end":     sens_end,
            "sens_delta":   sens_delta,
            # frame accounting
            "total_frames": total_f,
            "captured":     captured_f,
            "drops":        drops,
            # trigger
            "stale_trigger":  stale_trigger,
            "trigger_utc":    utc_times[0] if utc_times else "?",
            "trigger_wait_ms": wait_ms,
            # raw issues count + camera settings flag
            "n_issues":     len(summary.get("raw_issues", [])),
            "cam_settings_invalid": any(
                i.get("category") == "camera_init"
                and "settings" in i.get("message", "").lower()
                for i in summary.get("raw_issues", [])
            ),
        })

    if not sessions:
        return {"error": "No sessions could be parsed."}

    # ── Cross-session grouping ─────────────────────────────────────────────
    n            = len(sessions)
    ok_sessions  = [s for s in sessions if not s.get("error")]

    firmware_groups: Dict[str, List] = {}
    for s in ok_sessions:
        fw = s.get("firmware", "unknown").split()[0]
        firmware_groups.setdefault(fw, []).append(s["name"])

    exp_clamped_list = [s for s in ok_sessions if s.get("exp_clamped")]
    fps_capped_list  = [s for s in ok_sessions if s.get("fps_capped")]
    arg_fail_list    = [s for s in ok_sessions if not s.get("args_ok")]
    stale_trig_list  = [s for s in ok_sessions if s.get("stale_trigger")]
    drop_list        = [s for s in ok_sessions if (s.get("drops") or 0) > 0]
    slow_proc_list   = [s for s in ok_sessions if (s.get("proc_time_sec") or 0) > 8]

    # Sensor temp groups
    sens_groups: Dict[str, List] = {}
    for s in ok_sessions:
        t = s.get("sens_start")
        if t is not None:
            group = "HIGH" if t > 40 else "LOW" if t < 20 else "MID"
            sens_groups.setdefault(group, []).append(s)

    # ── Helpers ────────────────────────────────────────────────────────────
    lines: List[str] = []

    def emit(sev: str, text: str, detail: str = ""):
        icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "⚪", "OK": "✅"}.get(sev, "•")
        lines.append(f"{icon} [{sev}] {text}")
        if detail:
            for d_line in detail.strip().splitlines():
                lines.append(f"    → {d_line.strip()}")

    def sep(title: str):
        lines.append("")
        lines.append(f"── {title} {'─' * max(0, 55 - len(title))}")

    # ── Header ─────────────────────────────────────────────────────────────
    lines.append(f"SESSION LOG REPORT  |  {n} log(s)  |  {root_folder}")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── ARGUMENT INTEGRITY ─────────────────────────────────────────────────
    sep("ARGUMENT INTEGRITY")
    if arg_fail_list:
        for s in arg_fail_list:
            missing = 14 - s["arg_count"]
            emit("CRITICAL",
                 f"{s['name']}: only {s['arg_count']}/14 arguments received "
                 f"({missing} missing) — Processed[{s['proc_count']}]",
                 f"Parameter file was truncated or corrupt.\n"
                 f"All fields after position {s['arg_count']} fell back to defaults "
                 f"and are UNVERIFIED.")
    else:
        emit("OK", f"All {n} sessions received and processed all 14 arguments.")

    # ── EXPOSURE CLAMPING ──────────────────────────────────────────────────
    sep("EXPOSURE CLAMPING")
    if exp_clamped_list:
        emit("WARNING",
             f"Camera silently clamped exposure in {len(exp_clamped_list)}/{n} sessions. "
             f"Log prints 'Passed' but applied value differs from requested.",
             "Root cause: requested ExposureTime exceeded MaxExpTime for the active\n"
             "TDI mode + FPS combination. The 'Set Exposure Time=X Passed' log line\n"
             "is misleading — always verify applied vs requested numerically.")
        for s in exp_clamped_list:
            emit("WARNING", f"  {s['name']}: {s['exp_clamp_note']}")
    else:
        emit("OK", "No exposure clamping detected in any session.")

    # ── FPS CAPPING ────────────────────────────────────────────────────────
    sep("FPS CAPPING")
    if fps_capped_list:
        emit("INFO",
             f"FPS silently capped in {len(fps_capped_list)}/{n} sessions "
             f"(hardware limit for active TDI mode).",
             "Fix: set FPS in the parameter file to the capped value to suppress the warning.")
        for s in fps_capped_list:
            emit("INFO",
                 f"  {s['name']}: requested {s.get('fps_req')} fps "
                 f"→ capped to {s.get('fps_cap_val')} fps")
    else:
        emit("OK", "No FPS capping in any session.")

    # ── FRAME DROPS ────────────────────────────────────────────────────────
    sep("FRAME DROPS")
    if drop_list:
        for s in drop_list:
            emit("CRITICAL",
                 f"{s['name']}: {s['drops']} frame(s) lost "
                 f"— captured {s.get('captured')}/{s.get('total_frames')}")
    else:
        emit("OK", f"Zero frame drops across all {n} sessions.")

    # ── TRIGGER TIMING ─────────────────────────────────────────────────────
    sep("TRIGGER TIMING")
    if len(stale_trig_list) == n:
        utc_ex = next((s["trigger_utc"] for s in ok_sessions if s.get("trigger_utc")), "?")
        emit("WARNING",
             f"ALL {n} sessions have stale UTC trigger timestamps (W01).",
             f"Parameter files reference an outdated trigger date ({utc_ex}).\n"
             f"System falls back to a 5-second default wait every time.\n"
             f"GPS/orbital trigger timing is NOT being used.\n"
             f"Fix: update the trigger timestamp in the parameter file before each pass.")
    elif stale_trig_list:
        names = ", ".join(s["name"] for s in stale_trig_list)
        emit("WARNING",
             f"Stale trigger timestamp (W01) in {len(stale_trig_list)}/{n} sessions: {names}")
    else:
        emit("OK", "All trigger timestamps are current.")

    # ── SENSOR STATISTICS ─────────────────────────────────────────────────
    sep("SENSOR DATA AGGREGATION")
    all_sens_start = [s["sens_start"] for s in ok_sessions if s.get("sens_start") is not None]
    all_sens_end   = [s["sens_end"] for s in ok_sessions if s.get("sens_end") is not None]
    all_dev_start  = [s["dev_start"] for s in ok_sessions if s.get("dev_start") is not None]
    all_dev_end    = [s["dev_end"] for s in ok_sessions if s.get("dev_end") is not None]
    
    if all_sens_start:
        mean_sens = _stats.mean(all_sens_start)
        med_sens  = _stats.median(all_sens_start)
        min_sens  = min(all_sens_start)
        max_sens  = max(all_sens_start)
        stdev_sens = _stats.stdev(all_sens_start) if len(all_sens_start) > 1 else 0
        emit("INFO",
             f"Sensor Temperature (start of acquisition):",
             f"Mean: {mean_sens:.1f}°C  |  Median: {med_sens:.1f}°C  |  "
             f"Range: {min_sens:.1f}–{max_sens:.1f}°C  |  Std Dev: {stdev_sens:.2f}°C")
    
    if all_dev_start:
        mean_dev = _stats.mean(all_dev_start)
        med_dev  = _stats.median(all_dev_start)
        min_dev  = min(all_dev_start)
        max_dev  = max(all_dev_start)
        stdev_dev = _stats.stdev(all_dev_start) if len(all_dev_start) > 1 else 0
        emit("INFO",
             f"Core Temperature (start of acquisition):",
             f"Mean: {mean_dev:.1f}°C  |  Median: {med_dev:.1f}°C  |  "
             f"Range: {min_dev:.1f}–{max_dev:.1f}°C  |  Std Dev: {stdev_dev:.2f}°C")

    # ── TEMPERATURE ────────────────────────────────────────────────────────
    sep("TEMPERATURE VARIANCE")
    if len(sens_groups) > 1:
        group_descs = []
        for group, members in sens_groups.items():
            temps = [s["sens_start"] for s in members]
            fws   = sorted({s.get("firmware","?").split()[0] for s in members})
            names = ", ".join(s["name"] for s in members[:4])
            suffix = f" (+{len(members)-4} more)" if len(members) > 4 else ""
            group_descs.append(
                f"{group} ({min(temps):.0f}–{max(temps):.0f}°C, fw {'/'.join(fws)}): "
                f"{names}{suffix}"
            )
        emit("WARNING",
             f"Sensor temperature splits into {len(sens_groups)} distinct groups "
             f"— correlated with firmware version.",
             "\n".join(group_descs))
    else:
        all_temps = [s["sens_start"] for s in ok_sessions if s.get("sens_start")]
        if all_temps:
            emit("OK",
                 f"Sensor temperatures consistent across all sessions "
                 f"({min(all_temps):.0f}–{max(all_temps):.0f}°C).")

    # Flag any large within-session sensor drift
    for s in ok_sessions:
        if (s.get("sens_delta") or 0) > 5:
            emit("INFO",
                 f"{s['name']}: sensor temp drifted {s['sens_delta']:.1f}°C "
                 f"during acquisition ({s.get('sens_start')}→{s.get('sens_end')}°C)")

    # ── CAMERA SETTINGS PATH ───────────────────────────────────────────────
    sep("CAMERA SETTINGS")
    # Use the flag populated during session parsing — the regex against "" was always False.
    cam_path_fail = [s for s in ok_sessions if s.get("cam_settings_invalid")]
    if cam_path_fail:
        names = ", ".join(s["name"] for s in cam_path_fail)
        emit("WARNING",
             f"Camera settings ZIP path invalid in {len(cam_path_fail)}/{n} session(s): {names}",
             "System fell back to factory defaults. Custom calibration not applied.\n"
             "Verify the settings file path in the configuration.")
    else:
        emit("INFO",
             "Camera settings path — no invalid-path warnings detected. "
             "If E09 appears in any log, the system falls back to factory defaults.")

    # ── TDI / TDIYSHIFT ───────────────────────────────────────────────────
    sep("TDI & TDIYSHIFT")
    emit("INFO",
         "TDI argument (arg[8]) encodes TDI on/off + stage count — it is NOT the "
         "TDI_Modes value applied. Applied TDI_Modes is derived separately by the driver.")
    tdiy_sessions = [s for s in ok_sessions if s.get("tdiy_warn")]
    if tdiy_sessions:
        ex = tdiy_sessions[0]
        if ex.get("tdi_stages") and ex.get("rh_applied"):
            expected = ex["rh_applied"] / ex["tdi_stages"]
            emit("INFO",
                 f"G_TDIYShift 'greater than default' warning appeared in "
                 f"{len(tdiy_sessions)}/{n} sessions — this is expected behaviour.",
                 f"G_TDIYShift = RegionHeight / TDI_stages "
                 f"(e.g. {ex['rh_applied']} / {ex['tdi_stages']} = {expected:.0f}). "
                 f"The log warning is informational only; no action needed.")

    # ── BINNING ────────────────────────────────────────────────────────────
    sep("BINNING")
    emit("INFO",
         "No 'Applied Binning=' confirmation line exists in any log. "
         "Binning argument is received and echoed but cannot be verified from the log alone.",
         "This is a firmware log coverage gap. "
         "Raise with firmware team to add an applied-binning confirmation line.")

    # ── FIRMWARE ───────────────────────────────────────────────────────────
    sep("FIRMWARE")
    if len(firmware_groups) > 1:
        emit("INFO",
             f"{len(firmware_groups)} different firmware versions present in this session set.")
        for fw, names in sorted(firmware_groups.items()):
            emit("INFO", f"  {fw}: {len(names)} session(s) — {', '.join(n[-6:] for n in names[:6])}"
                         f"{'...' if len(names) > 6 else ''}")
    else:
        fw = list(firmware_groups.keys())[0] if firmware_groups else "?"
        emit("OK", f"All sessions on firmware {fw}.")

    # ── SLOW PROCESSING ────────────────────────────────────────────────────
    if slow_proc_list:
        sep("SLOW DATA PROCESSING")
        for s in slow_proc_list:
            emit("INFO",
                 f"{s['name']}: data processing took {s.get('proc_time_sec'):.1f}s "
                 f"(typical <4s) — FPS remained at default {s.get('fps_applied')}.")

    # ── FOOTER ─────────────────────────────────────────────────────────────
    lines.append("")
    criticals = len(arg_fail_list) + len(drop_list)
    warnings  = len(exp_clamped_list) + len(stale_trig_list) + (1 if len(sens_groups) > 1 else 0)
    infos     = len(fps_capped_list)
    lines.append(
        f"── TOTALS: {n} sessions  |  "
        f"🔴 {criticals} critical  🟡 {warnings} warning  ⚪ {infos} info ──"
    )

    return {
        "report":   "\n".join(lines),
        "sessions": n,
        "folder":   root_folder,
    }


def tool_get_histogram_state() -> Dict:
    """
    Return the current histogram viewer state — what the user is looking at
    in the Histogram tab right now.

    Returns the dataset, frame, display mode, pixel value axis range,
    per-frame min/max, which bands are visible, and per-band statistics
    (mean, std, min, max, saturation %, black %).

    Also returns derived observations: saturation warnings, low dynamic range,
    cross-band outliers.
    """
    h = state.histogram
    if h is None:
        return {
            "histogram_active": False,
            "note": "Histogram tab has not been opened or no frame has been rendered yet."
        }

    age_sec = time.time() - h.updated_at
    result: Dict = {
        "histogram_active": True,
        "dataset":          h.dataset_name,
        "folder":           h.folder,
        "frame_index":      h.frame_index,
        "display_mode":     h.display_mode,
        "axis_min":         h.axis_min,
        "axis_max":         h.axis_max,
        "frame_pixel_min":  h.frame_min,
        "frame_pixel_max":  h.frame_max,
        "visible_bands":    h.visible_bands,
        "band_stats":       h.band_stats,
        "data_age_sec":     round(age_sec, 1),
    }

    if h.display_mode == "frame_range":
        result["range_start"] = h.range_start
        result["range_end"]   = h.range_end

    # Derived observations Iris can use directly
    observations = []

    # Saturation / black pixel check per band
    for bidx, bs in h.band_stats.items():
        sat = bs.get("saturated_pct", 0)
        blk = bs.get("black_pct", 0)
        if sat > 1.0:
            observations.append(
                f"Band {bidx} is saturating: {sat:.1f}% of pixels at max DN "
                f"({h.axis_max:.0f}). Exposure or gain may be too high."
            )
        if blk > 5.0:
            observations.append(
                f"Band {bidx} has {blk:.1f}% black pixels (DN≈0). "
                f"Camera may be covered, or band is inactive."
            )

    # Dynamic range utilisation
    if h.frame_max > 0:
        usage_pct = (h.frame_max - h.frame_min) / max(h.axis_max - h.axis_min, 1) * 100
        if usage_pct < 10:
            observations.append(
                f"Very low dynamic range utilisation: pixel values span only "
                f"{h.frame_min:.0f}–{h.frame_max:.0f} "
                f"({usage_pct:.1f}% of the full {h.axis_max:.0f}-count range). "
                f"Scene may be dark or sensor may be covered."
            )
        elif usage_pct > 95:
            observations.append(
                f"Near-full dynamic range utilised ({h.frame_min:.0f}–{h.frame_max:.0f}). "
                f"Check for clipping — some bands may be at or near saturation."
            )

    # Cross-band mean outlier check
    means = {bidx: bs["mean"] for bidx, bs in h.band_stats.items() if "mean" in bs}
    if len(means) >= 2:
        mean_vals = list(means.values())
        overall_mean = sum(mean_vals) / len(mean_vals)
        if overall_mean > 0:
            for bidx, m in means.items():
                ratio = m / overall_mean
                if ratio < 0.3 or ratio > 3.0:
                    direction = "lower" if ratio < 1 else "higher"
                    observations.append(
                        f"Band {bidx} mean DN ({m:.1f}) is significantly {direction} "
                        f"than the cross-band mean ({overall_mean:.1f}). "
                        f"Check binning byte — binned bands have a different expected DN."
                    )

    if observations:
        result["observations"] = observations

    return result


def tool_refresh_scan(
    folder: str = "",
    mode: str = "quick",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """
    Explicitly invalidate cached scan for a folder and run a fresh scan.
    Use this when the user says data changed, resolution was fixed, or
    results look stale. Always clears the old result first.
    """
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No dataset is open. Load a dataset first or provide the folder path."}
    state.invalidate_scan(folder, reason="Manual refresh requested by user.")
    return tool_run_scan(folder, mode, _force=True, progress_cb=progress_cb)


def tool_force_scan(
    folder: str,
    mode: str = "quick",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Force a fresh scan of a folder (clears caches first)."""
    return tool_run_scan(folder, mode, _force=True, progress_cb=progress_cb)


def tool_clear_cache(scope: str = "all") -> Dict:
    """
    Clear cached scan results and/or cached reports.
    scope: "all" (default), "scans", or "reports"
    """
    scope = (scope or "all").strip().lower()
    clear_scans = scope in ("all", "scans", "scan")
    clear_reports = scope in ("all", "reports", "report")
    state.clear_all_caches(clear_reports=clear_reports, clear_scans=clear_scans)
    return {
        "status": "OK",
        "cleared_scans": bool(clear_scans),
        "cleared_reports": bool(clear_reports),
    }


def _is_dataset_folder(folder: str) -> bool:
    """Return True if folder contains at least one .bandN file."""
    try:
        for f in os.listdir(folder):
            if re.search(r"\.band\d+$", f, re.I):
                return True
    except PermissionError:
        pass
    return False


def _find_logs_upward(start_folder: str, max_levels: int = 4) -> List[str]:
    """
    Walk upward from start_folder looking for .log files.
    Also checks sibling directories.
    Returns list of found log file paths.
    """
    found = []
    current = os.path.abspath(start_folder)
    for _ in range(max_levels):
        parent = os.path.dirname(current)
        if parent == current:
            break
        try:
            for f in os.listdir(parent):
                fp = os.path.join(parent, f)
                if os.path.isfile(fp) and f.lower().endswith((".log", ".txt")):
                    found.append(fp)
        except PermissionError:
            pass
        current = parent
    return found


def tool_browse_folder_tree(root_folder: str,
                             scan_mode: str = "full",
                             logs_only: bool = False,
                             progress_cb: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Walk an entire directory tree under root_folder.
    For each subfolder that looks like a dataset (contains .bandN files),
    runs a scan (or logs-only if logs_only=True) and collects results.

    Returns a summary of all datasets found and their health scores.
    Use when user points to a folder containing multiple datasets,
    or when asked to scan everything in a directory.
    """
    if not root_folder or not os.path.isdir(root_folder):
        return {"error": f"Folder not found or not a directory: {root_folder}"}

    datasets_found = []
    logs_only_folders = []
    scan_mode_effective = "quick" if logs_only else scan_mode

    # Walk entire tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Skip hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        if _is_dataset_folder(dirpath):
            datasets_found.append(dirpath)
        else:
            # Check if there are log files even without band files
            has_logs = any(f.lower().endswith((".log",)) for f in filenames)
            if has_logs:
                logs_only_folders.append(dirpath)

    if not datasets_found and not logs_only_folders:
        return {
            "root": root_folder,
            "datasets_found": 0,
            "message": "No datasets or log files found in this directory tree.",
        }

    results = []
    errors  = []

    # Scan each dataset
    for idx, folder in enumerate(datasets_found, start=1):
        try:
            if progress_cb:
                try:
                    progress_cb(f"Dataset {idx}/{len(datasets_found)}: {os.path.basename(folder)}")
                except Exception:
                    pass

            def _wrap_progress(msg: str):
                if not progress_cb:
                    return
                prefix = f"{os.path.basename(folder)}"
                try:
                    if (msg or "").startswith("PROGRESS:"):
                        body = msg.split(":", 1)[1]
                        pct_str, text = body.split("|", 1)
                        progress_cb(f"PROGRESS:{pct_str}|{prefix} — {text}")
                    else:
                        progress_cb(f"{prefix} — {msg}")
                except Exception:
                    try:
                        progress_cb(f"{prefix} — {msg}")
                    except Exception:
                        pass

            result = tool_run_scan(folder, scan_mode_effective, progress_cb=_wrap_progress)
            if "error" in result:
                errors.append({"folder": folder, "error": result["error"]})
            else:
                results.append({
                    "folder":        folder,
                    "name":          os.path.basename(folder),
                    "health_score":  result.get("health_score"),
                    "findings":      result.get("findings_count", 0),
                    "critical":      result.get("critical_count", 0),
                    "warnings":      result.get("warning_count", 0),
                    "frames":        result.get("total_frames", 0),
                    "scan_type":     scan_mode_effective,
                })
        except Exception as e:
            errors.append({"folder": folder, "error": str(e)})

    # Log-only folders (no band files but has logs)
    log_results = []
    if logs_only or logs_only_folders:
        for idx, folder in enumerate(logs_only_folders, start=1):
            try:
                if progress_cb:
                    try:
                        progress_cb(f"Logs-only {idx}/{len(logs_only_folders)}: {os.path.basename(folder)}")
                    except Exception:
                        pass
                result = tool_run_scan(folder, "quick", progress_cb=progress_cb)
                if "error" in result:
                    errors.append({"folder": folder, "error": result["error"]})
                    continue
                results.append({
                    "folder":        folder,
                    "name":          os.path.basename(folder),
                    "health_score":  result.get("health_score"),
                    "findings":      result.get("anomaly_count", 0),
                    "critical":      result.get("critical_count", 0),
                    "warnings":      result.get("warning_count", 0),
                    "frames":        0,
                    "scan_type":     "quick",
                    "logs_only":     True,
                })
                scan = state.get_scan_result(folder)
                log_sum = scan.log_summary if scan else analyze_logs(folder, progress_cb=progress_cb)
                log_results.append({
                    "folder":       folder,
                    "name":         os.path.basename(folder),
                    "errors":       log_sum.get("error_count", 0),
                    "warnings":     log_sum.get("warning_count", 0),
                    "frame_drops":  (log_sum.get("frame_accounting") or {}).get("frames_lost", 0),
                    "issues":       len(log_sum.get("raw_issues", [])),
                })
            except Exception as e:
                errors.append({"folder": folder, "error": str(e)})

    # Sort by health score (worst first)
    results.sort(key=lambda x: x.get("health_score") or 100)

    all_dataset_folders = list(datasets_found) + list(logs_only_folders)

    return {
        "root":             root_folder,
        "datasets_found":   len(all_dataset_folders),
        "datasets_found_list": all_dataset_folders,
        "datasets_scanned": len(results),
        "log_only_folders": len(log_results),
        "errors":           len(errors),
        "dataset_results":  results,
        "log_results":      log_results,
        "error_details":    errors[:10],
        "summary": (
            f"Found {len(all_dataset_folders)} dataset(s) under {os.path.basename(root_folder)}. "
            f"Worst health: {results[0]['name']} ({results[0]['health_score']:.0f}%)"
            if results else
            f"Found {len(log_results)} log folder(s), no band data."
        ),
    }


def tool_force_browse_folder_tree(root_folder: str,
                                  scan_mode: str = "quick",
                                  logs_only: bool = False,
                                  progress_cb: Optional[Callable[[str], None]] = None) -> Dict:
    """
    Force-refresh scan every dataset/log folder under root_folder, bypassing
    cached scan results for each discovered dataset.
    """
    if not root_folder or not os.path.isdir(root_folder):
        return {"error": f"Folder not found or not a directory: {root_folder}"}

    datasets_found = []
    logs_only_folders = []
    scan_mode_effective = "quick" if logs_only else scan_mode

    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _is_dataset_folder(dirpath):
            datasets_found.append(dirpath)
        elif any(f.lower().endswith((".log",)) for f in filenames):
            logs_only_folders.append(dirpath)

    if not datasets_found and not logs_only_folders:
        return {
            "root": root_folder,
            "datasets_found": 0,
            "message": "No datasets or log files found in this directory tree.",
        }

    results = []
    errors = []
    log_results = []

    for idx, folder in enumerate(datasets_found, start=1):
        try:
            if progress_cb:
                try:
                    progress_cb(f"Dataset {idx}/{len(datasets_found)}: {os.path.basename(folder)}")
                except Exception:
                    pass

            def _wrap_progress(msg: str):
                if not progress_cb:
                    return
                prefix = os.path.basename(folder)
                try:
                    if (msg or "").startswith("PROGRESS:"):
                        body = msg.split(":", 1)[1]
                        pct_str, text = body.split("|", 1)
                        progress_cb(f"PROGRESS:{pct_str}|{prefix} — {text}")
                    else:
                        progress_cb(f"{prefix} — {msg}")
                except Exception:
                    try:
                        progress_cb(f"{prefix} — {msg}")
                    except Exception:
                        pass

            result = tool_run_scan(folder, scan_mode_effective, _force=True, progress_cb=_wrap_progress)
            if "error" in result:
                errors.append({"folder": folder, "error": result["error"]})
            else:
                results.append({
                    "folder": folder,
                    "name": os.path.basename(folder),
                    "health_score": result.get("health_score"),
                    "findings": result.get("findings_count", 0),
                    "critical": result.get("critical_count", 0),
                    "warnings": result.get("warning_count", 0),
                    "frames": result.get("total_frames", 0),
                    "scan_type": scan_mode_effective,
                })
        except Exception as e:
            errors.append({"folder": folder, "error": str(e)})

    if logs_only or logs_only_folders:
        for idx, folder in enumerate(logs_only_folders, start=1):
            try:
                if progress_cb:
                    try:
                        progress_cb(f"Logs-only {idx}/{len(logs_only_folders)}: {os.path.basename(folder)}")
                    except Exception:
                        pass
                result = tool_run_scan(folder, "quick", _force=True, progress_cb=progress_cb)
                if "error" in result:
                    errors.append({"folder": folder, "error": result["error"]})
                    continue
                results.append({
                    "folder": folder,
                    "name": os.path.basename(folder),
                    "health_score": result.get("health_score"),
                    "findings": result.get("anomaly_count", 0),
                    "critical": result.get("critical_count", 0),
                    "warnings": result.get("warning_count", 0),
                    "frames": 0,
                    "scan_type": "quick",
                    "logs_only": True,
                })
                scan = state.get_scan_result(folder)
                log_sum = scan.log_summary if scan else analyze_logs(folder, progress_cb=progress_cb)
                log_results.append({
                    "folder": folder,
                    "name": os.path.basename(folder),
                    "errors": log_sum.get("error_count", 0),
                    "warnings": log_sum.get("warning_count", 0),
                    "frame_drops": (log_sum.get("frame_accounting") or {}).get("frames_lost", 0),
                    "issues": len(log_sum.get("raw_issues", [])),
                })
            except Exception as e:
                errors.append({"folder": folder, "error": str(e)})

    results.sort(key=lambda x: x.get("health_score") or 100)
    all_dataset_folders = list(datasets_found) + list(logs_only_folders)
    return {
        "root": root_folder,
        "datasets_found": len(all_dataset_folders),
        "datasets_found_list": all_dataset_folders,
        "datasets_scanned": len(results),
        "log_only_folders": len(log_results),
        "errors": len(errors),
        "dataset_results": results,
        "log_results": log_results,
        "error_details": errors[:10],
        "summary": (
            f"Force-scanned {len(all_dataset_folders)} dataset(s) under {os.path.basename(root_folder)}. "
            f"{len(results)} completed, {len(errors)} errors."
        ),
    }


def tool_list_open_datasets() -> Dict:
    """
    Return all currently open datasets with their names, folders, and scan status.
    Used to build the 'choose a dataset' picker when user wants to compare
    or get a report for a specific dataset.
    """
    tabs = state.all_tabs()
    if not tabs:
        return {
            "open_count": 0,
            "datasets": [],
            "message": "No datasets are currently open.",
        }

    datasets = []
    for tab in tabs:
        scan = state.get_scan_result(tab.folder)
        stale = state.get_stale_reason(tab.folder)
        datasets.append({
            "tab_index":    tab.tab_index,
            "name":         tab.dataset_name,
            "folder":       tab.folder,
            "frames":       tab.frame_count,
            "bands":        tab.band_count,
            "is_active":    tab.is_active,
            "has_scan":     scan is not None,
            "scan_stale":   bool(stale),
            "stale_reason": stale,
            "health_score": scan.health_score if scan else None,
        })

    return {
        "open_count": len(datasets),
        "datasets":   datasets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY TOOLS  — Iris's persistent cross-session memory (Supabase)
# ─────────────────────────────────────────────────────────────────────────────

def tool_save_memory(title: str, detail: str,
                     memory_type: str = "finding",
                     dataset: str = "",
                     tags: list = None) -> Dict:
    """
    Save a key finding to Iris's long-term Supabase memory.
    Memories persist across sessions — recalled semantically without
    re-uploading or re-scanning any files.

    Call automatically after any CRITICAL finding.
    Also call when user says 'remember this', 'save this finding'.

    memory_type: finding | pattern | note | resolved
    """
    folder  = state.active_folder or ""
    dataset = dataset or (os.path.basename(folder) if folder else "")
    return save_memory(title=title, detail=detail, memory_type=memory_type,
                       dataset=dataset, folder=folder, tags=tags or [])


def tool_recall_memory(query: str, top_k: int = 5) -> Dict:
    """
    Search Iris's long-term memory for past findings related to the query.
    Returns past observations from previous sessions without re-scanning.

    Call at the start of any analysis to check if this dataset was seen before.
    Call when user asks 'what did you find last time?', 'do you remember X?'.
    """
    return recall_memory(query=query, top_k=top_k)


def tool_memory_summary() -> Dict:
    """
    Show what Iris currently remembers (findings/patterns/notes counts)
    plus which datasets have cached scan results that load instantly.
    """
    summary = memory_summary()
    cached  = state.list_cached_scans()
    return {
        "memory":       summary,
        "cached_scans": {
            "count":    len(cached),
            "datasets": [
                {"name": c["name"], "health": c["health_score"],
                 "scan_type": c["scan_type"], "age_days": c["age_days"]}
                for c in cached[:10]
            ],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE TOOLS  — semantic search over indexed reference files
# ─────────────────────────────────────────────────────────────────────────────

def tool_knowledge_search(query: str, top_k: int = 4) -> Dict:
    """
    Semantic search over Iris's knowledge base — sensor specs, SOPs,
    camera manuals, calibration guides indexed from your reference files.

    Call when:
      - User asks about sensor limits, parameter ranges, or specs
      - User asks 'what does the manual say about X'
      - Before making claims about hardware specifications — search first
      - A scan finding needs context from specs (e.g. 'is this FPS cap expected?')
    """
    if not query or not query.strip():
        return {"error": "Query cannot be empty."}
    return search_knowledge(query=query.strip(), top_k=max(1, min(top_k, 8)))


def tool_index_knowledge(force: bool = False) -> Dict:
    """
    Index (or re-index) all files in the Iris knowledge folder.
    Triggered by ?index command. Reads PDF, Word, Excel, CSV, Markdown, text.
    force=False: skip unchanged files. force=True: re-index everything.
    """
    def _log(msg): print(f"[Iris /index] {msg}")
    folder = _KNOWLEDGE_FOLDER
    if not os.path.isdir(folder):
        return {"error": (
            f"Knowledge folder not found: {folder}\n"
            "Create it and add your reference files, or set IRIS_KNOWLEDGE_FOLDER."
        )}
    return run_indexing(knowledge_folder=folder, progress_cb=_log, force=force)


def tool_knowledge_status() -> Dict:
    """
    Return knowledge-base indexing status:
    indexed files, chunk counts, and readiness for semantic search.
    """
    return knowledge_base_status()


def tool_forget_knowledge(file_name: str) -> Dict:
    """
    Remove a specific file from the knowledge index.
    Use when user says '?forget filename' or 'remove X from knowledge base'.
    The file_name should match the original filename (e.g. 'VisLinxM_Spec.pdf').
    """
    if not file_name or not file_name.strip():
        return {"error": "file_name cannot be empty."}
    return delete_from_index(knowledge_folder=_KNOWLEDGE_FOLDER, file_name=file_name.strip())


# ─────────────────────────────────────────────────────────────────────────────
# TOOL SCHEMAS  (sent to Claude API)
# ─────────────────────────────────────────────────────────────────────────────

def tool_parse_companion_files(folder: str = "") -> Dict:
    """Parse .meta and *_ephemeris.txt for the dataset. Enriches scan result."""
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No folder resolved."}
    meta_sum  = parse_meta_file(folder)
    ephem_sum = parse_ephemeris_file(folder)
    scan = state.get_scan_result(folder)
    log_sum = {}
    if scan:
        log_sum            = scan.log_summary or {}
        scan.meta_summary  = meta_sum
        scan.ephem_summary = ephem_sum
        state.store_scan_result(scan)
    fps = None
    if log_sum:
        fps = (log_sum.get("parameters_applied", {}).get("FPS") or
               log_sum.get("procmode", {}).get("decoded", {}).get("fps_requested"))
    merged = merge_summaries(log_sum, meta_sum, ephem_sum, fps)
    return {
        "folder":            folder,
        "meta":              meta_sum,
        "ephemeris":         ephem_sum,
        "cross_validation":  merged.get("cross_validation_findings", []),
        "gsd_along_track_m": merged.get("gsd_along_track_m"),
    }


def tool_detect_mission_type(folder: str = "") -> Dict:
    """Detect mission type from log alone — no band files needed."""
    folder = _resolve_folder(folder)
    if not folder:
        return {"error": "No folder resolved."}
    log_text = ""
    for fn in (os.listdir(folder) if os.path.isdir(folder) else []):
        if fn.lower().endswith(".log"):
            try:
                with open(os.path.join(folder, fn), errors="ignore") as f:
                    log_text += f.read()
            except Exception:
                pass
    scan     = state.get_scan_result(folder)
    log_sum  = scan.log_summary  if scan else {}
    meta_sum = scan.meta_summary if scan else {}
    return detect_mission_type(log_text, log_sum, meta_sum)


def tool_template_status() -> Dict:
    """Show all learned mission baselines — means, stds, sample counts per field."""
    return _learner.status()


def tool_reset_template(mission_type: str = "") -> Dict:
    """Reset one mission template (or all if empty). Use after hardware changes."""
    return _learner.reset(mission_type.strip() or None)


def tool_bulk_scan_files(root_folder: str, file_types: str = "log,meta,ephemeris") -> Dict:
    """
    Recursively find and parse all .log / .meta / ephemeris files under root_folder.
    Works without band files. Returns per-session summaries + cross-session analytics.
    """
    from .meta_parser import bulk_scan_files as _bulk
    if not root_folder or not os.path.isdir(root_folder):
        return {"error": f"Folder not found: {root_folder}"}
    ft = set()
    for t in file_types.lower().split(","):
        t = t.strip()
        if t in ("log","logs"):       ft.add(".log")
        if t == "meta":               ft.add(".meta")
        if t in ("ephemeris","ephem"):ft.add("_ephemeris.txt")
    if not ft:
        ft = {".log", ".meta", "_ephemeris.txt"}
    result = _bulk(root_folder, list(ft))
    return {
        "root":         result.get("root"),
        "sessions":     result.get("sessions", 0),
        "analytics":    result.get("analytics", {}),
        "errors":       result.get("error_count", 0),
        "session_list": [
            {k: s[k] for k in
             ("name","mission_type","sensor_temp","fps","drops",
              "n_critical","n_warnings","stale_trigger","has_meta","has_ephem")
             if k in s}
            for s in result.get("sessions_data", [])
        ],
    }


def tool_cross_session_analytics(root_folder: str = "",
                                   field: str = "",
                                   mission_type_filter: str = "") -> Dict:
    """
    Compute analytics (mean, std, min, max, outliers) across all sessions.
    field: e.g. sensor_temp, fps, exposure, gain, drops
    Handles: "compare all temperatures", "average FPS", "find temperature outliers"
    """
    from .meta_parser import bulk_scan_files as _bulk, compute_cross_session_analytics
    root_folder = root_folder or state.active_folder
    if not root_folder or not os.path.isdir(root_folder):
        return {"error": "Provide a valid root folder."}
    result   = _bulk(root_folder, [".log", ".meta"])
    sessions = result.get("sessions_data", [])
    if mission_type_filter:
        sessions = [s for s in sessions
                    if s.get("mission_type","").lower() == mission_type_filter.lower()]
    if not sessions:
        return {"error": "No sessions found.", "root": root_folder}
    analytics = compute_cross_session_analytics(sessions)
    if field:
        fmap = {
            "sensor_temp": "sensor_temperature", "temperature": "sensor_temperature",
            "temp": "sensor_temperature", "core_temp": "core_temperature",
            "fps": "fps", "exposure": "exposure", "gain": "gain",
            "frames": "frames_captured", "drops": "frame_drops",
        }
        key  = fmap.get(field.lower().replace(" ", "_"), field)
        data = analytics.get(key, analytics.get(field, {}))
        if isinstance(data, dict) and "mean" in data:
            fd = data
            return {
                "field":   field,
                "n":       fd["n"],
                "mean":    fd["mean"],
                "std":     fd["std"],
                "min":     fd["min"],
                "max":     fd["max"],
                "median":  fd["median"],
                "outliers": analytics.get("sensor_temperature", {}).get("outliers", []),
                "sessions": len(sessions),
                "root":    root_folder,
            }
    return {"root": root_folder, "sessions": len(sessions), "analytics": analytics}



TOOL_SCHEMAS = [
    {
        "name": "get_app_state",
        "description": (
            "Get a complete snapshot of the current application state: "
            "which datasets are open, which tab is active, current frame, "
            "and whether scan results exist. "
            "ALWAYS call this first if you're unsure what's currently open."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_folder",
        "description": (
            "List and categorize all files in a folder. "
            "Shows band files, log files, metadata JSON, and computed geometry. "
            "Use before scanning to understand what's present."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Full path to folder"}
            },
            "required": ["folder"],
        },
    },
    {
        "name": "read_logs",
        "description": (
            "Read and parse all log files in a folder (including subfolders). "
            "Extracts: FPS, TotalNoOfFrames, CapturedCount, frame drops, "
            "SensorTemp, DeviceTemp, errors, warnings, timing stability. "
            "Use when asked about capture conditions, errors, or performance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Folder containing log files"}
            },
            "required": ["folder"],
        },
    },
    {
        "name": "run_scan",
        "description": (
            "Scan a dataset for anomalies. "
            "Default behavior is quick metadata/log analysis only. "
            "Frame/pixel scanning must only be used when the user explicitly asks for frame-level, pixel-level, deep, or full analysis. "
            "Detects: frame count mismatches, log errors, parameter mismatches, and in deep modes also black frames, dead/hot pixels, dead columns, vertical striping, saturation, cross-band outliers. "
            "Returns health score (0-100) and anomaly frame numbers. "
            "mode 'quick' (default): metadata + logs only, ~3 seconds. "
            "mode 'full': every frame, 1-5 min, explicit deep scan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder path"},
                "mode":   {"type": "string", "enum": ["full", "quick"],
                           "description": "Scan depth — default is 'quick'"},
            },
            "required": ["folder"],
        },
    },
    {
        "name": "force_scan",
        "description": (
            "Force a fresh scan ignoring all caches. "
            "Still use quick by default; only use full if the user explicitly asks for frame/pixel/deep analysis. "
            "Use when user says 'force scan' or 'fscan'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder path"},
                "mode":   {"type": "string", "enum": ["full", "quick"],
                           "description": "Scan depth — default is 'quick'"},
            },
            "required": ["folder"],
        },
    },
    {
        "name": "get_scan_results",
        "description": (
            "Retrieve existing scan results for a folder without running a new scan. "
            "Returns health score, anomaly frame list, and detailed findings. "
            "Use this before run_scan to avoid redundant scanning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder path (optional, defaults to active)"}
            },
        },
    },
    {
        "name": "get_histogram_state",
        "description": (
            "Return the current histogram viewer state — dataset, frame, display mode, "
            "pixel value axis range, per-frame min/max, visible bands, and per-band stats. "
            "Use when the user asks about: the histogram, pixel values, exposure, DN levels, "
            "dynamic range, clipping, saturation, or any visual question about the "
            "currently displayed histogram."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_anomaly_frames",
        "description": (
            "Get the list of frames that contain anomalies, optionally filtered by type. "
            "Types include: black_frame, dead_columns, vertical_striping, "
            "saturation, dead_pixels, cross_band_outlier. "
            "Use when user asks 'which frames have issues' or 'show me the dead pixel frames'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder":       {"type": "string"},
                "anomaly_type": {"type": "string", "description": "Optional filter by anomaly type"},
            },
        },
    },
    {
        "name": "get_frame_info",
        "description": (
            "Get all known information about a specific frame: "
            "scan findings at that frame, whether it's an anomaly, what to look for. "
            "Use when user navigates to a frame and asks 'what's wrong here?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder":      {"type": "string"},
                "frame_index": {"type": "integer"},
            },
            "required": ["frame_index"],
        },
    },
    {
        "name": "navigate_to_frame",
        "description": (
            "Move the application display to a specific frame number. "
            "The frame slider will move and the viewer will update immediately. "
            "Use when user says 'go to frame X', 'show me frame X', "
            "or after identifying an anomaly at a specific frame."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_index": {"type": "integer", "description": "Frame number (0-based)"},
                "folder":      {"type": "string",  "description": "Optional: which dataset tab to navigate"},
            },
            "required": ["frame_index"],
        },
    },
    {
        "name": "open_dataset",
        "description": (
            "Open a dataset folder in a new application tab. "
            "Use when user says 'open', 'load', or 'show me' a dataset path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_path": {"type": "string", "description": "Full path to dataset folder"},
            },
            "required": ["folder_path"],
        },
    },
    {
        "name": "open_last_session",
        "description": (
            "Open the most recent dataset from memory/session history. "
            "Use when user asks to resume/open/load the last or previous session."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_zoom",
        "description": (
            "Set the zoom level on the active frame viewer. "
            "1.0 = fit to window, 2.0 = 2× zoom, 4.0 = 4× zoom. "
            "Use when user says 'zoom in', 'zoom out', 'fit to window'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "number", "description": "Zoom multiplier (1.0 = fit)"},
            },
            "required": ["level"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Generate a structured text report combining scan results and log analysis. "
            "Use when user asks for a report, summary, or diagnosis of a dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder (optional, defaults to active)"},
            },
        },
    },
    {
        "name": "llm_log_audit",
        "description": (
            "Ask the local Ollama model to audit raw .log files for suspicious gaps, "
            "resets, or anomalies that rules might miss. Returns a short bullet list."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder (optional, defaults to active)"},
                "max_chars": {"type": "integer", "description": "Max chars of log text to review"},
            },
        },
    },
    {
        "name": "clear_cache",
        "description": (
            "Clear cached scan results and/or cached reports. "
            "Use when user asks to delete cache, clear past reports, or reset cached scans."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "all (default), scans, or reports"},
            },
        },
    },
    {
        "name": "close_tab",
        "description": (
            "Close a specific tab by its tab index. "
            "Use when user says 'close this tab', 'close tab N', 'close the current tab'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tab_index": {"type": "integer", "description": "Tab index to close"},
            },
            "required": ["tab_index"],
        },
    },
    {
        "name": "control_app",
        "description": (
            "Control the application UI directly. "
            "Actions supported: switch_tab, add_tab, set_theme, toggle_theme, "
            "play_pause, play, pause, next_frame, prev_frame, refresh, reload, "
            "save_progress, export_image, fit_to_screen, actual_size, auto_contrast, "
            "open_terminal, close_terminal, toggle_terminal, close_tab, "
            "open_view, close_view, set_band_gap, set_histogram_bands, "
            "open_magnifier, close_magnifier, set_magnifier_center, "
            "set_magnifier_zoom, set_contrast. "
            "Use when the user wants Iris to operate the app instead of only analyzing data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Control action name"},
                "tab_index": {"type": "integer", "description": "Optional target tab index"},
                "mode": {"type": "string", "description": "Tab mode for add_tab: band/raw/video/live/tiled"},
                "value": {"type": "string", "description": "Optional value, e.g. dark/light for set_theme"},
                "folder": {"type": "string", "description": "Optional dataset folder to target a tab"},
                "dataset_name": {"type": "string", "description": "Optional dataset/tab name to target"},
                "x": {"type": "number", "description": "Optional x coordinate for actions like set_magnifier_center"},
                "y": {"type": "number", "description": "Optional y coordinate for actions like set_magnifier_center"},
                "min_value": {"type": "number", "description": "Optional minimum value for actions like set_contrast"},
                "max_value": {"type": "number", "description": "Optional maximum value for actions like set_contrast"},
                "enabled": {"type": "boolean", "description": "Optional enable flag for actions like set_contrast"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "compare_datasets",
        "description": (
            "Compare two datasets side by side. "
            "Shows health score diff, anomaly frames unique to each, shared anomaly frames, "
            "band mean differences, and finding type breakdown. "
            "If no folders specified, automatically compares the two currently open tabs. "
            "Both datasets must be scanned first — call run_scan on each if needed. "
            "Use when user says 'compare', 'how do these differ', "
            "'is this worse than the other one', 'same issue as before', "
            "'compare the two open datasets'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder_a": {"type": "string", "description": "First dataset folder (optional — uses open tabs if blank)"},
                "folder_b": {"type": "string", "description": "Second dataset folder (optional)"},
            },
        },
    },
    {
        "name": "detect_repeating_pattern",
        "description": (
            "Detect repeating tile patterns and anomalies in a specific band/frame. "
            "Auto-detects tile size and flags tiles that deviate from the dominant repeat."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder (optional, defaults to active)"},
                "band": {"type": "string", "description": "Band key like b0, b1 (optional)"},
                "frame_index": {"type": "integer", "description": "Frame index (optional, defaults to 0)"},
            },
        },
    },
    {
        "name": "refresh_scan",
        "description": (
            "Invalidate the cached scan for a dataset and run a fresh scan. "
            "Use when: user says 'scan again', 'results are wrong', 'I fixed the resolution', "
            "'data changed', 'reload', 're-scan', or any time the user signals the "
            "previous results are stale. Always clears old results first. "
            "Also use automatically when context shows SCAN CACHE INVALIDATED."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "Dataset folder (default: active tab)"},
                "mode":   {"type": "string", "enum": ["full", "quick"], "description": "Scan depth"},
            },
        },
    },
    {
        "name": "browse_folder_tree",
        "description": (
            "Walk an entire directory tree to find and scan all datasets inside. "
            "Use when user points to a parent folder that contains multiple datasets, "
            "or says 'scan everything in this folder', 'check all datasets here'. "
            "Set logs_only=true when user says 'just check logs' or 'logs only'. "
            "Returns a ranked list of all datasets found with health scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_folder": {"type": "string", "description": "Root directory to walk"},
                "scan_mode":   {"type": "string", "enum": ["full", "quick"], "description": "Scan depth for each dataset"},
                "logs_only":   {"type": "boolean", "description": "If true, only parse logs — skip pixel scan"},
            },
            "required": ["root_folder"],
        },
    },
    {
        "name": "session_report",
        "description": (
            "Scan all .log files under a root folder and produce a cross-session "
            "report covering ALL acquisitions in one pass. "
            "Reports: argument integrity (all 14 args), applied vs requested for every "
            "parameter (FPS, exposure, gain, xshift, TDIYShift), exposure clamp detection, "
            "FPS cap detection, frame drops, temperatures (start/end/delta per session), "
            "trigger timestamp staleness, firmware groups, binning verification gap, "
            "and a summary pass/fail table for every session. "
            "Use when user points to a folder of .log files and says 'give me a report', "
            "'analyze all logs', 'what happened in this pass', 'check all acquisitions'. "
            "Does NOT require band files — works on logs alone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_folder": {
                    "type": "string",
                    "description": "Folder containing .log files (searched recursively). "
                                   "Defaults to active folder if blank."
                },
            },
        },
    },
    {
        "name": "list_open_datasets",
        "description": (
            "List all currently open datasets with their names, folders, frame counts, "
            "and whether they have been scanned. "
            "Use when you need to show the user a picker — e.g. 'which dataset do you want to compare?' "
            "or 'which dataset should I report on?'. "
            "Present the returned list as options for the user to choose from."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_memory",
        "description": (
            "Save a key finding to Iris's long-term Supabase memory. "
            "Memories persist across sessions — Iris can recall them next session "
            "without re-uploading or re-scanning any files. "
            "Call automatically after any CRITICAL finding. "
            "Also call when user says 'remember this', 'save this finding'. "
            "memory_type: 'finding' (confirmed issue), 'pattern' (cross-session trend), "
            "'note' (general note), 'resolved' (fixed issue)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title":       {"type": "string", "description": "Short label e.g. 'Acq073 — corrupt parameter file'"},
                "detail":      {"type": "string", "description": "Full explanation with context"},
                "memory_type": {"type": "string", "description": "finding | pattern | note | resolved"},
                "dataset":     {"type": "string", "description": "Dataset name (auto-detected from active tab if blank)"},
                "tags":        {"type": "array", "items": {"type": "string"}, "description": "Searchable tags"},
            },
            "required": ["title", "detail"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Search Iris's long-term memory for past findings related to the query. "
            "Returns past observations from previous sessions without re-scanning. "
            "Call at the start of analysis to check if this dataset was seen before. "
            "Call when user asks 'what did you find last time?', 'do you remember X?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in past memories"},
                "top_k": {"type": "integer", "description": "Number of memories to return (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_summary",
        "description": (
            "Show what Iris currently remembers (findings/patterns/notes) "
            "plus which datasets have cached scan results that load instantly. "
            "Use when user asks 'what do you remember?', 'what scans do you have cached?'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "knowledge_search",
        "description": (
            "Semantic search over Iris's knowledge base — your indexed reference files "
            "(sensor specs, camera manuals, SOPs, calibration guides). "
            "Call before making any claim about hardware limits, parameter ranges, or procedures. "
            "Call when user asks 'what does the manual say about X', "
            "'what's the safe gain range', 'is this FPS cap expected?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Specific question or phrase to search for"},
                "top_k": {"type": "integer", "description": "Number of results (default 4, max 8)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "index_knowledge",
        "description": (
            "Index or re-index the Iris knowledge folder. "
            "Triggered by ?index command. Supports PDF, Word, Excel, CSV, Markdown, text. "
            "force=false: skip unchanged files. force=true: re-index everything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "description": "Re-index all files even if unchanged"},
            },
        },
    },
    {
        "name": "knowledge_status",
        "description": (
            "Show what files are indexed in the knowledge base: "
            "file names, types, chunk counts, last indexed date. "
            "Use when user asks 'what files are indexed?', 'is the manual loaded?'."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "forget_knowledge",
        "description": (
            "Remove a specific file from the knowledge index. "
            "Use when user says '?forget filename' or 'remove X from the knowledge base'. "
            "Auto-delete also runs on every ?index for missing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string",
                              "description": "Exact filename to remove e.g. 'old_manual_v1.pdf'"},
            },
            "required": ["file_name"],
        },
    },
        {
        "name": "parse_companion_files",
        "description": (
            "Parse the .meta and *_ephemeris.txt files for the active dataset. "
            "Extracts satellite identity, ground track, orbital altitude, GSD, "
            "and cross-validates against log. Call after scan when .meta or "
            "ephemeris files are present."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"folder": {"type": "string"}},
        },
    },
    {
        "name": "detect_mission_type",
        "description": (
            "Detect mission type from the log file alone — no band files needed. "
            "Returns mission_type (e.g. tdi_8_standard, test_pattern_calibration), "
            "confidence, and evidence. Use before generating a report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"folder": {"type": "string"}},
        },
    },
    {
        "name": "template_status",
        "description": (
            "Show all learned mission baselines: mean, std, min, max, sample count "
            "per field per mission type. Shows what Iris considers 'normal'. "
            "Use when asked 'what does normal look like?' or 'how many sessions learned?'"
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "reset_template",
        "description": (
            "Reset the learned baseline for a mission type, or all if empty. "
            "Use after hardware changes or when baselines are stale."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"mission_type": {"type": "string"}},
        },
    },
    {
        "name": "bulk_scan_files",
        "description": (
            "Recursively scan ALL .log, .meta, and/or ephemeris files under a folder. "
            "Works without band files. Returns per-session summaries and cross-session analytics. "
            "Use when asked: 'scan all logs in /data/', 'scan all .meta files', "
            "'find all sessions', 'scan all subdirectories'. "
            "file_types: comma-separated 'log,meta,ephemeris' (default: all)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_folder": {"type": "string", "description": "Root folder to walk recursively"},
                "file_types":  {"type": "string", "description": "log,meta,ephemeris (default: all)"},
            },
            "required": ["root_folder"],
        },
    },
    {
        "name": "cross_session_analytics",
        "description": (
            "Analytics across all sessions under a root folder. "
            "Handles: 'compare all temperatures', 'average FPS', "
            "'find sessions with low sensor temperature', 'temperature trend', "
            "'highest core temperature'. "
            "field: sensor_temp | fps | exposure | gain | drops. "
            "mission_type_filter: restrict to one type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "root_folder":          {"type": "string"},
                "field":                {"type": "string"},
                "mission_type_filter":  {"type": "string"},
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table — used by agent.py to route tool calls
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DISPATCH = {
    "get_app_state":       lambda inp: tool_get_app_state(),
    "list_folder":         lambda inp: tool_list_folder(inp.get("folder", "")),
    "read_logs":           lambda inp: tool_read_logs(inp.get("folder", "")),
    "run_scan":            lambda inp: tool_run_scan(inp.get("folder", ""), inp.get("mode", "quick")),
    "force_scan":          lambda inp: tool_force_scan(inp.get("folder", ""), inp.get("mode", "quick")),
    "refresh_scan":        lambda inp: tool_refresh_scan(inp.get("folder", ""), inp.get("mode", "quick")),
    "get_scan_results":    lambda inp: tool_get_scan_results(inp.get("folder", "")),
    "get_histogram_state": lambda inp: tool_get_histogram_state(),
    "find_anomaly_frames": lambda inp: tool_find_anomaly_frames(inp.get("folder",""), inp.get("anomaly_type")),
    "get_frame_info":      lambda inp: tool_get_frame_info(inp.get("folder",""), inp.get("frame_index", 0)),
    "navigate_to_frame":   lambda inp: tool_navigate_to_frame(inp.get("frame_index", 0), inp.get("folder")),
    "open_dataset":        lambda inp: tool_open_dataset(inp.get("folder_path", "")),
    "open_last_session":   lambda inp: tool_open_last_session(),
    "set_zoom":            lambda inp: tool_set_zoom(inp.get("level", 1.0)),
    "generate_report":     lambda inp: tool_generate_report(inp.get("folder", "")),
    "llm_log_audit":       lambda inp: tool_llm_log_audit(inp.get("folder", ""), inp.get("max_chars", 120000)),
    "clear_cache":         lambda inp: tool_clear_cache(inp.get("scope", "all")),
    "close_tab":           lambda inp: tool_close_tab(inp.get("tab_index", 0)),
    "control_app":         lambda inp: tool_control_app(
                               inp.get("action", ""),
                               inp.get("tab_index", -1),
                               inp.get("mode", ""),
                               inp.get("value", ""),
                               inp.get("folder", ""),
                               inp.get("dataset_name", ""),
                               inp.get("x"),
                               inp.get("y"),
                               inp.get("min_value"),
                               inp.get("max_value"),
                               inp.get("enabled"),
                           ),
    "compare_datasets":    lambda inp: tool_compare_datasets(inp.get("folder_a",""), inp.get("folder_b","")),
    "detect_repeating_pattern": lambda inp: tool_detect_repeating_pattern(
                               inp.get("folder", ""), inp.get("band", "b0"), inp.get("frame_index", 0)),
    "browse_folder_tree":  lambda inp: tool_browse_folder_tree(
                               inp.get("root_folder",""),
                               inp.get("scan_mode","quick"),
                               inp.get("logs_only", False)),
    "extract_all_parameters": lambda inp: tool_extract_all_log_parameters(inp.get("root_folder", "")),
    "extract_sensor_data": lambda inp: tool_extract_sensor_data(inp.get("root_folder", "")),
    "session_report":      lambda inp: tool_session_report(inp.get("root_folder", "")),
    "list_open_datasets":  lambda inp: tool_list_open_datasets(),
    # Memory
    "save_memory":         lambda inp: tool_save_memory(
                               inp.get("title",""), inp.get("detail",""),
                               inp.get("memory_type","finding"),
                               inp.get("dataset",""), inp.get("tags",[])),
    "recall_memory":       lambda inp: tool_recall_memory(inp.get("query",""), inp.get("top_k",5)),
    "memory_summary":      lambda inp: tool_memory_summary(),
    # Knowledge base
    "knowledge_search":    lambda inp: tool_knowledge_search(inp.get("query",""), inp.get("top_k",4)),
    "index_knowledge":     lambda inp: tool_index_knowledge(inp.get("force",False)),
    "knowledge_status":    lambda inp: tool_knowledge_status(),
    "forget_knowledge":    lambda inp: tool_forget_knowledge(inp.get("file_name","")),
    "parse_companion_files":   lambda inp: tool_parse_companion_files(inp.get("folder","")),
    "detect_mission_type":     lambda inp: tool_detect_mission_type(inp.get("folder","")),
    "template_status":         lambda inp: tool_template_status(),
    "reset_template":          lambda inp: tool_reset_template(inp.get("mission_type","")),
    "bulk_scan_files":         lambda inp: tool_bulk_scan_files(
                                   inp.get("root_folder",""), inp.get("file_types","log,meta,ephemeris")),
    "cross_session_analytics": lambda inp: tool_cross_session_analytics(
                                   inp.get("root_folder",""), inp.get("field",""),
                                   inp.get("mission_type_filter","")),
}

_REPORT_TOOL_NAMES = {
    "get_app_state",
    "list_folder",
    "read_logs",
    "run_scan",
    "get_scan_results",
    "find_anomaly_frames",
    "get_frame_info",
    "open_dataset",
    "generate_report",
    "llm_log_audit",
    "knowledge_search",
}

TOOL_SCHEMAS = [schema for schema in TOOL_SCHEMAS if schema.get("name") in _REPORT_TOOL_NAMES]
TOOL_DISPATCH = {name: fn for name, fn in TOOL_DISPATCH.items() if name in _REPORT_TOOL_NAMES}
