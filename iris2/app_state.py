"""
iris/app_state.py

Single source of truth for all application state.
Merged with scan_cache.py — SQLite persistence is handled here directly.
scan_cache.py is no longer needed and can be deleted.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import time
import threading
import json
import gzip
import sqlite3
import hashlib
from pathlib import Path
from app_paths import get_app_data_path, migrate_legacy_file

from .event_bus import bus, AppEvent, EventType


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TabState:
    tab_index:    int
    mode:         str                         # "band", "raw", "video", "live", "tiled"
    folder:       str = ""
    dataset_name: str = ""
    frame_count:  int = 0
    current_frame:int = 0
    band_count:   int = 0
    meta:         Dict = field(default_factory=dict)
    widget_ref:   Any = None
    loaded_at:    float = field(default_factory=time.time)
    is_active:    bool = False


@dataclass
class ScanResult:
    folder:         str
    scan_type:      str                       # "quick", "sample", "full"
    health_score:   float = 100.0
    anomaly_frames: List[int] = field(default_factory=list)
    findings:       List[Dict] = field(default_factory=list)
    band_summary:   Dict = field(default_factory=dict)
    log_summary:    Dict = field(default_factory=dict)
    # Extended: from .meta and ephemeris (populated by meta_parser after scan)
    meta_summary:   Dict = field(default_factory=dict)
    ephem_summary:  Dict = field(default_factory=dict)
    # Cross-source merged summary (GSD, cross-validation findings, etc.)
    merged_summary: Dict = field(default_factory=dict)
    scanned_at:     float = field(default_factory=time.time)
    duration_sec:   float = 0.0


@dataclass
class HistogramState:
    folder:        str   = ""
    dataset_name:  str   = ""
    frame_index:   int   = 0
    display_mode:  str   = "single_frame"
    range_start:   int   = 0
    range_end:     int   = 0
    axis_min:      float = 0.0
    axis_max:      float = 1023.0
    frame_min:     float = 0.0
    frame_max:     float = 0.0
    visible_bands: List[int] = field(default_factory=list)
    band_stats:    Dict[int, Dict] = field(default_factory=dict)
    updated_at:    float = field(default_factory=time.time)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN CACHE  (formerly scan_cache.py — merged here)
# ══════════════════════════════════════════════════════════════════════════════

_MAX_CACHE_AGE_DAYS = 90
_DB_FILENAME        = "iris_scans.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS scan_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    folder       TEXT    NOT NULL,
    fingerprint  TEXT    NOT NULL,
    scan_type    TEXT    NOT NULL,
    health_score REAL    NOT NULL DEFAULT 100.0,
    result_json  TEXT    NOT NULL,
    scanned_at   REAL    NOT NULL,
    created_at   REAL    NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS scan_cache_folder_fp
    ON scan_cache (folder, fingerprint);
CREATE INDEX IF NOT EXISTS scan_cache_folder
    ON scan_cache (folder);
"""


class _ScanCache:
    """Thread-safe SQLite cache for ScanResult objects. Used internally by AppState."""

    def __init__(self):
        candidate = Path(migrate_legacy_file(
            get_app_data_path(_DB_FILENAME),
            os.path.join(os.path.dirname(__file__), _DB_FILENAME)
        ))
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.touch(exist_ok=True)
            self._path = str(candidate)
        except OSError:
            fb = Path.home() / ".iris" / _DB_FILENAME
            fb.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(fb)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            # Check if we need to migrate from old unixepoch() schema
            try:
                cursor = conn.execute("PRAGMA table_info(scan_cache)")
                columns = {row[1]: row for row in cursor.fetchall()}
                if 'created_at' in columns:
                    # Check if the default value contains unixepoch
                    default_val = columns['created_at'][4]  # column default is at index 4
                    if default_val and 'unixepoch' in default_val.lower():
                        print("[ScanCache] Migrating database schema...")
                        conn.execute("DROP TABLE IF EXISTS scan_cache")
            except sqlite3.Error:
                pass  # Table might not exist yet
            
            # Create/recreate table with new schema
            conn.executescript(_CREATE_SQL)

    @staticmethod
    def fingerprint(folder: str) -> str:
        fp = Path(folder)
        if not fp.is_dir():
            return ""
        parts = []
        for f in sorted(fp.iterdir()):
            if f.suffix.lower() in {".band0",".band1",".band2",".band3",
                                     ".band4",".band5",".band6",".log",".meta"}:
                try:
                    s = f.stat()
                    parts.append(f"{f.name}:{s.st_size}:{s.st_mtime:.0f}")
                except OSError:
                    pass
        if not parts:
            try:
                parts.append(f"folder:{fp.stat().st_mtime:.0f}")
            except OSError:
                return ""
        return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]

    def store(self, result: ScanResult) -> bool:
        folder = result.folder
        fp     = self.fingerprint(folder)
        if not fp:
            return False
        try:
            rj = json.dumps({
                "folder":        result.folder,
                "scan_type":     result.scan_type,
                "health_score":  result.health_score,
                "anomaly_frames":result.anomaly_frames,
                "findings":      result.findings,
                "band_summary":  result.band_summary,
                "log_summary":   result.log_summary,
                "meta_summary":  result.meta_summary,
                "ephem_summary": result.ephem_summary,
                "merged_summary":result.merged_summary,
                "scanned_at":    result.scanned_at,
                "duration_sec":  result.duration_sec,
            })
        except (TypeError, ValueError) as e:
            print(f"[ScanCache] Serialisation error: {e}")
            return False
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("""
                        INSERT INTO scan_cache
                            (folder,fingerprint,scan_type,health_score,result_json,scanned_at)
                        VALUES (?,?,?,?,?,?)
                        ON CONFLICT(folder,fingerprint) DO UPDATE SET
                            scan_type=excluded.scan_type,
                            health_score=excluded.health_score,
                            result_json=excluded.result_json,
                            scanned_at=excluded.scanned_at
                    """, (folder,fp,result.scan_type,result.health_score,rj,result.scanned_at))
                return True
            except sqlite3.Error as e:
                print(f"[ScanCache] Write error: {e}")
                return False

    def load(self, folder: str) -> Optional[ScanResult]:
        fp = self.fingerprint(folder)
        if not fp:
            return None
        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute("""
                        SELECT result_json, scanned_at FROM scan_cache
                        WHERE folder=? AND fingerprint=?
                        ORDER BY scanned_at DESC LIMIT 1
                    """, (folder, fp)).fetchone()
            except sqlite3.Error:
                return None
        if not row:
            return None
        if (time.time() - row["scanned_at"]) / 86400 > _MAX_CACHE_AGE_DAYS:
            return None
        try:
            d = json.loads(row["result_json"])
        except Exception:
            return None
        return ScanResult(
            folder        = d["folder"],
            scan_type     = d["scan_type"],
            health_score  = d.get("health_score", 100.0),
            anomaly_frames= d.get("anomaly_frames", []),
            findings      = d.get("findings", []),
            band_summary  = d.get("band_summary", {}),
            log_summary   = d.get("log_summary", {}),
            meta_summary  = d.get("meta_summary", {}),
            ephem_summary = d.get("ephem_summary", {}),
            merged_summary= d.get("merged_summary", {}),
            scanned_at    = d.get("scanned_at", 0),
            duration_sec  = d.get("duration_sec", 0),
        )

    def is_cached(self, folder: str) -> bool:
        return self.load(folder) is not None

    def list_cached(self) -> list:
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute("""
                        SELECT folder,scan_type,health_score,scanned_at
                        FROM scan_cache ORDER BY scanned_at DESC
                    """).fetchall()
                return [{
                    "folder":       r["folder"],
                    "name":         os.path.basename(r["folder"]),
                    "scan_type":    r["scan_type"],
                    "health_score": r["health_score"],
                    "scanned_at":   r["scanned_at"],
                    "age_days":     round((time.time()-r["scanned_at"])/86400, 1),
                } for r in rows]
            except sqlite3.Error:
                return []

    def invalidate(self, folder: str):
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM scan_cache WHERE folder=?", (folder,))
            except sqlite3.Error as e:
                print(f"[ScanCache] Invalidate error: {e}")

    def purge_old(self):
        cutoff = time.time() - _MAX_CACHE_AGE_DAYS * 86400
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM scan_cache WHERE scanned_at<?", (cutoff,))
            except sqlite3.Error:
                pass

    def stats(self) -> Dict:
        with self._lock:
            try:
                with self._connect() as conn:
                    row = conn.execute("""
                        SELECT COUNT(*) as total_entries,
                               COUNT(DISTINCT folder) as unique_folders,
                               AVG(health_score) as avg_health,
                               MIN(scanned_at) as oldest,
                               MAX(scanned_at) as newest,
                               SUM(LENGTH(result_json))/1024 as total_kb
                        FROM scan_cache
                    """).fetchone()
                    return dict(row) if row else {}
            except sqlite3.Error:
                return {}


# ══════════════════════════════════════════════════════════════════════════════
# APP STATE
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    """
    Single source of truth for Iris.
    Manages tabs, scan results, report cache, and histogram state.
    scan_cache is embedded directly — no separate import needed.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._tabs: Dict[int, TabState] = {}
        self._active_tab_index: int = 0
        self._scan_results: Dict[str, ScanResult] = {}
        self._report_cache: Dict[str, Dict] = {}
        self._report_cache_file = migrate_legacy_file(
            get_app_data_path(".iris_reports.json.gz"),
            os.path.join(os.path.dirname(__file__), ".iris_reports.json.gz")
        )
        self._report_cache_max = 200
        self._stale_reason: Dict[str, str] = {}
        self._histogram: Optional[HistogramState] = None
        self._session_folders: List[str] = []
        self._max_session = 50

        # Embedded scan cache (replaces scan_cache.py)
        self._scan_cache = _ScanCache()
        self._scan_cache.purge_old()

        bus.subscribe(EventType.DATASET_LOADED,    self._on_dataset_loaded)
        bus.subscribe(EventType.DATASET_CLOSED,    self._on_dataset_closed)
        bus.subscribe(EventType.FRAME_CHANGED,     self._on_frame_changed)
        bus.subscribe(EventType.TAB_ACTIVATED,     self._on_tab_activated)
        bus.subscribe(EventType.TAB_CLOSED,        self._on_tab_closed)
        bus.subscribe(EventType.HISTOGRAM_UPDATED, self._on_histogram_updated)

        self._load_report_cache()

    # ── Report cache persistence ──────────────────────────────────────────

    def _load_report_cache(self):
        try:
            if not os.path.exists(self._report_cache_file):
                return
            with gzip.open(self._report_cache_file, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._report_cache = data
        except Exception:
            self._report_cache = {}

    def _save_report_cache(self):
        try:
            os.makedirs(os.path.dirname(self._report_cache_file) or ".", exist_ok=True)
            with gzip.open(self._report_cache_file, "wt", encoding="utf-8") as f:
                json.dump(self._report_cache, f, separators=(",", ":"))
        except Exception:
            pass

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_dataset_loaded(self, event: AppEvent):
        p = event.payload
        tab_index = p.get("tab_index", 0)
        folder    = p.get("folder", "")
        new_fc    = p.get("frame_count", 0)
        with self._lock:
            old_tab = self._tabs.get(tab_index)
            if old_tab and old_tab.folder == folder:
                if new_fc != old_tab.frame_count and new_fc > 0:
                    self._scan_results.pop(folder, None)
                    self._stale_reason[folder] = (
                        f"Frame count changed: {old_tab.frame_count} → {new_fc}. Re-scan required.")
            elif folder in self._scan_results:
                old_scan = self._scan_results[folder]
                if new_fc > 0:
                    old_fc = max(
                        (v.get("n_frames", 0) for v in old_scan.band_summary.values()
                         if isinstance(v, dict)), default=0)
                    if old_fc > 0 and old_fc != new_fc:
                        self._scan_results.pop(folder, None)
                        self._stale_reason[folder] = (
                            f"Dataset reloaded with different frame count ({old_fc} → {new_fc}).")
            self._tabs[tab_index] = TabState(
                tab_index=tab_index, mode=p.get("mode", "band"),
                folder=folder, dataset_name=os.path.basename(folder),
                frame_count=new_fc, band_count=p.get("band_count", 0),
                meta=p.get("meta", {}), widget_ref=p.get("widget"),
            )
            if folder and folder not in self._session_folders:
                self._session_folders.append(folder)
                self._session_folders = self._session_folders[-self._max_session:]

    def _on_dataset_closed(self, event: AppEvent):
        with self._lock:
            self._tabs.pop(event.payload.get("tab_index", -1), None)

    def _on_frame_changed(self, event: AppEvent):
        p = event.payload
        idx = p.get("tab_index", self._active_tab_index)
        with self._lock:
            if idx in self._tabs:
                self._tabs[idx].current_frame = p.get("index", 0)

    def _on_tab_activated(self, event: AppEvent):
        p = event.payload
        idx = p.get("tab_index", 0)
        with self._lock:
            self._active_tab_index = idx
            for i, tab in self._tabs.items():
                tab.is_active = (i == idx)
            if idx in self._tabs and p.get("widget"):
                self._tabs[idx].widget_ref = p.get("widget")

    def _on_tab_closed(self, event: AppEvent):
        with self._lock:
            self._tabs.pop(event.payload.get("tab_index", -1), None)

    def _on_histogram_updated(self, event: AppEvent):
        p = event.payload
        with self._lock:
            self._histogram = HistogramState(
                folder        = p.get("folder", ""),
                dataset_name  = os.path.basename(p.get("folder", "")),
                frame_index   = p.get("frame_index", 0),
                display_mode  = p.get("display_mode", "single_frame"),
                range_start   = p.get("range_start", 0),
                range_end     = p.get("range_end", 0),
                axis_min      = p.get("axis_min", 0.0),
                axis_max      = p.get("axis_max", 1023.0),
                frame_min     = p.get("frame_min", 0.0),
                frame_max     = p.get("frame_max", 0.0),
                visible_bands = p.get("visible_bands", []),
                band_stats    = p.get("band_stats", {}),
            )

    # ── Public read API ───────────────────────────────────────────────────

    @property
    def histogram(self) -> Optional[HistogramState]:
        with self._lock:
            return self._histogram

    @property
    def active_tab(self) -> Optional[TabState]:
        with self._lock:
            return self._tabs.get(self._active_tab_index)

    @property
    def active_folder(self) -> str:
        tab = self.active_tab
        return tab.folder if tab else ""

    @property
    def active_frame(self) -> int:
        tab = self.active_tab
        return tab.current_frame if tab else 0

    @property
    def active_widget(self):
        tab = self.active_tab
        return tab.widget_ref if tab else None

    def all_tabs(self) -> List[TabState]:
        with self._lock:
            return list(self._tabs.values())

    def tab_by_folder(self, folder: str) -> Optional[TabState]:
        with self._lock:
            for tab in self._tabs.values():
                if tab.folder == folder:
                    return tab
        return None

    def clear_all(self) -> None:
        """Delete all cached scan rows."""
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("DELETE FROM scan_cache")
            except sqlite3.Error as e:
                print(f"[ScanCache] Clear error: {e}")

    def session_folders(self) -> List[str]:
        with self._lock:
            return list(reversed(self._session_folders))

    # ── Scan result API ───────────────────────────────────────────────────

    def store_scan_result(self, result: ScanResult):
        with self._lock:
            self._scan_results[result.folder] = result
            self._stale_reason.pop(result.folder, None)
            self._report_cache.pop(result.folder, None)
        try:
            self._scan_cache.store(result)
        except Exception as e:
            print(f"[AppState] Scan cache write warning: {e}")

    def get_scan_result(self, folder: str) -> Optional[ScanResult]:
        with self._lock:
            if folder in self._scan_results:
                return self._scan_results[folder]
        # Try persistent cache
        try:
            cached = self._scan_cache.load(folder)
            if cached:
                with self._lock:
                    self._scan_results[folder] = cached
                    self._stale_reason.pop(folder, None)
                return cached
        except Exception as e:
            print(f"[AppState] Scan cache read failed: {e}")
        return None

    def has_scan(self, folder: str) -> bool:
        with self._lock:
            if folder in self._scan_results:
                return True
        return self._scan_cache.is_cached(folder)

    def invalidate_scan(self, folder: str, reason: str = ""):
        with self._lock:
            self._scan_results.pop(folder, None)
            self._report_cache.pop(folder, None)
            self._save_report_cache()
            if reason:
                self._stale_reason[folder] = reason
        self._scan_cache.invalidate(folder)

    def clear_all_caches(self, clear_reports: bool = True, clear_scans: bool = True):
        """Clear in-memory + persistent scan/report caches."""
        with self._lock:
            if clear_scans:
                self._scan_results.clear()
                self._stale_reason.clear()
            if clear_reports:
                self._report_cache.clear()
        if clear_scans:
            if hasattr(self._scan_cache, "clear_all"):
                self._scan_cache.clear_all()
            else:
                try:
                    for row in self._scan_cache.list_cached():
                        self._scan_cache.invalidate(row.get("folder", ""))
                except Exception:
                    pass
        if clear_reports:
            try:
                if os.path.exists(self._report_cache_file):
                    os.remove(self._report_cache_file)
            except Exception as e:
                print(f"[AppState] Report cache delete warning: {e}")

    def get_stale_reason(self, folder: str) -> str:
        with self._lock:
            return self._stale_reason.get(folder, "")

    def all_open_folders(self) -> List[str]:
        with self._lock:
            return [t.folder for t in self._tabs.values() if t.folder]

    def list_cached_scans(self) -> list:
        """Expose scan cache listing for tools."""
        return self._scan_cache.list_cached()

    def scan_cache_stats(self) -> Dict:
        return self._scan_cache.stats()

    # ── Report cache API ──────────────────────────────────────────────────

    def get_report(self, folder: str) -> str:
        with self._lock:
            entry = self._report_cache.get(folder, {})
            return entry.get("report", "") if isinstance(entry, dict) else ""

    def store_report(self, folder: str, report: str):
        with self._lock:
            if report:
                if folder not in self._report_cache and len(self._report_cache) >= self._report_cache_max:
                    oldest = sorted(self._report_cache.items(),
                                    key=lambda kv: kv[1].get("updated_at", 0))
                    if oldest:
                        self._report_cache.pop(oldest[0][0], None)
                self._report_cache[folder] = {"report": report, "updated_at": time.time()}
                self._save_report_cache()

    # ── Context summary for Claude ────────────────────────────────────────

    def context_for_claude(self) -> str:
        lines = []
        with self._lock:
            tab       = self._tabs.get(self._active_tab_index)
            all_tabs  = list(self._tabs.values())
            stale     = dict(self._stale_reason)
            scans     = dict(self._scan_results)

        if not tab and not all_tabs:
            return "NO DATASET LOADED. No tabs are open."

        if tab:
            lines.append(
                f"ACTIVE TAB: {tab.dataset_name} | Folder: {tab.folder} | "
                f"Bands: {tab.band_count} | Frames: {tab.frame_count} | "
                f"Current frame: {tab.current_frame}"
            )
            if tab.meta:
                lines.append(
                    f"Sensor: {tab.meta.get('width','?')}×{tab.meta.get('height','?')} px, "
                    f"{tab.meta.get('bit_depth','?')}-bit"
                )
            if stale.get(tab.folder):
                lines.append(f"⚠ SCAN CACHE INVALIDATED: {stale[tab.folder]}")
            scan = scans.get(tab.folder)
            if scan:
                n = len(scan.anomaly_frames)
                lines.append(
                    f"Scan: {scan.scan_type} | Health: {scan.health_score:.0f}% | "
                    f"Anomaly frames: {n} | {scan.anomaly_frames[:10]}{'...' if n > 10 else ''}"
                )
                if scan.meta_summary:
                    ms = scan.meta_summary
                    lines.append(
                        f"Meta: SAT={ms.get('sat_id','?')} Orbit={ms.get('orbit_number','?')} "
                        f"Task={ms.get('task_id','?')} Frames={ms.get('total_frames','?')} "
                        f"Lat={ms.get('lat_start','?'):.4f}→{ms.get('lat_end','?'):.4f} "
                        f"BandsUsed={ms.get('bands_used','?')}"
                    )
                if scan.ephem_summary:
                    es = scan.ephem_summary
                    lines.append(
                        f"Ephemeris: Alt={es.get('alt_mean_km','?'):.0f}km "
                        f"Valid={es.get('all_valid',False)} Records={es.get('total_records','?')}"
                    )
            else:
                lines.append("Scan: NOT RUN for active dataset.")

        if len(all_tabs) > 1:
            lines.append(f"OPEN TABS ({len(all_tabs)}):")
            for t in all_tabs:
                am = " ← active" if t.tab_index == self._active_tab_index else ""
                scan = scans.get(t.folder)
                ss = (f"scanned/{scan.scan_type}/health={scan.health_score:.0f}%"
                      if scan else "not scanned")
                sl = " [STALE]" if stale.get(t.folder) else ""
                lines.append(f"  Tab {t.tab_index}: {t.dataset_name} | frames={t.frame_count} | {ss}{sl}{am}")

        with self._lock:
            sess = list(reversed(self._session_folders))
            hist = self._histogram
        if sess:
            lines.append(f"Session folders: {', '.join(sess[:5])}")

        if hist and (time.time() - hist.updated_at) < 300:
            lines.append(
                f"HISTOGRAM: {hist.dataset_name} frame={hist.frame_index} "
                f"mode={hist.display_mode} range={hist.axis_min:.0f}–{hist.axis_max:.0f}"
            )

        return "\n".join(lines)


# Global singleton
state = AppState()