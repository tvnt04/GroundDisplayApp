from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List
import time
import threading


class EventType(Enum):
    # Dataset lifecycle
    DATASET_LOADED    = auto()   # {folder, band_count, frame_count, meta, widget}
    DATASET_CLOSED    = auto()   # {folder, tab_index}

    # Navigation
    FRAME_CHANGED     = auto()   # {index, folder, widget}
    TAB_ACTIVATED     = auto()   # {tab_index, widget, mode}
    TAB_CLOSED        = auto()   # {tab_index}

    # UI interactions
    ZOOM_CHANGED      = auto()   # {level, center_x, center_y}

    # Histogram viewer
    HISTOGRAM_UPDATED = auto()   # {folder, frame_index, mode, range_min, range_max,
                                 #  visible_bands, band_stats, frame_min, frame_max,
                                 #  display_mode}  "single_frame" | "frame_range"

    # User → Iris
    USER_MESSAGE      = auto()   # {text}

    # Iris → App (Iris emits these to control the app)
    NAVIGATE_TO_FRAME = auto()   # {index}
    OPEN_DATASET      = auto()   # {folder}
    SET_ZOOM          = auto()   # {level}
    CLOSE_TAB         = auto()   # {tab_index}
    CONTROL_APP       = auto()   # {action, tab_index?, mode?, value?, folder?, dataset_name?}


@dataclass
class AppEvent:
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""          # which module emitted this


class EventBus:
    """
    Thread-safe pub/sub bus.
    Subscribers are called on the thread that calls emit().
    For Qt safety, wrap subscriber callbacks in QTimer.singleShot(0, fn)
    if they touch UI elements.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._history: List[AppEvent] = []
        self._max_history = 200

    def subscribe(self, event_type: EventType, callback: Callable[[AppEvent], None]):
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def emit(self, event: AppEvent):
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            callbacks = list(self._subscribers.get(event.type, []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"[EventBus] Error in subscriber for {event.type}: {e}")

    def recent(self, event_type: EventType = None, limit: int = 20) -> List[AppEvent]:
        """Get recent events, optionally filtered by type."""
        with self._lock:
            events = self._history[-limit * 3:]
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]

    def last(self, event_type: EventType) -> AppEvent | None:
        """Get the most recent event of a given type."""
        events = self.recent(event_type, limit=1)
        return events[0] if events else None


# Global singleton — import this everywhere
bus = EventBus()
