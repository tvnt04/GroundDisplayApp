
import re

from .panel import IrisPanel
from .event_bus import bus, AppEvent, EventType
from .app_state import state
from image_viewer import GraphicsImageViewer


class Iris:

    def __init__(self, main_window):
        self._panel = IrisPanel(main_window)
        self.button = self._panel.make_toggle_button(main_window)
        bus.subscribe(EventType.OPEN_DATASET, self._on_open_dataset_request)
        bus.subscribe(EventType.CLOSE_TAB,    self._on_close_tab_request)
        bus.subscribe(EventType.CONTROL_APP,  self._on_control_app_request)
        self._main_window = main_window

    # ── App → Iris notifications ───────────────────────────────────────────

    def notify_dataset_loaded(self, tab_index: int, folder: str,
                               frame_count: int, band_count: int,
                               meta: dict, widget):
        """Call this when a dataset finishes loading in any tab."""
        bus.emit(AppEvent(EventType.DATASET_LOADED, {
            "tab_index":   tab_index,
            "folder":      folder,
            "frame_count": frame_count,
            "band_count":  band_count,
            "meta":        meta or {},
            "widget":      widget,
            "mode":        "band",
        }, source="app"))

    def notify_frame_changed(self, tab_index: int, frame_index: int):
        """Call this when the user moves to a new frame."""
        bus.emit(AppEvent(EventType.FRAME_CHANGED, {
            "tab_index": tab_index,
            "index":     frame_index,
        }, source="app"))

    def notify_tab_activated(self, tab_index: int, widget=None, mode: str = "band"):
        """Call this when the user switches to a different tab."""
        bus.emit(AppEvent(EventType.TAB_ACTIVATED, {
            "tab_index": tab_index,
            "widget":    widget,
            "mode":      mode,
        }, source="app"))

    def notify_tab_closed(self, tab_index: int):
        """Call this when a tab is closed."""
        bus.emit(AppEvent(EventType.TAB_CLOSED, {
            "tab_index": tab_index,
        }, source="app"))

    # ── Iris → App requests ────────────────────────────────────────────────

    def _on_open_dataset_request(self, event: AppEvent):
        from PyQt5.QtCore import QTimer
        folder = event.payload.get("folder", "")
        if folder and hasattr(self._main_window, "_add_band_tab"):
            QTimer.singleShot(0, lambda: self._do_open(folder))

    def _do_open(self, folder: str):
        try:
            win = self._main_window
            win._add_band_tab()
            widget = win.tab_widget.currentWidget()
            if hasattr(widget, "_open_recent_folder"):
                widget._open_recent_folder(folder)
            elif hasattr(widget, "current_folder"):
                widget.current_folder = folder
                if hasattr(widget, "load_folder_data"):
                    widget.load_folder_data()
        except Exception as e:
            print(f"[Iris] open_dataset error: {e}")

    def _on_close_tab_request(self, event: AppEvent):
        from PyQt5.QtCore import QTimer
        tab_index = event.payload.get("tab_index", -1)
        if tab_index >= 0 and hasattr(self._main_window, "close_tab"):
            QTimer.singleShot(0, lambda: self._main_window.close_tab(tab_index))

    def _resolve_tab_index(self, payload: dict) -> int:
        try:
            explicit = int(payload.get("tab_index", -1))
        except Exception:
            explicit = -1
        if explicit >= 0:
            return explicit

        folder = (payload.get("folder") or "").strip()
        dataset_name = (payload.get("dataset_name") or "").strip().lower()
        try:
            for i in range(self._main_window.tab_widget.count()):
                widget = self._main_window.tab_widget.widget(i)
                widget_folder = getattr(widget, "current_folder", "") or getattr(widget, "folder", "")
                title = self._main_window.tab_widget.tabText(i) or ""
                if folder and widget_folder == folder:
                    return i
                if dataset_name and (
                    dataset_name == title.strip().lower()
                    or dataset_name == getattr(widget, "base_name", "").strip().lower()
                ):
                    return i
        except Exception:
            pass
        try:
            return int(self._main_window.tab_widget.currentIndex())
        except Exception:
            return -1

    def _on_control_app_request(self, event: AppEvent):
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._do_control_app(event.payload or {}))

    def _do_control_app(self, payload: dict):
        win = self._main_window
        action = (payload.get("action") or "").strip().lower()
        if not action:
            return

        try:
            if action == "switch_tab":
                idx = self._resolve_tab_index(payload)
                if idx >= 0:
                    win.tab_widget.setCurrentIndex(idx)
                return

            if action == "add_tab":
                mode = (payload.get("mode") or "band").strip().lower()
                if mode == "band" and hasattr(win, "_add_band_tab"):
                    win._add_band_tab()
                elif mode == "raw" and hasattr(win, "_add_raw_tab"):
                    win._add_raw_tab()
                elif mode == "video" and hasattr(win, "_add_video_tab"):
                    win._add_video_tab()
                elif mode == "live" and hasattr(win, "_add_live_tab"):
                    win._add_live_tab()
                elif mode == "tiled" and hasattr(win, "_add_tiled_tab"):
                    win._add_tiled_tab()
                return

            if action in ("set_theme", "toggle_theme"):
                btn = getattr(win, "dark_mode_button", None)
                if btn is None:
                    return
                if action == "toggle_theme":
                    btn.setChecked(not bool(btn.isChecked()))
                    return
                value = (payload.get("value") or "").strip().lower()
                if value in ("dark", "moon", "night"):
                    btn.setChecked(True)
                elif value in ("light", "sun", "day"):
                    btn.setChecked(False)
                return

            idx = self._resolve_tab_index(payload)
            if idx < 0:
                return
            widget = win.tab_widget.widget(idx)
            if widget is None:
                return

            def _find_viewer():
                try:
                    if hasattr(widget, "view_tabs"):
                        current_idx = widget.view_tabs.currentIndex()
                        if current_idx >= 0:
                            tab_name = widget.view_tabs.tabText(current_idx)
                            if tab_name == "All Bands":
                                return getattr(widget, "all_bands_viewer", None)
                            if tab_name == "RGB Fusion":
                                return getattr(widget, "rgb_preview_viewer", None)
                            if tab_name == "Individual Bands":
                                notebook = getattr(widget, "individual_bands_notebook", None)
                                current_sub = notebook.currentWidget() if notebook is not None else None
                                return current_sub.findChild(GraphicsImageViewer) if current_sub is not None else None
                    return getattr(widget, "all_bands_viewer", None)
                except Exception:
                    return None

            def _set_view_visibility(view_name: str, visible: bool):
                cb = getattr(widget, "view_checkboxes", {}).get(view_name) if hasattr(widget, "view_checkboxes") else None
                if cb is not None:
                    cb.setChecked(bool(visible))
                if visible and hasattr(widget, "view_tabs"):
                    for i in range(widget.view_tabs.count()):
                        if widget.view_tabs.tabText(i) == view_name:
                            widget.view_tabs.setCurrentIndex(i)
                            break

            if action == "close_tab" and hasattr(win, "close_tab"):
                win.close_tab(idx)
                return

            if action in ("open_view", "close_view"):
                name = (payload.get("value") or payload.get("mode") or "").strip().lower()
                mapping = {
                    "all bands": "All Bands",
                    "all_bands": "All Bands",
                    "individual bands": "Individual Bands",
                    "individual_bands": "Individual Bands",
                    "rgb fusion": "RGB Fusion",
                    "rgb": "RGB Fusion",
                    "histogram": "Histogram",
                }
                view_name = mapping.get(name)
                if view_name:
                    _set_view_visibility(view_name, action == "open_view")
                return

            if action == "set_band_gap" and hasattr(widget, "gap_var"):
                try:
                    widget.gap_var.setValue(int(float(payload.get("value", 0))))
                    widget.update_views()
                except Exception:
                    pass
                return

            if action == "set_histogram_bands" and hasattr(widget, "histogram_viewer"):
                _set_view_visibility("Histogram", True)
                hv = widget.histogram_viewer
                raw = (payload.get("value") or "").strip()
                requested = []
                for part in raw.split(","):
                    part = part.strip().lower()
                    if not part:
                        continue
                    if re.match(r"^band\d+$", part):
                        requested.append("b" + part[4:])
                    elif re.match(r"^b\d+$", part):
                        requested.append(part)
                if not requested:
                    hv.clear_focus()
                    return
                hv._selected_bands = set(requested)
                hv._focused_band = None
                try:
                    hv._apply_selection_visibility()
                    hv._apply_zoom_view()
                    hv._update_table_emphasis()
                except Exception:
                    pass
                return

            if action in ("open_magnifier", "close_magnifier"):
                viewer = _find_viewer()
                if viewer is None and action == "open_magnifier":
                    _set_view_visibility("All Bands", True)
                    viewer = _find_viewer()
                gv = getattr(viewer, "graphics_view", None) if viewer is not None else None
                if gv is not None and hasattr(gv, "toggle_magnifier"):
                    gv.toggle_magnifier(action == "open_magnifier")
                return

            if action == "set_magnifier_center":
                from PyQt5.QtCore import QPointF
                viewer = _find_viewer()
                if viewer is None:
                    _set_view_visibility("All Bands", True)
                    viewer = _find_viewer()
                gv = getattr(viewer, "graphics_view", None) if viewer is not None else None
                if gv is not None:
                    try:
                        x = float(payload.get("x", 0))
                        y = float(payload.get("y", 0))
                        if not getattr(gv, "magnifier_enabled", False):
                            gv.toggle_magnifier(True)
                        ox, oy = x, y
                        try:
                            fw = max(1, int(getattr(viewer, "full_width", 1)))
                            fh = max(1, int(getattr(viewer, "full_height", 1)))
                            ox = max(0.0, min(float(fw - 1), float(ox)))
                            oy = max(0.0, min(float(fh - 1), float(oy)))
                        except Exception:
                            pass
                        try:
                            if hasattr(viewer, "map_original_to_scene") and hasattr(viewer, "get_original_coords"):
                                corrected_x, corrected_y = ox, oy
                                for _ in range(6):
                                    probe_scene = viewer.map_original_to_scene(corrected_x, corrected_y)
                                    actual_x, actual_y = viewer.get_original_coords(probe_scene)
                                    err_x = ox - float(actual_x)
                                    err_y = oy - float(actual_y)
                                    if abs(err_x) <= 0.75 and abs(err_y) <= 0.75:
                                        break
                                    corrected_x += err_x
                                    corrected_y += err_y
                                    try:
                                        corrected_x = max(0.0, min(float(fw - 1), float(corrected_x)))
                                        corrected_y = max(0.0, min(float(fh - 1), float(corrected_y)))
                                    except Exception:
                                        pass
                                ox, oy = corrected_x, corrected_y
                        except Exception:
                            pass
                        gv.magnifier_center = QPointF(ox, oy)
                        try:
                            if hasattr(viewer, "map_original_to_scene"):
                                target_scene = viewer.map_original_to_scene(ox, oy)
                                gv.centerOn(target_scene)
                        except Exception:
                            pass
                        gv.viewport().update()
                    except Exception:
                        pass
                return

            if action == "set_magnifier_zoom":
                viewer = _find_viewer()
                if viewer is None:
                    _set_view_visibility("All Bands", True)
                    viewer = _find_viewer()
                gv = getattr(viewer, "graphics_view", None) if viewer is not None else None
                if gv is not None and hasattr(gv, "set_magnifier_zoom"):
                    try:
                        value = float(payload.get("value", 80))
                        slider_value = int(round(value * 10.0 if value <= 50 else value))
                        slider_value = max(10, min(500, slider_value))
                        gv.set_magnifier_zoom(slider_value)
                    except Exception:
                        pass
                return

            if action == "set_contrast" and hasattr(widget, "contrast_min_var") and hasattr(widget, "contrast_max_var"):
                try:
                    min_v = payload.get("min")
                    max_v = payload.get("max")
                    enhance = payload.get("enabled")
                    if enhance is not None and hasattr(widget, "contrast_enhance_var"):
                        widget.contrast_enhance_var.setChecked(bool(enhance))
                    if min_v is not None:
                        widget.contrast_min_var.setValue(float(min_v))
                    if max_v is not None:
                        widget.contrast_max_var.setValue(float(max_v))
                    widget.update_views()
                except Exception:
                    pass
                return

            if action in ("play_pause", "play", "pause") and hasattr(widget, "toggle_play"):
                is_playing = bool(getattr(widget, "playing", False))
                if action == "play_pause" or (action == "play" and not is_playing) or (action == "pause" and is_playing):
                    widget.toggle_play()
                return

            if action == "next_frame" and hasattr(widget, "change_frame"):
                widget.change_frame(1)
                return

            if action == "prev_frame" and hasattr(widget, "change_frame"):
                widget.change_frame(-1)
                return

            if action == "refresh" and hasattr(widget, "refresh_current_tab"):
                widget.refresh_current_tab()
                return

            if action == "reload" and hasattr(widget, "reload_folder_data"):
                widget.reload_folder_data()
                return

            if action == "save_progress" and hasattr(widget, "save_parameters"):
                widget.save_parameters()
                return

            if action == "export_image" and hasattr(widget, "export_current_image"):
                widget.export_current_image()
                return

            if action == "fit_to_screen" and hasattr(widget, "fit_to_screen"):
                widget.fit_to_screen()
                return

            if action == "actual_size" and hasattr(widget, "actual_size"):
                widget.actual_size()
                return

            if action == "auto_contrast" and hasattr(widget, "set_auto_contrast"):
                widget.set_auto_contrast()
                return

            if action in ("open_terminal", "close_terminal", "toggle_terminal") and hasattr(widget, "toggle_terminal"):
                expanded = bool(getattr(widget, "_terminal_expanded", False))
                if action == "toggle_terminal":
                    widget.toggle_terminal()
                elif action == "open_terminal" and not expanded:
                    widget.toggle_terminal()
                elif action == "close_terminal" and expanded:
                    widget.toggle_terminal()
                return
        except Exception as e:
            print(f"[Iris] control_app error ({action}): {e}")
