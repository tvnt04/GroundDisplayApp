from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QGroupBox, QScrollArea, QTextEdit, QProgressBar,
    QCheckBox, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QFormLayout, QButtonGroup, QTabBar,
    QFileDialog, QMessageBox, QDialog, QLabel, QPushButton, QToolButton, QLineEdit, QApplication,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QStyle, QScrollBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QEvent
from PyQt5.QtGui import QTextCursor, QTextCharFormat, QColor, QBrush, QFont, QPalette
import numpy as np
import pyqtgraph as pg
from utils import _compute_hist_for_key, check_memory_requirement
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import psutil
import math
from PIL import Image
import traceback
import math

import concurrent.futures  
import re

try:
    from iris2.event_bus import bus, AppEvent, EventType
    _HAS_IRIS_BUS = True
except Exception:
    bus = None
    AppEvent = None
    EventType = None
    _HAS_IRIS_BUS = False

class HistogramWorker(QThread):
    finished = pyqtSignal(dict)  
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, band_frames, current_frame_index, frame_mode, start_frame=None, end_frame=None, ignore_extremes=True, parent=None):
        super().__init__(parent)
        self.band_frames = band_frames
        self.current_frame_index = current_frame_index
        self.frame_mode = frame_mode
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.ignore_extremes = ignore_extremes

    def run(self):
        try:
            self.progress.emit(0)
            if not self.band_frames:
                self.finished.emit({'results': {}, 'min_val': 255, 'max_val': 0})
                return

            band_items = list(self.band_frames.items())
            num_bands = len(band_items)

            # Prepare args_list
            args_list = [
                (key, frames, self.frame_mode, self.current_frame_index, self.start_frame, self.end_frame, self.ignore_extremes)
                for key, frames in band_items
            ]

            results = []
            use_processes = True
            for _, frames in band_items:
                if isinstance(frames, list) and frames and isinstance(frames[0], np.ndarray):
                    use_processes = False
                    break

            cpu_count = max(1, psutil.cpu_count(logical=False) or psutil.cpu_count() or 1)
            max_workers = min(max(1, cpu_count), num_bands)

            completed = 0
            if use_processes:
                with ProcessPoolExecutor(max_workers=max_workers) as execp:
                    futures = [execp.submit(_compute_hist_for_key, arg) for arg in args_list]
                    for future in concurrent.futures.as_completed(futures):  # Note: concurrent.futures (full module)
                        try:
                            res = future.result()
                            results.append(res)
                            completed += 1
                            progress = int((completed / num_bands) * 100)
                            self.progress.emit(progress)
                        except Exception as exc:
                            print(f"Process future error: {exc}")
                            # Continue with partial results
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as exect:
                    futures = [exect.submit(_compute_hist_for_key, arg) for arg in args_list]
                    for future in concurrent.futures.as_completed(futures):  # Note: concurrent.futures (full module)
                        try:
                            res = future.result()
                            results.append(res)
                            completed += 1
                            progress = int((completed / num_bands) * 100)
                            self.progress.emit(progress)
                        except Exception as exc:
                            print(f"Thread future error: {exc}")
                            # Continue with partial results

            self.progress.emit(100)

            # Build key_to_result matching original logic
            key_to_result = {r[0]: r[1:] for r in results}

            # Compute overall min/max
            overall_min = None
            overall_max = None
            for result_tuple in key_to_result.values():
                if len(result_tuple) != 6:
                    continue
                hist, gmin, gmax, count, sum_val, sum_sq = result_tuple
                if count > 0:
                    if overall_min is None or gmin < overall_min:
                        overall_min = int(gmin)
                    if overall_max is None or gmax > overall_max:
                        overall_max = int(gmax)

            gc.collect()
            self.finished.emit({
                'results': key_to_result,
                'min_val': 0 if overall_min is None else overall_min,
                'max_val': 0 if overall_max is None else overall_max
            })
        except Exception as e:
            self.error.emit(str(e))

class RangeHistogramThread(QThread):
    bins_ready = pyqtSignal(object, int, int)  # dict(key->bins), processed_frames, total_frames
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # final dict: {'bins':{key:bins}, 'min_val':int, 'max_val':int}
    error = pyqtSignal(str)

    def __init__(self, band_frames, start_frame, end_frame, sample_budget=20_000_000, batch_size=1, parent=None):
        super().__init__(parent)
        self.band_frames = band_frames  # dict: key -> LazyFrames OR key -> list/ndarray
        self.start_frame = int(start_frame)
        self.end_frame = int(end_frame)
        self.sample_budget = int(sample_budget)
        self.batch_size = int(max(1, batch_size))
        self._abort = threading.Event()

    def stop(self):
        self._abort.set()

    def _compute_stride(self, h, w, n_frames):
        # Determine integer stride so that (w/stride)*(h/stride)*n_frames <= sample_budget
        if n_frames <= 0:
            return 1
        total_pixels = int(w) * int(h) * int(n_frames)
        if total_pixels <= self.sample_budget:
            return 1
        # stride^2 >= (w*h*n_frames) / sample_budget
        ratio = (total_pixels + self.sample_budget - 1) // self.sample_budget
        # stride = ceil(sqrt(ratio)) -- use math.sqrt for estimate (small, inexpensive)
        
        stride = int(math.ceil(math.sqrt(ratio)))
        return max(1, stride)

    def run(self):
        try:
            keys = list(self.band_frames.keys())
            # Validate frames object types
            frames_objs = {k: self.band_frames[k] for k in keys}

            # Determine frame count and sample strategy from first LazyFrames-like object
            sample_h = sample_w = None
            n_frames = max(1, (self.end_frame - self.start_frame + 1))
            bitdepth = 8
            for obj in frames_objs.values():
                if hasattr(obj, 'get_raw') and hasattr(obj, 'bitdepth') and hasattr(obj, 'w'):
                    sample_h = getattr(obj, 'h', None) or getattr(obj, 'height', None)
                    sample_w = getattr(obj, 'w', None) or getattr(obj, 'width', None)
                    bitdepth = int(getattr(obj, 'bitdepth', 8))
                    break
                elif isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], np.ndarray):
                    sample_h, sample_w = obj[0].shape
                    bitdepth = 8 if obj[0].dtype == np.uint8 else 16
                    break

            if sample_h is None or sample_w is None:
                # Fallback: try to probe first frame
                for obj in frames_objs.values():
                    try:
                        if hasattr(obj, 'get_raw'):
                            arr = obj.get_raw(self.start_frame)
                            sample_h, sample_w = arr.shape
                            bitdepth = int(getattr(obj, 'bitdepth', 8))
                            break
                    except Exception:
                        continue

            if sample_h is None or sample_w is None:
                self.error.emit('Could not determine frame dimensions for range histogram')
                return

            stride = self._compute_stride(sample_h, sample_w, n_frames)

            # Keep range-mode histogram in native DN domain (not forced to 8-bit).
            if bitdepth <= 8:
                num_bins = 256
            else:
                num_bins = min(1 << int(bitdepth), 65536)
            accum = {k: np.zeros(num_bins, dtype=np.uint64) for k in keys}

            processed = 0
            total = n_frames

            for fi in range(self.start_frame, self.end_frame + 1):
                if self._abort.is_set():
                    return

                # Load each requested band for this frame (single-frame memory only)
                for k, obj in frames_objs.items():
                    try:
                        if hasattr(obj, 'get_raw'):
                            raw = obj.get_raw(fi)
                        elif isinstance(obj, (list, tuple)) and len(obj) > fi:
                            raw = obj[fi]
                        else:
                            # unsupported type; skip
                            continue

                        # Subsample spatially using integer stride (keeps memory small)
                        if stride > 1:
                            sampled = raw[::stride, ::stride]
                        else:
                            sampled = raw

                        vals = sampled.astype(np.uint32, copy=False).ravel()
                        if vals.size == 0:
                            continue
                        vals = np.clip(vals, 0, num_bins - 1)
                        bc = np.bincount(vals, minlength=num_bins)
                        accum[k] += bc.astype(np.uint64)

                    except Exception as ex:
                        # Continue processing other bands/frames; report later
                        print(f"RangeHistogramThread: skipping frame {fi} for {k}: {ex}")

                processed += 1
                # Emit progressive update after each frame (caller will redraw)
                self.bins_ready.emit({k: accum[k].copy() for k in keys}, processed, total)
                pct = int((processed / float(total)) * 100)
                self.progress.emit(pct)

                # Allow other threads to set abort
                if self._abort.is_set():
                    return

            # Finished
            overall_min = num_bins - 1
            overall_max = 0
            for k, b in accum.items():
                nz = np.nonzero(b)[0]
                if nz.size:
                    overall_min = min(int(nz[0]), overall_min)
                    overall_max = max(int(nz[-1]), overall_max)

            self.finished.emit({'bins': {k: v.copy() for k, v in accum.items()}, 'min_val': overall_min, 'max_val': overall_max})
        except Exception as e:
            
            traceback.print_exc()
            self.error.emit(str(e))


class HistogramViewer(QWidget):
    mode_changed = pyqtSignal()
    bandFocused = pyqtSignal(object)
    focusCleared = pyqtSignal()
    minmax_updated = pyqtSignal(int, int)  # Emits min_val, max_val

    def __init__(self, parent=None):
        super().__init__(parent)

        # Iris context (optional): set by caller when histogram is updated
        self._iris_folder = ""
        self._iris_frame_index = 0

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        self.setLayout(main_layout)

        self.single_frame_radio = QRadioButton("Single Frame")
        self.single_frame_radio.setChecked(True)

        self.frame_range_radio = QRadioButton("Frame Range")

        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.setToolTip("Show all histogram curves")
        self.show_all_btn.clicked.connect(self._show_all)

        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setToolTip("Zoom in on histogram plot")
        self.zoom_in_btn.clicked.connect(self._zoom_in)

        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setToolTip("Zoom out on histogram plot")
        self.zoom_out_btn.clicked.connect(self._zoom_out)

        self.hist_table_btn = QPushButton("Hide Table")
        self.hist_table_btn.setToolTip("Show or hide histogram statistics table")
        self.hist_table_btn.setCheckable(True)
        self.hist_table_btn.setEnabled(False)
        self.hist_table_btn.setVisible(False)
        self.hist_table_btn.clicked.connect(self._on_toggle_table_clicked)

        self.hist_fs_btn = QPushButton("Fullscreen")
        self.hist_fs_btn.setToolTip("Open histogram in fullscreen")
        self.hist_fs_btn.clicked.connect(self._on_hist_fs_btn_clicked)

        try:
            self.single_frame_radio.toggled.connect(lambda checked: self.mode_changed.emit() if checked else None)
            self.frame_range_radio.toggled.connect(lambda checked: self.mode_changed.emit() if checked else None)
        except Exception:
            pass

        content_layout = QHBoxLayout()
        self.content_layout = content_layout
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        main_layout.addLayout(content_layout, 1)

        left_panel = QWidget(self)
        self._left_panel = left_panel
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        self._left_stack_spacing = 4

        self.legend_area = QWidget(left_panel)
        self.legend_layout = QHBoxLayout(self.legend_area)
        self.legend_layout.setContentsMargins(2, 0, 2, 0)
        self.legend_layout.setSpacing(10)
        left_layout.addWidget(self.legend_area, 0)

        plot_row = QWidget(left_panel)
        self._plot_row = plot_row  # Store for later alignment calculations
        plot_row_layout = QHBoxLayout(plot_row)
        plot_row_layout.setContentsMargins(0, 0, 0, 0)
        plot_row_layout.setSpacing(2)

        self.plot = pg.PlotWidget(self)
        self.plot.setMinimumHeight(160)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plot_item = self.plot.getPlotItem()
        self.plot_item.setAspectLocked(False)
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.setMenuEnabled(False)
        self.plot_item.hideButtons()
        self.plot_item.showGrid(x=True, y=True, alpha=0.18)
        self.plot_item.setLabel("bottom", "Pixel Value")
        self.plot_item.setLabel("left", "Count")
        plot_row_layout.addWidget(self.plot, 1)

        self.y_scroll = QScrollBar(Qt.Vertical, plot_row)
        self.y_scroll.setVisible(False)
        self.y_scroll.valueChanged.connect(self._on_y_scroll)
        plot_row_layout.addWidget(self.y_scroll, 0)
        left_layout.addWidget(plot_row, 0)

        self.x_scroll = QScrollBar(Qt.Horizontal, left_panel)
        self.x_scroll.setVisible(False)
        self.x_scroll.valueChanged.connect(self._on_x_scroll)
        left_layout.addWidget(self.x_scroll, 0)

        self.minmax_bar = QWidget(left_panel)
        minmax_layout = QHBoxLayout(self.minmax_bar)
        minmax_layout.setContentsMargins(2, 0, 2, 0)
        minmax_layout.setSpacing(14)
        self.min_label = QLabel("Min: -", self.minmax_bar)
        self.max_label = QLabel("Max: -", self.minmax_bar)
        minmax_layout.addWidget(self.min_label)
        minmax_layout.addWidget(self.max_label)
        minmax_layout.addStretch()
        left_layout.addWidget(self.minmax_bar, 0)

        content_layout.addWidget(left_panel, 1)

        self.stats_table = QTableWidget(self)
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Data", "Mean", "Var", "Std"])
        self.stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stats_table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setStretchLastSection(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.stats_table.setMinimumWidth(260)
        self.stats_table.setMaximumWidth(260)
        self.stats_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.stats_table.setFocusPolicy(Qt.NoFocus)
        self.stats_table.cellClicked.connect(self._on_table_clicked)

        right_panel = QWidget(self)
        self._right_panel = right_panel
        right_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self._table_top_spacer = QWidget(right_panel)
        self._table_top_spacer.setFixedHeight(0)
        right_layout.addWidget(self._table_top_spacer, 0)
        right_layout.addWidget(self.stats_table, 0)
        right_layout.addStretch(1)
        content_layout.addWidget(right_panel, 0)
        content_layout.setStretch(0, 10)
        content_layout.setStretch(1, 0)

        self.hist_progress = QProgressBar(self)
        self.hist_progress.setFixedHeight(3)
        self.hist_progress.setTextVisible(False)
        self.hist_progress.setVisible(False)
        main_layout.addWidget(self.hist_progress)

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.single_frame_radio)
        control_layout.addWidget(self.frame_range_radio)
        control_layout.addWidget(self.show_all_btn)
        control_layout.addWidget(self.zoom_in_btn)
        control_layout.addWidget(self.zoom_out_btn)
        control_layout.addWidget(self.hist_table_btn)
        control_layout.addWidget(self.hist_fs_btn)
        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        self.min_val = 255
        self.max_val = 0
        self.band_items = []
        self.frame_mode = "Single"
        self.start_frame = 0
        self.end_frame = 0
        self.worker = None
        self._y_display_max = 1.0
        self._focused_band = None
        self._selected_bands = set()
        self._curves = {}
        self._curve_colors = {}
        self._row_to_band = {}
        self._band_to_row = {}
        self._legend_labels = {}
        self._legend_swatches = {}
        self._legend_colors = {}
        self._band_minmax = {}
        self._overall_minmax = (None, None)
        self._frame_set_minmax = (None, None)
        self._default_x_range = (0.0, 255.0)
        self._default_y_range = (0.0, 1.0)
        self._zoom_level = 0
        self._max_zoom_level = 0
        self._x_scroll_pos = 0.5
        self._x_scroll_user_override = False
        self._x_scroll_guard = False
        self._y_scroll_pos = 0.0
        self._y_scroll_user_override = False
        self._y_scroll_guard = False
        self._hist_is_fullscreen = False
        self._hist_table_visible = True
        self._hist_saved_ui_state = {}
        self._hist_original_parent = None
        self._hist_original_layout = None
        self._hist_original_index = -1
        self._hist_fullscreen_dialog = None
        self._hist_restoring = False
        self._hist_fs_requested = False
        # Bright, readable palette for dark backgrounds.
        self._palette = [
            (255, 99, 71), (46, 204, 113), (52, 152, 219), (26, 188, 156), (155, 89, 182),
            (241, 196, 15), (230, 126, 34), (231, 76, 60), (99, 205, 218), (174, 130, 255),
            (255, 173, 51), (180, 190, 196), (111, 207, 151), (114, 159, 207)
        ]

        self._apply_theme()
        self._deferred_plot_ratio_needed = True
        self._fit_table_width()
        self._update_zoom_buttons()

    def showEvent(self, event):
        """Called when the widget is shown. Apply deferred plot ratio if needed."""
        super().showEvent(event)
        if getattr(self, '_deferred_plot_ratio_needed', False):
            QTimer.singleShot(0, self._apply_plot_ratio)
            self._deferred_plot_ratio_needed = False

    def _is_dark_theme(self):
        app = QApplication.instance()
        palette = app.palette() if app else self.palette()
        return palette.color(QPalette.Window).lightness() < 128

    def _apply_theme(self):
        dark = self._is_dark_theme()
        if dark:
            bg = "#07090d"
            border = "#2c3340"
            axis = "#c3cedd"
            grid_alpha = 55
            table_bg = "#161b22"
            table_alt = "#1d2430"
            table_fg = "#d8e0ea"
            header_bg = "#202837"
            selection_bg = "#2b5ca8"
        else:
            bg = "#f5f7fb"
            border = "#b0bfd4"
            axis = "#1a2332"
            grid_alpha = 60
            table_bg = "#ffffff"
            table_alt = "#f0f4fa"
            table_fg = "#0d1117"
            header_bg = "#d4e1f3"
            selection_bg = "#80c9f4"

        self._axis_color_hex = axis
        self.plot.setBackground(bg)
        self.plot.setStyleSheet(f"border: 1px solid {border}; border-radius: 6px;")
        self.plot_item.showGrid(x=True, y=True, alpha=grid_alpha / 255.0)
        self.plot_item.setLabel("bottom", "Pixel Value", color=axis)
        self.plot_item.setLabel("left", "Count", color=axis)

        axis_pen = pg.mkPen(axis, width=1)
        for axis_name in ("left", "bottom"):
            ax = self.plot_item.getAxis(axis_name)
            ax.setPen(axis_pen)
            ax.setTextPen(axis_pen)

        self.stats_table.setStyleSheet(
            f"""
            QTableWidget {{
                background: {table_bg};
                alternate-background-color: {table_alt};
                color: {table_fg};
                border: 1px solid {border};
                gridline-color: {border};
                selection-background-color: {selection_bg};
            }}
            QHeaderView::section {{
                background: {header_bg};
                color: {table_fg};
                border: 0px;
                border-bottom: 1px solid {border};
                padding: 4px 6px;
                font-weight: 600;
            }}
            """
        )
        self._update_table_emphasis()
        self._style_legend_and_minmax()

    def _style_legend_and_minmax(self):
        dark = self._is_dark_theme()
        fg = "#d8e0ea" if dark else "#0d1117"
        dim = "#9cb0c8" if dark else "#3d4655"
        self.legend_area.setStyleSheet(f"QLabel {{ color: {fg}; }}")
        self.min_label.setStyleSheet(f"color: {dim}; font-weight: 600;")
        self.max_label.setStyleSheet(f"color: {dim}; font-weight: 600;")
        self.x_scroll.setStyleSheet("QScrollBar:horizontal { height: 12px; }")
        self.y_scroll.setStyleSheet("QScrollBar:vertical { width: 12px; }")

    def changeEvent(self, event):
        if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            self._apply_theme()
        super().changeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_plot_ratio()

    def _stats_from_hist(self, counts, x_values=None):
        counts = np.asarray(counts, dtype=np.float64)
        if counts.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        total = float(np.sum(counts))
        if total <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        if x_values is None:
            x_values = np.arange(counts.size, dtype=np.float64)
        else:
            x_values = np.asarray(x_values, dtype=np.float64)
            if x_values.size != counts.size:
                x_values = np.arange(counts.size, dtype=np.float64)
        mean = float(np.sum(x_values * counts) / total)
        var = float(np.sum(((x_values - mean) ** 2) * counts) / total)
        sd = math.sqrt(max(0.0, var))
        return (mean, var, sd, total)

    def _make_pen(self, color, width=2, alpha=255):
        c = QColor(color)
        c.setAlpha(int(alpha))
        return pg.mkPen(c, width=width)

    def _fmt_num(self, value):
        if value is None:
            return "-"
        try:
            return f"{float(value):.4g}"
        except Exception:
            return "-"

    def _reset_plot_data(self):
        self.plot_item.clear()
        self._curves.clear()
        self._curve_colors.clear()
        self._row_to_band.clear()
        self._band_to_row.clear()
        self._legend_labels.clear()
        self._legend_swatches.clear()
        self._legend_colors.clear()
        self._band_minmax.clear()
        self._overall_minmax = (None, None)
        self._frame_set_minmax = (None, None)
        self._default_x_range = (0.0, 255.0)
        self._default_y_range = (0.0, 1.0)
        self._zoom_level = 0
        self._max_zoom_level = 0
        self._x_scroll_pos = 0.5
        self._x_scroll_user_override = False
        self.x_scroll.setVisible(False)
        self._y_scroll_pos = 0.0
        self._y_scroll_user_override = False
        self.y_scroll.setVisible(False)
        self._selected_bands = set()
        self._clear_legend()
        self._fit_table_width()
        self._update_minmax_labels()
        self._update_zoom_buttons()

    def _clear_legend(self):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_legend_item(self, band, color):
        swatch = QLabel("")
        swatch.setFixedSize(11, 11)
        swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid rgba(255,255,255,40); border-radius: 2px;")
        name = QLabel(str(band))
        self.legend_layout.addWidget(swatch)
        self.legend_layout.addWidget(name)
        self._legend_swatches[band] = swatch
        self._legend_labels[band] = name
        self._legend_colors[band] = QColor(color)

    def _update_legend_emphasis(self):
        focused = self._focused_band
        selected = set(getattr(self, "_selected_bands", set()))
        dark = self._is_dark_theme()
        normal_text = "#d8e0ea" if dark else "#243042"
        for band, label in self._legend_labels.items():
            swatch = self._legend_swatches.get(band)
            c = self._legend_colors.get(band, QColor(180, 180, 180))
            if selected:
                if band in selected:
                    label.setStyleSheet(f"color: {normal_text}; font-weight: 700;")
                    swatch.setStyleSheet(
                        f"background-color: rgba({c.red()},{c.green()},{c.blue()},255);"
                        f"border: 2px solid rgba(255,255,255,230); border-radius: 2px;"
                    )
                else:
                    label.setStyleSheet(f"color: {normal_text}; font-weight: 500;")
                    swatch.setStyleSheet(
                        f"background-color: rgba({c.red()},{c.green()},{c.blue()},70);"
                        f"border: 1px solid rgba(255,255,255,70); border-radius: 2px;"
                    )
            elif focused is None:
                label.setStyleSheet(f"color: {normal_text}; font-weight: 500;")
                swatch.setStyleSheet(
                    f"background-color: rgba({c.red()},{c.green()},{c.blue()},255);"
                    f"border: 1px solid rgba(255,255,255,120); border-radius: 2px;"
                )
            elif band == focused:
                label.setStyleSheet(f"color: {normal_text}; font-weight: 700;")
                swatch.setStyleSheet(
                    f"background-color: rgba({c.red()},{c.green()},{c.blue()},255);"
                    f"border: 2px solid rgba(255,255,255,230); border-radius: 2px;"
                )
            else:
                label.setStyleSheet(f"color: {normal_text}; font-weight: 500;")
                swatch.setStyleSheet(
                    f"background-color: rgba({c.red()},{c.green()},{c.blue()},255);"
                    f"border: 1px solid rgba(255,255,255,120); border-radius: 2px;"
                )

    def _format_dn(self, v):
        if v is None:
            return "-"
        try:
            fv = float(v)
            iv = int(round(fv))
            if abs(fv - iv) < 1e-6:
                return str(iv)
            return f"{fv:.2f}"
        except Exception:
            return "-"

    def _extreme_bands(self, value, candidates=None, tol=1e-6):
        """Return band names whose min/max exactly matches the provided value."""
        if value is None:
            return []
        try:
            target = float(value)
        except Exception:
            return []
        names = candidates if candidates is not None else list(self._band_minmax.keys())
        out = []
        for b in names:
            mm = self._band_minmax.get(b)
            if not mm:
                continue
            bmin, bmax = mm
            if abs(float(bmin) - target) <= tol or abs(float(bmax) - target) <= tol:
                out.append(str(b))
        return out

    def _update_minmax_labels(self):
        selected = [b for b in getattr(self, "_selected_bands", set()) if b in self._band_minmax]
        min_bands = []
        max_bands = []
        if len(selected) == 1:
            band = selected[0]
            mn, mx = self._band_minmax[band]
            prefix = f"{band} "
        elif len(selected) >= 2:
            vals = [self._band_minmax[b] for b in selected]
            mn = min(v[0] for v in vals)
            mx = max(v[1] for v in vals)
            prefix = "Selected "
            min_bands = self._extreme_bands(mn, candidates=selected)
            max_bands = self._extreme_bands(mx, candidates=selected)
        elif self._focused_band is not None and self._focused_band in self._band_minmax:
            mn, mx = self._band_minmax[self._focused_band]
            prefix = f"{self._focused_band} "
        else:
            fmn, fmx = self._frame_set_minmax
            if fmn is not None and fmx is not None:
                mn, mx = fmn, fmx
                mode_txt = str(getattr(self, "frame_mode", "")).strip().lower()
                if mode_txt.startswith("range"):
                    prefix = "All Bands (range) "
                else:
                    prefix = "All Bands (frame) "
                min_bands = self._extreme_bands(mn)
                max_bands = self._extreme_bands(mx)
            else:
                mn, mx = self._overall_minmax
                prefix = "All Bands "
                min_bands = self._extreme_bands(mn)
                max_bands = self._extreme_bands(mx)

        min_owner = f" ({', '.join(min_bands)})" if min_bands else ""
        max_owner = f" ({', '.join(max_bands)})" if max_bands else ""
        self.min_label.setText(f"{prefix}Min: {self._format_dn(mn)}{min_owner}")
        self.max_label.setText(f"{prefix}Max: {self._format_dn(mx)}{max_owner}")

    def _set_frame_set_minmax(self, mn, mx):
        try:
            if mn is None or mx is None:
                self._frame_set_minmax = (None, None)
            else:
                self._frame_set_minmax = (int(mn), int(mx))
        except Exception:
            self._frame_set_minmax = (None, None)
        self._update_minmax_labels()

    def _fit_view(self):
        if not self._curves:
            return
        self._zoom_level = 0
        self._x_scroll_user_override = False
        self._x_scroll_pos = 0.5
        self._y_scroll_user_override = False
        self._y_scroll_pos = 0.0
        self._apply_zoom_view()

    def _zoom_in(self):
        if not self._curves:
            return
        if self._zoom_level >= self._max_zoom_level:
            return
        self._zoom_level += 1
        self._x_scroll_user_override = False
        self._y_scroll_user_override = False
        self._apply_zoom_view()

    def _zoom_out(self):
        if not self._curves:
            return
        if self._zoom_level <= 0:
            self._zoom_level = 0
            self._x_scroll_user_override = False
            self._y_scroll_user_override = False
            self._apply_zoom_view()
            return
        self._zoom_level -= 1
        if self._zoom_level == 0:
            self._x_scroll_user_override = False
            self._y_scroll_user_override = False
        self._apply_zoom_view()

    def _update_zoom_buttons(self):
        has_data = bool(self._curves)
        self.zoom_in_btn.setEnabled(has_data and self._zoom_level < self._max_zoom_level)
        self.zoom_out_btn.setEnabled(has_data and self._zoom_level > 0)

    def _active_x_range(self):
        selected = [b for b in getattr(self, "_selected_bands", set()) if b in self._band_minmax]
        if selected:
            vals = [self._band_minmax[b] for b in selected]
            return (min(v[0] for v in vals), max(v[1] for v in vals))
        if self._focused_band is not None and self._focused_band in self._band_minmax:
            return self._band_minmax[self._focused_band]
        return self._overall_minmax

    def _apply_zoom_view(self):
        if not self._curves:
            self._update_zoom_buttons()
            return

        x0, x1 = self._default_x_range
        y0, y1 = self._default_y_range
        x0 = float(x0)
        x1 = max(float(x1), x0 + 1.0)
        y1 = max(1.0, float(y1))

        if self._zoom_level <= 0:
            self.plot.setXRange(x0, x1, padding=0)
            self.plot.setYRange(y0, y1, padding=0)
            self.x_scroll.setVisible(False)
            self.y_scroll.setVisible(False)
            self._update_zoom_buttons()
            return

        a0, a1 = self._active_x_range()
        if a0 is None or a1 is None:
            a0, a1 = x0, x1
        span_default = max(1.0, x1 - x0)
        center = (float(a0) + float(a1)) * 0.5
        span = span_default * (0.78 ** self._zoom_level)
        span = min(span, span_default)
        min_start = x0
        max_start = max(x0, x1 - span)
        if max_start > min_start:
            if not self._x_scroll_user_override:
                start = min(max(center - span * 0.5, min_start), max_start)
                self._x_scroll_pos = (start - min_start) / (max_start - min_start)
            else:
                pos = max(0.0, min(1.0, float(self._x_scroll_pos)))
                start = min_start + (max_start - min_start) * pos
            xmin = start
            xmax = start + span
        else:
            xmin = x0
            xmax = min(x1, x0 + span)

        y_span_default = max(1.0, y1 - y0)
        y_span = max(1.0, y_span_default * (0.78 ** self._zoom_level))
        y_span = min(y_span, y_span_default)
        y_min_start = y0
        y_max_start = max(y0, y1 - y_span)
        if y_max_start > y_min_start:
            if not self._y_scroll_user_override:
                ystart = y_min_start
                self._y_scroll_pos = 0.0
            else:
                ypos = max(0.0, min(1.0, float(self._y_scroll_pos)))
                ystart = y_min_start + (y_max_start - y_min_start) * ypos
            ymin = ystart
            ymax = ystart + y_span
        else:
            ymin = y0
            ymax = min(y1, y0 + y_span)
        self.plot.setXRange(xmin, xmax, padding=0)
        self.plot.setYRange(ymin, ymax, padding=0)
        self._sync_x_scrollbar(xmin, xmax)
        self._sync_y_scrollbar(ymin, ymax)
        self._update_zoom_buttons()

    def _sync_x_scrollbar(self, xmin, xmax):
        if self._zoom_level <= 0:
            self.x_scroll.setVisible(False)
            return
        x0, x1 = self._default_x_range
        span = max(1.0, float(xmax) - float(xmin))
        total = max(1.0, float(x1) - float(x0))
        min_start = float(x0)
        max_start = max(float(x0), float(x1) - span)
        can_scroll = max_start > min_start + 1e-9
        self.x_scroll.setVisible(can_scroll)
        if not can_scroll:
            return
        pos = max(0.0, min(1.0, (float(xmin) - min_start) / (max_start - min_start)))
        self._x_scroll_pos = pos
        self._x_scroll_guard = True
        try:
            self.x_scroll.setRange(0, 1000)
            self.x_scroll.setPageStep(max(1, int(round((span / total) * 1000))))
            self.x_scroll.setValue(int(round(pos * 1000)))
        finally:
            self._x_scroll_guard = False

    def _sync_y_scrollbar(self, ymin, ymax):
        if self._zoom_level <= 0:
            self.y_scroll.setVisible(False)
            return
        y0, y1 = self._default_y_range
        span = max(1.0, float(ymax) - float(ymin))
        total = max(1.0, float(y1) - float(y0))
        min_start = float(y0)
        max_start = max(float(y0), float(y1) - span)
        can_scroll = max_start > min_start + 1e-9
        self.y_scroll.setVisible(can_scroll)
        if not can_scroll:
            return
        pos = max(0.0, min(1.0, (float(ymin) - min_start) / (max_start - min_start)))
        self._y_scroll_pos = pos
        self._y_scroll_guard = True
        try:
            self.y_scroll.setRange(0, 1000)
            self.y_scroll.setPageStep(max(1, int(round((span / total) * 1000))))
            self.y_scroll.setValue(int(round((1.0 - pos) * 1000)))
        finally:
            self._y_scroll_guard = False

    def _on_x_scroll(self, value):
        if self._x_scroll_guard:
            return
        if not self._curves or self._zoom_level <= 0:
            return
        self._x_scroll_pos = max(0.0, min(1.0, float(value) / 1000.0))
        self._x_scroll_user_override = True
        self._apply_zoom_view()

    def _on_y_scroll(self, value):
        if self._y_scroll_guard:
            return
        if not self._curves or self._zoom_level <= 0:
            return
        self._y_scroll_pos = max(0.0, min(1.0, 1.0 - (float(value) / 1000.0)))
        self._y_scroll_user_override = True
        self._apply_zoom_view()

    def _apply_plot_ratio(self):
        # Expand plot to available vertical space (do not reserve centered slack).
        if not hasattr(self, "_left_panel"):
            return

        legend_h = self.legend_area.sizeHint().height() if hasattr(self, "legend_area") else 0
        minmax_h = self.minmax_bar.sizeHint().height() if hasattr(self, "minmax_bar") else 0
        margins = 20
        available_h = max(120, self._left_panel.height() - legend_h - minmax_h - margins)
        
        # Use the same calculation regardless of whether data is loaded or not
        # This ensures the histogram stays within screen bounds in both cases
        plot_h = max(160, int(available_h * 0.93))
        
        self.plot.setFixedHeight(plot_h)
        # Force layout update, then align table to the actual plotted widget geometry.
        lay = self._left_panel.layout()
        if lay is not None:
            lay.activate()
        # Get the actual top position of plot_row within left_panel
        # This accounts for all spacing and margins
        if hasattr(self, "_plot_row"):
            plot_top = self._plot_row.y()
        else:
            plot_top = legend_h + self._left_stack_spacing
        self._table_top_spacer.setFixedHeight(max(0, plot_top))
        self.stats_table.setFixedHeight(max(120, self.plot.height()))
        self._fit_table_rows_to_height()

    def _show_all(self):
        self.clear_focus()
        self._fit_view()

    def _set_hist_table_visible(self, visible):
        visible = bool(visible)
        self._hist_table_visible = visible
        if hasattr(self, "_right_panel") and self._right_panel is not None:
            self._right_panel.setVisible(visible)
        if hasattr(self, "content_layout"):
            try:
                self.content_layout.setStretch(0, 10)
                self.content_layout.setStretch(1, 0)
            except Exception:
                pass
        if hasattr(self, "hist_table_btn"):
            self.hist_table_btn.blockSignals(True)
            self.hist_table_btn.setChecked(not visible)
            self.hist_table_btn.setText("Show Table" if not visible else "Hide Table")
            self.hist_table_btn.blockSignals(False)
        self._apply_plot_ratio()

    def _on_toggle_table_clicked(self, checked):
        # checked=True means "table hidden" for this button semantics.
        self._set_hist_table_visible(not bool(checked))

    def _on_hist_fs_btn_clicked(self):
        self._hist_fs_requested = True
        try:
            self.toggle_histogram_fullscreen()
        finally:
            self._hist_fs_requested = False

    def _restore_from_histogram_fullscreen(self):
        if getattr(self, "_hist_restoring", False):
            return
        self._hist_restoring = True
        try:
            dialog = getattr(self, "_hist_fullscreen_dialog", None)
            state = dict(getattr(self, "_hist_saved_ui_state", {}) or {})

            try:
                if dialog is not None and self.parentWidget() is dialog:
                    if dialog.layout() is not None:
                        dialog.layout().removeWidget(self)
            except Exception:
                pass

            parent = getattr(self, "_hist_original_parent", None)
            layout = getattr(self, "_hist_original_layout", None)
            idx = int(getattr(self, "_hist_original_index", -1))
            if parent is not None and layout is not None:
                self.setParent(parent)
                if idx >= 0 and idx <= layout.count():
                    layout.insertWidget(idx, self)
                else:
                    layout.addWidget(self)

            self._hist_is_fullscreen = False
            self.hist_fs_btn.setText("Fullscreen")
            self.hist_table_btn.setEnabled(False)
            self.hist_table_btn.setVisible(False)
            self._set_hist_table_visible(bool(state.get("table_visible", True)))
            self._apply_plot_ratio()

            self._hist_saved_ui_state = {}
            self._hist_original_parent = None
            self._hist_original_layout = None
            self._hist_original_index = -1
            self._hist_fullscreen_dialog = None
        finally:
            self._hist_restoring = False

    def toggle_histogram_fullscreen(self):
        # Enter fullscreen only from explicit button click.
        if not bool(getattr(self, "_hist_is_fullscreen", False)) and not bool(getattr(self, "_hist_fs_requested", False)):
            return
        if not bool(getattr(self, "_hist_is_fullscreen", False)):
            parent = self.parentWidget()
            if parent is None:
                return
            layout = parent.layout()
            if layout is None:
                return

            idx = -1
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is not None and item.widget() is self:
                    idx = i
                    break

            self._hist_saved_ui_state = {
                "table_visible": bool(getattr(self, "_hist_table_visible", True)),
            }
            self._hist_original_parent = parent
            self._hist_original_layout = layout
            self._hist_original_index = idx

            owner = self.window()
            dialog = QDialog(owner if owner is not None else None)
            dialog.setWindowTitle("Histogram Fullscreen")
            dialog.setWindowFlag(Qt.Window, True)
            dialog.setModal(False)
            dlay = QVBoxLayout(dialog)
            dlay.setContentsMargins(0, 0, 0, 0)
            dlay.setSpacing(0)

            layout.removeWidget(self)
            self.setParent(dialog)
            dlay.addWidget(self)
            dialog.finished.connect(lambda _r: self._restore_from_histogram_fullscreen())

            self._hist_fullscreen_dialog = dialog
            self._hist_is_fullscreen = True
            self.hist_fs_btn.setText("Exit Fullscreen")
            self.hist_table_btn.setEnabled(True)
            self.hist_table_btn.setVisible(True)
            self._set_hist_table_visible(True)
            try:
                # Keep fullscreen on the same screen as the owning main window.
                screen = None
                if owner is not None and owner.windowHandle() is not None:
                    screen = owner.windowHandle().screen()
                if screen is not None:
                    dialog.setGeometry(screen.availableGeometry())
            except Exception:
                pass
            dialog.showFullScreen()
            self._apply_plot_ratio()
        else:
            dialog = getattr(self, "_hist_fullscreen_dialog", None)
            if dialog is not None:
                try:
                    dialog.close()
                except Exception:
                    self._restore_from_histogram_fullscreen()
            else:
                self._restore_from_histogram_fullscreen()

    def _apply_selection_visibility(self):
        selected = set(getattr(self, "_selected_bands", set()))
        has_multi = len(selected) > 0
        for key, curve in self._curves.items():
            color = self._curve_colors[key]
            if has_multi:
                is_selected = key in selected
                curve.setVisible(is_selected)
                curve.setPen(self._make_pen(color, width=3 if is_selected else 1, alpha=255 if is_selected else 28))
                curve.setZValue(3 if is_selected else 1)
            elif self._focused_band is None:
                curve.setVisible(True)
                curve.setPen(self._make_pen(color, width=2, alpha=255))
                curve.setZValue(2)
            elif key == self._focused_band:
                curve.setVisible(True)
                curve.setPen(self._make_pen(color, width=4, alpha=255))
                curve.setZValue(3)
            else:
                curve.setVisible(True)
                curve.setPen(self._make_pen(color, width=1, alpha=28))
                curve.setZValue(1)

        self.stats_table.blockSignals(True)
        self.stats_table.clearSelection()
        if has_multi:
            for band in selected:
                row = self._band_to_row.get(band)
                if row is not None:
                    self.stats_table.selectRow(row)
        elif self._focused_band is not None:
            row = self._band_to_row.get(self._focused_band)
            if row is not None:
                self.stats_table.selectRow(row)
        self.stats_table.blockSignals(False)

    def _update_table_emphasis(self):
        dark = self._is_dark_theme()
        focused = self._focused_band
        selected = set(getattr(self, "_selected_bands", set()))
        active = QColor("#ffffff" if dark else "#000000")
        faint = QColor("#8796ab" if dark else "#8a96a7")
        normal = QColor("#d8e0ea" if dark else "#243042")
        for row in range(self.stats_table.rowCount()):
            band = self._row_to_band.get(row)
            for col in range(self.stats_table.columnCount()):
                item = self.stats_table.item(row, col)
                if item is None:
                    continue
                f = item.font()
                if selected:
                    if band in selected:
                        f.setBold(True)
                        item.setFont(f)
                        item.setForeground(QBrush(active))
                    else:
                        f.setBold(False)
                        item.setFont(f)
                        item.setForeground(QBrush(faint))
                elif focused is None:
                    f.setBold(False)
                    item.setFont(f)
                    item.setForeground(QBrush(normal))
                elif band == focused:
                    f.setBold(True)
                    item.setFont(f)
                    item.setForeground(QBrush(active))
                else:
                    f.setBold(False)
                    item.setFont(f)
                    item.setForeground(QBrush(faint))
        self._update_legend_emphasis()
        self._update_minmax_labels()

    def _fit_table_width(self):
        # Make table width tightly match actual column content (no stretched empty area).
        self.stats_table.resizeColumnsToContents()
        width = self.stats_table.frameWidth() * 2
        for i in range(self.stats_table.columnCount()):
            width += self.stats_table.columnWidth(i)
        width += 6  # small breathing room for grid/borders
        width = max(150, min(360, int(width)))
        self.stats_table.setFixedWidth(width)
        if hasattr(self, "_right_panel") and self._right_panel is not None:
            self._right_panel.setFixedWidth(width)

    def _fit_table_rows_to_height(self):
        rows = self.stats_table.rowCount()
        if rows <= 0:
            return
        header_h = self.stats_table.horizontalHeader().height()
        frame_h = self.stats_table.frameWidth() * 2
        avail = max(0, self.stats_table.height() - header_h - frame_h - 1)
        row_h = max(14, int(avail / rows))
        for r in range(rows):
            self.stats_table.setRowHeight(r, row_h)

    def set_histograms(self, band_histograms, title_text):
        previous_focus = self._focused_band
        previous_selected = set(getattr(self, "_selected_bands", set()))
        self._focused_band = None
        self._reset_plot_data()
        self.stats_table.clearSelection()
        self.stats_table.setRowCount(len(band_histograms))
        self._clear_legend()

        ymax = 1.0
        xmax = 255.0
        overall_min = None
        overall_max = None
        for row, entry in enumerate(band_histograms):
            band = entry.get("band")
            x = np.asarray(entry.get("x", []))
            y = np.asarray(entry.get("y", []))
            if x.size == 0:
                x = np.arange(len(y), dtype=np.float64)
            if y.size == 0:
                continue

            color = QColor(*self._palette[row % len(self._palette)])
            if x.size > 1:
                dx = float(np.median(np.diff(x)))
                if dx <= 0:
                    dx = 1.0
            else:
                dx = 1.0
            x_edges = np.empty(x.size + 1, dtype=np.float64)
            x_edges[:-1] = x - (dx * 0.5)
            x_edges[-1] = x[-1] + (dx * 0.5)

            curve = pg.PlotCurveItem(pen=self._make_pen(color, width=2, alpha=255), antialias=False)
            curve.setData(x=x_edges, y=y, stepMode=True)
            try:
                curve.setClickable(True, width=9)
                curve.sigClicked.connect(lambda *_, b=band: self._on_curve_clicked(b))
            except Exception:
                pass

            self.plot_item.addItem(curve)
            self._curves[band] = curve
            self._curve_colors[band] = color
            self._row_to_band[row] = band
            self._band_to_row[band] = row
            self._add_legend_item(band, color)

            ymax = max(ymax, float(np.max(y)))
            xmax = max(xmax, float(np.max(x)))
            nz = np.flatnonzero(y > 0)
            if nz.size > 0:
                bmin = float(x[int(nz[0])])
                bmax = float(x[int(nz[-1])])
                self._band_minmax[band] = (bmin, bmax)
                overall_min = bmin if overall_min is None else min(overall_min, bmin)
                overall_max = bmax if overall_max is None else max(overall_max, bmax)

            band_item = QTableWidgetItem(str(band))
            mean_item = QTableWidgetItem(self._fmt_num(entry.get("mean")))
            var_item = QTableWidgetItem(self._fmt_num(entry.get("var")))
            std_item = QTableWidgetItem(self._fmt_num(entry.get("std")))
            mean_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            var_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            std_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(row, 0, band_item)
            self.stats_table.setItem(row, 1, mean_item)
            self.stats_table.setItem(row, 2, var_item)
            self.stats_table.setItem(row, 3, std_item)

        self.legend_layout.addStretch()
        self._overall_minmax = (overall_min, overall_max)
        self._fit_table_width()
        self._fit_table_rows_to_height()
        title_color = getattr(self, "_axis_color_hex", "#2f3a4a")
        self.plot_item.setTitle(f"<span style='color:{title_color}; font-weight:600;'>{title_text}</span>")
        self._default_x_range = (0.0, xmax * 1.01)
        self._default_y_range = (0.0, ymax * 1.05)
        self._max_zoom_level = int(max(1, min(24, math.ceil(math.log2(max(1.0, self._default_y_range[1]))))))
        self._zoom_level = min(self._zoom_level, self._max_zoom_level)
        self._x_scroll_user_override = False
        self._x_scroll_pos = 0.5
        self._y_scroll_user_override = False
        self._y_scroll_pos = 0.0
        self._apply_zoom_view()
        self._update_table_emphasis()
        self._apply_plot_ratio()

        remaining_selected = {b for b in previous_selected if b in self._curves}
        if remaining_selected:
            self._selected_bands = remaining_selected
            self._focused_band = None
            self._apply_selection_visibility()
            self._x_scroll_user_override = False
            self._x_scroll_pos = 0.5
            self._y_scroll_user_override = False
            self._y_scroll_pos = 0.0
            self._apply_zoom_view()
            self._update_table_emphasis()
            self.focusCleared.emit()
        elif previous_focus in self._curves:
            self.focus_band(previous_focus)
        else:
            self.clear_focus()

    def focus_band(self, band):
        if band not in self._curves:
            self.clear_focus()
            return

        self._selected_bands = set()
        self._focused_band = band
        self._apply_selection_visibility()
        self._x_scroll_user_override = False
        self._x_scroll_pos = 0.5
        self._y_scroll_user_override = False
        self._y_scroll_pos = 0.0
        self._apply_zoom_view()
        self._update_table_emphasis()
        self.bandFocused.emit(band)

    def clear_focus(self):
        self._selected_bands = set()
        self._focused_band = None
        self._apply_selection_visibility()
        self._x_scroll_user_override = False
        self._x_scroll_pos = 0.5
        self._y_scroll_user_override = False
        self._y_scroll_pos = 0.0
        self._apply_zoom_view()
        self._update_table_emphasis()
        self.focusCleared.emit()

    def _on_table_clicked(self, row, _column):
        band = self._row_to_band.get(row)
        if band is None:
            return
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            if band in self._selected_bands:
                self._selected_bands.discard(band)
            else:
                self._selected_bands.add(band)
            self._focused_band = None
            self._apply_selection_visibility()
            self._x_scroll_user_override = False
            self._x_scroll_pos = 0.5
            self._y_scroll_user_override = False
            self._y_scroll_pos = 0.0
            self._apply_zoom_view()
            self._update_table_emphasis()
            if len(self._selected_bands) == 0:
                self.focusCleared.emit()
        else:
            if self._focused_band == band and not self._selected_bands:
                self.clear_focus()
            else:
                self.focus_band(band)

    def _on_curve_clicked(self, band):
        ctrl = bool(QApplication.keyboardModifiers() & Qt.ControlModifier)
        if ctrl:
            if band in self._selected_bands:
                self._selected_bands.discard(band)
            else:
                self._selected_bands.add(band)
            self._focused_band = None
            self._apply_selection_visibility()
            self._x_scroll_user_override = False
            self._x_scroll_pos = 0.5
            self._y_scroll_user_override = False
            self._y_scroll_pos = 0.0
            self._apply_zoom_view()
            self._update_table_emphasis()
            if len(self._selected_bands) == 0:
                self.focusCleared.emit()
        else:
            if self._focused_band == band and not self._selected_bands:
                self.clear_focus()
            else:
                self.focus_band(band)

    def _band_key_to_index(self, band):
        if isinstance(band, int):
            return band
        if isinstance(band, str):
            m = re.search(r"(\d+)$", band)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return band

    def _sorted_band_refs(self, bands):
        refs = [self._band_key_to_index(b) for b in bands]
        return sorted(refs, key=lambda v: (0, int(v)) if isinstance(v, int) else (1, str(v)))

    def _build_band_stats(self, payload):
        stats = {}
        for entry in payload:
            band = entry.get("band")
            band_idx = self._band_key_to_index(band)
            y = np.asarray(entry.get("y", []))
            total = float(y.sum()) if y.size else 0.0
            black_pct = (float(y[0]) / total * 100.0) if total > 0 else 0.0
            sat_pct = (float(y[-1]) / total * 100.0) if total > 0 else 0.0
            bmin, bmax = self._band_minmax.get(band, (None, None))
            stats[band_idx] = {
                "mean": entry.get("mean"),
                "std":  entry.get("std"),
                "min":  bmin,
                "max":  bmax,
                "saturated_pct": sat_pct,
                "black_pct": black_pct,
            }
        return stats

    def _emit_histogram_state(self, payload):
        if not _HAS_IRIS_BUS:
            return
        if not payload:
            return

        display_mode = "frame_range" if str(self.frame_mode).lower().startswith("range") else "single_frame"
        visible = None
        selected = set(getattr(self, "_selected_bands", set()))
        if selected:
            visible = self._sorted_band_refs(selected)
        else:
            visible = self._sorted_band_refs(self._curves.keys())

        overall_min, overall_max = getattr(self, "_overall_minmax", (None, None))
        frame_min = overall_min if overall_min is not None else self.min_val
        frame_max = overall_max if overall_max is not None else self.max_val

        payload_stats = self._build_band_stats(payload)

        try:
            bus.emit(AppEvent(
                EventType.HISTOGRAM_UPDATED,
                {
                    "folder":        self._iris_folder,
                    "frame_index":   int(self._iris_frame_index),
                    "display_mode":  display_mode,
                    "range_start":   int(self.start_frame),
                    "range_end":     int(self.end_frame),
                    "axis_min":      float(self.min_val),
                    "axis_max":      float(self.max_val),
                    "frame_min":     float(frame_min) if frame_min is not None else 0.0,
                    "frame_max":     float(frame_max) if frame_max is not None else 0.0,
                    "visible_bands": visible,
                    "band_stats":    payload_stats,
                },
                source="histogram_viewer"
            ))
        except Exception:
            pass

    def update_histogram(self, band_frames, current_frame_index, frame_mode, start_frame=None, end_frame=None, smooth=True, ignore_extremes=True, folder: str = ""):
        self._reset_plot_data()
        if not band_frames:
            self.stats_table.setRowCount(0)
            self._fit_table_width()
            self._apply_plot_ratio()
            self.hist_progress.hide()
            return

        self.band_items = list(band_frames.items())
        self.frame_mode = frame_mode
        self.start_frame = start_frame if start_frame is not None else current_frame_index
        self.end_frame = end_frame if end_frame is not None else current_frame_index
        self._iris_frame_index = int(current_frame_index)
        if folder:
            self._iris_folder = str(folder)

        if getattr(self, 'worker', None):
            try:
                if hasattr(self.worker, 'stop'):
                    self.worker.stop()
                else:
                    self.worker.quit()
                self.worker.wait(1000)
            except Exception:
                pass

        if getattr(self, 'single_frame_radio', None) and self.single_frame_radio.isChecked():
            requested_mode = 'single'
        elif getattr(self, 'frame_range_radio', None) and self.frame_range_radio.isChecked():
            requested_mode = 'range'
        else:
            requested_mode = str(frame_mode).lower()

        use_range_worker = requested_mode.startswith('range')
        effective_frame_index = int(current_frame_index)

        if not use_range_worker and requested_mode.startswith('single'):
            bf = {}
            for k, v in band_frames.items():
                if hasattr(v, 'get_raw'):
                    try:
                        bf[k] = [v.get_raw(current_frame_index)]
                    except Exception:
                        bf[k] = [np.zeros((1, 1), dtype=np.uint8)]
                else:
                    bf[k] = v
            band_frames = bf
            effective_frame_index = 0

        if use_range_worker:
            sf = int(self.start_frame)
            ef = int(self.end_frame)
            self.worker = RangeHistogramThread(band_frames, sf, ef, parent=self)
            self.worker.bins_ready.connect(self.update_from_bins)
            self.worker.progress.connect(self.hist_progress.setValue)
            self.worker.finished.connect(self._on_range_finished)
            self.worker.error.connect(self._on_histogram_error)
            self.hist_progress.setVisible(True)
            self.hist_progress.setValue(0)
            self._y_display_max = 1.0
            self.worker.start()
            return

        self.hist_progress.show()
        self.hist_progress.setValue(0)
        self.worker = HistogramWorker(
            band_frames, effective_frame_index, frame_mode,
            start_frame, end_frame, ignore_extremes, self
        )
        self.worker.finished.connect(self._on_histogram_finished)
        self.worker.error.connect(self._on_histogram_error)
        self.worker.progress.connect(self.hist_progress.setValue)
        self.worker.start()

    def update_from_bins(self, bins_dict, processed, total):
        try:
            if not bins_dict:
                return

            payload = []
            y_max = 1.0
            range_min = None
            range_max = None
            for key, bins in bins_dict.items():
                b = np.asarray(bins, dtype=np.uint64)
                if b.size == 0 or b.sum() == 0:
                    continue
                nz = np.flatnonzero(b > 0)
                if nz.size > 0:
                    bmin = int(nz[0])
                    bmax = int(nz[-1])
                    range_min = bmin if range_min is None else min(range_min, bmin)
                    range_max = bmax if range_max is None else max(range_max, bmax)
                x = np.arange(len(b), dtype=np.float64)
                mean, var, sd, cnt = self._stats_from_hist(b, x)
                y_max = max(y_max, float(np.max(b)))
                payload.append({
                    "band": key,
                    "x": x,
                    "y": b,
                    "mean": mean if cnt > 0 else None,
                    "var": var if cnt > 0 else None,
                    "std": sd if cnt > 0 else None
                })

            prev = float(getattr(self, '_y_display_max', 1.0))
            self._y_display_max = max(1.0, float(max(y_max, prev * 0.92)))
            mode_str = 'Frames' if self.frame_mode.lower().startswith('range') else 'Single Frame'
            self.set_histograms(payload, f"Histogram - {mode_str} ({processed}/{total})")
            if range_min is not None and range_max is not None:
                self.min_val = int(range_min)
                self.max_val = int(range_max)
                self._set_frame_set_minmax(self.min_val, self.max_val)
                self.minmax_updated.emit(self.min_val, self.max_val)
            self.plot.setYRange(0, self._y_display_max * 1.05, padding=0)

            pct = int((processed / float(max(1, total))) * 100)
            self.hist_progress.setValue(pct)
            if processed >= total:
                self.hist_progress.hide()
            self._emit_histogram_state(payload)
        except Exception as e:
            print(f"update_from_bins error: {e}")

    def _on_histogram_finished(self, data):
        key_to_result = data['results']
        self.min_val = data['min_val']
        self.max_val = data['max_val']
        frame_mode_str = 'Single Frame' if self.frame_mode == 'Single' else f'Frames {self.start_frame+1}-{self.end_frame+1}'

        num_bins = min(max(256, self.max_val + 1), 65536)
        bins = np.arange(0, num_bins + 1, dtype=np.int32)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        payload = []
        plotted_keys = set()

        for key, _frames in self.band_items:
            if key in plotted_keys:
                continue
            plotted_keys.add(key)
            hist_data = key_to_result.get(key, (np.zeros(num_bins, dtype=np.int64), self.min_val, self.max_val, 0, 0.0, 0.0))
            hist, _gmin, _gmax, count, sum_val, sum_sq = hist_data
            if count == 0:
                continue

            if len(hist) < num_bins:
                hist = np.pad(hist, (0, num_bins - len(hist)), mode='constant', constant_values=0)
            elif len(hist) > num_bins:
                hist = hist[:num_bins]

            mean = sum_val / count
            var = max(0.0, (sum_sq / count) - mean ** 2)
            sd = math.sqrt(max(0.0, var))
            payload.append({
                "band": key,
                "x": bin_centers[:len(hist)],
                "y": hist,
                "mean": mean,
                "var": var,
                "std": sd
            })

        self.set_histograms(payload, f"Histogram - {frame_mode_str} (Range: {self.min_val}-{self.max_val})")
        self._set_frame_set_minmax(self.min_val, self.max_val)
        self.minmax_updated.emit(self.min_val, self.max_val)
        self.hist_progress.hide()
        self._emit_histogram_state(payload)
        gc.collect()

    def _on_range_finished(self, data):
        try:
            bins_map = data.get('bins', {})
            minv = int(data.get('min_val', 0))
            maxv = int(data.get('max_val', 255))
            self.min_val = minv
            self.max_val = maxv
            total = (self.end_frame - self.start_frame + 1)
            self.update_from_bins({k: v for k, v in bins_map.items()}, processed=total, total=total)
            self._set_frame_set_minmax(self.min_val, self.max_val)
            self.minmax_updated.emit(self.min_val, self.max_val)
            # update_from_bins already emits histogram state
        except Exception as e:
            print(f"_on_range_finished error: {e}")
        finally:
            try:
                self.hist_progress.hide()
            except Exception:
                pass
            gc.collect()

    def _on_histogram_error(self, err):
        print(f"Histogram computation error: {err}")
        self._reset_plot_data()
        self.plot_item.setTitle("Error computing histogram")
        self.stats_table.setRowCount(0)
        self.hist_progress.hide()

    def clear(self):
        if getattr(self, 'worker', None):
            try:
                if hasattr(self.worker, 'stop'):
                    self.worker.stop()
                else:
                    self.worker.quit()
                self.worker.wait(1000)
            except Exception:
                pass
        self._reset_plot_data()
        self.stats_table.setRowCount(0)
        self.hist_progress.hide()
        self.min_val = 255
        self.max_val = 0

class PixelInfoBox(QWidget):
    def __init__(self, parent=None, matrix_size_var=None):
        super().__init__(parent)
        self.matrix_size_var = matrix_size_var
        self.measure_mode = False
        self.calculate_mode = False
        # Floating/movable state
        self._is_floating = False
        self._drag_pos = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        control_layout.addWidget(QLabel("Matrix Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["3", "5", "7", "9"])
        try:
            self.size_combo.setCurrentText(str(matrix_size_var.value()))
        except Exception:
            pass
        self.size_combo.currentTextChanged.connect(lambda v: self.matrix_size_var.setValue(int(v)))
        control_layout.addWidget(self.size_combo)

        layout.addWidget(QLabel("Pixel Info"))
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFontFamily("Consolas")
        self.info_text.setFontPointSize(9)
        self.info_text.setFixedHeight(250)
        layout.addWidget(self.info_text)

        self.last_x = None
        self.last_y = None
        self.last_values = None
        self.last_is_rgb = False
        self.last_dn_value = None
        self.last_oa = 0
        self.last_ob = 0
        self.last_ab = 0
        self.last_calc_mean = None
        self.last_calc_variance = None
        self.last_calc_std = None
        self.last_calc_min = None
        self.last_calc_max = None
        self.last_calc_count = None

    # --- Floating control API ---
    def make_floating(self, start_pos=None):
        try:
            self._is_floating = True
            self.setParent(None)
            flags = Qt.Window | Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
            self.setWindowFlags(flags)
            # Optional: give a small border so users can see it
            self.setStyleSheet(self.styleSheet() + "QWidget{border:1px solid rgba(200,200,200,0.6); background: rgba(30,30,30,0.92); color: white;}")
            if start_pos:
                try:
                    self.move(start_pos)
                except Exception:
                    pass
            self.show()
            self.raise_()
        except Exception as e:
            print("make_floating error:", e)

    def make_embedded(self, parent_panel, insert_index=None):
        try:
            self._is_floating = False
            self.hide()
            self.setParent(parent_panel)
            self.setWindowFlags(Qt.Widget)
            # reset any floating styles (optional)
            self.setStyleSheet("")
            # attach back to layout if available
            layout = getattr(parent_panel, "layout", None)
            if callable(layout):
                # if parent_panel.layout() exists use that
                pl = parent_panel.layout()
                if pl is not None:
                    if insert_index is not None and 0 <= insert_index < pl.count():
                        pl.insertWidget(insert_index, self)
                    else:
                        pl.addWidget(self)
            else:
                try:
                    parent_layout = parent_panel.layout()
                    if parent_layout is not None:
                        if insert_index is not None and 0 <= insert_index < parent_layout.count():
                            parent_layout.insertWidget(insert_index, self)
                        else:
                            parent_layout.addWidget(self)
                except Exception:
                    pass
            self.show()
        except Exception as e:
            print("make_embedded error:", e)

    # --- Mouse handlers to drag when floating ---
    def mousePressEvent(self, event):
        if self._is_floating and event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_floating and self._drag_pos is not None:
            new_pos = event.globalPos() - self._drag_pos
            # keep widget inside screen roughly
            try:
                self.move(new_pos)
            except Exception:
                pass
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def build_info_text(self):
        self.info_text.clear()
        no_values = self.last_values is None
        calc_available = (
            self.last_calc_mean is not None and
            self.last_calc_variance is not None and
            self.last_calc_std is not None
        )
        if no_values and not (self.measure_mode or self.calculate_mode or getattr(self, 'last_force_measure', False) or calc_available):
            return

        app = QApplication.instance()
        if app is not None:
            is_dark = app.palette().color(QPalette.Window).lightness() < 128
        else:
            is_dark = self.palette().color(QPalette.Window).lightness() < 128

        if is_dark:
            highlight_bg = "#5400C2"  # Dodgerblue for dark mode (white text visible)
            highlight_text = "white"
        else:
            highlight_bg = "yellow"  # Yellow for light mode (black text visible)
            highlight_text = "black"

        # Pixel matrix header
        try:
            size = int(self.matrix_size_var.value())
            if size <= 0:
                size = 1
        except Exception:
            size = 1
        self.info_text.append(f"\n{size}x{size} Pixel Matrix:")

        # Helper to format a single value (fixed width)
        def _fmt_val(v):
            try:
                return f"{int(v):1d}"   # single-digit or multi-digit without extra padding
            except Exception:
                try:
                    return f"{float(v):.0f}"
                except Exception:
                    return "0"

        # Render matrix
        values = self.last_values
        is_rgb = self.last_is_rgb
        if isinstance(values, np.ndarray):
            center_i = size // 2
            center_j = size // 2

            if is_rgb and values.ndim == 3:
                # Compact HTML table: small font, minimal padding, preserve spaces with white-space:pre
                rows = []
                for i in range(size):
                    r_cells = []
                    g_cells = []
                    b_cells = []
                    for j in range(size):
                        try:
                            r_val = values[i, j, 0] if i < values.shape[0] and j < values.shape[1] else 0
                            g_val = values[i, j, 1] if i < values.shape[0] and j < values.shape[1] else 0
                            b_val = values[i, j, 2] if i < values.shape[0] and j < values.shape[1] else 0
                        except Exception:
                            r_val = g_val = b_val = 0

                        r_fmt = _fmt_val(r_val)
                        g_fmt = _fmt_val(g_val)
                        b_fmt = _fmt_val(b_val)

                        if i == center_i and j == center_j:
                            r_fmt = f"<span style='background-color:{highlight_bg}; color:{highlight_text}'>{r_fmt}</span>"
                            g_fmt = f"<span style='background-color:{highlight_bg}; color:{highlight_text}'>{g_fmt}</span>"
                            b_fmt = f"<span style='background-color:{highlight_bg}; color:{highlight_text}'>{b_fmt}</span>"

                        r_cells.append(r_fmt)
                        g_cells.append(g_fmt)
                        b_cells.append(b_fmt)

                    # No extra spaces between numbers except single space; minimal padding between columns
                    r_str = " ".join(r_cells)
                    g_str = " ".join(g_cells)
                    b_str = " ".join(b_cells)

                    rows.append(
                        "<tr>"
                        "<td style='font-family:monospace; font-size:11px; white-space:pre; padding:0 6px 0 0;'>"
                        f"R:[{r_str}]</td>"
                        "<td style='font-family:monospace; font-size:11px; white-space:pre; padding:0 6px 0 0;'>"
                        f"G:[{g_str}]</td>"
                        "<td style='font-family:monospace; font-size:11px; white-space:pre; padding:0 0 0 0;'>"
                        f"B:[{b_str}]</td>"
                        "</tr>"
                    )

                matrix_html = (
                    "<table cellspacing='0' cellpadding='0' style='border-collapse:collapse; margin:0;'>"
                    "<tbody>"
                    + "".join(rows) +
                    "</tbody></table>"
                )
                self.info_text.append(matrix_html)

            else:
                # Non-RGB: keep previous compact inline rendering
                rows_html = []
                for i in range(size):
                    indent_html = "<span style='display:inline-block; width:3ch'></span>" if i != center_i else ""
                    cells = []
                    for j in range(size):
                        try:
                            val = values[i, j] if i < values.shape[0] and j < values.shape[1] else 0
                        except Exception:
                            val = 0

                        fmt = _fmt_val(val)
                        if i == center_i and j == center_j:
                            cells.append(f"<span style='display:inline-block; width:2ch; text-align:right; background-color:{highlight_bg}; color:{highlight_text}'>{fmt}</span>")
                        else:
                            cells.append(f"<span style='display:inline-block; width:2ch; text-align:right'>{fmt}</span>")

                    rows_html.append(
                        f"<div style='font-family:monospace; font-size:11px; white-space:nowrap; margin:0;'>"
                        f"{indent_html}[{' '.join(cells)}]</div>"
                    )

                matrix_html = "<div style='font-family:monospace;'>" + "".join(rows_html) + "</div>"
                self.info_text.append(matrix_html)

            try:
                self.info_text.repaint()
            except Exception:
                try:
                    self.info_text.update()
                except Exception:
                    pass
        else:
            self.info_text.append("Value: Unknown")
            print("No valid pixel data provided")

        # Geolocation (unchanged)
        try:
            if self.last_dn_value:
                if isinstance(self.last_dn_value, tuple) and len(self.last_dn_value) >= 3:
                    lat, lon, band_idx = self.last_dn_value[0], self.last_dn_value[1], self.last_dn_value[2]
                    self.info_text.append("\nGeolocation:")
                    self.info_text.append(f"  Lat: {lat:.8f}")
                    self.info_text.append(f"  Lon: {lon:.8f}")
                    self.last_geo = (lat, lon)
                elif isinstance(self.last_dn_value, dict):
                    lat = self.last_dn_value.get('lat')
                    lon = self.last_dn_value.get('lon')
                    if lat is not None and lon is not None:
                        self.info_text.append("\nGeolocation:")
                        self.info_text.append(f"  Lat: {lat:.8f}")
                        self.info_text.append(f"  Lon: {lon:.8f}")
                        self.last_geo = (lat, lon)
        except Exception:
            pass

        # Append measurements if in measure mode — or when a measurement was
        # explicitly forced by another component (fallback for Raw mode).
        if self.measure_mode or getattr(self, 'last_force_measure', False):
            try:
                self.info_text.append(f"\n({int(self.last_oa)},{int(self.last_ob)},{int(self.last_ab)})")
            except Exception:
                pass
        if self.calculate_mode and calc_available:
            try:
                self.info_text.append("\nCalculated:")
                # (Region WxH removed — only pixel count will be shown)
                self.info_text.append(f"  Mean: {self.last_calc_mean:.6f}")
                self.info_text.append(f"  Variance: {self.last_calc_variance:.6f}")
                self.info_text.append(f"  Standard deviation: {self.last_calc_std:.6f}")
            except Exception:
                pass
        # Also show extended statistics if available (min/max/count)
        if calc_available:
            try:
                if self.last_calc_min is not None:
                    self.info_text.append(f"  Min: {self.last_calc_min:.6f}")
                if self.last_calc_max is not None:
                    self.info_text.append(f"  Max: {self.last_calc_max:.6f}")
                if self.last_calc_count is not None:
                    self.info_text.append(f"  Pixels: {int(self.last_calc_count)}")
            except Exception:
                pass

        # Ensure visibility if floating
        if self._is_floating:
            self.show()
            self.raise_()

    def update_info(self, x, y, values, is_rgb=False, dn_value=None):
        # Clear any previously-forced measurement when normal pixel info is updated
        try:
            self.last_force_measure = False
        except Exception:
            pass
        self.last_x = x
        self.last_y = y
        self.last_values = values.copy() if isinstance(values, np.ndarray) else values
        self.last_is_rgb = is_rgb
        self.last_dn_value = dn_value
        self.build_info_text()

    def set_measure_mode(self, enabled):
        self.measure_mode = enabled
        self.calculate_mode = False
        if not enabled:
            self.last_oa = 0
            self.last_ob = 0
            self.last_ab = 0
            try:
                self.last_force_measure = False
            except Exception:
                pass
            self.build_info_text()

    def set_interaction_mode(self, mode):
        mode = str(mode).lower()
        # Support combined 'both' mode
        self.measure_mode = (mode == "measure" or mode == "both")
        self.calculate_mode = (mode == "calculate" or mode == "both")
        if not self.measure_mode:
            self.last_oa = 0
            self.last_ob = 0
            self.last_ab = 0
            try:
                self.last_force_measure = False
            except Exception:
                pass
        if not self.calculate_mode:
            self.last_calc_mean = None
            self.last_calc_variance = None
            self.last_calc_std = None
            self.last_calc_min = None
            self.last_calc_max = None
            self.last_calc_count = None
        self.build_info_text()

    def update_measurements(self, oa, ob, ab):
        self.last_oa = oa
        self.last_ob = ob
        self.last_ab = ab
        self.build_info_text()

    def update_calculations(self, mean, variance, std, vmin=None, vmax=None, count=None):
        try:
            self.last_calc_mean = float(mean)
        except Exception:
            self.last_calc_mean = None
        try:
            self.last_calc_variance = float(variance)
        except Exception:
            self.last_calc_variance = None
        try:
            self.last_calc_std = float(std)
        except Exception:
            self.last_calc_std = None
        try:
            self.last_calc_min = float(vmin) if vmin is not None else None
        except Exception:
            self.last_calc_min = None
        try:
            self.last_calc_max = float(vmax) if vmax is not None else None
        except Exception:
            self.last_calc_max = None
        # Allow `count` to be either an int or a tuple (count, width, height)
        try:
            if isinstance(count, (tuple, list)) and len(count) >= 1:
                self.last_calc_count = int(count[0])
            else:
                self.last_calc_count = int(count) if count is not None else None
        except Exception:
            self.last_calc_count = None
        # Also store region width/height if caller attached them via count tuple
        try:
            if isinstance(count, (tuple, list)) and len(count) >= 3:
                _, w, h = count
                self.last_calc_region = (int(w), int(h))
            else:
                if not hasattr(self, 'last_calc_region'):
                    self.last_calc_region = None
        except Exception:
            self.last_calc_region = None
        self.build_info_text()

    def force_show_measurements(self, oa, ob, ab):
        """Force the measurement tuple to appear in the rendered info box.
        This is used as a fallback when the normal measure-mode boolean may not
        be synchronised between widgets (e.g. RawViewer -> main app path).
        The forced measurement persists until the next normal update_info() call.
        """
        try:
            self.last_oa = int(oa)
            self.last_ob = int(ob)
            self.last_ab = int(ab)
            self.last_force_measure = True
            # Ensure the visual uses the same formatting path as measure_mode
            self.build_info_text()
        except Exception:
            pass

class CustomTabBar(QTabBar):
    layoutChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(False)
        self.setMovable(True)
        self.setDrawBase(False)
        # How many pixels we have manually scrolled the strip
        self._scroll_offset = 0

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._hide_native_scroll_buttons()
        # Clamp offset so we never over-scroll after a resize
        self._clamp_offset()
        self.layoutChanged.emit()

    def tabLayoutChange(self):
        super().tabLayoutChange()
        self._hide_native_scroll_buttons()
        self._clamp_offset()
        self.layoutChanged.emit()

    # ------------------------------------------------------------------
    # Native button suppression
    # ------------------------------------------------------------------
    def _scroll_button(self, object_name):
        for child in self.findChildren(QToolButton):
            if child.objectName() == object_name:
                return child
        return None

    def _native_scroll_enabled(self, object_name):
        btn = self._scroll_button(object_name)
        if btn is None:
            return None
        return btn.isEnabled()

    def _hide_native_scroll_buttons(self):
        for name in ("ScrollLeftButton", "ScrollRightButton"):
            btn = self._scroll_button(name)
            if btn is None:
                continue
            # Make them transparent instead of hiding completely
            btn.setStyleSheet("background: transparent; border: none; color: transparent;")
            btn.setFixedSize(1, 1)  # Minimal size so they exist but are invisible

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _total_tabs_width(self):
        """Sum of all tab widths."""
        total = 0
        for i in range(self.count()):
            total += self.tabRect(i).width()
        return total

    def _visible_width(self):
        """Usable width of the tab bar (excluding any corner widgets Qt steals)."""
        return self.width()

    def _max_offset(self):
        overflow = self._total_tabs_width() - self._visible_width()
        return max(0, overflow)

    def _clamp_offset(self):
        self._scroll_offset = max(0, min(self._scroll_offset, self._max_offset()))

    # ------------------------------------------------------------------
    # Public API used by MainApp
    # ------------------------------------------------------------------
    @property
    def scroll_offset(self):
        return self._scroll_offset

    def set_scroll_offset(self, offset):
        self._hide_native_scroll_buttons()
        self._scroll_offset = int(offset)
        self._clamp_offset()
        self.update()
        self.layoutChanged.emit()

    def can_scroll_left(self):
        native_state = self._native_scroll_enabled("ScrollLeftButton")
        if native_state is not None:
            return native_state
        return self._scroll_offset > 0

    def can_scroll_right(self):
        native_state = self._native_scroll_enabled("ScrollRightButton")
        if native_state is not None:
            return native_state
        return self._scroll_offset < self._max_offset()

    def _do_scroll(self, delta_px):
        """Scroll by delta_px pixels (positive = right, negative = left)."""
        # Use the native ScrollLeftButton/ScrollRightButton to actually move
        # the tab bar viewport — Qt handles the pixel maths internally.
        if delta_px < 0:
            btn_name = "ScrollLeftButton"
        else:
            btn_name = "ScrollRightButton"

        btn = self._scroll_button(btn_name)
        if btn:
            # Temporarily restore a usable size so Qt accepts the click.
            original_size = btn.size()
            btn.setFixedSize(16, 16)
            btn.show()
            btn.click()
            # Don't hide the button - let it stay visible but styled by CSS
            btn.setFixedSize(original_size)

        # Keep our own offset in sync so can_scroll_* are accurate
        if delta_px < 0:
            self._scroll_offset = max(0, self._scroll_offset - abs(delta_px))
        else:
            self._scroll_offset = min(self._max_offset(),
                                      self._scroll_offset + abs(delta_px))
        self.layoutChanged.emit()

    def scroll_strip_left(self):
        # One "step" = width of first visible tab (feels natural like a browser)
        step = 80
        if self.count() > 0:
            step = max(40, self.tabRect(0).width())
        self._do_scroll(-step)

    def scroll_strip_right(self):
        step = 80
        if self.count() > 0:
            step = max(40, self.tabRect(0).width())
        self._do_scroll(step)


class ParameterDialog(QDialog):
    def __init__(self, parent=None, dataset_params=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Image Parameters")
        layout = QFormLayout()
        self.setLayout(layout)

        default_width = "8448"
        default_raw_height = "384"
        default_tdi_stage = "0"
        default_bitdepth = "10"
        try:
            if parent is not None:
                if hasattr(parent, "width_entry"):
                    default_width = str(parent.width_entry.text())
                if hasattr(parent, "raw_height"):
                    default_raw_height = str(int(getattr(parent, "raw_height", 384) or 384))
                elif hasattr(parent, "height_entry"):
                    default_raw_height = str(parent.height_entry.text())
                if hasattr(parent, "tdi_stage"):
                    default_tdi_stage = str(int(getattr(parent, "tdi_stage", 0) or 0))
                if hasattr(parent, "bitdepth_var"):
                    default_bitdepth = str(parent.bitdepth_var.currentText())
        except Exception:
            pass

        if isinstance(dataset_params, dict):
            try:
                if dataset_params.get("width"):
                    default_width = str(int(dataset_params["width"]))
                if dataset_params.get("raw_height"):
                    default_raw_height = str(int(dataset_params["raw_height"]))
                if "tdi_stage" in dataset_params:
                    default_tdi_stage = str(int(dataset_params["tdi_stage"]))
                if dataset_params.get("bit_depth"):
                    default_bitdepth = str(int(dataset_params["bit_depth"]))
            except Exception:
                pass

        self.width_entry = QLineEdit(default_width)
        layout.addRow("Width:", self.width_entry)

        self.height_entry = QLineEdit(default_raw_height)
        layout.addRow("Height (RegionHeight):", self.height_entry)

        self.tdi_stage_var = QComboBox()
        self.tdi_stage_var.addItems(["0", "2", "4", "8", "16", "32"])
        self.tdi_stage_var.setCurrentText(default_tdi_stage if default_tdi_stage in ["0", "2", "4", "8", "16", "32"] else "0")
        layout.addRow("TDI Stage:", self.tdi_stage_var)

        self.bitdepth_var = QComboBox()
        self.bitdepth_var.addItems(["8", "10", "12", "16"])
        self.bitdepth_var.setCurrentText(default_bitdepth if default_bitdepth in ["8", "10", "12", "16"] else "10")
        layout.addRow("Bit Depth:", self.bitdepth_var)

        self.effective_height_label = QLabel()
        self.effective_height_label.setWordWrap(True)
        layout.addRow("Computed Band Height:", self.effective_height_label)
        self.height_entry.textChanged.connect(self._update_effective_height_hint)
        self.tdi_stage_var.currentTextChanged.connect(self._update_effective_height_hint)
        self._update_effective_height_hint()
        
        buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.setToolTip("Apply parameters")
        ok_btn.clicked.connect(self.accept)
        buttons.addWidget(ok_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Close without changes")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)
        
        layout.addRow(buttons)

    def _update_effective_height_hint(self):
        try:
            raw_height = int(self.height_entry.text())
        except Exception:
            raw_height = 0
        try:
            tdi_stage = int(self.tdi_stage_var.currentText())
        except Exception:
            tdi_stage = 0
        effective_height = raw_height if tdi_stage == 0 else max(1, raw_height // tdi_stage)
        if raw_height <= 0:
            self.effective_height_label.setText("Enter RegionHeight to compute band height.")
        elif tdi_stage == 0:
            self.effective_height_label.setText(f"{effective_height} px (TDI off, no division)")
        else:
            self.effective_height_label.setText(
                f"{effective_height} px ({raw_height} / {tdi_stage}; TDI stages are used to derive band height)"
            )
    
    def get_parameters(self):
        try:
            raw_height = int(self.height_entry.text())
        except Exception:
            raw_height = 384
        tdi_stage = int(self.tdi_stage_var.currentText())
        eff_height = raw_height if tdi_stage == 0 else max(1, raw_height // tdi_stage)
        return {
            "width": self.width_entry.text(),
            "height": str(eff_height),
            "raw_height": raw_height,
            "bit_depth": int(self.bitdepth_var.currentText()),
            "tdi_stage": tdi_stage
        }
