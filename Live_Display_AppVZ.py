import sys
import math
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QFormLayout, QGroupBox,
    QScrollArea, QFileDialog, QMessageBox, QTabWidget
)
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from PIL import Image
import threading
import time
import subprocess
from datetime import datetime
import glob
from help_tab import create_help_tab

class VideoModeHandler(QWidget):
    start_play_signal = pyqtSignal()  # Signal to start playback in GUI thread
    message_signal = pyqtSignal(str, str, str)  # msg_type, title, text
    update_ui_signal = pyqtSignal() 
    def __init__(self, parent=None, filepath=None, width=9344, height=384, fps=1, ExposureTime=29999, Gain=1, TestPattern=0, framesPerTrigger=1):
        super().__init__(parent)
        self.file_path = ""
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.frame_rate = float(fps)
        self.ExposureTime = float(ExposureTime)
        self.Gain = float(Gain)
        self.TestPattern = int(TestPattern)
        self.bytes_per_frame = (self.frame_width * self.frame_height * 10) // 8
        self.buffer_size = 100
        self.running = False
        self.is_paused = False
        self.is_start = False
        self.frame_index = 0
        self.save = False
        self.frame_buffer = []
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_frames)
        self.triggerPath = None
        self.imagePath = self._default_capture_dir()
        self.frame_count = 0
        self.connected = False
        self.start = False
        self.exited = False
        self.region_modes_visible = False
        self.output_dir = None
        self.triggerFile = None
        self.pixel_stretch_enabled = False  # NEW: Flag for pixel stretching
        self.createLogFile()
        self.setup_ui()
        self.start_play_signal.connect(self.play_video)
        self.message_signal.connect(self.show_message)
        self.update_ui_signal.connect(self.update_ui_fields)
        self.load_video()

    def _default_capture_dir(self):
        # Use an OS-appropriate default capture directory.
        if os.name == "nt":
            return os.path.join(os.path.expanduser("~"), "Pictures", "Display X Studio", "Capture")
        return "/opt/KAYA_Instruments/Examples/Vision Point API/Display_live/Capture"

    def _resolve_camera_binary(self):
        # Support both Linux and Windows camera helper binaries.
        cam_dir = os.path.dirname(__file__)
        candidates = []
        if os.name == "nt":
            candidates.extend([
                os.path.join(cam_dir, "Xdlinx_Cam.exe"),
                os.path.join(cam_dir, "Xdlinx_Cam"),
            ])
        else:
            candidates.extend([
                os.path.join(cam_dir, "Xdlinx_Cam"),
                os.path.join(cam_dir, "Xdlinx_Cam.bin"),
            ])
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _update_host_tab_name(self, folder_path):
        try:
            base = os.path.basename(folder_path.rstrip(os.sep)) if folder_path else ""
            if not base:
                return
            main_app = getattr(self, "_main_app", None)
            if main_app is not None and hasattr(main_app, "update_tab_name_for_widget"):
                main_app.update_tab_name_for_widget(self, base)
        except Exception:
            pass

    def show_message(self, msg_type, title, text):
        if msg_type == "warning":
            QMessageBox.warning(self, title, text)
        elif msg_type == "info":
            QMessageBox.information(self, title, text)
        elif msg_type == "critical":
            QMessageBox.critical(self, title, text)

    def createLogFile(self):
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = os.path.join(log_dir, f"output_log_{timestamp}.txt")
        self.log_file = open(self.log_file_path, 'a')

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.left_widget.setFixedWidth(500)
        main_layout.addWidget(self.left_widget)

        form = QFormLayout()

        self.image_entry = QLineEdit(self.imagePath )
        browse_image_btn = QPushButton("Browse")
        browse_image_btn.setToolTip("Choose image output path")
        browse_image_btn.clicked.connect(self.browse_image_path)
        image_hbox = QHBoxLayout()
        image_hbox.addWidget(self.image_entry)
        image_hbox.addWidget(browse_image_btn)
        form.addRow(QLabel("Image Path:"), image_hbox)

        self.width_entry = QLineEdit(str(self.frame_width))
        form.addRow(QLabel("Frame Width:"), self.width_entry)

        self.height_entry = QLineEdit(str(self.frame_height))
        form.addRow(QLabel("Frame Height:"), self.height_entry)

        self.frame_rate_entry = QLineEdit(str(self.frame_rate))
        form.addRow(QLabel("Frame Rate (FPS):"), self.frame_rate_entry)

        self.ExposureTime_entry = QLineEdit(str(self.ExposureTime))
        form.addRow(QLabel("Exposure Time:"), self.ExposureTime_entry)

        self.Gain_entry = QLineEdit(str(self.Gain))
        form.addRow(QLabel("Gain:"), self.Gain_entry)

        self.TestPattern_entry = QLineEdit(str(self.TestPattern))
        form.addRow(QLabel("TestPattern:"), self.TestPattern_entry)

        self.left_layout.addLayout(form)

        self.toggle_region_btn = QPushButton("▶ RegionModes")
        self.toggle_region_btn.setToolTip("Show or hide region mode settings")
        self.toggle_region_btn.clicked.connect(self.toggle_region_modes)
        self.left_layout.addWidget(self.toggle_region_btn)

        self.region_group = QGroupBox("RegionModes")
        region_layout = QFormLayout()
        offset_values = [1906, 2372, 2838, 3304, 3770, 4236, 4702]
        self.region_mode_combos = []
        self.region_offset_entries = []
        for i in range(7):
            mode_combo = QComboBox()
            mode_combo.addItems(["On", "Off"])
            mode_combo.setCurrentText("On")
            self.region_mode_combos.append(mode_combo)
            offset_entry = QLineEdit(str(offset_values[i]))
            self.region_offset_entries.append(offset_entry)
            mode_combo.currentTextChanged.connect(lambda text, entry=offset_entry: self.update_region_offset_state(text, entry))
            self.update_region_offset_state("On", offset_entry)
            region_layout.addRow(QLabel(f"Region{i} Mode:"), mode_combo)
            region_layout.addRow(QLabel(f"Region{i} Offset:"), offset_entry)
        self.region_group.setLayout(region_layout)
        self.region_group.setVisible(False)
        self.left_layout.addWidget(self.region_group)

        self.fit_to_screen_checkbox = QCheckBox("Fit to Screen")
        self.fit_to_screen_checkbox.setChecked(True)
        self.fit_to_screen_checkbox.stateChanged.connect(self.toggle_fit_to_screen)
        self.left_layout.addWidget(self.fit_to_screen_checkbox)

        # NEW: Add Pixel Stretch Checkbox
        self.pixel_stretch_checkbox = QCheckBox("Enable Pixel Stretching")
        self.pixel_stretch_checkbox.setChecked(False)
        self.pixel_stretch_checkbox.stateChanged.connect(self.toggle_pixel_stretch)
        self.left_layout.addWidget(self.pixel_stretch_checkbox)

        button_hbox = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.connect_button.setToolTip("Connect to camera")
        self.connect_button.clicked.connect(self.connectCamera)
        button_hbox.addWidget(self.connect_button)

        self.start_button = QPushButton("Start")
        self.start_button.setToolTip("Start or stop live streaming")
        self.start_button.clicked.connect(self.toggle_start_stop)
        button_hbox.addWidget(self.start_button)

        self.toggle_save_button = QPushButton("Start Save")
        self.toggle_save_button.setToolTip("Start or stop saving incoming frames")
        self.toggle_save_button.clicked.connect(self.toggle_save)
        button_hbox.addWidget(self.toggle_save_button)

        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems([".raw", ".bmp"])
        self.save_format_combo.setCurrentText(".raw")
        button_hbox.addWidget(self.save_format_combo)

        self.exit_button = QPushButton("Exit")
        self.exit_button.setToolTip("Close live display tab")
        self.exit_button.clicked.connect(self.exit_app)
        button_hbox.addWidget(self.exit_button)

        self.left_layout.addLayout(button_hbox)

        self.status_label = QLabel("Frame: 0     |     Brightest: N/A")
        self.left_layout.addWidget(self.status_label)

        self.right_tabs = QTabWidget()
        main_layout.addWidget(self.right_tabs, 1)

        live_view_tab = QWidget()
        live_view_layout = QVBoxLayout()
        live_view_layout.setContentsMargins(0, 0, 0, 0)
        live_view_tab.setLayout(live_view_layout)

        self.right_scroll = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.right_scroll.setWidget(self.image_label)
        self.right_scroll.setWidgetResizable(True)
        live_view_layout.addWidget(self.right_scroll)
        self.right_tabs.addTab(live_view_tab, "Live View")

        try:
            self.help_tab = create_help_tab(main_app=getattr(self, "_main_app", None), mode="live")
            self.right_tabs.addTab(self.help_tab, "Help")
        except Exception as e:
            print(f"Live help tab unavailable: {e}")

        main_layout.setStretch(1, 1)

    def update_region_offset_state(self, text, entry):
        entry.setReadOnly(text == "Off")

    def browse_image_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.imagePath = os.path.normpath(path)
            self.image_entry.setText(path)
            self._update_host_tab_name(path)

    # NEW: Toggle pixel stretching
    def toggle_pixel_stretch(self, state):
        self.pixel_stretch_enabled = (state == Qt.Checked)
        print(f"Pixel Stretching: {'Enabled' if self.pixel_stretch_enabled else 'Disabled'}")
        if self.frame_data is not None:
            self.display_frame()

    def connectCamera(self):
        if not self.connected:
            if hasattr(self, 'file') and self.file:
                self.file.close()
            if hasattr(self, 'fileSave') and self.fileSave:
                self.fileSave.close()
            self.load_video()
            try:
                self.frame_width = int(self.width_entry.text() or self.frame_width)
                self.frame_height = int(self.height_entry.text() or self.frame_height)
                self.frame_rate = float(self.frame_rate_entry.text() or self.frame_rate)
                self.ExposureTime = float(self.ExposureTime_entry.text() or self.ExposureTime)
                self.Gain = float(self.Gain_entry.text() or self.Gain)
                self.TestPattern = int(self.TestPattern_entry.text() or self.TestPattern)
                self.bytes_per_frame = (self.frame_width * self.frame_height * 10) // 8
            except ValueError:
                self.message_signal.emit("warning", "Invalid Input", "Please enter valid values.")
                return
            self.imagePath = self.image_entry.text() or self._default_capture_dir()
            self.imagePath = os.path.normpath(os.path.expanduser(self.imagePath))
            self.image_entry.setText(self.imagePath)
            print(f"Paths are {self.triggerPath} and {self.imagePath}")
            self._update_host_tab_name(self.imagePath)

            region_modes = [combo.currentText() for combo in self.region_mode_combos]
            region_offsets = [entry.text() for entry in self.region_offset_entries]
            mode_map = {"On": "1", "Off": "0"}
            region_modes_numeric = [mode_map.get(m, "0") for m in region_modes]

            print("Connecting camera with the following parameters:")
            print(f"  Frame Width      : {self.frame_width}")
            print(f"  Frame Height     : {self.frame_height}")
            print(f"  Frame Rate       : {self.frame_rate}")
            print(f"  Exposure Time    : {self.ExposureTime}")
            print(f"  Gain             : {self.Gain}")
            print(f"  Test Pattern     : {self.TestPattern}")
            for i, (mode, offset) in enumerate(zip(region_modes_numeric, region_offsets)):
                print(f"  Region {i} Mode   : {mode}")
                print(f"  Region {i} Offset : {offset}")

            frames = 0
            cam_dir = os.path.dirname(__file__)
            binary_path = self._resolve_camera_binary()
            if not os.path.exists(binary_path):
                self.message_signal.emit("critical", "Error", f"Binary not found at {binary_path}. Adjust path in code.")
                return

            command = [
                binary_path,
                str(self.frame_width),
                str(self.frame_height),
                str(self.frame_rate),
                str(self.ExposureTime),
                str(self.Gain),
                str(self.TestPattern),
                str(frames),
            ] + region_modes_numeric + region_offsets

            try:
                self.process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    universal_newlines=True,
                    close_fds=True,
                    cwd=cam_dir
                )
            except Exception as e:
                self.message_signal.emit("critical", "Error", f"Failed to start subprocess: {e}")
                return

            self.output_thread = threading.Thread(target=self.read_output, args=(self.process,))
            self.output_thread.daemon = True
            self.output_thread.start()

            self.err_thread = threading.Thread(target=self.read_err, args=(self.process,))
            self.err_thread.daemon = True
            self.err_thread.start()

            self.connected = True
            self.connect_button.setText("Disconnect")
            self.connect_button.setStyleSheet("background-color: #dc3545;")
        else:
            if self.start:
                self.send_command('s')
                time.sleep(0.5)
            self.send_command('e')
            time.sleep(1)

            start_time = time.time()
            while self.process.poll() is None and time.time() - start_time < 10:
                print("Waiting for process to exit gracefully...")
                time.sleep(0.2)
            
            if self.process.poll() is None:
                print("Process did not exit gracefully, terminating...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("Terminate timed out, killing...")
                    self.process.kill()
                    self.process.wait()
            
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except Exception as e:
                    print(f"Error closing stdin: {e}")
            if self.process.stdout:
                try:
                    self.process.stdout.close()
                except Exception as e:
                    print(f"Error closing stdout: {e}")
            if self.process.stderr:
                try:
                    self.process.stderr.close()
                except Exception as e:
                    print(f"Error closing stderr: {e}")
            
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join(timeout=10)
                if self.output_thread.is_alive():
                    print("Warning: Output thread did not join in time")
            if self.err_thread and self.err_thread.is_alive():
                self.err_thread.join(timeout=10)
                if self.err_thread.is_alive():
                    print("Warning: Err thread did not join in time")
            
            self.process = None
            self.connected = False
            self.save = False
            self.toggle_save_button.setText("Start Save")
            self.toggle_save_button.setStyleSheet("background-color: #28a745;")
            self.frame_count = 0
            self.frame_index = 0
            self.status_label.setText(f"Frame: {self.frame_count}     |     Brightest: {0}")
            self.start = False
            self.frame_height = 384
            self.height_entry.setText(str(self.frame_height))
            self.running = False
            self.start_button.setText("Start")
            self.stop_video()
            self.connect_button.setText("Connect")
            self.connect_button.setStyleSheet("background-color: #007ACC;")
            print("Camera disconnected.")

    def read_err(self, process):
        try:
            lines_iter = iter(process.stderr.readline, "")
            for line in lines_iter:
                if line:
                    print("STDERR: " + line.strip())
        except Exception as e:
            print(f"Error reading stderr: {e}")

    def toggle_region_modes(self):
        self.region_modes_visible = not self.region_modes_visible
        self.region_group.setVisible(self.region_modes_visible)
        self.toggle_region_btn.setText("▼ RegionModes" if self.region_modes_visible else "▶ RegionModes")

    def update_ui_fields(self):
        self.width_entry.setText(str(self.frame_width))
        self.height_entry.setText(str(self.frame_height))
        self.frame_rate_entry.setText(str(self.frame_rate))
        self.ExposureTime_entry.setText(str(self.ExposureTime))
        self.Gain_entry.setText(str(self.Gain))
        self.TestPattern_entry.setText(str(self.TestPattern))

        mode_display = {1: "On", 0: "Off"}
        for i in range(7):
            self.region_mode_combos[i].setCurrentText(mode_display.get(self.region_modes_numeric[i], "Off") if hasattr(self, 'region_modes_numeric') else "Off")
            self.region_offset_entries[i].setText(str(self.region_offsets[i]) if hasattr(self, 'region_offsets') else "0")

    def read_output(self, process):
        try:
            lines_iter = iter(process.stdout.readline, "")
            for line in lines_iter:
                if line:
                    line_strip = line.strip()
                    if "TemperatureInfo" not in line_strip:
                        print(line_strip)
                        if "Good callback stream's buffer handle" in line_strip:
                            try:
                                bytes_written = int(line_strip.split("Bytes Written=")[1].strip())
                                self.bytes_per_frame = bytes_written
                            except (IndexError, ValueError):
                                print("Failed to parse bytes written from stream callback")

                    if "Could not connect to grabber #0" in line_strip:
                        self.message_signal.emit("warning", "Connection Error", "Could not connect to grabber: Restart the hardware")
                        self.connected = False

                    if "Camera #0 was connected successfully" in line_strip:
                        self.message_signal.emit("info", "Info", "Camera connected successfully.")
                        self.connected = True

                    if 'Updated Variables after settings:' in line_strip:
                        vars_found, region_modes, region_offsets = self.parse_updated_variables(lines_iter)
                        self.frame_width = vars_found.get("Width", self.frame_width)
                        self.frame_height = vars_found.get("Height", self.frame_height)
                        self.frame_rate = vars_found.get("AcquisitionFrameRate", self.frame_rate)
                        self.ExposureTime = vars_found.get("ExposureTime", self.ExposureTime)
                        self.Gain = vars_found.get("Gain", self.Gain)
                        self.TestPattern = vars_found.get("TestPattern", self.TestPattern)
                        self.region_modes_numeric = region_modes
                        self.region_offsets = region_offsets
                        self.bytes_per_frame = (self.frame_width * self.frame_height * 10) // 8
                        print("Updated Variables:")
                        print(f"  Frame Width      : {self.frame_width}")
                        print(f"  Frame Height     : {self.frame_height}")
                        print(f"  Frame Rate       : {self.frame_rate}")
                        print(f"  Exposure Time    : {self.ExposureTime}")
                        print(f"  Gain             : {self.Gain}")
                        print(f"  Test Pattern     : {self.TestPattern}")
                        for i, (mode, offset) in enumerate(zip(self.region_modes_numeric, self.region_offsets)):
                            print(f"  Region {i} Mode   : {mode}")
                            print(f"  Region {i} Offset : {offset}")
                        self.update_ui_signal.emit()
                        self.start_play_signal.emit()

                    if 'TemperatureInfo' in line_strip:
                        try:
                            next_line = next(lines_iter).strip()
                            if not self.start:
                                self.log_file.write(f"{next_line} ----- Idle\n")
                            else:
                                self.log_file.write(f"{next_line} ----- Streaming\n")
                            self.log_file.flush()
                        except StopIteration:
                            print("Reached EOF while expecting temperature data.")

                if process.poll() is not None:
                    break
        except Exception as e:
            print(f"Error reading process output: {e}")

    def parse_updated_variables(self, lines_iter):
        vars_found = {}
        region_modes = []
        region_offsets = []
        keys_single = [
            "Width", "Height", "AcquisitionFrameRateMax", "AcquisitionFrameRate",
            "ExposureTime", "Gain", "TDIMode", "TestPattern"
        ]
        region_mode_keys = [f"Region{i}Mode" for i in range(7)]
        region_offset_keys = [f"Region{i}OffsetY" for i in range(7)]
        expected_lines = len(keys_single) + len(region_mode_keys) + len(region_offset_keys)
        count = 0
        while count < expected_lines:
            try:
                line = next(lines_iter).strip()
            except StopIteration:
                break
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                try:
                    if '.' in val or 'e' in val.lower():
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    pass
                if key in keys_single:
                    vars_found[key] = val
                elif key in region_mode_keys:
                    index = region_mode_keys.index(key)
                    while len(region_modes) <= index:
                        region_modes.append(None)
                    region_modes[index] = val
                elif key in region_offset_keys:
                    index = region_offset_keys.index(key)
                    while len(region_offsets) <= index:
                        region_offsets.append(None)
                    region_offsets[index] = val
                count += 1
        return vars_found, region_modes, region_offsets

    def send_command(self, command):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except Exception as e:
                print(f"Error sending command: {e}")

    def toggle_start_stop(self):
        if not self.connected:
            self.message_signal.emit("info", "Start Error", "Please connect the camera first.")
            return
        self.send_command('s')
        self.start = not self.start
        self.start_button.setText("Stop" if self.start else "Start")
        if self.start:
            print("[Stream] Started.")
        else:
            print("[Stream] Stopped.")

    def exit_app(self):
        self.parent().close()

    def load_video(self):
        if self.running:
            self.stop_video()
        if self.file_path:
            try:
                self.file = open(self.file_path, "rb")
                self.fileSave = open(self.file_path, "rb")
                self.frame_count = 0
                self.frame_index = 0
                self.is_paused = False
            except Exception as e:
                self.message_signal.emit("critical", "Error", f"Failed to load video: {e}")

    def toggle_save(self):
        if not self.start:
            self.message_signal.emit("info", "Save Error", "Start the stream before saving.")
            return
        if not self.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(self.imagePath, f"Capture_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)
            self.save = True
            self.frame_index = 0
            self.current_save_format = self.save_format_combo.currentText()
            selected_format = self.current_save_format
            print(f"Started saving as {selected_format} to {self.output_dir}")
            self.toggle_save_button.setText("Stop Save")
            self.toggle_save_button.setStyleSheet("background-color: #dc3545;")
            if selected_format == ".raw":
                threading.Thread(target=self.save_frames, daemon=True).start()
        else:
            self.save = False
            self.frame_index = 0
            self.toggle_save_button.setText("Start Save")
            self.toggle_save_button.setStyleSheet("background-color: #28a745;")
            print(f"Stopped saving at frame {self.frame_index}")

    def save_frames(self):
        frame_size_bytes = self.bytes_per_frame
        offset = self.frame_count * frame_size_bytes
        print(f"frame count: {self.frame_count}")
        self.fileSave.seek(offset)
        while self.save:
            current_pos = self.fileSave.tell()
            self.fileSave.seek(0, os.SEEK_END)
            file_size = self.fileSave.tell()
            self.fileSave.seek(current_pos)
            remaining_bytes = file_size - current_pos
            if remaining_bytes >= self.bytes_per_frame:
                frame_data = self.fileSave.read(self.bytes_per_frame)
                raw_filename = os.path.join(self.output_dir, f"Capture_{self.frame_index:04d}.raw")
                with open(raw_filename, 'wb') as raw_file:
                    raw_file.write(frame_data)
                self.frame_index += 1
            else:
                time.sleep(0.2)
        print(f"[Info] Finished saving {self.frame_index} frames.")

    def play_video(self):
        if not self.file_path or not os.path.exists(self.file_path):
            self.message_signal.emit("info", "No file loaded", "Please load a video file first.")
            return
        try:
            self.frame_width = int(self.width_entry.text() or self.frame_width)
            self.frame_height = int(self.height_entry.text() or self.frame_height)
            self.frame_rate = float(self.frame_rate_entry.text() or self.frame_rate)
            self.bytes_per_frame = (self.frame_width * self.frame_height * 10) // 8
            print(f"Read- Width: {self.frame_width}, Height: {self.frame_height}, Framesize: {self.bytes_per_frame}")
        except ValueError:
            self.message_signal.emit("critical", "Invalid Input", "Please enter valid numeric values for frame dimensions and frame rate.")
            return
        if not self.running:
            self.running = True
            self.is_paused = False
            threading.Thread(target=self.read_frames, daemon=True).start()
            self.update_timer.start(int(1000 / (self.frame_rate + 1)))

    def read_frames(self):
        while self.running:
            if len(self.frame_buffer) < self.buffer_size:
                try:
                    current_pos = self.file.tell()
                    self.file.seek(0, os.SEEK_END)
                    file_size = self.file.tell()
                    self.file.seek(current_pos)
                    remaining_bytes = file_size - current_pos
                    if remaining_bytes >= self.bytes_per_frame:
                        frame_data = self.file.read(self.bytes_per_frame)
                        if len(frame_data) == self.bytes_per_frame:
                            unpacked_frame = self.unpack_10bit_vectorized(frame_data, self.frame_width, self.frame_height)
                            self.frame_buffer.append(unpacked_frame)
                        else:
                            print(f"Warning: Read incomplete frame of size {len(frame_data)} bytes")
                    else:
                        if not self.frame_buffer:
                            time.sleep(0.2)
                except Exception as e:
                    print(f"Error reading frame: {e}")
                    self.message_signal.emit("critical", "Error", f"Error reading frame: {e}")
                    break
            else:
                time.sleep(0.01)

    def update_frames(self):
        if self.frame_buffer:
            self.frame_data = self.frame_buffer.pop(0)
            if self.save and self.current_save_format == ".bmp":
                bmp_filename = os.path.join(self.output_dir, f"frame_{self.frame_index:04d}.bmp")
                threading.Thread(target=self.save_bmp_frame, args=(self.frame_data.copy(), bmp_filename), daemon=True).start()
                self.frame_index += 1
            self.display_frame()

    def save_bmp_frame(self, frame_data, filename):
        try:
            if self.pixel_stretch_enabled:
                frame_data = self.apply_pixel_stretching(frame_data)
            Image.fromarray(frame_data, mode='L').save(filename)
        except Exception as e:
            print(f"Error saving BMP: {e}")

    # NEW: Apply pixel stretching
    def apply_pixel_stretching(self, frame_data):
        min_px = np.min(frame_data)
        max_px = np.max(frame_data)
        if max_px == min_px:
            return frame_data  # Avoid division by zero
        stretched = ((frame_data - min_px) / (max_px - min_px) * 255).astype(np.uint8)
        return stretched

    def display_frame(self):
        if self.frame_data is not None:
            frame_data = self.frame_data
            if self.pixel_stretch_enabled:
                frame_data = self.apply_pixel_stretching(frame_data)
            height, width = frame_data.shape
            qimage = QImage(frame_data.tobytes(), width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            if self.fit_to_screen_checkbox.isChecked():
                pixmap = pixmap.scaled(self.right_scroll.viewport().size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.adjustSize()
            if frame_data.size > 0:
                brightest_pixel = frame_data.max()
            else:
                brightest_pixel = 0
            self.frame_count += 1
            self.status_label.setText(f"Frame: {self.frame_count}     |     Brightest: {brightest_pixel}")

    def toggle_fit_to_screen(self, state):
        if state == Qt.Checked:
            print("Fit to Screen: Enabled")
        else:
            print("Fit to Screen: Disabled")
        if self.frame_data is not None:
            self.display_frame()

    def stop_video(self):
        self.running = False
        if hasattr(self, 'update_timer'):
            self.update_timer.stop()
        if hasattr(self, 'file') and self.file:
            self.file.close()
            self.file = None
        if hasattr(self, 'fileSave') and self.fileSave:
            self.fileSave.close()
            self.fileSave = None
        self.frame_buffer.clear()
        self.image_label.clear()

    def unpack_10bit_vectorized(self, packed_data, width, height):
        total_pixels = width * height
        if len(packed_data) < self.bytes_per_frame:
            print(f"Warning: Insufficient data ({len(packed_data)} bytes, expected {self.bytes_per_frame})")
            return np.zeros((height, width), dtype=np.uint8)
        num_full_groups = len(packed_data) // 5
        data = np.frombuffer(packed_data[:num_full_groups * 5], dtype=np.uint8).reshape(-1, 5)
        expanded_data = np.zeros(data.shape[0], dtype=np.uint64)
        for i in range(5):
            expanded_data += data[:, i].astype(np.uint64) << (8 * i)
        unpacked_pixels = np.zeros(num_full_groups * 4, dtype=np.uint16)
        for i in range(4):
            unpacked_pixels[i::4] = (expanded_data >> (10 * i)) & 0x3FF
        remaining_bytes = len(packed_data) % 5
        if remaining_bytes:
            last_bits = int.from_bytes(packed_data[-remaining_bytes:], 'little')
            extra_pixels = np.array([(last_bits >> (10 * i)) & 0x3FF for i in range((remaining_bytes * 8) // 10)])
            if extra_pixels.size + unpacked_pixels.size > total_pixels:
                extra_pixels = extra_pixels[:total_pixels - unpacked_pixels.size]
            unpacked_pixels = np.concatenate([unpacked_pixels, extra_pixels])
        if unpacked_pixels.size < total_pixels:
            unpacked_pixels = np.pad(unpacked_pixels, (0, total_pixels - unpacked_pixels.size), 'constant')
        unpacked_pixels = np.clip((unpacked_pixels * 255.0 / 1023), 0, 255).astype(np.uint8)
        return unpacked_pixels.reshape((height, width))

class SatelliteDataApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Display")
        self.resize(1000, 600)
        self.video_mode = VideoModeHandler(self)
        self.setCentralWidget(self.video_mode)

    def closeEvent(self, event):
        self.on_close()
        event.accept()

    def on_close(self):
        print("Closing application...")
        try:
            if hasattr(self.video_mode, 'send_command'):
                if self.video_mode.start:
                    self.video_mode.send_command('s')
                    # Remove blocking sleep during shutdown
                    # time.sleep(0.5)
                self.video_mode.send_command('e')
                # Remove blocking sleep during shutdown
                # time.sleep(1)
        except Exception as e:
            print(f"Error during command send: {e}")
        self.video_mode.stop_video()
        if hasattr(self.video_mode, 'log_file'):
            self.video_mode.log_file.close()
        if hasattr(self.video_mode, 'process') and self.video_mode.process:
            if self.video_mode.process.poll() is None:
                try:
                    self.video_mode.process.terminate()
                    self.video_mode.process.wait(timeout=2)
                    if self.video_mode.process.stdin:
                        self.video_mode.process.stdin.close()
                    if self.video_mode.process.stdout:
                        self.video_mode.process.stdout.close()
                    if self.video_mode.process.stderr:
                        self.video_mode.process.stderr.close()
                except subprocess.TimeoutExpired:
                    self.video_mode.process.kill()
                    self.video_mode.process.wait()
                except Exception as e:
                    print(f"Error during process cleanup: {e}")
            if hasattr(self.video_mode, 'output_thread') and self.video_mode.output_thread.is_alive():
                self.video_mode.output_thread.join(timeout=2)
            if hasattr(self.video_mode, 'err_thread') and self.video_mode.err_thread.is_alive():
                self.video_mode.err_thread.join(timeout=2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SatelliteDataApp()
    window.show()
    sys.exit(app.exec_())
