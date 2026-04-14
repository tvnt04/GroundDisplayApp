# Iris Log Code Reference
# Source: XDLX_CamApp_V6.cpp, XDLX_Camera.cpp, XDLX_Stream.cpp, XDLX_Kaya_UI.cpp, XDLX_Kaya_Grabber.cpp
# Purpose: RAG knowledge base — maps every log code to its meaning, source, and diagnostic interpretation

---

## Log Format

Every log line is produced by PRINT_LOG(code, message). The code appears as a bracketed tag:

    [I##]  — Informational. Normal operation milestone.
    [W##]  — Warning. Abnormal but recoverable condition.
    [E##]  — Error. Something failed; check severity to decide action.
    [SYS]  — OS-level signal or system event.

Lines without a code (empty bracket "") are continuation lines for the preceding entry and carry no independent severity.

The log file is written to: <userDirectory>/Capture/KayaProcess.log
At session end the log is flushed, closed, and transferred to the dataset folder via Stream::transferFile().

---

## SECTION 1 — STARTUP & SYSTEM (I01–I03, I81–I86, E01–E03, E44–E45, E52, E99)

### I01  Program started at: <datetime>
Source: XDLX_CamApp_V6.cpp — startMSIApp()
Meaning: Application entry point reached. The timestamp is the wall-clock session start time.
Use in report: Session start time. Always present in a valid log.

### I02  Welcome to Xdlinx Cam App Version-7.0
Source: XDLX_Kaya_UI.cpp — readInput()
Meaning: App banner. Version number embedded in message string (e.g., v6.0, v7.0).
Use in report: Extract app version from this line.

### I03  Camera Initialized...done...
Source: XDLX_Camera.cpp — after SettingsDefault()
Meaning: Default camera parameter block applied successfully.
Use in report: Confirms camera init phase completed.

### I81  System Update Information
Source: XDLX_CamApp_V6.cpp — getSystemUpdate()
Meaning: Host system telemetry block starts here.

### I82  Disk free space: X.XX GB
Source: XDLX_CamApp_V6.cpp — getSystemUpdate() via statvfs()
Meaning: Available disk at session start. Value also stored in latestTelemetry.storageInfo.storageStatus.
Use in report: Flag if low (< 5 GB for large captures).

### I83  Memory available: X.XX GB
Source: XDLX_CamApp_V6.cpp — getSystemUpdate() via /proc/meminfo MemAvailable
Meaning: Free RAM at session start.
Use in report: Cross-check against I64/I65 DMA allocation lines.

### I84  CPU Model: <string>
Source: XDLX_CamApp_V6.cpp — getSystemUpdate() via /proc/cpuinfo
Meaning: Host processor model name.
Use in report: Platform identification (i7-13700K = dev host, Atom x6425E = embedded payload).

### I85  CPU Cores: <n>
Source: XDLX_CamApp_V6.cpp — getSystemUpdate()
Meaning: Logical processor count.

### I86  Duration since System Up: <H>h <M>m <S>s
Source: XDLX_CamApp_V6.cpp — getSystemUpdate() via /proc/uptime
Meaning: Time since last OS boot at session start.
Use in report: Useful context; a very fresh boot (< 1 min) can explain slow PCIe enumeration.

### I87  Device Core Temperature: <°C>
Source: XDLX_CamApp_V6.cpp — appears twice: once at session start, once at session end
Meaning: PCIe grabber board core temperature read via KYFG_GetGrabberValueInt("DeviceTemperature").
Use in report: Report start and end values. Flag if > MAX_TEMP threshold.

### I91  Telemetry initialized successfully  /  Exiting...
Source: XDLX_CamApp_V6.cpp
Meaning (first): Telemetry subsystem started.
Meaning (second): Normal program exit reached. Session completed cleanly.

### E01  Library initialization failed
Source: XDLX_CamApp_V6.cpp — KYFGLib_Initialize() returned non-OK
Meaning: KAYA frame grabber library could not be initialised. Fatal — no capture follows.
Diagnosis: Check KYFGLib installation and driver version.

### E02  Invalid Time Difference. Capture Time has elapsed. Exiting...
Source: XDLX_CamApp_V6.cpp — after KayaUI.readInput()
Meaning: The UTC trigger time in the parameter file is so far in the past that even after W01 recovery the timing is unusable. Fatal exit.
Diagnosis: Parameter file date is severely stale. Update date/time field before mission.

### E03  Invalid Total Frames: <n>. It should be greater than 0.
Source: XDLX_CamApp_V6.cpp — after floor(G_FPS * G_Duration)
Meaning: Computed frame count is zero or negative. Caused by FPS=0, Duration=0, or both.
Diagnosis: Check FPS and Duration parameters in the parameter file.

### E44  Failed to get disk space info
Source: XDLX_CamApp_V6.cpp — statvfs() returned error
Meaning: Could not read filesystem stats. Non-fatal; disk free will be unknown.

### E45  Failed to get Memory info
Source: XDLX_CamApp_V6.cpp — /proc/meminfo parse failed
Meaning: Could not read available RAM. Non-fatal.

### E52  Grabber is not connected after <N> sec of grace time. Exiting...
Source: XDLX_CamApp_V6.cpp — CheckGrabberStatus() watchdog thread
Meaning: The grabber never reported connected status (grabberStatus != 2) within the 10-second grace window after startup. Fatal exit.
Diagnosis: PCIe board not powered, not seated, or driver not loaded.

### E99  AutoConnect timed out after <N> seconds
Source: XDLX_CamApp_V6.cpp — RunAutoConnectWithTimeout()
Meaning: Grabber AutoConnect took longer than the configured timeout. Fatal.
Diagnosis: Hardware not responding. Check PCIe slot, cable, and camera power-on timing.

### E101  Wait interrupted due to temperature threshold exceeded: <°C>
Source: XDLX_CamApp_V6.cpp — temperature watchdog during trigger wait
Meaning: Core temperature exceeded MAX_TEMP before capture started. Capture aborted.
Diagnosis: Thermal management issue. Allow cooldown before next attempt.

### E102  Payload temperature exceeded threshold: <°C>
Source: XDLX_CamApp_V6.cpp — post-capture temperature check
Meaning: Temperature exceeded threshold after capture completed. Sets g_temp_exceeded flag, which triggers payload-off command in payload_app.cc.
Note: Capture data is intact; only the post-capture power-off path fires.

---

## SECTION 2 — PARAMETER FILE & TIMING (I07, I28, I25–I27, W01, E4/E05/E54)

### I07  Arguments received from parameter file: <raw string>
Source: XDLX_Kaya_UI.cpp — readInput()
Meaning: The raw 14-field parameter line exactly as read from the .txt param file before any processing.
Use in report: Always log this verbatim. Compare against I28 to detect auto-correction.
Format: OrbitID TaskID JsonID Date Time Duration BandSelection TDI FPS ExposureTime Gain XShift Binning TDIYShift

### I28  Argument Processed[14]: <processed string>
Source: XDLX_Kaya_UI.cpp — ProcessInput() / convertUI2Arg()
Meaning: The 14 arguments after firmware parsing and any date/time correction. If I07 date was stale, I28 shows the firmware-corrected date and time.
Use in report: This is the ground truth for what the camera actually used. Always compare I07 vs I28.
Companion check: If I07 date != I28 date, W01 fired and the date was auto-corrected.

### I25  UTC Trigger Time: <datetime>
Source: XDLX_Kaya_UI.cpp / XDLX_Stream.cpp — trigger wait logic
Meaning: The parsed trigger timestamp the firmware computed from the parameter file or from system time after W01 correction.

### I26  System Time Now: <datetime>
Source: XDLX_Kaya_UI.cpp / XDLX_Stream.cpp
Meaning: Wall-clock time at the moment the trigger check ran.

### I27  Waiting Time = <N> msec
Source: XDLX_Kaya_UI.cpp / XDLX_Stream.cpp
Meaning: Calculated wait in milliseconds before the camera starts. Positive = future trigger. Values > 0 and < 60000 are normal.
Use in report: Flag if = 5000 (exactly 5000 ms means W01 fired and fell back to ADDED_DELAY default).

### W01  Time Difference: <N> msec is out of range. setting wait time to 5000 msec for default capture.
Source: XDLX_Kaya_UI.cpp
Meaning: The delta between UTC trigger time and system time is outside MIN_TIME_TH_MSEC…MAX_TIME_TH_MSEC. Firmware falls back to 5-second wait. The firmware then re-derives a corrected timestamp (visible as I28).
Root cause: Parameter file date/time not updated before mission. Stale date causes a negative delta of days/months.
Severity: Warning. Capture proceeds correctly via I28 correction. No data lost.
Action: Update parameter file date/time before every mission.

### E4  Unable to open ParamFile: <path>
Source: XDLX_Kaya_UI.cpp
Meaning: The parameter file path does not exist or is not readable. Fatal — no parameters loaded.

### E05  Invalid Time Difference: <N> msec. It should be between MIN and MAX.
Source: XDLX_Kaya_UI.cpp (early path, before W01 fallback path was added)
Meaning: Same condition as W01 but on an older code path. Session exits.

### E54  Invalid Time Difference: <N> msec. It should be between MIN and MAX.
Source: XDLX_Kaya_UI.cpp (newer path)
Meaning: Time delta outside range and W01 fallback not triggered. Fatal.

---

## SECTION 3 — GRABBER (I04–I06, I92–I99, E55–E59, I78–I79, E42)

### I04  Number of KAYA PCI devices found: <n>
Source: XDLX_Kaya_Grabber.cpp — DisplayList()
Meaning: Total physical + virtual KAYA PCIe grabber boards detected after KY_DeviceScan().

### I05  Grabber[i]: <name> on PCI slot {B:S:F}: Protocol 0x<hex> Generation <n>
Source: XDLX_Kaya_Grabber.cpp — DisplayList()
Meaning: Per-device info for each detected grabber.
Use in report: Extract grabber name, PCI slot, and generation for the Mission Identity section.

### I06  Autoconnect Initiated...
Source: XDLX_Kaya_Grabber.cpp — AutoConnect()
Meaning: Grabber auto-connection starting. Timestamp this line; compute delta to I96 to get connection duration.

### I94  Grabber[i]: <name> on PCI slot ... (duplicate of I05 format inside ConnectByID)
Source: XDLX_Kaya_Grabber.cpp — ConnectByID()
Meaning: Info printout for the specific grabber being connected.

### I95  Selected device #<n> is not a grabber
Source: XDLX_Kaya_Grabber.cpp — ConnectByID()
Meaning: The device at index n does not have the KY_DEVICE_STREAM_GRABBER flag. Skipped.

### I96  Good connection to grabber #<n>, FgHandles=0x<hex>
Source: XDLX_Kaya_Grabber.cpp — ConnectByID() after KYFG_Open()
Meaning: Grabber opened successfully. Sets grabberStatus = 2 (connected).
Use in report: Timestamp this line. Delta from I06 = grabber connection duration.
Normal duration: < 30 seconds. Flag if > 60 seconds as WARNING.

### I97  KYDeviceEventCallBackImpl() registered
Source: XDLX_Kaya_Grabber.cpp
Meaning: CXP2 event callback registered. Normal setup step.

### I98  Device Firmware Version: <version string>
Source: XDLX_Kaya_Grabber.cpp — getHardwareInfo()
Meaning: Grabber firmware version read from hardware register. Also stored as latestTelemetry.softwateFirmInfo.firmwareVer.
Use in report: Report as "Firmware: X.X.X".

### I99  Device Core Temperature: <°C>  (inside getHardwareInfo)
Source: XDLX_Kaya_Grabber.cpp
Meaning: Grabber board temperature at connection time. Same physical sensor as I87.

### I78  Closing the Grabber...
Source: XDLX_Kaya_Grabber.cpp — Close()
Meaning: Grabber close sequence started.

### I79  Grabber closed successfully
Source: XDLX_Kaya_Grabber.cpp — Close() after KYFG_Close()
Meaning: Grabber handle released cleanly.
Use in report: Confirms clean shutdown of hardware layer.

### E42  Failed to close Grabber.
Source: XDLX_Kaya_Grabber.cpp
Meaning: KYFG_Close() failed or handle was already invalid. Non-fatal post-capture.

### E55  Out of Range ID: <n>
Source: XDLX_Kaya_Grabber.cpp — ConnectByID()
Meaning: Requested grabber index out of detected device range.

### E56  wasn't able to retrieve information from device #<n>
Source: XDLX_Kaya_Grabber.cpp — ConnectByID(), flags check failed
Meaning: KY_DeviceInfo() returned non-OK for this device. Skip or fatal depending on context.

### E57  Could not connect to grabber #<n>
Source: XDLX_Kaya_Grabber.cpp — KYFG_Open() returned INVALID_FGHANDLE
Meaning: Open failed. Hardware not responding or already locked by another process.

### E58  grabber #<n> does not support queued buffers
Source: XDLX_Kaya_Grabber.cpp — DEVICE_QUEUED_BUFFERS_SUPPORTED check
Meaning: Device capability missing. Fatal — DMA streaming requires queued buffer support.

### E59  KYDeviceEventCallBackRegister() failed
Source: XDLX_Kaya_Grabber.cpp
Meaning: CXP2 event callback could not be registered. Non-fatal — heartbeat/event notifications will be unavailable.

### E03 (grabber context)  Autoconnect Failed...
Source: XDLX_Kaya_Grabber.cpp — AutoConnect() after all devices tried
Meaning: No physical grabber could be opened. Fatal — no capture follows.

---

## SECTION 4 — CAMERA (I10–I23, I29–I30, I33–I36, I54, E07–E18, E21, E24, E33–E34)

### I10  Camera Detection:
Source: XDLX_Camera.cpp — Detect()
Meaning: Camera scan on this grabber starting.

### I11  Number of cameras detected to the current Grabber: <n>
Source: XDLX_Camera.cpp — Detect()
Meaning: Camera count found via KYFG_UpdateCameraList().

### I12  Camera[i]: deviceModelName= <name>
Source: XDLX_Camera.cpp — Detect()
Meaning: Per-camera model name. "VisLinxM" is the expected value.

### I13  Connecting the desired cameras:
Source: XDLX_Camera.cpp — ConnectDesired()
Meaning: Camera connection loop starting.

### I14  Camera[i]:  (+ "Primary Camera" continuation)
Source: XDLX_Camera.cpp — ConnectDesired(), also getHardwareInfo() path
Two uses:
  1. Camera[i]: Camera Settings path: <path> — shows ZIP settings path. If E09 follows, path is invalid.
  2. Camera[i]: Primary Camera — role assignment during multi-camera connect.

### I15  The Camera was connected successfully
Source: XDLX_Camera.cpp — ConnectDesired() or ConnectByID(), after KYFG_CameraOpen()
Meaning: Camera handle acquired.
Use in report: Confirms camera online. If E09 followed, camera connected but with factory defaults.

### I16  Camera is already connected
Source: XDLX_Camera.cpp
Meaning: Attempted to connect a camera that was already open. Non-fatal, skipped.

### I17  Default settings are being applied...
Source: XDLX_Camera.cpp — SettingsDefault()
Meaning: Factory/default parameter block being written to camera registers.

### I18  Applied default TDI_Modes=<n>
Source: XDLX_Camera.cpp — SettingsDefault()
Values: 0=off, 1=unknown, 2=8-stage ON. This is the DEFAULT; overridden by I34 user settings.

### I19  Applied default TDI_Stages=<n>
Source: XDLX_Camera.cpp — SettingsDefault()

### I20  Applied default FPS = <value>
Source: XDLX_Camera.cpp — SettingsDefault()

### I21  Applied default ExposureTime = <value>
Source: XDLX_Camera.cpp — SettingsDefault()

### I22  Applied default Gain = <value>
Source: XDLX_Camera.cpp — SettingsDefault()

### I23  Applied default BandXShift = <value>
Source: XDLX_Camera.cpp — SettingsDefault()

### I29  Connecting the desired Regions
Source: XDLX_Camera.cpp — ConnectDesiredRegion()
Meaning: Band/region activation loop starting. Applies BandSelection bits from parameter file.

### I30  Camera: <id> has turned <on|off> <regionMode>: <mode>
Source: XDLX_Camera.cpp — ConnectDesiredRegion() and SettingsDefault()
Meaning: Per-region activation status. One line per band.
Use in report: Verify against BandSelection bitmask. Regions for inactive bands will say "off".

### I33  User settings are being applied...
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: Parameter file values being written to camera. Follows I17 block.

### I34  Set TDI_Modes=<n> Passed.../Failed...  Applied TDI_Modes=<n>
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: TDI mode write result. "Passed" = register write OK. "Failed" = write rejected by camera.
Values: 0=off, 2=8-stage TDI on. Must match TDI byte in parameter file (34 → mode 2).

### I35  Set TDI_Stages=<n> Passed.../Failed...  Applied TDI_Stages=<n>
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: TDI stage count write result.

### I35 (second use)  Applied TDIYShift=<n> [Default=<default>]
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: TDI Y-shift applied. If requested value > default value, E log fires (expected, see E section).
Formula: G_TDIYShift = TDIYShift from parameter file (field 14). Default = RegionHeight / TDI_Stages.

### I36  Set FrameCount=<n> [MaxFrameCount=<n>] Passed.../Failed...
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: Frame count register written to camera.

### I54  SensorTemp: <value>
Source: XDLX_CamApp_V6.cpp — appears twice: start and end of session
Meaning: Camera sensor temperature read via KYFG_GetCameraValueFloat("DeviceTemperature") with DeviceTemperatureSelector=0.
Firmware bug note: Firmware 2.3.2 reports artificially low values (~8–14°C). Firmware 2.2.2 values are credible (~46°C under load).
Use in report: Report start and end values; note firmware version for credibility.

### E07  Camera detect error. Please try again
Source: XDLX_Camera.cpp — Detect(), KYFG_UpdateCameraList() failed
Meaning: Camera list could not be refreshed. Possibly camera not powered.

### E08  No cameras detected. Please connect at least one camera
Source: XDLX_Camera.cpp — Detect(), nDetected = 0
Meaning: No cameras found on grabber. Fatal.

### E09  Invalid Camera Settings Path
Source: XDLX_Camera.cpp — SettingsDefault() when loading camera.zip
Meaning: The camera configuration ZIP path (from CamSettingsPathR) does not exist or is not readable.
Consequence: Camera falls back to factory default settings. Custom calibration NOT applied.
Use in report: Flag as WARNING if present. Affects radiometric accuracy.
Common cause in fleet: camera.zip not deployed at the expected path on this host machine.

### E10  The Camera was not connected
Source: XDLX_Camera.cpp — ConnectDesired() or ConnectByID()
Meaning: KYFG_CameraOpen() failed. Camera not responding.

### E11  Failed to set default TDIMode
Source: XDLX_Camera.cpp — SettingsDefault()
Meaning: Could not write default TDI mode. Camera register access issue.

### E12  Failed to set default TDIStages
Source: XDLX_Camera.cpp — SettingsDefault()

### E13  Failed to set default FPS
Source: XDLX_Camera.cpp — SettingsDefault()

### E14  Failed to set default ExposureTime
Source: XDLX_Camera.cpp — SettingsDefault()

### E15  Failed to set default Gain
Source: XDLX_Camera.cpp — SettingsDefault()

### E16  Failed to set default BandXShift
Source: XDLX_Camera.cpp — SettingsDefault()

### E18  Camera: <id> failed to set <on|off>
Source: XDLX_Camera.cpp — ConnectDesiredRegion() or SettingsDefault()
Meaning: Region enable/disable command rejected. If BandSelection marks a band as OFF, error 0x301a from firmware is expected.
Expected vs unexpected: 0x301a on an inactive region = expected, not an error. Any other status = unexpected.

### E21  Failed to set API for SensorTemp
Source: XDLX_CamApp_V6.cpp — KYFG_SetCameraValueEnum("DeviceTemperatureSelector", 0) failed
Meaning: Could not select temperature channel 0 before reading sensor temp. Non-fatal; SensorTemp reading may be invalid.

### E24  Buffer overflow detected! Offset: <n>, Buffer Size: <n>, File Size: <n>
Source: XDLX_Camera.cpp — callback during streaming
Meaning: Frame write pointer exceeded pre-allocated buffer. Critical — frame data may be corrupted at this point.
Diagnosis: TotalFrames or frame size calculation was wrong. Check I67 vs actual buffer size.

### E33  Error: KYFG_BufferQueueAll for CamID: <n>
Source: XDLX_Camera.cpp — StartCamera()
Meaning: Buffer queuing failed before camera start. No frames will be captured.

### E34  Error: KYFG_CameraStart for CamID: <n>
Source: XDLX_Camera.cpp — StartCamera()
Meaning: Camera start command rejected. Camera may be in wrong state.

---

## SECTION 5 — STREAM & MEMORY (I55–I75, I88–I90, E23–E40, E46–E51)

### I55  FPS: <f>  Duration: <d> = Total no of frames: <n>
Source: XDLX_CamApp_V6.cpp — after ProcessInput()
Meaning: Computed frame target. Formula: floor(G_FPS * G_Duration). If TDI mode 2, initialFrames = floor(G_FPS/2) added.
Use in report: This is the expected frame count. Compare against I70 CapturedCount.

### I56  Wait for time to trigger...
Source: XDLX_CamApp_V6.cpp
Meaning: Trigger wait phase starting. Duration = I27 Waiting Time value.

### I57  Waiting Time Completed
Source: XDLX_CamApp_V6.cpp
Meaning: Trigger wait finished. Camera acquisition starts immediately after.

### I58  Stream Created...
Source: XDLX_Stream.cpp — CreateStreamMap()
Meaning: KAYA stream handle and DMA buffer map created successfully.

### I59  Directory already exists: <path>
Source: XDLX_Stream.cpp — CreateStreamMap()
Meaning: Output dataset directory found. Reusing existing folder.

### I60  Directory created: <path>
Source: XDLX_Stream.cpp — CreateStreamMap()
Meaning: Output dataset directory created fresh.

### I63  Building with user allocated buffers  (compile-time pragma message)
Source: XDLX_Stream.cpp
Meaning: DMA buffers are user-allocated (not OS-managed). This is the standard build mode.

### I64  Total Memory Available: <n> (MB)
Source: XDLX_Stream.cpp — after buffer sizing
Meaning: RAM available for DMA allocation at time of buffer setup.

### I65  Total Memory can Allocate: <n> (MB)
Source: XDLX_Stream.cpp
Meaning: Computed usable DMA memory (subset of I64 after safety margin).

### I66  NOTE: Reduced TotalFrames from <old> to <new> and duration from <old_d> to <new_d> due to Memory limits.
Source: XDLX_Stream.cpp
Meaning: RAM was insufficient for the full requested frame count. Capture was shortened automatically.
Use in report: Flag as WARNING. Actual captured frames = new value, not parameter file duration.

### I67  File_size=<bytes> (FrameSize=<bytes> TotalFrames=<n>)
Source: XDLX_Stream.cpp
Meaning: Total pre-allocated output file size. FrameSize = width * height_active * bytes_per_pixel.
Use in report: Cross-check against actual output file size at post-capture.

### I68  Camera successfully stopped...
Source: XDLX_Stream.cpp — DeleteStreamMap()
Meaning: KYFG_CameraStop() succeeded.

### I69  Stream successfully deleted...
Source: XDLX_Stream.cpp — DeleteStreamMap()
Meaning: KYFG_StreamDelete() succeeded.
Use in report: Confirms clean stream teardown.

### I70  TotalNoOfFrames: <expected>  CapturedCount: <actual>
Source: XDLX_Stream.cpp — DeleteStreamMap()
Meaning: Final frame accounting. TotalNoOfFrames = I55 value. CapturedCount = Param.CaptureCount incremented per frame callback.
Use in report: If CapturedCount < TotalNoOfFrames → frame drop. Compute drop count and flag CRITICAL.
Perfect capture: CapturedCount == TotalNoOfFrames.

### I71  Successfully wrote all <n> bytes to file.
Source: XDLX_Stream.cpp — writeData()
Meaning: Output band file written completely.

### I73  Allocated Buffers free done.
Source: XDLX_Stream.cpp
Meaning: DMA buffer memory released.

### I74  Configuration saved to JSON successfully.
Source: XDLX_Stream.cpp — saveConfigInfo()
Meaning: Session config written to .json file in dataset folder.

### I75  (implied by I74 sequence) — dataset folder write complete
Use in report: Confirms dataset folder is complete.

### I80  Common Configuration saved to JSON successfully.
Source: XDLX_Stream.cpp — saveCommonConfig()
Meaning: Shared config (cross-session state) saved.

### I88  Data written successfully / Data write unsuccessful
Source: XDLX_CamApp_V6.cpp — writeData() path (non-BINNING build)
Meaning: Raw band data write result.

### I88 (second use in Stream)  Computed BAND_HEIGHT = <n>
Source: XDLX_Stream.cpp — ProcessData() binning path
Meaning: Computed per-band pixel height for binned output. Formula: RegionHeight / TDI_Stages / 2 (if binned).

### I89  Data Processing completed successfully / Data Processing unsuccessful
Source: XDLX_CamApp_V6.cpp — ProcessData() path (BINNING build)
Meaning: Binning/CCSDS post-processing result.

### I90  Total time took for data processing: <seconds> seconds
Source: XDLX_CamApp_V6.cpp — BINNING build
Meaning: Elapsed time for post-processing. Normal range: 0.2–1.5s for typical captures.
Flag if: > 5s on a high-performance host (may indicate binning + CCSDS overhead — see Acq000069 at 12.9s which is normal for 6-band CCSDS).

### E23  Error creating directory <path>: <errno string>
Source: XDLX_Stream.cpp — CreateStreamMap()
Meaning: mkdir() failed. Disk full or permissions problem.

### E25  Error: KYFG_StreamCreate
Source: XDLX_Stream.cpp — CreateStreamMap()
Meaning: Stream handle creation failed.

### E26  Error: KYFG_StreamBufferCallbackRegister
Source: XDLX_Stream.cpp
Meaning: Frame callback could not be registered. No frames will be received.

### E27  Error: KYFG_StreamGetInfo, KY_STREAM_INFO_PAYLOAD_SIZE
Source: XDLX_Stream.cpp
Meaning: Could not get frame payload size from stream. Buffer sizing impossible.

### E28  Error: KYFG_StreamGetInfo, KY_STREAM_INFO_BUF_ALIGNMENT
Source: XDLX_Stream.cpp
Meaning: Could not get buffer alignment requirement.

### E30  Error: KYFG_BufferAnnounce
Source: XDLX_Stream.cpp
Meaning: DMA buffer announcement to hardware failed.

### E31  ERROR: Not enough Memory to allocate even 1 frame buffer!
Source: XDLX_Stream.cpp
Meaning: Available RAM is less than a single frame. Fatal — no capture possible.
Diagnosis: RAM too low, or another process consuming memory.

### E32  Unsuccessful Camera not opened
Source: XDLX_Stream.cpp — DeleteStreamMap() path when camera handle invalid
Meaning: Cannot delete stream because camera was never successfully opened.

### E35  Failed to stop Camera
Source: XDLX_Stream.cpp — DeleteStreamMap()
Meaning: KYFG_CameraStop() returned error. Camera may still be streaming.

### E36  Failed to delete Stream.
Source: XDLX_Stream.cpp — DeleteStreamMap()
Meaning: KYFG_StreamDelete() failed. Resource leak possible.

### E37  Error: Less data acquired with file size: <n>
Source: XDLX_Stream.cpp — DeleteStreamMap()
Meaning: File size in memory is smaller than expected. Partial capture.
Diagnosis: Combined with I70 CapturedCount < TotalNoOfFrames, confirms actual frame drops.

### E38  Failed to open file for writing
Source: XDLX_Stream.cpp — writeData()
Meaning: Output file could not be created or opened. Disk permissions or path issue.

### E39  write failed
Source: XDLX_Stream.cpp — writeData()
Meaning: write() syscall failed mid-stream. Disk full or I/O error.

### E40 (two uses)
1. Failed to open JSON file for writing configurations: <path>  — saveConfigInfo()
2. Partial write: <written> of <total> bytes — writeData()
Both are non-fatal but indicate incomplete output data.

### E43  Failed to open JSON file for writing current folder name: <path>
Source: XDLX_Stream.cpp — saveCommonConfig()
Meaning: Common config JSON could not be saved.

### E46  No data to process — binning buffer is null.
Source: XDLX_Stream.cpp — ProcessData()
Meaning: ProcessData() called with no data in buffer. Skip or abort post-processing.

### E47  Failed to open file: <path>
Source: XDLX_Stream.cpp — ProcessData() band file output
Meaning: Band output file could not be opened.

### E48  Failed to open split files: <left_path> / <right_path>
Source: XDLX_Stream.cpp — ProcessData() unbinned band split
Meaning: For unbinned bands the system writes .bandX0 (left half) and .bandX1 (right half). If either file can't be opened, band data is lost.

### E49  Failed to open output.meta
Source: XDLX_Stream.cpp — ProcessData()
Meaning: Metadata output file could not be created. Frame-level telemetry will be missing.

### E50  Frame <n>: insufficient data for metadata.
Source: XDLX_Stream.cpp — ProcessData()
Meaning: Frame buffer is too small to contain the expected metadata chunk. Partial capture or buffer sizing issue.

---

## SECTION 6 — SETTINGS APPLIED & JSON (I87-context, E-context from SettingsApplied)

### getSettingsApplied output block
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: After SettingUserInput(), this block reads back the applied values from the camera. Logged as raw PRINT_LOG lines without codes.
Key lines to parse:
  "FPS = <value>" — applied FPS (may differ slightly from requested due to hardware rounding)
  "MaxFPS = <value>" — hardware maximum for this config
  "ExposureTime = <value>" — applied exposure in µs
  "MaxExpTime = <value>" — hardware maximum exposure for this FPS
  "Gain = <value>"
  "Applied TDI_Modes = <n>"
  "Applied TDI_Stages = <n>"
  "Applied TDIYShift = <n>"
  "RegionHeight = <n>"
  "Width = <n>"
  "Height = <n>" — active pixel height
  "G_TDIYShift = <n>" — physical Y-shift per stage (= RegionHeight / TDI_Stages)

---

## SECTION 7 — E LOG EXPECTED CASES (important for Iris reasoning)

The following E-level log entries appear during normal operation and must NOT be treated as errors:

### E (no number)  G_TDIYShift: <n> is greater than default TDIYShift: <default>. Value should less than default TDIYShift.
Source: XDLX_Camera.cpp — SettingUserInput() TDIYShift check
Meaning: Fired whenever TDIYShift parameter >= default (RegionHeight / TDI_Stages).
When TDIYShift=384 and TDI_Stages=8: default = 384/8 = 48. 384 > 48 → this E log fires.
Interpretation: EXPECTED and INFORMATIONAL. TDIYShift=384 is the full region height and is the correct value for 8-stage TDI operation. Not an error.

### E18  Camera: <id> failed to set <on|off>  with error 0x301a
Source: XDLX_Camera.cpp — ConnectDesiredRegion()
Meaning: Region OffsetY or enable write on a disabled band.
Interpretation: EXPECTED. The firmware rejects parameter writes on regions that are marked OFF by BandSelection. This is correct behaviour. Do not flag as an error in the report.

---

## SECTION 8 — PARAMETER DECODING REFERENCE

### BandSelection byte (parameter field 7)
Bits 6–0, one per band. Bit 0 = Band 1 (Region 0), Bit 6 = Band 7 (Region 6).
A bit value of 1 = band active.
Examples:
  127 = 0b1111111 = all 7 bands active
   30 = 0b0011110 = bands 2,3,4,5 active (bits 1–4 set)
    0 = no bands active (invalid)

### TDI byte (parameter field 8)
   0 = TDI OFF
  10 = TDI ON, 2-stage
  18 = TDI ON, 4-stage
  34 = TDI ON, 8-stage  (most common operational mode)
  66 = TDI ON, 64-stage

### Binning byte (parameter field 13)
Bit 7 = CCSDS compression flag (1 = CCSDS on)
Bits 6–0 = per-band binning (1 = that band is binned, 0 = unbinned)
Examples:
    0 = 0b00000000 = no binning, no CCSDS
  247 = 0b11110111 = CCSDS ON, bands 0,1,2,4,5,6 binned, band 3 unbinned
Unbinned band output: two files per band — .bandX0 (left half, width/2) and .bandX1 (right half, width/2)
Binned band output: single .bandX file (height/2)

### Frame count formula
TotalFrames = floor(G_FPS × G_Duration)
If TDI_Modes == 2 (8-stage TDI ON): initialFrames = floor(G_FPS / 2) added for ramp-up; subtracted from CapturedCount comparison.

### Band file size formula
file_size = (height_active / TDI_stages) × width × (bit_depth / 8) × num_frames
Where:
  height_active = RegionHeight = 384 px (standard)
  width = 8448 px full, or 4224 px per half if unbinned band
  bit_depth = 10 bits = 1.25 bytes/pixel
  If binned: height / 2

---

## SECTION 9 — FRAME CALLBACK (per-frame, high-frequency)

### Per-frame line (no code)  FrameNo=<n>, instantFps=<value>
Source: XDLX_Camera.cpp — frame callback
Meaning: Logged for every received frame. FrameNo starts at 0.
Normal: instantFps tracks applied FPS (e.g., 56.09–56.11 for a 56 FPS session).
Flag: instantFps = 0 on Frame 0 is expected (no prior frame to compute ratio).
Flag: Large gaps in FrameNo or FPS spikes indicate frame drops between callback invocations.

### Per-frame line (no code)  TimeDifference: <ms>
Source: XDLX_Camera.cpp — SettingUserInput() callback or stream callback
Meaning: Time delta between consecutive frames. Should equal 1000/FPS ms.
Normal: Constant value to 4 decimal places (e.g., 17.8572 ms for 56 FPS, 76.2491 ms for 13.114 FPS).
Flag: Varying values indicate timing instability.

---

## SECTION 10 — SIGNAL / OS EVENTS

### [SYS]  Signal received: <name> (<number>), cleaning up...
Source: XDLX_CamApp_V6.cpp — handle_signal()
Meaning: OS signal caught. Registered signals: SIGINT (2), SIGTERM (15), SIGSEGV (11), SIGBUS (7), SIGABRT (6).
Behaviour: g_terminate set, log flushed, log transferred, signal re-raised (allows core dump on SIGSEGV/SIGABRT).
Use in report: If present, session did not complete normally. Identify signal to diagnose (11=crash, 2=user interrupt, 15=kill).

---

## SECTION 11 — MULTI-SESSION PATTERNS (fleet-level notes for bulk report)

### Stale parameter date (W01 in all sessions)
If W01 appears in every session log, the parameter file date field has never been updated.
The same raw date string in all I07 lines confirms a systematic process gap, not a one-off mistake.
Action: Update the Date and Time fields (fields 4 and 5) in the parameter file before each mission.

### E09 across sessions (camera.zip path)
If E09 appears in 4 of 5 sessions, camera calibration settings are not being deployed consistently.
Check: Is the camera.zip path in CamSettingsPathR hard-coded or platform-specific?
Action: Standardise camera.zip path or embed per-platform config.

### Firmware version split
Firmware 2.2.2 → sensor temperature credible (~46°C at load).
Firmware 2.3.2 → sensor temperature bug: reads 8–14°C regardless of actual temperature. Do not use for radiometric calibration.
Core temperature (I87, I99) is credible on all firmware versions.

### Grabber connection time (I06 → I96 delta)
Normal: < 5 seconds.
Slow (> 30s): PCIe re-enumeration or camera power-up delay. Monitor for recurrence.
Timeout (E99): Hardware or driver fault.

---

## QUICK DIAGNOSTIC TABLE

| Symptom | Log codes to check | Likely cause |
|---|---|---|
| Session never started | E01, E02, E52, E99 | Library, timing, grabber |
| No camera found | E07, E08 | Camera power / cable |
| Camera uses factory defaults | E09 | camera.zip path invalid |
| Frame count mismatch | I55 vs I70 CapturedCount | Frame drops, E37 |
| Capture shortened | I66 | RAM insufficient |
| Stale trigger, 5s wait | W01 | Parameter file date not updated |
| Temperature warning | E101, E102 | Thermal limit exceeded |
| Sensor temp looks wrong | I54 | Firmware 2.3.2 bug (8–14°C) |
| Slow grabber connect | I06→I96 delta | PCIe enumeration delay |
| E log for TDIYShift | E (no num), TDIYShift > default | Expected — not a defect |
| E18 on inactive regions | E18 with 0x301a | Expected — not a defect |
| Clean shutdown | I68, I69, I76, I79, I91 | All handles released |

---

## SECTION 12 — PREVIOUSLY UNDOCUMENTED CODES (added after full audit)

### I08  Camera successfully stopped  (COMMENTED OUT)
Source: XDLX_Camera.cpp — old path, now replaced by I68
Status: Code exists in source as a comment (;// PRINT_LOG). Will NOT appear in logs. Superseded by I68.

### I09  Camera Stream successfully deleted  (COMMENTED OUT)
Source: XDLX_Camera.cpp — old path, now replaced by I69
Status: Code exists in source as a comment. Will NOT appear in logs. Superseded by I69.

### I24  Applied default BandYShift  (COMMENTED OUT)
Source: XDLX_Camera.cpp — SettingsDefault()
Status: Fully commented out. Will NOT appear in logs. BandYShift not implemented in current firmware.

### I31  Setter and Getters through JSON...
Source: XDLX_Camera.cpp — SettingsJSON()
Meaning: JSON-based settings block starting. Camera parameters read from the config JSON file are being applied.

### I32  Configuration json file used: <path>
Source: XDLX_Camera.cpp — SettingsJSON()
Meaning: Path to the .json config file being used for camera settings (distinct from the parameter file).
Use in report: Confirms which JSON config was active. Log in Mission Identity section.

### I37  (two uses)
1. Updated FPS from: <old> to supported max FPS: <new>
   Meaning: Requested FPS exceeded camera hardware maximum. Auto-clamped to MaxFPS. Flag as WARNING — actual FPS differs from parameter file.
2. Set FPS=<value>
   Meaning: FPS write command sent to camera register.

### I38  Set Exposure Time=<value> [MaxExpTime=<value>]
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: Exposure time write command. MaxExpTime is the hardware ceiling for this FPS setting. If requested > MaxExpTime, exposure would have been clamped (check I50 for applied value).

### I39  Set Gain=<value>
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: Gain write command sent.

### I40  Set BandXShift=<value>
Source: XDLX_Camera.cpp — SettingUserInput()
Meaning: Horizontal pixel shift write command sent.

### I41  Set BandYShift  (COMMENTED OUT)
Source: XDLX_Camera.cpp — SettingUserInput()
Status: Commented out. Will NOT appear in logs. BandYShift not active.

### I42  Final Settings applied:
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: All parameter writes complete. The I43–I52 readback block follows this line.
Use in report: Everything after I42 up to the next section is the confirmed applied-values block.

### I43  <regionModeName>:<mode>
Source: XDLX_Camera.cpp — getSettingsApplied(), region loop
Meaning: Per-region mode readback. One line per active band region. Confirms which regions are on/off after all settings applied.

### I44  Applied Width=<value>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: Sensor width confirmed applied. Expected: 8448 px.

### I45  Applied RegionHeight=<value>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: Region height confirmed applied. Expected: 384 px.

### I46  Applied Height=<value>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: Active pixel height (after TDI and band config). For 4-band config: ~193 px. For 7-band: ~384 px.

### I47  Applied TDI_Modes=<n>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: TDI mode readback. 0=off, 2=8-stage on. This is the ground-truth confirmed value.

### I48  (two uses, same code)
1. Applied TDI_Stages=<n> — stage count readback
2. Applied G_TDIYShift: <n> — physical Y-shift per stage readback (= RegionHeight / TDI_Stages = 384/8 = 48)

### I49  Applied FPS = <value>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: Confirmed FPS from camera register readback. Compare against parameter file request.

### I50  Applied ExposureTime = <value>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: Confirmed exposure time in µs. Compare against I38 requested value and MaxExpTime.

### I51  Applied Gain = <value>
Source: XDLX_Camera.cpp — getSettingsApplied()

### I52  Applied BandXShift = <value>
Source: XDLX_Camera.cpp — getSettingsApplied()
Meaning: Confirmed horizontal shift. Expected: 0 for standard capture.

### I53  Applied BandYShift  (COMMENTED OUT)
Source: XDLX_Camera.cpp — getSettingsApplied()
Status: Commented out. Will NOT appear in logs.

### I76  Successfully closed the camera.
Source: XDLX_Camera.cpp — CloseAll()
Meaning: Camera handle released cleanly.
Use in report: Confirms clean camera shutdown. Pair with I79 (grabber) and I69 (stream) for full clean-shutdown confirmation.

### I77  Camera not open
Source: XDLX_Camera.cpp — CloseAll()
Meaning: Attempted to close a camera that was never opened. Non-fatal — occurs when a camera slot was never used.

### I92  Received KYDEVICE_EVENT_CXP2_HEARTBEAT: cameraTime=<n>
Source: XDLX_Kaya_Grabber.cpp — ProcessHeartbeatEvent()
Meaning: CXP2 heartbeat received from camera. Only printed if printHeartbeats flag is ON (off by default).
Normal: Will not appear in standard logs. If present, heartbeat logging was manually enabled.

### I93  Received KYDEVICE_EVENT_CXP2_EVENT: tag=0x<hex>
Source: XDLX_Kaya_Grabber.cpp — ProcessCxp2Event()
Meaning: CXP2 protocol event received. Only printed if printCxp2Events flag is ON (off by default).
Normal: Will not appear in standard logs.

### I100  (COMMENTED OUT in source)
Source: XDLX_Kaya_Grabber.cpp — getHardwareInfo(), DevicePciGeneration read
Status: Commented out. Will NOT appear in logs.

### I101  (COMMENTED OUT in source)
Source: XDLX_Kaya_Grabber.cpp — getHardwareInfo(), DevicePciLanes read
Status: Commented out. Will NOT appear in logs.

### E06  Stream delete unsuccessful  (COMMENTED OUT)
Source: XDLX_Camera.cpp — old path, now replaced by E36
Status: Code exists as a comment. Will NOT appear in logs. Superseded by E36.

### E17  (COMMENTED OUT in source — BandYShift failure)
Source: XDLX_Camera.cpp — SettingsDefault()
Status: Commented out. Will NOT appear in logs.

### E19  Failed to open Configuration Settings file. <path>
Source: XDLX_Camera.cpp — SettingsJSON()
Meaning: The JSON config file (not the parameter file, not camera.zip) could not be opened. Camera JSON settings not applied.
Severity: WARNING — camera runs without JSON-specified region offsets or tuning values.

### E20  JSON parsing error: <what>
Source: XDLX_Camera.cpp — SettingsJSON()
Meaning: Config JSON file exists but contains malformed JSON. Settings not applied.
Diagnosis: Check the JSON config file for syntax errors.

### E41  Failed to close camera
Source: XDLX_Camera.cpp — CloseAll()
Meaning: KYFG_CameraClose() failed. Camera handle may be stale. Non-fatal post-capture.

### E51  (COMMENTED OUT in source — invalid frame range in split)
Source: XDLX_Stream.cpp — ProcessData() raw split path
Status: Commented out. Will NOT appear in logs.

### E53  Telemetry initialization failed.
Source: XDLX_CamApp_V6.cpp — after telemetry.initialize()
Meaning: The telemetry subsystem (temperature polling thread) could not start. Non-fatal — capture proceeds but temperature HM data will not be collected.
Diagnosis: Check telemetry hardware connection or driver.

---

## SECTION 13 — getSettingsApplied FULL READBACK SEQUENCE

The I42–I52 block is the confirmed applied-values block. It appears after all parameter writes and is the authoritative source for what the camera actually ran with.

Full sequence in log order:
  I42  Final Settings applied:
  I43  <region0Name>:<mode>  (one line per active region)
  I43  <region1Name>:<mode>
  ...
  I44  Applied Width=<n>
  I45  Applied RegionHeight=<n>
  I46  Applied Height=<n>
  I47  Applied TDI_Modes=<n>
  I48  Applied TDI_Stages=<n>
  I48  Applied G_TDIYShift: <n>
  I49  Applied FPS = <n>
  I50  Applied ExposureTime = <n>
  I51  Applied Gain = <n>
  I52  Applied BandXShift = <n>

Use in report: Section 2 "Requested vs Applied" should be built from parameter file (I07/I28) vs this block.

---

## UPDATED QUICK DIAGNOSTIC TABLE ADDITIONS

| Symptom | Log codes to check | Likely cause |
|---|---|---|
| JSON config not applied | E19, E20 | JSON file missing or malformed |
| Telemetry/temp not collected | E53 | Telemetry init failed |
| FPS auto-reduced | I37 (first form) | Requested FPS > hardware MaxFPS |
| Camera closed with error | E41 | Camera handle invalid at shutdown |
| Applied values block | I42–I52 | getSettingsApplied() readback |
| I08/I09 in log | I08, I09 | Old code path — should not appear in v6+ |
| Heartbeat spam in log | I92, I93 | printHeartbeats/printCxp2Events enabled |
