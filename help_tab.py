from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
import os
import html as html_module

_MODE_TITLES = {
    "onboarding": "Mode Selection Help",
    "band": "Band Mode Help",
    "raw": "Raw Mode Help",
    "video": "Video Mode Help",
    "live": "Live Display Mode Help",
    "tiled": "Tiled Mode Help",
}

_MODE_BODIES = {
    "onboarding": """
<h3>DisplayGround – Quick Start</h3>
<p><strong>Choose Your Data Type:</strong></p>
<ul>
  <li><strong>Band Mode</strong> — Multi-spectral <code>.bandXX</code> files, stitching, RGB fusion, histogram, full analysis toolkit.</li>
  <li><strong>Raw Mode</strong> — Direct <code>.raw</code> sensor file inspection with frame-by-frame navigation and stack preview.</li>
  <li><strong>Video Mode</strong> — Generate and playback video sequences from bands with RGB composition and speed control.</li>
  <li><strong>Live Display Mode</strong> — Real-time camera stream capture, recording, and parameter tuning.</li>
  <li><strong>Tiled Mode</strong> — Assemble large frames from matrix of RAW tiles with flexible ordering and stitching.</li>
</ul>
<p><strong>Getting Started:</strong> Use <strong>+</strong> button (top-left) to add a new dataset or click mode buttons. Each mode has its own viewer and controls tuned for that data type.</p>
""",
    "band": """
<h3>Band Mode — Multi-Spectral Analysis</h3>
<hr/>

<h4>Quick Start</h4>
<ol>
  <li><strong>Select Display Mode</strong>: Check/uncheck <strong>All Bands</strong>, <strong>Individual Bands</strong>, <strong>RGB Fusion</strong>, <strong>Histogram</strong>.</li>
  <li>Click <strong>Select Folder &amp; Stitch</strong> — choose folder with <code>.bandXX</code> files. Dialog asks for width, height, bit depth.</li>
  <li><strong>Close a dataset</strong> with the X on the tab, or uncheck a view mode to hide it.</li>
  <li><strong>Enable other views</strong> from the Display Modes section to explore multiple representations.</li>
</ol>

<h4>Top / Global Controls</h4>
<ul>
  <li><strong>+</strong> — Add new dataset (create new dataset tab).</li>
  <li><strong>Add Live Tab</strong> — Open live-streaming tab (connects to device/watched folder).</li>
  <li><strong>Theme (sun icon)</strong> — Toggle light / dark UI theme.</li>
</ul>

<h4>Display Modes &amp; Folder</h4>
<ul>
  <li><strong>All Bands</strong> — Show stitched stacked bands in one view.</li>
  <li><strong>Individual Bands</strong> — Create per-band tabs for selective inspection.</li>
  <li><strong>RGB Fusion</strong> — Enable RGB composite mode with channel mapping.</li>
  <li><strong>Histogram</strong> — Display histogram for intensity analysis.</li>
  <li><strong>Select Folder &amp; Stitch</strong> — Choose folder containing <code>.bandXX</code> files and supply width/height/bitdepth if prompted.</li>
  <li><strong>Recent menu (▼)</strong> — Quick access to recently opened folders.</li>
  <li><strong>Folder label</strong> — Shows selected folder path or "No folder selected".</li>
</ul>

<h4>Frame Controls</h4>
<ul>
  <li><strong>Frame slider</strong> — Drag to jump to a frame.</li>
  <li><strong>Frame label</strong> (e.g., 0/0) — Shows current / total frames.</li>
  <li><strong>Play (▶)</strong> — Play/pause frame animation.</li>
  <li><strong>Prev (◀) / Next (▶)</strong> — Step frames backward/forward.</li>
  <li><strong>Video Mode button</strong> — Add Video Mode tab for video-style playback.</li>
  <li><strong>Single Frame / Frame Range</strong> — Choose single frame or range mode.</li>
  <li><strong>Start / End</strong> — Spinboxes set frame indices for range operations.</li>
</ul>

<h4>Band Offsets &amp; Display Settings</h4>
<ul>
  <li><strong>Band Offsets (collapsible ▶)</strong> — Expand to show per-band X/Y offset controls for alignment tuning.</li>
  <li><strong>Band Gap</strong> — Spacing (pixels) between stitched bands.</li>
  <li><strong>1:4 Layout</strong> — Toggle special 1:4 layout arrangement.</li>
  <li><strong>Fit to Screen / Actual Size</strong> — Choose display scaling mode.</li>
  <li><strong>Contrast Enhance</strong> — Enable/disable contrast enhancement.</li>
  <li><strong>Min / Max</strong> — Numeric fields to set display intensity window.</li>
  <li><strong>Auto</strong> — Compute sensible min/max automatically for current frame.</li>
</ul>

<h4>Data &amp; Session Management</h4>
<ul>
  <li><strong>Save Progress</strong> — Persist dataset settings (offsets, contrast, gap, view selections).</li>
  <li><strong>Change Params</strong> — Edit width/height/bit depth later without reloading.</li>
  <li><strong>Export Image</strong> — Save current view to PNG/BMP/TIFF.</li>
  <li><strong>Reload</strong> — Re-scan the folder (use after externally adding files).</li>
  <li><strong>Refresh</strong> — Re-render using current settings (no disk re-read).</li>
</ul>

<h4>Pixel Info &amp; Measurement</h4>
<ul>
  <li><strong>Matrix Size</strong> — Choose sampling block (1×1, 3×3, 5×5, ...) for pixel statistics.</li>
  <li><strong>Pixel Info Box</strong> — Live display: cursor X/Y, pixel DN values (per-band or RGB), matrix stats (mean/std).</li>
</ul>

<h4>Main Viewer Area</h4>
<ul>
  <li><strong>Viewer tabs</strong> — All Bands, individual band tabs, RGB preview, Histogram, Help. Close each with ×.</li>
  <li><strong>Position label</strong> (top-right) — Shows X, Y | Zoom: % | Rotation: °.</li>
</ul>

<h4>Bottom Viewer Tools (per-tab)</h4>
<ul>
  <li><strong>Enable Magnifier</strong> — Show floating magnifier at cursor for detail inspection.</li>
  <li><strong>Torch</strong> — Bright spotlight effect; darkens surroundings.</li>
  <li><strong>Magnifier Zoom (slider)</strong> — Adjust magnifier magnification (zoom-in factor).</li>
  <li><strong>Zoom Out</strong> — Step zoom out.</li>
  <li><strong>Reset</strong> — Reset zoom to fit or actual based on mode.</li>
  <li><strong>Zoom In</strong> — Step zoom in.</li>
  <li><strong>Mouse Zoom (toggle)</strong> — Enable/disable Ctrl+scroll wheel zoom; affects all tabs.</li>
  <li><strong>Grid (toggle)</strong> — Show overlay grid with interactive positioning.</li>
  <li><strong>Measure (toggle)</strong> — Click 2 points to measure distance and coordinates.</li>
  <li><strong>Calculate (toggle)</strong> — Drag rectangle to get region statistics (mean, median, stddev).</li>
  <li><strong>Flip (cycles Off → Select → All)</strong> — Apply horizontal/vertical flip; Select = individual tile, All = global.</li>
  <li><strong>Rotate (cycles 0° → 90° → 180° → 270°)</strong> — Rotate display (non-destructive, view-only).</li>
</ul>

<h4>Per-Tab Specifics</h4>
<ul>
  <li><strong>All Bands tab</strong> — Shows stitched bands; apply offsets/gaps, use all viewer tools.</li>
  <li><strong>Individual Band tabs</strong> — One tab per band (lazy-loaded); same viewer tools per tab.</li>
  <li><strong>RGB Fusion tab</strong> — Top: RGB preview viewer. Bottom: Red/Green/Blue band selectors, per-channel X/Y offsets, Preview and Auto-preview buttons.</li>
  <li><strong>Histogram tab</strong> — Histogram for current frame or range; mode selector for band/RGB/combined.</li>
  <li><strong>Help tab</strong> — This help document.</li>
</ul>

<h4>Keyboard Shortcuts</h4>
<ul>
  <li><strong>Shift + N</strong> — Add a new dataset tab.</li>
  <li><strong>Shift + Q</strong> — Close the current tab.</li>
  <li><strong>Shift + Enter</strong> — Select folder.</li>
  <li><strong>Ctrl + S</strong> — Save parameters.</li>
  <li><strong>Tab / Shift + Tab</strong> — Cycle view tabs forward / backward.</li>
  <li><strong>Right Arrow / Left Arrow</strong> — Next / Previous frame.</li>
  <li><strong>Ctrl + Up / Ctrl + Down</strong> — Zoom in / Zoom out.</li>
  <li><strong>Space</strong> — Play / Pause.</li>
  <li><strong>Enter</strong> — Update views.</li>
  <li><strong>Ctrl + Enter</strong> — Apply contrast enhancement.</li>
  <li><strong>Ctrl + Space</strong> — Export current image.</li>
</ul>
""",
    "raw": """
<h3>Raw Mode — Direct Raw File Inspection</h3>
<hr/>

<h4>Quick Start</h4>
<ol>
  <li>Click <strong>Load Raw File</strong> to select a <code>.raw</code> file.</li>
  <li>Enter or confirm <strong>width</strong>, <strong>height</strong>, and <strong>bit depth</strong>.</li>
  <li>Adjust <strong>contrast</strong> and use <strong>frame slider / index selector</strong> to browse frames.</li>
  <li>Use <strong>Stack</strong> button to build a downsampled preview of all frames at once.</li>
</ol>

<h4>File &amp; Frame Configuration</h4>
<ul>
  <li><strong>Load Raw File</strong> — Choose a <code>.raw</code> file and specify dimensions.</li>
  <li><strong>Width / Height</strong> — Frame dimensions in pixels.</li>
  <li><strong>Bit Depth</strong> — 8, 10, 12, 16, etc. (auto-unpacking for multi-byte formats).</li>
  <li><strong>Byte Order</strong> — Little-endian (Intel) or Big-endian (Motorola).</li>
</ul>

<h4>Frame Navigation &amp; Playback</h4>
<ul>
  <li><strong>Frame Index</strong> — Spinbox to jump to specific frame number.</li>
  <li><strong>Frame Slider</strong> — Drag to scrub through frames.</li>
  <li><strong>Frame Label</strong> — Shows current / total frames.</li>
  <li><strong>Play (▶)</strong> — Animate frames forward/backward.</li>
  <li><strong>Prev (◀) / Next (▶)</strong> — Step single frame.</li>
  <li><strong>Start / End Range</strong> — Define frame range for operations.</li>
</ul>

<h4>Contrast &amp; Display</h4>
<ul>
  <li><strong>Fit to Screen / Actual Size</strong> — Toggle scaling mode.</li>
  <li><strong>Contrast Enhance</strong> — Enable/disable dynamic range stretching.</li>
  <li><strong>Min / Max</strong> — Manual intensity window controls.</li>
  <li><strong>Auto</strong> — Auto-compute optimal min/max for current frame.</li>
</ul>

<h4>Stack &amp; Analysis</h4>
<ul>
  <li><strong>Create Stack</strong> — Build a downsampled composite image showing all frames in the file.</li>
  <li><strong>Stack Progress Bar</strong> — Real-time progress during stack generation.</li>
  <li><strong>Histogram</strong> — View intensity distribution of current frame with statistics (mean, median, std-dev).</li>
</ul>

<h4>Main Viewer Area</h4>
<ul>
  <li><strong>Viewer tabs</strong> — Current Frame, Histogram, Stack (if generated), Help.</li>
  <li><strong>Position label</strong> (top-right) — X, Y | Zoom: % | Rotation: °.</li>
</ul>

<h4>Interactive Tools (per-tab)</h4>
<ul>
  <li><strong>Enable Magnifier</strong> — Floating zoom magnifier with adjustable radius.</li>
  <li><strong>Torch</strong> — Spotlight effect with darkened surroundings.</li>
  <li><strong>Magnifier Zoom</strong> — Adjust magnification level.</li>
  <li><strong>Zoom In / Out / Reset</strong> — Scale controls.</li>
  <li><strong>Mouse Zoom</strong> — Ctrl+scroll wheel zoom toggle.</li>
  <li><strong>Grid</strong> — Display interactive overlay grid.</li>
  <li><strong>Measure</strong> — Click 2 points to measure distance &amp; coordinates.</li>
  <li><strong>Calculate</strong> — Drag rectangle for region statistics.</li>
  <li><strong>Flip</strong> — Horizontal/vertical mirroring.</li>
  <li><strong>Rotate</strong> — Rotate display 0°/90°/180°/270°.</li>
</ul>

<h4>Data Management</h4>
<ul>
  <li><strong>Export Image Button</strong> — Save current frame as PNG/BMP (top toolbar).</li>
  <li><strong>Export Image</strong> — Save current frame as PNG/BMP.</li>
  <li><strong>Save Progress</strong> — Store session settings.</li>
  <li><strong>Reload</strong> — Re-read file from disk.</li>
  <li><strong>Refresh</strong> — Re-render with current settings.</li>
  <li><strong>Recent Files (▼)</strong> — Quick access to previously opened .raw files.</li>
</ul>

<h4>Pixel Info</h4>
<ul>
  <li><strong>Matrix Size</strong> — Sampling block for statistics (1×1, 3×3, 5×5, ...).</li>
  <li><strong>Pixel Info Box</strong> — Live X/Y, DN value, and matrix statistics (mean/std).</li>
</ul>

<h4>Keyboard Shortcuts</h4>
<ul>
  <li><strong>Space</strong> — Play / Pause.</li>
  <li><strong>Right / Left Arrow</strong> — Next / Previous frame.</li>
  <li><strong>Ctrl + Up / Down</strong> — Zoom in / Zoom out.</li>
  <li><strong>Ctrl + S</strong> — Save progress.</li>
  <li><strong>Ctrl + Space</strong> — Export image (PNG/BMP).</li>
  <li><strong>Shift + Q</strong> — Close tab.</li>
</ul>
""",
    "video": """
<h3>Video Mode — Generate &amp; Playback Video Sequences</h3>
<hr/>

<h4>Quick Start</h4>
<ol>
  <li>Click <strong>Select Folder</strong> to choose a directory with <code>.bandXX</code> files.</li>
  <li>Configure RGB channels (Red/Green/Blue band selection, per-channel offsets).</li>
  <li>Set <strong>frame range</strong>, <strong>FPS</strong>, and output format; click <strong>Generate Video</strong>.</li>
  <li>Use <strong>playback controls</strong> to review generated video.</li>
</ol>

<h4>Folder &amp; Configuration</h4>
<ul>
  <li><strong>Select Folder</strong> — Choose directory with <code>.bandXX</code> band files.</li>
  <li><strong>Load Recent</strong> — Auto-load previously saved settings for a folder.</li>
  <li><strong>Frame Range</strong> — Define start and end frame indices for video generation.</li>
  <li><strong>FPS</strong> — Frames per second for playback speed.</li>
  <li><strong>Output Format</strong> — MP4, AVI, or image sequence.</li>
</ul>

<h4>RGB Fusion Configuration</h4>
<ul>
  <li><strong>Red Band</strong> — Select which band maps to red channel.</li>
  <li><strong>Green Band</strong> — Select which band maps to green channel.</li>
  <li><strong>Blue Band</strong> — Select which band maps to blue channel.</li>
  <li><strong>Per-Channel Offsets</strong> — X/Y shifts for each channel to correct registration.</li>
  <li><strong>Preview RGB</strong> — Generate quick preview at current frame before full generation.</li>
  <li><strong>Auto Preview</strong> — Automatically regenerate preview when settings change.</li>
</ul>

<h4>Video Generation &amp; Playback</h4>
<ul>
  <li><strong>Generate Video</strong> — Create video file from selected frame range.</li>
  <li><strong>Progress Bar</strong> — Real-time generation progress.</li>
  <li><strong>Play (▶)</strong> — Start/pause playback.</li>
  <li><strong>Seek Slider</strong> — Jump to any point in video.</li>
  <li><strong>Prev / Next Frame</strong> — Single-frame stepping during playback.</li>
  <li><strong>Speed Control</strong> — Adjust playback speed (0.5x — 4x).</li>
  <li><strong>Loop</strong> — Repeat video continuously.</li>
  <li><strong>Frame Counter</strong> — Display current / total frames.</li>
</ul>

<h4>Display &amp; Zoom</h4>
<ul>
  <li><strong>Fit to Screen / Actual Size</strong> — Toggle scaling mode.</li>
  <li><strong>Zoom In / Out / Reset</strong> — Magnification controls.</li>
  <li><strong>Position Label</strong> (top-right) — X, Y, Zoom %, Rotation °.</li>
</ul>

<h4>Keyboard Shortcuts</h4>
<ul>
  <li><strong>Space</strong> — Play / Pause.</li>
  <li><strong>Right / Left Arrow</strong> — Next / Previous frame.</li>
  <li><strong>Ctrl + Up / Down</strong> — Zoom in / Zoom out.</li>
  <li><strong>Ctrl + S</strong> — Save settings.</li>
</ul>
""",
    "live": """
<h3>Live Display Mode — Camera Stream &amp; Capture</h3>
<hr/>

<h4>Quick Start</h4>
<ol>
  <li>Configure <strong>frame width/height, FPS, exposure, gain</strong> parameters.</li>
  <li>Click <strong>Connect</strong> to establish camera connection.</li>
  <li>Click <strong>Start Stream</strong> to begin live preview.</li>
  <li>Select output folder and format; click <strong>Capture</strong> to save frames.</li>
  <li>Click <strong>Stop Stream</strong> and <strong>Disconnect</strong> when done.</li>
</ol>

<h4>Camera &amp; Connection</h4>
<ul>
  <li><strong>Connect</strong> — Establish connection to camera hardware/pipeline.</li>
  <li><strong>Disconnect</strong> — Close camera connection and free resources.</li>
  <li><strong>Status Indicator</strong> — Display connection state and available cameras.</li>
</ul>

<h4>Frame Configuration</h4>
<ul>
  <li><strong>Image Path / Output Folder</strong> — Directory for saving captured frames.</li>
  <li><strong>Frame Width</strong> — Sensor width in pixels.</li>
  <li><strong>Frame Height</strong> — Sensor height in pixels.</li>
  <li><strong>Frame Rate (FPS)</strong> — Live stream frame rate.</li>
  <li><strong>Exposure Time</strong> — Sensor exposure duration (microseconds).</li>
  <li><strong>Gain</strong> — Sensor amplification level.</li>
  <li><strong>Test Pattern</strong> — Internal test pattern selection (0=disabled).</li>
</ul>

<h4>Stream Control &amp; Capture</h4>
<ul>
  <li><strong>Start Stream</strong> — Begin capturing frames from camera.</li>
  <li><strong>Stop Stream</strong> — Halt live capture.</li>
  <li><strong>Frame Buffer</strong> — Configurable buffer size for smooth playback (default: 100 frames).</li>
  <li><strong>Save Mode</strong> — Choose output format: RAW binary, BMP, or PNG.</li>
  <li><strong>Capture</strong> — Save current frame or start batch capture.</li>
</ul>

<h4>Region Modes &amp; Display</h4>
<ul>
  <li><strong>Region Modes (▶)</strong> — Toggle to expand region configuration panel.</li>
  <li><strong>Region Selection</strong> — Define sub-regions of the sensor for focused capture.</li>
  <li><strong>Region Offsets</strong> — Apply XY shifts to selected region.</li>
  <li><strong>Multiple Regions</strong> — Configure and switch between preset regions.</li>
  <li><strong>Fit to Screen</strong> — Automatically scale live stream to fit display.</li>
  <li><strong>Pixel Stretching</strong> — Apply aspect ratio correction for rectangular pixels.</li>
</ul>

<h4>Session Management</h4>
<ul>
  <li><strong>Save Settings</strong> — Persist camera and capture configurations.</li>
  <li><strong>Load Settings</strong> — Restore previous session parameters.</li>
</ul>

<h4>Keyboard Shortcuts</h4>
<ul>
  <li><strong>Space</strong> — Start / Stop stream.</li>
  <li><strong>Ctrl + S</strong> — Save settings.</li>
  <li><strong>Shift + Q</strong> — Close tab.</li>
</ul>
""",
    "tiled": """
<h3>Tiled Mode — Multi-Tile Matrix Stitching</h3>
<hr/>

<h4>Quick Start</h4>
<ol>
  <li>Click <strong>Load / Settings</strong> to open configuration dialog.</li>
  <li>Select session folder with tile data (Folder-per-Frame or Flat Tiles).</li>
  <li>Specify grid dimensions (rows/cols), tile size, overlap, and bit depth.</li>
  <li>Choose tile ordering pattern (Row-Major, Serpentine, Column-Major).</li>
  <li>Review stitched preview and use Tile View to inspect individual tiles.</li>
  <li>Export stitched frames as needed.</li>
</ol>

<h4>Load &amp; Settings Configuration</h4>
<ul>
  <li><strong>Load / Settings</strong> — Open dialog to configure tile parameters.</li>
  <li><strong>Session Folder</strong> — Directory containing tile data.</li>
  <li><strong>Folder Mode</strong>:
    <ul>
      <li><strong>Folder per Frame</strong> — Each frame in its own subdirectory.</li>
      <li><strong>Flat Tiles</strong> — All tiles in single directory (e.g., tile_r0_c0_frame0.raw).</li>
    </ul>
  </li>
  <li><strong>Grid Columns / Rows</strong> — Number of tile columns and rows in matrix.</li>
  <li><strong>Tile Width / Height</strong> — Individual tile size in pixels.</li>
  <li><strong>Overlap</strong> — Pixel overlap between adjacent tiles for seamless stitching.</li>
  <li><strong>Bit Depth</strong> — 8, 16, 24, or 32 bits per tile.</li>
  <li><strong>Final Width / Height</strong> — Computed or custom stitched output dimensions.</li>
</ul>

<h4>Tile Ordering Patterns</h4>
<ul>
  <li><strong>Row-Major</strong> — Left→Right, Top→Bottom standard ordering.</li>
  <li><strong>Column-Major</strong> — Top→Bottom, Left→Right ordering.</li>
  <li><strong>Row Serpentine</strong> — Zigzag by row (alternating L→R and R→L).</li>
  <li><strong>Column Serpentine</strong> — Zigzag by column (alternating T→B and B→T).</li>
</ul>

<h4>Stitched Preview</h4>
<ul>
  <li><strong>Composite View</strong> — Full stitched frame from all tiles.</li>
  <li><strong>Stretch Toggle</strong> — Aspect ratio correction for rectangular tiles.</li>
  <li><strong>Frame Navigation</strong> — Slider/buttons to navigate frames (if multi-frame).</li>
  <li><strong>Contrast Controls</strong> — Adjust intensity of stitched output.</li>
  <li><strong>Position Label</strong> (top-right) — X, Y, Zoom %, Rotation °.</li>
</ul>

<h4>Tile View Inspector</h4>
<ul>
  <li><strong>Tile View (▶)</strong> — Toggle panel to show individual tile inspection.</li>
  <li><strong>Tile Grid</strong> — Visual matrix of tiles with status indicators.</li>
  <li><strong>Tile Index Selector</strong> — Jump to specific tile by index.</li>
  <li><strong>Per-Tile Display</strong> — Examine selected tile content and metadata.</li>
  <li><strong>Frame Scrubber</strong> — Step through frames in tile view.</li>
</ul>

<h4>Interactive Tools</h4>
<ul>
  <li><strong>Pixel Inspector</strong> — Hover/click to view exact coordinates and values in original tile space.</li>
  <li><strong>Zoom Controls</strong> — Zoom in/out on stitched composite.</li>
  <li><strong>Grid Overlay</strong> — Display tile boundaries and grid.</li>
  <li><strong>Measure Tool</strong> — Distance measurement across tiles.</li>
</ul>

<h4>Export &amp; Data Management</h4>
<ul>
  <li><strong>Export Stitched</strong> — Save current stitched frame as PNG/BMP.</li>
  <li><strong>Batch Export</strong> — Generate stitched frames for entire frame range.</li>
  <li><strong>Format Options</strong> — Compression and bit depth for export.</li>
  <li><strong>Save Settings</strong> — Persist tile configuration for later use.</li>
</ul>

<h4>Advanced Options</h4>
<ul>
  <li><strong>Pixel Matrix Size</strong> — Odd kernel (3, 5, 7, 9, ...) for statistics.</li>
  <li><strong>Final Width / Height Overrides</strong> — Custom output dimensions.</li>
</ul>

<h4>Keyboard Shortcuts</h4>
<ul>
  <li><strong>Right / Left Arrow</strong> — Next / Previous frame.</li>
  <li><strong>Ctrl + Up / Down</strong> — Zoom in / Zoom out.</li>
  <li><strong>Ctrl + S</strong> — Save settings.</li>
  <li><strong>Ctrl + Space</strong> — Export stitched frame.</li>
  <li><strong>Shift + Q</strong> — Close tab.</li>
</ul>
""",
}

_IRIS_HTML = """
<hr/>
<h3>Iris Assistant</h3>
<ul>
  <li><strong>Iris</strong> is available in this build.</li>
  <li>Use the <strong>Iris</strong> button in the top bar for assistant-driven help and workflow support.</li>
</ul>
"""


def _detect_iris_file(main_app=None):
    candidates = []
    try:
        candidates.append(os.getcwd())
    except Exception:
        pass
    try:
        candidates.append(os.path.dirname(__file__))
    except Exception:
        pass
    if main_app is not None:
        try:
            candidates.append(os.path.dirname(os.path.abspath(main_app.session_file)))
        except Exception:
            pass

    for d in candidates:
        if not d:
            continue
        try:
            if os.path.exists(os.path.join(d, "iris.py")):
                return True
        except Exception:
            pass
    return False


def _default_help_html(mode="band", include_iris=None, main_app=None):
    key = (mode or "band").strip().lower()
    if key not in _MODE_BODIES:
        key = "band"
    if include_iris is None:
        include_iris = _detect_iris_file(main_app=main_app)

    title = _MODE_TITLES.get(key, "Help")
    body = _MODE_BODIES.get(key, "")
    iris_section = _IRIS_HTML if include_iris else ""
    return f"<h2>{title}</h2><hr/>{body}{iris_section}"


def load_help_file(path):
    try:
        if not path:
            return None, False
        if not os.path.exists(path):
            return None, False
        _, ext = os.path.splitext(path.lower())
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        if ext in (".html", ".htm"):
            return data, True
        safe = html_module.escape(data).replace("\n", "<br/>\n")
        return safe, True
    except Exception:
        return None, False


def create_help_tab(main_app=None, help_file_path=None, use_html=True, mode="band", include_iris=None):
    w = QWidget()
    layout = QVBoxLayout()
    w.setLayout(layout)

    help_text = QTextEdit()
    help_text.setReadOnly(True)
    help_text.setLineWrapMode(QTextEdit.WidgetWidth)

    file_content, file_is_html = load_help_file(help_file_path)
    if file_content:
        if use_html and file_is_html:
            help_text.setHtml(file_content)
        elif use_html and not file_is_html:
            help_text.setHtml(file_content)
        else:
            help_text.setPlainText(html_module.unescape(file_content).replace("<br/>\n", "\n"))
    else:
        if use_html:
            help_text.setHtml(_default_help_html(mode=mode, include_iris=include_iris, main_app=main_app))
        else:
            help_text.setPlainText(html_module.unescape(_default_help_html(mode=mode, include_iris=include_iris, main_app=main_app)))

    layout.addWidget(help_text)

    w._help_widget = help_text
    w._help_file_path = help_file_path
    w._help_mode = mode

    def update_help(new_text=None, as_html=True, new_mode=None):
        active_mode = new_mode or w._help_mode
        if new_text is None:
            fc, _ = load_help_file(help_file_path)
            if fc:
                if as_html:
                    help_text.setHtml(fc)
                else:
                    help_text.setPlainText(html_module.unescape(fc).replace("<br/>\n", "\n"))
            else:
                rendered = _default_help_html(mode=active_mode, include_iris=include_iris, main_app=main_app)
                if as_html:
                    help_text.setHtml(rendered)
                else:
                    help_text.setPlainText(html_module.unescape(rendered))
        else:
            if as_html:
                help_text.setHtml(new_text)
            else:
                help_text.setPlainText(new_text)

    w.update_help = update_help
    return w
