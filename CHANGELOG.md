# Change Log

## Unreleased

### Added
- Dynamic band-file discovery in `band_app.py` so the loader can handle raw, binned, and split band variants instead of only fixed band indices.
- Support for `32`-bit input across the band, raw, and video paths.
- A second tab host / secondary-screen window flow in `main.py` for multi-monitor use.
- Histogram fills and improved curve emphasis in `ui_components.py`.
- Play/pause controls and frame playback speed selection in the raw viewer.

### Improved
- Memory-aware frame loading in `band_views.py` to reduce pressure on large datasets.
- Safer raw data unpacking and display scaling in `raw_mode.py` and `utils.py`.
- Fullscreen behavior in `image_viewer.py` so extended desktops can span across the virtual display area instead of only maximizing on one screen.
- Tab lookup and Iris dataset notifications in `tiled_viewer.py`.
- More explicit error handling in `video_mode.py` when no folder is selected or no band files are found.

### Changed
- Session state is now saved using the application directory for `last_session.json`.
- Bit-depth selectors now include `32` where supported.
- The editor grid option label was simplified from `Force Grid Mode (equal-size swap only)` to `Force Grid Mode`.
- Tab visibility state is now persisted in the main app session data.

### Notes
- This repository currently contains several generated/runtime files such as `__pycache__` entries, `*.db`, and `*.json` state files. Those are usually best kept out of a release commit unless you intentionally want them versioned.
