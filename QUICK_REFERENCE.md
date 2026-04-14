# IRIS Report Generator - Quick Reference

## Quick Start -  API (Recommended)

### Generate a single report:

from iris2.meta_parser import batch_reports

# Get report for one dataset
reports = batch_reports('/home/xdlinx/Capture', pattern='Acq000001*')
for name, report in reports:
    print(report)
    # Save if needed:
    # with open(f'{name}_report.txt', 'w') as f:
    #     f.write(report)


### Generate reports with template comparison:

from iris2.meta_parser import batch_reports

# Enable template comparison to see deviations from learned baselines
reports = batch_reports('/home/xdlinx/Capture', enable_template_comparison=True)
for name, report in reports:
    print(f"=== {name} ===")
    print(report)


**Note:** Template comparison is disabled by default to avoid confusion. Only enable it if you want statistical deviation analysis.

## In  Code


from iris2.meta_parser import generate_full_report, batch_reports
from iris2.tools import tool_run_scan

# Method 1: Single dataset with scan
scan_result = tool_run_scan('/path/to/Acq000004_11091104052026', mode='quick')
if 'error' not in scan_result:
    from iris2.app_state import state
    scan_obj = state.get_scan_result('/path/to/Acq000004_11091104052026')
    report = generate_full_report(scan_obj, '/path/to/Acq000004_11091104052026')
    print(report)

# Method 2: Batch process multiple datasets
reports = batch_reports('/home/xdlinx/Capture', pattern='Acq*')
for name, report_text in reports:
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"{'='*70}")
    print(report_text)


## Report Contents

| Section | What It Contains |
|---------|-----------------|
| **0. MISSION IDENTITY** | App version, firmware, camera, CPU, disk, RAM |
| **1. PARAMETER FILE CHECK** | All 14 acquisition parameters |
| **2. REQUESTED VS APPLIED** | FPS, exposure, gain accuracy |
| **3. FRAME ACCOUNTING** | Frames captured, drops, completeness |
| **4. TEMPERATURE** | Core, sensor, grabber temps |
| **6. FPS STABILITY** | FPS mean, std dev, jitter |
| **7. CAMERA SETUP** | Camera status, regions, memory |
| **8. ORBITAL & GEOLOCATION** | Ephemeris, GPS, PPS status |
| **10. TEMPLATE STATUS** | Calibration template info |
| **11. POST-CAPTURE** | Output size, processing time |

## Status Indicators

- ✅ **OK** - Check passed
- ⚪ **INFO** - Informational message
- 🟡 **WARNING** - Issue detected but non-critical
- ❌ **ERROR** - Critical issue

## Files at a Glance

| File | Purpose |
|------|---------|
| `iris2/meta_parser.py` | Main report generation functions (integrated) |
| `iris2/agent.py` | Dataset scanner that feeds data to reports |
| `REPORT_GENERATOR_README.md` | Full documentation |
| `QUICK_REFERENCE.md` | This file |

## Troubleshooting

**Report not generating?**
- Ensure dataset folder contains `.json` and `.log` files
- Check file permissions
- Try verbose mode: `--verbose`

**Empty sections?**
- Some sections require data in `.log` file
- Check if log file is complete

**Can't find CLI?**
- Run from workspace root: `/home/xdlinx/Downloads/DisplayGoundADV`
- Or add to PATH

## Report Output Example


╔══════════════════════════════════════════════════════════════════╗
║                      IRIS ACQUISITION REPORT                     ║
║  Dataset : Acq000004_11091104052026                              ║
║  Sources : log + .meta                                           ║
║  Mission : Test Pattern Calibration                              ║
╚══════════════════════════════════════════════════════════════════╝

  HEALTH: 95/100  ✅ OK
  Summary: 0 confirmed · 1 warnings · 3 info

──────────────────────────────────────────────────────────────────────
  0. MISSION IDENTITY
──────────────────────────────────────────────────────────────────────
  App          : Xdlinx Cam App v8.0
  Firmware     : 2.4.4
  Camera       : VisLinxM
  Grabber      : Vislinx-M PCIe Camera on PCI slot {1:0:0}: Protocol 0xFFFF, Generation 2
  CPU          : Intel Atom(R) x6425E Processor @ 2.00GHz (4 cores)
  Disk free    : 52.41 GB
  RAM free     : 14.24 GB
  
... [more sections] ...

══════════════════════════════════════════════════════════════════════
  END — Acq000004_11091104052026  |  Health: 95/100  |  Mission: test_pattern_calibration
══════════════════════════════════════════════════════════════════════


## Common Commands

### View report in terminal
bash
3 iris2/report_cli.py /home/xdlinx/Capture/Acq000004_11091104052026 | less


### Save report for each dataset
bash
3 iris2/batch_report_gen.py /home/xdlinx/Capture --verbose


### Generate reports in separate folder
bash
mkdir -p ./iris_reports
3 iris2/batch_report_gen.py /home/xdlinx/Capture \
    --output-dir ./iris_reports


### Find all generated reports
bash
find /home/xdlinx/Capture -name "REPORT.txt"


## API Summary

### Main Functions

**`generate_full_report(dataset_folder: str) -> str`**
- Generates complete report as string
- Takes dataset folder path
- Returns formatted report text

**`LogParser(log_path: str)`**
- Parses `.log` file
- Properties: `data`, `get_fps_stats()`

**`ReportBuilder(dataset_name, json_data, log_parser)`**
- Builds report from parsed data
- Method: `build() -> str`

## Notes

- Reports are self-contained text files (can email, archive, etc.)
- Reports include all critical acquisition information
- Health score indicates overall acquisition quality
- Warnings flag potential issues for review
- Reports are human-readable and machine-parseable

## Next Steps

1. ✅ Generate reports for all datasets
2. ✅ Review warnings and errors
3. ✅ Archive reports with datasets
4. ✅ Integrate with data processing pipeline
