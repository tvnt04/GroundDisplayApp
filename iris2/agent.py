from __future__ import annotations
import json
import os
import re
import base64
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from .app_state import state
from .tools import TOOL_SCHEMAS, TOOL_DISPATCH
from .local_engine import analyze, build_context_for_llm
from .ollama import (
    is_available as ollama_available,
    ask_iris,
    classify_intent,
    chat,
    FAST_MODEL,
)


# ── Model ─────────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-6"

try:
    import anthropic as _sdk
    _HAS_SDK = True
except ImportError:
    _sdk = None
    _HAS_SDK = False


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """You are Iris — an expert AI agent embedded in DisplayGroundX, \
a multispectral satellite and sensor imagery analysis application.

## YOUR IDENTITY
You are not a chatbot. You are an active participant in the application. \
You can see everything that's open, read files directly, run complete pixel scans, \
and control the application — opening datasets, navigating frames, zooming viewers. \
You act; you don't just advise.

## SENSOR KNOWLEDGE

### Capture pipeline
Sensor → Camera driver → Frame grabber → Region extraction → Packed raw band files + metadata → Dataset folder

### Sensor baseline (flag deviations)
- Width: 8448 px | Height: 384 px (RegionHeight)
- Bit depth: 10-bit packed (5 bytes per 4 pixels — NOT 2 bytes/pixel)
- Bands: 7 spectral bands | Typical frame count: ~500

### Band file naming
- `.band0`-`.band6`  -> full band files
- `.band20`, `.band21` -> band 2 left/right halves (width = full_width / 2)
- `.band22`           -> band 2 binned

### ProcMode string (14 space-separated fields)
OrbitID TaskID JsonID Date Time Duration BandSelection TDI FPS ExposureTime Gain XShift Binning TDIYShift

TDI byte: 0=OFF, 10=ON/2-stage, 18=ON/4-stage, 34=ON/8-stage, 66=ON/64-stage
BandSelection: bits 6-0, one per band, 1=active (127=all 7)
Binning byte: bit7=CCSDS, bits 6-0=per-band binning (1=binned, 0=unbinned)

## HOW TO REASON ABOUT FINDINGS

### The golden rule
The scanner flags what it sees in pixels. You decide what it means.
A flag is an observation. Your job is to explain whether it is:
  (a) a real defect requiring action
  (b) an expected artifact of the capture configuration
  (c) inconclusive - needs more context or a scene-data capture to confirm

Every finding in the scan results has a context_note field added by the scanner.
READ IT. It already contains the reasoned explanation - use it, don't ignore it.

### Context checklist - run through this before concluding anything

**Before calling something a hardware defect:**
  - Are ALL frames affected, or just some? (all = systematic, some = transient)
  - Was the camera covered or shutter closed? (all-black = not a sensor failure)
  - Did any parameter FAIL to apply? (XShift fail -> apparent dead columns may be misalignment)
  - Is the pattern consistent with the TDI configuration?

**Black frames:**
  - ALL frames black -> camera covered, dark reference, or shutter test. Say so. Never diagnose hardware failure.
  - SOME frames black -> real mid-sequence dropout. Flag as confirmed critical.

**Dead columns:**
  - BandXShift FAILED: "May be misaligned fill pixels due to XShift failure. Retry with correct shift before concluding hardware failure."
  - XShift OK AND valid scene data: real column amplifier failure.

**Alternating light/dark row pattern:**
  - TDI ON: "Observed pattern consistent with TDI phase alignment and dual-ADC even/odd row interleaving. Not necessarily a defect. Verify TDI Y-shift and ADC calibration if radiometric accuracy is required."
  - TDI OFF: "ADC even/odd channel gain mismatch. Flat-field calibration required."

**Vertical striping:**
  - All bands, all frames: ADC fixed-pattern noise. Flat-field correction removes it. Not a defect.
  - Specific bands only: band-specific ADC or readout issue.
  - Near-zero DN data: unreliable, camera may have been covered.

**Saturation:**
  - High gain (>1.5x) or long exposure: expected on bright targets. Suggest reducing settings if unwanted.
  - Normal settings: genuine hot spot.

**Cross-band outlier:**
  - Band is BINNED per binning byte: expected - binned bands have different mean DN. Not a defect.
  - Band not binned but reads differently: degraded spectral channel.

**Truncated file:**
  - CapturedCount = TotalFrames in log: disk write/transfer failure, not a sensor issue.
  - CapturedCount < TotalFrames: both capture drop and file truncation.

**Trigger timing [W01]:**
  - Always state clearly: "Parameter file had a stale UTC timestamp. GPS/orbital sync did NOT occur. Capture fell back to 5-second default delay. Timing metadata for this dataset is unreliable."

**Parameter mismatches:**
  - FAILED: camera rejected the value. State what was requested, applied, and what it means for data quality.
  - AUTO_ADJUSTED: hardware limit reached. Always explain the practical implication.

### How to structure your answer
1. State what was **observed** — never what was concluded.
2. Confirmed log-level issues first (frame drops, parameter failures, timing sync failure) — these are real.
3. Pixel-level observations second — always qualified: "observed", "noted", "possible", "requires scene data to confirm".
4. Informational items last (dark capture notice, expected TDI patterns).

**Severity translation for your responses:**
- CRITICAL finding → "Confirmed issue: ..."
- WARNING finding → "Observed pattern: ... (confirm with scene data)"
- INFO finding → "Noted: ... (expected / inconclusive)"

**Health score language:**
- 90-100 → "No issues observed"
- 70-89  → "Minor patterns noted"
- 50-69  → "Several patterns observed — scene capture recommended"
- 30-49  → "Log issues confirmed, pixel observations inconclusive"
- Never say "0%", "failed", or "critical health" for any dataset

### What you never do
- Never say "hardware failure", "dead", "defect", "column amplifier failure" as a conclusion — only as a possibility requiring confirmation.
- Never diagnose pixel findings on a dark capture dataset — only report the mean DN and say "re-evaluate with scene data."
- Never call INFO findings "warnings" or "issues" in your response.
- Never translate an observed pattern into a verdict.
- Never list raw flag names — translate: "alternating_row_banding" → "alternating light/dark row pattern."
- Never repeat the same finding twice.
- Never say a dataset "scored 0%" — minimum is 30%.

## HOW YOU WORK

### Default behaviour — always do this first
1. Call `get_app_state` to see what is open and whether scans exist.
2. If context shows `SCAN CACHE INVALIDATED` → call `refresh_scan` immediately without asking.
3. If user says "data changed", "I fixed the resolution", "re-scan", "scan again" → call `refresh_scan`, not `run_scan`.

### Reporting — active tab by default
- Report on the **active tab** unless the user specifies another dataset.
- Do not ask which dataset to report on when there is only one open.
- If multiple tabs are open and it is ambiguous, call `list_open_datasets` and emit a CHOICE block so the user can pick.

### Compare flow
When user says "compare" and multiple datasets are open:
1. Call `list_open_datasets` to get current open datasets.
2. Emit a CHOICE block with mode=multi so user can pick which two to compare.
3. After user picks, scan both if needed, then call `compare_datasets`.

When user says "compare" and only one dataset is open:
- Say "I only see one dataset open. Which second dataset should I compare to?"
- Offer to open a new folder or ask user to load one first.

### No dataset loaded
- Never stop working. If no dataset is open and user asks to scan:
  1. Say "No dataset is loaded. Please select a folder to scan."
  2. Emit a CHOICE block asking user to specify the folder path, OR
  3. Ask the user to paste the folder path directly.
- Never say "I can't help" — always offer the next step.

### Folder tree scanning
- When user says "scan this folder" and points to a parent directory with multiple datasets:
  - Call `browse_folder_tree` with the root folder.
- When user says "logs only" or "just check the logs":
  - Call `browse_folder_tree` with logs_only=true, OR call `read_logs` for a single folder.

### Scanning
- Default: `run_scan` mode="full" for a fresh scan of a specific folder.
- `refresh_scan`: when results are stale or user signals data changed.
- `browse_folder_tree`: when user points to a directory containing multiple datasets.
- mode="quick" only if user explicitly says so.
- After scan: report health score, confirmed issues, observed patterns — in that order.

### Interactive pickers — CHOICE block format
When you need the user to choose from a list, emit this exact format at the end of your message:

For single selection:
[CHOICE:single|prompt="Which dataset should I report on?"|opts="DatasetA||DatasetB||DatasetC"]

For multi selection:
[CHOICE:multi|prompt="Which datasets do you want to compare? (select two)"|opts="DatasetA||DatasetB||DatasetC"]

Rules for CHOICE blocks:
- Always place the CHOICE block at the END of your message, after any text.
- Use `list_open_datasets` to get the actual dataset names before building the opts list.
- Only emit one CHOICE block per message.
- Never emit a CHOICE block when there is only one option.

### Multiple open tabs
- `get_app_state` shows all tabs. Say which dataset you are analyzing if ambiguous.
- Use `compare_datasets` for side-by-side. Scan both first if needed.

## COMMUNICATION RULES
1. One question -> one direct answer with specific values.
2. Plain language always. Translate every technical flag name.
3. For each finding: what it is, which band, which frames, cause, real problem or expected.
4. If downgraded to INFO by context: say "observed but expected given [reason]" not "warning."
5. Short greetings: one sentence only.
6. No filler. No "Certainly!" No repeating the question back.
7. Frame numbers as **NUMBER** so they render highlighted.
8. Technical reports must be anomaly-only: never spend lines on "everything else is normal".
9. If evidence is weak, say "possible" or "inconclusive" instead of asserting hardware failure.
10. Be very deatiled and give full report with all noteable observations 
11. Do not restate UI sections unless the user asks for a walkthrough.
12. Prefer tool calls and concrete answers over speculation.

## MEMORY & PERSISTENT SCAN CACHE

**Scan cache (automatic, local SQLite)**
Every scan result is saved to `iris_scans.db` after it completes. When a dataset
is opened again, the result loads instantly — no re-scanning, no pixel reading,
no tokens spent on raw file content. Never re-scan a dataset that has a valid
cache entry unless the user explicitly asks.

**Long-term memory (Supabase, semantic)**
Key findings are stored as embeddings. You can recall them by meaning.

What to do automatically:
- At the start of any analysis, call `recall_memory` with the dataset name —
  if a memory exists, surface it: "I've seen this dataset before — [finding].
  Shall I use the cached result or re-scan?"
- After any CRITICAL finding, call `save_memory` to preserve it for future sessions.
- When user says "remember this", "save this finding", "note this down".

What you never do:
- Never re-scan when a cached result exists (unless user asks).
- Never ask user to re-upload logs you've already processed.
- Never save INFO-level observations as memories — only significant findings.

## KNOWLEDGE BASE

You have a semantic knowledge base of indexed reference files — sensor specs,
camera manuals, SOPs, calibration guides.

When to search:
- Any question about specific hardware limits, parameter ranges, or sensor specs.
- Before making any claim about a hardware specification — search first, don't guess.
- When the user asks "what does the manual say about X".
- When a scan finding needs context from specs.

What you never do:
- Never invent hardware specifications — search or say you don't know.
- Never say "according to my training data" when you have a knowledge base.

## HISTOGRAM VIEWER AWARENESS

You always know what the Histogram tab is showing. The current histogram state is
injected into every message via CURRENT APPLICATION CONTEXT. When it is present, use it.

When the user asks anything about the histogram, pixel values, exposure, DN levels,
saturation, dynamic range, or what a band looks like — call `get_histogram_state`
to get the full current state including per-band statistics and derived observations.

**How to interpret histogram state:**
- `frame_pixel_min` / `frame_pixel_max` — actual pixel value range in the current frame.
  Compare against `axis_max` (e.g. 1023 for 10-bit) to assess dynamic range utilisation.
- `band_stats.mean` — average DN per band. Cross-band outliers suggest a binned band,
  an inactive band, or a degraded spectral channel.
- `saturated_pct > 1%` — band is clipping at the sensor ceiling.
- `black_pct > 5%` — band has near-zero pixels. Camera may be covered, or band inactive.
- `visible_bands` — only the bands the user has checked on in the legend.

**What you never do with histogram data:**
- Never report a mean DN and call it "the pixel value" — it's an average across the frame.
- Never diagnose saturation from a dark capture — say "re-evaluate with scene data."
- Never say a band is "dead" based on histogram alone — it may be binned or hidden.

## CURRENT APPLICATION CONTEXT
{CONTEXT}
"""

_cached_system: str = ""
_cached_context_key: str = ""


def build_system_prompt(force: bool = False) -> str:
    """
    Build system prompt with live application context injected.
    Cached within a request — call with force=True at the start of each
    new user message to pick up the latest tab/scan state.
    """
    global _cached_system, _cached_context_key
    ctx = state.context_for_claude()
    if not force and ctx == _cached_context_key and _cached_system:
        return _cached_system
    _cached_context_key = ctx
    _cached_system = _SYSTEM.replace("{CONTEXT}", ctx)
    return _cached_system


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_question(messages: List[Dict]) -> str:
    """Pull text content from the last user message."""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                for block in c:
                    if block.get("type") == "text":
                        return block["text"]
            elif isinstance(c, str):
                return c
    return ""


def _build_history(messages: List[Dict]) -> List[Dict]:
    """
    Convert the messages list (minus the last user turn) into a clean
    [{"role": ..., "content": str}] history for Ollama.
    Image blocks are stripped — local modes are text-only.
    """
    history = []
    for m in messages[:-1]:
        role    = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(b["text"] for b in content if b.get("type") == "text")
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    return history


# ── Claude API worker ─────────────────────────────────────────────────────────

class IrisAgentWorker(QThread):
    """Online mode — full Claude API with tool use."""
    token     = pyqtSignal(str)
    tool_call = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed    = pyqtSignal(str)

    def __init__(self, api_key: str, messages: List[Dict],
                 image_bytes: Optional[bytes] = None,
                 folder: str = "",
                 parent=None):
        super().__init__(parent)
        self._key      = api_key
        self._messages = [dict(m) for m in messages]
        self._image    = image_bytes
        self._folder   = folder

    def run(self):
        try:
            self.completed.emit(self._agent_loop())
        except Exception as e:
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg.lower():
                self.failed.emit("Invalid API key. Use /apikey to set it.")
            elif "429" in msg:
                self.failed.emit("Rate limit. Wait a moment and try again.")
            elif "529" in msg or "overloaded" in msg.lower():
                self.failed.emit("API overloaded. Try again in a moment.")
            else:
                self.failed.emit(f"Error: {msg[:300]}")

    def _agent_loop(self) -> str:
        messages = list(self._messages)

        # Inject screenshot if provided
        if self._image:
            b64 = base64.b64encode(self._image).decode("ascii")
            img_block = {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}}
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    c = messages[i]["content"]
                    messages[i]["content"] = (
                        [{"type": "text", "text": c}, img_block]
                        if isinstance(c, str) else c + [img_block])
                    break

        final_text = ""

        # ── Ensure AppState knows the active folder ───────────────────────
        # If the event bus didn't fire (e.g. user loaded dataset manually before
        # Iris was ready), state.active_folder is empty and every tool fails.
        # Force-register the folder from panel so tools can resolve it.
        if self._folder and os.path.isdir(self._folder):
            if not state.active_folder:
                with state._lock:
                    # Register as session folder so _resolve_folder() finds it
                    if self._folder not in state._session_folders:
                        state._session_folders.append(self._folder)
                    # Create a minimal tab so active_folder works
                    if not state._tabs:
                        from .app_state import TabState as _TS
                        idx = 0
                        state._tabs[idx] = _TS(
                            tab_index=idx, mode="band",
                            folder=self._folder,
                            dataset_name=os.path.basename(self._folder),
                        )
                        state._active_tab_index = idx

        # Build system prompt — now has correct context because folder is registered
        system = build_system_prompt(force=True)

        for _loop in range(12):
            resp        = self._call_api(messages, system)
            stop_reason = resp.get("stop_reason", "end_turn")
            content     = resp.get("content", [])

            text_parts = []
            tool_uses  = []

            for blk in content:
                btype = blk.get("type", "")
                if btype == "text":
                    t = blk.get("text", "")
                    if t:
                        text_parts.append(t)
                elif btype == "tool_use":
                    tool_uses.append({
                        "id":    blk.get("id", ""),
                        "name":  blk.get("name", ""),
                        "input": blk.get("input", {}),
                    })

            if stop_reason == "end_turn" or not tool_uses:
                final_text = "\n".join(text_parts).strip()
                for word in final_text.split(" "):
                    self.token.emit(word + " ")
                break

            # Build assistant turn
            asst_content = []
            for blk in content:
                btype = blk.get("type", "")
                if btype == "text":
                    asst_content.append({"type": "text", "text": blk.get("text", "")})
                elif btype == "tool_use":
                    asst_content.append({
                        "type":  "tool_use",
                        "id":    blk.get("id", ""),
                        "name":  blk.get("name", ""),
                        "input": blk.get("input", {}),
                    })
            messages.append({"role": "assistant", "content": asst_content})

            # Execute tools
            tool_results = []
            for tc in tool_uses:
                name = tc["name"]
                self.tool_call.emit(name)
                result = self._exec_tool(name, tc["input"])
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": tc["id"],
                    "content":     json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_results})

        return final_text or "(No response)"

    def _call_api(self, messages: List[Dict], system: str) -> Dict:
        return self._call_sdk(messages, system) if _HAS_SDK else self._call_http(messages, system)

    def _call_sdk(self, messages: List[Dict], system: str) -> Dict:
        client = _sdk.Anthropic(api_key=self._key)
        resp   = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        content = []
        for blk in resp.content:
            btype = getattr(blk, "type", "")
            if btype == "text":
                content.append({"type": "text", "text": blk.text})
            elif btype == "tool_use":
                content.append({"type": "tool_use", "id": blk.id,
                                 "name": blk.name, "input": blk.input})
        return {"stop_reason": resp.stop_reason, "content": content}

    def _call_http(self, messages: List[Dict], system: str) -> Dict:
        payload = json.dumps({
            "model": CLAUDE_MODEL,
            "max_tokens": 4096,
            "system": system,
            "tools": TOOL_SCHEMAS,
            "messages": messages,
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())

    def _exec_tool(self, name: str, inputs: Dict) -> Dict:
        fn = TOOL_DISPATCH.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(inputs)
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}


# ── Ollama worker — handles both Local and Smart modes ────────────────────────

class IrisOllamaWorker(QThread):
    """
    Local + Smart mode worker powered by Ollama.

    Fast path routing via classify_intent() — zero extra LLM calls:
      "casual"   → skip all context building, straight to Ollama  (~50ms)
      "question" → build scan context if available, Ollama answers (~100ms)
      "task"     → run rule engine / tool if needed, then Ollama   (~200ms+)

    mode="local"  — Ollama only, never touches the API.
    mode="smart"  — same as local, but emits needs_api if user explicitly
                    requests Claude or if question is far beyond Ollama.
    """
    token     = pyqtSignal(str)
    tool_call = pyqtSignal(str)
    completed = pyqtSignal(str)
    failed    = pyqtSignal(str)
    needs_api = pyqtSignal(str)   # smart mode: escalate to Claude API

    def __init__(
        self,
        messages: List[Dict],
        folder:   str  = "",
        api_key:  str  = "",
        mode:     str  = "local",   # "local" | "smart"
        allow_api: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._messages  = messages
        self._folder    = folder
        self._api_key   = api_key
        self._mode      = mode
        self._allow_api = allow_api

    def run(self):
        try:
            self._dispatch()
        except Exception as e:
            self.failed.emit(f"Ollama worker error: {e}")

    # ── Internal ──────────────────────────────────────────────────────────────

    # ── Routing system prompt — used in Smart mode only ──────────────────────
    #
    # Ollama sees the question + scan context and replies with ONE of:
    #   QUERY_MEMORY   — needs past session data from Supabase memory
    #   QUERY_KB       — needs sensor specs / manuals from knowledge base
    #   QUERY_BOTH     — needs both
    #   ESCALATE       — question needs Claude API (deep reasoning / planning)
    #   ANSWER: <text> — can answer directly with what it has
    #
    # Using FAST_MODEL (gemma3:4b) — classification only, ~1 s, deterministic.
    _ROUTING_SYSTEM = (
        "You are a routing assistant for Iris, an AI inside a satellite camera analysis tool.\n"
        "You will receive a question and any available scan context.\n"
        "Decide what additional information is needed, or if Claude API is required.\n\n"
        "Reply with EXACTLY ONE of these tokens (nothing else, no explanation):\n\n"
        "  QUERY_MEMORY   — you need past session findings, saved anomalies, or dataset history\n"
        "                   stored in long-term memory to answer well\n"
        "  QUERY_KB       — you need sensor specs, firmware docs, SOPs, or technical manuals\n"
        "                   from the knowledge base to answer well\n"
        "  QUERY_BOTH     — you need both memory and knowledge base\n"
        "  ESCALATE       — the question requires deep multi-step reasoning, a detailed\n"
        "                   improvement plan, root-cause analysis across multiple datasets,\n"
        "                   or nuanced technical judgment beyond what you can provide\n"
        "  ANSWER: <text> — you can answer fully with the context already provided\n\n"
        "Rules:\n"
        "- If scan data is provided and the question is a simple factual lookup → ANSWER\n"
        "- If user asks about past sessions, previous findings, history → QUERY_MEMORY\n"
        "- If user asks about limits, specs, firmware, max/min values → QUERY_KB\n"
        "- If you need both session history AND spec knowledge → QUERY_BOTH\n"
        "- If the question needs a strategic recommendation or deep root-cause → ESCALATE\n"
        "- Simple questions about current scan data → ANSWER directly\n\n"
        "Output ONLY the token or ANSWER: <text> — no other text."
    )

    _TASK_PLANNER_SYSTEM = (
        "You are an action planner for Iris. "
        "Detect whether the user is asking for a direct app action.\n"
        "Return strict JSON only with keys: tool, confidence, args.\n"
        "Allowed tools: open_dataset, run_scan, generate_report, none.\n"
        "Rules:\n"
        "- If user gives a folder path to open -> open_dataset with args.folder_path\n"
        "- If user asks scan/check dataset -> run_scan\n"
        "- If user asks generate/show report -> generate_report\n"
        "- If no clear action request -> none\n"
        "confidence is a float 0..1."
    )

    def _route(self, question: str, context: str) -> str:
        """
        Ask Ollama (fast model, non-streaming) how to handle this question.

        Returns one of: "QUERY_MEMORY", "QUERY_KB", "QUERY_BOTH",
                        "ESCALATE", "ANSWER"

        Uses gemma3:4b — ~1 s, temperature=0 for deterministic routing.
        Falls back to "ANSWER" on any error so the conversation never breaks.
        """
        from .ollama import chat, FAST_MODEL
        user_msg = (
            f"Scan context:\n{context}\n\nQuestion: {question}" if context
            else f"Question: {question}"
        )
        try:
            raw = chat(
                messages=[{"role": "user", "content": user_msg}],
                system=self._ROUTING_SYSTEM,
                temperature=0.0,
                stream=False,
                model=FAST_MODEL,
            ).strip()
            token = raw.upper().split()[0] if raw else ""
            if token in ("QUERY_MEMORY", "QUERY_KB", "QUERY_BOTH", "ESCALATE"):
                return token
            return "ANSWER"
        except Exception:
            return "ANSWER"

    def _plan_task(self, question: str) -> Dict:
        try:
            raw = chat(
                messages=[{"role": "user", "content": question}],
                system=self._TASK_PLANNER_SYSTEM,
                temperature=0.0,
                stream=False,
                model=FAST_MODEL,
            ).strip()
            plan = json.loads(raw) if raw else {}
            tool = str(plan.get("tool", "none")).strip()
            conf = float(plan.get("confidence", 0.0))
            args = plan.get("args", {}) or {}
            if tool not in {
                "open_dataset", "run_scan", "generate_report", "none",
            }:
                tool = "none"
            return {"tool": tool, "confidence": conf, "args": args}
        except Exception:
            return {"tool": "none", "confidence": 0.0, "args": {}}

    def _extract_path_from_question(self, question: str) -> str:
        """
        Extract a file path from the question text.
        Looks for patterns like /path/to/folder or absolute paths.
        Returns the first match or empty string.
        """
        import re

        # Try quoted path first (handles spaces):
        quoted = re.search(r"['\"]([^'\"]+/[^'\"]+)['\"]", question)
        if quoted:
            path = quoted.group(1).strip()
            if os.path.isdir(path):
                return path

        # Unix absolute paths (with optional spaces if quoted)
        match_unix = re.search(r"(/[^\s'\"]+)", question)
        if match_unix:
            path = match_unix.group(1).strip('"\'')
            if os.path.isdir(path):
                return path

        # Windows absolute paths (C:\... style)
        match_win = re.search(r"([A-Za-z]:\\[^\s'\"]+)", question)
        if match_win:
            path = match_win.group(1).strip('"\'')
            if os.path.isdir(path):
                return path

        return ""

    def _try_llm_task_execution(self, question: str, folder: str) -> bool:
 
        q = question.lower()
        
        # Try to extract path from question if not already provided
        extracted_path = self._extract_path_from_question(question)
        effective_folder = extracted_path or folder

        # Direct keyword intercept — no Ollama planner needed
        _REPORT_KW = ("report", "give report", "show report", "findings",
                      "what's wrong", "whats wrong", "summarize", "summary",
                      "analyze", "analyse", "full report", "what happened")
        _SCAN_KW   = ("scan", "run scan", "re-scan", "rescan",
                      "quick scan", "full scan", "check this", "health check")

        _PARAM_KW = ()
        if any(k in q for k in _PARAM_KW):
            if not effective_folder:
                self.completed.emit("No dataset provided. Either load a folder first or provide a path in your request.")
                return True
            self.tool_call.emit("extract_all_parameters")
            pr = TOOL_DISPATCH["extract_all_parameters"]({"root_folder": effective_folder})
            if pr.get("error"):
                self.completed.emit(f"Parameter extraction failed: {pr['error']}")
                return True

            stats = pr.get("parameter_stats", {})
            def get_param(p):
                return stats.get(p, {})

            if "least fps" in q or "lowest fps" in q:
                candidate = get_param("fps_applied") or get_param("fps_requested") or get_param("fps_capped")
                if candidate:
                    self.completed.emit(f"Lowest FPS recorded: {candidate.get('min')} (from {candidate.get('count')} entries)")
                else:
                    self.completed.emit("No FPS data found.")
                return True

            if "average fps" in q or "mean fps" in q:
                candidate = get_param("fps_applied") or get_param("fps_requested") or get_param("fps_capped")
                if candidate:
                    self.completed.emit(f"Average FPS: {candidate.get('mean')} (min {candidate.get('min')}, max {candidate.get('max')})")
                else:
                    self.completed.emit("No FPS data found.")
                return True

            if "exposure" in q:
                candidate = get_param("exposure_applied_us") or get_param("exposure_requested_us")
                if candidate:
                    self.completed.emit(f"Exposure stats (µs): mean {candidate.get('mean')}, min {candidate.get('min')}, max {candidate.get('max')}")
                else:
                    self.completed.emit("No exposure data found.")
                return True

            if "gain" in q:
                candidate = get_param("gain_db")
                if candidate:
                    self.completed.emit(f"Gain stats (dB): mean {candidate.get('mean')}, min {candidate.get('min')}, max {candidate.get('max')}")
                else:
                    self.completed.emit("No gain data found.")
                return True

            if "kelvin" in q or "temperature" in q:
                sens = get_param("sensor_temp_c")
                if sens:
                    meank = round(sens.get("mean",0) + 273.15,2)
                    self.completed.emit(f"Sensor temp mean: {sens.get('mean')}°C = {meank} K (min {sens.get('min')}°C, max {sens.get('max')}°C)")
                else:
                    self.completed.emit("No sensor temp data found.")
                return True

            # Generic response when no specific token recognized
            self.completed.emit(
                "Parameter stats summary: "
                + ", ".join([f"{k}: mean={v.get('mean')}" for k,v in list(stats.items())[:5]])
                + "..."
            )
            return True

        _LOG_KW = ()
        if any(k in q for k in _LOG_KW):
            if not effective_folder:
                self.completed.emit("No dataset provided. Either load a folder first or provide a path in your request.")
                return True
            self.tool_call.emit("session_report")
            result = TOOL_DISPATCH["session_report"]({"root_folder": effective_folder})
            if result.get("error"):
                self.completed.emit(f"Log report failed: {result['error']}")
            else:
                self.completed.emit(result.get("report", "Session log report generated."))
            return True

        if any(k in q for k in _REPORT_KW):
            if not effective_folder:
                self.completed.emit("No dataset provided. Either load a folder first or provide a path in your request.")
                return True
            self.tool_call.emit("generate_report")
            result = TOOL_DISPATCH["generate_report"]({"folder": effective_folder})
            if result.get("error"):
                self.completed.emit(f"Report failed: {result['error']}")
            else:
                self.completed.emit(result.get("report", "Report generated."))
            return True

        if any(k in q for k in _SCAN_KW):
            if not effective_folder:
                self.completed.emit("No dataset provided. Either load a folder first or provide a path in your request.")
                return True
            mode = "quick" if "quick" in q else "full"
            self.tool_call.emit("run_scan")
            result = TOOL_DISPATCH["run_scan"]({"folder": effective_folder, "mode": mode})
            if result.get("error"):
                self.completed.emit(f"Scan failed: {result['error']}")
                return True
            rep = TOOL_DISPATCH["generate_report"]({"folder": effective_folder})
            self.completed.emit(rep.get("report") or
                f"Scan done. Health: {result.get('health_score','?')}/100, "
                f"{result.get('anomaly_count',0)} anomalies.")
            return True

        plan = self._plan_task(question)
        tool = plan.get("tool", "none")
        conf = float(plan.get("confidence", 0.0))
        args = dict(plan.get("args", {}) or {})
        if tool == "none" or conf < 0.70:
            return False

        fn = TOOL_DISPATCH.get(tool)
        if not fn:
            return False

        if tool in ("run_scan", "generate_report") and not args.get("folder"):
            args["folder"] = effective_folder

        self.tool_call.emit(tool)
        result = fn(args)
        if result.get("error"):
            self.completed.emit(f"{tool} failed: {result['error']}")
            return True

        # For scan actions, return the full structured report immediately.
        if tool == "run_scan":
            report_fn = TOOL_DISPATCH.get("generate_report")
            report_folder = args.get("folder") or effective_folder or state.active_folder or ""
            if report_fn and report_folder:
                rep = report_fn({"folder": report_folder})
                if rep.get("report"):
                    self.completed.emit(rep["report"])
                    return True
                if rep.get("error"):
                    self.completed.emit(
                        f"Scan completed, but report generation failed: {rep['error']}"
                    )
                    return True

        # Best-effort learning: store successful action + phrasing in memory.
        try:
            save_fn = TOOL_DISPATCH.get("save_memory")
            if save_fn:
                save_fn({
                    "title": f"Action learned: {tool}",
                    "detail": f"User said: {question}\nResult: {result}",
                    "memory_type": "note",
                    "dataset": os.path.basename(effective_folder) if effective_folder else "",
                    "tags": ["llm_action", tool],
                })
        except Exception:
            pass

        if tool == "generate_report":
            txt = result.get("report", "") or result.get("message", "Report generated.")
            self.completed.emit(txt)
            return True

        msg = (
            result.get("message")
            or result.get("status")
            or f"Executed {tool}."
        )
        self.completed.emit(msg)

        return True

    def _dispatch(self):
        question = _extract_question(self._messages)
        history  = _build_history(self._messages)

        # ── Explicit API request — user typed "ask claude" etc. ───────────────
        if self._mode == "smart" and self._allow_api and self._api_key:
            q_lower = question.lower()
            if any(k in q_lower for k in ("ask claude", "use api", "get better answer", "online")):
                self.needs_api.emit(question)
                return

        # ── Zero-cost intent classification ───────────────────────────────────
        intent = classify_intent(question)

        from .ollama import available_models as _avail_fn
        if not _avail_fn():
            self._fallback_no_ollama(question, intent)
            return

        # ── Casual — skip all routing, just chat ──────────────────────────────
        if intent == "casual":
            self._stream(question, context="", history=history, action="casual")
            return

        folder = self._folder or state.active_folder or ""

        # ── Task: LLM action planner for free-language commands ──────────────
        if intent == "task":
            if self._try_llm_task_execution(question, folder):
                return   # action executed, report/confirmation already emitted
            # LLM planner didn't fire (low confidence / no match) → rule engine
            context = self._handle_task(question, folder, history)
            if context is None:
                return   # rule engine emitted directly (shouldn't happen here)
        else:
            context = build_context_for_llm(folder) if folder else ""

        # ── Smart mode: Ollama routes question/analysis requests ──────────────
        #
        # Only runs for intent == "question" (or task that fell through).
        # Scan/report tasks are already handled above — they never reach here.
        # Routing adds ~1 s but only on analytical questions, never on actions.
        #
        #   QUERY_MEMORY → fetch past session findings from Supabase memory
        #   QUERY_KB     → fetch sensor specs / manuals from knowledge base
        #   QUERY_BOTH   → fetch both
        #   ESCALATE     → hand off to Claude API (deep reasoning needed)
        #   ANSWER       → stream directly, no Supabase needed
        #
        if self._mode == "smart" and self._allow_api and self._api_key and intent == "question":
            self.tool_call.emit("routing_check")
            route = self._route(question, context)

            if route == "ESCALATE":
                self.tool_call.emit("escalating_to_api")
                self.needs_api.emit(question)
                return

            dataset_name = os.path.basename(folder) if folder else ""
            mem_query    = f"{dataset_name} {question}".strip() if dataset_name else question

            if route in ("QUERY_MEMORY", "QUERY_BOTH"):
                try:
                    from .retrieval import recall_memory
                    mem = recall_memory(query=mem_query, top_k=3, threshold=0.35)
                    if mem.get("total_found", 0) > 0:
                        context = mem["context_text"] + "\n\n" + context
                        self.tool_call.emit("recall_memory")
                except Exception:
                    pass

            if route in ("QUERY_KB", "QUERY_BOTH"):
                try:
                    from .ollama import search_local_kb_fast
                    kb_ctx = search_local_kb_fast(question, top_k=4)
                    if kb_ctx:
                        context = kb_ctx + "\n\n" + context
                        self.tool_call.emit("knowledge_search")
                except Exception:
                    # Fallback: Supabase pgvector (requires network + credentials)
                    try:
                        from .retrieval import search_knowledge
                        kb = search_knowledge(query=question, top_k=3, threshold=0.32)
                        if kb.get("total_found", 0) > 0:
                            context = kb["context_text"] + "\n\n" + context
                            self.tool_call.emit("knowledge_search")
                    except Exception:
                        pass

        # ── Stream the answer ──────────────────────────────────────────────────
        self._stream(question, context=context, history=history, action=intent)


    def _handle_task(self, question: str, folder: str, history: List[Dict]) -> Optional[str]:
        """
        Handle a task-intent message.
        Returns context string to pass to Ollama, or None if already emitted.

        Routing logic:
        - Local mode:  emit the structured report directly — no LLM rewrite.
        - Smart mode:  run the rule engine, fetch KB + memory context from
                       Supabase, then pass everything to Ollama so it produces
                       a narrative explanation on top of the structured data.
        - Online mode: return the structured report as context — Claude receives
                       it as a tool result and writes its own narrative with
                       full knowledge-base access via the knowledge_search tool.
        """
        q_lower = question.lower()

        is_report_request = any(k in q_lower for k in (
            "report", "findings", "what's wrong", "whats wrong",
            "analyze", "analyse", "summary", "summarize",
        ))
        is_scan_request = any(k in q_lower for k in (
            "scan", "re-scan", "rescan", "refresh scan", "run scan",
            "check this", "health",
        ))
        needs_scan = is_report_request or is_scan_request

        scan    = state.get_scan_result(folder) if folder else None
        context = build_context_for_llm(folder) if folder else ""

        # Run scan if needed and not already cached
        if needs_scan and folder and not scan:
            self.tool_call.emit("run_scan")
            tool_fn = TOOL_DISPATCH.get("run_scan")
            if tool_fn:
                scan_mode = "full" if any(k in q_lower for k in (
                    "pixel", "pixels", "frame", "frames", "full", "deep", "anomaly frames"
                )) else "quick"
                tool_fn({"folder": folder, "mode": scan_mode})
            scan = state.get_scan_result(folder)

        # Build engine result
        engine_result = analyze(folder, question) if folder else {}
        report_text   = engine_result.get("report_text", "")

        # ── Local mode: emit structured report directly, no LLM ──────────
        # In local mode the structured output IS the final answer.
        if self._mode == "local":
            if is_report_request and report_text:
                self.completed.emit(report_text)
                return None
            if engine_result.get("answer"):
                context += f"\n\nRule answer:\n{engine_result['answer']}"
            return context

        # ── Smart / Online: emit structured report, append short Ollama reasoning ──

        if is_report_request and report_text:
            # Emit the full structured report immediately
            self.completed.emit(report_text)

            # Build a compact findings summary for Ollama (not the full report)
            if scan and scan.findings:
                criticals = [f for f in scan.findings if f["severity"] == "CRITICAL"]
                warnings  = [f for f in scan.findings if f["severity"] == "WARNING"]
                top = (criticals + warnings)[:5]
                findings_summary = "; ".join(f.get("message","")[:80] for f in top)
                dataset_name = os.path.basename(folder) if folder else "dataset"
                short_prompt = (
                    f"{dataset_name} — health {scan.health_score:.0f}/100. "
                    f"Top findings: {findings_summary}. "
                    f"In 2-3 sentences: what is the most important thing the engineer "
                    f"should know and do? Plain text only, no bullet points."
                )
                # Stream Ollama reasoning as a follow-up 
                try:
                    from .ollama import ask_iris as _ask
                    tokens = []
                    def _tok(t):
                        tokens.append(t)
                        self.token.emit(t)
                    self.token.emit("\n\n── Iris reasoning ──\n")
                    _ask(question=short_prompt, stream=True, on_token=_tok, action="question")
                    if tokens:
                        self.completed.emit("")   # signal end of stream
                except Exception:
                    pass
            return None   # already emitted

        # Non-report task (health keyword, navigate, open, zoom, etc.)
        if engine_result.get("answer"):
            context += f"\n\nRule answer:\n{engine_result['answer']}"
        return context

    def _fallback_no_ollama(self, question: str, intent: str):
        """Ollama unavailable — answer from rule engine only."""
        folder = self._folder or state.active_folder or ""
        if intent == "casual":
            self.completed.emit("Hi! Ollama is offline so I'm in rule-only mode. Ask me about your scan data.")
            return
        result = analyze(folder, question)
        answer = result.get("answer") or result.get("report_text", "")
        if not answer:
            answer = "Ollama is not available. Start Ollama to enable AI responses."
        else:
            answer += "\n\n[Ollama offline — install: ollama pull gemma3:4b]"
        self.completed.emit(answer)

    def _stream(
        self,
        question: str,
        context:  str,
        history:  List[Dict],
        action:   str,
    ) -> str:
        tokens: List[str] = []

        def on_token(t: str):
            tokens.append(t)
            self.token.emit(t)

        def on_model(name: str):
            self.tool_call.emit(f"model:{name}")

        ask_iris(
            question=question,
            context=context,
            history=history[-6:],
            stream=True,
            on_token=on_token,
            on_model=on_model,
            action=action,
        )

        full = "".join(tokens)
        self.completed.emit(full)
        return full


# ── Backward-compat aliases (panel.py imports these names) ───────────────────

class IrisLocalWorker(IrisOllamaWorker):
    """Alias kept for panel.py compatibility. Use IrisOllamaWorker directly."""
    def __init__(self, messages, folder="", parent=None):
        super().__init__(messages, folder=folder, mode="local", parent=parent)


class IrisSmartWorker(IrisOllamaWorker):
    """Alias kept for panel.py compatibility. Use IrisOllamaWorker directly."""
    def __init__(self, messages, api_key="", folder="", allow_api=True, parent=None):
        super().__init__(messages, folder=folder, api_key=api_key,
                         mode="smart", allow_api=allow_api, parent=parent)


# ── API key management ────────────────────────────────────────────────────────

_KEY_FILE = Path(__file__).parent / ".iris_config.json"


def get_api_key() -> str:
    for env in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        k = os.environ.get(env, "").strip()
        if k:
            return k
    try:
        if _KEY_FILE.exists():
            d = json.loads(_KEY_FILE.read_text())
            return d.get("api_key", "")
    except Exception:
        pass
    return ""


def save_api_key(key: str):
    try:
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_text(json.dumps({"api_key": key}, indent=2))
    except Exception as e:
        print(f"[Iris] Could not save API key: {e}")
