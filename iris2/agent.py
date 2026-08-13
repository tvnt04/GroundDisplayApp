from __future__ import annotations
import json
import os
import re
import base64
import urllib.request
import urllib.error
from pathlib import Path
from app_paths import get_app_data_path, migrate_legacy_file
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

_SYSTEM = """You are Iris — a helpful AI assistant embedded in Display X Studio, a multispectral satellite and sensor imagery analysis application.

## YOUR IDENTITY
Your primary goal is to help the user understand and navigate the application. Whenever the user is stuck, you should be able to answer questions regarding the application. You act as a simple, helpful assistant. You can also perform tasks based on the user's text requests when appropriate, but your main focus is on answering questions and guiding the user. Keep your answers clear, helpful, and concise.

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
        
        import difflib
        def fuzzy_has(kw_list):
            if any(k in q for k in kw_list):
                return True
            words = q.split()
            single_kws = [k for k in kw_list if " " not in k]
            for w in words:
                # 0.75 cutoff handles 'ope' for 'open', 'lod' for 'load', etc.
                if difflib.get_close_matches(w, single_kws, n=1, cutoff=0.75):
                    return True
            return False
        
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
        if fuzzy_has(_PARAM_KW):
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
        if fuzzy_has(_LOG_KW):
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

        if fuzzy_has(_REPORT_KW):
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

        if fuzzy_has(_SCAN_KW):
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

        _OPEN_KW = ("open", "load", "view", "show")
        if fuzzy_has(_OPEN_KW):
            if extracted_path:
                self.tool_call.emit("open_dataset")
                in_current = any(x in q for x in ("current tab", "same tab", "this tab", "here"))
                result = TOOL_DISPATCH["open_dataset"]({"folder_path": extracted_path, "in_current_tab": in_current})
                if result.get("error"):
                    self.completed.emit(f"Failed to open: {result['error']}")
                else:
                    self.completed.emit(f"Opened {os.path.basename(extracted_path)}.")
                return True

        # Fallback to LLM planner for complex tasks like "open last folder"
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

        if tool == "run_scan":
            report_fn = TOOL_DISPATCH.get("generate_report")
            report_folder = args.get("folder") or effective_folder or state.active_folder or ""
            if report_fn and report_folder:
                rep = report_fn({"folder": report_folder})
                if rep.get("report"):
                    self.completed.emit(rep["report"])
                    return True

        if tool == "generate_report":
            txt = result.get("report", "") or result.get("message", "Report generated.")
            self.completed.emit(txt)
            return True

        msg = result.get("message") or result.get("status") or f"Executed {tool}."
        self.completed.emit(msg)
        return True

    def _dispatch(self):
        question = _extract_question(self._messages)
        intent = classify_intent(question)
        folder = self._folder or state.active_folder or ""

        if intent == "task":
            if self._try_llm_task_execution(question, folder):
                return
                
        self._fallback_no_ollama(question, intent)


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
        """Answer from rule engine only (zero API/LLM dependencies)."""
        folder = self._folder or state.active_folder or ""
        if intent == "casual":
            self.completed.emit("Hi! I'm operating in instant local mode. Ask me about your scan data or give me a command.")
            return
        result = analyze(folder, question)
        answer = result.get("answer") or result.get("report_text", "")
        if not answer:
            answer = "I couldn't find a direct answer to that from the scan data using my local engine."
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

_KEY_FILE = Path(migrate_legacy_file(
    get_app_data_path(".iris_config.json"),
    os.path.join(os.path.dirname(__file__), ".iris_config.json")
))


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
