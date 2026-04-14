"""
iris/ollama.py

Ollama spine — the always-present local AI layer.

Two models, each used for what they are best at:

  gemma3:4b  — fast, low memory, excellent for:
    • Casual chat / greetings / small-talk
    • Short single-turn questions with no context
    • App navigation confirmations ("navigated to frame 21")
    • Any response where speed matters more than depth
    Target latency: ~1–3 s first token

  qwen2.5:7b-instruct-q4_K_M — smarter, better instruction-following, for:
    • Multi-finding analysis questions ("why is band 3 different?")
    • Reasoning over scan context (health score, cause/effect)
    • Knowledge-base RAG answers (spec lookups, firmware questions)
    • Cross-session pattern questions with memory context
    • Any response requiring structured multi-step reasoning
    Target latency: ~4–8 s first token

classify_intent() — zero-cost regex router (<1 ms):
  "casual"   → gemma3:4b, no context loaded
  "task"     → rule engine / tool handles it first (no LLM needed usually)
  "question" → pick_model() decides fast vs smart based on question depth

No API key needed. Requires ollama running on localhost:11434.
Install:
    ollama pull gemma3:4b
    ollama pull qwen2.5:7b-instruct-q4_K_M
    ollama pull nomic-embed-text
"""

from __future__ import annotations

import json
import re
import os
import urllib.request
import urllib.error
from typing import Callable, Dict, List, Optional

OLLAMA_URL   = "http://localhost:11434"
TIMEOUT_SECS = 120

# ── Model names — override via env vars if you swap models ───────────────────
FAST_MODEL  = os.environ.get("OLLAMA_FAST_MODEL",  "mistral")  # smaller and faster than gemma3:4b, but still good at short instructions
SMART_MODEL = os.environ.get("OLLAMA_SMART_MODEL", "llama-3.1-8b")  # better than qwen2.5:7b at reasoning and following complex instructions, and still reasonably fast for 7

# ── Availability cache — avoids an HTTP round-trip on every message ───────────
# Refreshed on first call and every 30 s. Hitting localhost is still ~5 ms,
# and doing it on every token stream callback compounds to real latency.
import time as _time
_avail_cache: list = []
_avail_ts: float = 0.0
_AVAIL_TTL: float = 30.0   # seconds between actual checks
_resolved_models: dict = {}  # preferred → resolved name cache


# ── Intent classifier — zero cost, <1 ms ─────────────────────────────────────

_RE_CASUAL = re.compile(
    r"^(hi+|hey+|hello+|hola|sup|yo+|howdy|hiya|heya|greetings)"
    r"|^(how are you|what can you do|who are you|what are you)"
    r"|^(thanks?|thank you|ok|okay|got it|cool|nice|great|awesome|perfect|sounds good)"
    r"|^(good morning|good evening|good afternoon|good night)"
    r"|^(bye|goodbye|see you|cya|later)$",
    re.I,
)

_RE_TASK = re.compile(
    r"\b(scan|re-?scan|refresh|report|analyz|find|check|open|load|close|"
    r"compare|navigate|go to|jump to|show me frame|"
    r"band|contrast|magnifier|histogram|zoom|theme|terminal|gap|tab|"
    r"play|pause|frame|health|log|error|anomal|dataset|folder|"
    r"remember|save|recall|index|memory)\b",
    re.I,
)

# Questions that need multi-step reasoning → SMART model
_RE_DEEP = re.compile(
    r"\b(why|explain|what (does|is|causes?|happened|went wrong)|"
    r"how (does|did|can|should)|interpret|reason|pattern|compare|"
    r"difference|relationship|firmware|spec|manual|calibrat|"
    r"tdi|binning|procmode|parameter|exposure|gain|fps|"
    r"across (bands?|frames?)|trend|history|session)\b",
    re.I,
)


def classify_intent(question: str) -> str:
    """
    Classify a user message with zero LLM calls.

    Returns:
        "casual"   — greeting / ack / small-talk
        "task"     — action needed (scan, open, report, navigate…)
        "question" — knowledge / analysis / reasoning question
    """
    q = (question or "").strip()
    q = re.sub(r"\b(?:chnage|cahnge|chnage)\b", "change", q, flags=re.I)
    q = re.sub(r"\b(?:enhnace|enhace|ehnance|enhacne)\b", "enhance", q, flags=re.I)
    q = re.sub(r"\b(?:contrsat|constrast|contast)\b", "contrast", q, flags=re.I)
    q = re.sub(r"\b(?:histgram|histogra)\b", "histogram", q, flags=re.I)
    q = re.sub(r"\b(?:individula|indivdual)\b", "individual", q, flags=re.I)
    q = re.sub(r"\b(?:magnifer|magnfier|maginifier)\b", "magnifier", q, flags=re.I)
    if _RE_CASUAL.search(q) and len(q.split()) < 8:
        return "casual"
    if _RE_TASK.search(q):
        return "task"
    return "question"


def pick_model(action: str, question: str, has_context: bool = False) -> str:
    """
    Choose between FAST_MODEL and SMART_MODEL based on what the response needs.

    gemma3:4b  wins when: casual, short, no context, pure navigation
    qwen2.5:7b wins when: reasoning, specs, RAG, scan context analysis

    Rules (priority order):
      1. Casual / navigation / confirmation action  → FAST
      2. Short question with no context             → FAST
      3. Deep reasoning / spec / firmware keywords  → SMART
      4. Has scan context AND analytical question   → SMART
      5. Default                                    → FAST
    """
    _FAST_ACTIONS = {"casual", "navigate_frame", "open_dataset",
                     "close_tab", "simple_question"}
    if action in _FAST_ACTIONS:
        return FAST_MODEL

    q = (question or "").strip()

    # Short question, no context — gemma handles it fine and is 2-3x faster
    if len(q.split()) < 6 and not has_context:
        return FAST_MODEL

    # Deep reasoning patterns — qwen is measurably better here
    if _RE_DEEP.search(q):
        return SMART_MODEL

    # Has scan context = needs to reason over data — use qwen
    if has_context:
        return SMART_MODEL

    return FAST_MODEL


# ── Availability ──────────────────────────────────────────────────────────────

def is_available() -> bool:
    try:
        return len(available_models()) > 0
    except Exception:
        return False


def available_models(force: bool = False) -> List[str]:
    """
    Return installed model names. Cached for 30 s to avoid an HTTP call on
    every message. Pass force=True to bypass the cache (e.g. after ollama pull).
    """
    global _avail_cache, _avail_ts
    now = _time.time()
    if not force and _avail_cache and (now - _avail_ts) < _AVAIL_TTL:
        return _avail_cache
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            _avail_cache = [m["name"] for m in data.get("models", [])]
            _avail_ts = now
            _resolved_models.clear()   # invalidate resolved-name cache too
            return _avail_cache
    except Exception:
        return _avail_cache or []  # return stale cache on network error


def _resolve_model(preferred: str) -> str:
    """
    Resolve a preferred model name to one that is actually installed.
    Result is cached per preferred name so repeated calls cost nothing.
    """
    if preferred in _resolved_models:
        return _resolved_models[preferred]

    models = available_models()
    if not models:
        return preferred

    resolved = preferred
    if preferred not in models:
        family = preferred.split(":")[0].lower()
        for m in models:
            if family in m.lower():
                resolved = m
                break
        else:
            for m in models:
                if any(k in m.lower() for k in ("instruct", "chat", "qwen", "gemma", "mistral", "llama")):
                    resolved = m
                    break
            else:
                resolved = models[0]

    _resolved_models[preferred] = resolved
    return resolved


# ── System prompts — tuned per model ─────────────────────────────────────────

# gemma3:4b: keep it SHORT — gemma follows concise prompts better
_SYSTEM_FAST = (
    "You are Iris, a camera QA assistant inside DisplayGroundX.\n"
    "Answer in plain conversational sentences — no bullet points, no markdown.\n"
    "Be direct and specific. Answer the latest user request only.\n"
    "Do not recycle the same explanation unless the user asked the same question again.\n"
    "If you don't know, say so."
)

# qwen2.5:7b: full context — qwen handles long system prompts well
_SYSTEM_SMART = (
    "You are Iris, a satellite camera QA analyst inside DisplayGroundX.\n"
    "You are talking to an engineer. Be direct, specific, and natural — like a knowledgeable colleague.\n"
    "Sensor: pushbroom, 8448×384px, 7 bands, 10-bit. TDI byte: 0=OFF 34=8-stage 66=64-stage.\n"
    "When given scan data: explain what you see, what likely caused it, and what to do.\n"
    "Keep answers under 150 words unless more detail is genuinely needed.\n"
    "Answer the latest request, not the previous one.\n"
    "Do not repeat earlier explanations unless they are directly relevant.\n"
    "Plain text only — no bullet points, no markdown, no headers."
)


# ── Core chat ─────────────────────────────────────────────────────────────────

def chat(
    messages:    List[Dict],
    system:      str = "",
    temperature: float = 0.3,
    stream:      bool = False,
    on_token:    Optional[Callable[[str], None]] = None,
    on_model:    Optional[Callable[[str], None]] = None,
    model:       Optional[str] = None,
) -> str:
    """
    Send a conversation to Ollama and return the full response.
    model: pass FAST_MODEL or SMART_MODEL — resolved to installed name automatically.
    """
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    chosen = _resolve_model(model or FAST_MODEL)

    if on_model:
        try:
            on_model(chosen)
        except Exception:
            pass

    payload = json.dumps({
        "model":    chosen,
        "messages": full_messages,
        "stream":   stream,
        "options":  {"temperature": temperature, "num_predict": 1024},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
            if not stream:
                return json.loads(resp.read()).get("message", {}).get("content", "")

            full_text: List[str] = []
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_text.append(token)
                        if on_token:
                            on_token(token)
                except json.JSONDecodeError:
                    continue
            return "".join(full_text)

    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="ignore").strip()
        except Exception:
            pass
        raise RuntimeError(
            f"Ollama HTTP {e.code}." + (f" {body[:200]}" if body else "") + f" Model: {chosen}."
        )
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_URL}. Is Ollama running? {e}")


# ── Main ask function ─────────────────────────────────────────────────────────

def ask_iris(
    question:  str,
    context:   str = "",
    history:   Optional[List[Dict]] = None,
    stream:    bool = False,
    on_token:  Optional[Callable[[str], None]] = None,
    on_model:  Optional[Callable[[str], None]] = None,
    action:    str = "",
) -> str:
    """
    Ask Iris — automatically picks gemma3:4b or qwen2.5:7b based on need.

    gemma3:4b  → casual / short / navigation (fast path, no reasoning needed)
    qwen2.5:7b → analysis / specs / RAG / scan context (quality path)

    action:  intent from classify_intent() — "casual" / "task" / "question"
    context: scan data or rule engine output — non-empty triggers SMART model
    """
    has_context = bool(context and context.strip())
    model       = pick_model(action=action, question=question, has_context=has_context)
    system      = _SYSTEM_FAST if model == FAST_MODEL else _SYSTEM_SMART

    # Augment context BEFORE constructing the user message; otherwise the model
    # never actually receives the merged KB context.
    if model == SMART_MODEL and _RE_DEEP.search(question):
        kb_ctx = search_local_kb_fast(question, top_k=3)
        if kb_ctx and not context:
            context = kb_ctx
        elif kb_ctx and context:
            context = kb_ctx + "\n\n---\n\n" + context

    messages = list(history or [])
    user_content = f"{context}\n\nUser question: {question}" if context else question
    messages.append({"role": "user", "content": user_content})

    return chat(
        messages=messages,
        system=system,
        temperature=0.3,
        stream=stream,
        on_token=on_token,
        on_model=on_model,
        model=model,
    )


def search_local_kb_fast(query: str, top_k: int = 3) -> str:
    """
    Search the local knowledge folder using BM25 (no network, no embeddings).
    Returns a ready-to-inject context string, or "" if nothing found.
    Typical latency: <5 ms after the first warm-up call.
    """
    try:
        from .local_rag import search_local_kb
        result = search_local_kb(query, top_k=top_k)
        return result.get("context_text", "")
    except Exception as e:
        print(f"[Ollama] local_rag search failed: {e}")
        return ""


def ask_with_rag(
    question: str,
    context:  str = "",
    history:  Optional[List[Dict]] = None,
    stream:   bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    on_model: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Ask Iris with knowledge-base RAG context.

    RAG pipeline (fully offline, no Supabase needed):
      1. Search local knowledge folder via BM25 (~1–5 ms, in-process).
      2. Merge with any extra context passed in by the caller.
      3. Send to SMART_MODEL with a spec-aware system prompt.

    Always uses SMART_MODEL — RAG requires careful reasoning over spec text.
    """
    # ── Step 1: local BM25 search ─────────────────────────────────────────
    local_kb = search_local_kb_fast(question, top_k=4)

    # ── Step 2: merge context (local KB first — most authoritative) ───────
    if local_kb and context:
        merged_context = local_kb + "\n\n---\n\n" + context
    else:
        merged_context = local_kb or context

    rag_system = _SYSTEM_SMART + (
        "\n\nYou have been provided with relevant excerpts from the knowledge base "
        "(sensor specs, SOPs, camera manuals). Use this information to answer the question. "
        "Cite the source document name when you use knowledge base content."
    )
    messages = list(history or [])
    messages.append({
        "role": "user",
        "content": (
            f"Knowledge base context:\n{merged_context}\n\nUser question: {question}"
            if merged_context else question
        ),
    })

    return chat(
        messages=messages,
        system=rag_system,
        temperature=0.3,
        stream=stream,
        on_token=on_token,
        on_model=on_model,
        model=SMART_MODEL,
    )


# ── Streaming worker ─────────────────────────────────────────────────────────

class OllamaWorker:
    """Async streaming worker. Set on_token/on_finished/on_error, then start()."""

    def __init__(
        self,
        question: str,
        context:  str = "",
        history:  Optional[List[Dict]] = None,
        action:   str = "",
    ):
        self.question = question
        self.context  = context
        self.history  = history or []
        self.action   = action

        self.on_token:    Optional[Callable[[str], None]] = None
        self.on_finished: Optional[Callable[[str], None]] = None
        self.on_error:    Optional[Callable[[str], None]] = None

    def start(self):
        import threading
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            tokens: List[str] = []

            def _tok(t: str):
                tokens.append(t)
                if self.on_token:
                    self.on_token(t)

            ask_iris(
                question=self.question,
                context=self.context,
                history=self.history,
                stream=True,
                on_token=_tok,
                action=self.action,
            )
            if self.on_finished:
                self.on_finished("".join(tokens))
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
