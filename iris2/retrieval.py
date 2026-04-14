"""
iris/retrieval.py

Single module for all semantic retrieval operations.
Consolidates: embedder.py + knowledge.py + memory.py  (those 3 files can be deleted)

  - Local embedding via Ollama nomic-embed-text (768-dim, free, offline)
  - Knowledge base search against Supabase pgvector
  - Long-term memory: save + recall + manage

Install:  ollama pull nomic-embed-text
Requires: ollama running on localhost:11434
          supabase credentials in iris_knowledge.cfg or env vars
          (SUPABASE_URL, SUPABASE_SERVICE_KEY)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL      = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM   = 768


def _cfg(key: str) -> str:
    env_map = {
        "supabase_url":         "SUPABASE_URL",
        "supabase_service_key": "SUPABASE_SERVICE_KEY",
    }
    env_val = os.environ.get(env_map.get(key, ""), "")
    if env_val:
        return env_val
    cfg_path = Path(__file__).parent / "iris_knowledge.cfg"
    if not cfg_path.exists():
        return ""
    for line in cfg_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return ""


def _supabase():
    from supabase import create_client
    url = _cfg("supabase_url")
    key = _cfg("supabase_service_key")
    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not configured. "
            "Set SUPABASE_URL + SUPABASE_SERVICE_KEY env vars or iris_knowledge.cfg."
        )
    return create_client(url, key)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_one(text: str) -> List[float]:
    """Embed a single string. Returns 768-dim float vector via Ollama."""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    payload = json.dumps({"model": EMBEDDING_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            vec  = data.get("embedding", [])
            if len(vec) != EMBEDDING_DIM:
                print(f"[Retrieval] Warning: expected {EMBEDDING_DIM} dims, got {len(vec)}")
            return vec
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}. Is Ollama running? Error: {e}"
        )


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns list of 768-dim vectors."""
    return [embed_one(t) for t in texts] if texts else []


def embedding_available() -> bool:
    """Check if Ollama embedding service is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


# ── Knowledge base search ─────────────────────────────────────────────────────

def search_knowledge(
    query:       str,
    top_k:       int   = 4,
    threshold:   float = 0.30,
    file_filter: Optional[str] = None,
) -> Dict:
    """
    Semantic search over the Iris knowledge base (user guide, specs, docs).

    Args:
        query:       Natural language question or keyword phrase.
        top_k:       Max chunks to return.
        threshold:   Min cosine similarity (0–1). Lower = broader.
        file_filter: Restrict to a specific filename if set.

    Returns dict: results, query, total_found, context_text.
    """
    if not query or not query.strip():
        return {"error": "Empty query", "results": []}

    try:
        vec = embed_one(query.strip())
    except Exception as e:
        return {"error": f"Embedding failed: {e}", "results": []}

    try:
        sb   = _supabase()
        rows = sb.rpc("iris_search", {
            "query_embedding": vec,
            "match_threshold": threshold,
            "match_count":     top_k,
        }).execute().data or []
    except Exception as e:
        return {"error": f"Supabase search failed: {e}", "results": []}

    if file_filter:
        rows = [r for r in rows if file_filter.lower() in r.get("file_name", "").lower()]

    if not rows:
        return {
            "query":        query,
            "total_found":  0,
            "results":      [],
            "context_text": f"No relevant content found in knowledge base for: '{query}'",
        }

    results       = []
    context_parts = []

    for row in rows:
        similarity = round(float(row.get("similarity", 0)), 3)
        file_name  = row.get("file_name", "unknown")
        heading    = row.get("heading", "")
        page_sheet = row.get("page_or_sheet", "")
        chunk_text = row.get("chunk_text", "")

        source_parts = [file_name]
        if heading:    source_parts.append(f"§ {heading}")
        if page_sheet: source_parts.append(f"[{page_sheet}]")
        source = " — ".join(source_parts)

        results.append({
            "source":     source,
            "file_name":  file_name,
            "file_type":  row.get("file_type", ""),
            "heading":    heading,
            "location":   page_sheet,
            "text":       chunk_text,
            "similarity": similarity,
        })
        context_parts.append(
            f"[Source: {source}  similarity={similarity:.2f}]\n{chunk_text}"
        )

    return {
        "query":        query,
        "total_found":  len(results),
        "results":      results,
        "context_text": (
            f"Knowledge base results for '{query}':\n\n"
            + "\n\n---\n\n".join(context_parts)
        ),
    }


def knowledge_base_status() -> Dict:
    """Return a summary of what's currently indexed in the knowledge base."""
    try:
        sb   = _supabase()
        rows = sb.table("iris_knowledge_index_status").select("*").execute().data or []
        if not rows:
            return {
                "indexed": False,
                "message": "Knowledge base is empty. Use /index to index your files.",
                "files":   [],
            }
        return {
            "indexed":      True,
            "file_count":   len(rows),
            "total_chunks": sum(r.get("chunks", 0) for r in rows),
            "files": [
                {
                    "name":         r["file_name"],
                    "type":         r["file_type"],
                    "chunks":       r["chunks"],
                    "last_indexed": r["last_indexed"],
                }
                for r in rows
            ],
        }
    except Exception as e:
        return {"error": str(e), "indexed": False}


# ── Long-term memory ──────────────────────────────────────────────────────────
#
# Supabase schema required (run in SQL editor if not done):
#
# create table if not exists iris_memory (
#     id           bigserial primary key,
#     memory_type  text        not null default 'finding',
#     dataset      text        not null default '',
#     folder       text        not null default '',
#     title        text        not null,
#     detail       text        not null,
#     tags         text[]      not null default '{}',
#     embedding    vector(768),
#     created_at   timestamptz not null default now(),
#     session_date text        not null default ''
# );
# create or replace function iris_memory_search(
#     query_embedding  vector(768),
#     match_threshold  float   default 0.30,
#     match_count      int     default 5
# )
# returns table (id bigint, memory_type text, dataset text,
#                title text, detail text, tags text[],
#                similarity float, created_at timestamptz)
# language sql stable as $$
#     select id, memory_type, dataset, title, detail, tags,
#            1 - (embedding <=> query_embedding) as similarity,
#            created_at
#     from iris_memory
#     where 1 - (embedding <=> query_embedding) > match_threshold
#     order by embedding <=> query_embedding
#     limit match_count;
# $$;

def save_memory(
    title:       str,
    detail:      str,
    memory_type: str       = "finding",
    dataset:     str       = "",
    folder:      str       = "",
    tags:        List[str] = None,
) -> Dict:
    """Save a memory entry to Supabase iris_memory table."""
    if not title or not detail:
        return {"error": "title and detail are required"}

    tags = tags or []

    embed_text = f"{title}\n{detail}"
    if dataset: embed_text = f"{dataset}: {embed_text}"
    if tags:    embed_text += f"\nTags: {', '.join(tags)}"

    try:
        vec = embed_one(embed_text)
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

    try:
        _supabase().table("iris_memory").insert({
            "memory_type":  memory_type,
            "dataset":      dataset,
            "folder":       folder,
            "title":        title,
            "detail":       detail,
            "tags":         tags,
            "embedding":    vec,
            "session_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }).execute()
    except Exception as e:
        return {"error": f"Supabase insert failed: {e}"}

    return {
        "saved":       True,
        "title":       title,
        "memory_type": memory_type,
        "dataset":     dataset,
    }


def recall_memory(
    query:       str,
    top_k:       int   = 5,
    threshold:   float = 0.30,
    memory_type: Optional[str] = None,
) -> Dict:
    """Semantically search Iris's long-term memory for past findings."""
    if not query.strip():
        return {"error": "Empty query", "results": []}

    try:
        vec = embed_one(query.strip())
    except Exception as e:
        return {"error": f"Embedding failed: {e}", "results": []}

    try:
        rows = _supabase().rpc("iris_memory_search", {
            "query_embedding": vec,
            "match_threshold": threshold,
            "match_count":     top_k,
        }).execute().data or []
    except Exception as e:
        return {"error": f"Supabase search failed: {e}", "results": []}

    if memory_type:
        rows = [r for r in rows if r.get("memory_type") == memory_type]

    if not rows:
        return {
            "query":        query,
            "total_found":  0,
            "results":      [],
            "context_text": f"No past memories found for: '{query}'",
        }

    _ICONS = {"finding": "🔴", "pattern": "🟡", "note": "⚪", "resolved": "✅"}
    results       = []
    context_parts = []

    for row in rows:
        similarity = round(float(row.get("similarity", 0)), 3)
        mtype      = row.get("memory_type", "note")
        dataset    = row.get("dataset", "")
        title      = row.get("title", "")
        detail     = row.get("detail", "")
        created_at = row.get("created_at", "")[:10]
        tags       = row.get("tags", [])

        results.append({
            "type": mtype, "dataset": dataset, "title": title,
            "detail": detail, "date": created_at, "tags": tags,
            "similarity": similarity,
        })
        icon   = _ICONS.get(mtype, "•")
        header = f"{icon} [{mtype.upper()}] {title}"
        if dataset: header += f" — {dataset}"
        header += f" (observed {created_at}, similarity={similarity:.2f})"
        context_parts.append(f"{header}\n{detail}")

    return {
        "query":        query,
        "total_found":  len(results),
        "results":      results,
        "context_text": (
            f"Iris memory — past observations related to '{query}':\n\n"
            + "\n\n---\n\n".join(context_parts)
        ),
    }


def list_memories(
    dataset:     Optional[str] = None,
    memory_type: Optional[str] = None,
    limit:       int = 20,
) -> List[Dict]:
    """List recent memories, optionally filtered by dataset or type."""
    try:
        sb = _supabase()
        q  = (sb.table("iris_memory")
                .select("id, memory_type, dataset, title, session_date, tags")
                .order("created_at", desc=True)
                .limit(limit))
        if dataset:     q = q.ilike("dataset", f"%{dataset}%")
        if memory_type: q = q.eq("memory_type", memory_type)
        return q.execute().data or []
    except Exception as e:
        return [{"error": str(e)}]


def delete_memory(memory_id: int) -> Dict:
    """Delete a specific memory by ID."""
    try:
        _supabase().table("iris_memory").delete().eq("id", memory_id).execute()
        return {"deleted": True, "id": memory_id}
    except Exception as e:
        return {"error": str(e)}


def memory_summary() -> Dict:
    """Return memory counts by type."""
    try:
        rows = _supabase().table("iris_memory").select("memory_type").execute().data or []
        counts: Dict[str, int] = {}
        for r in rows:
            t = r.get("memory_type", "note")
            counts[t] = counts.get(t, 0) + 1
        return {"total": len(rows), "by_type": counts}
    except Exception as e:
        return {"error": str(e)}
