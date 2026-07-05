#!/usr/bin/env python3
"""JARVIS — Private AI Assistant Server.
   Runs on 1bit.systems NPU+GPU+CPU hardware.
   API: http://localhost:8080
   Web UI: http://localhost:8080/chat
"""
import os, sys
import json
import base64
import uuid
import logging
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from config import load_config, JarvisConfig
from agent import JarvisAgent
from voice import VoiceIO
from knowledge import OpenKnowledge, KnowledgeEntry, fact, document_entry, conversation_entry

# ─── Setup ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("jarvis")

config = load_config()
knowledge = OpenKnowledge(config.data_dir)
agent = JarvisAgent(config, knowledge=knowledge)
voice = VoiceIO(config)

# Ensure data directories exist
for d in ["rag_docs", "knowledge", "recordings", "uploads", "conversations"]:
    (Path(config.data_dir) / d).mkdir(parents=True, exist_ok=True)

# ─── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(title="JARVIS — 1bit.systems", version="1.0.0")

# Mount static files (web UI)
static_dir = Path(config.jarvis_dir) / "web"
static_dir.mkdir(exist_ok=True)

# Main page
@app.get("/", response_class=HTMLResponse)
async def root():
    return (static_dir / "index.html").read_text() if (static_dir / "index.html").exists() else HTMLResponse("<h1>JARVIS — Building...</h1>")

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    return (static_dir / "index.html").read_text() if (static_dir / "index.html").exists() else HTMLResponse("<h1>JARVIS — Building...</h1>")

# ─── API Endpoints ────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    """System status — check all backends."""
    import httpx
    results = {}
    # Check unified daemon
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{config.api_base}/v1/models")
            results["fused_engine"] = {"ok": r.status_code == 200}
    except Exception as e:
        results["fused_engine"] = {"ok": False, "error": str(e)}

    # Check FLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"http://127.0.0.1:{config.flm_port}/v1/models")
            results["flm_npu"] = {"ok": r.status_code == 200}
    except Exception as e:
        results["flm_npu"] = {"ok": False, "error": str(e)}

    # Check TTS
    results["tts"] = {"ok": voice.tts_voice is not None}

    kstats = knowledge.stats()
    return {
        "name": "JARVIS — 1bit.systems",
        "version": "1.0.0",
        "hardware": "NPU (XDNA2) + GPU (Radeon 8060S) + CPU (Zen 5)",
        "status": results,
        "model": config.llm_model,
        "tools": config.tools_enabled,
        "knowledge": kstats,
    }

@app.post("/api/chat")
async def chat(message: str = Form(...), session_id: str = Form(None), image: Optional[UploadFile] = File(None)):
    """Send a chat message and get response."""
    sid = session_id or str(uuid.uuid4())
    image_b64 = None
    if image:
        image_data = await image.read()
        image_b64 = base64.b64encode(image_data).decode()

    from fastapi.responses import StreamingResponse
    async def generate():
        async for chunk in agent.chat_stream(sid, message, image_b64):
            yield json.dumps({"type": "chunk", "text": chunk}) + "\n"
        yield json.dumps({"type": "done", "session_id": sid}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket chat with streaming."""
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "message")
            session_id = msg.get("session_id", session_id)

            if msg_type == "message":
                text = msg.get("text", "")
                image_b64 = msg.get("image", None)

                # Stream response
                async for chunk in agent.chat_stream(session_id, text, image_b64):
                    if websocket.client_state == WebSocketState.DISCONNECTED:
                        break
                    await websocket.send_text(json.dumps({
                        "type": "chunk",
                        "text": chunk,
                        "session_id": session_id,
                    }))
                await websocket.send_text(json.dumps({
                    "type": "done",
                    "session_id": session_id,
                }))

            elif msg_type == "stt":
                """Speech-to-text: receive audio, return text."""
                import base64
                audio_b64 = msg.get("audio", "")
                if audio_b64:
                    audio_data = base64.b64decode(audio_b64)
                    text = await voice.stt(audio_data)
                    await websocket.send_text(json.dumps({
                        "type": "stt_result",
                        "text": text,
                    }))

            elif msg_type == "tts":
                """Text-to-speech: receive text, return audio."""
                text = msg.get("text", "")
                audio = await voice.tts(text)
                if audio:
                    await websocket.send_bytes(json.dumps({
                        "type": "tts_result",
                        "audio": base64.b64encode(audio).decode(),
                    }).encode())

            elif msg_type == "rag_query":
                query = msg.get("query", "")
                results = await agent.rag_query(query)
                await websocket.send_text(json.dumps({
                    "type": "rag_result",
                    "results": results,
                }))

            elif msg_type == "clear":
                agent.conversations[session_id] = [
                    {"role": "system", "content": agent.system_prompt}
                ]
                await websocket.send_text(json.dumps({
                    "type": "cleared",
                    "session_id": session_id,
                }))

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        log.error(f"WebSocket error: {e}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), category: str = Form("rag")):
    """Upload a file for RAG or vision."""
    if category == "rag":
        save_dir = Path(config.data_dir) / "rag_docs"
        save_dir.mkdir(parents=True, exist_ok=True)
        fpath = save_dir / file.filename
        content = await file.read()
        fpath.write_bytes(content)
        return {"status": "ok", "path": str(fpath), "size": len(content), "category": "rag"}
    elif category == "image":
        save_dir = Path(config.data_dir) / "uploads"
        save_dir.mkdir(parents=True, exist_ok=True)
        fpath = save_dir / file.filename
        content = await file.read()
        fpath.write_bytes(content)
        return {"status": "ok", "path": str(fpath), "size": len(content), "category": "image"}
    return JSONResponse({"status": "error", "message": "Unknown category"}, status_code=400)

# ─── Knowledge API ────────────────────────────────────────────────────

@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """Knowledge base statistics."""
    return knowledge.stats()

@app.get("/api/knowledge/search")
async def knowledge_search(q: str = "", type: str = "", max: int = 10):
    """Search the open knowledge base."""
    types = [t.strip() for t in type.split(",")] if type else None
    results = knowledge.search(q, max_results=max, entry_types=types)
    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "path": e.path,
                "title": e.title,
                "type": e.entry_type,
                "tags": e.tags,
                "source": e.source,
                "confidence": e.confidence,
                "created": e.created.isoformat(),
                "snippet": e.content[:300],
            }
            for e in results
        ],
    }

@app.get("/api/knowledge/list")
async def knowledge_list(type: str = ""):
    """List all knowledge entries, optionally filtered by type."""
    if type:
        entries = knowledge.list_by_type(type)
    else:
        entries = knowledge.all()
    return {
        "count": len(entries),
        "entries": [
            {"path": e.path, "title": e.title, "type": e.entry_type,
             "tags": e.tags, "created": e.created.isoformat()}
            for e in entries
        ],
    }

@app.get("/api/knowledge/read")
async def knowledge_read(path: str = ""):
    """Read a knowledge entry by path."""
    entry = knowledge.get(path)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {
        "path": entry.path,
        "title": entry.title,
        "type": entry.entry_type,
        "content": entry.content,
        "tags": entry.tags,
        "source": entry.source,
        "confidence": entry.confidence,
        "created": entry.created.isoformat(),
        "updated": entry.updated.isoformat(),
    }

@app.post("/api/knowledge/add")
async def knowledge_add(title: str = Form(...), content: str = Form(...),
                        type: str = Form("fact"), tags: str = Form(""),
                        source: str = Form("user"), confidence: float = Form(1.0)):
    """Add a new knowledge entry."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    entry = KnowledgeEntry(entry_type=type, title=title, content=content,
                           tags=tag_list, source=source, confidence=confidence)
    path = knowledge.add(entry)
    return {"status": "ok", "path": path, "title": title}

@app.delete("/api/knowledge/delete")
async def knowledge_delete(path: str = Form(...)):
    """Delete a knowledge entry."""
    if knowledge.delete(path):
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Entry not found")


@app.post("/api/tts/download")
async def download_voice(voice_name: str = Form("en_US-lessac-medium")):
    """Download a TTS voice model."""
    stdout, stderr = await VoiceIO.download_voice(voice_name)
    voice._init_tts()
    return {"status": "ok" if voice.tts_voice else "error", "stdout": stdout, "stderr": stderr}

# ─── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    JARVIS v1.0                           ║
║         Private AI — 1bit.systems                       ║
║                                                         ║
║  🔹 NPU:  XDNA2 (94 tok/s INT8)                        ║
║  🔹 GPU:  Radeon 8060S (Vulkan)                        ║
║  🔹 CPU:  Zen 5 (16C/32T)                              ║
║  🔹 Model: {config.llm_model:<40s}║
║                                                         ║
║  Web UI:  http://localhost:{config.port}/chat           ║
║  TTS:     {'✅ Loaded' if voice.tts_voice else '⬜ Not loaded'}                              ║
╚══════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
