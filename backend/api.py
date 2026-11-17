"""Web API. CLI and Web share one service stack (GraphService) and one storage mode."""
import asyncio
import json
import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from backend.schemas import ChatRequest, ChatResponse, Citation, DocumentInfo, SessionInfo, SessionSummary
from backend.session_manager import SessionManager
from rag.settings import load_env, load_settings, build_runtime

load_env()
router = APIRouter()

_settings = load_settings()
service = build_runtime(_settings)
session_manager = SessionManager(max_history=10, dsn=_settings.pg_dsn or None)

SAMPLE_DIR = "sample_data"

_startup_sync: dict = {
    "changed": False,
    "status": "unknown",
    "sample_dir": SAMPLE_DIR,
    "files_total": 0,
    "files_reindexed": 0,
    "files": [],
}


def set_startup_sync(summary: dict):
    global _startup_sync
    _startup_sync = summary or {"changed": False, "status": "unknown"}


def _citations(chunks):
    return [Citation(id=i + 1,
                     source=Path(c["metadata"]["source"]).name,
                     score=round(float(c.get("score", 0.0)), 6),
                     snippet=c["text"][:150],
                     citation_id=c.get("citation_id", f"S{i + 1}"))
            for i, c in enumerate(chunks)]


_GUIDANCE = {
    "insufficient_evidence": "检索到的材料不足以支持答案，已拒绝回答以避免编造。",
    "no_relevant": "知识库中没有找到与问题相关的内容。",
    "no_docs": "知识库中还没有已导入的文档，请先上传资料。",
}


@router.post("/sessions", response_model=SessionInfo)
def create_session():
    session_id = session_manager.create_session()
    return session_manager.load_session(session_id)


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions():
    return session_manager.list_sessions()


@router.get("/startup/sync")
def get_startup_sync():
    return _startup_sync


@router.get("/sessions/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    session = session_manager.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"status": "ok"}


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return [{"filename": Path(row["source"]).name, "chunks": row["chunks"]}
            for row in service.list_sources()]


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "").name
    if safe_name.rsplit(".", 1)[-1].lower() not in ("md", "txt", "pdf"):
        raise HTTPException(status_code=400, detail="Only .md / .txt / .pdf files are supported.")
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    file_path = os.path.join(SAMPLE_DIR, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    task_id = service.submit_ingest(file_path)
    return {"filename": safe_name, "task_id": task_id, "status": "queued"}


@router.get("/tasks")
def list_tasks():
    return service.task_list(50)


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = service.task_get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/chat", response_model=ChatResponse)
def chat_sync(req: ChatRequest):
    session_id = req.session_id or session_manager.create_session()
    history_context = session_manager.get_history_context(session_id)
    full_query = history_context + "\n[Current Question]:\n" + req.message if history_context else req.message

    result = service.query(full_query, mode=req.search_mode, top_k=req.top_k,
                           threshold=req.threshold, rerank=req.use_reranker)

    session_manager.add_message(session_id, "user", req.message)
    session_manager.add_message(session_id, "assistant", result["answer"])

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        status=result["status"],
        user_guidance=_GUIDANCE.get(result["status"], ""),
        retrieval_quality={"status": result["status"], "elapsed_ms": result.get("elapsed_ms"),
                           "chunks_used": len(result.get("chunks_used", [])),
                           "channels": result.get("retrieval_trace", {}).get("counts", {})},
        session_id=session_id,
        citations=_citations(result.get("chunks_used", [])),
        mode=req.search_mode,
        cached=bool(result.get("cached")),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or session_manager.create_session()
    history_context = session_manager.get_history_context(session_id)
    full_query = history_context + "\n[Current Question]:\n" + req.message if history_context else req.message

    def work():
        return service.query(full_query, mode=req.search_mode, top_k=req.top_k,
                             threshold=req.threshold, rerank=req.use_reranker)

    async def event_generator():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield sse({"type": "progress", "session_id": session_id, "stage": "session.ready",
                   "message": "🧠 已加载会话上下文，开始处理问题..."})
        result = await asyncio.get_event_loop().run_in_executor(None, work)
        yield sse({"type": "progress", "session_id": session_id, "stage": "retrieval.done",
                   "message": f"🔎 检索完成，命中 {len(result.get('chunks_used', []))} 个候选片段。"})
        citations = _citations(result.get("chunks_used", []))
        if citations:
            yield sse({"type": "citations", "session_id": session_id, "citations": [c.model_dump() for c in citations]})
        if result["status"] != "ok":
            yield sse({"type": "notice", "session_id": session_id,
                       "message": _GUIDANCE.get(result["status"], "未能基于知识库回答该问题。")})
        # Answers stream only after citation validation: replay the checked answer.
        session_manager.add_message(session_id, "user", req.message)
        session_manager.add_message(session_id, "assistant", result["answer"])
        yield sse({"type": "progress", "session_id": session_id, "stage": "generation.start",
                   "message": "✍️ 引用校验完成，开始输出回答..."})
        text = result["answer"]
        step = 48
        for i in range(0, len(text), step):
            yield sse({"type": "answer.delta", "chunk": text[i:i + step]})
        yield sse({"type": "progress", "session_id": session_id, "stage": "generation.done",
                   "message": "✅ 回答生成完成。"})
        yield sse({"type": "answer.final", "answer": text})
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
