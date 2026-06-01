import json
import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.schemas import ChatRequest, ChatResponse, SessionInfo, SessionSummary, DocumentInfo
from backend.session_manager import SessionManager
from rag.pipeline import RAGPipeline
from rag.retriever import Retriever
from rag.generator import generate_chat, generate

router = APIRouter()
session_manager = SessionManager(max_history=10)

# Initialize pipeline once
pipeline = RAGPipeline()

_startup_sync: dict = {
    "changed": False,
    "status": "unknown",
    "sample_dir": "sample_data",
    "files_total": 0,
    "files_reindexed": 0,
    "files": [],
}


def set_startup_sync(summary: dict):
    global _startup_sync
    _startup_sync = summary or {"changed": False, "status": "unknown"}

@router.post("/sessions", response_model=SessionInfo)
def create_session():
    session_id = session_manager.create_session()
    session = session_manager.load_session(session_id)
    return session

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
    sources = pipeline.list_documents()
    # Mock chunk counts for now, could be enhanced
    return [{"filename": src, "chunks": 0} for src in sources]

@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    os.makedirs("sample_data", exist_ok=True)
    file_path = os.path.join("sample_data", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Ingest the file
    chunks = pipeline.ingest(file_path)
    return {"filename": file.filename, "chunks": chunks}


class UrlIngestRequest(BaseModel):
    url: str


@router.post("/documents/ingest-url")
def ingest_url(req: UrlIngestRequest):
    url = (req.url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")
    chunks = pipeline.ingest(url)
    return {"url": url, "chunks": chunks}

@router.post("/chat", response_model=ChatResponse)
def chat_sync(req: ChatRequest):
    session_id = req.session_id
    if not session_id:
        session_id = session_manager.create_session()

    history_context = session_manager.get_history_context(session_id)
    full_query = history_context + "\n[Current Question]:\n" + req.message if history_context else req.message

    result = pipeline.query(
        full_query,
        verbose=False,
        search_mode=req.search_mode,
        top_k=req.top_k,
        threshold=req.threshold,
        use_reranker=req.use_reranker,
    )
    
    session_manager.add_message(session_id, "user", req.message)
    session_manager.add_message(session_id, "assistant", result["answer"])

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        status=result.get("status", "ok"),
        user_guidance=result.get("user_guidance", ""),
        retrieval_quality=result.get("retrieval_quality", {}),
        session_id=session_id
    )

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id
    if not session_id:
        session_id = session_manager.create_session()

    # Get history context
    history_context = session_manager.get_history_context(session_id)
    full_query = history_context + "\n[Current Question]:\n" + req.message if history_context else req.message

    async def event_generator():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield sse({
            "type": "progress",
            "session_id": session_id,
            "stage": "session.ready",
            "message": "🧠 已加载会话上下文，开始处理问题..."
        })

        # Setup retrieval
        if len(pipeline.store) == 0:
            quality = {"status": "no_docs"}
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "retrieval.skipped",
                "message": "📂 当前还没有索引文档，跳过检索。"
            })
            yield sse({
                "type": "notice",
                "session_id": session_id,
                "message": "No documents indexed. Answering from general knowledge."
            })
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "generation.start",
                "message": "✍️ 正在基于通用知识流式生成回答..."
            })
            
            answer_parts = []
            for chunk in generate_chat(full_query, stream=True, yield_chunks=True):
                answer_parts.append(chunk)
                yield sse({"type": "answer.delta", "chunk": chunk})
            
            final_answer = "".join(answer_parts)
            session_manager.add_message(session_id, "user", req.message)
            session_manager.add_message(session_id, "assistant", final_answer)
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "generation.done",
                "message": "✅ 回答生成完成。"
            })
            yield sse({"type": "answer.final", "answer": final_answer})
            yield "data: [DONE]\n\n"
            return

        yield sse({
            "type": "progress",
            "session_id": session_id,
            "stage": "retrieval.start",
            "message": "🔎 正在检索相关文档片段..."
        })
        retriever = Retriever(
            pipeline.store,
            top_k=req.top_k,
            threshold=req.threshold,
            search_mode=req.search_mode,
            use_reranker=req.use_reranker,
        )
        results = retriever.retrieve(full_query)
        quality = pipeline._analyze_retrieval_quality(
            results,
            search_mode=req.search_mode,
            threshold=req.threshold,
            top_k=req.top_k,
        )
        yield sse({
            "type": "progress",
            "session_id": session_id,
            "stage": "retrieval.done",
            "message": f"📚 检索完成，命中 {len(results)} 个候选片段。"
        })

        if quality["status"] in ["no_relevant", "low_confidence"]:
            if quality["status"] == "no_relevant":
                msg = "No relevant documents found. Answering from general knowledge."
            else:
                msg = f"Low retrieval confidence (top score: {quality['top_score']:.2f}). Answering from general knowledge."
                
            yield sse({"type": "notice", "session_id": session_id, "message": msg})
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "generation.start",
                "message": "🤖 文档上下文不足，切换为通用知识流式回答..."
            })
            
            answer_parts = []
            for chunk in generate_chat(full_query, stream=True, yield_chunks=True):
                answer_parts.append(chunk)
                yield sse({"type": "answer.delta", "chunk": chunk})
            
            final_answer = "".join(answer_parts)
            session_manager.add_message(session_id, "user", req.message)
            session_manager.add_message(session_id, "assistant", final_answer)
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "generation.done",
                "message": "✅ 回答生成完成。"
            })
            yield sse({"type": "answer.final", "answer": final_answer})
        else:
            # Send citations first
            citations = []
            for i, r in enumerate(results):
                citations.append({
                    "id": i + 1,
                    "source": r.get("metadata", {}).get("source", "unknown"),
                    "score": r.get("score", 0.0),
                    "snippet": r.get("text", "")[:150] + "..."
                })
                
            yield sse({"type": "citations", "session_id": session_id, "citations": citations})
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "generation.start",
                "message": f"✍️ 已锁定 {len(citations)} 个来源，开始流式生成回答..."
            })
            
            answer_parts = []
            for chunk in generate(full_query, results, stream=True, yield_chunks=True):
                answer_parts.append(chunk)
                yield sse({"type": "answer.delta", "chunk": chunk})
            
            final_answer = "".join(answer_parts)
            session_manager.add_message(session_id, "user", req.message)
            session_manager.add_message(session_id, "assistant", final_answer)
            yield sse({
                "type": "progress",
                "session_id": session_id,
                "stage": "generation.done",
                "message": "✅ 文档回答生成完成。"
            })
            yield sse({"type": "answer.final", "answer": final_answer})
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
