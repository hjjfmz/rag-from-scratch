from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    search_mode: str = "vector"
    top_k: int = 5
    threshold: float = 0.3
    use_reranker: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    status: str
    user_guidance: str
    retrieval_quality: Dict[str, Any]
    session_id: str

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    messages: List[Message]
    title: str = "New Chat"

class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    title: str = "New Chat"

class DocumentInfo(BaseModel):
    filename: str
    chunks: int
