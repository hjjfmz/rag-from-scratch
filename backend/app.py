import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import router, service, set_startup_sync

app = FastAPI(title="RAG From Scratch API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# Sync the sample_data directory once at startup (explicit local files only).
@app.on_event("startup")
def startup_event():
    summary = {"changed": False, "status": "no_sample_dir", "files": []}
    sample_dir = Path("sample_data")
    if sample_dir.is_dir():
        summary = service.sync(str(sample_dir))
    set_startup_sync(summary)


# Mount frontend
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
