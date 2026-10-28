import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api import router, pipeline, set_startup_sync

app = FastAPI(title="Local Search RAG API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Auto-ingest on startup
@app.on_event("startup")
def startup_event():
    summary = pipeline.sync_sample_data()
    set_startup_sync(summary)

# Mount frontend
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
