"""
Data Agent Web Application

FastAPI-based web interface for Data Agent text-to-SQL system.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from data_agent.web.routes.chat import router as chat_router
from data_agent.web.routes.training import router as training_router


# Get the directory containing this file
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Data Agent Web UI starting...")
    yield
    # Shutdown
    print("👋 Data Agent Web UI shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Data Agent",
    description="LangGraph-based Text-to-SQL System",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Include routers
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(training_router, prefix="/api", tags=["training"])


@app.get("/")
async def index(request: Request):
    """Serve the main chat page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


def run():
    """Run the development server."""
    import uvicorn
    uvicorn.run(
        "data_agent.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
