"""
Data Agent Web UI - FastAPI Version

Usage:
    python run_web.py
    
Then open http://localhost:8000 in your browser.
"""

import sys
sys.path.insert(0, "src")

import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path

from sqlalchemy import create_engine

from data_agent import DataAgent

import numpy as np

def make_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    return obj

# ============================================================
# 配置
# ============================================================
CHINOOK_SCHEMA = """
CREATE TABLE Customer (
    CustomerId INTEGER PRIMARY KEY,
    FirstName NVARCHAR(40) NOT NULL,
    LastName NVARCHAR(20) NOT NULL,
    Company NVARCHAR(80),
    City NVARCHAR(40),
    Country NVARCHAR(40),
    Email NVARCHAR(60) NOT NULL,
    SupportRepId INTEGER
);

CREATE TABLE Invoice (
    InvoiceId INTEGER PRIMARY KEY,
    CustomerId INTEGER NOT NULL,
    InvoiceDate DATETIME NOT NULL,
    BillingCity NVARCHAR(40),
    BillingCountry NVARCHAR(40),
    Total NUMERIC(10,2) NOT NULL
);

CREATE TABLE Artist (
    ArtistId INTEGER PRIMARY KEY,
    Name NVARCHAR(120)
);

CREATE TABLE Album (
    AlbumId INTEGER PRIMARY KEY,
    Title NVARCHAR(160) NOT NULL,
    ArtistId INTEGER NOT NULL
);

CREATE TABLE Track (
    TrackId INTEGER PRIMARY KEY,
    Name NVARCHAR(200) NOT NULL,
    AlbumId INTEGER,
    GenreId INTEGER,
    Milliseconds INTEGER NOT NULL,
    UnitPrice NUMERIC(10,2) NOT NULL
);

CREATE TABLE Genre (
    GenreId INTEGER PRIMARY KEY,
    Name NVARCHAR(120)
);

CREATE TABLE InvoiceLine (
    InvoiceLineId INTEGER PRIMARY KEY,
    InvoiceId INTEGER NOT NULL,
    TrackId INTEGER NOT NULL,
    UnitPrice NUMERIC(10,2) NOT NULL,
    Quantity INTEGER NOT NULL
);

CREATE TABLE Employee (
    EmployeeId INTEGER PRIMARY KEY,
    LastName NVARCHAR(20) NOT NULL,
    FirstName NVARCHAR(20) NOT NULL,
    Title NVARCHAR(30)
);
"""

# Global agent
agent: DataAgent = None
_schema = CHINOOK_SCHEMA


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    global agent
    
    print("🚀 启动 Data Agent Web UI...")
    
    # Database
    engine = create_engine("sqlite:///Chinook.sqlite")
    conn = engine.connect()
    print("✓ 数据库连接成功")
    
    # Agent (启用新的 Agent 模式)
    agent = DataAgent(
        llm_provider="openai_compatible",
        base_url="https://api.siliconflow.cn/v1",
        llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        llm_api_key="sk-",
        database_connection=conn,
        embedding_model="BAAI/bge-m3",
        use_agent_mode=True,  # 启用新的 Agent 模式
        enable_hitl=False,     # 关闭人工审核 (避免拦截 execute_sql)
        max_model_calls=20,    # 增加调用次数限制
    )
    agent.set_schema(CHINOOK_SCHEMA)
    print("✓ DataAgent 已配置 (Agent 模式)")
    
    print("\n" + "=" * 50)
    print("🌐 访问 http://localhost:8000")
    print("=" * 50 + "\n")
    
    yield
    
    print("👋 关闭中...")


# FastAPI app
BASE_DIR = Path(__file__).parent / "src" / "data_agent" / "web"
app = FastAPI(title="Data Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ============================================================
# Request/Response Models
# ============================================================
class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class SchemaRequest(BaseModel):
    schema_text: str

class TrainRequest(BaseModel):
    question: Optional[str] = None
    sql: Optional[str] = None
    ddl: Optional[str] = None
    documentation: Optional[str] = None


# ============================================================
# Routes
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/ask/stream")
async def ask_stream(
    q: str = Query(...),
    conversation_id: Optional[str] = Query(None),
):
    """SSE streaming endpoint with step-by-step results."""
    
    async def generate():
        try:
            # Send thinking
            print(f"[SSE] Processing: {q}")
            yield f"data: {json.dumps({'type': 'thinking', 'data': '正在分析问题...'})}\n\n"
            
            # Get result
            print("[SSE] Calling agent.ask()...")
            result = await asyncio.to_thread(
                agent.ask,
                question=q,
                conversation_id=conversation_id,
            )
            
            # Send examples if any
            if result.get("similar_examples"):
                examples = make_serializable(result["similar_examples"])
                yield f"data: {json.dumps({'type': 'examples', 'data': examples})}\n\n"
            
            sql_preview = (result.get('sql') or 'N/A')[:50]
            print(f"[SSE] Got result: {sql_preview}...")
            
            # Send todo_list (Agent 模式任务规划)
            if result.get("todo_list"):
                todo_list = make_serializable(result["todo_list"])
                yield f"data: {json.dumps({'type': 'todo_list', 'data': todo_list})}\n\n"
                print(f"[SSE] Sent todo_list: {len(todo_list)} items")
            
            # Send execution_steps (Agent 模式执行过程)
            if result.get("execution_steps"):
                execution_steps = make_serializable(result["execution_steps"])
                yield f"data: {json.dumps({'type': 'execution_steps', 'data': execution_steps})}\n\n"
                print(f"[SSE] Sent execution_steps: {len(execution_steps)} steps")
            
            # Send sub_questions with step results (Graph 模式)
            sub_questions = result.get("sub_questions")
            sub_results = result.get("sub_results") or {}
            if sub_questions:
                # Combine questions with their results (sub_results is keyed by sq_id like 'sq1', 'sq2')
                enriched_subs = []
                for i, sq in enumerate(sub_questions):
                    sq_id = f"sq{i + 1}"  # Generate key: sq1, sq2, sq3...
                    enriched = {
                        "question": sq.get("question", ""),
                        "sql": sq.get("sql", ""),
                    }
                    # Add result if available
                    if sq_id in sub_results:
                        enriched["result"] = make_serializable(sub_results[sq_id])
                    enriched_subs.append(enriched)
                yield f"data: {json.dumps({'type': 'sub_questions', 'data': enriched_subs})}\n\n"
            
            # Send SQL
            if result.get("sql"):
                yield f"data: {json.dumps({'type': 'sql', 'data': result['sql']})}\n\n"
            
            # Send result
            if result.get("result"):
                result_data = make_serializable(result['result'])
                yield f"data: {json.dumps({'type': 'result', 'data': result_data})}\n\n"
            
            # Send answer
            yield f"data: {json.dumps({'type': 'answer', 'data': result.get('answer', '')})}\n\n"
            
            # Done
            yield f"data: {json.dumps({'type': 'done', 'data': result.get('conversation_id', '')})}\n\n"
            print("[SSE] Complete")
            
        except Exception as e:
            print(f"[SSE] Error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/ask")
async def ask(request: AskRequest):
    """Non-streaming ask endpoint."""
    result = await asyncio.to_thread(
        agent.ask,
        question=request.question,
        conversation_id=request.conversation_id,
    )
    return result


@app.get("/api/schema")
async def get_schema():
    return {"schema": _schema}


@app.post("/api/schema")
async def set_schema(request: SchemaRequest):
    global _schema
    _schema = request.schema_text
    agent.set_schema(_schema)
    return {"success": True}


@app.post("/api/train")
async def train(request: TrainRequest):
    result_id = agent.train(
        question=request.question,
        sql=request.sql,
        ddl=request.ddl,
        documentation=request.documentation,
    )
    return {"success": True, "id": result_id or ""}


@app.get("/api/training")
async def get_training():
    examples = agent.get_training_data()
    return examples


@app.delete("/api/training/{example_id}")
async def delete_training(example_id: str):
    success = agent.remove_training_data(example_id)
    return {"success": success}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
