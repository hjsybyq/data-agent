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
        llm_api_key="sk-cidkntehcueyomwlamjivnosmcworzwmrokzajrdlpdlfchz",
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
            
            # 获取底层 agent
            lang_agent = agent.get_agent()
            if lang_agent is None:
                # Fallback to non-streaming
                result = await asyncio.to_thread(agent.ask, question=q, conversation_id=conversation_id)
                if result.get("answer"):
                    yield f"data: {json.dumps({'type': 'answer', 'data': result['answer']})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'data': result.get('conversation_id', '')})}\n\n"
                return
            
            # 构建消息
            messages = [{"role": "user", "content": q}]
            
            # 收集数据用于历史记录
            final_answer = ""
            execution_steps = []
            current_sql = None
            sql_result = None
            
            # 工具名称映射
            TOOL_DISPLAY = {
                "get_database_schema": "📋 获取 Schema",
                "search_examples": "🔍 检索样例",
                "validate_sql": "✓ 验证 SQL",
                "execute_sql": "▶ 生成 SQL",  # 只显示生成SQL，执行结果单独一行
            }
            
            # 真正的流式输出 (增加 recursion_limit 防止复杂查询失败)
            async for event in lang_agent.astream_events(
                {"messages": messages},
                version="v2",
                config={"recursion_limit": 100},
            ):
                kind = event.get("event")
                name = event.get("name", "")
                
                # 详细日志：写入文件便于调试
                with open("stream_events.log", "a", encoding="utf-8") as f:
                    f.write(f"[STREAM] Event: {kind} | Name: {name}\n")
                    if kind in ["on_tool_start", "on_tool_end"]:
                        data = event.get("data", {})
                        f.write(f"  Data keys: {data.keys() if isinstance(data, dict) else type(data)}\n")
                        if kind == "on_tool_end":
                            output = data.get("output", "")
                            f.write(f"  Output: {str(output)[:500]}\n")
                        f.write("\n")
                
                # LLM Token 流式输出
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        final_answer += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'data': chunk.content})}\n\n"
                
                # 工具调用开始
                elif kind == "on_tool_start":
                    tool_name = name
                    tool_input = event.get("data", {}).get("input", {})
                    
                    if tool_name in TOOL_DISPLAY:
                        step = {
                            "tool": TOOL_DISPLAY[tool_name],
                            "tool_id": tool_name,
                            "summary": "执行中...",
                            "status": "running",
                        }
                        if tool_name == "execute_sql":
                            current_sql = tool_input.get("sql")
                            step["sql"] = current_sql
                        execution_steps.append(step)
                        yield f"data: {json.dumps({'type': 'execution_steps', 'data': make_serializable(execution_steps)})}\n\n"
                
                # 工具调用结束
                elif kind == "on_tool_end":
                    tool_name = name
                    output_obj = event.get("data", {}).get("output", "")
                    
                    # 处理 output 对象 - 提取 content 属性
                    if hasattr(output_obj, "content"):
                        output = output_obj.content
                    else:
                        output = str(output_obj)
                    
                    # 处理 write_todos 工具
                    if tool_name == "write_todos":
                        try:
                            # 从 Command 对象中提取 todos
                            if hasattr(output_obj, "update") and "todos" in getattr(output_obj, "update", {}):
                                todos = output_obj.update.get("todos", [])
                            elif isinstance(output, str) and "todos" in output:
                                import re
                                # 尝试解析 todos
                                todos_match = re.search(r"'todos':\s*\[(.*?)\]", output, re.DOTALL)
                                if todos_match:
                                    # 简化解析：提取 content 字段
                                    contents = re.findall(r"'content':\s*'([^']+)'", output)
                                    statuses = re.findall(r"'status':\s*'([^']+)'", output)
                                    todos = [{"content": c, "status": s} for c, s in zip(contents, statuses)]
                            else:
                                todos = []
                            
                            if todos:
                                yield f"data: {json.dumps({'type': 'todo_list', 'data': make_serializable(todos)})}\n\n"
                        except Exception as e:
                            print(f"[STREAM] Error parsing write_todos: {e}")
                    
                    # 处理其他核心工具
                    elif tool_name in TOOL_DISPLAY and execution_steps:
                        # 更新最后一个步骤
                        for step in reversed(execution_steps):
                            if step.get("tool_id") == tool_name:
                                step["status"] = "success"
                                
                                # 根据工具类型设置摘要
                                if tool_name == "get_database_schema":
                                    import re
                                    tables = re.findall(r'CREATE TABLE (\w+)', output)
                                    step["summary"] = f"发现 {len(tables)} 张表"
                                elif tool_name == "search_examples":
                                    # 解析样例数量并发送样例数据
                                    try:
                                        import json as json_mod
                                        if "[]" in output or output.strip() == "[]":
                                            step["summary"] = "无相似样例"
                                        elif "[" in output:
                                            # 尝试解析 JSON 格式的样例
                                            try:
                                                examples_data = json_mod.loads(output)
                                                if isinstance(examples_data, list) and len(examples_data) > 0:
                                                    step["summary"] = f"发现 {len(examples_data)} 个样例"
                                                    # 发送样例数据给前端
                                                    yield f"data: {json.dumps({'type': 'examples', 'data': make_serializable(examples_data)})}\n\n"
                                                else:
                                                    step["summary"] = "无相似样例"
                                            except:
                                                # 非标准 JSON，用正则解析
                                                import re
                                                examples = re.findall(r'\{[^{}]+\}', output)
                                                if examples:
                                                    step["summary"] = f"发现 {len(examples)} 个样例"
                                                else:
                                                    step["summary"] = "检索完成"
                                        else:
                                            step["summary"] = "检索完成"
                                    except:
                                        step["summary"] = "检索完成"
                                elif tool_name == "validate_sql":
                                    step["summary"] = "验证通过" if "true" in output.lower() else "验证失败"
                                elif tool_name == "execute_sql":
                                    try:
                                        import json as json_mod
                                        result_data = json_mod.loads(output) if isinstance(output, str) else output
                                        if result_data.get("success"):
                                            row_count = result_data.get("row_count", 0)
                                            step["summary"] = "SQL 已生成"
                                            sql_result = result_data
                                            
                                            # 添加单独的"执行结果"步骤
                                            result_step = {
                                                "tool": "📊 执行结果",
                                                "tool_id": "execute_result",
                                                "summary": f"返回 {row_count} 行",
                                                "status": "success",
                                                "result_data": result_data,
                                            }
                                            execution_steps.append(result_step)
                                            
                                            # 发送完整的执行结果数据
                                            yield f"data: {json.dumps({'type': 'result', 'data': make_serializable(result_data)})}\n\n"
                                        else:
                                            step["summary"] = f"失败: {result_data.get('error', '')[:30]}"
                                            step["status"] = "error"
                                    except:
                                        step["summary"] = "执行完成"
                                
                                break
                        
                        yield f"data: {json.dumps({'type': 'execution_steps', 'data': make_serializable(execution_steps)})}\n\n"
            
            # 真流式已通过 token 事件完成，不需要再发送 answer
            # 如果没有收到任何 token，final_answer 会是空的，这时可以发送一个提示
            if not final_answer:
                yield f"data: {json.dumps({'type': 'answer', 'data': '处理完成'})}\n\n"
            
            # 保存到会话历史（使用原有的 ask 方法逻辑）
            # 这里简化处理，实际应该调用 conversation 存储
            
            yield f"data: {json.dumps({'type': 'done', 'data': conversation_id or ''})}\n\n"
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
