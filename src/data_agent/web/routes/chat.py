"""
Chat API Routes

Handles question-answering and SSE streaming for the chat interface.
"""

import json
import asyncio
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from data_agent import DataAgent


router = APIRouter()

# Global agent instance (will be configured at startup)
_agent: Optional[DataAgent] = None
_schema: str = ""


class AskRequest(BaseModel):
    """Request model for ask endpoint."""
    question: str
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    """Response model for ask endpoint."""
    sql: Optional[str]
    result: Optional[dict]
    answer: str
    conversation_id: str
    question_analysis: Optional[dict] = None
    sub_questions: Optional[list] = None
    # Agent mode: TodoList task planning
    todo_list: Optional[list] = None
    execution_steps: Optional[list] = None


class SchemaRequest(BaseModel):
    """Request model for schema endpoint."""
    schema_text: str


def get_agent() -> DataAgent:
    """Get or create the DataAgent instance."""
    global _agent
    if _agent is None:
        _agent = DataAgent(
            llm_provider="openai_compatible",
            base_url="https://api.siliconflow.cn/v1",
            llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            llm_api_key="",  # Will be set via environment or config
            use_agent_mode=True,  # 启用新的 Agent 模式
        )
        if _schema:
            _agent.set_schema(_schema)
    return _agent


def configure_agent(
    llm_provider: str = "openai",
    base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    database_connection = None,
    use_agent_mode: bool = True,  # 默认启用 Agent 模式
):
    """Configure the global agent instance."""
    global _agent
    _agent = DataAgent(
        llm_provider=llm_provider,
        base_url=base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        database_connection=database_connection,
        use_agent_mode=use_agent_mode,
    )
    if _schema:
        _agent.set_schema(_schema)
    return _agent


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get SQL + answer.
    
    Returns the complete response after processing.
    """
    agent = get_agent()
    
    try:
        result = agent.ask(
            question=request.question,
            conversation_id=request.conversation_id,
        )
        
        return AskResponse(
            sql=result.get("sql"),
            result=result.get("result"),
            answer=result.get("answer", ""),
            conversation_id=result.get("conversation_id", ""),
            question_analysis=result.get("question_analysis"),
            sub_questions=result.get("sub_questions"),
            todo_list=result.get("todo_list"),
            execution_steps=result.get("execution_steps"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ask/stream")
async def ask_stream(
    q: str = Query(..., description="The question to ask"),
    conversation_id: Optional[str] = Query(None, description="Conversation ID"),
):
    """
    Ask a question with SSE streaming response.
    
    Events:
    - {"type": "thinking", "data": "..."} - Processing stage
    - {"type": "sql", "data": "SELECT..."} - Generated SQL
    - {"type": "result", "data": {...}} - Query result
    - {"type": "answer", "data": "..."} - Natural language answer
    - {"type": "done", "data": null} - Stream complete
    - {"type": "error", "data": "..."} - Error message
    """
    async def event_generator():
        agent = get_agent()
        
        try:
            # Send thinking event
            yield f"data: {json.dumps({'type': 'thinking', 'data': '正在分析问题...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Get the result (this blocks, but we wrap it)
            result = await asyncio.to_thread(
                agent.ask,
                question=q,
                conversation_id=conversation_id,
            )
            
            # Send SQL event
            if result.get("sql"):
                yield f"data: {json.dumps({'type': 'sql', 'data': result['sql']})}\n\n"
                await asyncio.sleep(0.1)
            
            # Send todo_list event (Agent mode task planning)
            if result.get("todo_list"):
                yield f"data: {json.dumps({'type': 'todo_list', 'data': result['todo_list']})}\n\n"
                await asyncio.sleep(0.1)
            
            # Send execution_steps event
            if result.get("execution_steps"):
                yield f"data: {json.dumps({'type': 'execution_steps', 'data': result['execution_steps']})}\n\n"
                await asyncio.sleep(0.1)
            
            # Send sub-questions if any (Graph mode)
            if result.get("sub_questions"):
                yield f"data: {json.dumps({'type': 'sub_questions', 'data': result['sub_questions']})}\n\n"
                await asyncio.sleep(0.1)
            
            # Send result event
            if result.get("result"):
                yield f"data: {json.dumps({'type': 'result', 'data': result['result']})}\n\n"
                await asyncio.sleep(0.1)
            
            # Send answer event
            yield f"data: {json.dumps({'type': 'answer', 'data': result.get('answer', '')})}\n\n"
            
            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'data': result.get('conversation_id', '')})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/schema")
async def get_schema():
    """Get the current database schema."""
    global _schema
    return {"schema": _schema}


@router.post("/schema")
async def set_schema(request: SchemaRequest):
    """Set the database schema."""
    global _schema, _agent
    _schema = request.schema_text
    
    if _agent:
        _agent.set_schema(_schema)
    
    return {"success": True, "message": "Schema updated"}
