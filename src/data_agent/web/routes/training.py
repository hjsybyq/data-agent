"""
Training API Routes

Handles training data management (add, list, delete examples).
"""

from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from data_agent.web.routes.chat import get_agent


router = APIRouter()


class TrainRequest(BaseModel):
    """Request model for training endpoint."""
    question: Optional[str] = None
    sql: Optional[str] = None
    ddl: Optional[str] = None
    documentation: Optional[str] = None


class TrainResponse(BaseModel):
    """Response model for training endpoint."""
    success: bool
    id: str
    message: str


class TrainingExample(BaseModel):
    """Training example model."""
    id: str
    question: str
    sql: str
    metadata: Optional[dict] = None


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    question: str
    k: int = 5


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """
    Add training example.
    
    Supports:
    - Question + SQL pair
    - DDL statements
    - Documentation
    """
    agent = get_agent()
    
    try:
        result_id = agent.train(
            question=request.question,
            sql=request.sql,
            ddl=request.ddl,
            documentation=request.documentation,
        )
        
        return TrainResponse(
            success=True,
            id=result_id or "",
            message="Training data added successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training", response_model=List[TrainingExample])
async def get_training_data():
    """Get all training examples."""
    agent = get_agent()
    
    try:
        examples = agent.get_training_data()
        return [
            TrainingExample(
                id=ex.get("id", ""),
                question=ex.get("question", ""),
                sql=ex.get("sql", ""),
                metadata=ex.get("metadata"),
            )
            for ex in examples
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/training/{example_id}")
async def delete_training_data(example_id: str):
    """Delete a training example by ID."""
    agent = get_agent()
    
    try:
        success = agent.remove_training_data(example_id)
        
        if success:
            return {"success": True, "message": f"Deleted example {example_id}"}
        else:
            raise HTTPException(status_code=404, detail="Example not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/search")
async def search_similar(request: SearchRequest):
    """Search for similar training examples."""
    agent = get_agent()
    
    try:
        results = agent.search_similar_examples(request.question, k=request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
