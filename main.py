from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from langGraph.pipeline import build_pipeline

# Initialize FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    """Request model for research report endpoint"""
    question: str
    search_type: str
    selected_periods: List[str]
    agents: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Analyze NVIDIA's performance",
                "search_type": "Specific Quarter",
                "selected_periods": ["2023q4"],
                "agents": ["RAG Agent", "Web Search Agent"]
            }
        }

@app.post("/research_report")
async def research_report(request: QueryRequest):
    """Generate research report based on query and selected agents"""
    try:
        # Create initial state for pipeline
        state = {
            "input": request.question,
            "question": request.question,
            "search_type": request.search_type,
            "selected_periods": request.selected_periods,
            "chat_history": [],
            "intermediate_steps": [],
            "selected_agents": request.agents  # Pass selected agents to pipeline
        }

        # Initialize pipeline with selected agents
        pipeline = build_pipeline(selected_agents=request.agents)
        
        # Execute pipeline
        result = pipeline.invoke(state)
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Pipeline execution failed to produce results"
            )
            
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

def format_report(report_dict: Dict) -> str:
    """Format report dictionary into markdown text"""
    # ... existing format_report code ...