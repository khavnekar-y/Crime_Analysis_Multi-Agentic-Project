# main.py

from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator
import os
import sys

# Adjust the path to access your internal LangGraph modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from langGraph.pipeline import build_pipeline  # Ensure this matches your pipeline file

app = FastAPI()

# ---------------------------
# 📦 Request Model
# ---------------------------
class CrimeReportRequest(BaseModel):
    """Request model for generating the crime report"""
    question: str
    search_mode: str = "all_years"  # NEW: either "all_years" or "specific_range"
    start_year: Optional[int] = None  # e.g., 1995
    end_year: Optional[int] = None  # e.g., 2018
    selected_regions: List[str]  # Required list of regions to analyze
    model_type: Optional[str] = None  # allows user to choose the model type

    @field_validator('search_mode')
    def validate_search_mode(cls, v):
        if v not in ["all_years", "specific_range"]:
            raise ValueError('search_mode must be either "all_years" or "specific_range"')
        return v

    @field_validator('selected_regions')
    def validate_selected_regions(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one region must be selected')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Generate a full crime report for Chicago",
                "search_mode": "specific_range",  # or "all_years"
                "start_year": 1995,
                "end_year": 2018,
                "selected_regions": ["Chicago", "New York"],
                "model_type": "claude-3-haiku-20240307"
            }
        }
# ---------------------------
# 🚀 API Endpoint (Modified Part)
# ---------------------------
@app.post("/generate_crime_report")
async def generate_crime_report(request: CrimeReportRequest):
    """Main endpoint to generate a crime analysis report using LangGraph pipeline"""
    try:
        state = {
            "input": request.question,
            "question": request.question,
            "search_mode": request.search_mode,
            "start_year": request.start_year,
            "end_year": request.end_year,
            "selected_regions": request.selected_regions,
            "model_type": request.model_type,
            "chat_history": [],
            "intermediate_steps": []
        }

        # 2. Let pipeline internally determine the necessary agents
        pipeline = build_pipeline()

        # 3. Invoke the pipeline
        result = pipeline.invoke(state)

        if not result or "final_report" not in result:
            raise HTTPException(status_code=500, detail="Pipeline did not return a final report")

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )