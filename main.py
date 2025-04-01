# main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator
import base64
from pathlib import Path
import os
import sys
import json
from datetime import datetime
# Adjust the path to access your pipeline
from langGraph.pipeline import build_pipeline
from agents.llmselection import LLMSelector

# Create a reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

app = FastAPI(title="Crime Report API")

# Simple storage for report status
reports = {}

# ---------------------------
# Helper Functions
# ---------------------------
def convert_image_to_base64(image_path):
    """Convert an image file to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error converting image {image_path}: {str(e)}")
        return None

def process_markdown_with_base64_images(content: str, report_id: str) -> str:
    """Convert all local image references to base64 in markdown"""
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        if '![' in line and '](' in line and not line.startswith('![]'):
            try:
                # Extract image path - handle both ./image_path and just image_path formats
                if '](./' in line:
                    img_path = line.split('](./')[1].split(')')[0]
                elif '](' in line:
                    img_path = line.split('](')[1].split(')')[0]
                    if img_path.startswith('http'):
                        # Skip external URLs
                        processed_lines.append(line)
                        continue
                
                if os.path.exists(img_path):
                    base64_img = convert_image_to_base64(img_path)
                    if base64_img:
                        # Replace local path with base64
                        img_ext = Path(img_path).suffix[1:]  # Get extension without dot
                        if not img_ext:
                            img_ext = "png"  # Default extension
                        line = line.replace(f']({img_path})', 
                                          f'](data:image/{img_ext};base64,{base64_img})')
            except Exception as e:
                print(f"Error processing image in line: {line}, error: {str(e)}")
                
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)
# ---------------------------
# Request and Response Models
# ---------------------------
class CrimeReportRequest(BaseModel):
    """Request model for generating the crime report"""
    question: str
    search_mode: str = "all_years"
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    selected_regions: List[str]
    model_type: str = "Claude 3 Haiku"

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
    
    @field_validator('model_type')
    def validate_model_type(cls, v):
        valid_models = ["Claude 3 Haiku", "Claude 3 Sonnet", "Gemini Pro"]
        if v not in valid_models:
            raise ValueError(f'model_type must be one of: {", ".join(valid_models)}')
        return v

class ReportResponse(BaseModel):
    """Response with report ID and status URL"""
    report_id: str
    status_url: str

class ReportStatus(BaseModel):
    """Status of the report generation"""
    report_id: str
    status: str
    judge_score: Optional[float] = None
    judge_feedback: Optional[Dict] = None  # Add judge_feedback field
    evaluation: Optional[Dict] = None      # Add full evaluation data
    markdown_url: Optional[str] = None

# ---------------------------
# API Endpoints
# ---------------------------
@app.post("/generate_report", response_model=ReportResponse)
async def generate_report(request: CrimeReportRequest):
    """Generate a crime analysis report"""
    report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Store initial report status
    reports[report_id] = {
        "report_id": report_id,
        "status": "processing",
        "start_time": datetime.now().isoformat()
    }
    
        # Start async processing
    async def process_report():
        try:
            # Build and run the pipeline
            pipeline = build_pipeline()
            result = pipeline.invoke({
                "question": request.question,
                "search_mode": request.search_mode,
                "start_year": request.start_year,
                "end_year": request.end_year,
                "selected_regions": request.selected_regions,
                "model_type": request.model_type,
                "chat_history": [],
                "intermediate_steps": []
            })
            
            final_report = result.get("final_report", {})
            judge_feedback = result.get("judge_feedback", {})
            
            md_filename = f"{report_id}.md"
            with open(md_filename, "w", encoding="utf-8") as f:
                # Write cover image
                if final_report.get("cover_image"):
                    f.write(f"![Cover]({final_report.get('cover_image')})\n\n")
                
                # Write title and date
                f.write(f"# {final_report.get('title', 'Crime Report')}\n\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write each section with its images
                for section in final_report.get('sections', []):
                    f.write(f"## {section.get('title', '')}\n\n")
                    f.write(f"{section.get('content', '')}\n\n")
                    
                    if section.get("visualizations"):
                        for viz in section["visualizations"]:
                            f.write(f"![Visualization]({viz})\n\n")
                    
                    if section.get("images"):
                        for img in section["images"]:
                            if isinstance(img, dict) and "path" in img:
                                f.write(f"![{img.get('description', 'Image')}]({img['path']})\n\n")
                
                # Write any remaining contextual images
                if final_report.get("contextual_images"):
                    f.write("## Additional Images\n\n")
                    for img in final_report["contextual_images"]:
                        if isinstance(img, dict) and "path" in img:
                            f.write(f"![{img.get('description', 'Image')}]({img['path']})\n\n")
            
            # Get judge feedback
            evaluation = final_report.get("evaluation", {})
            judge_score = evaluation.get("judge_feedback", {}).get("overall_score", 0)
            judge_feedback = evaluation.get("judge_feedback", {})
            
            # Simple status update
            reports[report_id] = {
                "report_id": report_id,
                "status": "completed",
                "judge_score": float(judge_score),
                "judge_feedback": judge_feedback, # Store full feedback
                "report_file": md_filename,
                "evaluation": evaluation # Store full evaluation data
            }
            
        except Exception as e:
            print(f"Error: {str(e)}")
            reports[report_id] = {
                "report_id": report_id,
                "status": "failed",
                "error": str(e)
            }
    
    # Start processing in background
    import asyncio
    asyncio.create_task(process_report())
    
    # Return immediately with the report ID
    return {"report_id": report_id, "status_url": f"/report_status/{report_id}"}

@app.get("/download_report/{report_id}")
async def download_report(report_id: str):
    """Get the markdown report with base64-encoded images for download"""
    if report_id not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if reports[report_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready")
    
    # Check for the download version with base64 images
    if "download_file" in reports[report_id]:
        download_path = reports[report_id]["download_file"]
        if os.path.exists(download_path):
            return FileResponse(
                download_path, 
                media_type="text/markdown", 
                filename=f"crime_report_{report_id}.md"
            )
    
    # If no download file exists, raise an error
    raise HTTPException(status_code=404, detail="Downloadable report file not found")




@app.get("/report_status/{report_id}", response_model=ReportStatus)
async def get_report_status(report_id: str):
    """Get the status of a report"""
    if report_id not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Ensure all required fields are present in the response
    report_data = dict(reports[report_id])  # Create a copy to avoid modifying the original
    
    # Make sure the report_id is included
    report_data["report_id"] = report_id
    
    # Add judge feedback to status response if it exists
    if "judge_feedback" in report_data:
        report_data["judge_feedback"] = report_data["judge_feedback"]
    
    # Add the markdown URL if the report is completed
    if report_data.get("status") == "completed" and "report_file" in report_data:
        report_data["markdown_url"] = f"/report/{report_id}"
    
    return ReportStatus(**report_data)
@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Get the markdown report for viewing (without base64 images)"""
    if report_id not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if reports[report_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready")
    
    # First check if content is stored directly in memory
    if "content" in reports[report_id]:
        return Response(content=reports[report_id]["content"], media_type="text/markdown")
    
    # Otherwise check for the file
    if "report_file" in reports[report_id]:
        report_path = reports[report_id]["report_file"]
        if os.path.exists(report_path):
            return FileResponse(report_path, media_type="text/markdown")
    
    # If all else fails, check for standard filenames
    standard_path = f"{report_id}.md"
    if os.path.exists(standard_path):
        return FileResponse(standard_path, media_type="text/markdown")
        
    # If no file is found, raise an error
    raise HTTPException(status_code=404, detail="Report file not found")

@app.get("/available_models")
async def get_available_models():
    """Get the list of available models"""
    # Get all models from LLMSelector without filtering
    available_models = LLMSelector.get_available_models()
    return {
        "models": list(available_models.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)