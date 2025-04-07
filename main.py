# main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, field_validator
import base64
from pathlib import Path
import os
import sys
import json
from datetime import datetime
# Adjust the path to access your pipeline
from langGraph.pipeline import build_pipeline,generate_markdown_report
from agents.llmselection import LLMSelector

# Create a reports directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

app = FastAPI(title="Crime Report API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for CORS
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods for CORS
    allow_headers=["*"],  # Allow all headers for CORS
)

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
    

class ReportResponse(BaseModel):
    """Response with report ID and status URL"""
    report_id: str
    status_url: str

class ReportStatus(BaseModel):
    """Status of the report generation"""
    report_id: str
    status: str
    evaluation: Optional[Dict] = None      
    markdown_url: Optional[str] = None
    token_usage_summary: Optional[Dict] = None
    final_report: Optional[Dict] = None

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
        """Process report generation asynchronously"""
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
            token_usage_summary = final_report.get("token_usage_summary", {
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_node": {},
            "model_info": {}
            })
            
            # Generate markdown using pipeline's function
            md_filename = f"{report_id}.md"
            generate_markdown_report(final_report, md_filename)
            
            # Create downloadable version with base64 images
            download_filename = f"download_{report_id}.md"
            with open(md_filename, 'r', encoding='utf-8') as f:
                content = f.read()
            processed_content = process_markdown_with_base64_images(content, report_id)
            with open(download_filename, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            evaluation = final_report.get("evaluation", {})
            
            # Update report status
            reports[report_id] = {
                "report_id": report_id,
                "status": "completed",
                "report_file": md_filename,
                "download_file": download_filename,
                "content": processed_content,
                "evaluation": evaluation,
                "token_usage_summary": token_usage_summary,
                "final_report": final_report,
                "model_type": request.model_type
            }
                
        except Exception as e:
            print(f"Error processing report: {str(e)}")
            reports[report_id].update({
                "status": "failed",
                "error": str(e)
            })
        
    
    # Start processing in background
    import asyncio
    asyncio.create_task(process_report())
    
    # Return immediately with the report ID
    return {"report_id": report_id, "status_url": f"/report_status/{report_id}"}

@app.get("/report_status/{report_id}")
async def get_report_status(report_id: str):
    if report_id not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report_data = dict(reports[report_id])
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


@app.get("/forecast_code/{report_id}")
async def get_forecast_code(report_id: str):
    """Get the forecast code for a report"""
    if report_id not in reports:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if reports[report_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Report not ready")
    
    if "forecast_code" in reports[report_id]:
        return Response(
            content=reports[report_id]["forecast_code"],
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=crime_forecast_{report_id}.py"}
        )
    
    
    raise HTTPException(status_code=404, detail="Forecast code not found")

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



@app.get("/available_models")
async def get_available_models():
    """Get the list of available models"""
    # Get all models from LLMSelector without filtering
    available_models = LLMSelector.get_available_models()
    return {
        "models": list(available_models.keys())
    }    

