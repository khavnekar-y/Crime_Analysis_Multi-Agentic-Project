"""
Criminal Report Generation Pipeline using LangGraph
Integrates multiple agents for comprehensive criminal report analysis
"""

import os
import sys
import operator
import traceback
import json
from typing import TypedDict, Dict, Any, List, Annotated, Optional
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64
import tempfile
import requests
import re
import google.generativeai as genai
# Add to imports at top of file
import numpy as np
# LangChain and LangGraph imports
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from graphviz import Digraph

# Add parent directory to path for imports from other project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent modules
from agents.websearch_agent import tavily_search, tavily_extract, build_markdown_report
from agents.rag_agent import RAGAgent
# Make sure this is at the top
from agents.snowflake_utils import CrimeDataAnalyzer, initialize_connections, CrimeReportRequest
from agents.Comparision_agent import ComparisonAgent
from agents.judge_agent import JudgeAgent
from agents.llmselection import LLMSelector as llmselection
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

###############################################################################
# State definition for Criminal Report Generation
###############################################################################
class CrimeReportState(TypedDict, total=False):
    """State definition for the crime report generation pipeline"""
    # Input parameters
    question: str  # User's query (removing redundant 'input')
    search_mode: str  # "all_years" or "specific_range"
    start_year: Optional[int]  # Starting year for analysis
    end_year: Optional[int]  # Ending year for analysis
    selected_regions: List[str]  # Cities/regions to analyze
    model_type: str  # LLM model to use
    
    # Agent outputs
    web_output: Dict[str, Any]  # Results from WebSearchAgent
    rag_output: Dict[str, Any]  # Results from RAGAgent
    snowflake_output: Dict[str, Any]  # Results from CrimeDataAnalyzer
    comparison_output: Dict[str, Any]  # Results from ComparisonAgent
    forecast_output: Dict[str, Any]  # Crime trend forecasts
    #Report organization 
    report_sections: Dict[str, Any]  # Structured report sections with content
    visualizations: Dict[str, Any]  # All visualizations from different sources
    
    # Processing metadata
    chat_history: List[Dict[str, Any]]  # Conversation history with feedback
    intermediate_steps: Annotated[List[tuple[Any, str]], operator.add]  # Agent reasoning steps
    
    # Final outputs and evaluation
    final_report: Dict[str, Any]  # Final structured report including:
                                 # - sections
                                 # - visualizations
                                 # - metadata
                                 # - quality assessment
    
    evaluation: Dict[str, Any]  # Combined quality assessment including:
                               # - judge feedback
                               # - quality scores
                               # - improvement suggestions
                               # - historical feedback

###############################################################################
# Node Functions
###############################################################################
def start_node(state: CrimeReportState) -> Dict:
    """Initial node that processes the input query and initializes the pipeline."""
    print(f"\n🚀 Starting crime report generation for: {state['question']}")
    print(f"📊 Analysis parameters:")
    print(f"- Regions: {', '.join(state['selected_regions'])}")
    print(f"- Time period: {state['search_mode']}")
    if state['search_mode'] == "specific_range":
        print(f"- Years: {state['start_year']} - {state['end_year']}")
    print(f"- Model: {state['model_type']}")
    
    return {
        "question": state["question"],
        "search_mode": state["search_mode"],
        "selected_regions": state["selected_regions"],
        "model_type": state["model_type"],
        "start_year": state.get("start_year"),
        "end_year": state.get("end_year")
    }


def web_search_node(state: CrimeReportState) -> Dict:
    """Execute web search for crime data using websearch_agent functions."""
    try:
        print("\n🔍 Executing web search for latest crime reports and news...")
        
        # Create query based on state parameters
        query = state["question"]
        if state["search_mode"] == "specific_range":
            query += f" between {state['start_year']} and {state['end_year']}"
        
        # Execute tavily search using functions from websearch_agent.py
        search_response = tavily_search(
            query=query,
            selected_regions=state["selected_regions"],
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            search_mode=state["search_mode"],
            topic="news",
            max_results=8
        )
        
        if not search_response:
            raise Exception("Search failed - no response from Tavily API")
            
        # Extract content from URLs
        urls = [item["url"] for item in search_response.get("results", []) 
                if "url" in item]
        extract_response = tavily_extract(urls=urls)
        
        # Build the report
        result_json = build_markdown_report(query, search_response, extract_response)
        result = json.loads(result_json)
        
        print(f"✅ Web search complete - found {result['metadata']['result_count']} results")
        return {"web_output": result}
    
    except Exception as e:
        print(f"❌ Web search error: {str(e)}")
        traceback.print_exc()
        return {"web_output": {
            "markdown_report": f"Error during web search: {str(e)}",
            "images": [],
            "links": [],
            "metadata": {"error": str(e)}
        }}

def rag_node(state: CrimeReportState) -> Dict:
    try:
        print("\n📚 Retrieving historical crime data using RAG...")
        rag_agent = RAGAgent()
        
        # Remove filters parameter that's causing the error
        result = rag_agent.process(
            query=state["question"],
            search_mode=state["search_mode"],
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            selected_regions=state["selected_regions"]
        )
        
        return {"rag_output": result}
        
    except Exception as e:
        print(f"❌ RAG analysis error: {str(e)}")
        traceback.print_exc()
        return {"rag_output": {"error": str(e)}}
    
def snowflake_node(state: CrimeReportState) -> Dict:
    """Execute Snowflake analysis for crime data visualizations."""
    try:
        print("\n📊 Analyzing crime data from Snowflake...")
        
        # Initialize connections
        from agents.snowflake_utils import initialize_connections
        engine, llm = initialize_connections()
        
        # Initialize the CrimeDataAnalyzer
        analyzer = CrimeDataAnalyzer(engine, llm)
        
        # Import the request model
        from agents.snowflake_utils import CrimeReportRequest
        from pydantic import BaseModel
        
        # Create the request
        request = CrimeReportRequest(
            question=state["question"],
            search_mode=state["search_mode"],
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            selected_regions=state["selected_regions"],
            model_type=state["model_type"]
        )
        
        # Execute the analysis
        result = analyzer.analyze_crime_data(request)
        
        print(f"✅ Snowflake analysis complete - generated {len(result.get('visualizations', {}).get('paths', {}))} visualizations")
        return {"snowflake_output": result}
    
    except Exception as e:
        print(f"❌ Snowflake analysis error: {str(e)}")
        traceback.print_exc()
        return {"snowflake_output": {"error": str(e), "status": "failed"}}


def contextual_image_node(state: CrimeReportState) -> Dict:
    """Generate contextual images based on crime data insights using Gemini native image generation."""
    try:
        print("\n🎨 Generating contextual images for the report...")
        
        # Collect insights from various sources
        snowflake_insights = state.get("snowflake_output", {}).get("analysis", "")
        rag_insights = state.get("rag_output", {}).get("insights", "")
        web_insights = state.get("web_output", {}).get("markdown_report", "")[:500]
        
        # Extract key themes for image generation
        llm = llmselection.get_llm(state["model_type"])
        
        theme_prompt = f"""
        Based on the following crime data insights, identify 3 key visual themes that would 
        enhance the report with meaningful contextual images. Each theme should be specific 
        enough for image generation.
        
        INSIGHTS FROM DATA:
        {snowflake_insights[:300]}...
        {rag_insights[:300]}...
        
        For each theme:
        1. Provide a descriptive title (3-5 words)
        2. Create a detailed image prompt (50-80 words) for an image generator - focus on professional, 
           data visualization style imagery appropriate for a crime report
        3. Explain why this visual would enhance the report
        
        Format as JSON:
        {{
            "themes": [
                {{
                    "title": "Theme title",
                    "image_prompt": "Detailed prompt for image generator",
                    "rationale": "Why this image matters"
                }}
            ]
        }}
        """
        
        themes_response = llmselection.get_response(llm, theme_prompt)
        
        # Parse themes
        try:
            themes_data = json.loads(themes_response)
            themes = themes_data.get("themes", [])
        except:
            # Fallback if JSON parsing fails
            print("Warning: Could not parse themes JSON, using default themes")
            themes = [
                {
                    "title": "Crime Trend Visualization",
                    "image_prompt": f"Create a professional data visualization showing crime trends in {', '.join(state['selected_regions'])} with clear decreasing or increasing patterns. Use a clean blue color scheme with data points and trend lines. Include a legend and axis labels. Modern analytical style.",
                    "rationale": "Provides visual overview of key trends"
                },
                {
                    "title": "Safety Measures Illustration",
                    "image_prompt": f"A professional illustration of community safety measures in an urban setting resembling {state['selected_regions'][0]}. Show police presence, neighborhood watch systems, and well-lit streets with security cameras. Use a cool color palette with blue and green tones. Informative style with labels.",
                    "rationale": "Visualizes key safety recommendations"
                },
                {
                    "title": "Crime Prevention Concept",
                    "image_prompt": f"Create a conceptual image of crime prevention strategies in {state['selected_regions'][0]}. Show community engagement programs, youth activities, and environmental design improvements that deter crime. Use a modern, clean style with infographic elements and a professional color scheme. Include small explanatory labels.",
                    "rationale": "Supports prevention recommendations"
                }
            ]
        
        # Generate images using Gemini native image generation
        contextual_images = {}
        
        try:
            # Initialize Google Gemini API with proper credentials
            from google import genai
            from google.genai import types
            from PIL import Image 
            from io import BytesIO
            import base64
            
            # Set up the Gemini client
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
            client = genai.Client()
            
            print("✅ Connected to Google Gemini API for image generation")
            
            for i, theme in enumerate(themes[:3]):  # Limit to 3 images
                try:
                    # Get the image prompt
                    img_prompt = theme.get("image_prompt")
                    print(f"Generating image for: {theme.get('title')}")
                    print(f"Using prompt: {img_prompt[:100]}...")
                    
                    # Call Gemini image generation model
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp-image-generation",
                        contents=img_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=['Text', 'Image']
                        )
                    )
                    
                    # Extract and save the generated image
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            # Save the generated image
                            img_filename = f"contextual_image_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            image = Image.open(BytesIO(part.inline_data.data))
                            image.save(img_filename)
                            
                            # Add to results
                            contextual_images[theme.get("title")] = {
                                "path": img_filename,
                                "prompt": img_prompt,
                                "rationale": theme.get("rationale")
                            }
                            
                            print(f"✅ Generated contextual image: {theme.get('title')}")
                            break
                        elif part.text is not None:
                            print(f"Image generation model returned text: {part.text[:100]}...")
                except Exception as e:
                    print(f"Error generating image '{theme.get('title')}': {e}")
                    # Continue with next theme if one fails
        
        except Exception as e:
            print(f"Could not initialize Gemini image generation: {e}")
            
        # If Gemini image generation fails, fall back to Unsplash
        if not contextual_images:
            print("⚠️ Falling back to Unsplash for images")
            for i, theme in enumerate(themes[:3]):
                try:
                    img_prompt = theme.get("image_prompt")
                    keywords = re.sub(r'[^\w\s]', '', img_prompt).replace(' ', '+')
                    img_url = f"https://source.unsplash.com/featured/?{keywords}"
                    img_response = requests.get(img_url)
                    
                    # Save the image
                    img_filename = f"contextual_image_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    with open(img_filename, "wb") as f:
                        f.write(img_response.content)
                    
                    # Add to results
                    contextual_images[theme.get("title")] = {
                        "path": img_filename,
                        "prompt": img_prompt,
                        "rationale": theme.get("rationale")
                    }
                    
                    print(f"✅ Generated fallback image: {theme.get('title')}")
                except Exception as e:
                    print(f"Error generating fallback image '{theme.get('title')}': {e}")
        
        # Final fallback to static URLs if no images generated
        if not contextual_images:
            print("⚠️ Using placeholder image URLs as final fallback")
            contextual_images = {
                "Crime Hotspots": {
                    "path": "https://source.unsplash.com/featured/?crime,map",
                    "prompt": "Crime hotspots map visualization",
                    "rationale": "Shows geographic distribution of crime"
                },
                "Prevention Strategies": {
                    "path": "https://source.unsplash.com/featured/?community,safety",
                    "prompt": "Community crime prevention visualization",
                    "rationale": "Illustrates prevention strategies"
                }
            }
                    
        print(f"✅ Contextual image generation complete - created {len(contextual_images)} images")
        return {"contextual_images": contextual_images}
        
    except Exception as e:
        print(f"❌ Contextual image generation error: {str(e)}")
        traceback.print_exc()
        return {"contextual_images": {}}


def _create_temporal_comparison(state: CrimeReportState) -> Dict:
    """Create temporal comparison when only one region is selected."""
    try:
        region = state["selected_regions"][0]
        snowflake_data = state.get("snowflake_output", {})
        rag_data = state.get("rag_output", {})
        
        # Use LLM to generate temporal comparison
        llm = llmselection.get_llm(state["model_type"])
        
        # Extract statistics if available
        stats = snowflake_data.get("statistics", {})
        
        # Create comparison prompt based on available data
        temporal_prompt = f"""
        Create a detailed temporal comparison of crime patterns in {region} across different time periods.
        
        Statistical data:
        {json.dumps(stats, indent=2)[:1000]}...
        
        Historical context:
        {rag_data.get("insights", "No historical context available")[:500]}...
        
        Please provide:
        1. Year-over-year changes in overall crime rates
        2. Specific crime types showing notable trends over time
        3. Seasonal patterns if discernible
        4. Key turning points or trend changes
        5. Long-term trajectory analysis
        
        Format your response with clear headings, bullet points, and temporal analysis language.
        """
        
        temporal_analysis = llmselection.get_response(llm, temporal_prompt)
        
        # Get visualizations from snowflake output
        visualizations = {}
        if "visualizations" in snowflake_data and "paths" in snowflake_data["visualizations"]:
            visualizations = snowflake_data["visualizations"]["paths"]
        
        comparison_output = {
            "analysis": temporal_analysis,
            "visualizations": visualizations,
            "status": "success",
            "comparison_type": "temporal"  # Flag that this is temporal, not cross-region
        }
        
        print(f"✅ Temporal comparison analysis complete")
        return {"comparison_output": comparison_output}
        
    except Exception as e:
        print(f"❌ Temporal comparison error: {str(e)}")
        traceback.print_exc()
        return {"comparison_output": {"error": str(e), "status": "failed"}}

def comparison_node(state: CrimeReportState) -> Dict:
    """Create comparative analysis using ComparisonAgent with memory."""
    try:
        print("\n🔄 Creating comparative analysis...")
        
        # Get or create comparison agent
        if not hasattr(comparison_node, 'agent'):
            comparison_node.agent = ComparisonAgent(state["model_type"])
        
        # Prepare analysis request
        analysis_request = {
            "regions": state["selected_regions"],
            "snowflake_data": state.get("snowflake_output", {}),
            "rag_data": state.get("rag_output", {}),
            "web_data": state.get("web_output", {})
        }
        
        # Execute analysis
        comparison_type = "cross_region" if len(state["selected_regions"]) > 1 else "temporal"
        result = comparison_node.agent.analyze(analysis_request, comparison_type)
        
        # Store any feedback from previous runs
        if state.get("previous_feedback"):
            comparison_node.agent.store_feedback(
                analysis_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                feedback=state["previous_feedback"]
            )
        
        print(f"✅ Comparison analysis complete - Memory size: {len(comparison_node.agent.memory.chat_memory.messages)} messages")
        return {"comparison_output": result}
        
    except Exception as e:
        print(f"❌ Comparison analysis error: {str(e)}")
        traceback.print_exc()
        return {"comparison_output": {
            "error": str(e),
            "status": "failed"
        }}

def forecast_node(state: CrimeReportState) -> Dict:
    """Generate crime trend forecasts for future periods."""
    try:
        print("\n🔮 Generating crime trend forecasts...")
        
        # Check if we have Snowflake data to work with
        if "snowflake_output" not in state or state["snowflake_output"].get("status") != "success":
            print("⚠️ Skipping forecast - no valid data available")
            return {"forecast_output": {
                "status": "skipped",
                "reason": "No valid data available for forecasting"
            }}
        
        # Use the LLM to generate forecasts based on historical data
        llm = llmselection.get_llm(state["model_type"])
        
        # Extract historical stats from the Snowflake output
        stats = state["snowflake_output"]["statistics"]
        yearly_trends = stats["incident_analysis"]["yearly_trends"]
        
        # Convert to a format the LLM can understand
        trend_data = json.dumps(yearly_trends, indent=2)
        
        # Create prompt for forecasting
        forecast_prompt = f"""
        You are a crime data forecasting expert. Based on the historical crime data provided,
        generate future crime trend forecasts for {', '.join(state['selected_regions'])}.
        
        Historical Crime Data:
        {trend_data}
        
        Please provide:
        1. Short-term forecast (next 1-2 years)
        2. Medium-term forecast (3-5 years)
        3. Long-term forecast (5-10 years)
        4. Key indicators to monitor
        5. Potential intervention points
        
        Format your response with clear headings and bullet points.
        """
        
        # Generate forecast
        forecast = llmselection.get_response(llm, forecast_prompt)
        
        # Create a forecast visualization if possible
        forecast_viz = generate_forecast_visualization(
            yearly_trends,
            state["selected_regions"][0],  # Use first region for visualization
            f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        forecast_output = {
            "forecast": forecast,
            "visualization": forecast_viz,
            "status": "success"
        }
        
        print("✅ Forecast generation complete")
        return {"forecast_output": forecast_output}
        
    except Exception as e:
        print(f"❌ Forecast generation error: {str(e)}")
        traceback.print_exc()
        return {"forecast_output": {"error": str(e), "status": "failed"}}


def safety_assessment_node(state: CrimeReportState) -> Dict:
    """Generate safety assessment and recommendations."""
    try:
        print("\n🛡️ Generating safety assessment...")
        
        # Use the LLM to generate safety assessment
        llm = llmselection.get_llm(state["model_type"])
        
        # Create a simple safety assessment agent
        safety_tool = Tool(
            name="safety_assessment",
            description="Assess safety based on crime data",
            func=lambda x: x  
        )
        
        safety_agent = initialize_agent(
            tools=[safety_tool],
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=1
        )
        
        # Gather data from previous nodes
        web_data = state.get("web_output", {}).get("markdown_report", "No web data available")
        rag_data = state.get("rag_output", {}).get("insights", "No historical insights available")
        snowflake_data = state.get("snowflake_output", {}).get("analysis", "No analysis available")
        
        # Create prompt for safety assessment
        safety_prompt = f"""
        You are a public safety expert. Based on the crime data and analysis provided,
        generate a comprehensive safety assessment for {', '.join(state['selected_regions'])}.
        
        Web Search Data:
        {web_data[:2000]}...
        
        Historical Insights:
        {rag_data}
        
        Statistical Analysis:
        {snowflake_data}
        
        Please provide:
        1. Current Safety Rating (scale 1-10)
        2. High-Risk Areas and Times
        3. Vulnerable Demographics
        4. Safety Recommendations for Residents
        5. Recommendations for Law Enforcement
        6. Potential Policy Interventions
        
        Format your response with clear headings and bullet points.
        """
        
        # Generate safety assessment using the agent
        safety_assessment = safety_agent.run(safety_prompt)
        
        print("✅ Safety assessment complete")
        return {"safety_assessment": safety_assessment}
        
    except Exception as e:
        print(f"❌ Safety assessment error: {str(e)}")
        traceback.print_exc()
        return {"safety_assessment": f"Error generating safety assessment: {str(e)}"}


def report_organization_node(state: CrimeReportState) -> Dict:
    """Organize all data into structured report sections."""
    try:
        print("\n📝 Organizing report sections...")
        
        report_sections = {
            "executive_summary": {
                "title": "Executive Summary",
                "content": "",
                "order": 1
            },
            "methodology": {
                "title": "Methodology and Data Sources",
                "content": "",
                "order": 2
            },
            "historical_context": {
                "title": "Historical Context and Trends",
                "content": "",
                "order": 3
            },
            "current_analysis": {
                "title": "Current Crime Landscape Analysis",
                "content": "",
                "order": 4
            },
            "regional_comparison": {
                "title": "Regional Comparison Analysis",
                "content": "",
                "order": 5,
                "visualizations": []
            },
            "safety_assessment": {
                "title": "Safety Assessment",
                "content": "",
                "order": 6
            },
            "forecast": {
                "title": "Crime Trend Forecast",
                "content": "",
                "order": 7,
                "visualizations": []
            },
            "recommendations": {
                "title": "Recommendations and Interventions",
                "content": "",
                "order": 8
            },
            "appendix": {
                "title": "Appendix: Additional Data and Visualizations",
                "content": "",
                "order": 9,
                "visualizations": []
            }
        }
        
        # Collect visualizations from all sources
        visualizations = {}
        
        # Add Snowflake visualizations
        if "snowflake_output" in state and state["snowflake_output"].get("status") == "success":
            viz_paths = state["snowflake_output"].get("visualizations", {}).get("paths", {})
            for viz_type, path in viz_paths.items():
                visualizations[f"snowflake_{viz_type}"] = path
                report_sections["current_analysis"]["visualizations"] = \
                    report_sections["current_analysis"].get("visualizations", []) + [path]
        
        # Add Comparison visualizations
        if "comparison_output" in state and state["comparison_output"].get("status") == "success":
            for viz_type, path in state["comparison_output"].get("visualizations", {}).items():
                visualizations[f"comparison_{viz_type}"] = path
                report_sections["regional_comparison"]["visualizations"] = \
                    report_sections["regional_comparison"].get("visualizations", []) + [path]
        
        # Add Forecast visualization
        if "forecast_output" in state and state["forecast_output"].get("status") == "success":
            forecast_viz = state["forecast_output"].get("visualization")
            if forecast_viz:
                visualizations["forecast"] = forecast_viz
                report_sections["forecast"]["visualizations"] = [forecast_viz]
        
        # Add web search images
        if "web_output" in state:
            images = state["web_output"].get("images", [])
            for i, img_url in enumerate(images[:3]):  # Limit to first 3 images
                visualizations[f"web_image_{i}"] = img_url
                report_sections["appendix"]["visualizations"] = \
                    report_sections["appendix"].get("visualizations", []) + [img_url]
        
        print(f"✅ Report organization complete - {len(visualizations)} visualizations included")
        return {
            "report_sections": report_sections,
            "visualizations": visualizations
        }
        
    except Exception as e:
        print(f"❌ Report organization error: {str(e)}")
        traceback.print_exc()
        return {"report_sections": {}, "visualizations": {}}


def synthesis_node(state: CrimeReportState) -> Dict:
    """
    Synthesize findings from all sources and populate report sections.
    Use LLM to generate coherent content for each section.
    """
    try:
        print("\n🔄 Synthesizing information across all sources...")
        report_sections = state.get("report_sections", {}) 
        if not report_sections:
            print("⚠️ No report sections available for synthesis")
            org_result = report_organization_node(state)
            report_sections = org_result.get("report_sections", {})

        visualizations = state.get("visualizations", {})
        # Use the LLM for synthesis
        llm = llmselection.get_llm(state["model_type"])
        # Create a synthesis agent
        synthesis_tool = Tool(
            name="information_synthesis",
            description="Synthesize information from multiple sources",
            func=lambda x: x  # Simple pass-through function
        )
        
        synthesis_agent = initialize_agent(
            tools=[synthesis_tool],
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=1
        )
        
        # Gather all available data
        web_data = state.get("web_output", {}).get("markdown_report", "No web data available")
        rag_data = state.get("rag_output", {}).get("insights", "No historical insights available")
        snowflake_data = state.get("snowflake_output", {}).get("analysis", "No analysis available")
        comparison_data = state.get("comparison_output", {}).get("analysis", "No comparison available")
        forecast_data = state.get("forecast_output", {}).get("forecast", "No forecast available")
        safety_data = state.get("safety_assessment", "No safety assessment available")
        
        # Get report sections structure
        report_sections = state["report_sections"]
        if not report_sections:
            organization_result = report_organization_node(state)
            report_sections = organization_result.get("report_sections", {})
        
        # Generate executive summary
        exec_summary_prompt = f"""
        Create a concise executive summary (max 250 words) of the crime analysis report for {', '.join(state['selected_regions'])}. 
        Include key findings from all data sources and the most important recommendations.
        
        Key data points to consider:
        1. {snowflake_data[:200]}...
        2. {rag_data[:200]}...
        3. {web_data[:200]}...
        """
        
        report_sections["executive_summary"]["content"] = synthesis_agent.run(exec_summary_prompt)
        
        # Generate methodology section
        methodology_prompt = f"""
        Create a methodology section (250-300 words) describing the data sources and analytical approaches used in this crime report.
        
        Include these data sources:
        1. Historical crime records (Retrieval Augmented Generation)
        2. Current data analysis (Snowflake database)
        3. Latest news articles and reports (Web search)
        4. Comparative regional analysis
        5. Forecasting methodology
        
        Be specific about the time period ({state.get('start_year', 'all available history')} to {state.get('end_year', 'present')})
        and regions analyzed ({', '.join(state['selected_regions'])}).
        """
        
        report_sections["methodology"]["content"] = synthesis_agent.run(methodology_prompt)
        
        # Generate historical context
        report_sections["historical_context"]["content"] = rag_data
        
        # Generate current analysis
        report_sections["current_analysis"]["content"] = snowflake_data
        
        # Generate regional comparison if available
        if "comparison_output" in state and state["comparison_output"].get("status") == "success":
            report_sections["regional_comparison"]["content"] = comparison_data
        else:
            # Generate single region analysis if comparison unavailable
            single_region_prompt = f"""
            Create a detailed analysis (300-400 words) of crime patterns in {state['selected_regions'][0]} 
            based on the available data. Focus on unique characteristics of this region and notable trends.
            
            Data to consider:
            {snowflake_data[:500]}...
            """
            report_sections["regional_comparison"]["content"] = synthesis_agent.run(single_region_prompt)
            report_sections["regional_comparison"]["title"] = f"Crime Analysis: {state['selected_regions'][0]}"
        
        # Add safety assessment
        report_sections["safety_assessment"]["content"] = safety_data
        
        # Add forecast
        report_sections["forecast"]["content"] = forecast_data if "forecast_output" in state else \
            "Insufficient data for accurate forecasting."
        
        # Generate recommendations
        recommendations_prompt = f"""
        Based on all the crime data and analysis for {', '.join(state['selected_regions'])}, 
        provide comprehensive recommendations for:
        
        1. Law enforcement strategies
        2. Community safety measures
        3. Policy interventions
        4. Resource allocation
        5. Prevention programs
        
        Consider these insights:
        {snowflake_data[:300]}...
        {safety_data[:300] if isinstance(safety_data, str) else ''}...
        {forecast_data[:300] if isinstance(forecast_data, str) and "forecast_output" in state else ''}...
        
        Format with clear headings, bullet points, and prioritize recommendations by impact.
        """
        
        report_sections["recommendations"]["content"] = llmselection.get_response(llm, recommendations_prompt)
        
        print("✅ Content synthesis complete")
        return {"report_sections": report_sections,
                "visualizations": visualizations,
                "synthesis_complete":True
                }
        
    except Exception as e:
        print(f"❌ Synthesis error: {str(e)}")
        traceback.print_exc()
        return {"synthesis_error": str(e)}


def final_report_node(state: CrimeReportState) -> Dict:
    """Assemble the final report with all sections, visualizations and contextual images."""
    try:
        print("\n📊 Generating final crime report...")
        
        # Get report sections and visualizations
        report_sections = state["report_sections"]
        visualizations = state["visualizations"]
        contextual_images = state.get("contextual_images", {})
        
        # Create final report structure
        final_report = {
            "title": f"Crime Analysis Report: {', '.join(state['selected_regions'])}",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "generated_by": os.getenv("USER_NAME", "crime-analysis-system"),
            "query": state["question"],
            "parameters": {
                "regions": state["selected_regions"],
                "time_period": f"{state.get('start_year', 'all history')} to {state.get('end_year', 'present')}",
                "model": state["model_type"]
            },
            "sections": [],
            "visualizations": visualizations,
            "contextual_images": contextual_images,
            "metadata": {
                "source_count": len(visualizations),
                "contextual_images_count": len(contextual_images),
                "word_count": sum(len(section["content"].split()) 
                                 for section in report_sections.values() 
                                 if isinstance(section["content"], str)),
                "section_count": len(report_sections)
            }
        }
        
        # Add sections in correct order
        ordered_sections = sorted(
            [section for section in report_sections.values()],
            key=lambda x: x.get("order", 999)
        )
        
        # Insert contextual images strategically within the sections
        if contextual_images:
            # Add contextual images to relevant sections
            for section in ordered_sections:
                section_title = section.get("title", "").lower()
                section_content = section.get("content", "")
                
                # Find relevant contextual images for this section
                relevant_images = []
                for img_title, img_data in contextual_images.items():
                    # Check if image theme matches section
                    if any(kw in section_title.lower() for kw in img_title.lower().split()):
                        relevant_images.append(img_data)
                    # Or check if image rationale mentions key terms in section
                    elif "rationale" in img_data and any(kw in section_content.lower() for kw in img_data["rationale"].lower().split()):
                        relevant_images.append(img_data)
                
                # Add relevant images to this section
                if relevant_images:
                    if "contextual_images" not in section:
                        section["contextual_images"] = []
                    section["contextual_images"].extend([img["path"] for img in relevant_images])
        
        final_report["sections"] = ordered_sections
        
        # Generate a sample cover image with title
        cover_image_path = generate_report_cover(
            title=final_report["title"],
            regions=state["selected_regions"],
            time_period=final_report["parameters"]["time_period"]
        )
        
        final_report["cover_image"] = cover_image_path
        
        print("✅ Final report generated successfully")
        print(f"📈 Report includes {final_report['metadata']['source_count']} visualizations")
        print(f"🎨 Report includes {final_report['metadata']['contextual_images_count']} contextual images")
        print(f"📝 Report contains {final_report['metadata']['word_count']} words")
        print(f"📚 Report has {final_report['metadata']['section_count']} sections")
        
        return {"final_report": final_report}
        
    except Exception as e:
        print(f"❌ Final report generation error: {str(e)}")
        traceback.print_exc()
        return {"final_report": {
            "error": str(e),
            "title": "ERROR: Report Generation Failed",
            "sections": [{
                "title": "Error Details",
                "content": str(e)
            }]
        }}


def judge_node(state: CrimeReportState) -> Dict:
    """Evaluate report quality using JudgeAgent with memory."""
    try:
        print("\n⚖️ Evaluating report quality...")
        
        # Get or create judge agent
        if not hasattr(judge_node, 'agent'):
            judge_node.agent = JudgeAgent()
        
        # Execute evaluation
        evaluation = judge_node.agent.evaluate(
            report_data=state["final_report"],
            state=state
        )
        
        # Get improvement suggestions for future reports
        improvements = judge_node.agent.get_improvement_suggestions()
        
        print(f"✅ Report evaluation complete - Overall score: {evaluation.get('overall_score', 'N/A')}/10")
        print(f"📝 Stored {len(judge_node.agent.feedback_history)} previous evaluations")
        
        return {
            "judge_feedback": evaluation,
            "quality_scores": evaluation.get("scores", {}),
            "improvement_suggestions": improvements,
            "feedback_history": judge_node.agent.feedback_history[-5:]  # Last 5 evaluations
        }
        
    except Exception as e:
        print(f"❌ Report evaluation error: {str(e)}")
        traceback.print_exc()
        return {
            "judge_feedback": {"error": str(e)},
            "quality_scores": {"overall": 5}
        }

###############################################################################
# Helper Functions for Visualization
###############################################################################
def generate_forecast_visualization(yearly_trends, region: str, output_path: str) -> str:
    """Generate a forecast visualization extending the current trends."""
    try:
        # Extract relevant data
        if not yearly_trends or not isinstance(yearly_trends, dict):
            return ""
            
        # Get years and values
        years = []
        values = []
        
        # Format may vary, try to extract data safely
        if isinstance(yearly_trends, dict):
            for year, data in yearly_trends.items():
                if isinstance(data, dict):
                    # For nested structure
                    years.append(int(year))
                    values.append(sum(val for val in data.values() if isinstance(val, (int, float))))
                else:
                    # For flattened structure
                    years.append(int(year))
                    values.append(float(data))
        
        if not years or not values:
            return ""
            
        # Sort by year
        years, values = zip(*sorted(zip(years, values)))
        
        # Create the forecast (simple linear projection)
        future_years = list(range(max(years) + 1, max(years) + 6))  # 5 years into future
        
        # Calculate trend for forecasting (simple linear regression)
        import numpy as np
        from scipy import stats
        
        slope, intercept, _, _, _ = stats.linregress(years, values)
        future_values = [slope * year + intercept for year in future_years]
        
        # Plot with historical and forecast data
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(years, values, 'b-', label=f'Historical Data ({region})')
        
        # Forecast data
        plt.plot(future_years, future_values, 'r--', label=f'Forecast ({region})')
        
        # Add shading to future area to indicate uncertainty
        plt.fill_between(
            future_years,
            [v * 0.8 for v in future_values],  # Lower bound (20% lower)
            [v * 1.2 for v in future_values],  # Upper bound (20% higher)
            color='red', alpha=0.2
        )
        
        # Add labels and title
        plt.title(f'Crime Trend Forecast for {region}', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Incidents', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to file
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    except Exception as e:
        print(f"Error generating forecast visualization: {e}")
        return ""


def generate_report_cover(title: str, regions: List[str], time_period: str) -> str:
    """Generate a cover image for the report with title and key info."""
    try:
        # Create a blank image
        width, height = 800, 1100
        cover = Image.new('RGB', (width, height), color=(245, 245, 245))
        draw = ImageDraw.Draw(cover)
        
        try:
            # Try to load a nice font, fall back to default if not available
            title_font = ImageFont.truetype("Arial.ttf", 40)
            subtitle_font = ImageFont.truetype("Arial.ttf", 30)
            info_font = ImageFont.truetype("Arial.ttf", 20)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            info_font = ImageFont.load_default()
        
        # Add a decorative header bar
        draw.rectangle([(0, 0), (width, 100)], fill=(30, 50, 100))
        
        # Add title
        title_wrapped = "\n".join([title[i:i+30] for i in range(0, len(title), 30)])
        draw.text((width/2, 200), title_wrapped, font=title_font, fill=(30, 50, 100), anchor="mm")
        
        # Add regions
        regions_text = "Regions Analyzed: " + ", ".join(regions)
        draw.text((width/2, 300), regions_text, font=subtitle_font, fill=(60, 80, 120), anchor="mm")
        
        # Add time period
        period_text = f"Time Period: {time_period}"
        draw.text((width/2, 350), period_text, font=subtitle_font, fill=(60, 80, 120), anchor="mm")
        
        # Add generation info
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((width/2, 450), f"Generated: {gen_time}", font=info_font, fill=(100, 100, 100), anchor="mm")
        
        # Add system info
        draw.text((width/2, 480), "Multi-Agent Crime Analysis System", font=info_font, fill=(100, 100, 100), anchor="mm")
        
        # Add a decorative footer
        draw.rectangle([(0, height-50), (width, height)], fill=(30, 50, 100))
        
        # Save the cover
        output_path = f"report_cover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cover.save(output_path)
        return output_path
        
    except Exception as e:
        print(f"Error generating report cover: {e}")
        return ""


###############################################################################
# Pipeline Building
###############################################################################
def build_pipeline():
    """Build and compile the pipeline for crime report generation with parallel processing."""
    graph = StateGraph(CrimeReportState)
    def parallel_data_gathering(state: CrimeReportState) -> Dict:
        """Execute web search, RAG, and snowflake analysis in parallel."""
        print("\n🚀 Starting parallel data gathering...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            web_future = executor.submit(web_search_node, state)
            rag_future = executor.submit(rag_node, state)
            snowflake_future = executor.submit(snowflake_node, state)
            
            web_result = web_future.result()
            rag_result = rag_future.result()
            snowflake_result = snowflake_future.result()
        
        print("✅ Parallel data gathering complete")
        return {
            **web_result,
            **rag_result,
            **snowflake_result
        }
    # Add nodes
    graph.add_node("start", start_node)
    graph.add_node("parallel_gathering", parallel_data_gathering)
    graph.add_node("organization", report_organization_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("report_generation", final_report_node)
    graph.add_node("judge", judge_node)
    
    # Set entry point
    graph.set_entry_point("start")
    
    # Add edges using the correct LangGraph syntax
    # The add_edge method only takes source and target parameters
    
    graph.add_edge("start", "parallel_gathering")
    
    # Check for required outputs before organization
    graph.add_conditional_edges(
        "parallel_gathering", 
        lambda x: all(k in x for k in ["snowflake_output", "rag_output", "web_output"]),
        {True: "organization"}
    )
    
    # Check for report sections and visualizations
    graph.add_conditional_edges(
        "organization",
        lambda x: "report_sections" in x and "visualizations" in x,
        {True: "synthesis"}
    )
    
    # Check for synthesis completion
    graph.add_conditional_edges(
        "synthesis",
        lambda x: "synthesis_complete" in x,
        {True: "report_generation"}
    )
    
    # Check for final report
    graph.add_conditional_edges(
        "report_generation",
        lambda x: "final_report" in x,
        {True: "judge"}
    )
    
    # Final edge to END state
    graph.add_edge("judge", END)
    
    return graph.compile()

    
###############################################################################
# Main Invocation
###############################################################################
def cleanup_matplotlib():
    """Clean up matplotlib resources"""
    plt.close('all')

if __name__ == "__main__":
    try:
        # Build the crime report pipeline
        pipeline = build_pipeline()
        
        # Default query
        default_query = "Analyze recent criminal incidents trends and patterns"
        
        # Get user input or use default
        print("\n🔍 Crime Report Generator 🔍")
        print("============================")
        query = input(f"Enter your query [press Enter for default: '{default_query}']: ")
        if not query:
            query = default_query
        
        # Get regions
        regions_input = input("Enter regions to analyze (comma-separated) [default: Chicago, New York]: ")
        regions = [r.strip() for r in regions_input.split(",")] if regions_input else ["Chicago", "New York"]
        
        # Get time range
        time_range = input("Enter time range (all_years/specific_range) [default: specific_range]: ")
        time_range = time_range if time_range else "specific_range"
        
        start_year = None
        end_year = None
        if time_range == "specific_range":
            start_year_input = input("Enter start year [default: 2015]: ")
            start_year = int(start_year_input) if start_year_input else 2015
            
            end_year_input = input("Enter end year [default: 2024]: ")
            end_year = int(end_year_input) if end_year_input else 2024
        
        # Get model
        model_input = input("Enter LLM model (Claude 3 Haiku/Claude 3 Sonnet/Gemini Pro) [default: Claude 3 Haiku]: ")
        model = model_input if model_input else "Claude 3 Haiku"
        
        # Initialize state
        initial_state = {
            "question": query,
            "search_mode": time_range,
            "start_year": start_year,
            "end_year": end_year,
            "selected_regions": regions,
            "model_type": model,
            "chat_history": [],
            "intermediate_steps": []
        }
        
        print("\n🚀 Starting analysis pipeline...")
        result = pipeline.invoke(initial_state)
        
        # Display results
        print("\n✅ Analysis Complete!")
        
        final_report = result.get("final_report", {})
        print(f"\n📊 Report: {final_report.get('title', 'No title')}")
        print(f"📊 Sections: {len(final_report.get('sections', []))}")
        print(f"📊 Visualizations: {len(final_report.get('visualizations', []))}")
        
        # Save report to JSON file
        output_filename = f"crime_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n💾 Full report saved to: {output_filename}")
        
        # Save markdown version
        md_filename = output_filename.replace(".json", ".md")
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(f"# {final_report.get('title', 'Crime Report')}\n\n")
            f.write(f"Generated at: {final_report.get('generated_at')}\n\n")
            
            for section in final_report.get('sections', []):
                f.write(f"## {section.get('title', 'Section')}\n\n")
                f.write(f"{section.get('content', '')}\n\n")
                
                # Add visualizations if any
                if "visualizations" in section and section["visualizations"]:
                    f.write("### Visualizations\n\n")
                    for viz in section["visualizations"]:
                        if isinstance(viz, str):
                            if viz.startswith("http"):
                                f.write(f"![Visualization]({viz})\n\n")
                            else:
                                f.write(f"![Visualization](./{viz})\n\n")
        
        print(f"📝 Markdown report saved to: {md_filename}")
        
        # Show quality feedback
        if "judge_feedback" in result and "overall_assessment" in result["judge_feedback"]:
            print("\n⚖️ Quality Assessment:")
            print(f"Overall: {result['judge_feedback'].get('overall_score', 'N/A')}/10")
            print(f"Assessment: {result['judge_feedback'].get('overall_assessment')[:100]}...")
        cleanup_matplotlib()
    except Exception as e:
        print(f"\n❌ Error running pipeline: {str(e)}")
        traceback.print_exc()
    finally:
        # Cleanup matplotlib resources
        cleanup_matplotlib()
        print("\n🧹 Cleaned up resources.")