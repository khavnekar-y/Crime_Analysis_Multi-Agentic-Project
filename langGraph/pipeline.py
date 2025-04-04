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
import os
from openai import OpenAI
# Add to imports at top of file
import numpy as np
import scipy.stats as stats
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
    contextual_images: Dict[str, Any]  # Contextual images generated for the report
    previous_feedback: Optional[str]  # Feedback from previous runs for comparison
    safety_assessment: Dict[str, Any]  # Safety assessment results
    
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
    """Execute RAG analysis for historical crime data."""
    try:
        print("\n📚 Retrieving historical crime data using RAG...")
        
        # Get the model type from state
        model_type = state.get("model_type")
        if model_type:
            print(f"RAG node using model: {model_type}")
        
        # Initialize the RAG agent with the user-selected model
        rag_agent = RAGAgent(model_name=model_type)
        
        # Process the query with the RAG agent
        result = rag_agent.process(
            query=state["question"],
            search_mode=state.get("search_mode", "all_years"),
            start_year=state.get("start_year"),
            end_year=state.get("end_year"),
            selected_regions=state.get("selected_regions", []) 
        )
        
        print(f"✅ RAG analysis complete using {result.get('model_used', model_type)}")
        return {"rag_output": result}
        
    except Exception as e:
        print(f"❌ RAG analysis error: {str(e)}")
        traceback.print_exc()
        return {"rag_output": {"error": str(e), "status": "failed"}}
        
def snowflake_node(state: CrimeReportState) -> Dict:
    """Execute Snowflake analysis for crime data visualizations."""
    try:
        print("\n📊 Analyzing crime data from Snowflake...")
        
        # Initialize connections with the user-selected model
        from agents.snowflake_utils import initialize_connections
        engine, llm = initialize_connections(model_type=state["model_type"])
        
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
    """Generate contextual images for the report."""
    try:
        print("\n🎨 Generating contextual images for the report...")
        
        # Load environment variables
        load_dotenv()
        
        # Configure X.AI client
        XAI_API_KEY = os.getenv("GROK_API_KEY")
        client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
        
        # Define regions for context
        regions = state['selected_regions']
        regions_str = ", ".join(regions)
        
        # Define prompts for different contextual images
        prompts = [
            {
                "title": "Crime Prevention Strategies",
                "prompt": f"A photorealistic image showing modern crime prevention strategies in {regions_str}. Community watch programs, police presence, and advanced surveillance technology working together. Professional style for a data report.",
                "prefix": "crime_prevention"
            },
            {
                "title": f"Urban Safety in {regions[0]}",
                "prompt": f"A photorealistic image of urban safety features in {regions[0]}. Well-lit streets, security cameras, police patrols, and people feeling safe in public spaces. Professional style for a crime analysis report.",
                "prefix": "urban_safety"
            },
            {
                "title": "Community Policing Impact",
                "prompt": "A photorealistic image of effective community policing. Police officers interacting positively with diverse community members, participating in neighborhood events, and building trust. Professional style for a crime analysis report.",
                "prefix": "community_policing"
            }
        ]
        
        # Generate images for each prompt
        contextual_images = {}
        
        for prompt_data in prompts:
            print(f"\n🎨 Generating image for: {prompt_data['prefix']}")
            print(f"Prompt: {prompt_data['prompt']}")
            
            try:
                # Request image from X.AI
                response = client.images.generate(
                    model="grok-2-image-1212",
                    prompt=prompt_data['prompt'],
                    n=1
                )
                
                # Get the image URL
                image_url = response.data[0].url
                print(f"✅ Image URL: {image_url}")
                
                # Download the image
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    # Create timestamped filename
                    image_path = f"{prompt_data['prefix']}.png"
                    
                    # Save the image
                    img = Image.open(BytesIO(img_response.content))
                    img.save(image_path)
                    print(f"✅ Image saved to: {image_path}")
                    
                    contextual_images[prompt_data['title']] = {
                        "path": image_path,
                        "prompt": prompt_data['prompt'],
                        "description": "AI-generated image",
                        "rationale": f"Illustrative image for {prompt_data['title']}"
                    }
            except Exception as e:
                print(f"❌ Error generating image for {prompt_data['title']}: {str(e)}")
        
        # Print summary
        print(f"\n✅ Generated {len(contextual_images)} contextual images")
        for title, data in contextual_images.items():
            print(f"- {title}: {data['path']}")
            
        return {"contextual_images": contextual_images}
        
    except Exception as e:
        print(f"❌ Contextual image generation error: {str(e)}")
        traceback.print_exc()
        return {"contextual_images": {}}

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
        
        # Get insights from RAG and web search
        rag_insights = state.get("rag_output", {}).get("insights", "No historical insights available")
        web_trends = state.get("web_output", {}).get("markdown_report", "No web search data available")
        
        # Limit text size to avoid token issues
        rag_insights_sample = rag_insights[:1000] + "..." if len(rag_insights) > 1000 else rag_insights
        web_trends_sample = web_trends 
        
        # Convert to a format the LLM can understand
        trend_data = json.dumps(yearly_trends, indent=2)
        
        # Create prompt for forecasting with additional context
        forecast_prompt = f"""
        You are a crime data forecasting expert. Based on the historical crime data and insights provided,
        generate future crime trend forecasts for {', '.join(state['selected_regions'])}.
        
        Historical Crime Data:
        {trend_data}
        
        Historical Context (from RAG):
        {rag_insights_sample}
        
        Recent News and Trends (from Web):
        {web_trends_sample}
        
        Please provide:
        1. Short-term forecast (next 1-2 years)
        2. Medium-term forecast (3-5 years)
        3. Key indicators to monitor
        4. Potential intervention points
        
        Format your response with clear headings and bullet points.
        """
        
        # Generate forecast
        forecast = llmselection.get_response(llm, forecast_prompt)
        
        # Create a forecast visualization if possible
        forecast_viz = generate_forecast_visualization(
            yearly_trends,
            state["selected_regions"][0],  
            f"forecast_crime.png"
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
        {rag_data[:2000]}
        
        Statistical Analysis:
        {snowflake_data[:1000]}
        
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
    """Organize all data into structured report sections and generate missing content."""
    try:
        print("\n📝 Organizing report sections and synthesizing content...")
        
        # Define the report structure with empty sections
        report_sections = {
            "executive_summary": {
                "title": "Executive Summary",
                "content": "",
                "order": 1,
                "images": []
            },
            "methodology": {
                "title": "Methodology and Data Sources",
                "content": "",
                "order": 2,
                "images": []
            },
            "historical_context": {
                "title": "Historical Context and Trends",
                "content": "",
                "order": 3,
                "visualizations": [],
                "images": []
            },
            "current_analysis": {
                "title": "Current Crime Landscape Analysis",
                "content": "",
                "order": 4,
                "visualizations": [],
                "images": []
            },
            "regional_comparison": {
                "title": "Regional Comparison Analysis",
                "content": "",
                "order": 5,
                "visualizations": [],
                "images": []
            },
            "safety_assessment": {
                "title": "Safety Assessment",
                "content": state.get("safety_assessment", ""),
                "order": 6,
                "images": []
            },
            "forecast": {
                "title": "Crime Trend Forecast",
                "content": state.get("forecast_output", {}).get("forecast", ""),
                "order": 7,
                "visualizations": [],
                "images": []
            },
            "recommendations": {
                "title": "Recommendations and Interventions",
                "content": "",
                "order": 8,
                "images": []
            },
            "appendix": {
                "title": "Appendix: Additional Data and Visualizations",
                "content": "",
                "order": 9,
                "visualizations": [],
                "images": [],
                "links": []
            }
        }
        
        # Begin populating sections with available content
        
        # Historical context from RAG output
        if "rag_output" in state:
            historical_insights = state["rag_output"].get("insights", "")
            report_sections["historical_context"]["content"] = historical_insights
            
            # Get any historical visualizations
            if "visualizations" in state["rag_output"]:
                rag_viz = state["rag_output"].get("visualizations", [])
                report_sections["historical_context"]["visualizations"] = rag_viz
        
        # Current analysis from Snowflake output
        if "snowflake_output" in state and state["snowflake_output"].get("status") == "success":
            analysis = state["snowflake_output"].get("analysis", "")
            report_sections["current_analysis"]["content"] = analysis
            
            # Add visualizations
            viz_paths = state["snowflake_output"].get("visualizations", {}).get("paths", {})
            for viz_type, path in viz_paths.items():
                report_sections["current_analysis"]["visualizations"].append(path)
        
        # Regional comparison from comparison output
        if "comparison_output" in state:
            comparison = state["comparison_output"].get("comparison", "")
            report_sections["regional_comparison"]["content"] = comparison
            
            # Add comparison visualizations if any
            comp_viz = state["comparison_output"].get("visualizations", [])
            report_sections["regional_comparison"]["visualizations"] = comp_viz
        
        # Safety assessment from safety output
        if "safety_assessment" in state:
            report_sections["safety_assessment"]["content"] = state["safety_assessment"]
        
        # Forecast from forecast output
        if "forecast_output" in state and state["forecast_output"].get("status") == "success":
            forecast = state["forecast_output"].get("forecast", "")
            report_sections["forecast"]["content"] = forecast
            
            # Add forecast visualization
            forecast_viz = state["forecast_output"].get("visualization")
            if forecast_viz:
                report_sections["forecast"]["visualizations"] = [forecast_viz]
        
        # Methodology - standard content based on what was used
        methodology_content = [
            f"This report analyzes crime data for {', '.join(state['selected_regions'])} using multiple data sources:",
            "- Historical crime records through Retrieval Augmented Generation",
            "- Statistical analysis using Snowflake database",
            "- Latest news articles and reports from web searches",
            "- Comparative analysis across regions and time periods",
            "- AI-driven forecasting and trend analysis",
            "",
            f"The analysis covers {state.get('start_year', 'all available history')} to {state.get('end_year', 'present')}."
        ]
        report_sections["methodology"]["content"] = "\n".join(methodology_content)
        
        # ========== SYNTHESIS: GENERATE MISSING CONTENT ==========
        # Get the LLM for synthesis of missing content
        llm = llmselection.get_llm(state["model_type"])
        
        # Generate missing content for empty sections
        for section_key, section in report_sections.items():
            if not section.get("content"):
                print(f"Generating content for {section.get('title')}")
                
                if section_key == "executive_summary":
                    # Create an executive summary based on all available data
                    summary_prompt = f"""
                    Create a concise executive summary (max 250 words) of the crime analysis report for {', '.join(state['selected_regions'])}. 
                    Include key findings about crime rates, patterns, and notable trends.
                    Focus on the most important insights that a decision-maker would need to know.
                    """
                    section["content"] = llmselection.get_response(llm, summary_prompt)
                
                elif section_key == "recommendations":
                    # Generate recommendations based on all analyses
                    safety_assessment = state.get("safety_assessment", "")
                    forecast = state.get("forecast_output", {}).get("forecast", "")
                    
                    recommendations_prompt = f"""
                    Based on the crime data and analysis for {', '.join(state['selected_regions'])}, 
                    provide specific, actionable recommendations for:
                    1. Law enforcement strategies
                    2. Community safety measures
                    3. Policy interventions
                    
                    Safety Assessment: {safety_assessment[:500]}...
                    
                    Forecast Insights: {forecast[:500]}...
                    
                    Format with clear bullet points and prioritize by potential impact.
                    """
                    section["content"] = llmselection.get_response(llm, recommendations_prompt)
                
                elif section_key == "appendix":
                    # Generate appendix content
                    appendix_prompt = f"""
                    Create a brief appendix section for a crime report including:
                    1. Data sources and methodologies
                    2. Statistical methods used
                    3. Glossary of crime-related terms
                    4. References
                    """
                    section["content"] = llmselection.get_response(llm, appendix_prompt)
        
        # Collect all visualizations for reference
        visualizations = {}
        
        # Add Snowflake visualizations
        if "snowflake_output" in state and state["snowflake_output"].get("status") == "success":
            viz_paths = state["snowflake_output"].get("visualizations", {}).get("paths", {})
            for viz_type, path in viz_paths.items():
                visualizations[f"snowflake_{viz_type}"] = path
        
        # Add Forecast visualization
        if "forecast_output" in state and state["forecast_output"].get("status") == "success":
            forecast_viz = state["forecast_output"].get("visualization")
            if forecast_viz:
                visualizations["forecast"] = forecast_viz
        
        # Add contextual images to appropriate sections
        if "contextual_images" in state and state["contextual_images"]:
            # Match contextual images to sections by keywords
            for title, image_data in state["contextual_images"].items():
                if "prevention" in title.lower():
                    report_sections["recommendations"]["images"].append(image_data)
                elif "safety" in title.lower():
                    report_sections["safety_assessment"]["images"].append(image_data)
                elif "policing" in title.lower():
                    report_sections["current_analysis"]["images"].append(image_data)
                else:
                    # Default to executive summary
                    report_sections["executive_summary"]["images"].append(image_data)
        
        # Add web search images to appendix
        if "web_output" in state:
            # Add images
            images = state["web_output"].get("images", [])
            for i, img_url in enumerate(images):
                visualizations[f"web_image_{i}"] = img_url
                report_sections["appendix"]["visualizations"].append(img_url)
            
            # Add links to the appendix
            links = state["web_output"].get("links", [])
            if links and isinstance(links, list):
                # Format links for appendix
                link_content = ["### Reference Links", ""]
                for i, link_data in enumerate(links, start=1):
                    if isinstance(link_data, dict):
                        title = link_data.get("title", "Untitled")
                        url = link_data.get("url", "#")
                        source = link_data.get("source", "Unknown")
                        pub_date = link_data.get("published_date", "Unknown")
                        link_content.append(f"{i}. [{title}]({url}) - {source} ({pub_date})")
                    elif isinstance(link_data, str):
                        link_content.append(f"{i}. [{link_data}]({link_data})")
                
                # Store the formatted links in the appendix section
                report_sections["appendix"]["links"] = links
                
                # Add link details to appendix content
                if report_sections["appendix"]["content"]:
                    report_sections["appendix"]["content"] += "\n\n" + "\n".join(link_content)
                else:
                    report_sections["appendix"]["content"] = "\n".join(link_content)
        
        print(f"✅ Report organization and content synthesis complete with {len(visualizations)} visualizations")
        return {
            "report_sections": report_sections,
            "visualizations": visualizations,
            "synthesis_complete": True  # Keep this flag for pipeline flow control
        }
        
    except Exception as e:
        print(f"❌ Report organization error: {str(e)}")
        traceback.print_exc()
        return {"report_sections": {}, "visualizations": {}}
        

# def synthesis_node(state: CrimeReportState) -> Dict:
#     """
#     Synthesize findings from all sources and populate empty report sections.
#     """
#     try:
#         print("\n🔄 Synthesizing information across all sources...")
        
#         # Get existing report sections and visualizations
#         report_sections = state.get("report_sections", {})
#         visualizations = state.get("visualizations", {})
        
#         # If report_sections is empty, something went wrong
#         if not report_sections:
#             print("⚠️ No report sections available - creating default sections")
#             org_result = report_organization_node(state)
#             report_sections = org_result.get("report_sections", {})
#             visualizations = org_result.get("visualizations", {})
        
#         # Get the LLM for synthesis
#         llm = llmselection.get_llm(state["model_type"])
        
#         # Only generate content for empty sections
#         for section_key, section in report_sections.items():
#             if not section.get("content"):
#                 print(f"Generating content for {section.get('title')}")
                
#                 if section_key == "executive_summary":
#                     # Create an executive summary based on all available data
#                     summary_prompt = f"""
#                     Create a concise executive summary (max 250 words) of the crime analysis report for {', '.join(state['selected_regions'])}. 
#                     Include key findings about crime rates, patterns, and notable trends.
#                     Focus on the most important insights that a decision-maker would need to know.
#                     """
#                     section["content"] = llmselection.get_response(llm, summary_prompt)
                
#                 elif section_key == "recommendations":
#                     # Generate recommendations based on all analyses
#                     safety_assessment = state.get("safety_assessment", "")
#                     forecast = state.get("forecast_output", {}).get("forecast", "")
                    
#                     recommendations_prompt = f"""
#                     Based on the crime data and analysis for {', '.join(state['selected_regions'])}, 
#                     provide specific, actionable recommendations for:
#                     1. Law enforcement strategies
#                     2. Community safety measures
#                     3. Policy interventions
                    
#                     Safety Assessment: {safety_assessment[:500]}...
                    
#                     Forecast Insights: {forecast[:500]}...
                    
#                     Format with clear bullet points and prioritize by potential impact.
#                     """
#                     section["content"] = llmselection.get_response(llm, recommendations_prompt)
                
#                 elif section_key == "appendix":
#                     # Generate appendix content
#                     appendix_prompt = f"""
#                     Create a brief appendix section for a crime report including:
#                     1. Data sources and methodologies
#                     2. Statistical methods used
#                     3. Glossary of crime-related terms
#                     4. References
#                     """
#                     section["content"] = llmselection.get_response(llm, appendix_prompt)
        
#         print("✅ Content synthesis complete")
#         return {
#             "report_sections": report_sections,
#             "visualizations": visualizations,
#             "synthesis_complete": True
#         }
        
#     except Exception as e:
#         print(f"❌ Synthesis error: {str(e)}")
#         traceback.print_exc()
#         return {"synthesis_error": str(e)}
        

def final_report_node(state: CrimeReportState) -> Dict:
    """Assemble the final report with all sections, visualizations and contextual images."""
    try:
        print("\n📊 Generating final crime report...")
        
        # Get report sections and visualizations
        report_sections = state.get("report_sections", {})
        visualizations = state.get("visualizations", {})
        contextual_images = state.get("contextual_images", {})
        
        # Log for debugging
        print(f"Found {len(report_sections)} report sections")
        section_titles = [section.get("title") for section in report_sections.values()]
        print(f"Section titles: {section_titles}")
        print(f"Found {len(visualizations)} visualizations")
        print(f"Found {len(contextual_images)} contextual images")
        
        # Ensure safety assessment is included
        if "safety_assessment" in state and "safety_assessment" not in report_sections:
            print("Adding safety assessment to report sections")
            report_sections["safety_assessment"] = {
                "title": "Safety Assessment",
                "content": state["safety_assessment"],
                "order": 6
            }
        
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
            "visualizations": list(visualizations.values()) if visualizations else [],
            "contextual_images": list(contextual_images.values()) if contextual_images else [],
            "cover_image": None,  # Will be set later if generated
            "metadata": {
                "source_count": len(visualizations),
                "word_count": sum(len(section.get("content", "").split()) 
                               for section in report_sections.values() 
                               if isinstance(section.get("content", ""), str)),
                "section_count": len(report_sections)
            }
        }
        
        # Add sections in correct order
        ordered_sections = sorted(
            [section for section in report_sections.values()],
            key=lambda x: x.get("order", 999)
        )
        
        # Generate a sample cover image with title
        cover_image_path = generate_report_cover(
            title=final_report["title"],
            regions=state["selected_regions"],
            time_period=final_report["parameters"]["time_period"]
        )
        
        if cover_image_path:
            final_report["cover_image"] = cover_image_path
            print(f"✅ Cover image generated: {cover_image_path}")
        
        final_report["sections"] = ordered_sections
        
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
            judge_node.agent = JudgeAgent(model_type=state["model_type"])
        
        # Create evaluation context
        evaluation_context = {
            'report': state["final_report"],
            'regions': state["selected_regions"],
            'time_period': f"{state.get('start_year', 'all history')} to {state.get('end_year', 'present')}"
        }
        
        # Execute evaluation
        evaluation = judge_node.agent.evaluate(evaluation_context)
        
        # Make sure feedback_history exists
        if not hasattr(judge_node.agent, 'feedback_history'):
            judge_node.agent.feedback_history = []
        
        print(f"✅ Report evaluation complete - Overall score: {evaluation.get('overall_score', 'N/A')}/10")
        print(f"📝 Stored {len(judge_node.agent.feedback_history)} previous evaluations")
        
        evaluation_data = {
            "judge_feedback": evaluation,
            "quality_scores": evaluation.get("scores", {}),
            "improvement_suggestions": evaluation.get("improvement_suggestions", []),
            "feedback_history": judge_node.agent.feedback_history[-5:]  # Last 5 evaluations
        }
        
        # Include evaluation in the final report for storage
        if "final_report" in state:
            state["final_report"]["evaluation"] = evaluation_data
        
        return evaluation_data
        
    except Exception as e:
        print(f"❌ Report evaluation error: {str(e)}")
        traceback.print_exc()
        return {
            "judge_feedback": {"error": str(e)},
            "quality_scores": {"overall": 5}
        }
        if "final_report" in state:
            state["final_report"]["evaluation"] = error_data
            
        return error_data
    
###############################################################################
# Helper Functions for Visualization
###############################################################################
def generate_forecast_visualization(yearly_trends, region: str, output_path: str) -> str:
    """Generate a forecast visualization extending the current trends."""
    try:
        print(f"Generating forecast visualization for {region}...")
        
        # Extract relevant data
        if not yearly_trends or not isinstance(yearly_trends, dict):
            print("No valid yearly trends data for forecast visualization")
            return ""
            
        # Get years and values
        years = []
        values = []
        
        # Format may vary, try to extract data safely
        if isinstance(yearly_trends, dict):
            for year_str, data in yearly_trends.items():
                try:
                    # Only process if year is numeric
                    year = int(year_str)
                    
                    # Extract total crime count for the year
                    if isinstance(data, dict):
                        total = 0
                        for incident_type, count in data.items():
                            # Skip non-numeric incident types and "all incidents"
                            if isinstance(incident_type, str) and incident_type.strip().lower() == 'all incidents':
                                continue
                                
                            try:
                                # Try to convert to number
                                if isinstance(count, (int, float)):
                                    total += count
                                else:
                                    total += float(count)
                            except (ValueError, TypeError):
                                # Skip if conversion fails
                                continue
                        
                        if total > 0:  # Only add if we have valid data
                            years.append(year)
                            values.append(total)
                    elif isinstance(data, (int, float)):
                        years.append(year)
                        values.append(data)
                        
                except (ValueError, TypeError):
                    # Skip non-numeric years
                    continue
        
        if len(years) < 2:
            print("Not enough data points for forecast (need at least 2 years)")
            return ""
        
        # Sort data by year
        sorted_data = sorted(zip(years, values))
        years = [y for y, v in sorted_data]
        values = [v for y, v in sorted_data]
        
        print(f"Processed data: {len(years)} years with valid data")
        
        # Create the forecast (simple linear regression)
        # Convert to numpy arrays
        x = np.array(years, dtype=float)
        y = np.array(values, dtype=float)
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Generate future years (5 years into the future)
        future_years = list(range(max(years) + 1, max(years) + 6))
        
        # Predict future values
        future_values = [max(0, slope * year + intercept) for year in future_years]
        
        # Plot the data with consistent color theme
        plt.figure(figsize=(12, 6))
        
        # Use a consistent color palette based on the region
        region_colors = {
            'Chicago': '#1f77b4',  # blue
            'New York': '#ff7f0e',  # orange
            'Los Angeles': '#2ca02c',  # green
            'default': '#d62728'  # red
        }
        
        hist_color = region_colors.get(region, region_colors['default'])
        
        # Historical data
        plt.plot(years, values, marker='o', color=hist_color, linewidth=2, 
                 label=f'Historical Data ({region})')
        
        # Forecast
        plt.plot(future_years, future_values, 'r--o', linewidth=2, label='Forecast')
        
        # Add confidence interval
        plt.fill_between(
            future_years, 
            [max(0, val * 0.8) for val in future_values],  # Lower bound (80% of prediction)
            [val * 1.2 for val in future_values],  # Upper bound (120% of prediction)
            color='red', alpha=0.2, label='Forecast Range (±20%)'
        )
        
        # Improve styling
        plt.title(f'Crime Trend Forecast for {region}', fontsize=16, pad=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Incidents', fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add data labels to historical points
        for i, (x_val, y_val) in enumerate(zip(years, values)):
            plt.annotate(f'{int(y_val):,}', 
                        xy=(x_val, y_val),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9)
        
        # Add data labels to forecast points
        for i, (x_val, y_val) in enumerate(zip(future_years, future_values)):
            plt.annotate(f'{int(y_val):,}', 
                        xy=(x_val, y_val),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=9)
        
        # Add a watermark/footer with generation details
        plt.figtext(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", 
                   fontsize=8, ha='right', color='gray', alpha=0.7)
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Forecast visualization saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating forecast visualization: {str(e)}")
        traceback.print_exc()
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
        output_path = f"report_cover.png"
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
    if not hasattr(build_pipeline, 'llm_cache'):
        build_pipeline.llm_cache = {}
    
    try:
        graph = StateGraph(CrimeReportState)
        
        # def get_cached_llm(state):
        #     """Get a cached LLM instance or create a new one."""
        #     model_type = state.get("model_type")
        #     if model_type not in build_pipeline.llm_cache:
        #         build_pipeline.llm_cache[model_type] = llmselection.get_llm(model_type)
        #     return build_pipeline.llm_cache[model_type]
        
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
        graph.add_node("comparison", comparison_node)
        graph.add_node("forecast", forecast_node)
        graph.add_node("safety", safety_assessment_node)
        graph.add_node("contextual_img", contextual_image_node)
        graph.add_node("organization", report_organization_node)  # Now includes synthesis
        graph.add_node("report_generation", final_report_node)
        graph.add_node("judge", judge_node)
        
        # Set entry point
        graph.set_entry_point("start")

        # Define edge conditions and add edges
        print("🔀 Configuring streamlined pipeline flow...")
        
        # Start -> Parallel Gathering
        graph.add_edge("start", "parallel_gathering")

        # Parallel Gathering -> Contextual Images
        graph.add_conditional_edges(
            "parallel_gathering",
            lambda x: all(k in x for k in ["snowflake_output", "rag_output", "web_output"]),
            {True: "contextual_img", False: "contextual_img"}  # Continue even if some data is missing
        )

        # Contextual Images -> Comparison
        graph.add_edge("contextual_img", "comparison")

        # Comparison -> Forecast
        graph.add_edge("comparison", "forecast")

        # Forecast -> Safety Assessment
        graph.add_edge("forecast", "safety")

        # Safety Assessment -> Organization (which now includes synthesis)
        graph.add_edge("safety", "organization")

        # Organization -> Report Generation (directly)
        graph.add_conditional_edges(
            "organization",
            lambda x: x.get("synthesis_complete", False),
            {True: "report_generation"}
        )

        # Report Generation -> Judge
        graph.add_conditional_edges(
            "report_generation",
            lambda x: "final_report" in x,
            {True: "judge"}
        )

        # Judge -> END
        graph.add_edge("judge", END)

        print("✅ Streamlined pipeline build complete")
        return graph.compile()
    except Exception as e:
        print(f"❌ Error building pipeline: {str(e)}")
        traceback.print_exc()
        return None

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
        output_filename = f"crime_report.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n💾 Full report saved to: {output_filename}")
        
        # Save markdown version
        md_filename = output_filename.replace(".json", ".md")
        with open(md_filename, "w", encoding="utf-8") as f:
            # Add cover image if available
            if final_report.get("cover_image"):
                f.write(f"![Cover](./{final_report.get('cover_image')})\n\n")
            
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
                
                # Add images if any (from contextual images)
                if "images" in section and section["images"]:
                    f.write("### Illustrative Images\n\n")
                    for img in section["images"]:
                        if isinstance(img, dict) and "path" in img:
                            img_path = img["path"]
                            img_desc = img.get("description", "Contextual image")
                            if img_path.startswith("http"):
                                f.write(f"![{img_desc}]({img_path})\n\n")
                            else:
                                f.write(f"![{img_desc}](./{img_path})\n\n")
                
            # Add any contextual images that weren't assigned to sections
            if final_report.get("contextual_images"):
                f.write("## Additional Contextual Images\n\n")
                for img in final_report.get("contextual_images", []):
                    if isinstance(img, dict) and "path" in img:
                        img_path = img["path"]
                        img_desc = img.get("description", "Contextual image")
                        f.write(f"![{img_desc}](./{img_path})\n\n")
        
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