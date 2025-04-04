import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any, List, Tuple
import warnings
from io import BytesIO
import base64
import json
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.llmselection import LLMSelector as llmselection

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Constants and Configuration
# ---------------------------------------------------------------------------
SYSTEM_CONFIG = {
    "CURRENT_UTC": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    "CURRENT_USER": os.getenv("USER_NAME", "user"),
    "MIN_YEAR": 2010,
    "MAX_YEAR": 2024,
    "DEFAULT_CITIES": ["Chicago", "New York", "Los Angeles"],
    "CHART_COLORS": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    "DATE_FORMAT": "%Y-%m-%d",
    "DEFAULT_DPI": 150,
    "FIGURE_SIZE": (12, 6)
}

# Load environment variables
load_dotenv(override=True)

class ComparisonAgent:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.llm = llmselection.get_llm(model_type)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="comparison_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = [
            Tool(
                name="analyze_trends",
                func=self._analyze_crime_trends,
                description="Analyze crime trends across regions"
            ),
            Tool(
                name="compare_statistics",
                func=self._compare_regions,
                description="Compare statistical data between regions"
            )
        ]
        
        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
        
        self.feedback_history = []

    def _analyze_crime_trends(self, data: dict) -> str:
        """Tool for analyzing crime trends in the data."""
        try:
            stats = data.get("statistics", {})
            trends = stats.get("incident_analysis", {}).get("yearly_trends", {})
            
            return json.dumps({
                "trend_analysis": trends,
                "key_findings": self._extract_key_trends(trends)
            })
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"

    def _compare_regions(self, regions: list, data: dict) -> str:
        """Tool for comparing statistics between regions."""
        try:
            stats = data.get("statistics", {})
            return json.dumps({
                "regional_comparison": stats.get("regional_analysis", {}),
                "key_differences": self._extract_regional_differences(stats, regions)
            })
        except Exception as e:
            return f"Error comparing regions: {str(e)}"

    def _analyze_temporal(self, region: str,
                         snowflake_data: Dict,
                         rag_data: Dict) -> str:
        """Analyze temporal patterns for a single region."""
        prompt = f"""
        Analyze temporal crime patterns for {region}.
        
        Statistical Data:
        {json.dumps(snowflake_data.get('statistics', {}), indent=2)[:1000]}...
        
        Historical Context:
        {rag_data.get('insights', 'No historical context available')[:500]}...
        
        Provide:
        1. Year-over-year changes
        2. Seasonal patterns
        3. Long-term trends
        4. Key turning points
        5. Future projections
        """
        
        return self.agent.run(prompt)

    def store_feedback(self, analysis_id: str, feedback: dict) -> None:
        """Store feedback for improving future analyses."""
        feedback_entry = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "model_type": self.model_type
        }
        self.feedback_history.append(feedback_entry)
        
        # Add feedback to conversation memory
        self.memory.save_context(
            {"input": "Analysis Feedback"},
            {"output": f"Feedback received: {json.dumps(feedback)}"}
        )

        # Update analyze method in ComparisonAgent class
    def analyze(self, analysis_request, comparison_type="cross_region"):
        """
        Analyze crime data and create comparative insights.
        
        Args:
            analysis_request (dict): Request containing data sources
            comparison_type (str): Type of comparison to perform
            
        Returns:
            dict: Comparison results
        """
        try:
            # Extract data from request
            regions = analysis_request.get("regions", [])
            snowflake_data = analysis_request.get("snowflake_data", {})
            rag_data = analysis_request.get("rag_data", {})
            web_data = analysis_request.get("web_data", {})
            
            # If there's an error in any data source, provide a placeholder comparison
            if (snowflake_data.get("status") == "failed" or 
                "error" in rag_data or 
                "error" in snowflake_data):
                # Create a basic fallback comparison using whatever data is available
                return self._generate_fallback_comparison(regions)
                
            # Generate comparison based on available data
            comparison = self._generate_comparison(regions, snowflake_data, rag_data, web_data, comparison_type)
            return comparison
            
        except Exception as e:
            print(f"Comparison agent error: {str(e)}")
            # Return a minimal valid result structure
            return {
                "comparison": f"Unable to generate comparison due to error: {str(e)}. Please check data sources.",
                "visualizations": [],
                "status": "error"
            }
    def _generate_comparison(self, regions, snowflake_data, rag_data, web_data, comparison_type):
        """Generate a comprehensive comparison between regions or time periods."""
        try:
            # Create a prompt for the comparison based on available data
            regions_str = ", ".join(regions)
            
            # Extract key data points
            snowflake_analysis = snowflake_data.get("analysis", "No Snowflake analysis available") 
            if len(snowflake_analysis) > 1000:  # Trim if too long
                snowflake_analysis = snowflake_analysis[:1000] + "..."
                
            rag_insights = rag_data.get("insights", "No historical insights available")
            if len(rag_insights) > 1000:
                rag_insights = rag_insights[:1000] + "..."
                
            web_report = web_data.get("markdown_report", "No web search data available")
            if len(web_report) > 1000:
                web_report = web_report[:1000] + "..."
            
            # Determine comparison type and construct prompt
            if comparison_type == "cross_region" and len(regions) > 1:
                prompt = f"""
                Generate a comprehensive comparison of crime patterns between {regions_str}.
                
                Data from statistical analysis:
                {snowflake_analysis}
                
                Historical insights:
                {rag_insights}
                
                Web search findings:
                {web_report}
                
                Please provide:
                1. Direct comparison of crime rates and types between regions
                2. Regional differences in crime trends
                3. Socioeconomic or demographic factors that might explain differences
                4. Policing and prevention strategies that differ between regions
                5. Which region appears safer overall and why
                
                Format your response with clear headings and use markdown formatting.
                """
            else:
                # Temporal comparison for a single region
                prompt = f"""
                Generate a comprehensive analysis of crime trends over time for {regions_str}.
                
                Data from statistical analysis:
                {snowflake_analysis}
                
                Historical insights:
                {rag_insights}
                
                Web search findings:
                {web_report}
                
                Please provide:
                1. How crime rates have changed over time
                2. Key turning points or significant changes in crime patterns
                3. Factors that might explain these changes
                4. Whether the overall crime situation is improving or worsening
                5. Predictions for future trends based on historical patterns
                
                Format your response with clear headings and use markdown formatting.
                """
            
            # Get the LLM response
            llm = llmselection.get_llm(self.model_type)
            comparison_content = llmselection.get_response(llm, prompt)
            
            # Return the comparison data
            return {
                "comparison": comparison_content,
                "visualizations": [],
                "status": "success"
            }
        
        except Exception as e:
            print(f"Error generating comparison: {str(e)}")
            return {
                "comparison": f"Error generating comparison: {str(e)}",
                "visualizations": [],
                "status": "error"
            }
        
        def _generate_fallback_comparison(self, regions):
            """Generate a fallback comparison when data is insufficient."""
            region_str = ", ".join(regions)
            
            # Generate a basic comparison that acknowledges the data limitations
            basic_comparison = f"""
            # Regional Comparison Analysis for {region_str}
            
            **Note: This analysis is limited due to data availability issues.**
            
            ## Available Information:
            
            Based on the limited data available, crime patterns across {region_str} show some variations:
            
            - Each region has unique demographic and socioeconomic factors affecting crime rates
            - Urban density, economic conditions, and policing approaches differ between regions
            - Historical crime trends suggest regional differences in types and frequencies of incidents
            
            More comprehensive analysis would require additional data sources.
            """
            
            return {
                "comparison": basic_comparison,
                "visualizations": [],
                "status": "limited"
            }


    def _analyze_cross_region(self, regions: List[str], 
                            snowflake_data: Dict, 
                            rag_data: Dict) -> str:
        """Analyze cross-region comparisons."""
        prompt = f"""
        Compare crime patterns between {', '.join(regions)}.
        
        Statistical Data:
        {json.dumps(snowflake_data.get('statistics', {}), indent=2)[:1000]}...
        
        Historical Context:
        {rag_data.get('insights', 'No historical context available')[:500]}...
        
        Provide:
        1. Overall crime rate comparison
        2. Specific crime type differences
        3. Temporal trends comparison
        4. Key factors explaining differences
        5. Best practices recommendations
        """
        
        return self.agent.run(prompt)

    def _structure_analysis(self, response: str, comparison_type: str) -> dict:
        """Structure the agent's response into a formatted analysis."""
        try:
            # Parse the response into sections
            sections = response.split('\n\n')
            
            return {
                "analysis": response,
                "insights": self._extract_insights(sections),
                "recommendations": self._extract_recommendations(sections),
                "metadata": {
                    "comparison_type": comparison_type,
                    "generated_at": datetime.now().isoformat(),
                    "model_used": self.model_type
                },
                "status": "success"
            }
        except Exception as e:
            return {
                "error": f"Error structuring analysis: {str(e)}",
                "status": "failed"
            }
    def _extract_key_trends(self, trends: Dict) -> List[str]:
        """Extract key trends from the data."""
        key_findings = []
        try:
            if not trends:
                return ["No trend data available"]
                
            # Analyze year-over-year changes
            years = sorted(list(trends.keys()))
            for i in range(1, len(years)):
                prev_year = years[i-1]
                curr_year = years[i]
                change = trends[curr_year] - trends[prev_year]
                pct_change = (change / trends[prev_year]) * 100
                key_findings.append(
                    f"{curr_year}: {'Increase' if change > 0 else 'Decrease'} of {abs(pct_change):.1f}%"
                )
                
            return key_findings
        except Exception as e:
            return [f"Error extracting trends: {str(e)}"]

    def _extract_regional_differences(self, stats: Dict, regions: List[str]) -> List[str]:
        """Extract key differences between regions."""
        differences = []
        try:
            if not stats or not regions:
                return ["No regional data available"]
                
            regional_data = stats.get("regional_analysis", {})
            for region in regions:
                region_stats = regional_data.get(region, {})
                differences.append(f"{region}:")
                differences.extend([
                    f"- Total incidents: {region_stats.get('total_incidents', 'N/A')}",
                    f"- Top crime: {region_stats.get('top_crime', 'N/A')}",
                    f"- YoY change: {region_stats.get('yoy_change', 'N/A')}%"
                ])
                
            return differences
        except Exception as e:
            return [f"Error extracting regional differences: {str(e)}"]

    def _extract_insights(self, sections: List[str]) -> List[str]:
        """Extract key insights from analysis sections."""
        insights = []
        try:
            for section in sections:
                if ":" in section:
                    title, content = section.split(":", 1)
                    if any(keyword in title.lower() for keyword in ["finding", "insight", "trend", "pattern"]):
                        insights.extend([line.strip() for line in content.split("\n") if line.strip()])
            return insights if insights else ["No specific insights found"]
        except Exception as e:
            return [f"Error extracting insights: {str(e)}"]

    def _extract_recommendations(self, sections: List[str]) -> List[str]:
        """Extract recommendations from analysis sections."""
        recommendations = []
        try:
            for section in sections:
                if ":" in section:
                    title, content = section.split(":", 1)
                    if any(keyword in title.lower() for keyword in ["recommend", "suggest", "action", "improve"]):
                        recommendations.extend([line.strip() for line in content.split("\n") if line.strip()])
            return recommendations if recommendations else ["No specific recommendations found"]
        except Exception as e:
            return [f"Error extracting recommendations: {str(e)}"]