import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
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
    """Agent for comparing crime statistics across different cities and time periods."""
    
    def __init__(self, model_type=None):
        """Initialize the comparison agent with database connection and LLM."""
        self.engine = create_engine(
            f"snowflake://{os.environ.get('SNOWFLAKE_USER')}:{os.environ.get('SNOWFLAKE_PASSWORD')}"
            f"@{os.environ.get('SNOWFLAKE_ACCOUNT')}/{os.environ.get('SNOWFLAKE_DATABASE')}/"
            f"{os.environ.get('SNOWFLAKE_SCHEMA')}?warehouse={os.environ.get('SNOWFLAKE_WAREHOUSE')}"
        )
        self.llm = llmselection.get_llm(model_type or "Claude 3 Haiku")
        self.initialize_agent()

    def initialize_agent(self):
        """Initialize LangChain agent with tools."""
        comparison_tool = Tool(
            name="compare_crime_stats",
            description="Compare crime statistics between cities and time periods",
            func=self.compare_crime_stats
        )
        
        self.agent = initialize_agent(
            tools=[comparison_tool],
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=2,
            early_stopping_method="generate"
        )

    def compare_crime_stats(self, cities: list, start_year: int = None, end_year: int = None) -> dict:
        """Compare crime statistics between specified cities."""
        try:
            # Build query
            query = """
            SELECT 
                EXTRACT(YEAR FROM date) as year,
                city,
                incident,
                SUM(value) as incident_count
            FROM CLEAN_CRIME_DATASET
            WHERE UPPER(city) IN ({})
            {}
            GROUP BY year, city, incident
            ORDER BY year, city, incident
            """.format(
                ','.join([f"UPPER('{city}')" for city in cities]),
                f"AND EXTRACT(YEAR FROM date) BETWEEN {start_year} AND {end_year}" 
                if start_year and end_year else ""
            )
            
            # Execute query
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                raise ValueError("No data found for the specified cities and time period.")
            
            # Generate comparison visualizations
            viz_data = self.generate_comparison_visualizations(df)
            
            # Get LLM analysis
            analysis = self.get_comparison_analysis(df)
            
            return {
                "visualizations": viz_data,
                "analysis": analysis,
                "status": "success"
            }
            
        except Exception as e:
            print(f"Error in compare_crime_stats: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }

    def generate_comparison_visualizations(self, df: pd.DataFrame) -> dict:
        """Generate comparison visualizations."""
        viz_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. Total Incidents by City Over Time
            plt.figure(figsize=SYSTEM_CONFIG["FIGURE_SIZE"])
            for city in df['city'].unique():
                city_data = df[df['city'] == city].groupby('year')['incident_count'].sum()
                plt.plot(city_data.index, city_data.values, marker='o', label=city)
            
            plt.title('Total Crime Incidents by City Over Time', fontsize=14)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Number of Incidents', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            total_incidents_path = f"total_incidents_comparison_{timestamp}.png"
            plt.savefig(total_incidents_path, dpi=SYSTEM_CONFIG["DEFAULT_DPI"], bbox_inches='tight')
            plt.close()
            viz_paths['total_incidents'] = total_incidents_path

            # 2. Incident Type Distribution by City
            plt.figure(figsize=SYSTEM_CONFIG["FIGURE_SIZE"])
            for i, city in enumerate(df['city'].unique()):
                city_data = df[df['city'] == city].groupby('incident')['incident_count'].sum()
                plt.bar([x + i*0.25 for x in range(len(city_data))], 
                       city_data.values, 
                       width=0.25, 
                       label=city)
            
            plt.title('Incident Type Distribution by City', fontsize=14)
            plt.xlabel('Incident Type', fontsize=12)
            plt.ylabel('Total Number of Incidents', fontsize=12)
            plt.xticks(range(len(df['incident'].unique())), 
                      df['incident'].unique(), 
                      rotation=45, 
                      ha='right')
            plt.legend()
            plt.tight_layout()
            
            distribution_path = f"incident_distribution_comparison_{timestamp}.png"
            plt.savefig(distribution_path, dpi=SYSTEM_CONFIG["DEFAULT_DPI"], bbox_inches='tight')
            plt.close()
            viz_paths['incident_distribution'] = distribution_path

            return viz_paths
            
        except Exception as e:
            print(f"Error generating comparison visualizations: {str(e)}")
            return {}

    def get_comparison_analysis(self, df: pd.DataFrame) -> str:
        """Get LLM analysis of comparison data."""
        try:
            # Prepare context for analysis
            cities = df['city'].unique()
            years_range = f"{df['year'].min()} to {df['year'].max()}"
            total_by_city = df.groupby('city')['incident_count'].sum()
            
            context = f"""
            Analyze crime statistics comparison between {', '.join(cities)}:
            
            Time Period: {years_range}
            
            Total Incidents by City:
            {total_by_city.to_string()}
            
            Top Incidents by City:
            {df.groupby(['city', 'incident'])['incident_count'].sum().reset_index().sort_values(['city', 'incident_count'], ascending=[True, False]).to_string()}
            
            Please provide a comprehensive comparative analysis including:
            1. Overall trends and patterns for each city
            2. Notable differences between cities
            3. Common patterns or correlations
            4. Key insights and recommendations
            5. Areas requiring attention
            """
            
            # Get analysis from LLM
            response = self.llm.invoke(context)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f"Error getting comparison analysis: {str(e)}")
            return f"Error generating analysis: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize agent
    agent = ComparisonAgent(model_type="Claude 3 Haiku")
    
    # Test comparison
    result = agent.compare_crime_stats(
        cities=["Chicago", "New York"],
        start_year=2020,
        end_year=2024
    )
    
    if result["status"] == "success":
        print("\nComparison Analysis:")
        print("===================")
        print(result["analysis"])
        print("\nVisualization files:")
        for viz_type, path in result["visualizations"].items():
            print(f"- {viz_type}: {path}")
    else:
        print(f"Error: {result['error']}")