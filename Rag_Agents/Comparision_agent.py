# agents/snowflake_agent.py
import os
import pandas as pd
import numpy as np  # Add numpy import
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import warnings
from io import BytesIO
import base64
import time
from typing import Dict, Any

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv(override=True)

# Initialize Snowflake SQLAlchemy engine
engine = create_engine(
    f"snowflake://{os.environ.get('SNOWFLAKE_USER')}:{os.environ.get('SNOWFLAKE_PASSWORD')}@{os.environ.get('SNOWFLAKE_ACCOUNT')}/{os.environ.get('SNOWFLAKE_DATABASE')}/{os.environ.get('SNOWFLAKE_SCHEMA')}?warehouse={os.environ.get('SNOWFLAKE_WAREHOUSE')}"
)


def get_valuation_summary(query: str = None) -> dict:
    """Get top 5 elements from the cybersyn.urban_crime_timeseries table."""
    try:
        # Use base query to fetch data from cybersyn.urban_crime_timeseries
        df = pd.read_sql("SELECT * FROM cybersyn.urban_crime_timeseries ORDER BY DATE DESC LIMIT 5", engine)
        df.columns = df.columns.str.upper().str.strip()
        
        if df.empty:
            raise ValueError("No data returned from Snowflake. Ensure the table contains data.")
        
        # Save the DataFrame to a CSV file
        csv_file_path = "top_5_urban_crime_timeseries.csv"
        df.to_csv(csv_file_path, index=False)
        
        # Print the DataFrame
        print("Top 5 Elements from cybersyn.urban_crime_timeseries:")
        print(df)
        
        # Generate visualization
        plt.figure(figsize=(10, 6))
        for date in df["DATE"].unique():
            subset = df[df["DATE"] == date]
            plt.bar(subset.columns[1:], subset.iloc[0, 1:], label=str(date))
        
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.title("Urban Crime Timeseries Metrics")
        plt.xticks(rotation=45)
        plt.legend()
        
        # Convert plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        return {
            "chart": img_str,
            "summary": df.to_string(),
            "csv_file_path": csv_file_path,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

def get_graph_specs_from_llm(data: list) -> dict:
    """Get graph specifications from LLM based on the Snowflake data."""
    try:
        # Convert the data summary to a JSON-like string for LLM input
        prompt = f"""
        Based on the following urban crime timeseries data:
        {data}

        Generate a graph specification in this format:
        - Title: [Graph title]
        - Type: [line/bar/scatter]
        - X-axis: [label and settings]
        - Y-axis: [label and settings]
        - Grouping: [e.g., by CITY or VARIABLE_NAME]
        - Colors: [color scheme]
        - Additional elements: [grid, legend position, etc.]

        Focus on making the graph visually informative and easy to interpret.
        """
        
        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        if not response or not hasattr(response, "content"):
            raise ValueError("LLM did not return a valid response.")
        
        # Extract the content from the AIMessage object
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse the LLM response into a dictionary
        specs = {}
        for line in response_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                specs[key.strip()] = value.strip()
        
        return specs
    except Exception as e:
        print(f"Error getting graph specs from LLM: {str(e)}")
        return {}

def create_graph_from_llm_specs(data: pd.DataFrame, specs: dict) -> str:
    """Create a graph based on LLM specifications with grouping by CITY or VARIABLE_NAME."""
    try:
        if 'DATE' not in data.columns:
            raise ValueError("The 'DATE' column is missing from the data.")
        
        # Create a copy of the DataFrame for normalization
        df_normalized = data.copy()
        
        # Normalize all numeric columns except DATE, CITY, and VARIABLE_NAME
        for col in df_normalized.columns:
            if col not in ['DATE', 'CITY', 'VARIABLE_NAME']:
                # Convert column to numeric, coercing errors to NaN
                df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
                
                # Skip normalization if the column contains non-numeric data
                if df_normalized[col].isnull().all():
                    print(f"Skipping column '{col}' as it contains non-numeric data.")
                    continue
                
                # Perform normalization
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val - min_val != 0:  # Avoid division by zero
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        # Group by CITY or VARIABLE_NAME if specified
        group_by = specs.get("Grouping", None)
        if group_by and group_by in df_normalized.columns:
            grouped_data = df_normalized.groupby(group_by)
        else:
            grouped_data = [("", df_normalized)]
        
        # Create figure
        plt.figure(figsize=(15, 8))
        for group_name, group_df in grouped_data:
            x = pd.to_datetime(group_df["DATE"]).dt.strftime('%b %d, %Y')
            for col in group_df.select_dtypes(include=[np.number]).columns:
                plt.plot(
                    x,
                    group_df[col],
                    label=f"{group_name} - {col.replace('_', ' ').title()}",
                    marker="o"
                )
        
        # Apply formatting
        plt.title(specs.get("Title", "Urban Crime Timeseries Metrics"), fontsize=16)
        plt.xlabel(specs.get("X-axis", "Date"), fontsize=14)
        plt.ylabel(specs.get("Y-axis", "Value"), fontsize=14)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the graph
        chart_file_path = "llm_generated_graph.png"
        plt.savefig(chart_file_path, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        
        return chart_file_path
    except Exception as e:
        print(f"Error creating graph: {str(e)}")
        return None

def get_valuation_summary_with_llm_graph() -> dict:
    """Get urban crime timeseries metrics and generate a graph using LLM."""
    try:
        # Fetch data from Snowflake
        df = pd.read_sql("SELECT * FROM cybersyn.urban_crime_timeseries ORDER BY DATE DESC LIMIT 10", engine)
        
        # Normalize column names
        df.columns = df.columns.str.upper().str.strip()

        # Debugging
        print("DEBUG: Normalized DataFrame columns:", df.columns)
        print("DEBUG: First few rows of the DataFrame:\n", df.head())
        
        if df.empty:
            raise ValueError("No data returned from Snowflake. Ensure the table contains data.")
        
        # Ensure the DATE column is parsed as datetime
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            if df['DATE'].isnull().all():
                raise ValueError("The 'DATE' column could not be parsed as datetime.")
        else:
            raise ValueError("The 'DATE' column is missing from the data.")
        
        # Include CITY and VARIABLE_NAME in the data summary
        data_summary = df[['DATE', 'CITY', 'VARIABLE_NAME'] + [col for col in df.columns if col not in ['DATE', 'CITY', 'VARIABLE_NAME']]].to_dict(orient="records")
        
        # Get graph specifications from LLM
        graph_specs = get_graph_specs_from_llm(data_summary)
        if not graph_specs:
            print("LLM failed to generate graph specifications. Using default graph settings.")
            graph_specs = {
                "Title": "Urban Crime Timeseries Metrics",
                "Type": "line",
                "X-axis": "Date",
                "Y-axis": "Value",
                "Additional elements": "grid, legend"
            }
        
        # Create the graph based on LLM specifications
        chart_file_path = create_graph_from_llm_specs(df, graph_specs)
        if not chart_file_path:
            raise ValueError("Failed to create graph from LLM specifications.")
        
        # Return the summary and graph path
        return {
            "summary": data_summary,
            "chart_path": chart_file_path,
            "graph_specs": graph_specs,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in get_valuation_summary_with_llm_graph: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

def get_ai_analysis_with_graph(user_query: dict):
    """Get AI-generated analysis of urban crime metrics with LLM-generated graph."""
    try:
        # Query the data based on user specifications
        result = query_and_generate_graph_with_langchain(user_query)
        if result["status"] == "failed":
            return f"Error: {result['error']}"
        
        # Fetch the filtered data with limited columns and rows
        base_query = """
            SELECT "DATE", "CITY", "INCIDENT", "VALUE" 
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        if "CITY" in user_query:
            base_query += f" AND CITY = '{user_query['CITY']}'"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND DATE BETWEEN '{start_date}' AND '{end_date}'"
        
        # Limit the data for analysis
        base_query += " LIMIT 50"  # Limiting to 50 rows to avoid token limit
        
        df = pd.read_sql(base_query, engine)
        
        # Ensure the INCIDENT column exists
        if 'INCIDENT' not in df.columns:
            raise ValueError("The 'INCIDENT' column is missing from the data.")
        
        # Create a concise summary of the data
        summary = f"""
        City: {user_query.get('CITY')}
        Date Range: {user_query.get('DATE_RANGE', ['', ''])[0]} to {user_query.get('DATE_RANGE', ['', ''])[1]}
        Total Records: {len(df)}
        Crime Types: {', '.join(df['INCIDENT'].unique())}
        """
        
        # Create a shorter prompt for the LLM
        prompt = f"""
        Analyze the crime data summary for {user_query.get('CITY')}:
        
        {summary}
        
        Provide a brief analysis focusing on:
        1. Most frequent crime types
        2. Notable trends
        3. Key insights
        Limit the analysis to 3-4 sentences.
        """
        
        # Get the analysis from the LLM
        response = llm.invoke(prompt)
        
        # Handle the AIMessage response
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        return f"""
Analysis Results for {user_query.get('CITY')}:
----------------
{analysis}

Graph Path: {result['chart_path']}
"""
    except Exception as e:
        print(f"Error during AI analysis with graph: {str(e)}")
        return "Analysis unavailable - Please try again later."


# Example Usage
if __name__ == "__main__":
    user_query = {
        "CITY": "Chicago",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    
    # Generate AI analysis with graph
    analysis = get_ai_analysis_with_graph(user_query)
    print(analysis)

def get_ai_analysis():
    """Get AI-generated analysis of NVIDIA metrics"""
    prompt = """Analyze NVIDIA financial metrics using the nvidia_financial_metrics tool.
    Provide a brief summary of key insights."""
    
    try:
        response = agent.invoke({"input": prompt})
        return response.get("output", str(response))
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return "Analysis unavailable - Rate limit exceeded. Please try again later."  

def query_and_generate_graph_with_langchain(user_query: dict) -> dict:
    try:
        # Base query with proper case handling
        base_query = """
            SELECT 
                CAST("DATE" as DATE) as "DATE",
                "CITY",
                "INCIDENT",
                CAST("VALUE" as INTEGER) as "VALUE"
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        
        if "CITY" in user_query:
            base_query += f" AND UPPER(\"CITY\") = UPPER('{user_query['CITY']}')"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND \"DATE\" BETWEEN '{start_date}' AND '{end_date}'"
        
        # Debug: Print the query
        print(f"Executing query: {base_query}")
        
        # Fetch data and immediately normalize column names
        df = pd.read_sql(base_query, engine)
        
        # Debug: Print raw DataFrame columns
        print("Raw DataFrame columns:", df.columns)
        
        # Normalize column names to uppercase
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        # Debug: Print normalized DataFrame
        print("Normalized DataFrame columns:", df.columns)
        print("First few rows:\n", df.head())
        
        # Verify data exists
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Ensure required columns exist
        required_columns = ['DATE', 'CITY', 'INCIDENT', 'VALUE']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Group data by INCIDENT and sum the values
        grouped_data = df.groupby('INCIDENT')['VALUE'].sum().reset_index()
        grouped_data = grouped_data.sort_values(by='VALUE', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(grouped_data['INCIDENT'], grouped_data['VALUE'], color='skyblue')
        plt.title(f"Crime Data for {user_query.get('CITY', 'All Cities')}", fontsize=16)
        plt.xlabel("Crime Type", fontsize=14)
        plt.ylabel("Total Incidents", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save the graph
        chart_file_path = "user_filtered_crime_data.png"
        plt.savefig(chart_file_path, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        
        return {
            "chart_path": chart_file_path,
            "status": "success"
        }
    except Exception as e:
        print(f"Error querying and generating graph: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

def generate_crime_data_summary(user_query: dict) -> dict:
    """
    Generate a detailed summary of crime data from Snowflake for a specific city and time period.
    
    Args:
        user_query (dict): Dictionary containing city and date range filters
    
    Returns:
        dict: Summary results and status
    """
    try:
        # Base query with proper case handling
        base_query = """
            SELECT 
                CAST("DATE" as DATE) as "DATE",
                "CITY",
                "INCIDENT",
                CAST("VALUE" as INTEGER) as "VALUE"
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        
        if "CITY" in user_query:
            base_query += f" AND UPPER(\"CITY\") = UPPER('{user_query['CITY']}')"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND \"DATE\" BETWEEN '{start_date}' AND '{end_date}'"
        
        # Debug: Print the query
        print(f"Executing query: {base_query}")
        
        # Fetch data and normalize column names
        df = pd.read_sql(base_query, engine)
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Calculate summary statistics
        total_incidents = df['VALUE'].sum()
        incident_counts = df.groupby('INCIDENT')['VALUE'].sum().sort_values(ascending=False)
        most_common_incidents = incident_counts.head(5)
        dates_range = pd.to_datetime(df['DATE'])
        
        # Create a detailed data summary
        data_summary = {
            "city": user_query.get('CITY'),
            "date_range": f"{dates_range.min().strftime('%Y-%m-%d')} to {dates_range.max().strftime('%Y-%m-%d')}",
            "total_incidents": total_incidents,
            "unique_incident_types": len(incident_counts),
            "top_incidents": most_common_incidents.to_dict(),
            "daily_average": total_incidents / len(dates_range.unique()),
            "total_days": len(dates_range.unique())
        }
        
        # Create prompt for LLM analysis
        prompt = f"""
        Analyze the following crime statistics for {data_summary['city']}:
        
        Time Period: {data_summary['date_range']}
        Total Incidents: {data_summary['total_incidents']}
        Number of Different Crime Types: {data_summary['unique_incident_types']}
        Daily Average: {data_summary['daily_average']:.2f} incidents
        
        Top 5 Most Common Incidents:
        {chr(10).join([f"- {incident}: {count}" for incident, count in data_summary['top_incidents'].items()])}
        
        Please provide a comprehensive analysis including:
        1. Overall crime trends and patterns
        2. Significance of the most common crime types
        3. Daily/periodic patterns if any
        4. Notable insights or concerns
        5. Comparison with expected averages
        """
        
        # Get analysis from LLM
        response = llm.invoke(prompt)
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "summary": f"""
Crime Analysis for {data_summary['city']}
Time Period: {data_summary['date_range']}
----------------------------------------
{analysis}

Statistical Overview:
- Total Incidents: {data_summary['total_incidents']:,}
- Daily Average: {data_summary['daily_average']:.2f}
- Unique Crime Types: {data_summary['unique_incident_types']}
- Days Analyzed: {data_summary['total_days']}
""",
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error generating crime data summary: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

def generate_crime_analysis(user_query: dict) -> dict:
    """
    Generate a detailed analysis of crime data from Snowflake without graphs.
    
    Args:
        user_query (dict): Dictionary containing city and date range filters
    
    Returns:
        dict: Analysis results and status
    """
    try:
        # Base query with proper case handling
        base_query = """
            SELECT 
                CAST("DATE" as DATE) as "DATE",
                "CITY",
                "VARIABLE",
                CAST("VALUE" as INTEGER) as "VALUE"
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        
        if "CITY" in user_query:
            base_query += f" AND UPPER(\"CITY\") = UPPER('{user_query['CITY']}')"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND \"DATE\" BETWEEN '{start_date}' AND '{end_date}'"
        
        # Debug: Print the query
        print(f"Executing query: {base_query}")
        
        # Fetch data and normalize column names
        df = pd.read_sql(base_query, engine)
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        # Debug: Print DataFrame info
        print("DEBUG: DataFrame columns:", df.columns)
        print("DEBUG: First few rows:")
        print(df.head())
        
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Calculate key metrics
        total_incidents = df['VALUE'].sum()
        crime_types = df.groupby('VARIABLE')['VALUE'].sum().sort_values(ascending=False)
        dates_range = pd.to_datetime(df['DATE'])
        date_range_str = f"{dates_range.min().strftime('%Y-%m-%d')} to {dates_range.max().strftime('%Y-%m-%d')}"
        
        # Prepare context for LLM
        context = f"""
        Crime Analysis Context for {user_query.get('CITY')}:
        
        Time Period: {date_range_str}
        Total Reported Incidents: {total_incidents:,}
        Number of Days: {len(dates_range.unique())}
        Daily Average: {total_incidents / len(dates_range.unique()):.2f}
        
        Crime Type Distribution:
        {crime_types.to_string()}
        
        Please analyze this data and provide:
        1. Overall crime trends and patterns
        2. Most significant crime types and their impact
        3. Any notable temporal patterns
        4. Key insights and potential areas of concern
        5. Recommendations based on the data
        """
        
        # Get analysis from LLM
        response = llm.invoke(context)
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "city": user_query.get('CITY'),
            "date_range": date_range_str,
            "total_incidents": total_incidents,
            "daily_average": total_incidents / len(dates_range.unique()),
            "analysis": analysis,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error generating crime analysis: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

# Example Usage
if __name__ == "__main__":
    user_query = {
        "CITY": "Chicago",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    
    # Generate analysis
    result = generate_crime_analysis(user_query)
    if result["status"] == "success":
        print(f"""
Crime Analysis Report
===================
City: {result['city']}
Date Range: {result['date_range']}
Total Incidents: {result['total_incidents']:,}
Daily Average: {result['daily_average']:.2f}

Analysis:
{result['analysis']}
""")
    else:
        print(f"Error: {result['error']}")

# Create LangChain tool for the Snowflake agent
snowflake_tool = Tool(
    name="nvidia_financial_metrics",
    description="Get NVIDIA financial valuation metrics from Snowflake",
    func=get_valuation_summary
)

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",  
    temperature=0,
    anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')  # Get from environment instead of hardcoding
) 

try:
    # Create agent with the tool
    # Simplify agent initialization
    agent = initialize_agent(
        tools=[snowflake_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Add specific agent type
        handle_parsing_errors=True,
        max_iterations=2,  # Limit iterations to reduce token usage
        early_stopping_method="generate"  # Add early stopping
    )
except Exception as e:
    print(f"Error initializing agent: {str(e)}")
    print("Available Claude models: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307")
    raise
 
# LangChain LLM Initialization
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
)

# LangChain Tool for Querying and Generating Graph
query_tool = Tool(
    name="query_and_generate_graph",
    description="Query Snowflake data based on user-specified criteria and generate a graph for the filtered data.",
    func=query_and_generate_graph_with_langchain
)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=[query_tool],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=2,
    early_stopping_method="generate"
)

def query_with_langchain(user_query: dict) -> str:
    """
    Use LangChain agent to query Snowflake data and generate a graph.

    Args:
        user_query (dict): A dictionary containing user-specified filters.

    Returns:
        str: Path to the generated graph or an error message.
    """
    try:
        # Convert user query to a string for the agent
        query_description = f"""
        Query Snowflake data with the following criteria:
        {user_query}
        Generate a graph for the filtered data.
        """
        
        # Invoke the LangChain agent
        response = agent.invoke({"input": query_description})
        return response.get("output", str(response))
    except Exception as e:
        print(f"Error during LangChain query: {str(e)}")
        return "Error: Unable to process the query."

# Example Usage
if __name__ == "__main__":
    user_query = {
        "CITY": "Chicago",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    result = query_and_generate_graph_with_langchain(user_query)
    if result["status"] == "success":
        print(f"Graph generated successfully: {result['chart_path']}")
        # Pass the user_query to get_ai_analysis_with_graph
        analysis = get_ai_analysis_with_graph(user_query)
        print(analysis)
    else:
        print(f"Error: {result['error']}")

    analysis = get_ai_analysis_with_graph(user_query)
    print(analysis)

if __name__ == "__main__":
    # Define user query
    user_query = {
        "CITY": "Chicago",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    
    # Generate graph and analysis
    result = query_and_generate_graph_with_langchain(user_query)
    if result["status"] == "success":
        print(f"Graph generated successfully: {result['chart_path']}")
        analysis = get_ai_analysis_with_graph(user_query)
        print(analysis)
    else:
        print(f"Error: {result['error']}")

def generate_table_summary(user_query: dict) -> dict:
    """
    Generate a summary of Snowflake table data based on user query.
    
    Args:
        user_query (dict): Dictionary containing filters like city and date range
        
    Returns:
        dict: Summary results and status
    """
    try:
        # Base query with proper case handling
        base_query = """
            SELECT 
                CAST("DATE" as DATE) as "DATE",
                "CITY",
                "VARIABLE",
                CAST("VALUE" as INTEGER) as "VALUE"
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        
        # Add filters based on user query
        if "CITY" in user_query:
            base_query += f" AND UPPER(\"CITY\") = UPPER('{user_query['CITY']}')"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND \"DATE\" BETWEEN '{start_date}' AND '{end_date}'"
        
        # Debug: Print the query
        print(f"Executing query: {base_query}")
        
        # Fetch data and normalize column names
        df = pd.read_sql(base_query, engine)
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Calculate summary statistics
        total_incidents = df['VALUE'].sum()
        crime_types = df.groupby('VARIABLE')['VALUE'].sum().sort_values(ascending=False)
        dates_range = pd.to_datetime(df['DATE'])
        num_days = len(dates_range.unique())
        daily_avg = total_incidents / num_days if num_days > 0 else 0
        
        # Create context for LLM analysis
        context = f"""
        Data Summary for {user_query.get('CITY')}:
        
        Time Period: {dates_range.min().strftime('%Y-%m-%d')} to {dates_range.max().strftime('%Y-%m-%d')}
        Total Incidents: {total_incidents:,}
        Number of Days Analyzed: {num_days}
        Daily Average: {daily_avg:.2f}
        
        Incident Distribution:
        {crime_types.to_string()}
        
        Please provide a comprehensive analysis including:
        1. Overall trends and patterns
        2. Most significant incident types
        3. Notable temporal patterns
        4. Key insights and concerns
        5. Recommendations based on the data
        """
        
        # Initialize LLM if not already initialized
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
        
        # Get analysis from LLM
        response = llm.invoke(context)
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "city": user_query.get('CITY'),
            "date_range": f"{dates_range.min().strftime('%Y-%m-%d')} to {dates_range.max().strftime('%Y-%m-%d')}",
            "total_incidents": total_incidents,
            "daily_average": daily_avg,
            "analysis": analysis,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


def generate_crime_summary(user_query: dict) -> dict:
    """
    Generate a focused crime data summary from Snowflake without graphs or extra analysis.
    
    Args:
        user_query (dict): Dictionary containing city and date range filters
    
    Returns:
        dict: Summary results and status
    """
    try:
        # Base query with proper case handling
        base_query = """
            SELECT 
                CAST("DATE" as DATE) as "DATE",
                "CITY",
                "VARIABLE",
                CAST("VALUE" as INTEGER) as "VALUE"
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        
        if "CITY" in user_query:
            base_query += f" AND UPPER(\"CITY\") = UPPER('{user_query['CITY']}')"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND \"DATE\" BETWEEN '{start_date}' AND '{end_date}'"
        
        # Fetch data and normalize column names
        df = pd.read_sql(base_query, engine)
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Calculate summary statistics
        total_incidents = df['VALUE'].sum()
        crime_types = df.groupby('VARIABLE')['VALUE'].sum().sort_values(ascending=False)
        top_crimes = crime_types.head(5)
        dates_range = pd.to_datetime(df['DATE'])
        num_days = len(dates_range.unique())
        daily_avg = total_incidents / num_days if num_days > 0 else 0
        
        return {
            "city": user_query.get('CITY'),
            "date_range": f"{dates_range.min().strftime('%Y-%m-%d')} to {dates_range.max().strftime('%Y-%m-%d')}",
            "total_incidents": total_incidents,
            "daily_average": daily_avg,
            "unique_crime_types": len(crime_types),
            "top_crimes": top_crimes.to_dict(),
            "days_analyzed": num_days,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error generating crime summary: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }





def compare_cities_safety(num_cities: int = 5) -> dict:
    """
    Compare crime statistics between multiple cities using Snowflake data from 2020-2025.
    Lower daily average means safer city.
    """
    try:
        # Base query to get data for all cities with proper date range
        base_query = """
            SELECT 
                CAST("DATE" as DATE) as "DATE",
                "CITY",
                "VARIABLE",
                CAST("VALUE" as INTEGER) as "VALUE"
            FROM CLEANED_CRIME_DATASET 
            WHERE "DATE" BETWEEN '2020-01-01' AND '2025-12-31'
            AND "VARIABLE" LIKE 'Daily_count_of_incidents_%'
            ORDER BY "DATE" DESC
        """
        
        print(f"Executing query: {base_query}")
        df = pd.read_sql(base_query, engine)
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        if df.empty:
            raise ValueError("No data returned from Snowflake")

        # Calculate city-wise statistics
        city_stats = []
        for city in df['CITY'].unique():
            city_data = df[df['CITY'] == city].copy()
            
            # Calculate total incidents and daily average
            total_incidents = city_data['VALUE'].sum()
            num_days = len(city_data['DATE'].unique())
            daily_avg = total_incidents / num_days if num_days > 0 else 0
            
            # Get crime type distribution
            crime_types = city_data.groupby('VARIABLE')['VALUE'].sum().sort_values(ascending=False)
            most_common = crime_types.index[0].replace('Daily_count_of_incidents_', '').replace('_', ' ').title()
            
            # Calculate yearly trends
            city_data['YEAR'] = pd.to_datetime(city_data['DATE']).dt.year
            yearly_trend = city_data.groupby('YEAR')['VALUE'].sum()
            
            # Calculate trend percentage
            if len(yearly_trend) > 1:
                trend_pct = ((yearly_trend.iloc[-1] - yearly_trend.iloc[0]) / yearly_trend.iloc[0]) * 100
            else:
                trend_pct = 0
            
            city_stats.append({
                'city': city,
                'total_incidents': total_incidents,
                'daily_average': daily_avg,
                'most_common_crime': most_common,
                'yearly_trend': yearly_trend.to_dict(),
                'trend_percentage': trend_pct
            })
        
        # Create DataFrame and sort by safety (lower daily average = safer)
        city_stats_df = pd.DataFrame(city_stats)
        city_stats_df = city_stats_df.sort_values('daily_average')
        
        # Add safety rankings (1 = safest)
        city_stats_df['safety_rank'] = range(1, len(city_stats_df) + 1)
        
        # Calculate safety scores (100 = safest, 0 = least safe)
        max_daily_avg = city_stats_df['daily_average'].max()
        city_stats_df['safety_score'] = (1 - (city_stats_df['daily_average'] / max_daily_avg)) * 100
        
        # Add safety levels
        city_stats_df['safety_level'] = city_stats_df['safety_score'].apply(lambda x: 
            "Very Safe" if x >= 80 else
            "Safe" if x >= 60 else
            "Moderate" if x >= 40 else
            "Less Safe" if x >= 20 else "Concerning"
        )
        
        return {
            "city_stats": city_stats_df.to_dict('records'),
            "total_cities": len(city_stats_df),
            "date_range": "2020-01-01 to 2025-12-31",
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error comparing cities: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

def generate_city_safety_statistics(city_stats: list) -> dict:
    """Generate statistical summary for all cities"""
    stats = {
        'avg_daily_incidents': np.mean([city['daily_average'] for city in city_stats]),
        'total_incidents_by_year': {},
        'year_over_year_changes': {},
        'safest_year': None,
        'most_dangerous_year': None
    }
    
    # Calculate total incidents by year across all cities
    years = range(2020, 2025)  # Only 2020-2024
    for year in years:
        yearly_total = sum(city['yearly_trend'].get(year, 0) for city in city_stats)
        stats['total_incidents_by_year'][year] = yearly_total
    
    # Calculate year-over-year changes
    for year in range(2021, 2025):
        prev_year = stats['total_incidents_by_year'][year - 1]
        curr_year = stats['total_incidents_by_year'][year]
        pct_change = ((curr_year - prev_year) / prev_year) * 100
        stats['year_over_year_changes'][year] = pct_change
    
    # Find safest and most dangerous years
    min_year = min(stats['total_incidents_by_year'].items(), key=lambda x: x[1])
    max_year = max(stats['total_incidents_by_year'].items(), key=lambda x: x[1])
    stats['safest_year'] = min_year
    stats['most_dangerous_year'] = max_year
    
    return stats

if __name__ == "__main__":
    result = compare_cities_safety()
    
    if result["status"] == "success":
        print(f"""
Multi-City Safety Comparison Report (2020-2024)
============================================
Analysis Period: 2020-01-01 to 2024-12-31
Total Cities Analyzed: {result['total_cities']}

City Rankings (1 = Safest):
------------------------""")
        
        # Sort cities by safety rank
        cities = sorted(result['city_stats'], key=lambda x: x['safety_rank'])
        
        # Generate statistics
        stats = generate_city_safety_statistics(cities)
        
        for city in cities:
            trend_direction = "↑" if city['trend_percentage'] > 0 else "↓" if city['trend_percentage'] < 0 else "→"
            
            print(f"""
{city['safety_rank']}. {city['city']}:
   Safety Score: {city['safety_score']:.1f}%
   Safety Level: {city['safety_level']}
   Daily Average Incidents: {city['daily_average']:.2f}
   Total Incidents (2020-2024): {sum(v for k,v in city['yearly_trend'].items() if k <= 2024):,}
   Most Common Crime: {city['most_common_crime']}
   4-Year Trend: {trend_direction} {abs(city['trend_percentage']):.1f}%
   
   Yearly Incident Totals:""")
            
            # Show only 2020-2024 data
            for year in range(2020, 2025):
                if year in city['yearly_trend']:
                    print(f"   {year}: {city['yearly_trend'][year]:,}")
            
            print("-" * 50)
        
        # Print Statistical Summary
        print(f"""
Statistical Summary (2020-2024)
============================
Average Daily Incidents Across All Cities: {stats['avg_daily_incidents']:.2f}

Yearly Totals Across All Cities:
----------------------------""")
        for year, total in stats['total_incidents_by_year'].items():
            print(f"{year}: {total:,}")
        
        print("\nYear-over-Year Changes:")
        print("---------------------")
        for year, change in stats['year_over_year_changes'].items():
            print(f"{year}: {'↑' if change > 0 else '↓'} {abs(change):.1f}%")
        
        print(f"""
Key Findings:
-----------
Safest Year: {stats['safest_year'][0]} ({stats['safest_year'][1]:,} incidents)
Most Dangerous Year: {stats['most_dangerous_year'][0]} ({stats['most_dangerous_year'][1]:,} incidents)
Safest City: {cities[0]['city']} (Score: {cities[0]['safety_score']:.1f}%)
Most Concerning City: {cities[-1]['city']} (Score: {cities[-1]['safety_score']:.1f}%)
""")
    else:
        print(f"Error: {result['error']}")

def generate_table_summary(cities_data: dict) -> dict:
    """
    Generate a comparative summary of crime statistics across cities.
    
    Args:
        cities_data (dict): Dictionary containing city statistics and rankings
    
    Returns:
        dict: Comparative summary and analysis
    """
    try:
        # Sort cities by safety ranking
        cities = sorted(cities_data['city_stats'], key=lambda x: x['safety_rank'])
        
        # Calculate overall statistics
        total_cities = len(cities)
        avg_daily_incidents = np.mean([city['daily_average'] for city in cities])
        
        # Calculate yearly trends
        yearly_totals = {}
        for city in cities:
            for year, incidents in city['yearly_trend'].items():
                if 2020 <= year <= 2024:  # Only consider 2020-2024
                    yearly_totals[year] = yearly_totals.get(year, 0) + incidents
        
        # Calculate year-over-year changes
        yoy_changes = {}
        for year in range(2021, 2025):
            prev_year = yearly_totals[year - 1]
            curr_year = yearly_totals[year]
            pct_change = ((curr_year - prev_year) / prev_year) * 100
            yoy_changes[year] = pct_change
        
        # Identify safest and most dangerous years
        safest_year = min(yearly_totals.items(), key=lambda x: x[1])
        most_dangerous_year = max(yearly_totals.items(), key=lambda x: x[1])
        
        # Create summary context for LLM
        context = f"""
        Crime Statistics Comparative Analysis (2020-2024)
        
        Overall Statistics:
        - Total Cities Analyzed: {total_cities}
        - Average Daily Incidents Across All Cities: {avg_daily_incidents:.2f}
        - Safest City: {cities[0]['city']} (Score: {cities[0]['safety_score']:.1f}%)
        - Most Concerning City: {cities[-1]['city']} (Score: {cities[-1]['safety_score']:.1f}%)
        
        Yearly Totals:
        {chr(10).join([f"- {year}: {total:,} incidents" for year, total in yearly_totals.items()])}
        
        Year-over-Year Changes:
        {chr(10).join([f"- {year}: {'↑' if change > 0 else '↓'} {abs(change):.1f}%" for year, change in yoy_changes.items()])}
        
        Key Time Periods:
        - Safest Year: {safest_year[0]} ({safest_year[1]:,} incidents)
        - Most Dangerous Year: {most_dangerous_year[0]} ({most_dangerous_year[1]:,} incidents)
        
        City Safety Rankings:
        {chr(10).join([f"{city['safety_rank']}. {city['city']}: {city['safety_level']} ({city['safety_score']:.1f}%)" for city in cities])}
        
        Please provide a comprehensive analysis including:
        1. Overall safety trends across cities
        2. Notable patterns in yearly changes
        3. Significant differences between cities
        4. Key factors affecting safety scores
        5. Recommendations for improving city safety
        """
        
        # Get analysis from LLM
        response = llm.invoke(context)
        analysis = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "summary_stats": {
                "total_cities": total_cities,
                "avg_daily_incidents": avg_daily_incidents,
                "yearly_totals": yearly_totals,
                "yoy_changes": yoy_changes,
                "safest_year": safest_year,
                "most_dangerous_year": most_dangerous_year,
                "safest_city": cities[0],
                "most_concerning_city": cities[-1]
            },
            "analysis": analysis,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error generating comparative summary: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

# Example Usage
if __name__ == "__main__":
    # First get the city comparison data
    cities_data = compare_cities_safety()
    
    if cities_data["status"] == "success":
        # Generate comparative summary
        result = generate_table_summary(cities_data)
        
        if result["status"] == "success":
            stats = result["summary_stats"]
            print(f"""
Comparative Crime Analysis Report (2020-2024)
=========================================
Overview:
--------
Total Cities Analyzed: {stats['total_cities']}
Average Daily Incidents: {stats['avg_daily_incidents']:.2f}

Safety Rankings:
-------------
Most Safe: {stats['safest_city']['city']} (Score: {stats['safest_city']['safety_score']:.1f}%)
Least Safe: {stats['most_concerning_city']['city']} (Score: {stats['most_concerning_city']['safety_score']:.1f}%)

Temporal Trends:
-------------
Safest Year: {stats['safest_year'][0]} ({stats['safest_year'][1]:,} incidents)
Most Dangerous Year: {stats['most_dangerous_year'][0]} ({stats['most_dangerous_year'][1]:,} incidents)

Year-over-Year Changes:
--------------------""")
            for year, change in stats['yoy_changes'].items():
                print(f"{year}: {'↑' if change > 0 else '↓'} {abs(change):.1f}%")
            
            print("\nDetailed Analysis:")
            print("----------------")
            print(result["analysis"])
        else:
            print(f"Error: {result['error']}")
    else:
        print(f"Error: {cities_data['error']}")

# Get city comparison data
cities_data = compare_cities_safety()

# Generate comparative summary
if cities_data["status"] == "success":
    summary = generate_table_summary(cities_data)
    if summary["status"] == "success":
        print(summary["analysis"])