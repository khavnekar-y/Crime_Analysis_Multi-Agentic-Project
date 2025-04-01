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
            SELECT DATE, CITY, INCIDENT, VALUE 
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
    """
    Query Snowflake data based on user-specified criteria using LangChain and generate a graph for the filtered data.

    Args:
        user_query (dict): A dictionary containing user-specified filters, e.g.,
                           {"CITY": "Los Angeles", "DATE_RANGE": ["2025-01-01", "2025-03-30"]}

    Returns:
        dict: A dictionary containing the graph file path and status.
    """
    try:
        # Base query
        base_query = "SELECT DATE, CITY, INCIDENT, VALUE FROM CLEANED_CRIME_DATASET WHERE 1=1"
        
        # Add filters based on user query
        if "CITY" in user_query:
            base_query += f" AND CITY = '{user_query['CITY']}'"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND DATE BETWEEN '{start_date}' AND '{end_date}'"
        
        # Fetch data from Snowflake
        print(f"Executing query: {base_query}")
        df = pd.read_sql(base_query, engine)
        
        # Check if data is returned
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Normalize column names
        df.columns = df.columns.str.upper().str.strip()

        # Debugging
        print("DEBUG: Filtered DataFrame columns:", df.columns)
        print("DEBUG: First few rows of the filtered DataFrame:\n", df.head())
        
        # Ensure the required columns are present
        required_columns = ['CITY', 'INCIDENT', 'VALUE', 'DATE']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"The required column '{col}' is missing from the data.")
        
        # Group data by INCIDENT and sum the values
        grouped_data = df.groupby('INCIDENT')['VALUE'].sum().reset_index()
        
        # Sort by VALUE to get the top crimes
        grouped_data = grouped_data.sort_values(by='VALUE', ascending=False)
        
        # Create a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(grouped_data['INCIDENT'], grouped_data['VALUE'], color='skyblue')
        plt.title(f"Crime Data for {user_query.get('CITY', 'All Cities')}", fontsize=16)
        plt.xlabel("Crime Type", fontsize=14)
        plt.ylabel("Total Value", fontsize=14)
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

def query_and_generate_graph_for_incident(user_query: dict, incident_type: str) -> dict:
    """
    Query Snowflake data for a specific incident type and generate a graph focusing on ZIP codes with maximum crimes.

    Args:
        user_query (dict): A dictionary containing user-specified filters, e.g.,
                           {"CITY": "Chicago", "DATE_RANGE": ["2025-01-01", "2025-03-30"]}
        incident_type (str): The specific incident type to filter, e.g., "thefts", "battery_or_assault".

    Returns:
        dict: A dictionary containing the graph file path and status.
    """
    try:
        # Base query
        base_query = f"""
            SELECT DATE, ZIPCODE, CITY, INCIDENT, VALUE 
            FROM CLEANED_CRIME_DATASET 
            WHERE INCIDENT = '{incident_type}'
        """
        
        # Add filters based on user query
        if "CITY" in user_query:
            base_query += f" AND CITY = '{user_query['CITY']}'"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND DATE BETWEEN '{start_date}' AND '{end_date}'"
        
        # Fetch data from Snowflake
        print(f"Executing query: {base_query}")
        df = pd.read_sql(base_query, engine)
        
        # Check if data is returned
        if df.empty:
            raise ValueError(f"No data returned for the specified filters and incident type: {incident_type}.")
        
        # Normalize column names
        df.columns = df.columns.str.upper().str.strip()

        # Debugging
        print("DEBUG: Filtered DataFrame columns:", df.columns)
        print("DEBUG: First few rows of the filtered DataFrame:\n", df.head())
        
        # Ensure the required columns are present
        required_columns = ['ZIPCODE', 'INCIDENT', 'VALUE', 'DATE']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"The required column '{col}' is missing from the data.")
        
        # Ensure the INCIDENT column exists
        if 'INCIDENT' not in df.columns:
            raise ValueError("The 'INCIDENT' column is missing from the data.")

        # Filter data for the specific incident type
        incident_df = df[df['INCIDENT'] == incident_type]

        # Ensure the filtered DataFrame is not empty
        if incident_df.empty:
            raise ValueError(f"No data returned for the incident type: {incident_type}.")

        # Group data by ZIPCODE and sum the values
        grouped_data = incident_df.groupby('ZIPCODE')['VALUE'].sum().reset_index()
        
        # Sort by VALUE to get the top 5 ZIP codes
        grouped_data = grouped_data.sort_values(by='VALUE', ascending=False).head(5)
        
        # Create a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(grouped_data['ZIPCODE'].astype(str), grouped_data['VALUE'], color='skyblue')
        plt.title(f"Top ZIP Codes for {incident_type.title()} in {user_query.get('CITY', 'All Cities')}", fontsize=16)
        plt.xlabel("ZIP Code", fontsize=14)
        plt.ylabel("Total Incidents", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save the graph
        chart_file_path = f"{incident_type}_zipcodes_crime_data.png"
        plt.savefig(chart_file_path, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        
        return {
            "chart_path": chart_file_path,
            "status": "success"
        }
    except Exception as e:
        print(f"Error querying and generating graph for incident: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }

def query_and_generate_graphs_for_top_incidents(user_query: dict) -> dict:
    """
    Query Snowflake data for the top 5 incident types in a city and generate graphs for each incident
    focusing on the top 5 ZIP codes with maximum crimes.

    Args:
        user_query (dict): A dictionary containing user-specified filters, e.g.,
                           {"CITY": "Chicago", "DATE_RANGE": ["2025-01-01", "2025-03-30"]}

    Returns:
        dict: A dictionary containing the graph file paths and status.
    """
    try:
        # Base query to get all incidents for the specified city and date range
        base_query = """
            SELECT DATE, ZIPCODE, CITY, INCIDENT, VALUE 
            FROM CLEANED_CRIME_DATASET 
            WHERE 1=1
        """
        if "CITY" in user_query:
            base_query += f" AND CITY = '{user_query['CITY']}'"
        if "DATE_RANGE" in user_query and len(user_query["DATE_RANGE"]) == 2:
            start_date, end_date = user_query["DATE_RANGE"]
            base_query += f" AND DATE BETWEEN '{start_date}' AND '{end_date}'"
        
        # Fetch data from Snowflake
        print(f"Executing query: {base_query}")
        df = pd.read_sql(base_query, engine)
        
        # Ensure the DataFrame is not empty
        if df.empty:
            raise ValueError("No data returned for the specified filters. Please refine your query.")
        
        # Normalize column names
        df.columns = df.columns.str.upper().str.strip()

        # Ensure the INCIDENT column exists
        if 'INCIDENT' not in df.columns:
            raise ValueError("The 'INCIDENT' column is missing from the data.")

        # Group data by INCIDENT and sum the values to get the top 5 incidents
        top_incidents = (
            df.groupby('INCIDENT')['VALUE']
            .sum()
            .reset_index()
            .sort_values(by='VALUE', ascending=False)
            .head(5)
        )
        
        # Generate a graph for each top incident
        graph_paths = []
        for _, row in top_incidents.iterrows():
            incident_type = row['INCIDENT']
            incident_df = df[df['INCIDENT'] == incident_type]
            
            # Group data by ZIPCODE and sum the values
            grouped_data = (
                incident_df.groupby('ZIPCODE')['VALUE']
                .sum()
                .reset_index()
                .sort_values(by='VALUE', ascending=False)
                .head(5)  # Include only the top 5 ZIP codes
            )
            
            # Create a bar chart
            plt.figure(figsize=(12, 6))
            plt.bar(grouped_data['ZIPCODE'].astype(str), grouped_data['VALUE'], color='skyblue')
            plt.title(f"Top 5 ZIP Codes for {incident_type.title()} in {user_query.get('CITY', 'All Cities')}", fontsize=16)
            plt.xlabel("ZIP Code", fontsize=14)
            plt.ylabel("Total Incidents", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=12)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            # Save the graph
            chart_file_path = f"{incident_type}_top5_zipcodes_crime_data.png"
            plt.savefig(chart_file_path, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            
            graph_paths.append(chart_file_path)
        
        return {
            "graph_paths": graph_paths,
            "status": "success"
        }
    except Exception as e:
        print(f"Error querying and generating graphs for top incidents: {str(e)}")
        return {
            "error": str(e),
            "status": "failed"
        }


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

def generate_crime_forecast(user_query: dict, forecast_days: int = 30) -> dict:
    """
    Generate crime forecast based on historical data from Snowflake.
    
    Args:
        user_query (dict): Dictionary containing filters like:
            {"CITY": "Chicago", "DATE_RANGE": ["2025-01-01", "2025-03-30"]}
        forecast_days (int): Number of days to forecast into the future
        
    Returns:
        dict: Forecast results and status
    """
    try:
        # Base query to get historical data
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
        
        # Fetch data and prepare for forecasting
        df = pd.read_sql(base_query, engine)
        df.columns = [col.strip('"').upper() for col in df.columns]
        
        if df.empty:
            raise ValueError("No historical data available for forecasting")
        
        # Convert DATE to datetime and sort
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE')
        
        # Group by date and get daily totals
        daily_crimes = df.groupby('DATE')['VALUE'].sum().reset_index()
        daily_crimes = daily_crimes.set_index('DATE')
        
        # Calculate rolling statistics for the trend
        rolling_mean = daily_crimes['VALUE'].rolling(window=7).mean()
        rolling_std = daily_crimes['VALUE'].rolling(window=7).std()
        
        # Generate future dates
        last_date = daily_crimes.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=forecast_days, 
                                   freq='D')
        
        # Simple forecasting using exponential smoothing
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Fit the model
        model = ExponentialSmoothing(daily_crimes['VALUE'],
                                   seasonal_periods=7,  # Weekly seasonality
                                   trend='add',
                                   seasonal='add')
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(forecast_days)
        forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['FORECAST'])
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(daily_crimes.index, daily_crimes['VALUE'], 
                label='Historical Data', color='blue', alpha=0.6)
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['FORECAST'], 
                label='Forecast', color='red', linestyle='--')
        
        # Add confidence intervals
        plt.fill_between(daily_crimes.index,
                        rolling_mean - 2*rolling_std,
                        rolling_mean + 2*rolling_std,
                        color='gray', alpha=0.2, label='Confidence Interval')
        
        plt.title(f'Crime Forecast for {user_query["CITY"]}\n'
                 f'Historical Period: {start_date} to {end_date}\n'
                 f'Forecast Period: {forecast_days} days',
                 fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Incidents', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the forecast graph
        forecast_path = f"crime_forecast_{user_query['CITY']}.png"
        plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate forecast statistics
        forecast_stats = {
            "average_daily_forecast": forecast.mean(),
            "max_daily_forecast": forecast.max(),
            "min_daily_forecast": forecast.min(),
            "total_forecast_incidents": forecast.sum()
        }
        
        return {
            "forecast_path": forecast_path,
            "forecast_stats": forecast_stats,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error generating forecast: {str(e)}")
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

    # Define user query
    user_query = {
        "CITY": "Chicago",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    
    # Specify the incident type
    incident_type = "thefts"  # Change this to "battery_or_assault" or any other incident type as needed
    
    # Generate graph for the specified incident type
    result = query_and_generate_graph_for_incident(user_query, incident_type)
    if result["status"] == "success":
        print(f"Graph generated successfully: {result['chart_path']}")
    else:
        print(f"Error: {result['error']}")

    # Define user query
    user_query = {
        "CITY": "Los Angeles",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    
    # Generate graphs for the top 5 incidents
    result = query_and_generate_graphs_for_top_incidents(user_query)
    if result["status"] == "success":
        print("Graphs generated successfully:")
        for path in result["graph_paths"]:
            print(f"- {path}")
    else:
        print(f"Error: {result['error']}")

    # Example usage
    user_query = {
        "CITY": "Chicago",
        "DATE_RANGE": ["2025-01-01", "2025-03-30"]
    }
    
    # Generate forecast for next 30 days
    result = generate_crime_forecast(user_query, forecast_days=30)
    
    if result["status"] == "success":
        print(f"""
Crime Forecast Report
===================
City: {user_query['CITY']}

Forecast Statistics:
------------------
Average Daily Incidents: {result['forecast_stats']['average_daily_forecast']:.2f}
Maximum Daily Incidents: {result['forecast_stats']['max_daily_forecast']:.2f}
Minimum Daily Incidents: {result['forecast_stats']['min_daily_forecast']:.2f}
Total Forecast Incidents: {result['forecast_stats']['total_forecast_incidents']:.0f}

Forecast Graph: {result['forecast_path']}
""")
    else:
        print(f"Error: {result['error']}")

# Example usage
user_query = {
    "CITY": "Chicago",
    "DATE_RANGE": ["2025-01-01", "2025-03-30"]
}

# Generate 30-day forecast
result = generate_crime_forecast(user_query, forecast_days=30)