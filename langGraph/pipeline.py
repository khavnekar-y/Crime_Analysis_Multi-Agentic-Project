"""
Criminal Report Generation Pipeline using LangGraph
Integrates web search and Snowflake data for comprehensive criminal report analysis
"""

import os
import sys
import operator
import traceback
from typing import TypedDict, Dict, Any, List, Annotated
from datetime import datetime
from typing import Optional
from agents.rag_agent import RAGAgent  
from agents.snowflake_agent import CrimeDataAnalyzer
from dataclasses import dataclass

# LangChain and LangGraph imports
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from graphviz import Digraph

# Add parent directory to path for imports from other project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent functions (assumed to be implemented in your project)
from agents.websearch_agent import search_quarterly  # Web search function for crime reports
from agents.snowflake_agent import query_snowflake, get_valuation_summary  # Snowflake query functions

###############################################################################
# State definition for Criminal Report Generation
###############################################################################
class CriminalReportState(TypedDict, total=False):
    input: str  # User's original query
    question: str  # Processed question
    search_type: str  # e.g., "Specific Quarter" (if applicable)
    selected_periods: List[str]  # List of periods (if applicable)
    web_output: str  # Results from web search
    snowflake_output: Dict[str, Any]  # Results from Snowflake query
    valuation_data: Dict[str, Any]  # Financial/metrics visualization data (if applicable)
    chat_history: List[Dict[str, Any]]  # Conversation history
    intermediate_steps: Annotated[List[tuple[Any, str]], operator.add]  # Agent reasoning steps
    assistant_response: str  # Final agent response
    final_report: Dict[str, Any]  # Final structured report


###############################################################################
# Node Functions
###############################################################################
def start_node(state: CriminalReportState) -> Dict:
    """Initial node that processes the input query."""
    return {"question": state["input"]}


def web_search_node(state: CriminalReportState) -> Dict:
    """Execute web search for criminal report data."""
    try:
        result = search_quarterly(state["question"])
        return {"web_output": result}
    except Exception as e:
        return {"web_output": f"Web search error: {str(e)}"}


def snowflake_node(state: CriminalReportState) -> Dict:
    """Execute Snowflake query (for metrics or related data)."""
    try:
        query_result = {
            "metrics": "Valuation metrics and crime statistics available",
            "latest_date": "Recent date",
            "query_status": "success"
        }
        valuation_data = get_valuation_summary()
        return {"snowflake_output": query_result, "valuation_data": valuation_data}
    except Exception as e:
        return {"snowflake_output": {"error": str(e)}, "valuation_data": {"error": str(e)}}


def agent_node(state: CriminalReportState, report_gpt):
    """
    Execute the LLM agent to combine web search and Snowflake data.
    Produces a cohesive answer using the web summary and Snowflake data.
    """
    try:
        raw_web = state.get("web_output", "No web data available.")
        snowflake_data = state.get("snowflake_output", {}).get("metrics", "No metrics data.")
        
        prompt = f"""
You are an AI assistant tasked with generating a criminal report.
Below are two sources of data:

Web Search Summary:
{raw_web}

Structured Metrics:
{snowflake_data}

Question: {state['question']}

Please integrate these insights into a clear, professionally formatted criminal report.
Structure your response with clear headings:
1. Overview of Current Incidents
2. Metrics & Trends
3. Final Analysis and Conclusions
"""
        final_response = report_gpt.invoke({"input": prompt})
        if isinstance(final_response, dict) and "output" in final_response:
            return {"assistant_response": final_response["output"]}
        else:
            return {"assistant_response": str(final_response)}
    except Exception as e:
        return {"assistant_response": f"Agent analysis error: {str(e)}"}


def final_report_node(state: CriminalReportState) -> Dict:
    """Generate the final structured criminal report."""
    try:
        report = {
            "introduction": f"Criminal Report Analysis for: {state['question']}",
            "key_findings": [
                f"Web Summary: {state.get('web_output', 'N/A')[:200]}...",
                f"Metrics: {state.get('snowflake_output', {}).get('metrics', 'N/A')}"
            ],
            "analysis": state.get("assistant_response", "No analysis provided."),
            "conclusion": "Based on the gathered data, the criminal incidents and metrics indicate key trends that require attention.",
            "sources": ["Web Search", "Snowflake", "AI Analysis"]
        }
        return {"final_report": report}
    except Exception as e:
        return {"final_report": {
            "introduction": "Error generating report",
            "key_findings": [f"Error: {str(e)}"],
            "analysis": "Analysis unavailable due to error",
            "conclusion": "Unable to generate conclusion",
            "sources": []
        }}


###############################################################################
# Agent Initialization and Pipeline Building
###############################################################################
def create_tools():
    """Create LangChain tools for the agent (web search and snowflake only)."""
    from langchain_core.tools import Tool
    return [
        Tool(
            name="web_search",
            func=search_quarterly,
            description="Search for criminal reports from web sources"
        ),
        Tool(
            name="snowflake_query",
            func=query_snowflake,
            description="Query Snowflake for criminal metrics and financial data"
        ),
        Tool(
            name="generate_report",
            func=final_report_node,
            description="Generate a structured criminal report from analyzed information"
        )
    ]


def initialize_report_generator():
    """Initialize a focused report generator agent using an LLM."""
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0,
        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    
    tools = create_tools()
    system_message = (
        "You are a focused report generator for criminal report analysis. "
        "Your task is to produce clear, structured, and concise reports. "
        "DO NOT show your reasoning. Provide only the final report in professional format."
    )
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        handle_parsing_errors=True, 
        verbose=False,
        agent_kwargs={
            "prefix": system_message,
            "format_instructions": (
                "To use a tool, use the format:\n"
                "Action: <action name>\n"
                "Action Input: <input>\n"
                "When finished, respond with:\n"
                "Final Report: <your final report>\n"
                "Begin!"
            )
        }
    )
    return agent


def build_pipeline(selected_agents: List[str] = None):
    """
    Build and return the compiled pipeline.
    For criminal report generation, we include Web Search and Snowflake nodes.
    """
    if selected_agents is None:
        selected_agents = []

    report_gpt = initialize_report_generator()
    graph = StateGraph(CriminalReportState)
    graph.add_node("start", start_node)
    graph.set_entry_point("start")

    # Always include Web Search
    graph.add_node("web_search", web_search_node)
    graph.add_edge("start", "web_search")

    # Always include Snowflake query
    graph.add_node("snowflake", snowflake_node)
    graph.add_edge("web_search", "snowflake")

    # Agent node using LLM to combine results
    def agent_node_with_gpt(state):
        return agent_node(state, report_gpt)
    graph.add_node("agent", agent_node_with_gpt)
    graph.add_edge("snowflake", "agent")

    # Final report node
    graph.add_node("report_generator", final_report_node)
    graph.add_edge("agent", "report_generator")
    graph.add_edge("report_generator", END)

    return graph.compile()


###############################################################################
# Main Invocation
###############################################################################
if __name__ == "__main__":
    try:
        # Build pipeline with Web Search and Snowflake agents (no RAG)
        pipeline = build_pipeline(["Web Search Agent", "Snowflake Agent"])

        result = pipeline.invoke({
            "input": "Analyze recent criminal incidents in New York for Q4 2023",
            "question": "Analyze recent criminal incidents in New York for Q4 2023",
            "search_type": "Specific Quarter",
            "selected_periods": ["2023q4"],
            "chat_history": [],
            "intermediate_steps": []
        })

        print("\n✅ Analysis Complete!")
        final_report = result.get("final_report", {})
        print(f"INTRO: {final_report.get('introduction', '')}")
        print(f"FINDINGS: {final_report.get('key_findings', [])}")
        print(f"ANALYSIS: {final_report.get('analysis', '')}")
        print(f"CONCLUSION: {final_report.get('conclusion', '')}")

    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
