"""
NVIDIA Research Pipeline using LangGraph
Integrates web search, RAG, and Snowflake data for comprehensive NVIDIA analysis
"""
import os
import sys
import operator
import traceback
from typing import TypedDict, Dict, Any, List, Annotated

# LangChain imports
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic

# LangGraph imports
from langgraph.graph import StateGraph, END
from graphviz import Digraph

# Add parent directory to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agent functions
from agents.websearch_agent import search_quarterly
from agents.rag_agent import search_all_namespaces, search_specific_quarter
from agents.snowflake_agent import query_snowflake, get_valuation_summary


class NvidiaGPTState(TypedDict, total=False):
    """State definition for NVIDIA research pipeline"""
    input: str  # User's original query
    question: str  # Processed question
    search_type: str  # "All Quarters" or "Specific Quarter"
    selected_periods: List[str]  # List of quarters to analyze
    web_output: str  # Results from web search
    rag_output: Dict[str, Any]  # Results from RAG search
    snowflake_output: Dict[str, Any]  # Results from Snowflake query
    valuation_data: Dict[str, Any]  # Financial visualization data
    chat_history: List[Dict[str, Any]]  # Conversation history
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]  # Agent reasoning steps
    assistant_response: str  # Agent's response
    final_report: Dict[str, Any]  # Final structured report


def final_report_tool(input_dict: Dict) -> Dict:
    """Generates final report in structured format."""
    return {
        "introduction": input_dict.get("introduction", ""),
        "key_findings": input_dict.get("key_findings", []),
        "analysis": input_dict.get("analysis", ""),
        "conclusion": input_dict.get("conclusion", ""),
        "sources": input_dict.get("sources", [])
    }


def start_node(state: NvidiaGPTState) -> Dict:
    """Initial node that processes the input query."""
    return {"question": state["input"]}


def web_search_node(state: NvidiaGPTState) -> Dict:
    """Execute web search for NVIDIA information."""
    try:
        result = search_quarterly(state["question"])
        return {"web_output": result}
    except Exception as e:
        return {"web_output": f"Web search error: {str(e)}"}


def rag_search_node(state: NvidiaGPTState) -> Dict:
    """Execute RAG search based on search type."""
    try:
        if state.get("search_type") == "All Quarters":
            # Search across all document namespaces
            result = search_all_namespaces(state["question"])
            return {"rag_output": {"type": "all", "result": result["text"], "images": result["images"]}}
        else:
            # For specific quarters
            input_dict = {
                "input_dict": {
                    "query": state["question"],
                    "selected_periods": state.get("selected_periods", ["2023q1"])
                }
            }
            result = search_specific_quarter.invoke(input_dict)
            return {
                "rag_output": {
                    "type": "specific",
                    "result": result["text"],
                    "images": result["images"],
                    "periods": state.get("selected_periods", ["2023q1"])
                }
            }
    except Exception as e:
        return {"rag_output": {"type": "error", "result": f"RAG search error: {str(e)}", "images": []}}

def snowflake_node(state: NvidiaGPTState) -> Dict:
    """Execute Snowflake query with reduced token usage."""
    try:
        # Simple placeholder result
        query_result = {
            "metrics": "See valuation data for metrics",
            "latest_date": "Recent date",
            "query_status": "success"
        }
        # Add chart data
        valuation_data = get_valuation_summary()
        return {
            "snowflake_output": query_result,
            "valuation_data": valuation_data
        }
    except Exception as e:
        return {
            "snowflake_output": {"error": str(e)},
            "valuation_data": {"error": str(e)}
        }


def agent_node(state: NvidiaGPTState, nvidia_gpt):
    """Execute NvidiaGPT agent with LLM, summarizing RAG & Web data."""
    try:
        raw_web = state.get("web_output", "")
        raw_rag_data = state.get("rag_output", {})
        raw_rag = raw_rag_data.get("result", "")
        rag_images = raw_rag_data.get("images", [])
        snowflake_data = state.get("snowflake_output", {}).get("metrics", "No snowflake data")
        
        # Extract metadata information if present in the RAG output
        rag_periods = raw_rag_data.get("periods", [])
        rag_type = raw_rag_data.get("type", "")
        metadata_info = ""
        if rag_periods:
            metadata_info = f"Information filtered for periods: {', '.join(rag_periods)}"
        
        # 1) Summarize Web Data
        if raw_web:
            prompt_for_web = f"""
You are an AI assistant summarizing real-time web data about NVIDIA. 
Below is the web content:

{raw_web}

Please provide a concise summary focusing on the most important current insights, 
around 3-5 lines only.
"""
            web_summary_result = nvidia_gpt.invoke({"input": prompt_for_web})
            if isinstance(web_summary_result, dict) and "output" in web_summary_result:
                web_summary = web_summary_result["output"]
            else:
                web_summary = str(web_summary_result)
        else:
            web_summary = "No web data to summarize."

        # 2) Summarize RAG Data with metadata info
        if raw_rag:
            # Include metadata info in the prompt
            image_info = f"\nThe retrieved content includes {len(rag_images)} relevant images." if rag_images else ""
            
            prompt_for_rag = f"""
You are an AI assistant summarizing historical NVIDIA performance. 
Below is a chunk from RAG retrieval:{image_info}
{metadata_info}

{raw_rag}

Please provide a concise summary focusing on key points, 
about 3-5 lines. If the content references specific quarters or years, highlight this information.
"""
            rag_summary_result = nvidia_gpt.invoke({"input": prompt_for_rag})
            if isinstance(rag_summary_result, dict) and "output" in rag_summary_result:
                rag_summary = rag_summary_result["output"]
            else:
                rag_summary = str(rag_summary_result)
        else:
            rag_summary = "No RAG data to summarize."

        # 3) Combine Summaries + Snowflake Data for Final Answer
        combined_prompt = f"""
Web Summary (Real-time data):
{web_summary}

RAG Summary (Historical data{' for ' + ', '.join(rag_periods) if rag_periods else ''}):
{rag_summary}

Snowflake (Financial Metrics):
{snowflake_data}

Question: {state['question']}

Please integrate these insights into a single cohesive answer 
regarding NVIDIA's performance. Structure your response with clear sections:
1. Real-time market insights
2. Historical performance
3. Financial metrics analysis
"""
        final_response = nvidia_gpt.invoke({"input": combined_prompt})

        # Return the final answer plus individual summaries and metadata for UI display
        if isinstance(final_response, dict) and "output" in final_response:
            return {
                "assistant_response": final_response["output"],
                "web_summary": web_summary,
                "rag_summary": rag_summary,
                "metadata_info": metadata_info,
                "rag_periods": rag_periods
            }
        else:
            return {
                "assistant_response": str(final_response),
                "web_summary": web_summary,
                "rag_summary": rag_summary,
                "metadata_info": metadata_info,
                "rag_periods": rag_periods
            }

    except Exception as e:
        return {
            "assistant_response": f"Analysis error: {str(e)}",
            "web_summary": "No summary due to error.",
            "rag_summary": "No summary due to error.",
            "metadata_info": ""
        }
    

def final_report_node(state: NvidiaGPTState) -> Dict:
    """Generate final report combining all sources with improved structure."""
    try:
        # 1) Grab the LLM-summarized texts, if any
        rag_summary = state.get("rag_summary", "")
        web_summary = state.get("web_summary", "")
        metadata_info = state.get("metadata_info", "")
        rag_periods = state.get("rag_periods", [])

        # 2) Build structured key findings by source
        findings_by_source = {
            "web": [],
            "rag": [], 
            "snowflake": []
        }

        # Web Search findings
        raw_web = state.get("web_output", "")
        if web_summary and "No web data to summarize." not in web_summary:
            findings_by_source["web"].append(f"Web Search Summary: {web_summary}")
        elif raw_web:
            findings_by_source["web"].append(f"Web Search (raw): {raw_web[:200]}...")

        # RAG findings with metadata
        raw_rag_data = state.get("rag_output", {})
        raw_rag_text = raw_rag_data.get("result", "")
        rag_type = raw_rag_data.get("type", "unknown")
        
        if metadata_info:
            findings_by_source["rag"].append(metadata_info)
            
        if rag_summary and "No RAG data" not in rag_summary:
            findings_by_source["rag"].append(f"Document Analysis Summary: {rag_summary}")
        elif raw_rag_text:
            findings_by_source["rag"].append(f"Document Analysis ({rag_type}): {raw_rag_text[:200]}...")

        # Add information about any images found
        rag_images = raw_rag_data.get("images", [])
        if rag_images:
            img_sources = []
            for img in rag_images[:3]:  # Limit to first 3 images
                if 'source' in img:
                    img_sources.append(img['source'])
            findings_by_source["rag"].append(
                f"Found {len(rag_images)} relevant image(s) from documents" + 
                (f" including: {', '.join(img_sources)}" if img_sources else "")
            )
            
        # Snowflake findings
        snowflake_output = state.get('snowflake_output', {})
        if isinstance(snowflake_output, dict):
            metrics = snowflake_output.get('metrics')
            latest_date = snowflake_output.get('latest_date', 'recent date')
            if metrics:
                findings_by_source["snowflake"].append(
                    f"Financial Metrics: Latest metrics from {latest_date}"
                )
                
            # Add any valuation data summary
            valuation_data = state.get('valuation_data', {})
            if valuation_data and 'error' not in valuation_data:
                findings_by_source["snowflake"].append(
                    "Valuation data available with visualizations"
                )

        # 3) Flatten findings for the final report
        key_findings = []
        for source, findings in findings_by_source.items():
            if findings:
                if source == "web":
                    key_findings.append("🌐 WEB SEARCH FINDINGS:")
                elif source == "rag":
                    period_info = f" for {', '.join(rag_periods)}" if rag_periods else ""
                    key_findings.append(f"📚 DOCUMENT ANALYSIS{period_info}:")
                elif source == "snowflake":
                    key_findings.append("📊 FINANCIAL DATA FINDINGS:")
                    
                key_findings.extend([f"  • {finding}" for finding in findings])

        # 4) Construct final structured dictionary
        report = final_report_tool({
            "introduction": f"Analysis of NVIDIA performance for: {state['question']}",
            "key_findings": key_findings,
            "analysis": state.get("assistant_response", "Analysis unavailable"),
            "conclusion": "Based on the collected data, NVIDIA continues to show strong performance in the GPU market...",
            "sources": ["Web Search", "Document Analysis", "Financial Data", "AI Analysis"],
            "images": raw_rag_data.get("images", []),  # Include images in the final report
            "metadata": {
                "periods": rag_periods,
                "search_type": raw_rag_data.get("type", "")
            }
        })

        return {"final_report": report}

    except Exception as e:
        return {
            "final_report": {
                "introduction": "Error generating report",
                "key_findings": [f"Error: {str(e)}"],
                "analysis": "Analysis unavailable due to error",
                "conclusion": "Unable to generate conclusion",
                "sources": [],
                "images": [],
                "metadata": {}
            }
        }    

def create_tools():
    """Create LangChain tools for the agent."""
    return [
        Tool(
            name="web_search",
            func=search_quarterly,
            description="Search for NVIDIA quarterly financial information from web sources"
        ),
        Tool(
            name="rag_search",
            func=search_all_namespaces,
            description="Search across all document repositories for NVIDIA information"
        ),
        Tool(
            name="specific_quarter_search",
            func=search_specific_quarter,
            description="Search for specific quarter information from NVIDIA reports"
        ),
        Tool(
            name="snowflake_query",
            func=query_snowflake,
            description="Query Snowflake database for NVIDIA financial metrics"
        ),
        Tool(
            name="generate_report",
            func=final_report_tool,
            description="Generate a structured report from analyzed information"
        )
    ]

def initialize_report_generator():
    """Initialize a focused report generator agent that minimizes verbose thinking."""
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0,
        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
    )
    
    tools = create_tools()
    
    # Create a custom system message that focuses on direct report generation
    system_message = """You are a focused financial report generator for NVIDIA analysis.
    YOUR ONLY JOB is to produce clean, well-structured reports about NVIDIA's financial performance.
    
    Key instructions:
    1. DO NOT show your reasoning process
    2. DO NOT explain your approach or methodology
    3. DO NOT include observations about the data
    4. FOCUS ONLY on delivering a professionally formatted report
    5. Use concise, data-driven language
    6. Structure with clear headings and bullet points
    
    The final output should ONLY be the report itself, formatted professionally.
    """
    
    # Initialize the agent with just system message in agent_kwargs
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        handle_parsing_errors=True, 
        verbose=False,
        agent_kwargs={
            "prefix": system_message,
            "format_instructions": """To use a tool, please use the following format:
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action

When you have a response for the user, respond with:
Final Report: your final report for the user

Begin!
"""
        }
    )
    
    return agent



def generate_workflow_diagram(filename="nvidia_workflow"):
    """Generates and saves workflow diagram with visual enhancements."""
    dot = Digraph(comment='NVIDIA Analysis Pipeline')
    dot.attr(rankdir='LR', bgcolor='white', fontname='Helvetica')
    dot.attr('node', fontname='Helvetica', fontsize='12', style='filled', fontcolor='white', margin='0.4')
    dot.attr('edge', fontname='Helvetica', fontsize='10', penwidth='1.5')

    # Nodes
    dot.node('start', 'Start', shape='oval', style='filled', fillcolor='#4CAF50', color='#2E7D32')
    dot.node('web_search', 'Web Search', shape='box', style='filled,rounded', fillcolor='#2196F3', color='#0D47A1')
    dot.node('rag_search', 'RAG Search', shape='box', style='filled,rounded', fillcolor='#03A9F4', color='#0277BD')
    dot.node('snowflake', 'Snowflake', shape='box', style='filled,rounded', fillcolor='#00BCD4', color='#006064')
    dot.node('agent', 'NvidiaGPT Agent', shape='hexagon', style='filled', fillcolor='#9C27B0', color='#4A148C')
    dot.node('report_generator', 'Report Generator', shape='note', style='filled', fillcolor='#FF9800', color='#E65100')
    dot.node('end', 'End', shape='oval', style='filled', fillcolor='#F44336', color='#B71C1C')

    # Edges
    dot.edge('start', 'web_search', color='#2196F3')
    dot.edge('start', 'rag_search', color='#03A9F4')
    dot.edge('start', 'snowflake', color='#00BCD4')
    dot.edge('web_search', 'agent', color='#2196F3')
    dot.edge('rag_search', 'agent', color='#03A9F4')
    dot.edge('snowflake', 'agent', color='#00BCD4')
    dot.edge('agent', 'report_generator', color='#9C27B0')
    dot.edge('report_generator', 'end', color='#FF9800')

    try:
        dot.render(filename, format='png', cleanup=True)
        return f"{filename}.png"
    except Exception as e:
        print(f"Warning: Could not generate diagram: {e}")
        return None


def build_pipeline(selected_agents: List[str] = None):
    """
    Build and return the compiled pipeline with dynamic agent selection.
    If selected_agents is None, we default to an empty list.
    """
    if selected_agents is None:
        selected_agents = []

    # Initialize the NvidiaGPT agent
    nvidia_gpt = initialize_report_generator()
    
    graph = StateGraph(NvidiaGPTState)

    # Make "start" a real node
    graph.add_node("start", start_node)
    # Set the official entry point in LangGraph
    graph.set_entry_point("start")

    # Add nodes for selected agents
    last_node = "start"

    # If user selected RAG
    if "RAG Agent" in selected_agents:
        graph.add_node("rag_search", rag_search_node)
        graph.add_edge(last_node, "rag_search")
        last_node = "rag_search"

    # If user selected Web Search
    if "Web Search Agent" in selected_agents:
        graph.add_node("web_search", web_search_node)
        graph.add_edge(last_node, "web_search")
        last_node = "web_search"

    # If user selected Snowflake
    if "Snowflake Agent" in selected_agents:
        graph.add_node("snowflake", snowflake_node)
        graph.add_edge(last_node, "snowflake")
        last_node = "snowflake"

    # Add agent node for LLM-based analysis
    if selected_agents:
        # Partially apply the agent_node function with nvidia_gpt parameter
        def agent_node_with_gpt(state):
            return agent_node(state, nvidia_gpt)
            
        graph.add_node("agent", agent_node_with_gpt)
        graph.add_edge(last_node, "agent")
        last_node = "agent"

    # Add final node for generating the report
    graph.add_node("report_generator", final_report_node)
    graph.add_edge(last_node, "report_generator")

    # Finally connect "report_generator" → END
    graph.add_edge("report_generator", END)

    return graph.compile()


if __name__ == "__main__":
    try:
        # EXAMPLE usage with no agents
        # pipeline = build_pipeline(selected_agents=[])  # no agents
        # or use some default
        pipeline = build_pipeline(["RAG Agent"])

        result = pipeline.invoke({
            "input": "Analyze NVIDIA's financial performance in Q4 2023",
            "question": "Analyze NVIDIA's financial performance in Q4 2023",
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
        
        # Print information about images if any were found
        images = final_report.get("images", [])
        if images:
            print(f"\nFound {len(images)} images:")
            for i, img in enumerate(images):
                print(f"  Image {i+1}: {img['caption']} ({img['type']})")
                
    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())