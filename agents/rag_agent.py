import logging
from langchain_core.tools import tool
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Dict, Any, Union, Optional
from pinecone import Pinecone
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import boto3

# Initialize environment and configurations
load_dotenv(override=True)

SYSTEM_CONFIG = {
    "CURRENT_UTC": "2025-04-02 06:21:03",
    "CURRENT_USER": "user",
    "MIN_YEAR": 1995,
    "MAX_YEAR": 2018
}

# Initialize services
encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "crime-reports"))

# Import the LLMSelector from your llmselection file
from agents.llmselection import LLMSelector as llmselection

class SearchCrimeDataInput(BaseModel):
    tool_input: Dict[str, Any] = Field(..., description="The input parameters for the search")

class RAGAgent:
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        # Initialize the LLM using the LLMSelector
        self.initialize_agent(model_name)
        self.prompt = PromptTemplate.from_template("""
            You are a crime analysis expert. Based on the provided data, create a comprehensive analysis.
            
            SEARCH RESULTS:
            {context}
            
            QUERY:
            {query}
            
            Provide a detailed report with:
            1. Executive Summary
            2. Incident Details
            3. Evidence & Leads
            4. Context & Implications
            5. Recommendations
            
            ANALYSIS:
        """)
    
    def initialize_agent(self, model_name: str):
        """
        Initialize the LLM by retrieving it from the LLMSelector.
        This ensures that the correct model is used.
        """
        self.llm = llmselection.get_llm(model_name)
    
    def process(self, query: str, context: str) -> str:
        """
        Process the search results and return an analysis.
        If no quality context is provided, a prompt to refine the search is returned.
        """
        if not context or context.startswith("No results found"):
            return "No relevant information found. Please refine your search."
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context})

def get_chunk_from_s3(chunk_s3_path: str, chunk_index: int, s3_bucket: str) -> str:
    """Retrieve specific chunk data from S3 with improved error handling."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
            aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        response = s3_client.get_object(
            Bucket=s3_bucket,
            Key=chunk_s3_path
        )
        chunks_data = json.loads(response['Body'].read().decode('utf-8'))
        chunk_text = chunks_data.get(str(chunk_index), "")
        if len(chunk_text.strip()) < 100 or chunk_text.strip().startswith('#'):
            return ""
        return chunk_text
    except Exception as e:
        logging.error(f"Error retrieving chunk from S3: {e}")
        return ""

def rerank_results(query: str, results: list, top_k: int = 15) -> list:
    """Re-rank results using the cross-encoder for higher relevance."""
    if not results:
        return []
    passages = []
    for match in results:
        metadata = match['metadata']
        chunk_text = ""
        if metadata.get('chunks_s3_path') and metadata.get('chunk_index'):
            chunk_text = get_chunk_from_s3(
                chunk_s3_path=metadata['chunks_s3_path'],
                chunk_index=metadata['chunk_index'],
                s3_bucket=metadata.get('s3_bucket', 'crime-records')
            )
        text = chunk_text if chunk_text else metadata.get('text_preview', '').strip()
        if text:
            passages.append((match, text))
    if not passages:
        return []
    pairs = [[query, p[1]] for p in passages]
    scores = cross_encoder.predict(pairs)
    passage_scores = [(passages[i][0], score) for i, score in enumerate(scores)]
    reranked_results = sorted(passage_scores, key=lambda x: x[1], reverse=True)
    return [(item[0], float(item[1])) for item in reranked_results[:top_k]]

def format_results(matches: list) -> str:
    """Format search results with improved chunk data retrieval."""
    results = []
    for match, score in matches:
        metadata = match['metadata']
        chunk_text = ""
        if metadata.get('chunks_s3_path') and metadata.get('chunk_index'):
            chunk_text = get_chunk_from_s3(
                chunk_s3_path=metadata['chunks_s3_path'],
                chunk_index=metadata['chunk_index'],
                s3_bucket=metadata.get('s3_bucket', 'crime-records')
            )
        description = chunk_text if chunk_text else metadata.get('text_preview', '').strip()
        if len(description.strip()) < 50:
            continue
        results.append("\n".join([
            f"Year: {metadata.get('year', 'Unknown')}",
            f"Document ID: {metadata.get('document_id', 'Unknown')}",
            f"Description: {description}",
            f"Score: {score:.3f}\n"
        ]))
    return "\n---\n".join(results)

@tool("search_crime_data")
def search_crime_data(tool_input: Dict[str, Any]) -> Union[str, Dict]:
    """Search crime report data with improved retrieval and ranking."""
    try:
        # Extract query parameters
        query = tool_input.get("query")
        search_mode = tool_input.get("search_mode", "all_years")
        start_year = tool_input.get("start_year")
        end_year = tool_input.get("end_year")
        model_type = tool_input.get("model_type", "claude-3-haiku-20240307")
        
        if not query:
            return "Error: Query is required"
        
        # Encode query for vector search
        xq = encoder.encode([query])[0].tolist()
        
        # Determine year range based on search mode
        years_range = range(
            start_year or SYSTEM_CONFIG["MIN_YEAR"],
            (end_year or SYSTEM_CONFIG["MAX_YEAR"]) + 1
        ) if search_mode == "specific_range" else range(
            SYSTEM_CONFIG["MIN_YEAR"], 
            SYSTEM_CONFIG["MAX_YEAR"] + 1
        )
        
        # Collect initial results from each year namespace
        initial_results = []
        for year in years_range:
            try:
                response = index.query(
                    vector=xq,
                    top_k=10,
                    include_metadata=True,
                    namespace=str(year),
                    alpha=0.5
                )
                if response.get("matches"):
                    initial_results.extend(response["matches"])
            except Exception as e:
                logging.error(f"Error searching year {year}: {e}")
                continue
        
        if not initial_results:
            return "No results found for the specified time period."
        
        # Re-rank results using the cross-encoder
        reranked_results = rerank_results(query, initial_results, top_k=15)
        
        if not reranked_results:
            return "No quality results found after filtering."
            
        # Format results using improved chunk retrieval
        formatted_contexts = format_results(reranked_results)
        
        # Use RAG agent with the selected model to process the results
        agent = RAGAgent(model_name=model_type)
        insights = agent.process(query, formatted_contexts)
        
        return {
            "raw_contexts": formatted_contexts,
            "insights": insights,
            "metadata": {
                "query": query,
                "search_mode": search_mode,
                "time_range": f"{min(years_range)}-{max(years_range)}",
                "timestamp": SYSTEM_CONFIG["CURRENT_UTC"],
                "user": SYSTEM_CONFIG["CURRENT_USER"],
                "result_count": len(reranked_results)
            }
        }
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return f"Error performing search: {str(e)}"

# Test the tool if run directly
if __name__ == "__main__":
    test_queries = [
        {
            "tool_input": {
                "query": "Spike in vehicle theft incidents",
                "search_mode": "specific_range",
                "start_year": 2000,
                "end_year": 2005
            }
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query['tool_input']['search_mode']}")
        try:
            result = search_crime_data.invoke(query)
            if isinstance(result, dict):
                print(f"Success: Found {result['metadata']['result_count']} results")
                filename = f"crime_report_{query['tool_input']['search_mode']}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to {filename}")
                print(f"Insights: {result['insights']}")
            else:
                print(f"Error: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
