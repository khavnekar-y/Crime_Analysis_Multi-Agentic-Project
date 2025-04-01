import logging
from langchain_core.tools import tool
import boto3
import json
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Union
from pinecone import Pinecone
from PIL import Image
import time
import base64
from io import BytesIO
import re
import os
from sentence_transformers import SentenceTransformer
# add imports for the agents
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv(override=True)
# Initialize services
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_SERVER_PUBLIC_KEY"),
    aws_secret_access_key=os.getenv("AWS_SERVER_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION")
)
encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "nvidia-reports"))

class RAGAgent:
    def __init__(self, model_name):
        """Initialize the RAG Agent with a language model."""
        # Initialize the appropriate LLM based on the model name
        if "claude" in model_name:
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=0,
                anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        elif "gemini" in model_name:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=os.environ.get('GOOGLE_API_KEY')
            )
        elif "deepseek" in model_name:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=os.environ.get('DEEP_SEEK_API_KEY')
            )
        elif "grok" in model_name:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(
                model=model_name,
                temperature=0,
                api_key=os.environ.get('GROK_API_KEY')
            )
        else:
            self.llm = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0,
                anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
            )
        self.prompt = PromptTemplate.from_template("""
        You are a financial analyst specialized in analyzing NVIDIA quarterly reports and financial data.
        Based on the provided quarterly report excerpts, create a comprehensive analysis.

        SEARCH RESULTS:
        {context}

        QUERY:
        {query}

        Please provide a detailed report following this structure:
        1. Executive Summary
           - Key findings related to the query
           - Overall financial health indicators
        
        2. Financial Metrics Analysis
           - Revenue and growth trends
           - Segment performance
           - Key financial ratios (if available)
        
        3. Business Highlights
           - Major developments
           - Product launches or technological advancements
           - Strategic initiatives
        
        4. Market Position & Competition
           - Market share information
           - Competitive advantages
           - Industry trends
        
        5. Future Outlook
           - Company guidance
           - Growth opportunities
           - Potential challenges

        Summarize the information in a professional, analytical tone, focusing on the most relevant data points 
        from the provided quarterly reports.

        ANALYSIS:
    """)
    
    def process(self,query:str,context:str) -> str:
        """Process the search results and provide insights."""
        if not context or context.startswith("No results found"):
            return "No relevant information found for your query.Try refining your search terms or exploring different time periods."
        #Create a chain to process the search results
        chain = self.prompt | self.llm | StrOutputParser()
        # run the chain with the search results 
        result = chain.invoke({
            "query" : query,
            "context" : context
        })
        return result

def clean_base64_images(text: Any) -> str:
    """
    Remove base64 encoded images from text for cleaner content.
    
    Args:
        text: Text containing base64 encoded images
    
    Returns:
        Cleaned text with base64 image data removed
    """
    # Check if text is a string
    if not isinstance(text, str):
        return str(text) if text is not None else ""
        
    if not text:
        return ""
    
    # Pattern to find markdown image syntax with base64 data
    pattern = r'(?:!\[.*?\]|\[Image.*?\])\(data:image\/[^;]+;base64,[^)]+\)'
    
    # Replace base64 images with a placeholder
    cleaned_text = re.sub(pattern, '[IMAGE REMOVED]', text)
    
    return cleaned_text

def get_content_from_s3(json_source: str) -> Dict:
    """Retrieve content from S3 bucket"""
    bucket = json_source.split('/')[2]
    key = '/'.join(json_source.split('/')[3:])
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = json.loads(response['Body'].read().decode('utf-8'))
        return content
    except Exception as e:
        print(f"Error retrieving from S3: {e}")
        return {}

def rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """Rerank results using cross-encoder"""
    if not results:
        return []
    
    # Prepare pairs for reranking
    pairs = [(query, result['metadata']['text_preview']) for result in results]
    
    # Get scores from cross-encoder
    scores = cross_encoder.predict(pairs)
    
    # Combine results with new scores
    for result, score in zip(results, scores):
        result['score'] = float(score)
    
    # Sort by new scores and return top_k
    reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    return reranked_results

def format_rag_contexts(matches: List[Dict]) -> str:
    contexts = []
    
    for x in matches:
        # Get full content from S3 if available
        if 'json_source' in x['metadata']:
            full_content = get_content_from_s3(x['metadata']['json_source'])
            chunk_content = full_content.get(x['metadata']['chunk_id'], '')
        else:
            chunk_content = x['metadata'].get('text_preview', '')
        
        # Ensure chunk_content is a string and clean base64 images
        if not isinstance(chunk_content, str):
            chunk_content = str(chunk_content) if chunk_content is not None else ""
        
        # Clean base64 images from the content
        cleaned_content = clean_base64_images(chunk_content)
        
        text = (
            f"File Name: {x['metadata']['file_name']}\n"
            f"Year: {x['metadata']['year']}\n"
            f"Quarter: {x['metadata']['quarter']}\n"
            f"Content: {cleaned_content}\n"
            f"Source: {x['metadata']['source']}\n"
            f"Relevance Score: {x['score']:.3f}\n"
        )
        contexts.append(text)
    
    return "\n---\n".join(contexts)

@tool("search_all_namespaces")
def search_all_namespaces(query: str, alpha: float = 0.5,model_name:str = "claude-3-haiku-20240307"):
    """
    Searches across all quarterly report namespaces using hybrid search.
    """
    print(f"\nSearching for query: {query}")
    results = []
    xq = encoder.encode([query])[0].tolist()
    
    namespaces = [f"{year}q{quarter}" for year in range(2023, 2025) 
                 for quarter in range(1, 5)]
    
    print(f"Searching across namespaces: {namespaces}")
    
    for namespace in namespaces:
        try:
            xc = index.query(
                vector=xq,
                top_k=5,
                include_metadata=True,
                namespace=namespace,
                alpha=alpha,
            )
            if xc["matches"]:
                results.extend(xc["matches"])
            else:
                logging.info(f"No results found in namespace {namespace}.")
        except Exception as e:
            logging.error(f"Error searching namespace {namespace}: {str(e)}")
            continue
    
    print(f"\nTotal results found: {len(results)}")
    
    if results:
        results = rerank_results(query, results)
        formatted_contexts = format_rag_contexts(results)
        # Initialize and use the agent to process results
        agent = RAGAgent(model_name=model_name) 
        processed_result = agent.process(query, formatted_contexts)
        return{
            "raw_contexts":formatted_contexts,
            "insights": processed_result
        }
    else:
        return "No results found across any namespace."
    
@tool("search_specific_quarter")
def search_specific_quarter(input_dict: Dict) -> str:
    """
    Searches in specific quarterly report namespaces using hybrid search.
    Args:
        input_dict: Dictionary containing query and selected periods
    """
    if isinstance(input_dict, str) and "input_dict" in input_dict:
        input_dict = json.loads(input_dict)
    query = input_dict.get("query")
    selected_periods = input_dict.get("selected_periods", ["2023q1"]) 
    if not query:
        return "Error: No query provided."
    results = []
    
    # Encode query once for all searches
    xq = encoder.encode([query])[0].tolist()
    
    for period in selected_periods:
        try:
            # Query each selected namespace
            xc = index.query(
                vector=xq,
                top_k=5,
                include_metadata=True,
                namespace=period,
                alpha=0.5
            )
            if xc["matches"]:
                results.extend(xc["matches"])
        except Exception as e:
            logging.error(f"Error searching namespace {period}: {str(e)}")
    
    # Rerank combined results
    if results:
        results = rerank_results(query, results)
        formatted_contexts = format_rag_contexts(results)
        model_name = input_dict.get("model_name", "claude-3-haiku-20240307")
        agent = RAGAgent(model_name=model_name)
        process_results = agent.process(query,formatted_contexts)
        return {
            "raw_contexts": formatted_contexts,
            "insights": process_results
        }
    return "No results found in selected quarters."

    
if __name__ == "__main__":
    # Test search_all_namespaces with image extraction
    query = "NVIDIA GPU architecture diagrams"
    result = search_all_namespaces.invoke(query)
    if isinstance(result, dict):
        print("\n=== Search Results with ===")
        print(result["raw_contexts"])
        print("\n=== AI Agent Insights ===")
        print(result["insights"])
    else:
        print(result["raw_contexts"])
        
    # Test searching across multiple periods
    test_query = "NVIDIA revenue growth charts"
    # Define multiple periods to search across
    test_periods = ["2023q1", "2023q2", "2024q1"]
    
    input_dict = {
        "query": test_query,
        "selected_periods": test_periods
    }
    
    # specific_result = search_specific_quarter.invoke(input_dict)
    
    # print("\n=== Multiple Quarter Results with Images ===")
    # print(specific_result["raw_contexts"] if isinstance(specific_result, dict) else specific_result)
    # print("\n=== AI Agent Insights ===")
    # print(specific_result["insights"] if isinstance(specific_result, dict) else "")