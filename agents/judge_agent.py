from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import json
from datetime import datetime
from typing import Dict, List
from agents.llmselection import LLMSelector as llmselection

class JudgeAgent:
    def __init__(self, model_type: str = "Claude 3 Sonnet"):
        self.model_type = model_type
        self.llm = llmselection.get_llm(model_type)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="evaluation_history",
            return_messages=True,
            input_key="report",
            output_key="evaluation"
        )
        
        # Initialize tools
        self.tools = [
            Tool(
                name="evaluate_accuracy",
                func=self._evaluate_accuracy,
                description="Evaluate factual accuracy and correctness"
            ),
            Tool(
                name="evaluate_completeness",
                func=self._evaluate_completeness,
                description="Evaluate report comprehensiveness"
            ),
            Tool(
                name="compare_with_previous",
                func=self._compare_with_previous,
                description="Compare with previous report evaluations"
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
        
        # Store feedback history
        self.feedback_history = []
    
    def evaluate(self, report_data: Dict, state: Dict) -> Dict:
        """Evaluate report quality and provide feedback."""
        try:
            # Extract relevant data
            sections = report_data.get("sections", [])
            query = state.get("question", "")
            
            # Check previous evaluations in memory
            previous_feedback = self._get_relevant_feedback(state)
            
            # Create evaluation context
            eval_context = {
                "report": report_data,
                "query": query,
                "previous_feedback": previous_feedback,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store evaluation context in memory
            self.memory.save_context(
                {"report": str(eval_context)},
                {"evaluation": "Starting evaluation..."}
            )
            
            # Execute evaluation
            evaluation = self._run_evaluation(eval_context)
            
            # Store feedback for future reference
            self._store_feedback(evaluation, state)
            
            return evaluation
            
        except Exception as e:
            print(f"JudgeAgent evaluation error: {str(e)}")
            return self._create_error_response(str(e))
    def _evaluate_completeness(self, report_data: Dict) -> Dict:
        """Evaluate completeness of the report."""
        required_sections = [
            "executive_summary",
            "methodology",
            "analysis",
            "recommendations"
        ]
        
        completeness = {
            "score": 0,
            "missing_sections": [],
            "feedback": []
        }
        
        sections = report_data.get("sections", [])
        section_titles = [s.get("title", "").lower() for s in sections]
        
        for required in required_sections:
            if not any(required in title for title in section_titles):
                completeness["missing_sections"].append(required)
        
        completeness["score"] = 10 - (len(completeness["missing_sections"]) * 2)
        return completeness

    def _compare_with_previous(self, current_evaluation: Dict) -> List[str]:
        """Compare with previous evaluations from memory."""
        if not self.feedback_history:
            return ["No previous evaluations available"]
            
        previous = self.feedback_history[-1] if self.feedback_history else None
        if not previous:
            return ["First evaluation"]
            
        improvements = []
        current_score = current_evaluation.get("scores", {}).get("overall", 0)
        previous_score = previous.get("evaluation", {}).get("scores", {}).get("overall", 0)
        
        if current_score > previous_score:
            improvements.append(f"Overall score improved by {current_score - previous_score} points")
        elif current_score < previous_score:
            improvements.append(f"Overall score decreased by {previous_score - current_score} points")
            
        return improvements
    
    def _run_evaluation(self, context: Dict) -> Dict:
        """Run the actual evaluation using the agent."""
        # Create evaluation prompt
        eval_prompt = self._create_evaluation_prompt(context)
        
        # Get evaluation from agent
        response = self.agent.run(eval_prompt)
        
        # Process and structure the response
        try:
            evaluation = json.loads(response)
        except:
            evaluation = self._create_structured_evaluation(response)
        
        return evaluation
    
    def _evaluate_accuracy(self, report_data: Dict) -> Dict:
        """Evaluate factual accuracy of the report."""
        try:
            sections = report_data.get("sections", [])
            evaluation = {
                "score": 0,
                "feedback": [],
                "improvements": []
            }
            
            # Check for data citations and sources
            for section in sections:
                content = section.get("content", "")
                if content and isinstance(content, str):
                    # Score based on data references
                    data_references = len([line for line in content.split('\n') if any(term in line.lower() for term in ["data shows", "statistics indicate", "according to", "analysis reveals"])])
                    evaluation["score"] += min(data_references, 3)  # Max 3 points per section
                    
            # Normalize score to 1-10 range
            evaluation["score"] = min(max(evaluation["score"], 1), 10)
            return evaluation
        except Exception as e:
            return {"score": 5, "error": str(e)}
    
    def _store_feedback(self, evaluation: Dict, state: Dict) -> None:
        """Store feedback for future reference."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": state.get("question", ""),
            "regions": state.get("selected_regions", []),
            "model_type": state.get("model_type", ""),
            "evaluation": evaluation,
            "improvements_suggested": evaluation.get("improvements", {})
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update memory with structured feedback
        self.memory.save_context(
            {"feedback": json.dumps(feedback_entry)},
            {"stored": "Feedback stored successfully"}
        )
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get improvement suggestions based on feedback history."""
        if not self.feedback_history:
            return []
            
        # Analyze feedback history for common improvement areas
        improvements = []
        for entry in self.feedback_history[-5:]:  # Look at last 5 evaluations
            for improvement in entry.get("improvements_suggested", {}).values():
                improvements.append(improvement)
                
        return list(set(improvements))  # Remove duplicates