"""
Ask User Clarification Tool

LangChain Tool for asking users clarification questions when
their request is ambiguous or lacks necessary information.

This tool triggers an interrupt in the agent execution, allowing
the frontend to display the question and collect user input.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.types import interrupt


class AskUserInput(BaseModel):
    """Input schema for ask user clarification tool."""
    
    question: str = Field(
        description="The clarification question to ask the user"
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional context explaining why this clarification is needed"
    )


# Callback for handling user input (for non-LangGraph usage)
_user_input_callback: Optional[callable] = None


def set_user_input_callback(callback: callable) -> None:
    """
    Set a callback function for handling user input.
    
    This is used when running outside of LangGraph's interrupt mechanism.
    
    Args:
        callback: Function that takes a question and returns user's answer
    """
    global _user_input_callback
    _user_input_callback = callback


@tool("ask_user_clarification", args_schema=AskUserInput)
def ask_user_clarification(question: str, context: Optional[str] = None) -> str:
    """
    Ask the user a clarification question to gather more information.
    
    Use this tool when:
    - The user's question is ambiguous or unclear
    - Important details are missing (e.g., time range, specific filters)
    - You need to confirm assumptions before proceeding
    - Multiple interpretations are possible
    
    Examples of when to use this tool:
    - "Show me recent sales" → Ask about specific date range
    - "What's the best product?" → Ask about criteria (revenue, quantity, rating)
    - "Compare departments" → Ask which departments to compare
    
    Args:
        question: The clarification question to ask the user
        context: Optional explanation of why this information is needed
        
    Returns:
        The user's response to your question
    """
    global _user_input_callback
    
    # Build the full message
    if context:
        full_message = f"{context}\n\n{question}"
    else:
        full_message = question
    
    # Try to use LangGraph interrupt mechanism
    try:
        # This will pause the agent and wait for user input
        user_response = interrupt({
            "type": "clarification_request",
            "question": question,
            "context": context,
            "full_message": full_message,
        })
        return user_response
    except Exception:
        # Fallback to callback if available
        if _user_input_callback is not None:
            return _user_input_callback(full_message)
        
        # If no callback, return a placeholder indicating clarification needed
        return f"[CLARIFICATION_NEEDED] {full_message}"


class ClarificationState:
    """
    State manager for handling clarification requests in web UI.
    
    This class helps track pending clarification requests and
    resume agent execution after user provides input.
    """
    
    def __init__(self):
        self._pending_requests: Dict[str, Dict[str, Any]] = {}
    
    def add_request(
        self,
        session_id: str,
        question: str,
        context: Optional[str] = None,
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a pending clarification request.
        
        Args:
            session_id: The conversation session ID
            question: The clarification question
            context: Optional context
            agent_state: The agent's state at time of interrupt
            
        Returns:
            Request ID for tracking
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        self._pending_requests[request_id] = {
            "session_id": session_id,
            "question": question,
            "context": context,
            "agent_state": agent_state,
            "status": "pending",
        }
        
        return request_id
    
    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get a pending request by ID."""
        return self._pending_requests.get(request_id)
    
    def resolve_request(self, request_id: str, user_response: str) -> bool:
        """
        Resolve a pending request with user's response.
        
        Args:
            request_id: The request ID
            user_response: The user's answer
            
        Returns:
            True if request was found and resolved
        """
        if request_id in self._pending_requests:
            self._pending_requests[request_id]["response"] = user_response
            self._pending_requests[request_id]["status"] = "resolved"
            return True
        return False
    
    def get_pending_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get pending request for a session."""
        for req_id, req in self._pending_requests.items():
            if req["session_id"] == session_id and req["status"] == "pending":
                return {"request_id": req_id, **req}
        return None


# Global clarification state manager
clarification_state = ClarificationState()
