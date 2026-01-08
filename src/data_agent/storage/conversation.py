"""
Conversation Storage Module

Provides conversation history storage for multi-turn dialogue support.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid


class Message(BaseModel):
    """Single message in a conversation."""
    
    role: str = Field(description="Message role (user/assistant/system)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # SQL generation specific
    sql: Optional[str] = Field(default=None, description="Generated SQL if applicable")


class Conversation(BaseModel):
    """Conversation containing multiple messages."""
    
    id: str = Field(description="Unique conversation identifier")
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def get_history_for_llm(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM context.
        
        Args:
            max_turns: Maximum number of recent turns to include
            
        Returns:
            List of message dicts with role and content
        """
        # Get recent messages (limit turns)
        recent = self.messages[-max_turns * 2:] if max_turns else self.messages
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent
        ]
    
    def get_context_summary(self) -> str:
        """Get a summary of conversation context for SQL generation."""
        if not self.messages:
            return ""
        
        parts = ["## Conversation Context\n"]
        parts.append("Previous exchanges in this conversation:\n")
        
        for msg in self.messages[-6:]:  # Last 3 turns
            role_label = "User" if msg.role == "user" else "Assistant"
            content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            parts.append(f"- **{role_label}**: {content_preview}\n")
            if msg.sql:
                parts.append(f"  - SQL: `{msg.sql[:100]}...`\n" if len(msg.sql) > 100 else f"  - SQL: `{msg.sql}`\n")
        
        return "".join(parts)


class ConversationStore:
    """In-memory conversation storage."""
    
    def __init__(self):
        self._conversations: Dict[str, Conversation] = {}
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        
        conversation = Conversation(id=conversation_id)
        self._conversations[conversation_id] = conversation
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self._conversations.get(conversation_id)
    
    def get_or_create(self, conversation_id: Optional[str] = None) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        return self.create_conversation(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False
    
    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        """List all conversations, sorted by updated_at."""
        conversations = list(self._conversations.values())
        conversations.sort(key=lambda x: x.updated_at, reverse=True)
        return conversations[:limit]


class Session:
    """
    Conversation session for multi-turn dialogue.
    
    Example:
        session = vn.create_session()
        result1 = session.ask("鏈夊灏戝鎴凤紵")
        result2 = session.ask("浠栦滑涓湁澶氬皯鏄湰鏈堟敞鍐岀殑锛?)
    """
    
    def __init__(
        self,
        vanna_instance,
        conversation: Conversation,
    ):
        self._vanna = vanna_instance
        self._conversation = conversation
    
    @property
    def conversation_id(self) -> str:
        """Get the conversation ID."""
        return self._conversation.id
    
    @property
    def history(self) -> List[Message]:
        """Get conversation history."""
        return self._conversation.messages
    
    def ask(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Ask a question in this conversation context.
        
        Args:
            question: Natural language question
            
        Returns:
            Result dict with sql, answer, etc.
        """
        return self._vanna.ask(
            question,
            conversation_id=self._conversation.id,
            **kwargs,
        )
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL in this conversation context."""
        return self._vanna.generate_sql(
            question,
            conversation_id=self._conversation.id,
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation.messages.clear()
        self._conversation.updated_at = datetime.utcnow()
    
    def get_context(self) -> str:
        """Get conversation context summary."""
        return self._conversation.get_context_summary()
