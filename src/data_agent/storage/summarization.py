"""
Conversation Summarization Module

Provides conversation context summarization for long conversations.
When context gets too long, this module compresses earlier messages
while preserving key information for SQL generation.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from data_agent.storage.conversation import Conversation, Message


class SummarizationConfig(BaseModel):
    """Configuration for conversation summarization."""
    
    # Token thresholds
    trigger_token_count: int = Field(
        default=8000,
        description="Trigger summarization when estimated tokens exceed this"
    )
    keep_recent_turns: int = Field(
        default=3,
        description="Number of recent turns to keep in full (not summarized)"
    )
    max_summary_tokens: int = Field(
        default=500,
        description="Maximum tokens for the summary"
    )


SUMMARIZATION_PROMPT = """Summarize the following conversation history between a user and an SQL assistant.

Focus on:
1. Key entities mentioned (table names, column names, filters applied)
2. Previous queries and their results (briefly)
3. User's analytical intent and any refinements made
4. Important context that would help with future queries

Conversation:
{conversation_text}

Provide a concise summary that captures the essential context for SQL generation.
"""


class ConversationSummarizer:
    """
    Manages conversation context with automatic summarization.
    
    When conversations grow too long, earlier messages are summarized
    to keep the context window manageable while preserving key information.
    
    Example:
        summarizer = ConversationSummarizer()
        context = await summarizer.get_context(conversation)
        # Returns either full history or summary + recent messages
    """
    
    def __init__(self, config: Optional[SummarizationConfig] = None):
        self.config = config or SummarizationConfig()
        self._cached_summaries: Dict[str, str] = {}
    
    async def get_context(
        self,
        conversation: Conversation,
        force_summarize: bool = False,
    ) -> str:
        """
        Get conversation context, summarizing if necessary.
        
        Args:
            conversation: The conversation to get context from
            force_summarize: Force summarization even if under threshold
            
        Returns:
            Context string suitable for LLM prompts
        """
        messages = conversation.messages
        
        if not messages:
            return ""
        
        # Estimate token count
        estimated_tokens = self._estimate_tokens(messages)
        
        if not force_summarize and estimated_tokens < self.config.trigger_token_count:
            # Under threshold, return full context
            return conversation.get_context_summary()
        
        # Need to summarize
        return await self._get_summarized_context(conversation)
    
    async def _get_summarized_context(self, conversation: Conversation) -> str:
        """Generate summarized context for long conversations."""
        from data_agent.providers import get_llm
        
        messages = conversation.messages
        keep_count = self.config.keep_recent_turns * 2  # Each turn = 2 messages
        
        if len(messages) <= keep_count:
            # Not enough messages to summarize
            return conversation.get_context_summary()
        
        # Split into messages to summarize and recent messages to keep
        messages_to_summarize = messages[:-keep_count]
        recent_messages = messages[-keep_count:]
        
        # Check cache
        cache_key = f"{conversation.id}:{len(messages_to_summarize)}"
        if cache_key in self._cached_summaries:
            summary = self._cached_summaries[cache_key]
        else:
            # Generate summary
            summary = await self._summarize_messages(messages_to_summarize)
            self._cached_summaries[cache_key] = summary
        
        # Combine summary with recent messages
        context_parts = ["## Conversation Context\n"]
        context_parts.append("### Summary of Earlier Conversation\n")
        context_parts.append(summary)
        context_parts.append("\n\n### Recent Exchanges\n")
        
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            context_parts.append(f"- **{role_label}**: {content}\n")
            if msg.sql:
                context_parts.append(f"  - SQL: `{msg.sql[:100]}...`\n" if len(msg.sql) > 100 else f"  - SQL: `{msg.sql}`\n")
        
        return "".join(context_parts)
    
    async def _summarize_messages(self, messages: List[Message]) -> str:
        """Use LLM to summarize a list of messages."""
        from data_agent.providers import get_llm
        
        # Format messages for summarization
        conversation_text = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            conversation_text.append(f"{role}: {msg.content}")
            if msg.sql:
                conversation_text.append(f"  [Generated SQL: {msg.sql}]")
        
        prompt = SUMMARIZATION_PROMPT.format(
            conversation_text="\n".join(conversation_text)
        )
        
        try:
            llm = get_llm()
            response = await llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            # Fallback to simple truncation
            return self._simple_summary(messages)
    
    def _simple_summary(self, messages: List[Message]) -> str:
        """Simple fallback summary without LLM."""
        parts = ["Previous conversation discussed:"]
        
        # Extract key info from messages
        tables_mentioned = set()
        queries_count = 0
        
        for msg in messages:
            if msg.sql:
                queries_count += 1
                # Extract table names from SQL
                import re
                tables = re.findall(r'\bFROM\s+(\w+)', msg.sql, re.IGNORECASE)
                tables_mentioned.update(tables)
        
        if tables_mentioned:
            parts.append(f"- Tables referenced: {', '.join(tables_mentioned)}")
        if queries_count:
            parts.append(f"- {queries_count} SQL queries were generated")
        parts.append(f"- {len(messages)} messages exchanged")
        
        return "\n".join(parts)
    
    def _estimate_tokens(self, messages: List[Message]) -> int:
        """Estimate token count for messages (rough approximation)."""
        total_chars = sum(len(msg.content) + (len(msg.sql) if msg.sql else 0) for msg in messages)
        # Rough estimate: ~4 chars per token
        return total_chars // 4
    
    def clear_cache(self, conversation_id: Optional[str] = None) -> None:
        """Clear cached summaries."""
        if conversation_id:
            keys_to_remove = [k for k in self._cached_summaries if k.startswith(conversation_id)]
            for key in keys_to_remove:
                del self._cached_summaries[key]
        else:
            self._cached_summaries.clear()


# Module-level summarizer instance
_default_summarizer: Optional[ConversationSummarizer] = None


def get_summarizer() -> ConversationSummarizer:
    """Get the default summarizer instance."""
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = ConversationSummarizer()
    return _default_summarizer


def configure_summarizer(config: SummarizationConfig) -> None:
    """Configure the default summarizer."""
    global _default_summarizer
    _default_summarizer = ConversationSummarizer(config)
