"""
Text-to-SQL Agent Factory

Creates a LangChain 1.2.0 create_agent based Text-to-SQL agent
with middleware support for production use.
"""

from typing import Any, Dict, List, Optional, Union
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    TodoListMiddleware,
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from data_agent.agent.prompts import (
    TEXT2SQL_SYSTEM_PROMPT,
    FORCE_TODO_SYSTEM_PROMPT,
    FORCE_TODO_TOOL_DESCRIPTION,
)
from data_agent.tools.schema_tool import get_database_schema
from data_agent.tools.sql_execution_tool import execute_sql
from data_agent.tools.validation_tool import validate_sql


def create_text2sql_agent(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    enable_summarization: bool = True,
    enable_hitl: bool = False,
    enable_todo_list: bool = True,  # 新增：启用 TodoList 支持复杂问题拆分
    max_model_calls: int = 10,
    hitl_tools: Optional[List[str]] = None,
    **kwargs,
):
    """
    Create a Text-to-SQL Agent using LangChain 1.2.0 create_agent.
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-3.5-turbo")
        api_key: OpenAI API key (uses env var if not provided)
        base_url: Optional base URL for OpenAI-compatible APIs
        temperature: Model temperature (0.0 for deterministic)
        tools: Optional list of additional tools
        system_prompt: Optional custom system prompt
        enable_summarization: Enable conversation summarization middleware
        enable_hitl: Enable human-in-the-loop middleware
        max_model_calls: Maximum model calls to prevent infinite loops
        hitl_tools: Tool names that require human approval
        **kwargs: Additional arguments passed to create_agent
        
    Returns:
        Compiled agent ready for invocation
    """
    # Configure LLM
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
    }
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
        
    llm = ChatOpenAI(**llm_kwargs)
    
    # Configure tools - use defaults if not provided
    if tools is None:
        tools = get_default_tools()
    
    # Configure middleware
    middleware = []
    
    # Summarization middleware for long conversations
    if enable_summarization:
        middleware.append(
            SummarizationMiddleware(
                model=llm,  # 使用 model 参数而非 llm
                trigger=("tokens", 4000),  # 触发摘要的 token 阈值
            )
        )
    
    # Human-in-the-loop for sensitive operations
    if enable_hitl:
        tools_requiring_approval = hitl_tools or ["execute_sql"]
        # 配置需要人工审核的工具
        interrupt_config = {tool: True for tool in tools_requiring_approval}
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on=interrupt_config,
            )
        )
    
    # TodoList middleware for complex multi-step tasks (可选模式)
    if enable_todo_list:
        middleware.append(
            TodoListMiddleware()  # 使用默认配置，LLM 自行判断是否需要规划
        )
    
    # Model call limit to prevent infinite loops
    middleware.append(
        ModelCallLimitMiddleware(
            run_limit=max_model_calls,  # 每次运行的最大调用次数
        )
    )
    
    # Create agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt or TEXT2SQL_SYSTEM_PROMPT,
        middleware=middleware if middleware else None,
        **kwargs,
    )
    
    return agent


def get_default_tools() -> List[BaseTool]:
    """
    Get the default set of tools for Text-to-SQL agent.
    
    Returns:
        List of LangChain tools
    """
    from data_agent.tools.schema_tool import get_database_schema
    from data_agent.tools.sql_execution_tool import execute_sql
    from data_agent.tools.validation_tool import validate_sql
    
    # Import optional tools if available
    tools = [
        get_database_schema,
        execute_sql,
        validate_sql,
    ]
    
    # Try to add example search tool
    try:
        from data_agent.tools.example_search_tool import search_examples
        tools.append(search_examples)
    except ImportError:
        pass
    
    # Try to add ask user tool
    try:
        from data_agent.tools.ask_user_tool import ask_user_clarification
        tools.append(ask_user_clarification)
    except ImportError:
        pass
    
    return tools


class Text2SQLAgentExecutor:
    """
    Wrapper class for Text-to-SQL agent with additional features.
    
    Provides a simpler interface for common operations and
    handles multi-turn conversations.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the agent executor."""
        self.agent = create_text2sql_agent(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self._config = {
            "model": model,
            "base_url": base_url,
        }
    
    def invoke(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user question and return results.
        
        Args:
            question: User's natural language question
            conversation_history: Optional previous conversation messages
            
        Returns:
            Dictionary with response and metadata
        """
        # Build messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})
        
        # Invoke agent
        result = self.agent.invoke({"messages": messages})
        
        return self._format_result(result)
    
    async def ainvoke(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Async version of invoke.
        
        Args:
            question: User's natural language question
            conversation_history: Optional previous conversation messages
            
        Returns:
            Dictionary with response and metadata
        """
        # Build messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": question})
        
        # Invoke agent
        result = await self.agent.ainvoke({"messages": messages})
        
        return self._format_result(result)
    
    def _format_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format agent result into standard response format."""
        # Extract the final message
        messages = result.get("messages", [])
        
        # Find SQL and execution results from tool calls
        sql = None
        execution_result = None
        clarification_needed = False
        clarification_question = None
        
        for msg in messages:
            if hasattr(msg, "tool_calls"):
                for tool_call in msg.tool_calls:
                    if tool_call.get("name") == "execute_sql":
                        sql = tool_call.get("args", {}).get("sql")
            
            # Check for tool results
            if hasattr(msg, "name"):
                if msg.name == "execute_sql":
                    execution_result = msg.content
                elif msg.name == "ask_user_clarification":
                    clarification_needed = True
                    clarification_question = msg.content
        
        # Get final response
        final_response = messages[-1].content if messages else ""
        
        return {
            "answer": final_response,
            "sql": sql,
            "result": execution_result,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
            "messages": messages,
        }
