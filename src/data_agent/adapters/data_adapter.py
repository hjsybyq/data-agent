"""
DataAgent Adapter

Provides a Vanna-compatible API wrapper around the LangGraph implementation.
This enables easy migration from original Vanna while using the new architecture.
Supports multi-turn conversations with context preservation.
"""

import asyncio
import sys
import threading
import concurrent.futures
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field

# ============================================================
# Windows asyncio 修复
# ============================================================
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _run_async(coro):
    """
    Run an async coroutine safely using a dedicated thread.
    
    This approach works in both sync Flask and async FastAPI contexts.
    """
    result = None
    exception = None
    
    def run_in_thread():
        nonlocal result, exception
        try:
            # Create a fresh event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        except Exception as e:
            exception = e
    
    # Run in a daemon thread
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=300)  # 5 minute timeout
    
    if thread.is_alive():
        raise TimeoutError("Async operation timed out")
    
    if exception:
        raise exception
    
    return result


from data_agent.graph.builder import create_agent_graph, create_simple_graph
from data_agent.graph.state import create_initial_state, AgentState
from data_agent.providers.base import LLMConfig, configure_llm
from data_agent.tools.schema_tool import set_database_schema
from data_agent.tools.sql_execution_tool import set_database_connection, enable_mock_mode
from data_agent.storage.conversation import (
    Message,
    Conversation,
    ConversationStore,
    Session,
)

# New Agent imports
try:
    from data_agent.agent.agent import create_text2sql_agent, Text2SQLAgentExecutor
    from data_agent.tools.example_search_tool import set_rag_store
    from data_agent.tools.ask_user_tool import clarification_state
    AGENT_MODE_AVAILABLE = True
except ImportError:
    AGENT_MODE_AVAILABLE = False


@dataclass
class AgentConfig:
    """Configuration for DataAgent."""
    
    # LLM configuration
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    temperature: float = 0.0
    
    # Graph configuration
    max_retries: int = 3
    use_simple_graph: bool = False
    
    # Conversation configuration
    max_history_turns: int = 5
    
    # Database configuration
    database_schema: Optional[str] = None
    database_connection: Optional[Any] = None
    
    # New Agent mode (LangChain 1.2.0 create_agent)
    use_agent_mode: bool = False
    enable_hitl: bool = False  # Human-in-the-loop for SQL execution
    max_model_calls: int = 10  # Prevent infinite loops


class DataAgent:
    """
    LangGraph-based implementation with Vanna-compatible API.
    
    This class provides a familiar interface for users migrating from
    the original Vanna project while leveraging the new LangGraph architecture.
    
    Supports multi-turn conversations:
        ```python
        vn = DataAgent(...)
        
        # Single question (no history)
        result = vn.ask("How many customers?")
        
        # Multi-turn with session
        session = vn.create_session()
        result1 = session.ask("How many customers?")
        result2 = session.ask("And how many of them are active?")  # Understands context
        ```
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        max_history_turns: int = 5,
        use_simple_graph: bool = False,
        database_connection: Optional[Any] = None,
        enable_rag: bool = True,
        rag_persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        # New Agent mode parameters
        use_agent_mode: bool = False,
        enable_hitl: bool = False,
        max_model_calls: int = 10,
        **kwargs,
    ):
        """
        Initialize DataAgent.
        
        Args:
            llm_provider: LLM provider. Options:
                - "openai": Standard OpenAI API
                - "openai_compatible": Any OpenAI-compatible API (vLLM, LocalAI, etc.)
                - "ollama": Ollama local models
                - "azure": Azure OpenAI
                - "anthropic": Anthropic Claude
                - "google": Google Gemini
            llm_model: Specific model name (uses provider default if None)
            llm_api_key: API key for the LLM provider
            base_url: Base URL for OpenAI-compatible APIs
            temperature: LLM temperature setting
            max_retries: Maximum SQL correction attempts
            max_history_turns: Maximum conversation turns to include in context
            use_simple_graph: Use simplified graph without retry loops
            database_connection: SQLAlchemy connection for SQL execution
            enable_rag: Whether to enable RAG-based example retrieval
            rag_persist_dir: Directory for RAG persistence (None for in-memory)
            use_agent_mode: Use new LangChain 1.2.0 create_agent instead of LangGraph nodes
            enable_hitl: Enable human-in-the-loop for SQL execution (agent mode only)
            max_model_calls: Maximum model calls to prevent infinite loops (agent mode only)
        """
        self.config = AgentConfig(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            temperature=temperature,
            max_retries=max_retries,
            max_history_turns=max_history_turns,
            use_simple_graph=use_simple_graph,
            database_connection=database_connection,
            use_agent_mode=use_agent_mode,
            enable_hitl=enable_hitl,
            max_model_calls=max_model_calls,
        )
        
        # Store additional config
        self._base_url = base_url
        self._use_agent_mode = use_agent_mode and AGENT_MODE_AVAILABLE
        
        # Configure database connection
        if database_connection:
            set_database_connection(database_connection)
        else:
            enable_mock_mode()
        
        self._schema: Optional[str] = None
        
        # Conversation storage for multi-turn support
        self._conversation_store = ConversationStore()
        
        # Initialize mode-specific components
        if self._use_agent_mode:
            # New Agent mode using LangChain 1.2.0 create_agent
            self._init_agent_mode(
                model=llm_model or "gpt-4o",
                api_key=llm_api_key,
                base_url=base_url,
                temperature=temperature,
                enable_hitl=enable_hitl,
                max_model_calls=max_model_calls,
            )
        else:
            # Legacy LangGraph node mode
            self._init_graph_mode(
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_api_key=llm_api_key,
                base_url=base_url,
                temperature=temperature,
                use_simple_graph=use_simple_graph,
            )
        
        # RAG vector store for example retrieval
        self._enable_rag = enable_rag
        self._vector_store = None
        if enable_rag:
            self._init_rag(
                llm_api_key=llm_api_key,
                base_url=base_url,
                embedding_model=embedding_model,
                rag_persist_dir=rag_persist_dir,
            )
    
    def _init_agent_mode(
        self,
        model: str,
        api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        enable_hitl: bool,
        max_model_calls: int,
    ):
        """Initialize new Agent mode components."""
        self._agent = create_text2sql_agent(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            enable_hitl=enable_hitl,
            max_model_calls=max_model_calls,
        )
        self._graph = None  # Not used in agent mode
    
    def _init_graph_mode(
        self,
        llm_provider: str,
        llm_model: Optional[str],
        llm_api_key: Optional[str],
        base_url: Optional[str],
        temperature: float,
        use_simple_graph: bool,
    ):
        """Initialize legacy LangGraph node mode components."""
        # Configure LLM with base_url support
        llm_config = LLMConfig(
            provider=llm_provider,  # type: ignore
            model=llm_model,
            api_key=llm_api_key,
            base_url=base_url,
            temperature=temperature,
        )
        configure_llm(llm_config)
        
        # Create the graph
        if use_simple_graph:
            self._graph = create_simple_graph()
        else:
            self._graph = create_agent_graph()
        
        self._agent = None  # Not used in graph mode
    
    def _init_rag(
        self,
        llm_api_key: Optional[str],
        base_url: Optional[str],
        embedding_model: Optional[str],
        rag_persist_dir: Optional[str],
    ):
        """Initialize RAG vector store."""
        from data_agent.rag import FAISSStore, EmbeddingConfig, configure_embeddings
        
        # Configure embeddings
        embed_config = EmbeddingConfig(
            provider="openai",
            model=embedding_model or "text-embedding-3-small",
            api_key=llm_api_key,
            base_url=base_url,
        )
        configure_embeddings(embed_config)
        
        self._vector_store = FAISSStore(persist_dir=rag_persist_dir)
        
        # Connect to appropriate mode
        if self._use_agent_mode:
            # Set RAG store for agent tools
            set_rag_store(self._vector_store)
        else:
            # Connect vector store to the graph retrieval node
            from data_agent.graph.nodes import set_vector_store
            set_vector_store(self._vector_store)
    
    def set_schema(self, schema: str) -> None:
        """
        Set the database schema for SQL generation.
        
        Args:
            schema: Database schema as CREATE TABLE statements
        """
        self._schema = schema
        self.config.database_schema = schema
        set_database_schema(schema)
    
    def connect_to_database(self, connection: Any) -> None:
        """
        Connect to a database for SQL execution.
        
        Args:
            connection: SQLAlchemy engine or connection
        """
        self.config.database_connection = connection
        set_database_connection(connection)
    
    # ================================================================
    # Session Management for Multi-Turn Conversations
    # ================================================================
    
    def create_session(self, conversation_id: Optional[str] = None) -> Session:
        """
        Create a new conversation session for multi-turn dialogue.
        
        Args:
            conversation_id: Optional ID for the conversation
            
        Returns:
            Session object for continued conversation
            
        Example:
            session = vn.create_session()
            result1 = session.ask("閺堝顦跨亸鎴濐吂閹村嚖绱?)
            result2 = session.ask("娴犳牔婊戞稉顓熸箒婢舵艾鐨弰顖涙拱閺堝牊鏁為崘宀€娈戦敍?)
        """
        conversation = self._conversation_store.get_or_create(conversation_id)
        return Session(self, conversation)
    
    def get_session(self, conversation_id: str) -> Optional[Session]:
        """
        Get an existing conversation session.
        
        Args:
            conversation_id: The conversation ID
            
        Returns:
            Session object or None if not found
        """
        conversation = self._conversation_store.get_conversation(conversation_id)
        if conversation:
            return Session(self, conversation)
        return None
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List all conversation sessions.
        
        Returns:
            List of session info dicts
        """
        conversations = self._conversation_store.list_conversations(limit)
        return [
            {
                "id": conv.id,
                "message_count": len(conv.messages),
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            }
            for conv in conversations
        ]
    
    # ================================================================
    # Core API Methods
    # ================================================================
    
    def ask(
        self, 
        question: str,
        conversation_id: Optional[str] = None,
        print_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Ask a natural language question and get SQL + results.
        
        Args:
            question: Natural language question
            conversation_id: Optional ID to continue existing conversation
            print_results: Whether to print results (for compatibility)
            
        Returns:
            Dictionary with "sql", "result", "answer", and "conversation_id" keys
        """
        # Get or create conversation
        conversation = self._conversation_store.get_or_create(conversation_id)
        
        # Add user message to history
        conversation.add_message("user", question)
        
        # Route to appropriate mode
        if self._use_agent_mode and self._agent is not None:
            return self._ask_agent_mode(question, conversation)
        else:
            return self._ask_graph_mode(question, conversation)
    
    def _ask_agent_mode(
        self,
        question: str,
        conversation: "Conversation",
    ) -> Dict[str, Any]:
        """Handle ask using new Agent mode."""
        # Build conversation history for agent
        messages = []
        for msg in conversation.messages[:-1]:  # Exclude current question
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        messages.append({"role": "user", "content": question})
        
        # Invoke agent
        result = _run_async(self._agent.ainvoke({"messages": messages}))
        
        # Extract results from agent output
        agent_messages = result.get("messages", [])
        
        # Find SQL and answer from agent's tool calls and responses
        sql = None
        sql_result = None
        clarification_needed = False
        clarification_question = None
        
        # 提取任务规划和执行过程
        todo_list = []  # 任务规划列表
        execution_steps = []  # 执行步骤记录
        all_sqls = []  # 所有执行的 SQL 列表
        pending_sql = None  # 待匹配的 SQL
        
        for msg in agent_messages:
            # Check for tool calls
            if hasattr(msg, "tool_calls"):
                for tool_call in getattr(msg, "tool_calls", []):
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        
                        if tool_name == "execute_sql":
                            sql = tool_args.get("sql")
                            pending_sql = sql  # 保存 SQL 用于匹配结果
                            all_sqls.append(sql)
                        elif tool_name == "write_todos":
                            # 提取任务规划
                            todos = tool_args.get("todos", [])
                            if todos:
                                todo_list = todos
                        elif tool_name == "ask_user_clarification":
                            # 处理用户澄清请求
                            clarification_needed = True
                            clarification_question = tool_args.get("question")
            
            # Check for tool results
            if hasattr(msg, "name"):
                tool_name = msg.name
                tool_content = msg.content
                
                # 核心工具名称映射（用于展示）
                CORE_TOOLS = {
                    "get_database_schema": "📋 获取 Schema",
                    "search_examples": "🔍 检索样例",
                    "validate_sql": "✓ 验证 SQL",
                    "execute_sql": "▶ 生成 SQL",
                }
                
                # 只记录核心工具的执行步骤（过滤 write_todos 等）
                if tool_name in CORE_TOOLS:
                    # 生成内容摘要
                    summary = ""
                    result_preview = None  # 用于前端展示的完整结果数据
                    
                    if tool_name == "get_database_schema":
                        # 提取表名
                        import re
                        tables = re.findall(r'CREATE TABLE (\w+)', tool_content)
                        summary = f"发现 {len(tables)} 张表: {', '.join(tables[:5])}" + ("..." if len(tables) > 5 else "")
                    elif tool_name == "search_examples":
                        if "error" in tool_content.lower():
                            summary = "未找到相关样例"
                        else:
                            # 尝试计算样例数量
                            try:
                                import json
                                examples = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                                count = len(examples) if isinstance(examples, list) else 0
                                summary = f"找到 {count} 个参考样例" if count > 0 else "检索完成"
                            except:
                                # Fallback: count SQL blocks in text
                                count = tool_content.count("```sql")
                                summary = f"找到 {count} 个参考样例" if count > 0 else "检索完成"
                    elif tool_name == "validate_sql":
                        if '"valid": true' in tool_content or '"valid":true' in tool_content:
                            summary = "SQL 语法正确"
                        else:
                            summary = "SQL 语法有误"
                    elif tool_name == "execute_sql":
                        try:
                            import json
                            result_data = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                            if result_data.get("success"):
                                row_count = result_data.get("row_count", len(result_data.get("data", [])))
                                summary = f"返回 {row_count} 行"
                                result_preview = result_data  # 保存完整结果用于展示
                            else:
                                summary = f"失败: {result_data.get('error', '未知错误')[:30]}"
                        except:
                            summary = "执行完成"
                    
                    step_data = {
                        "tool": CORE_TOOLS[tool_name],
                        "tool_id": tool_name,
                        "summary": summary,
                        "result_data": result_preview, # 添加结果数据
                        "status": "success" if "error" not in tool_content.lower() else "error",
                    }
                    # 为 execute_sql 添加 SQL 内容
                    if tool_name == "execute_sql" and pending_sql:
                        step_data["sql"] = pending_sql
                        pending_sql = None  # 清除已使用的 SQL
                    
                    execution_steps.append(step_data)
                
                if tool_name == "execute_sql":
                    try:
                        import json
                        sql_result = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                    except:
                        sql_result = {"data": tool_content}
                elif tool_name == "ask_user_clarification":
                    clarification_needed = True
                    clarification_question = tool_content
                elif tool_name == "write_todos":
                    # 解析并更新 todo_list
                    try:
                        import json
                        todo_data = json.loads(tool_content) if isinstance(tool_content, str) else tool_content
                        if isinstance(todo_data, list):
                            todo_list = todo_data
                        elif isinstance(todo_data, dict) and "todos" in todo_data:
                            todo_list = todo_data["todos"]
                    except:
                        pass
        
        # Get final answer
        final_answer = agent_messages[-1].content if agent_messages else ""
        if clarification_needed and clarification_question:
            final_answer = clarification_question
        
        # Add assistant response to history
        conversation.add_message(
            "assistant", 
            final_answer,
            sql=sql,
        )
        
        return {
            "sql": sql,
            "result": sql_result,
            "answer": final_answer,
            "is_valid": sql is not None,
            "retry_count": 0,
            "conversation_id": conversation.id,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
            # TodoList 任务规划信息
            "todo_list": todo_list,
            "execution_steps": execution_steps,
            # No decomposition in agent mode (replaced by todo_list)
            "question_analysis": None,
            "sub_questions": None,
            "sub_results": None,
            "similar_examples": None,
        }
    
    def _ask_graph_mode(
        self,
        question: str,
        conversation: "Conversation",
    ) -> Dict[str, Any]:
        """Handle ask using legacy Graph mode."""
        # Build context from conversation history
        conversation_context = ""
        if len(conversation.messages) > 1:
            conversation_context = conversation.get_context_summary()
        
        # Create initial state with conversation context
        initial_state = create_initial_state(
            user_question=question,
            max_retries=self.config.max_retries,
            database_schema=self._schema,
            conversation_context=conversation_context,
        )
        
        # Run the graph synchronously to avoid Windows asyncio issues
        final_state = _run_async(self._graph.ainvoke(initial_state))
        
        # Extract results
        sql = final_state.get("generated_sql")
        answer = final_state.get("final_answer", "")
        
        # Add assistant response to history
        conversation.add_message(
            "assistant", 
            answer,
            sql=sql,
        )
        
        return {
            "sql": sql,
            "result": final_state.get("sql_result"),
            "answer": answer,
            "is_valid": final_state.get("is_sql_valid", False),
            "retry_count": final_state.get("retry_count", 0),
            "conversation_id": conversation.id,
            # Decomposition info
            "question_analysis": final_state.get("question_analysis"),
            "sub_questions": final_state.get("sub_questions"),
            "sub_results": final_state.get("sub_results"),
            "similar_examples": final_state.get("retrieved_examples"),
        }
    
    async def ask_async(
        self, 
        question: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of ask().
        
        Args:
            question: Natural language question
            conversation_id: Optional ID to continue existing conversation
            
        Returns:
            Dictionary with "sql", "result", "answer", and "conversation_id" keys
        """
        # Get or create conversation
        conversation = self._conversation_store.get_or_create(conversation_id)
        
        # Add user message to history
        conversation.add_message("user", question)
        
        # Build context from conversation history
        conversation_context = ""
        if len(conversation.messages) > 1:
            conversation_context = conversation.get_context_summary()
        
        # Create initial state with conversation context
        initial_state = create_initial_state(
            user_question=question,
            max_retries=self.config.max_retries,
            database_schema=self._schema,
            conversation_context=conversation_context,
        )
        
        # Run the graph
        final_state = await self._graph.ainvoke(initial_state)
        
        # Extract results
        sql = final_state.get("generated_sql")
        answer = final_state.get("final_answer", "")
        
        # Add assistant response to history
        conversation.add_message(
            "assistant", 
            answer,
            sql=sql,
        )
        
        return {
            "sql": sql,
            "result": final_state.get("sql_result"),
            "answer": answer,
            "is_valid": final_state.get("is_sql_valid", False),
            "retry_count": final_state.get("retry_count", 0),
            "conversation_id": conversation.id,
            # Decomposition info
            "question_analysis": final_state.get("question_analysis"),
            "sub_questions": final_state.get("sub_questions"),
            "sub_results": final_state.get("sub_results"),
        }
    
    def generate_sql(
        self, 
        question: str,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Generate SQL without executing.
        
        Args:
            question: Natural language question
            conversation_id: Optional ID to use conversation context
            
        Returns:
            Generated SQL query string
        """
        return _run_async(self.generate_sql_async(question, conversation_id))
    
    async def generate_sql_async(
        self, 
        question: str,
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        Async version of generate_sql().
        
        Args:
            question: Natural language question
            conversation_id: Optional ID to use conversation context
            
        Returns:
            Generated SQL query string
        """
        # Use simple graph for generation only
        simple_graph = create_simple_graph()
        
        # Build context if conversation exists
        conversation_context = ""
        if conversation_id:
            conversation = self._conversation_store.get_conversation(conversation_id)
            if conversation and conversation.messages:
                conversation_context = conversation.get_context_summary()
        
        initial_state = create_initial_state(
            user_question=question,
            max_retries=0,
            database_schema=self._schema,
            conversation_context=conversation_context,
        )
        
        final_state = await simple_graph.ainvoke(initial_state)
        return final_state.get("generated_sql", "")
    
    def run_sql(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL directly.
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Execution result dictionary
        """
        return _run_async(self.run_sql_async(sql))
    
    async def run_sql_async(self, sql: str) -> Dict[str, Any]:
        """
        Async version of run_sql().
        
        Args:
            sql: SQL query to execute
            
        Returns:
            Execution result dictionary
        """
        from data_agent.tools.sql_execution_tool import execute_sql_query
        return await execute_sql_query(sql)
    
    def train(
        self,
        question: Optional[str] = None,
        sql: Optional[str] = None,
        ddl: Optional[str] = None,
        documentation: Optional[str] = None,
    ) -> str:
        """
        Add training data for RAG-based example retrieval.
        
        Args:
            question: Example question  
            sql: Example SQL query
            ddl: Schema DDL (sets database schema)
            documentation: Additional documentation
            
        Returns:
            ID of the added training data, or empty string if nothing added
        """
        result_id = ""
        
        if ddl:
            self.set_schema(ddl)
        
        if question and sql and self._vector_store:
            result_id = self._vector_store.add_example(question, sql, {"source": "train"})
        
        if documentation and self._vector_store:
            result_id = self._vector_store.add_documentation(documentation, {"source": "train"})
        
        return result_id
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get all training examples.
        
        Returns:
            List of training examples with question, sql, and metadata
        """
        if not self._vector_store:
            return []
        
        examples = self._vector_store.get_all_examples()
        return [
            {
                "id": ex.id,
                "question": ex.question,
                "sql": ex.sql,
                "metadata": ex.metadata,
            }
            for ex in examples
        ]
    
    def remove_training_data(self, example_id: str) -> bool:
        """
        Remove a training example by ID.
        
        Args:
            example_id: ID of the example to remove
            
        Returns:
            True if removed, False if not found
        """
        if not self._vector_store:
            return False
        return self._vector_store.remove_example(example_id)
    
    def search_similar_examples(self, question: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar question-SQL examples.
        
        Args:
            question: Query to search for
            k: Number of results to return
            
        Returns:
            List of similar examples with scores
        """
        if not self._vector_store:
            return []
        
        examples = self._vector_store.search_examples(question, k)
        return [
            {
                "id": ex.id,
                "question": ex.question,
                "sql": ex.sql,
                "score": ex.score,
            }
            for ex in examples
        ]
    
    def get_related_ddl(self, question: str) -> str:
        """
        Get relevant DDL for a question (compatibility method).
        
        Args:
            question: User question
            
        Returns:
            Relevant schema DDL
        """
        return self._schema or ""
    
    @property
    def graph(self):
        """Access the underlying LangGraph for advanced usage."""
        return self._graph
    
    @property
    def conversation_store(self) -> ConversationStore:
        """Access the conversation store."""
        return self._conversation_store

