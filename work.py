"""
Data Agent 测试 - 使用 Chinook 真实数据库
"""

import sys
sys.path.insert(0, "src")

# ============================================================
# Windows socket.socketpair 修复
# 必须在导入 asyncio 和其他模块之前执行
# ============================================================
import socket
if sys.platform == "win32":
    _original_socketpair = socket.socketpair
    
    def _patched_socketpair(family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0):
        """Patched socketpair with retries for Windows network issues."""
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return _original_socketpair(family, type, proto)
            except (ConnectionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                # Last resort: use a different approach
                # Create a socket pair manually using a listener
                lsock = socket.socket(family, type, proto)
                try:
                    lsock.bind(('127.0.0.1', 0))
                    lsock.listen(1)
                    addr = lsock.getsockname()
                    csock = socket.socket(family, type, proto)
                    try:
                        csock.setblocking(True)
                        csock.connect(addr)
                        ssock, _ = lsock.accept()
                        csock.setblocking(True)
                        ssock.setblocking(True)
                        return (ssock, csock)
                    except Exception:
                        csock.close()
                        raise
                finally:
                    lsock.close()
        raise RuntimeError("Failed to create socketpair after retries")
    
    socket.socketpair = _patched_socketpair

import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import create_engine, text
from data_agent import DataAgent

# ============================================================
# Chinook 数据库 Schema
# ============================================================
# Chinook 是一个数字音乐商店数据库，包含以下表：
# - Artist: 艺术家
# - Album: 专辑
# - Track: 歌曲
# - Genre: 流派
# - MediaType: 媒体类型
# - Playlist: 播放列表
# - PlaylistTrack: 播放列表中的歌曲
# - Customer: 客户
# - Employee: 员工
# - Invoice: 发票/订单
# - InvoiceLine: 订单明细

CHINOOK_SCHEMA = """
CREATE TABLE Artist (
    ArtistId INTEGER PRIMARY KEY,
    Name NVARCHAR(120)
);

CREATE TABLE Album (
    AlbumId INTEGER PRIMARY KEY,
    Title NVARCHAR(160) NOT NULL,
    ArtistId INTEGER NOT NULL REFERENCES Artist(ArtistId)
);

CREATE TABLE Track (
    TrackId INTEGER PRIMARY KEY,
    Name NVARCHAR(200) NOT NULL,
    AlbumId INTEGER REFERENCES Album(AlbumId),
    MediaTypeId INTEGER NOT NULL,
    GenreId INTEGER,
    Composer NVARCHAR(220),
    Milliseconds INTEGER NOT NULL,
    Bytes INTEGER,
    UnitPrice NUMERIC(10,2) NOT NULL
);

CREATE TABLE Genre (
    GenreId INTEGER PRIMARY KEY,
    Name NVARCHAR(120)
);

CREATE TABLE Customer (
    CustomerId INTEGER PRIMARY KEY,
    FirstName NVARCHAR(40) NOT NULL,
    LastName NVARCHAR(20) NOT NULL,
    Company NVARCHAR(80),
    Address NVARCHAR(70),
    City NVARCHAR(40),
    State NVARCHAR(40),
    Country NVARCHAR(40),
    PostalCode NVARCHAR(10),
    Phone NVARCHAR(24),
    Email NVARCHAR(60) NOT NULL,
    SupportRepId INTEGER
);

CREATE TABLE Employee (
    EmployeeId INTEGER PRIMARY KEY,
    LastName NVARCHAR(20) NOT NULL,
    FirstName NVARCHAR(20) NOT NULL,
    Title NVARCHAR(30),
    ReportsTo INTEGER,
    BirthDate DATETIME,
    HireDate DATETIME,
    Address NVARCHAR(70),
    City NVARCHAR(40),
    State NVARCHAR(40),
    Country NVARCHAR(40),
    PostalCode NVARCHAR(10),
    Phone NVARCHAR(24),
    Email NVARCHAR(60)
);

CREATE TABLE Invoice (
    InvoiceId INTEGER PRIMARY KEY,
    CustomerId INTEGER NOT NULL REFERENCES Customer(CustomerId),
    InvoiceDate DATETIME NOT NULL,
    BillingAddress NVARCHAR(70),
    BillingCity NVARCHAR(40),
    BillingState NVARCHAR(40),
    BillingCountry NVARCHAR(40),
    BillingPostalCode NVARCHAR(10),
    Total NUMERIC(10,2) NOT NULL
);

CREATE TABLE InvoiceLine (
    InvoiceLineId INTEGER PRIMARY KEY,
    InvoiceId INTEGER NOT NULL REFERENCES Invoice(InvoiceId),
    TrackId INTEGER NOT NULL REFERENCES Track(TrackId),
    UnitPrice NUMERIC(10,2) NOT NULL,
    Quantity INTEGER NOT NULL
);

CREATE TABLE Playlist (
    PlaylistId INTEGER PRIMARY KEY,
    Name NVARCHAR(120)
);

CREATE TABLE PlaylistTrack (
    PlaylistId INTEGER NOT NULL,
    TrackId INTEGER NOT NULL,
    PRIMARY KEY (PlaylistId, TrackId)
);
"""

# ============================================================
# 连接真实数据库
# ============================================================
print("正在连接 Chinook.sqlite 数据库...")
engine = create_engine("sqlite:///Chinook.sqlite")

# 验证连接
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM Customer"))
    count = result.fetchone()[0]
    print(f"✓ 数据库连接成功，共有 {count} 位客户")

# ============================================================
# 初始化 DataAgent
# ============================================================
vn = DataAgent(
    llm_provider="openai_compatible",
    base_url="https://api.siliconflow.cn/v1",
    llm_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    llm_api_key="sk-",
    database_connection=engine.connect(),  # 真实数据库连接
    embedding_model="BAAI/bge-m3",  # SiliconFlow 向量模型
    # 新 Agent 模式配置
    use_agent_mode=True,           # 启用 LangChain 1.2.0 create_agent 架构
    enable_hitl=False,             # 是否启用人工审核（暂时关闭便于测试）
    max_model_calls=15,            # 最大模型调用次数，防止无限循环
)

vn.set_schema(CHINOOK_SCHEMA)
print("✓ DataAgent 初始化完成 (Agent 模式)\n")


def display_agent_result(result: dict, question: str):
    """展示 Agent 模式的执行结果，包括任务规划和执行过程"""
    print(f"问题: {question}")
    
    # 展示任务规划
    todo_list = result.get('todo_list', [])
    if todo_list:
        print("\n📋 [任务规划]")
        for i, todo in enumerate(todo_list, 1):
            if isinstance(todo, dict):
                status_icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(
                    todo.get("status", "pending"), "⏳"
                )
                print(f"  {status_icon} {i}. {todo.get('description', todo)}")
            else:
                print(f"  ⏳ {i}. {todo}")
    
    # 展示执行步骤
    execution_steps = result.get('execution_steps', [])
    if execution_steps:
        print("\n🔧 [执行过程]")
        for i, step in enumerate(execution_steps, 1):
            tool_name = step.get('tool', 'unknown')
            # 简化显示
            result_preview = step.get('result', '')[:100]
            if len(step.get('result', '')) > 100:
                result_preview += "..."
            print(f"  {i}. {tool_name}: {result_preview}")
    
    # 展示 SQL
    print(f"\n📝 SQL: {result.get('sql', 'None')}")
    
    # 展示结果
    sql_result = result.get('result')
    if sql_result:
        print(f"📊 结果: {sql_result}")
    
    # 展示最终答案
    print(f"\n💬 答案: {result.get('answer', '')}")
    print("-" * 60)


# ============================================================
# 测试 1: 简单查询
# ============================================================
print("=" * 60)
print("测试 1: 简单查询")
print("=" * 60)

result = vn.ask("数据库中有多少位客户？")
display_agent_result(result, "数据库中有多少位客户？")

# ============================================================
# # 测试 2: 聚合查询
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 2: 聚合查询")
# print("=" * 60)

# result = vn.ask("销售额最高的前5个国家是哪些？")
# display_agent_result(result, "销售额最高的前5个国家是哪些？")

# # ============================================================
# # 测试 3: JOIN 查询
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 3: JOIN 查询")
# print("=" * 60)

# result = vn.ask("哪位艺术家的专辑数量最多？")
# print(f"问题: 哪位艺术家的专辑数量最多？")
# print(f"SQL: {result['sql']}")
# print(f"结果: {result['result']}")
# print(f"答案: {result['answer']}")

# # ============================================================
# # 测试 4: 多轮对话
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 4: 多轮对话")
# print("=" * 60)

# session = vn.create_session()
# print(f"会话 ID: {session.conversation_id}\n")

# # 第一轮
# print("[第1轮]")
# result1 = session.ask("Rock 流派有多少首歌曲？")
# print(f"问题: Rock 流派有多少首歌曲？")
# print(f"SQL: {result1['sql']}")
# print(f"答案: {result1['answer']}")

# # 第二轮 - 使用代词
# print("\n[第2轮]")
# result2 = session.ask("它们的总时长是多少分钟？")
# print(f"问题: 它们的总时长是多少分钟？")
# print(f"SQL: {result2['sql']}")  # 应该能理解"它们"指的是 Rock 歌曲
# print(f"答案: {result2['answer']}")

# # 第三轮 - 继续追问
# print("\n[第3轮]")
# result3 = session.ask("其中最长的一首歌是什么？")
# print(f"问题: 其中最长的一首歌是什么？")
# print(f"SQL: {result3['sql']}")
# print(f"答案: {result3['answer']}")

# # ============================================================
# # 测试 5: 复杂查询
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 5: 复杂查询")
# print("=" * 60)

# result = vn.ask("每位员工的销售业绩如何？按销售额降序排列")
# print(f"问题: 每位员工的销售业绩如何？按销售额降序排列")
# print(f"SQL: {result['sql']}")
# print(f"结果: {result['result']}")
# print(f"答案: {result['answer']}")

# # ============================================================
# # 测试 6: RAG 样例检索
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 6: RAG 样例检索")
# print("=" * 60)

# # 添加训练样例
# print("\n[添加训练样例]")
# vn.train(question="有多少客户", sql="SELECT COUNT(*) FROM Customer")
# vn.train(question="所有艺术家", sql="SELECT * FROM Artist")
# vn.train(question="销售额最高的国家", sql="SELECT BillingCountry, SUM(Total) as TotalSales FROM Invoice GROUP BY BillingCountry ORDER BY TotalSales DESC LIMIT 1")
# print("✓ 添加了 3 个训练样例")

# # 查看训练数据
# training_data = vn.get_training_data()
# print(f"✓ 当前训练数据数量: {len(training_data)}")

# # 搜索相似样例
# print("\n[搜索相似样例]")
# similar = vn.search_similar_examples("客户总数是多少", k=3)
# print(f"查询: '客户总数是多少'")
# for i, ex in enumerate(similar, 1):
#     print(f"  {i}. 相似度: {ex['score']:.3f} - {ex['question']}")

# # 使用训练数据进行查询
# print("\n[使用训练数据查询]")
# result = vn.ask("数据库里有多少艺术家？")
# print(f"问题: 数据库里有多少艺术家？")
# print(f"SQL: {result['sql']}")
# print(f"答案: {result['answer']}")

# # ============================================================
# # 测试 7: 问题拆分（复杂查询）
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 7: 问题拆分（复杂查询）")
# print("=" * 60)

# # 复杂问题 - 需要多步分析
# complex_question = "比较Rock和Jazz两种流派的歌曲数量，哪个更多？分别有多少首？"
# print(f"\n问题: {complex_question}")

# result = vn.ask(complex_question)
# print(f"\nSQL: {result['sql']}")
# print(f"答案: {result['answer']}")

# # 另一个复杂问题 - 涉及多表关联和聚合
# complex_question2 = "找出购买金额最高的客户，以及他购买了多少首歌曲，总金额是多少？"
# print(f"\n问题: {complex_question2}")

# result2 = vn.ask(complex_question2)
# print(f"\nSQL: {result2['sql']}")
# print(f"答案: {result2['answer']}")

# # ============================================================
# # 测试 8: 多步依赖问题（更复杂，应触发拆分）
# # ============================================================
# print("\n" + "=" * 60)
# print("测试 8: 多步依赖问题（应触发拆分）")
# print("=" * 60)

# # 这类问题有明确的多步依赖关系
# decompose_question = """
# 找出购买金额最高的客户是谁，然后分析这个客户的购买偏好：
# 1. 他最喜欢哪个流派的音乐？
# 2. 他购买次数最多的艺术家是谁？
# 3. 他的平均每次购买金额是多少？
# """
# print(f"\n问题: {decompose_question}")

# result3 = vn.ask(decompose_question)

# # 显示问题分析结果
# analysis = result3.get('question_analysis')
# if analysis:
#     print(f"\n[问题分析]")
#     print(f"  复杂度: {analysis.get('complexity', 'unknown')}")
#     print(f"  需要拆分: {analysis.get('requires_decomposition', False)}")
#     print(f"  原因: {analysis.get('reasoning', 'N/A')}")

# # 显示子问题和执行过程
# sub_questions = result3.get('sub_questions')
# if sub_questions:
#     print(f"\n[问题拆分] - 共 {len(sub_questions)} 个子问题:")
#     for i, sq in enumerate(sub_questions, 1):
#         print(f"\n  子问题 {i}: {sq.get('question', 'N/A')}")
#         print(f"    目的: {sq.get('purpose', 'N/A')}")
#         print(f"    依赖: {sq.get('depends_on', [])}")
#         if sq.get('sql'):
#             print(f"    SQL: {sq.get('sql')[:100]}...")
#         if sq.get('result'):
#             result_data = sq['result']
#             if result_data.get('success'):
#                 data = result_data.get('data', [])
#                 print(f"    结果: {data[:2]}..." if len(data) > 2 else f"    结果: {data}")
#             else:
#                 print(f"    ❌ 错误: {result_data.get('error', 'Unknown')}")
# else:
#     # 没有拆分，显示单一SQL
#     print(f"\n[直接生成 SQL]")
#     print(f"SQL: {result3['sql']}")

# print(f"\n[最终答案]")
# print(f"{result3['answer']}")

# # 检查是否触发了问题拆分
# if sub_questions:
#     print(f"\n✓ 问题被拆分为 {len(sub_questions)} 个子问题")
# else:
#     print("\n⚠ 问题未被拆分（LLM判断可用单一查询解决）")

# print("\n" + "=" * 60)
# print("✅ 所有测试完成!")
# print("=" * 60)
