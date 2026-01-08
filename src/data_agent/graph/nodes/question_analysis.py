"""
Question Analysis Node

This node analyzes the complexity of a user's question and determines
if it needs to be decomposed into sub-questions. Uses a hybrid approach:
1. Fast rule-based check for obviously simple questions
2. LLM-based analysis for potentially complex questions
"""

import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from data_agent.graph.state import AgentState, QuestionAnalysis, SubQuestion


# Signals that indicate a potentially complex question
COMPLEXITY_PATTERNS = [
    # Comparison/contrast
    r"比较|对比|和.*比|versus|compare|vs\.?",
    # Multiple entities
    r"分别|各自|每个|respectively|each",
    # Sequential operations
    r"然后|接着|之后|并且|同时|then|and then|after that",
    # Multi-time dimensions
    r"本月.*上月|今年.*去年|同比|环比|year.over.year|month.over.month",
    # Aggregation with filtering
    r"(最|前\d+|top\s*\d+).*的.*中",
    # Trend analysis
    r"增长|下降|变化|趋势|trend|growth|decline",
    # Conditional logic
    r"如果|假设|条件|if|assuming|condition",
    # Multi-step reasoning
    r"原因|为什么|分析|解释|why|reason|analyze|explain",
]


def is_obviously_simple(question: str) -> bool:
    """
    Fast check for obviously simple questions.
    
    A question is obviously simple if:
    - It's short (< 30 chars) AND has no complexity signals
    - It asks for a single count, sum, or list
    """
    # Very short questions are usually simple
    if len(question) < 25:
        return True
    
    # Check for complexity signals
    for pattern in COMPLEXITY_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            return False
    
    # Single aggregation patterns are simple
    simple_patterns = [
        r"^(有多少|多少个|几个|how many|count)",
        r"^(总共|总计|合计|total|sum)",
        r"^(列出|显示|查看|show|list|display)",
        r"^(最新|最近|latest|recent)",
    ]
    for pattern in simple_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return True
    
    # If question is moderate length with no signals, likely simple
    return len(question) < 50


# LLM prompt for complex question analysis
ANALYSIS_PROMPT = """Analyze the following question and determine if it requires decomposition into multiple sub-queries.

## Question
{question}

## Database Schema Context
{schema_context}

## Analysis Criteria
A question is COMPLEX and requires decomposition if:
1. It compares data from multiple time periods (e.g., "compare this month vs last month")
2. It requires multiple aggregations that can't be done in a single query
3. It asks for analysis or explanation that requires multiple data points
4. It has dependencies between parts (e.g., "find X, then for those X, find Y")

A question is SIMPLE if:
1. It can be answered with a single SQL query
2. It asks for a straightforward aggregation, filter, or join
3. It queries a single fact or dataset

## Output Format
Respond in JSON format:
{{
    "complexity": "simple" | "complex",
    "reasoning": "Brief explanation of why",
    "requires_decomposition": true | false,
    "sub_questions": [
        {{
            "id": "sq1",
            "question": "First sub-question",
            "purpose": "What this finds",
            "depends_on": []
        }},
        {{
            "id": "sq2", 
            "question": "Second sub-question",
            "purpose": "What this finds",
            "depends_on": ["sq1"]
        }}
    ]
}}

If the question is simple, sub_questions should be an empty list.
"""


async def question_analysis(state: AgentState) -> Dict[str, Any]:
    """
    Analyze question complexity using hybrid approach.
    
    This node:
    1. First tries fast rule-based detection for simple questions
    2. Falls back to LLM analysis for potentially complex ones
    3. Returns analysis result with optional sub-questions
    
    Args:
        state: Current graph state with normalized_question
        
    Returns:
        State updates with question_analysis
    """
    from data_agent.providers import get_llm
    
    question = state.get("normalized_question") or state["user_question"]
    schema = state.get("database_schema", "")
    
    # Fast path: obviously simple questions skip LLM
    if is_obviously_simple(question):
        analysis = QuestionAnalysis(
            complexity="simple",
            reasoning="Question is short or matches simple query patterns",
            requires_decomposition=False,
            sub_questions=[],
        )
        return {
            "question_analysis": analysis.model_dump(),
        }
    
    # Slow path: use LLM for complex analysis
    prompt = ANALYSIS_PROMPT.format(
        question=question,
        schema_context=schema[:2000] if schema else "No schema provided",
    )
    
    try:
        llm = get_llm()
        
        # Try to use structured output if available
        try:
            structured_llm = llm.with_structured_output(QuestionAnalysis)
            analysis = await structured_llm.ainvoke(prompt)
        except (AttributeError, NotImplementedError):
            # Fallback to regular invoke and parse
            response = await llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON from response
            import json
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                # Convert sub_questions to SubQuestion objects
                sub_questions = []
                for sq in data.get("sub_questions", []):
                    sub_questions.append(SubQuestion(
                        id=sq.get("id", f"sq{len(sub_questions)+1}"),
                        question=sq.get("question", ""),
                        purpose=sq.get("purpose", ""),
                        depends_on=sq.get("depends_on", []),
                    ))
                analysis = QuestionAnalysis(
                    complexity=data.get("complexity", "simple"),
                    reasoning=data.get("reasoning", ""),
                    requires_decomposition=data.get("requires_decomposition", False),
                    sub_questions=sub_questions,
                )
            else:
                # Default to simple if parsing fails
                analysis = QuestionAnalysis(
                    complexity="simple",
                    reasoning="Could not parse LLM response, defaulting to simple",
                    requires_decomposition=False,
                    sub_questions=[],
                )
        
        # Convert sub_questions to state format
        sub_questions_data = None
        if analysis.requires_decomposition and analysis.sub_questions:
            sub_questions_data = [sq.model_dump() for sq in analysis.sub_questions]
        
        return {
            "question_analysis": analysis.model_dump(),
            "sub_questions": sub_questions_data,
        }
        
    except Exception as e:
        # On any error, default to simple path
        analysis = QuestionAnalysis(
            complexity="simple",
            reasoning=f"Error during analysis: {str(e)}, defaulting to simple",
            requires_decomposition=False,
            sub_questions=[],
        )
        return {
            "question_analysis": analysis.model_dump(),
        }
