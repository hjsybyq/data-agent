"""
Result Synthesizer Node

This node combines results from multiple sub-queries into a cohesive
final answer. Used after all sub-queries have been executed for
complex decomposed questions.
"""

from typing import Dict, Any, List

from data_agent.graph.state import AgentState


SYNTHESIS_PROMPT = """Synthesize the following sub-query results into a comprehensive answer.

**IMPORTANT: Always respond in Chinese (涓枃).**

## Original Question
{original_question}

## Sub-Query Results
{sub_results}

## Instructions
1. 灏嗘墍鏈夊瓙鏌ヨ缁撴灉鏁村悎鎴愪竴涓繛璐殑绛旀
2. 鐩存帴鍥炵瓟鍘熷闂
3. 鍖呭惈缁撴灉涓殑鐩稿叧鏁板瓧鍜屾暟鎹?
4. 濡傛灉闂娑夊強姣旇緝鎴栬秼鍔匡紝璇峰睍绀?
5. 绠€娲佷絾瀹屾暣

## 鍥炵瓟锛堣鐢ㄤ腑鏂囷級
"""


async def result_synthesizer(state: AgentState) -> Dict[str, Any]:
    """
    Synthesize results from multiple sub-queries into final answer.
    
    This node:
    1. Collects all sub-query results from state
    2. Uses LLM to synthesize into a coherent answer
    3. Returns the final answer
    
    Args:
        state: Current graph state with sub_results
        
    Returns:
        State updates with final_answer
    """
    from data_agent.providers import get_llm
    
    original_question = state["user_question"]
    sub_questions = state.get("sub_questions", [])
    sub_results = state.get("sub_results", {})
    
    # Format sub-results for the prompt
    results_text = []
    for sq in sub_questions or []:
        sq_id = sq.get("id", "unknown")
        sq_question = sq.get("question", "")
        sq_purpose = sq.get("purpose", "")
        sq_result = sub_results.get(sq_id, {}) if sub_results else {}
        
        result_data = sq_result.get("data", []) if sq_result else []
        result_summary = _format_result_data(result_data)
        
        results_text.append(f"""
### Sub-Query: {sq_question}
**Purpose**: {sq_purpose}
**Result**: {result_summary}
""")
    
    if not results_text:
        # No sub-results, just return what we have
        return {
            "final_answer": "Unable to synthesize results: no sub-query results available.",
            "should_terminate": True,
        }
    
    # Build synthesis prompt
    prompt = SYNTHESIS_PROMPT.format(
        original_question=original_question,
        sub_results="\n".join(results_text),
    )
    
    try:
        llm = get_llm()
        response = await llm.ainvoke(prompt)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "final_answer": answer.strip(),
            "should_terminate": True,
        }
        
    except Exception as e:
        # On error, provide a basic answer from raw results
        return {
            "final_answer": f"Analysis complete. Results:\n" + "\n".join(results_text),
            "should_terminate": True,
        }


def _format_result_data(data: List[Dict[str, Any]], max_rows: int = 5) -> str:
    """Format result data for display in prompt."""
    if not data:
        return "No data returned"
    
    if len(data) == 1:
        # Single row result
        row = data[0]
        parts = [f"{k}: {v}" for k, v in row.items()]
        return ", ".join(parts)
    
    # Multiple rows - show summary
    preview = data[:max_rows]
    lines = []
    for row in preview:
        parts = [f"{k}: {v}" for k, v in row.items()]
        lines.append("  - " + ", ".join(parts))
    
    result = "\n".join(lines)
    if len(data) > max_rows:
        result += f"\n  ... and {len(data) - max_rows} more rows"
    
    return result
