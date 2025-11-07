import pytest 

from pydantic_ai import Agent, AgentRunResult
from pydantic import BaseModel


from tests.utils import get_tool_calls
from agent_main import run_agent_sync, run_agent


judge_instructions = """
you are an expert judge evaluating the performance of an AI search agent.
"""


class JudgeCriterion(BaseModel):
    criterion_description: str
    passed: bool
    judgement: str


class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion]
    feedback: str


def create_judge():
    judge = Agent(
        name="judge",
        instructions=judge_instructions,
        model="openai:gpt-4o-mini",
        output_type=JudgeFeedback,
    )
    return judge


async def evaluate_agent_performance(
        criteria: list[str],
        result: AgentRunResult,
        output_transformer: callable = None) -> JudgeFeedback:
    
    search_tool_calls = get_tool_calls(result, 'get_search_queries')
    fetch_tool_calls = get_tool_calls(result, 'fetch_wiki_page')

    tool_calls = search_tool_calls + fetch_tool_calls

    tool_calls_str = [str(call) for call in tool_calls]

    output = result.output
    if output_transformer is not None:
        output = output_transformer(output)

    user_prompt = f"""
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{'\n'.join(criteria)}
</CRITERIA>

The agent's final output was:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls:
<TOOL_CALLS>
{'\n'.join([str(c) for c in tool_calls])}
</TOOL_CALLS>
    """

    print("Judge Prompt:", user_prompt)

    judge = create_judge()
    judge_result = await judge.run(user_prompt)
    return judge_result.output


@pytest.mark.asyncio
async def test_wikiagent():
    user_prompt = "Tell me about Nelson Mandela's life and where he lived."

    result = await run_agent(user_prompt)

    print(result.output.format_markdown())
    
    criteria = [
        "agent made at least one  get_search_queries tool calls",
        "agent made at least two  fetch_wiki_page tool calls",
        "article contains at least one reference with a valid wikipedia URL"
    ]

    judge_feedback = await evaluate_agent_performance(
        criteria,
        result,
        output_transformer=lambda output: output.format_markdown()
    )

    print(judge_feedback)

    for criterion in judge_feedback.criteria:
        assert criterion.passed, f"Criterion failed: {criterion.criterion_description}, {criterion.judgement}"