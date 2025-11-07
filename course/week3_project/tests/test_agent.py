from agent_main import run_agent_sync
from tests.utils import get_tool_calls
from pydantic import HttpUrl

def test_agent_makes_tool_calls():
    user_prompt = "Where did mandela live?"
    result = run_agent_sync(user_prompt)
    print(result.output.format_markdown( ))

    search_tool_calls = get_tool_calls(result, 'get_search_queries')
    fetch_tool_calls = get_tool_calls(result, 'fetch_wiki_page')
    assert len(search_tool_calls) >= 1, f"Expected at least 1 search tool calls, got {len(search_tool_calls)}"
    assert len(fetch_tool_calls) >= 2, f"Expected at least 2 fetch tool calls, got {len(search_tool_calls)}"

def test_agent_adds_references():
    user_prompt = "Where did mandela live?"
    result = run_agent_sync(user_prompt)
    print(result.output.format_markdown( ))

    search_tool_calls = get_tool_calls(result, 'get_search_queries')
    fetch_tool_calls = get_tool_calls(result, 'fetch_wiki_page')
    assert len(search_tool_calls) >= 1, f"Expected at least 1 search tool calls, got {len(search_tool_calls)}"
    assert len(fetch_tool_calls) >= 2, f"Expected at least 2 fetch tool calls, got {len(search_tool_calls)}"

    assert len(result.output.citations) > 0, "Expected at least one citation in the article"
    for citation in result.output.citations:
        assert isinstance(citation.url, HttpUrl), f"Citation URL is not a valid HttpUrl: {citation.url}"