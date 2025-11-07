import main 
from tests.utils import get_tool_calls
from search_agent import SearchResultArticle


def test_agent_makes_3_search_calls():
    user_prompt = "What is LLM evaluation?"
    result = main.run_agent_sync(user_prompt)

    print(result.output.format_article())

    tool_calls = get_tool_calls(result)
    assert len(tool_calls) >= 3, f"Expected at least 3 tool calls, got {len(tool_calls)}"

    article: SearchResultArticle = result.output
    assert len(article.sections) > 0, "Expected at least one section in the article"


def test_agent_adds_references():
    user_prompt = "What is LLM evaluation?"
    result = main.run_agent_sync(user_prompt)

    article: SearchResultArticle = result.output
    print(article.format_article())

    tool_calls = get_tool_calls(result)
    assert len(tool_calls) >= 3, f"Expected at least 3 tool calls, got {len(tool_calls)}"

    assert len(article.sections) > 0, "Expected at least one section in the article"
    assert len(article.references) > 0, "Expected at least one reference in the article"


def test_agent_code():
    user_prompt = "How do I implement LLM as a Judge eval?"
    result = main.run_agent_sync(user_prompt)

    article: SearchResultArticle = result.output
    print(article.format_article())

    assert len(article.sections) > 0, "Expected at least one section in the article"

    found_code = False

    for section in article.sections:
        if "```python" in section.content:
            found_code = True

    assert found_code, "Expected at least one code block in the article"