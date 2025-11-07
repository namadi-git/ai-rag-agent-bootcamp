from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    UserPromptPart,
)

import tools

instructions = """
You are an AI assistant specialized in helping users find accurate, relevant information from Wikipedia.

Your job is to:
1) Understand the user’s question.
2) Search Wikipedia for relevant pages.
3) Retrieve the most useful full pages.
4) Synthesize a clear, correct answer with citations.

You have access to two tools:

1. search_wikipedia(subject: str) -> List[SearchQuery]
   - Searches Wikipedia for pages related to a topic.
   - Returns SearchQuery objects containing a page title and snippet.

2. fetch_wikipedia_page(title: str) -> WebPageContent
   - Retrieves the full content of a Wikipedia page.
   - Returns the page title, URL, and content text.

---------------------------------
SEARCH & ANSWERING STRATEGY
---------------------------------

Step 1 — Understand the question
- Identify the key topic/entity (typically 1–3 words).
- If helpful, reformulate the query into a simpler search phrase.

Step 2 — Initial search & selection
- Call search_wikipedia on the identified subject.
- Inspect titles/snippets; pick the top 3–5 most promising pages that directly relate to the question.

Step 3 — Mandatory fetching before synthesis
- You MUST call fetch_wikipedia_page at least **3 times** (i.e., fetch at least three distinct pages) before attempting to synthesize any final answer.
- Prefer diversity: do not fetch obviously duplicate or near-duplicate titles.

Step 4 — Iterative clarification (allowed during synthesis)
- If after the initial 3+ fetches the information is incomplete or unclear, you MAY continue using both tools while drafting the answer:
  - refine the search phrase (search_wikipedia),
  - fetch additional specific pages (fetch_wikipedia_page).
- Avoid redundant calls (don’t re-fetch the same title unless needed for a different mode).

Step 5 — Tool-call budget (hard cap)
- The total number of tool calls (search_wikipedia + fetch_wikipedia_page combined) MUST NOT exceed **10**.
- Manage your budget: start with 1–2 searches, then fetch 3–5 pages, then iterate only if necessary.

Step 6 — Synthesize
- Create a concise, accurate answer that directly addresses the user’s question.
- When multiple pages are relevant, combine information logically and resolve conflicts conservatively (prefer pages that are directly about the topic).
- Do NOT invent facts—only rely on content retrieved via the tools.

Step 7 — Cite Sources
- At the end of your answer, list the Wikipedia pages you actually used:
  - Page Title — URL

Failure case
- If, after using your available tool budget, you still cannot find a reliable answer, clearly state that Wikipedia did not provide sufficient information for the query and summarize what you attempted.

Additional rules
- Do not fabricate facts or URLs.
- Only use information retrieved through the tools.
- Keep responses well-structured, precise, and easy to read.
""".strip()

# ---------- Citations & references ----------

class Citation(BaseModel):
    """A single Wikipedia source used in the answer."""
    title: str = Field(..., min_length=1, description="Wikipedia page title")
    url: HttpUrl = Field(..., description="Canonical Wikipedia URL")

class Reference(BaseModel):
    """
    Optional supporting reference (local artifact).
    Useful if you also save extracted files/snippets.
    """
    title: str = Field(..., min_length=1)
    filename: str = Field(..., min_length=1)


# ---------- Structured sections ----------

class Section(BaseModel):
    """A structured section of the final write-up."""
    heading: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    references: List[Reference] = Field(default_factory=list)

# ---------- Final agent output ----------

class SearchResultArticle(BaseModel):
    """
    The final, user-facing answer from the Wikipedia agent.
    """
    model_config = ConfigDict(str_min_length=1)

    found_answer: bool = Field(..., description="True if sufficient Wikipedia info was found")
    question: str = Field(..., description="The user's original question")
    answer: str = Field(..., description="Concise synthesized answer for the user")
    sections: List[Section] = Field(default_factory=list, description="Optional structured breakdown")
    citations: List[Citation] = Field(default_factory=list, description="Wikipedia pages used")
    notes: Optional[str] = Field(None, description="Optional caveats/limits/assumptions")

    def format_markdown(self, base_path: str = "file_path/") -> str:
        """
        Render the article to Markdown.

        Args:
            base_url: Base URL for local 'Reference.filename' links in sections and global references.

        Returns:
            A Markdown string with Question, Answer, Sections (with references),
            Sources (citations), and optional Notes.
        """
        lines: list[str] = []

        # Header
        lines.append(f"# {self.question.strip()}")
        lines.append("")  # blank line

        # Found/Not found banner (optional)
        if not self.found_answer:
            lines.append("> **Note:** No conclusive answer found from Wikipedia sources.")
            lines.append("")

        # Answer
        lines.append("## Answer")
        lines.append("")
        lines.append(self.answer.strip() if self.answer else "_No answer provided._")
        lines.append("")

        # Sections
        if self.sections:
            lines.append("## Sections")
            lines.append("")
            for sec in self.sections:
                lines.append(f"### {sec.heading.strip()}")
                lines.append("")
                lines.append(sec.content.strip())
                lines.append("")
                if sec.references:
                    lines.append("#### References")
                    for ref in sec.references:
                        lines.append(f"- [{ref.title}]({base_path.rstrip('/')}/{ref.filename.lstrip('/')})")
                    lines.append("")
        # Citations (Wikipedia sources)
        if self.citations:
            lines.append("## Sources (Wikipedia)")
            for c in self.citations:
                lines.append(f"- [{c.title}]({str(c.url)})")
            lines.append("")

        # Notes
        if self.notes:
            lines.append("## Notes")
            lines.append(self.notes.strip())
            lines.append("")

        return "\n".join(lines).rstrip()



# =========================
# History Processor
# =========================

def force_answer_after_10_fetches(messages: list[ModelMessage]) -> list[ModelMessage]: 
    """
    If the conversation history already includes >= MAX_FETCH_CALLS calls to
    fetch_wikipedia_page, append a system nudge to synthesize an answer now.
    """
    num_tool_calls = 0

    for m in messages:
        for p in m.parts:
            if p.part_kind == 'tool-call' :#and p.tool_name == 'fetch_wikipedia_page':
                num_tool_calls = num_tool_calls + 1

    if num_tool_calls >= 10:
        print('forcing output')
        last_message = messages[-1]
        finish_prompt = 'System message: The maximal number of searches has exceeded. Proceed to finishing the writeup'
        finish_prompt_part = UserPromptPart(finish_prompt)
        last_message.parts.append(finish_prompt_part)

    return messages


# =========================
# Agent Config / Factory
# =========================
@dataclass
class AgentConfig:
    model: str = "openai:gpt-4o-mini"


class NamedCallback:
    """
    Simple streaming callback that logs tool calls/results.
    Useful for debugging and tracing tool usage.
    """

    def __init__(self, agent):
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            print("TOOL CALL →", event.part.tool_name, event.part.args_as_dict(), event.tool_call_id)
        # elif isinstance(event, FunctionToolResultEvent):
        #     print("TOOL RES  ←", event.result.tool_name, event.tool_call_id, event.result.content)

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)
    
def create_agent(config: AgentConfig | None = None) -> Agent:
    """
    Construct the Wikipedia search agent with tools, instructions, schema, and history processor.
    """
    config = config or AgentConfig()

    agent = Agent(
        name="wikipedia-searcher",
        instructions=instructions,  # your improved prompt string
        tools=[tools.get_search_queries, tools.fetch_wiki_page],
        model=config.model,
        output_type=SearchResultArticle,
        history_processors=[force_answer_after_10_fetches],
    )
    return agent

