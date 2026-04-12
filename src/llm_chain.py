"""
Phase 3 — LLM Reasoning Layer

Connects to Ollama (llama3.1 by default) via LangChain and reasons over
MF-generated candidates to produce a structured, intent-aware recommendation.
"""

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an AI Life Recommender. A machine learning model has pre-selected \
candidate items across three domains based on the user's taste profile. \
Your job is to pick the single best fit from each domain given the user's \
current mood or goal, and explain why in one sentence.

You MUST respond in exactly this format with no extra text and no skipped lines:
Movie: <title> — <one-sentence reason>
Food: <dish or recipe name> — <one-sentence reason>
Book: <title> — <one-sentence reason>
Summary: <2-3 sentences connecting all three choices to the user's intent>\
"""

_HUMAN = """\
User's intent: {intent}

Movie candidates:
{movie_candidates}

Food candidates:
{food_candidates}

Book candidates:
{book_candidates}

Choose the best option from each list and explain your reasoning.\
"""

# ── public API ────────────────────────────────────────────────────────────────

def build_chain(model: str = "llama3.1", base_url: str = "http://host.docker.internal:11434"):
    """Return a LangChain runnable: dict → recommendation string."""
    llm    = ChatOllama(model=model, base_url=base_url, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])
    return prompt | llm | StrOutputParser()


def format_candidates(candidates: list[dict]) -> str:
    """Format a list of candidate dicts into a numbered prompt block."""
    lines = []
    for i, c in enumerate(candidates, 1):
        tags = f" [{c['tags']}]" if c.get("tags") else ""
        lines.append(f"{i}. {c['name']}{tags}")
    return "\n".join(lines)
