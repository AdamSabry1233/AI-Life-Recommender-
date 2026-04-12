"""
Learning Agent — Books domain.

Gets top-N book candidates from the MF model, then uses the LLM to pick
the single best fit for the user's current intent.
"""

import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState
from recommender import MultiDomainRecommender, DOMAIN_BOOKS
from llm_chain import format_candidates

_SYSTEM = """\
You are the Learning Agent in an AI Life Recommender system.
A machine learning model has ranked these books based on this user's taste profile.
Pick the single best book given the user's current mood and cognitive state.
Consider how much mental energy the user actually has right now.

Respond on ONE line in exactly this format:
Book: <title> — <one-sentence reason tied to the user's intent>\
"""

_HUMAN = """\
User intent: {intent}
{user_prefs}
Book candidates:
{candidates}

Choose the best fit.\
"""


def make_node(rec: MultiDomainRecommender, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    def learning_agent(state: AgentState) -> dict:
        # Fetch extra candidates and shuffle the tail so the LLM sees variety.
        # The top-3 stay fixed (best MF matches), the rest are shuffled.
        candidates = rec.get_top_n(DOMAIN_BOOKS, state["user_local_idx"], n=20)
        top, tail  = candidates[:3], candidates[3:]
        random.shuffle(tail)
        candidates = (top + tail)[:10]
        prefs = state.get("user_preferences") or ""
        raw = chain.invoke({
            "intent":     state["intent"],
            "user_prefs": f"User preferences: {prefs}\n" if prefs else "",
            "candidates": format_candidates(candidates),
        })
        name, reason = _parse(raw)
        return {"book_rec": {"name": name, "reason": reason}}

    return learning_agent


def _parse(text: str) -> tuple[str, str]:
    for line in text.strip().splitlines():
        if line.lower().startswith("book:"):
            body = line.split(":", 1)[-1].strip()
            if " — " in body:
                name, reason = body.split(" — ", 1)
                return name.strip(), reason.strip()
            return body.strip(), ""
    return "Unknown", text.strip()
