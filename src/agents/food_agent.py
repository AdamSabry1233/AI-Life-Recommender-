"""
Food Agent — Food domain.

Gets top-N recipe candidates from the MF model, then uses the LLM to pick
the single best fit for the user's current intent.
"""

import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState
from recommender import MultiDomainRecommender, DOMAIN_FOOD
from llm_chain import format_candidates

_SYSTEM = """\
You are the Food Agent in an AI Life Recommender system.
A machine learning model has ranked these recipes based on this user's taste profile.
Pick the single best meal given the user's current mood, energy level, and physical state.

Respond on ONE line in exactly this format:
Food: <dish name> — <one-sentence reason tied to the user's intent>\
"""

_HUMAN = """\
User intent: {intent}
{user_prefs}
Food candidates:
{candidates}

Choose the best fit.\
"""


def make_node(rec: MultiDomainRecommender, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    def food_agent(state: AgentState) -> dict:
        candidates = rec.get_top_n(DOMAIN_FOOD, state["user_local_idx"], n=30)
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
        return {"food_rec": {"name": name, "reason": reason}}

    return food_agent


def _parse(text: str) -> tuple[str, str]:
    for line in text.strip().splitlines():
        if line.lower().startswith("food:"):
            body = line.split(":", 1)[-1].strip()
            if " — " in body:
                name, reason = body.split(" — ", 1)
                return name.strip(), reason.strip()
            return body.strip(), ""
    return "Unknown", text.strip()
