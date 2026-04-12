"""
Orchestrator — synthesizes all 4 agent outputs into a final response.

This is the last node in the graph. It receives the completed AgentState
(all recs filled in) and produces a single cohesive recommendation.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState

_SYSTEM = """\
You are the Orchestrator of an AI Life Recommender system. Four specialized
agents have already chosen the best movie, food, book, and habit for the user.
Your only job is to write a 2-3 sentence Summary that explains how these four
choices work together to address the user's current state and intent.
Do NOT repeat or restate the individual recommendations — just synthesize them.\
"""

_HUMAN = """\
User intent: {intent}

{memory_section}
Agent recommendations:
- Movie (Entertainment Agent): {movie}
- Food (Food Agent): {food}
- Book (Learning Agent): {book}
- Habit (Habit Agent): {habit}

Write the Summary only.\
"""


def make_node(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    def synthesize(state: AgentState) -> dict:
        movie  = state.get("movie_rec")     or {}
        food   = state.get("food_rec")      or {}
        book   = state.get("book_rec")      or {}
        habit  = state.get("habit_rec")     or {}
        memory = state.get("memory_context") or ""

        memory_section = (f"{memory}\n" if memory else "")

        raw = chain.invoke({
            "intent":         state["intent"],
            "memory_section": memory_section,
            "movie":  f"{movie.get('name', '?')} — {movie.get('reason', '')}",
            "food":   f"{food.get('name',  '?')} — {food.get('reason',  '')}",
            "book":   f"{book.get('name',  '?')} — {book.get('reason',  '')}",
            "habit":  f"{habit.get('habit','?')} — {habit.get('reason', '')}",
        })
        return {"final_response": raw.strip()}

    return synthesize
