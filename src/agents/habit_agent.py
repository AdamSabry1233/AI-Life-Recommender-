"""
Habit Agent — LLM only (no MF model backing).

Reasons purely from the user's intent to suggest one actionable habit or
routine that fits their current physical and mental state.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agents.state import AgentState

_SYSTEM = """\
You are the Habit Agent in an AI Life Recommender system.
Your job is to suggest ONE small, actionable habit or routine that fits the
user's current physical and mental state. Think in terms of daily habits:
sleep hygiene, movement, focus rituals, recovery, mindfulness, etc.
Keep it realistic — if they're physically tired, don't suggest an intense workout.

Respond on ONE line in exactly this format:
Habit: <action> — <one-sentence reason tied to the user's intent>\
"""

_HUMAN = """\
User intent: {intent}
{user_prefs}
Suggest the single most helpful habit for this person right now.\
"""


def make_node(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human",  _HUMAN),
    ])
    chain = prompt | llm | StrOutputParser()

    def habit_agent(state: AgentState) -> dict:
        prefs = state.get("user_preferences") or ""
        raw = chain.invoke({
            "intent":     state["intent"],
            "user_prefs": f"User preferences: {prefs}\n" if prefs else "",
        })
        habit, reason = _parse(raw)
        return {"habit_rec": {"habit": habit, "reason": reason}}

    return habit_agent


def _parse(text: str) -> tuple[str, str]:
    for line in text.strip().splitlines():
        if line.lower().startswith("habit:"):
            body = line.split(":", 1)[-1].strip()
            if " — " in body:
                habit, reason = body.split(" — ", 1)
                return habit.strip(), reason.strip()
            return body.strip(), ""
    return "Unknown", text.strip()
