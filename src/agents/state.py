from typing import TypedDict, Optional


class AgentState(TypedDict):
    intent:         str
    user_local_idx: int
    movie_rec:      Optional[dict]   # {name, reason}  ← Entertainment Agent
    food_rec:       Optional[dict]   # {name, reason}  ← Food Agent
    book_rec:       Optional[dict]   # {name, reason}  ← Learning Agent
    habit_rec:      Optional[dict]   # {habit, reason} ← Habit Agent
    final_response: Optional[str]    #                 ← Orchestrator
    memory_context:   Optional[str]   # retrieved past interactions (Phase 5)
    user_preferences: Optional[str]  # onboarding profile injected into every agent
    messages:         list           # conversation history for multi-turn chat
