"""
Phase 4/5 — LangGraph Multi-Agent Pipeline + Memory

Graph flow:

    START
      ↓
    retrieve_memory       (Phase 5 — FAISS lookup, injects memory_context)
      ↓
    entertainment_agent   (Movies MF + LLM)
      ↓
    food_agent            (Food MF + LLM)
      ↓
    learning_agent        (Books MF + LLM)
      ↓
    habit_agent           (LLM only)
      ↓
    synthesize            (Orchestrator — combines all 4 + memory context)
      ↓
    END

Usage:
    python src/graph.py
    python src/graph.py --intent "I want to relax but still feel productive"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

from agents.state import AgentState
from agents import entertainment_agent, food_agent, learning_agent, habit_agent, orchestrator
from recommender import MultiDomainRecommender

OLLAMA_BASE_URL = "http://host.docker.internal:11434"


def _make_retrieve_node(store):
    """Returns a no-op node if store is None (memory disabled)."""
    def retrieve_memory(state: AgentState) -> dict:
        if store is None:
            return {"memory_context": None}
        past    = store.search(state["intent"], k=3)
        context = store.format_context(past) if past else None
        return {"memory_context": context}
    return retrieve_memory


def make_llm(ollama_model: str = "llama3.1"):
    """Return the correct LLM based on INFERENCE_BACKEND env var."""
    if os.getenv("INFERENCE_BACKEND") == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
    return ChatOllama(model=ollama_model, base_url=OLLAMA_BASE_URL, temperature=0.3)


def build_graph(
    checkpoint_path: str = "mf_best.pt",
    data_dir: str = "data",
    ollama_model: str = "llama3.1",
    history_store=None,
    llm=None,             # pass in pre-built LLM or let build_graph create one
):
    print("Loading MF model …")
    rec = MultiDomainRecommender(checkpoint_path, data_dir)
    if llm is None:
        llm = make_llm(ollama_model)

    # ── nodes ─────────────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)

    builder.add_node("retrieve_memory",    _make_retrieve_node(history_store))
    builder.add_node("entertainment_agent", entertainment_agent.make_node(rec, llm))
    builder.add_node("food_agent",          food_agent.make_node(rec, llm))
    builder.add_node("learning_agent",      learning_agent.make_node(rec, llm))
    builder.add_node("habit_agent",         habit_agent.make_node(llm))
    builder.add_node("synthesize",          orchestrator.make_node(llm))

    # ── edges ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("retrieve_memory")
    builder.add_edge("retrieve_memory",    "entertainment_agent")
    builder.add_edge("entertainment_agent", "food_agent")
    builder.add_edge("food_agent",          "learning_agent")
    builder.add_edge("learning_agent",      "habit_agent")
    builder.add_edge("habit_agent",         "synthesize")
    builder.add_edge("synthesize",          END)

    print("Graph ready.\n")
    return builder.compile(), rec, llm   # return rec+llm so chat.py can reuse them


def run(intent: str, user_local_idx: int = 0, **graph_kwargs) -> str:
    """Invoke the graph and return a fully formatted recommendation string."""
    graph, _, _ = build_graph(**graph_kwargs)
    initial_state: AgentState = {
        "intent":         intent,
        "user_local_idx": user_local_idx,
        "movie_rec":      None,
        "food_rec":       None,
        "book_rec":       None,
        "habit_rec":      None,
        "final_response": None,
        "memory_context": None,
        "messages":       [],
    }
    state = graph.invoke(initial_state)
    return _format_output(state)


def _format_output(state: AgentState) -> str:
    """
    Format the final state into structured output.
    Individual agent results are printed directly from state (not re-processed
    by the LLM), so each agent's contribution is clearly visible.
    The orchestrator's summary ties them together beneath.
    """
    movie = state.get("movie_rec") or {}
    food  = state.get("food_rec")  or {}
    book  = state.get("book_rec")  or {}
    habit = state.get("habit_rec") or {}

    lines = [
        f"Movie:   {movie.get('name', '?')} — {movie.get('reason', '')}",
        f"Food:    {food.get('name',  '?')} — {food.get('reason',  '')}",
        f"Book:    {book.get('name',  '?')} — {book.get('reason',  '')}",
        f"Habit:   {habit.get('habit','?')} — {habit.get('reason', '')}",
        "",
        f"Summary: {state.get('final_response', '')}",
    ]
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--intent", "-i",
                   default="I want to relax but still feel productive")
    p.add_argument("--checkpoint", default="mf_best.pt")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--model",      default="llama3.1")
    args = p.parse_args()

    print(f"Intent: {args.intent}\n" + "─" * 60)
    output = run(
        intent=args.intent,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        ollama_model=args.model,
    )
    print(output)
