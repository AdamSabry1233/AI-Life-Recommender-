"""
Shared recommendation engine — used by both chat.py (terminal) and api.py (FastAPI).

ChatEngine  : holds all shared resources (graph, LLM, rec model, FAISS store).
              One instance per process.
ChatSession : per-user state (current recommendations + conversation history).
              One instance per user/session.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState
from agents import habit_agent
from graph import build_graph, _format_output
from memory.embedder import Embedder
from memory.history_store import HistoryStore
from recommender import DOMAIN_MOVIES, DOMAIN_FOOD, DOMAIN_BOOKS
from llm_chain import format_candidates

OLLAMA_BASE_URL = "http://host.docker.internal:11434"

# ── prompts ───────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """\
You are a conversation classifier for an AI Life Recommender.
The user has received recommendations (movie, food, book, habit) and has typed a follow-up message.
Classify their message as exactly ONE of:
  question   — they want more info about a specific recommendation
  refinement — they want to change ONE recommendation with no concern or restriction
  concern    — they have a dietary, religious, health, or ethical concern about a recommendation
               (e.g. "I'm Muslim", "I'm vegan", "I'm allergic to nuts", "is there pork in this?")
  new_intent — they have a completely new mood or topic

Respond with ONE word only: question, refinement, concern, or new_intent\
"""

_DOMAIN_SYSTEM = """\
You are identifying which recommendation domain the user wants to refine.
Options: movie, food, book, habit
Respond with ONE word only: movie, food, book, or habit\
"""

_QA_SYSTEM = """\
You are an AI Life Recommender assistant. Answer helpfully and concisely
using the current recommendations and conversation history as context.\
"""

# ── onboarding questions ──────────────────────────────────────────────────────

ONBOARDING_QUESTIONS = [
    "👋 Welcome! Before I make recommendations, I have 4 quick questions to personalise them for you.\n\n"
    "**Step 1 of 4** — What movie or TV genres do you enjoy most? "
    "(e.g. action, comedy, drama, sci-fi, romance, documentary)",

    "**Step 2 of 4** — Do you have any dietary preferences or restrictions? "
    "(e.g. vegetarian, vegan, halal, gluten-free — or just say 'no restrictions')",

    "**Step 3 of 4** — Do you prefer fiction or non-fiction books? "
    "Any topics or authors you love?",

    "**Step 4 of 4 (last one!)** — What does your ideal evening look like when you want to unwind?",
]

# ── session ───────────────────────────────────────────────────────────────────

@dataclass
class ChatSession:
    current_state:      AgentState | None = None
    messages:           list = field(default_factory=list)
    onboarding_answers: list = field(default_factory=list)
    profile:            dict | None = None

    @property
    def onboarding_complete(self) -> bool:
        return self.profile is not None

# ── engine ────────────────────────────────────────────────────────────────────

class ChatEngine:
    """
    Initialise once per process. Pass a ChatSession per user request.
    All methods mutate the session and return (response_text, response_type).
    response_type: "recommendations" | "answer" | "refinement" | "history"
    """

    _DOMAIN_ID = {"movie": DOMAIN_MOVIES, "food": DOMAIN_FOOD, "book": DOMAIN_BOOKS}
    _STATE_KEY  = {"movie": "movie_rec",  "food": "food_rec",  "book": "book_rec"}
    _LABEL      = {"movie": "Movie",      "food": "Food",      "book": "Book"}

    def __init__(
        self,
        checkpoint_path: str = "mf_best.pt",
        data_dir: str = "data",
        ollama_model: str = "llama3.1",
    ):
        self.embedder = Embedder(model=ollama_model, base_url=OLLAMA_BASE_URL)
        self.store    = HistoryStore(store_dir=f"{data_dir}/memory", embedder=self.embedder)

        self.graph, rec, llm = build_graph(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            ollama_model=ollama_model,
            history_store=self.store,
        )
        self.llm        = llm
        self.rec        = rec
        self._habit_node = habit_agent.make_node(llm)

    # ── main dispatch ─────────────────────────────────────────────────────────

    def process(self, session: ChatSession, message: str) -> tuple[str, str]:
        """Route the message and return (response, type)."""
        # ── onboarding gate ───────────────────────────────────────────────────
        if not session.onboarding_complete:
            return self._process_onboarding(session, message)

        if session.current_state is None:
            if self.store.index is not None and self.store.index.ntotal > 0:
                if self.classify(session, message) == "question":
                    return self.answer_from_history(message), "history"
            return self.run_full(session, message), "recommendations"

        intent_type = self.classify(session, message)
        if intent_type == "question":
            return self.answer_question(session, message), "answer"
        elif intent_type == "refinement":
            return self.run_refinement(session, message), "refinement"
        elif intent_type == "concern":
            return self.handle_concern(session, message), "concern"
        else:
            self.save(session)
            session.messages = []
            return self.run_full(session, message), "recommendations"

    # ── full graph run ────────────────────────────────────────────────────────

    def run_full(self, session: ChatSession, intent: str) -> str:
        initial: AgentState = {
            "intent":           intent,
            "user_local_idx":   0,
            "movie_rec":        None,
            "food_rec":         None,
            "book_rec":         None,
            "habit_rec":        None,
            "final_response":   None,
            "memory_context":   None,
            "user_preferences": self._format_preferences(session.profile),
            "messages":         session.messages.copy(),
        }
        session.current_state = self.graph.invoke(initial)
        output = _format_output(session.current_state)
        session.messages.append({"role": "user",      "content": intent})
        session.messages.append({"role": "assistant", "content": output})
        return output

    # ── onboarding ────────────────────────────────────────────────────────────

    def start_onboarding(self) -> str:
        """Return the first onboarding question."""
        return ONBOARDING_QUESTIONS[0]

    def _process_onboarding(self, session: ChatSession, answer: str) -> tuple[str, str]:
        session.onboarding_answers.append(answer)
        step = len(session.onboarding_answers)

        if step < len(ONBOARDING_QUESTIONS):
            return ONBOARDING_QUESTIONS[step], "onboarding"

        # All answers collected — build and save profile
        keys = ["movie_genres", "dietary", "book_prefs", "lifestyle"]
        session.profile = dict(zip(keys, session.onboarding_answers))
        self.store.save({
            "intent":         f"onboarding: {self._format_preferences(session.profile)}",
            "movie_rec":      {},
            "food_rec":       {},
            "book_rec":       {},
            "habit_rec":      {},
            "final_response": "",
        })

        completion = (
            "Thanks! I've got your profile set up. 🎉\n\n"
            "Now tell me how you're feeling and I'll make recommendations "
            "tailored specifically to you."
        )
        return completion, "onboarding_complete"

    @staticmethod
    def _format_preferences(profile: dict | None) -> str:
        if not profile:
            return ""
        return (
            f"Movie genres: {profile.get('movie_genres', 'any')} | "
            f"Dietary: {profile.get('dietary', 'no restrictions')} | "
            f"Books: {profile.get('book_prefs', 'any')} | "
            f"Lifestyle: {profile.get('lifestyle', 'general')}"
        )

    # ── single-agent refinement ───────────────────────────────────────────────

    def run_refinement(self, session: ChatSession, message: str) -> str:
        domain = self.identify_domain(message)

        if domain == "habit":
            current_habit = (session.current_state.get("habit_rec") or {}).get("habit", "")
            refined_intent = (
                f"{session.current_state['intent']}. "
                f"Update: {message}. Do NOT suggest '{current_habit}' again."
            )
            update = self._habit_node({**session.current_state, "intent": refined_intent})
            session.current_state = {**session.current_state, **update}
            rec  = session.current_state.get("habit_rec") or {}
            line = f"Habit (updated): {rec.get('habit','?')} — {rec.get('reason','')}"
        else:
            domain_id    = self._DOMAIN_ID[domain]
            state_key    = self._STATE_KEY[domain]
            label        = self._LABEL[domain]
            current_name = (session.current_state.get(state_key) or {}).get("name", "")

            import random as _random
            candidates = self.rec.get_top_n(domain_id, 0, n=40)
            top, tail  = candidates[:3], candidates[3:]
            _random.shuffle(tail)
            candidates = (top + tail)
            filtered   = [c for c in candidates if c["name"] != current_name][:15]

            s = session.current_state
            current_recs = (
                f"Current recommendations (DO NOT repeat these):\n"
                f"  Movie: {(s.get('movie_rec') or {}).get('name', '?')}\n"
                f"  Food:  {(s.get('food_rec')  or {}).get('name', '?')}\n"
                f"  Book:  {(s.get('book_rec')  or {}).get('name', '?')}\n"
                f"  Habit: {(s.get('habit_rec') or {}).get('habit', '?')}\n"
            )
            prompt = (
                f"User intent: {session.current_state['intent']}\n"
                f"User refinement: {message}\n\n"
                f"{current_recs}\n"
                f"Do NOT suggest '{current_name}' — pick a different option.\n\n"
                f"{label} candidates:\n{format_candidates(filtered)}\n\n"
                f"Respond on ONE line: {label}: <name> — <one-sentence reason>"
            )
            result = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            name, reason = self._parse_line(result, label)
            session.current_state = {**session.current_state, state_key: {"name": name, "reason": reason}}
            line = f"{label} (updated): {name} — {reason}"

        session.messages.append({"role": "user",      "content": message})
        session.messages.append({"role": "assistant", "content": line})
        return line

    # ── follow-up Q&A ────────────────────────────────────────────────────────

    def answer_question(self, session: ChatSession, question: str) -> str:
        s     = session.current_state
        movie = (s.get("movie_rec")  or {})
        food  = (s.get("food_rec")   or {})
        book  = (s.get("book_rec")   or {})
        habit = (s.get("habit_rec")  or {})

        context = (
            f"Current recommendations:\n"
            f"  Movie: {movie.get('name','?')} — {movie.get('reason','')}\n"
            f"  Food:  {food.get('name','?')} — {food.get('reason','')}\n"
            f"  Book:  {book.get('name','?')} — {book.get('reason','')}\n"
            f"  Habit: {habit.get('habit','?')} — {habit.get('reason','')}\n"
        )
        history_str = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in session.messages[-6:]
        )
        msgs = [
            SystemMessage(content=_QA_SYSTEM),
            HumanMessage(content=f"{context}\nConversation:\n{history_str}\n\nQuestion: {question}"),
        ]
        answer = self.llm.invoke(msgs).content.strip()
        session.messages.append({"role": "user",      "content": question})
        session.messages.append({"role": "assistant", "content": answer})
        return answer

    # ── concern handler ───────────────────────────────────────────────────────

    def handle_concern(self, session: ChatSession, message: str) -> str:
        """
        1. Acknowledge and answer the concern (is there pork? is it vegan? etc.)
        2. Proactively replace the affected recommendation with the restriction in mind.
        """
        domain = self.identify_domain(message)
        s      = session.current_state or {}
        state_key = self._STATE_KEY.get(domain, "food_rec")
        current   = (s.get(state_key) or {}).get("name", "the recommendation")

        # Step 1 — answer the concern about the current item
        concern_prompt = (
            f"The user received this recommendation: {current}\n"
            f"They have raised this concern: {message}\n\n"
            f"In 1-2 sentences, honestly address whether their concern applies to '{current}'. "
            f"Be direct — if you're unsure, say so. "
            f"Then say you'll find a suitable alternative."
        )
        acknowledgement = self.llm.invoke(
            [HumanMessage(content=concern_prompt)]
        ).content.strip()

        # Step 2 — replace with restriction explicitly in the prompt
        replacement_line = self._replace_with_restriction(session, domain, message)

        response = f"{acknowledgement}\n\n{replacement_line}"
        session.messages.append({"role": "user",      "content": message})
        session.messages.append({"role": "assistant", "content": response})
        return response

    def _replace_with_restriction(self, session: ChatSession, domain: str, restriction: str) -> str:
        """Run a targeted replacement that bakes the dietary/ethical restriction into the prompt."""
        if domain == "habit":
            refined = (
                f"{session.current_state['intent']}. "
                f"Restriction: {restriction}. Suggest a different habit."
            )
            update = self._habit_node({**session.current_state, "intent": refined})
            session.current_state = {**session.current_state, **update}
            rec = session.current_state.get("habit_rec") or {}
            return f"Habit (updated): {rec.get('habit','?')} — {rec.get('reason','')}"

        domain_id    = self._DOMAIN_ID[domain]
        state_key    = self._STATE_KEY[domain]
        label        = self._LABEL[domain]
        current_name = (session.current_state.get(state_key) or {}).get("name", "")

        candidates = self.rec.get_top_n(domain_id, 0, n=20)
        filtered   = [c for c in candidates if c["name"] != current_name][:10]

        prompt = (
            f"User intent: {session.current_state['intent']}\n"
            f"Dietary/ethical restriction: {restriction}\n\n"
            f"Do NOT suggest '{current_name}'. "
            f"Pick an option that respects the restriction above.\n\n"
            f"{label} candidates:\n{format_candidates(filtered)}\n\n"
            f"Respond on ONE line: {label}: <name> — <one-sentence reason mentioning why it fits the restriction>"
        )
        result = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
        name, reason = self._parse_line(result, label)
        session.current_state = {**session.current_state, state_key: {"name": name, "reason": reason}}
        return f"{label} (updated): {name} — {reason}"

    # ── history recall (cross-session) ────────────────────────────────────────

    def answer_from_history(self, question: str) -> str:
        past = self.store.search(question, k=5)
        if not past:
            return "I don't have any past sessions saved yet."

        history_block = []
        for i, p in enumerate(past, 1):
            history_block.append(
                f"Session {i} (intent: \"{p['intent']}\"):\n"
                f"  Movie: {p.get('movie', {}).get('name', '?')}\n"
                f"  Food:  {p.get('food',  {}).get('name', '?')}\n"
                f"  Book:  {p.get('book',  {}).get('name', '?')}\n"
                f"  Habit: {p.get('habit', {}).get('habit', '?')}"
            )
        prompt = (
            "The following are the user's past recommendation sessions:\n\n"
            + "\n\n".join(history_block)
            + f"\n\nUser question: {question}\n\n"
            "Answer specifically and accurately using only the sessions above."
        )
        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

    # ── classifiers ───────────────────────────────────────────────────────────

    def classify(self, session: ChatSession, message: str) -> str:
        state = session.current_state or {}
        recs_summary = (
            f"Movie: {(state.get('movie_rec') or {}).get('name','?')}, "
            f"Food: {(state.get('food_rec')   or {}).get('name','?')}, "
            f"Book: {(state.get('book_rec')   or {}).get('name','?')}, "
            f"Habit: {(state.get('habit_rec') or {}).get('habit','?')}"
        )
        msgs = [
            SystemMessage(content=_CLASSIFY_SYSTEM),
            HumanMessage(content=f"Current recommendations: {recs_summary}\nUser message: {message}"),
        ]
        result = self.llm.invoke(msgs).content.strip().lower()
        for label in ("question", "refinement", "concern", "new_intent"):
            if label in result:
                return label
        return "new_intent"

    def identify_domain(self, message: str) -> str:
        msgs = [
            SystemMessage(content=_DOMAIN_SYSTEM),
            HumanMessage(content=f"User refinement request: {message}"),
        ]
        result = self.llm.invoke(msgs).content.strip().lower()
        for domain in ("movie", "food", "book", "habit"):
            if domain in result:
                return domain
        return "movie"

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, session: ChatSession):
        if session.current_state:
            self.store.save(session.current_state)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_line(text: str, label: str) -> tuple[str, str]:
        for line in text.strip().splitlines():
            if line.lower().startswith(label.lower() + ":"):
                body = line.split(":", 1)[-1].strip()
                if " — " in body:
                    name, reason = body.split(" — ", 1)
                    return name.strip(), reason.strip()
                return body.strip(), ""
        return "Unknown", text.strip()
