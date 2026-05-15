"""
HuggingFace Spaces entry point.

Downloads mf_best.pt from the HF Hub model repo at startup, then launches
the Gradio UI with direct ChatEngine calls (no FastAPI layer needed on Spaces).

Environment variables (set in Space settings):
    INFERENCE_BACKEND=groq
    GROQ_API_KEY=<your key>
    UPSTASH_REDIS_REST_URL=<your url>
    UPSTASH_REDIS_REST_TOKEN=<your token>
    HF_MODEL_REPO=asabry1233/ai-recommender-mf   (optional override)
    CHECKPOINT_FILENAME=mf_best.pt               (optional override)
"""

import os
import sys
import uuid

# ── ensure src/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── download checkpoint from HF Hub ──────────────────────────────────────────
from huggingface_hub import hf_hub_download

HF_MODEL_REPO       = os.getenv("HF_MODEL_REPO",      "asabry1233/ai-recommender-mf")
CHECKPOINT_FILENAME = os.getenv("CHECKPOINT_FILENAME", "mf_best.pt")
DATA_DIR            = os.getenv("DATA_DIR",            "data")

print(f"Downloading {CHECKPOINT_FILENAME} from {HF_MODEL_REPO} …")
checkpoint_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename=CHECKPOINT_FILENAME,
)
print(f"Checkpoint ready at: {checkpoint_path}")

# ── load engine + user store ──────────────────────────────────────────────────
from engine import ChatEngine, ChatSession
from user_store import load_profile, save_profile

# ── augmentation feature flags ────────────────────────────────────────────────
from semantic import live as s1_live
from clusters import live as s2_live
print(f"[features] semantic retrieval (S1): {'ON' if s1_live.enabled() else 'off'}")
print(f"[features] cluster conditioning (S2): {'ON' if s2_live.enabled() else 'off'}")

print("Loading ChatEngine …")
engine = ChatEngine(
    checkpoint_path=checkpoint_path,
    data_dir=DATA_DIR,
)
print("Engine ready.")

# ── in-process session store (per tab, current visit only) ────────────────────
_sessions: dict[str, ChatSession] = {}

def get_session(session_id: str) -> ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession()
    return _sessions[session_id]

# ── Gradio helpers ────────────────────────────────────────────────────────────

def load_user(user_id: str):
    """
    Called on page load. Creates a new tab session_id but checks Redis
    for a saved profile so returning users skip onboarding.
    Returns (history, session_id).
    """
    session_id = str(uuid.uuid4())
    session    = ChatSession()
    _sessions[session_id] = session

    saved_profile = load_profile(user_id) if user_id else None

    if saved_profile:
        # Returning user — restore profile, skip onboarding
        session.profile = saved_profile
        welcome = (
            f"Welcome back! 👋 I remember your preferences.\n\n"
            f"Tell me how you're feeling and I'll make recommendations for you."
        )
        return [{"role": "assistant", "content": welcome}], session_id
    else:
        # New user — start onboarding
        first_q = engine.start_onboarding()
        return [{"role": "assistant", "content": first_q}], session_id


def start_new_session(session_id: str, user_id: str):
    """New session button — save current session, start fresh chat (keep profile)."""
    if session_id and session_id in _sessions:
        engine.save(_sessions[session_id])

    new_id  = str(uuid.uuid4())
    session = ChatSession()
    _sessions[new_id] = session

    # Restore profile so they don't re-do onboarding
    saved_profile = load_profile(user_id) if user_id else None
    if saved_profile:
        session.profile = saved_profile
        msg = "Starting a new session! Tell me how you're feeling."
    else:
        msg = engine.start_onboarding()

    return [{"role": "assistant", "content": msg}], new_id


PLACEHOLDER_ONBOARDING = "Type your answer to the question above and press Enter …"
PLACEHOLDER_CHAT       = "Describe how you're feeling — e.g. I'm tired but want to be productive …"


def chat(message: str, history: list, session_id: str, user_id: str):
    """Streaming-style generator: show thinking placeholder, then real response."""
    if not session_id:
        session_id = str(uuid.uuid4())

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "⏳ Thinking — agents are running …"},
    ]
    yield "", history, session_id, gr.update(placeholder=PLACEHOLDER_ONBOARDING)

    session = get_session(session_id)
    try:
        response, _ = engine.process(session, message)
        # Save profile to Redis whenever onboarding completes
        if session.onboarding_complete and user_id:
            save_profile(user_id, session.profile)
    except Exception as e:
        response = f"Sorry, something went wrong: {e}"

    # Switch placeholder once onboarding is done
    placeholder = PLACEHOLDER_CHAT if session.onboarding_complete else PLACEHOLDER_ONBOARDING
    history[-1] = {"role": "assistant", "content": response}
    yield "", history, session_id, gr.update(placeholder=placeholder)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
import gradio as gr

with gr.Blocks(title="AI Life Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AI Life Recommender
        Tell me how you're feeling and I'll recommend a **movie**, **meal**, **book**, and **habit** tailored to your mood.
        Ask follow-up questions or request a different recommendation at any time.
        """
    )

    # Persists in the browser across page refreshes — identifies returning users
    user_id    = gr.BrowserState(str(uuid.uuid4()))
    session_id = gr.State("")

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        show_copy_button=True,
        type="messages",
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder=PLACEHOLDER_ONBOARDING,
            label="Your message",
            scale=5,
            lines=1,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("New session", variant="secondary")

    submit_btn.click(
        chat,
        inputs=[msg_box, chatbot, session_id, user_id],
        outputs=[msg_box, chatbot, session_id, msg_box],
        queue=True,
    )
    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, session_id, user_id],
        outputs=[msg_box, chatbot, session_id, msg_box],
        queue=True,
    )
    clear_btn.click(
        start_new_session,
        inputs=[session_id, user_id],
        outputs=[chatbot, session_id],
    )

    demo.load(
        load_user,
        inputs=[user_id],
        outputs=[chatbot, session_id],
    )

# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue()
    demo.launch()
