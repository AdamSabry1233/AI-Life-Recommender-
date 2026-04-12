"""
HuggingFace Spaces entry point.

Downloads mf_best.pt from the HF Hub model repo at startup, then launches
the Gradio UI with direct ChatEngine calls (no FastAPI layer needed on Spaces).

Environment variables (set in Space settings):
    INFERENCE_BACKEND=groq
    GROQ_API_KEY=<your key>
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

HF_MODEL_REPO      = os.getenv("HF_MODEL_REPO",      "asabry1233/ai-recommender-mf")
CHECKPOINT_FILENAME = os.getenv("CHECKPOINT_FILENAME", "mf_best.pt")
DATA_DIR            = os.getenv("DATA_DIR",            "data")

print(f"Downloading {CHECKPOINT_FILENAME} from {HF_MODEL_REPO} …")
checkpoint_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename=CHECKPOINT_FILENAME,
)
print(f"Checkpoint ready at: {checkpoint_path}")

# ── load the engine ───────────────────────────────────────────────────────────
from engine import ChatEngine, ChatSession

print("Loading ChatEngine …")
engine = ChatEngine(
    checkpoint_path=checkpoint_path,
    data_dir=DATA_DIR,
)
print("Engine ready.")

# ── session store (per browser tab via gr.State) ──────────────────────────────
_sessions: dict[str, ChatSession] = {}

def get_session(session_id: str) -> ChatSession:
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession()
    return _sessions[session_id]

# ── Gradio helpers ────────────────────────────────────────────────────────────

def start_new_session(session_id: str):
    """Save current session (if any), create a fresh one, fire first question."""
    if session_id and session_id in _sessions:
        engine.save(_sessions[session_id])

    new_id  = str(uuid.uuid4())
    session = ChatSession()
    _sessions[new_id] = session
    first_q = engine.start_onboarding()
    return [(None, first_q)], new_id


def chat(message: str, history: list, session_id: str):
    """Streaming-style generator: show thinking placeholder, then real response."""
    if not session_id:
        session_id = str(uuid.uuid4())

    history = history + [(message, "⏳ Thinking — agents are running …")]
    yield "", history, session_id

    session  = get_session(session_id)
    try:
        response, _ = engine.process(session, message)
    except Exception as e:
        response = f"Sorry, something went wrong: {e}"

    history[-1] = (message, response)
    yield "", history, session_id

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

    session_id = gr.State(str(uuid.uuid4()))

    chatbot = gr.Chatbot(
        label="Conversation",
        height=500,
        show_copy_button=True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. I'm feeling tired but want to do something productive …",
            label="Your message",
            scale=5,
            lines=1,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("New session", variant="secondary")

    submit_btn.click(
        chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[msg_box, chatbot, session_id],
        queue=True,
    )
    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, session_id],
        outputs=[msg_box, chatbot, session_id],
        queue=True,
    )
    clear_btn.click(
        start_new_session,
        inputs=[session_id],
        outputs=[chatbot, session_id],
    )

    demo.load(
        start_new_session,
        inputs=[session_id],
        outputs=[chatbot, session_id],
    )

# ── launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue()
    demo.launch()
