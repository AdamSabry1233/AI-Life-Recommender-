"""
Phase 6 — Gradio frontend.

Calls the FastAPI backend at API_URL (set via environment variable).
Each browser tab gets its own session_id so sessions don't bleed into each other.

Usage:
    python src/ui.py            (dev, expects API at localhost:8000)
    docker-compose up ui        (production, API_URL set by docker-compose)
"""

import os
import uuid
import time
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── wait for API to be ready ──────────────────────────────────────────────────

def wait_for_api(retries: int = 20, delay: float = 3.0):
    for i in range(retries):
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            if r.status_code == 200:
                print(f"API ready at {API_URL}")
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"Waiting for API … ({i+1}/{retries})")
        time.sleep(delay)
    raise RuntimeError(f"API at {API_URL} did not become ready in time.")

# ── API helpers ───────────────────────────────────────────────────────────────

def send_message(message: str, session_id: str) -> dict:
    r = requests.post(
        f"{API_URL}/chat",
        json={"message": message, "session_id": session_id},
        timeout=600,   # 5 agents × ~60s each on CPU
    )
    r.raise_for_status()
    return r.json()

def save_session(session_id: str):
    try:
        requests.post(
            f"{API_URL}/chat/save",
            json={"message": "", "session_id": session_id},
            timeout=10,
        )
    except Exception:
        pass

# ── Gradio UI ─────────────────────────────────────────────────────────────────

def chat(message: str, history: list, session_id: str) -> tuple[str, list, str]:
    if not session_id:
        session_id = str(uuid.uuid4())

    # Show user message immediately with a thinking placeholder
    history = history + [(message, "⏳ Thinking — this takes ~2 min while agents run locally …")]
    yield "", history, session_id

    try:
        data     = send_message(message, session_id)
        response = data["response"]
    except Exception as e:
        response = f"Error contacting API: {e}"

    history[-1] = (message, response)
    yield "", history, session_id


def start_new_session(session_id: str) -> tuple[list, str]:
    """Save current session, start fresh, and fire the first onboarding question."""
    if session_id:
        save_session(session_id)
    new_id = str(uuid.uuid4())
    try:
        r = requests.post(f"{API_URL}/session/start", params={"session_id": new_id}, timeout=10)
        first_q = r.json()["response"]
    except Exception:
        first_q = "Tell me how you're feeling and I'll make a recommendation!"
    return [(None, first_q)], new_id


# ── layout ────────────────────────────────────────────────────────────────────

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

    # submit on button click or Enter
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

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    wait_for_api()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
