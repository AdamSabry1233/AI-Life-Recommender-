"""
Phase 5 — Terminal multi-turn chat.

Thin I/O wrapper around ChatEngine (src/engine.py).

Usage:
    docker-compose exec api python src/chat.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from engine import ChatEngine, ChatSession


def main(checkpoint_path="mf_best.pt", data_dir="data", ollama_model="llama3.1"):
    print("Initialising …")
    engine  = ChatEngine(checkpoint_path, data_dir, ollama_model)
    session = ChatSession()

    print("\nAI Life Recommender — multi-turn chat")
    print("Tell me how you're feeling, ask follow-ups, or type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            engine.save(session)
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            engine.save(session)
            print("Goodbye!")
            break

        response, _ = engine.process(session, user_input)
        print(f"\nAssistant:\n{response}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="mf_best.pt")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--model",      default="llama3.1")
    args = p.parse_args()
    main(args.checkpoint, args.data_dir, args.model)
