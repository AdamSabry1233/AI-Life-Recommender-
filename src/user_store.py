"""
Persistent user profile store backed by Upstash Redis.

Each user is identified by a UUID stored in their browser (gr.BrowserState).
Their profile (onboarding answers + preferences) is saved as JSON in Redis
so recommendations are personalized across sessions and Space restarts.

Falls back to in-memory dict if Redis env vars are not set (local dev).
"""

import os
import json


def _make_store():
    url   = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        from upstash_redis import Redis
        print("Using Upstash Redis for user profiles.")
        return Redis(url=url, token=token)
    print("Redis not configured — using in-memory profile store.")
    return None


_redis = _make_store()
_local: dict = {}          # fallback for local dev

PROFILE_TTL = 60 * 60 * 24 * 90   # 90 days


def load_profile(user_id: str) -> dict | None:
    """Return saved profile dict for this user, or None if first visit."""
    if _redis:
        raw = _redis.get(f"profile:{user_id}")
        return json.loads(raw) if raw else None
    return _local.get(user_id)


def save_profile(user_id: str, profile: dict):
    """Persist the user's profile. Overwrites any existing entry."""
    if _redis:
        _redis.set(f"profile:{user_id}", json.dumps(profile), ex=PROFILE_TTL)
    else:
        _local[user_id] = profile


def append_feedback(user_id: str, feedback: dict):
    """
    Merge new feedback into the existing profile.
    feedback example: {"disliked_movies": ["Inception"], "dietary": "halal"}
    """
    profile = load_profile(user_id) or {}
    for key, val in feedback.items():
        if isinstance(val, list) and isinstance(profile.get(key), list):
            profile[key] = list(set(profile[key] + val))
        else:
            profile[key] = val
    save_profile(user_id, profile)
