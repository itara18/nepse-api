"""Small in-memory rate limiter for auth routes (per client IP).

Fine for one process. Behind several workers or replicas, use Redis instead.
"""
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_hits: dict[str, deque] = defaultdict(deque)


def reset() -> None:
    _hits.clear()


def limit(max_calls: int = 10, window_s: int = 60):
    def dependency(request: Request) -> None:
        key = f"{request.url.path}:{request.client.host if request.client else 'unknown'}"
        now = time.monotonic()
        q = _hits[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= max_calls:
            raise HTTPException(status_code=429, detail="Too many requests, slow down")
        q.append(now)

    return dependency
