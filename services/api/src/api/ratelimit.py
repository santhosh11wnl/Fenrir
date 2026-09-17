"""Per-client rate limiting.

A sliding window held in process memory. Two honest limitations:

* **Per-replica.** Three replicas means three times the configured rate. Move
  to Redis when you scale out.
* **Keyed by client IP**, which is a proxy for identity, not identity itself.
  Once real auth lands (see the auth task), key on the authenticated subject
  instead -- IP keying penalises everyone behind one NAT and is trivially
  sidestepped from a pool of addresses.

It is here anyway because the failure it prevents is expensive: an unbounded
client loop against a metered model API. A crude ceiling beats none.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, requests_per_minute: int, window_seconds: float = 60.0) -> None:
        self._limit = requests_per_minute
        self._window = window_seconds
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        """Record an attempt. Returns False when the caller is over the limit."""
        now = time.monotonic()
        window = self._hits[key]

        cutoff = now - self._window
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._limit:
            return False

        window.append(now)
        return True

    def retry_after(self, key: str) -> int:
        """Whole seconds until the caller's oldest hit falls out of the window."""
        window = self._hits.get(key)
        if not window:
            return 0
        return max(1, int(self._window - (time.monotonic() - window[0])) + 1)

    def prune(self) -> None:
        """Drop keys with no recent activity.

        Without this the map grows one entry per distinct client IP forever,
        which is a slow leak with a fast trigger on a public endpoint.
        """
        cutoff = time.monotonic() - self._window
        for key in [k for k, w in self._hits.items() if not w or w[-1] < cutoff]:
            del self._hits[key]


__all__ = ["SlidingWindowLimiter"]
