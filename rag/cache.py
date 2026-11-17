"""Redis answer cache with explicit degradation.

A cache outage falls back to a miss and is reported once. GraphService includes
the corpus content/version signature in each key; clearing on writes is only
best-effort cleanup, so a failed clear cannot make a later corpus reuse old keys.
"""
import hashlib
import json
import threading


def cache_key(fingerprint, question, mode, top_k, threshold, rerank, variant="query", corpus_signature=""):
    material = json.dumps([fingerprint, variant, corpus_signature, question, mode, top_k, threshold, rerank],
                          ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, url=None, client=None, ttl=3600, prefix="raganswer"):
        self.ttl, self.prefix = ttl, prefix
        self._client = client
        self._url = (url or "").strip()
        self._warned = threading.Event()
        if client is None and self._url:
            import redis  # optional dependency; only needed when a cache URL is configured
            self._client = redis.Redis.from_url(self._url, decode_responses=True,
                                                socket_connect_timeout=2, socket_timeout=2)

    @property
    def enabled(self):
        return self._client is not None

    def _degrade(self, action, error):
        if not self._warned.is_set():
            self._warned.set()
            print(f"[cache] Redis unavailable, answering without cache ({action}: {error})", flush=True)

    def get(self, key):
        if not self.enabled:
            return None
        try:
            raw = self._client.get(f"{self.prefix}:{key}")
            return json.loads(raw) if raw else None
        except Exception as error:
            self._degrade("read", error)
            return None

    def put(self, key, value):
        if not self.enabled:
            return
        try:
            self._client.set(f"{self.prefix}:{key}", json.dumps(value, ensure_ascii=False), ex=self.ttl)
        except Exception as error:
            self._degrade("write", error)

    def clear(self):
        if not self.enabled:
            return
        try:
            keys = list(self._client.scan_iter(f"{self.prefix}:*", count=500))
            if keys:
                self._client.delete(*keys)
        except Exception as error:
            self._degrade("clear", error)
