from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalResponse:
    content: str
    model: str
    tokens: int
    latency_ms: float
    from_local: bool


class LocalExecutor:
    """
    Runs tasks against a locally hosted model via Ollama or LM Studio.

    Ollama is tried first (port 11434). LM Studio is tried second (port 1234).
    If neither is reachable and a fallback Anthropic client is configured,
    the request is sent to Sonnet instead of failing outright.
    """

    def __init__(
        self,
        model: str = "qwen3:32b",
        ollama_url: str = "http://localhost:11434",
        lmstudio_url: str = "http://localhost:1234",
        fallback_client=None,
        fallback_model: str = "claude-sonnet-4-5",
        timeout: int = 120,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.lmstudio_url = lmstudio_url.rstrip("/")
        self._fallback = fallback_client
        self._fallback_model = fallback_model
        self._timeout = timeout
        self._local_calls = 0
        self._fallback_calls = 0
        self._total_tokens = 0
        self._available: Optional[bool] = None

    def run(
        self,
        task: str,
        system: str = "",
        max_tokens: int = 2000,
        force_local: bool = False,
    ) -> LocalResponse:
        start = time.time()

        if self._is_ollama_available():
            result = self._call_ollama(task, system, max_tokens)
            if result:
                self._local_calls += 1
                return LocalResponse(
                    content=result["content"],
                    model=self.model,
                    tokens=result.get("tokens", 0),
                    latency_ms=(time.time() - start) * 1000,
                    from_local=True,
                )

        result = self._call_lmstudio(task, system, max_tokens)
        if result:
            self._local_calls += 1
            return LocalResponse(
                content=result["content"],
                model=self.model,
                tokens=result.get("tokens", 0),
                latency_ms=(time.time() - start) * 1000,
                from_local=True,
            )

        if force_local:
            raise RuntimeError(
                f"Local model unavailable ({self.model}). "
                "Is Ollama or LM Studio running?"
            )

        if self._fallback:
            self._fallback_calls += 1
            return self._call_fallback(task, system, max_tokens, start)

        raise RuntimeError(
            "Local model unavailable and no fallback client configured."
        )

    def _call_ollama(self, task: str, system: str, max_tokens: int) -> Optional[dict]:
        payload: dict = {
            "model": self.model,
            "prompt": task,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        return self._post(f"{self.ollama_url}/api/generate", payload, "response", "eval_count")

    def _call_lmstudio(self, task: str, system: str, max_tokens: int) -> Optional[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": task})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.lmstudio_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
                content = body["choices"][0]["message"]["content"]
                tokens = body.get("usage", {}).get("completion_tokens", 0)
                return {"content": content, "tokens": tokens}
        except Exception:
            return None

    def _post(self, url: str, payload: dict, content_key: str, token_key: str) -> Optional[dict]:
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
                return {
                    "content": body.get(content_key, ""),
                    "tokens": body.get(token_key, 0),
                }
        except Exception:
            return None

    def _call_fallback(
        self,
        task: str,
        system: str,
        max_tokens: int,
        start: float,
    ) -> LocalResponse:
        kwargs: dict = {
            "model": self._fallback_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": task}],
        }
        if system:
            kwargs["system"] = system

        response = self._fallback.messages.create(**kwargs)
        content = response.content[0].text
        tokens = response.usage.output_tokens
        self._total_tokens += tokens

        return LocalResponse(
            content=content,
            model=self._fallback_model,
            tokens=tokens,
            latency_ms=(time.time() - start) * 1000,
            from_local=False,
        )

    def _is_ollama_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2):
                self._available = True
                return True
        except Exception:
            self._available = False
            return False

    def reset_availability_cache(self) -> None:
        """Force a fresh availability check on the next call."""
        self._available = None

    def stats(self) -> dict:
        total = self._local_calls + self._fallback_calls
        return {
            "local_calls": self._local_calls,
            "fallback_calls": self._fallback_calls,
            "total_calls": total,
            "local_rate": self._local_calls / max(1, total),
            "model": self.model,
            "ollama_url": self.ollama_url,
        }
