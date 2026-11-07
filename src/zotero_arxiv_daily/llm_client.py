"""Bounded failover for chat completions on one OpenAI-compatible endpoint."""
import time
from types import SimpleNamespace

import openai
from loguru import logger


class FailoverClient:
    def __init__(self, client, models, total_timeout=240, model_timeout=90, cooldown=900, clock=time.monotonic):
        self._client = client
        self.models = list(dict.fromkeys(models))
        self.total_timeout = float(total_timeout)
        self.model_timeout = float(model_timeout)
        self.cooldown = float(cooldown)
        self.clock = clock
        self.failed_until = {}
        self.preferred = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        requested = kwargs.get("model")
        models = list(self.models)
        if requested and requested not in models:
            models.insert(0, requested)
        if requested in models:
            models.remove(requested)
            models.insert(0, requested)
        if self.preferred in models:
            models.remove(self.preferred)
            models.insert(0, self.preferred)
        now = self.clock()
        healthy = [m for m in models if self.failed_until.get(m, 0) <= now]
        cooling = [m for m in models if self.failed_until.get(m, 0) > now]
        cooling.sort(key=lambda m: self.failed_until[m])
        available = healthy + cooling
        budget = self.total_timeout
        if kwargs.get("timeout") is not None:
            budget = min(budget, float(kwargs["timeout"]))
        deadline = now + budget
        last_error = None
        for index, model in enumerate(available):
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            attempt = dict(kwargs, model=model, timeout=min(self.model_timeout, remaining / (len(available) - index)))
            try:
                result = self._client.chat.completions.create(**attempt)
            except (openai.APIConnectionError, openai.APIStatusError) as error:
                status = getattr(error, "status_code", None)
                body = getattr(error, "body", None)
                details = body.get("error", body) if isinstance(body, dict) else {}
                details = details if isinstance(details, dict) else {}
                code = str(details.get("code", "")).lower()
                message = str(details.get("message", "")).lower()
                if status in (401, 403) or any(value in code or value in message for value in ("insufficient_quota", "insufficient_user_quota", "余额不足")):
                    raise
                retryable = isinstance(error, openai.APIConnectionError) or status in (408, 429) or (isinstance(status, int) and 500 <= status < 600)
                if status in (400, 404):
                    retryable = (code in ("model_not_found", "model_unavailable", "model_not_available")
                                 or ("model" in message and any(phrase in message for phrase in
                                     ("not found", "not available", "unavailable", "does not exist", "not supported"))))
                if not retryable:
                    raise
                logger.warning("LLM model {} failed: type={} status={}", model, type(error).__name__, status)
                self.failed_until[model] = self.clock() + self.cooldown
                if self.preferred == model:
                    self.preferred = None
                last_error = error
                continue
            if self.preferred != model:
                logger.info("LLM active model: {}", model)
            self.preferred = model
            self.failed_until.pop(model, None)
            return result
        if last_error is not None:
            raise last_error
        raise TimeoutError("LLM request budget exhausted")
