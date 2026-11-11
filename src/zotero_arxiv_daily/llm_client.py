"""Bounded failover for chat completions on one OpenAI-compatible endpoint."""
import asyncio
import time

import httpx
from types import SimpleNamespace

import openai
from loguru import logger


class FailoverClient:
    def __init__(self, client, models, total_timeout=240, model_timeout=90, cooldown=900, clock=time.monotonic, backup_client=None):
        self._client = client
        self.models = list(dict.fromkeys(models))
        self.total_timeout = float(total_timeout)
        self.model_timeout = float(model_timeout)
        self.cooldown = float(cooldown)
        self.clock = clock
        self.failed_until = {}
        self.preferred = None
        self.deadline = None  # Optional whole-generation monotonic deadline.
        self.backup_client = backup_client
        self.endpoint_failed_until = 0
        self.endpoint_error = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def _async_create(self, attempt):
        client = openai.AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=str(self._client.base_url),
            organization=self._client.organization,
            project=self._client.project,
            max_retries=0,
        )
        async with client:
            try:
                return await asyncio.wait_for(
                    client.chat.completions.create(**attempt), timeout=attempt["timeout"]
                )
            except asyncio.TimeoutError:
                request = httpx.Request("POST", str(self._client.base_url))
                raise openai.APITimeoutError(request=request) from None

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
        if self.deadline is not None:
            budget = min(budget, max(0, self.deadline - now))
        if kwargs.get("timeout") is not None:
            budget = min(budget, float(kwargs["timeout"]))
        request_deadline = now + budget
        if now < self.endpoint_failed_until and self.endpoint_error is not None:
            return self._backup_or_raise(kwargs, request_deadline, self.endpoint_error)
        # An independently configured endpoint gets part of this same budget.
        deadline = now + (budget / 2 if self.backup_client is not None else budget)
        last_error = None
        failed_attempts = 0
        for index, model in enumerate(available):
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            count = len(healthy) - index if index < len(healthy) else len(available) - index
            if model == self.preferred and model in healthy:
                count = 1
            attempt = dict(kwargs, model=model, timeout=min(self.model_timeout, remaining / count))
            try:
                if isinstance(self._client, openai.OpenAI):
                    result = asyncio.run(self._async_create(attempt))
                else:
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
                failed_attempts += 1
                self.failed_until[model] = self.clock() + self.cooldown
                if self.preferred == model:
                    self.preferred = None
                last_error = error
                continue
            if self.preferred != model:
                logger.info("LLM active model: {}", model)
            self.preferred = model
            self.endpoint_failed_until = 0
            self.endpoint_error = None
            self.failed_until.pop(model, None)
            return result
        if last_error is not None:
            if failed_attempts == len(available):
                self.endpoint_failed_until = self.clock() + min(60, self.cooldown)
                self.endpoint_error = last_error
            return self._backup_or_raise(kwargs, request_deadline, last_error)
        raise TimeoutError("LLM request budget exhausted")

    def _backup_or_raise(self, kwargs, deadline, error):
        remaining = deadline - self.clock()
        if self.backup_client is not None and remaining > 0:
            self.backup_client.deadline = self.deadline
            request = dict(kwargs, model=self.backup_client.models[0], timeout=remaining)
            return self.backup_client.chat.completions.create(**request)
        raise error
