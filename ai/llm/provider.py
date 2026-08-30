"""
Provider implementation for OpenAI-compatible HTTP chat completion endpoints and mock testing.
Includes exponential backoff retry logic, structured output support, and observability metadata.
"""
from __future__ import annotations
import asyncio
import json
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from .base import LLMProvider, LLMResponse
from .config import LLMConfig
from .errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger("satquery.llm")


class OpenAICompatibleProvider(LLMProvider):
    """
    Standard HTTP-based OpenAI-compatible Chat Completions provider.
    Works with OpenAI, OpenRouter, Groq, Together, Ollama, vLLM, and any compliant endpoint.
    """

    TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        config: LLMConfig,
        transport: Optional[httpx.BaseTransport] = None,
        async_transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.config = config
        self.transport = transport
        self.async_transport = async_transport
        self._validate_config()

    def _validate_config(self) -> None:
        """Ensure critical configuration is valid before issuing calls."""
        if not self.config.base_url:
            raise LLMConfigurationError("LLM base_url must not be empty.", provider=self.config.provider)
        if not self.config.model:
            raise LLMConfigurationError("LLM model name must not be empty.", provider=self.config.provider)

    def _get_headers(self) -> Dict[str, str]:
        """Construct authorization and content headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SatQueryAI-LLM/1.0",
            "HTTP-Referer": "https://github.com/satquery-ai/satquery",
            "X-Title": "SatQuery AI",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Build standard OpenAI-compatible chat completions payload."""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    def _parse_response(self, response_data: Dict[str, Any], latency_ms: float) -> LLMResponse:
        """Parse raw completions JSON into standard LLMResponse."""
        try:
            choices = response_data.get("choices")
            if not choices or not isinstance(choices, list):
                raise LLMResponseError("LLM response contains no choices.", provider=self.config.provider)

            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content")

            if content is None:
                raise LLMResponseError("LLM response choice contains null content.", provider=self.config.provider)

            finish_reason = first_choice.get("finish_reason")
            usage = response_data.get("usage")

            return LLMResponse(
                content=content.strip(),
                model=response_data.get("model", self.config.model),
                provider=self.config.provider,
                latency_ms=round(latency_ms, 2),
                usage=usage,
                finish_reason=finish_reason,
                raw_data=response_data,
            )
        except (KeyError, TypeError) as e:
            raise LLMResponseError(f"Malformed LLM response structure: {e}", provider=self.config.provider) from e

    def _handle_http_error(self, exc: httpx.HTTPStatusError) -> LLMError:
        """Map HTTP error status codes to explicit typed LLM exceptions."""
        status = exc.response.status_code
        err_msg = f"HTTP {status} from {self.config.provider} ({self.config.model})"
        try:
            body = exc.response.json()
            if "error" in body:
                err_msg += f": {body['error'].get('message', body['error'])}"
        except Exception:
            err_msg += f": {exc.response.text[:200]}"

        if status in {401, 403}:
            return LLMAuthenticationError(err_msg, provider=self.config.provider, status_code=status)
        elif status == 429:
            return LLMRateLimitError(err_msg, provider=self.config.provider, status_code=status)
        elif status in self.TRANSIENT_STATUS_CODES:
            return LLMResponseError(err_msg, provider=self.config.provider, status_code=status)
        else:
            return LLMResponseError(err_msg, provider=self.config.provider, status_code=status)

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Execute synchronous completion with retry on transient failures."""
        endpoint = f"{self.config.base_url}/chat/completions"
        headers = self._get_headers()
        payload = self._build_payload(messages, temperature, max_tokens, response_format)

        last_exception: Optional[Exception] = None
        base_delay = 0.5

        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.perf_counter()
            try:
                with httpx.Client(timeout=self.config.timeout, transport=self.transport) as client:
                    resp = client.post(endpoint, headers=headers, json=payload)
                    resp.raise_for_status()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return self._parse_response(resp.json(), latency_ms)

            except httpx.TimeoutException as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                last_exception = LLMTimeoutError(
                    f"LLM request timed out after {self.config.timeout}s: {e}",
                    provider=self.config.provider,
                )
                logger.warning(f"LLM timeout on attempt {attempt}/{self.config.max_retries}")

            except httpx.HTTPStatusError as e:
                mapped_error = self._handle_http_error(e)
                last_exception = mapped_error
                # Non-transient errors fail immediately
                if e.response.status_code not in self.TRANSIENT_STATUS_CODES:
                    raise mapped_error from e
                logger.warning(f"LLM transient HTTP {e.response.status_code} on attempt {attempt}/{self.config.max_retries}")

            except (httpx.NetworkError, httpx.ConnectError) as e:
                last_exception = LLMNetworkError(f"Network failure connecting to LLM provider: {e}", provider=self.config.provider)
                logger.warning(f"LLM network error on attempt {attempt}/{self.config.max_retries}: {e}")

            except Exception as e:
                if isinstance(e, LLMError):
                    raise
                raise LLMResponseError(f"Unexpected error calling LLM: {e}", provider=self.config.provider) from e

            # Backoff before retry
            if attempt < self.config.max_retries:
                sleep_sec = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1), 5.0)
                time.sleep(sleep_sec)

        if last_exception:
            raise last_exception
        raise LLMResponseError("Max retries exceeded without valid response.", provider=self.config.provider)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Execute asynchronous completion with retry on transient failures."""
        endpoint = f"{self.config.base_url}/chat/completions"
        headers = self._get_headers()
        payload = self._build_payload(messages, temperature, max_tokens, response_format)

        last_exception: Optional[Exception] = None
        base_delay = 0.5

        for attempt in range(1, self.config.max_retries + 1):
            start_time = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout, transport=self.async_transport) as client:
                    resp = await client.post(endpoint, headers=headers, json=payload)
                    resp.raise_for_status()
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    return self._parse_response(resp.json(), latency_ms)

            except httpx.TimeoutException as e:
                last_exception = LLMTimeoutError(
                    f"LLM request timed out after {self.config.timeout}s: {e}",
                    provider=self.config.provider,
                )
                logger.warning(f"LLM async timeout on attempt {attempt}/{self.config.max_retries}")

            except httpx.HTTPStatusError as e:
                mapped_error = self._handle_http_error(e)
                last_exception = mapped_error
                if e.response.status_code not in self.TRANSIENT_STATUS_CODES:
                    raise mapped_error from e
                logger.warning(f"LLM async transient HTTP {e.response.status_code} on attempt {attempt}/{self.config.max_retries}")

            except (httpx.NetworkError, httpx.ConnectError) as e:
                last_exception = LLMNetworkError(f"Network failure connecting to LLM provider: {e}", provider=self.config.provider)
                logger.warning(f"LLM async network error on attempt {attempt}/{self.config.max_retries}: {e}")

            except Exception as e:
                if isinstance(e, LLMError):
                    raise
                raise LLMResponseError(f"Unexpected error calling LLM: {e}", provider=self.config.provider) from e

            # Async backoff before retry
            if attempt < self.config.max_retries:
                sleep_sec = min(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1), 5.0)
                await asyncio.sleep(sleep_sec)

        if last_exception:
            raise last_exception
        raise LLMResponseError("Max retries exceeded without valid response.", provider=self.config.provider)


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for deterministic offline testing and CI workflows.
    Does not make external network requests.
    """

    def __init__(
        self,
        default_response: str = '{"task": "vqa", "target": "building", "confidence": 0.95}',
        custom_handler: Optional[Callable[..., str]] = None,
        latency_ms: float = 5.0,
    ):
        self.default_response = default_response
        self.custom_handler = custom_handler
        self.latency_ms = latency_ms
        self.call_history: List[Dict[str, Any]] = []

    def _execute(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, str]],
    ) -> LLMResponse:
        self.call_history.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        })

        if self.custom_handler:
            content = self.custom_handler(messages=messages, temperature=temperature, response_format=response_format)
        else:
            content = self.default_response

        return LLMResponse(
            content=content,
            model="mock-llm-v1",
            provider="mock",
            latency_ms=self.latency_ms,
            usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            finish_reason="stop",
            raw_data={"mock": True},
        )

    def generate_sync(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        return self._execute(messages, temperature, max_tokens, response_format)

    async def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        return self._execute(messages, temperature, max_tokens, response_format)


def get_llm_provider(
    config: Optional[LLMConfig] = None,
    transport: Optional[httpx.BaseTransport] = None,
    async_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> LLMProvider:
    """
    Factory function instantiating the appropriate LLM provider.
    Defaults to LLMConfig.from_env() if config is omitted.
    """
    cfg = config or LLMConfig.from_env()
    provider_type = cfg.provider.lower()

    if provider_type in {"mock", "test", "testing"}:
        return MockLLMProvider()
    elif provider_type in {"openai", "openai_compatible", "openrouter", "groq", "together", "ollama", "vllm"}:
        return OpenAICompatibleProvider(cfg, transport=transport, async_transport=async_transport)
    else:
        raise LLMConfigurationError(
            f"Unsupported LLM provider '{cfg.provider}'. Supported options: 'openai_compatible', 'mock', 'openrouter', 'groq', 'together'.",
            provider=cfg.provider,
        )
