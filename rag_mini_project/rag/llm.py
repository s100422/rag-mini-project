"""Gemini 호출 공통 래퍼: 재시도, 사용량·시간 집계, LangSmith 추적.

생성·Agent·판정이 모두 generate()로 호출한다. 호출마다 토큰(입력·출력·생각)과 시간을 돌려줘
문항별 사용량을 합산할 수 있다.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from langsmith import traceable

from . import config  # noqa: F401  (.env 로드)

GEN_MODEL = os.getenv("GEN_MODEL", "models/gemini-3.8-flash")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "models/gemini-2.5-pro")
RETRYABLE = ("429", "500", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE",
             "timeout", "Timeout", "disconnected", "ConnectError", "ReadError")

_client: genai.Client | None = None
_lock = threading.Lock()


def _get_client() -> genai.Client:
    """스레드가 동시에 처음 부르면 클라이언트가 둘 생기고, 덮어써진 쪽이 정리되며 쓰던 연결이 닫힌다. 잠금으로 막는다."""
    global _client
    with _lock:
        if _client is None:
            _client = genai.Client(http_options=types.HttpOptions(timeout=180_000))
        return _client


def _reset_client() -> None:
    global _client
    with _lock:
        _client = None


class PromptBlocked(RuntimeError):
    """API의 안전 장치가 프롬프트를 차단했다(block_reason). 답변 없음으로 기록한다."""


@dataclass
class LLMResult:
    text: str
    parsed: object
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: float
    model_version: str


def _inputs(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k != "schema"}


@traceable(run_type="llm", name="gemini", process_inputs=_inputs)
def _call(model: str, prompt: str, system: str | None, schema, temperature: float,
          thinking_budget: int | None, max_output_tokens: int) -> dict:
    cfg = dict(system_instruction=system, temperature=temperature, max_output_tokens=max_output_tokens)
    if schema is not None:
        cfg |= dict(response_mime_type="application/json", response_schema=schema)
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    t = time.perf_counter()
    r = _get_client().models.generate_content(model=model, contents=prompt,
                                              config=types.GenerateContentConfig(**cfg))
    fb = getattr(r, "prompt_feedback", None)
    if fb is not None and fb.block_reason:  # API가 요청을 막으면 모델은 아무것도 생성하지 않는다(재시도해도 같다)
        raise PromptBlocked(str(fb.block_reason.name))
    u = r.usage_metadata
    inp, out, think = (u.prompt_token_count or 0), (u.candidates_token_count or 0), (u.thoughts_token_count or 0)
    return {
        "text": r.text or "", "parsed": r.parsed if schema is not None else None,
        "input_tokens": inp, "output_tokens": out, "thinking_tokens": think,
        "latency_ms": (time.perf_counter() - t) * 1000, "model_version": r.model_version or model,
        "usage_metadata": {"input_tokens": inp, "output_tokens": out + think, "total_tokens": inp + out + think},
    }


def generate(model: str, prompt: str, system: str | None = None, schema=None, temperature: float = 0.0,
             thinking_budget: int | None = None, max_output_tokens: int = 2048, retries: int = 6) -> LLMResult:
    """재시도 포함 호출. schema(pydantic)를 주면 JSON으로 받아 parsed에 넣는다."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            d = _call(model, prompt, system, schema, temperature, thinking_budget, max_output_tokens)
            if schema is not None and d["parsed"] is None:
                raise ValueError("JSON 응답을 해석하지 못했습니다(출력 한도 초과일 수 있음)")
            return LLMResult(d["text"], d["parsed"], d["input_tokens"], d["output_tokens"], d["thinking_tokens"],
                             d["latency_ms"], d["model_version"])
        except Exception as e:  # noqa: BLE001
            last = e
            if isinstance(e, PromptBlocked):
                raise
            if "client has been closed" in str(e):
                _reset_client()
            retryable = (isinstance(e, ValueError) or "client has been closed" in str(e)
                         or any(s in str(e) or s in type(e).__name__ for s in RETRYABLE))
            if not retryable or attempt == retries - 1:
                break
            time.sleep(min(2 ** attempt, 30) + random.random())
    raise RuntimeError(f"{model} 호출 실패: {last}") from last
