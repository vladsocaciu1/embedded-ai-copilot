import logging
import math
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TypeVar

import httpx
from google.genai import errors

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BASE_DELAY_S = 1.0
MAX_DELAY_S = 8.0
REQUEST_SLEEP_BUDGET_S = 20.0
DEFAULT_RETRY_AFTER_S = 60
RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})
TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.ConnectError)

sleep = time.sleep
rand = random.random
now = lambda: datetime.now(UTC)

_RETRY_DELAY_RE = re.compile(r"^\d+(\.\d+)?s$")
_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"

T = TypeVar("T")


@dataclass
class RetryBudget:
    remaining: float = REQUEST_SLEEP_BUDGET_S


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, TRANSPORT_ERRORS):
        return True
    return isinstance(error, errors.APIError) and error.code in RETRYABLE_CODES


def retry_hint(error: BaseException) -> float | None:
    delay = _retry_info_delay(error)
    if delay is not None:
        return delay
    return _retry_after_delay(error)


def retry_after_seconds(error: errors.APIError) -> int:
    hint = retry_hint(error)
    if hint is None:
        return DEFAULT_RETRY_AFTER_S
    return max(1, math.ceil(hint))


def call_with_retries(fn: Callable[[], T], budget: RetryBudget, what: str) -> T:
    attempt = 1
    while True:
        try:
            return fn()
        except (errors.APIError, *TRANSPORT_ERRORS) as error:
            if not is_retryable(error) or attempt >= MAX_ATTEMPTS:
                raise
            delay = _delay(error, attempt)
            if delay > budget.remaining:
                raise
            budget.remaining -= delay
            _log_retry(what, error, attempt, delay)
            sleep(delay)
            attempt += 1


def _delay(error: BaseException, attempt: int) -> float:
    hint = retry_hint(error)
    if hint is not None:
        return hint
    cap = min(MAX_DELAY_S, BASE_DELAY_S * 2 ** (attempt - 1))
    return cap * rand()


def _log_retry(what: str, error: BaseException, attempt: int, delay: float) -> None:
    if isinstance(error, errors.APIError):
        code: object = error.code
    else:
        code = type(error).__name__
    logger.warning(
        f"Gemini {what} failed ({code}); retry {attempt} of {MAX_ATTEMPTS - 1} "
        f"in {delay:.1f} s."
    )


def _detail_entries(error: BaseException) -> list[object] | None:
    body = getattr(error, "details", None)
    if not isinstance(body, dict):
        return None
    nested = body.get("error")
    source = nested if isinstance(nested, dict) else body
    entries = source.get("details")
    if not isinstance(entries, list):
        return None
    return entries


def _retry_info_delay(error: BaseException) -> float | None:
    entries = _detail_entries(error)
    if entries is None:
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("@type") != _RETRY_INFO_TYPE:
            continue
        delay = entry.get("retryDelay")
        if isinstance(delay, str) and _RETRY_DELAY_RE.fullmatch(delay):
            return float(delay[:-1])
    return None


def _retry_after_delay(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("retry-after")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if text.isdigit():
        return float(text)
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    try:
        seconds = (parsed - now()).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return None
    if seconds < 0:
        return 0.0
    return seconds
