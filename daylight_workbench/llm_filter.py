"""Opt-in DeepSeek planning for local, deterministic candidate filtering.

Only schema and aggregate ranges leave this module. Candidate rows never do.
The returned plan is data, never Python/SQL or an expression to evaluate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import ssl
from typing import Any, Callable
from urllib import error, request
from urllib.parse import urlsplit

import numpy as np
import pandas as pd


ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
MAX_QUERY = 2000
MAX_COLUMNS = 100
MAX_FILTERS = 12
MAX_SORTS = 5
MAX_LIMIT = 1000
MAX_RESPONSE_BYTES = 65536


class PlanError(ValueError):
    """Invalid or unsafe model plan. The message must not contain secrets."""


class RequestError(RuntimeError):
    """Safe network/API failure, without response bodies or credentials."""


@dataclass(frozen=True)
class RequestSettings:
    model: str = MODEL
    timeout_seconds: float = 20.0
    endpoint: str = ENDPOINT


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanError("Duplicate JSON key in model response.")
        result[key] = value
    return result


def _bad_constant(_: str) -> None:
    raise PlanError("Non-finite JSON number in model response.")


def _parse_json(value: str | bytes) -> Any:
    try:
        return json.loads(value, object_pairs_hook=_no_duplicate_keys,
                          parse_constant=_bad_constant)
    except (UnicodeError, ValueError) as exc:
        raise PlanError("Model response is not valid JSON.") from exc


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _column_schema(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise PlanError("A DataFrame with unique columns is required.")
    cols = []
    for name in frame.columns:
        if not isinstance(name, str) or not name or len(name) > 150:
            raise PlanError("Column names must be short, nonempty strings.")
        # IDs must be preserved, not sent or used as semantic criteria.
        if name == "candidate_id":
            continue
        series = frame[name]
        if name == "pareto" and pd.api.types.is_bool_dtype(series.dtype):
            cols.append({"name": name, "type": "boolean"})
        elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
            numbers = pd.to_numeric(series, errors="coerce")
            finite = numbers[np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan))]
            item: dict[str, Any] = {"name": name, "type": "number"}
            if len(finite):
                low, high = float(finite.min()), float(finite.max())
                if math.isfinite(low) and math.isfinite(high):
                    item.update({"min": low, "max": high})
            cols.append(item)
    if len(cols) > MAX_COLUMNS:
        raise PlanError("Too many columns for an LLM request.")
    return cols


def build_request_preview(frame: pd.DataFrame, query: str) -> dict[str, Any]:
    """Create exactly the user-data preview sent to the provider.

    No row values, IDs, dataset names, file paths or sample records are sent.
    Numeric minima/maxima are aggregated over entire columns.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
        raise PlanError(f"Query must contain 1 to {MAX_QUERY} characters.")
    return {"query": query.strip(), "columns": _column_schema(frame)}


def _validate_preview(preview: dict[str, Any]) -> dict[str, str]:
    if not isinstance(preview, dict) or set(preview) != {"query", "columns"}:
        raise PlanError("Invalid request preview.")
    query, cols = preview["query"], preview["columns"]
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY:
        raise PlanError("Invalid request query.")
    if not isinstance(cols, list) or len(cols) > MAX_COLUMNS:
        raise PlanError("Invalid request columns.")
    schema = {}
    for item in cols:
        if not isinstance(item, dict) or set(item) not in ({"name", "type"}, {"name", "type", "min", "max"}):
            raise PlanError("Invalid request column schema.")
        name, kind = item["name"], item["type"]
        if not isinstance(name, str) or not name or len(name) > 150 or name == "candidate_id" or name in schema:
            raise PlanError("Invalid request column name.")
        if kind not in ("number", "boolean"):
            raise PlanError("Invalid request column type.")
        if "min" in item and (kind != "number" or not _finite_number(item["min"])
                               or not _finite_number(item["max"]) or item["min"] > item["max"]):
            raise PlanError("Invalid aggregate range.")
        schema[name] = kind
    return schema


SYSTEM_PROMPT = """You translate a user's candidate-filtering request into one JSON object only. No Markdown, code, SQL, expressions, or candidate IDs. The candidate data stays local. Available columns, their types, and optional aggregate min/max appear in the next message. Use only supplied columns. Return exactly these keys: filters, sort, limit, needs_clarification, explanation. filters is an AND-only array, at most 12, of {column, operator, value}; numeric operators are <, <=, >, >=, ==, !=, between (two numbers). For boolean columns, only == or != with true or false. sort is an array, at most 5, of {column, direction}, where direction is asc or desc. limit is null or an integer 1..1000. Never infer a numeric threshold, objective weight, or preference not explicitly given. Words such as 'balanced', 'better', '更好', or '平衡' without precise criteria require needs_clarification true, empty filters/sort, null limit and a short clarification question in explanation. If a request cannot be represented safely, use the same clarification shape. Never claim predictions are verified physics."""


def request_body(preview: dict[str, Any], settings: RequestSettings = RequestSettings()) -> dict[str, Any]:
    """Exact JSON body sent in the HTTPS request; useful for UI inspection."""
    _validate_preview(preview)
    _validate_settings(settings)
    return {"model": settings.model, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(preview, ensure_ascii=False, allow_nan=False, separators=(",", ":"))},
    ], "temperature": 0, "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"}, "max_tokens": 2048, "stream": False}


def _validate_settings(settings: RequestSettings) -> str:
    if not isinstance(settings, RequestSettings) or not isinstance(settings.model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", settings.model):
        raise PlanError("Invalid model name.")
    if not _finite_number(settings.timeout_seconds) or not 1 <= settings.timeout_seconds <= 60:
        raise PlanError("Timeout must be 1 to 60 seconds.")
    # This version intentionally uses one vetted provider, preventing token exfiltration
    # through an arbitrary endpoint or redirection. Model remains editable.
    if settings.endpoint != ENDPOINT:
        raise PlanError("Only the official DeepSeek HTTPS endpoint is supported.")
    parts = urlsplit(settings.endpoint)
    if (parts.scheme != "https" or parts.netloc != "api.deepseek.com"
            or parts.path != "/chat/completions" or parts.query or parts.fragment
            or parts.username or parts.password):
        raise PlanError("Invalid DeepSeek endpoint.")
    return parts.hostname or ""


def provider_host(settings: RequestSettings = RequestSettings()) -> str:
    return _validate_settings(settings)


def _ambiguous(query: str) -> bool:
    """Conservative guard for vague requests lacking any explicit criterion.

    The prompt still asks the provider to clarify genuinely ambiguous trade-offs;
    this guard does not claim to understand the user's semantics.
    """
    vague = re.search(r"\b(?:balanced|better)\b|平衡|更好|较好", query, re.I)
    explicit = re.search(r">=|<=|[><=≥≤]|升序|降序|最大|最小|最高|最低|"
                         r"\b(?:ascending|descending|maximize|minimize)\b", query, re.I)
    return bool(vague and not explicit)


def validate_plan(raw_plan: dict | str | bytes, frame_or_preview: pd.DataFrame | dict,
                  query: str | None = None) -> dict[str, Any]:
    """Reject anything beyond a bounded, typed, AND-only filtering grammar."""
    if isinstance(raw_plan, (str, bytes)):
        raw_plan = _parse_json(raw_plan)
    if not isinstance(raw_plan, dict) or set(raw_plan) != {"filters", "sort", "limit", "needs_clarification", "explanation"}:
        raise PlanError("Model plan has missing or unsupported fields.")
    if isinstance(frame_or_preview, pd.DataFrame):
        schema = _validate_preview(build_request_preview(frame_or_preview, query or "filter"))
    else:
        schema = _validate_preview(frame_or_preview)
        query = query or frame_or_preview["query"]
    filters, sorts, limit = raw_plan["filters"], raw_plan["sort"], raw_plan["limit"]
    clarification, explanation = raw_plan["needs_clarification"], raw_plan["explanation"]
    if not isinstance(clarification, bool) or not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 1000:
        raise PlanError("Invalid clarification or explanation.")
    if not isinstance(filters, list) or len(filters) > MAX_FILTERS or not isinstance(sorts, list) or len(sorts) > MAX_SORTS:
        raise PlanError("Too many or invalid filters or sort keys.")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT):
        raise PlanError("Invalid result limit.")
    if clarification:
        if filters or sorts or limit is not None:
            raise PlanError("A clarification plan cannot also filter candidates.")
        return {"filters": [], "sort": [], "limit": None,
                "needs_clarification": True, "explanation": explanation.strip()}
    if query and _ambiguous(query):
        raise PlanError("Ambiguous 'balanced/better' requests require clarification.")
    clean_filters = []
    for item in filters:
        if not isinstance(item, dict) or set(item) != {"column", "operator", "value"}:
            raise PlanError("Invalid filter shape.")
        column, op, value = item["column"], item["operator"], item["value"]
        if not isinstance(column, str) or column not in schema:
            raise PlanError("Unknown filter column.")
        if schema[column] == "boolean":
            if op not in ("==", "!=") or not isinstance(value, bool):
                raise PlanError("Invalid boolean filter.")
        elif op == "between":
            if not isinstance(value, list) or len(value) != 2 or not all(_finite_number(v) for v in value) or value[0] > value[1]:
                raise PlanError("Invalid numeric range.")
        elif op not in ("<", "<=", ">", ">=", "==", "!=") or not _finite_number(value):
            raise PlanError("Invalid numeric filter.")
        clean_filters.append({"column": column, "operator": op, "value": value})
    clean_sorts = []
    seen = set()
    for item in sorts:
        if not isinstance(item, dict) or set(item) != {"column", "direction"}:
            raise PlanError("Invalid sort shape.")
        column, direction = item["column"], item["direction"]
        if not isinstance(column, str) or column not in schema or direction not in ("asc", "desc") or column in seen:
            raise PlanError("Invalid or duplicate sort key.")
        seen.add(column)
        clean_sorts.append({"column": column, "direction": direction})
    return {"filters": clean_filters, "sort": clean_sorts, "limit": limit,
            "needs_clarification": False, "explanation": explanation.strip()}


def apply_plan(frame: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    """Apply a reviewed plan locally; return only original rows and values."""
    plan = validate_plan(plan, frame)
    if plan["needs_clarification"]:
        raise PlanError("Clarification is required before applying a plan.")
    mask = pd.Series(True, index=frame.index)
    for item in plan["filters"]:
        values, op, bound = frame[item["column"]], item["operator"], item["value"]
        if op == "between":
            match = values.ge(bound[0]) & values.le(bound[1])
        else:
            match = {"<": values.lt, "<=": values.le, ">": values.gt,
                     ">=": values.ge, "==": values.eq, "!=": values.ne}[op](bound)
        mask &= match.fillna(False)
    result = frame.loc[mask].copy()
    if plan["sort"]:
        result = result.sort_values([item["column"] for item in plan["sort"]],
                                    ascending=[item["direction"] == "asc" for item in plan["sort"]],
                                    kind="stable")
    if plan["limit"] is not None:
        result = result.head(plan["limit"])
    return result.copy()


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RequestError("Provider redirected the request; nothing was sent to the new host.")


def _https_transport(req: request.Request, timeout: float, max_bytes: int) -> bytes:
    opener = request.build_opener(_NoRedirect, request.HTTPSHandler(context=ssl.create_default_context()))
    try:
        with opener.open(req, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except RequestError:
        raise
    except error.HTTPError as exc:
        # Never read/echo the provider's error body; it can include request data.
        raise RequestError(f"Provider returned HTTP {exc.code}.") from None
    except Exception:
        # Unexpected socket/TLS errors can contain request details.
        raise RequestError("Could not complete the HTTPS request.") from None
    if len(data) > max_bytes:
        raise RequestError("Provider response exceeded the size limit.")
    return data


def request_plan(settings: RequestSettings, key: str, preview: dict[str, Any],
                 transport: Callable[[request.Request, float, int], bytes] | None = None) -> dict[str, Any]:
    """One explicit POST; no retry, redirect, credential persistence, or row upload."""
    _validate_settings(settings)
    _validate_preview(preview)
    if not isinstance(key, str) or not re.fullmatch(r"[!-~]{1,512}", key.strip()):
        raise PlanError("A valid API key is required.")
    body = json.dumps(request_body(preview, settings), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    req = request.Request(settings.endpoint, data=body, method="POST",
                          headers={"Authorization": f"Bearer {key.strip()}",
                                   "Content-Type": "application/json", "Accept": "application/json"})
    try:
        if transport is None:
            response = _https_transport(req, settings.timeout_seconds, MAX_RESPONSE_BYTES)
        else:
            try:
                response = transport(req, settings.timeout_seconds, MAX_RESPONSE_BYTES)
            except Exception:
                # A custom/mock transport might include the Authorization header in
                # its exception text; never expose that text to callers.
                raise RequestError("Could not complete the HTTPS request.") from None
        if not isinstance(response, bytes) or len(response) > MAX_RESPONSE_BYTES:
            raise RequestError("Invalid or oversized provider response.")
        payload = _parse_json(response)
        content = payload["choices"][0]["message"]["content"]
        return validate_plan(content, preview)
    except (RequestError, PlanError):
        raise
    except (KeyError, IndexError, TypeError, ValueError):
        raise RequestError("Provider response has an unexpected shape.") from None


# The original numerical-filter API above is deliberately preserved for old
# projects/tests. The dedicated conversation page uses this richer strict schema.
# Lazy imports avoid a circular dependency and reuse the same vetted transport.
def build_preference_preview(frame, query, objectives, preference_state=None):
    from .preference_engine import build_preference_preview as build
    return build(frame, query, objectives, preference_state)


def preference_request_body(preview, settings=RequestSettings()):
    from .preference_engine import preference_request_body as build
    return build(preview, settings)


def validate_preference_plan(raw_plan, frame_or_preview, objectives=None):
    from .preference_engine import validate_preference_plan as validate
    return validate(raw_plan, frame_or_preview, objectives)


def request_preference_plan(settings, key, preview, transport=None):
    from .preference_engine import request_preference_plan as send
    return send(settings, key, preview, transport)
