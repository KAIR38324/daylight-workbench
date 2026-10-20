"""Offline contract tests: no test uses a real provider or real credential."""
import json

import pandas as pd
import pytest

from daylight_workbench.llm_filter import (
    MAX_RESPONSE_BYTES, PlanError, RequestError, RequestSettings, _NoRedirect,
    apply_plan, build_request_preview, request_body, request_plan, validate_plan,
)


@pytest.fixture
def frame():
    return pd.DataFrame({
        "candidate_id": [101, 102, 103],
        "predicted::UDI": [50.0, 80.0, 70.0],
        "predicted::EUI": [30.0, 25.0, 40.0],
        "pareto": [True, True, False],
    })


def plan(**changes):
    base = {"filters": [], "sort": [], "limit": None,
            "needs_clarification": False, "explanation": "Local selection."}
    base.update(changes)
    return base


def test_preview_does_not_send_rows_or_ids(frame):
    preview = build_request_preview(frame, "UDI at least 70")
    serialized = json.dumps(request_body(preview))
    assert "candidate_id" not in serialized
    assert '"101"' not in serialized and '"102"' not in serialized
    assert set(preview) == {"query", "columns"}
    assert {c["name"] for c in preview["columns"]} == {"predicted::UDI", "predicted::EUI", "pareto"}
    assert {"name": "predicted::UDI", "type": "number", "min": 50.0, "max": 80.0} in preview["columns"]


def test_local_apply_preserves_rows_values_and_ids(frame):
    original = frame.copy(deep=True)
    reviewed = validate_plan(plan(filters=[{"column": "predicted::UDI", "operator": ">=", "value": 70},
                                   {"column": "pareto", "operator": "==", "value": True}],
                                  sort=[{"column": "predicted::EUI", "direction": "asc"}], limit=1), frame)
    actual = apply_plan(frame, reviewed)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(actual, frame.loc[[1]])


@pytest.mark.parametrize("bad", [
    "{garbage", '{"filters":[],"filters":[],"sort":[],"limit":null,"needs_clarification":false,"explanation":"x"}',
    '{"filters":[{"column":"predicted::UDI","operator":">","value":NaN}],"sort":[],"limit":null,"needs_clarification":false,"explanation":"x"}',
])
def test_bad_json_is_rejected(frame, bad):
    with pytest.raises(PlanError):
        validate_plan(bad, frame)


@pytest.mark.parametrize("bad", [
    plan(filters=[{"column": "made_up", "operator": ">", "value": 2}]),
    plan(filters=[{"column": "candidate_id", "operator": "==", "value": 101}]),
    plan(filters=[{"column": "predicted::UDI", "operator": "eval", "value": "__import__('os')"}]),
    plan(filters=[{"column": "predicted::UDI", "operator": ">", "value": float("inf")}]),
    plan(filters=[{"column": "pareto", "operator": ">", "value": 0}]),
    plan(code="frame.dropna()"),
    plan(limit=1001),
    plan(limit=True),
    plan(filters=[{"column": "predicted::UDI", "operator": "between", "value": [90, 20]}]),
])
def test_invalid_plan_rejected(frame, bad):
    with pytest.raises(PlanError):
        validate_plan(bad, frame)


def test_ambiguous_words_require_clarification(frame):
    with pytest.raises(PlanError, match="clarification"):
        validate_plan(plan(sort=[{"column": "predicted::UDI", "direction": "desc"}]), frame,
                      query="Find a balanced better scheme")
    suggested = validate_plan(plan(needs_clarification=True, explanation="Which targets matter to you?"),
                              frame, query="Find a balanced scheme")
    with pytest.raises(PlanError):
        apply_plan(frame, suggested)


def test_mock_transport_only_and_key_not_in_body(frame):
    preview = build_request_preview(frame, "UDI at least 70")
    selected = plan(filters=[{"column": "predicted::UDI", "operator": ">=", "value": 70}])
    response = json.dumps({"choices": [{"message": {"content": json.dumps(selected)}}]}).encode()
    captured = []

    def fake(req, timeout, max_bytes):
        captured.append((req, timeout, max_bytes))
        return response

    result = request_plan(RequestSettings(), "only-for-test", preview, fake)
    assert result["filters"] == selected["filters"]
    req, timeout, max_bytes = captured.pop()
    assert req.full_url == "https://api.deepseek.com/chat/completions"
    assert req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer only-for-test"
    assert b"only-for-test" not in req.data
    assert b"candidate_id" not in req.data
    assert timeout == 20.0 and max_bytes == MAX_RESPONSE_BYTES


def test_secret_not_in_errors_or_settings(frame):
    preview = build_request_preview(frame, "UDI > 70")
    secret = "only-for-test"
    def fake(req, timeout, max_bytes):
        return b'{"error": "' + secret.encode() + b'"}'
    with pytest.raises(RequestError) as info:
        request_plan(RequestSettings(), secret, preview, fake)
    assert secret not in str(info.value)
    assert secret not in repr(RequestSettings())


def test_redirect_cannot_replay_authorization():
    import urllib.request
    redirect = _NoRedirect()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions",
                                 headers={"Authorization": "Bearer test"})
    with pytest.raises(RequestError, match="redirected"):
        redirect.redirect_request(req, None, 302, "Moved", {}, "https://evil.example/steal")


def test_unapproved_endpoint_rejected_before_transport(frame):
    preview = build_request_preview(frame, "UDI > 70")
    called = []
    with pytest.raises(PlanError):
        request_plan(RequestSettings(endpoint="https://evil.example/steal"), "test", preview,
                     lambda *args: called.append(True))
    assert called == []


def test_explicit_criteria_not_blocked_by_better_word(frame):
    clear = plan(filters=[{"column": "predicted::UDI", "operator": ">=", "value": 60}],
                 sort=[{"column": "predicted::EUI", "direction": "asc"}])
    valid = validate_plan(clear, frame, query="找更好的方案，UDI>=60，按能耗升序")
    assert list(apply_plan(frame, valid).candidate_id) == [102, 103]


def test_arbitrary_transport_error_is_redacted(frame):
    preview = build_request_preview(frame, "UDI > 70")
    secret = "secret-from-test"
    def bad_transport(*args):
        raise RuntimeError(f"request with bearer {secret} failed")
    with pytest.raises(RequestError) as info:
        request_plan(RequestSettings(), secret, preview, bad_transport)
    assert secret not in str(info.value)
