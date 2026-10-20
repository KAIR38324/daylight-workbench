# Optional DeepSeek natural-language filter

This feature is **opt-in and online**. The offline modeling/optimization pipeline still works without an API key or network connection. It produces a *suggested selection plan*, not new candidate data, new predictions, physical validation, or design advice.

## Provider and use

The preset uses the official DeepSeek HTTPS API at `https://api.deepseek.com/chat/completions` with model `deepseek-flash`. The model field can be edited as DeepSeek model names change; this version intentionally does **not** allow a custom endpoint, to avoid sending an API key to an unintended host. No DeepSeek SDK or LangChain dependency is needed. The call is a single JSON chat-completions POST, with `response_format: {"type":"json_object"}`, thinking disabled, and no streaming. The user supplies their own API key in the password-masked dialog. The key is held in memory for the dialog session only and is never saved to a project, config, log, or request preview. The dialog warns of possible provider charges. Pricing and model availability must be checked against DeepSeek's current documentation and the user's account.

1. Type a request, set a model if desired, and click **生成发送预览**.
2. Check the exact JSON body and provider host shown before clicking **发送给 DeepSeek**. No network action happens while typing or previewing.
3. Inspect the structured conditions/sort/limit returned. Click **应用筛选** to execute it locally, or close without applying. Ambiguous wording may ask for clarification instead.

Only the user query, available numeric/boolean column names and types, and numeric column min/max aggregates are sent. Candidate rows, IDs, source dataset, file paths, and project contents are not sent by the code. The user's *own query* is transmitted as entered, so users should not put private row data or secrets in it. The preview shows the complete outbound JSON body, excluding the Authorization header, whose API key remains masked.

## Safety boundary

`llm_filter.py` validates every response against a strict data-only grammar. At most 12 AND-connected predicates can use `<`, `<=`, `>`, `>=`, `==`, `!=`, or numeric `between`; booleans permit only equality/inequality. There are at most 5 sort keys and an optional limit from 1 to 1000. Unknown columns, IDs, unknown operators, duplicate JSON keys, NaN/Infinity, extra fields, code and executable expressions are rejected. `pareto` can be used only when present in the supplied candidate DataFrame. The plan's short explanation is display text only. Vague requests such as "balanced" or "better" without explicit filter or ranking criteria require clarification instead of inferred weights or thresholds. When clear criteria accompany such wording, the explicit conditions can be applied after user review; the model prompt still asks for clarification of any unresolved trade-off. An accepted plan is applied with pandas comparisons on a copy of the existing frame; it can only select and reorder existing original rows.

The HTTPS request has TLS certificate validation, a 20-second timeout, a 64 KiB maximum response, no automatic retries, and no redirects. Provider error bodies are never displayed. A canceled/closed dialog ignores a late response. The key is not serialized by any module function. The provider necessarily receives it as an Authorization header when the user explicitly clicks Send.

## Development and evidence

Pure functions are `build_request_preview(frame, query)`, `request_body(preview, settings)`, `validate_plan(raw_plan, frame_or_preview, query=None)`, and `apply_plan(frame, plan)`. `request_plan(settings, key, preview, transport=None)` permits an injected mocked transport for offline tests. Qt integration is `NaturalLanguageFilterDialog(frame, parent=None)`; after `exec()` returns Accepted, read `dialog.result_frame` and `dialog.plan`, or connect `planApplied(object)`.

`tests/test_llm_filter.py` exercises the contract using a mock transport, including rejected malformed/duplicate JSON, unknown columns/operators, code fields, non-finite numbers, local data invariance, outbound privacy, endpoint restrictions, redirect rejection, and error redaction. **It does not test a live DeepSeek API request**; that requires the user's key and an explicit Send action in the app. The UI only claims a local filter was applied after the user presses Apply.

Official references: [DeepSeek API overview](https://api-docs.deepseek.com/zh-cn/), [Create Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion/).
