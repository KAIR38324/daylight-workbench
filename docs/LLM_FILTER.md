# Conversational preference decisions / DeepSeek

The fifth workflow page, **05 偏好决策 / DeepSeek**, is an optional online language interface with an entirely local recommendation engine. Modeling, optimization, manual preference editing and visual comparison work without an API key. Model output is a *proposed declarative preference plan*, not new candidate data, physical validation, or a uniquely correct design decision.

## User workflow

1. Generate candidates or import an external result table. Open the fifth page, or use **DeepSeek 筛选** on the results page.
2. On **映射与发送**, check the original objective column, display name, explicit min/max direction, aliases and unit. Recognizable names can have suggested aliases, but the user confirms them once for this candidate source. Arbitrary CSV headers are not physical evidence. Unit fields remain unspecified rather than guessed. Editing the mapping requires a new confirmation, not another confirmation every turn.
3. Type natural preferences such as **帮我选更考虑采光的** or **我要能耗和眩光平衡的方案**. A clear semantic mapping allows a proposal without demanding a numeric threshold. A genuinely ambiguous mapping prompts a specific question. Relative follow-ups retain the session's current weights, scope and hard constraints unless changed or reset.
4. Click **预览发送内容** and inspect the complete JSON body on **映射与发送**. Set a model and masked session-only API key, then explicitly click **发送**. Typing, previewing and changing tabs make no request.
5. The response appears in the conversation and **偏好审核**. Inspect/edit every target weight, scope, normalization, recommendation count and original-unit AND constraint. The model has not selected anything yet. Click **应用审核后的偏好并联动选择** to compute locally.
6. **推荐与权衡** displays original candidate IDs, original metric values and units, original global Pareto membership and a separate local preference score. Initially up to three recommended IDs are linked to both interactive plots and the results table. Click or multi-select recommendations to compare normalized per-objective losses. The results table retains its comparison view rather than shrinking a current-view scope to its own top-k. A background-plot selection outside that view reveals the original rows. Use the existing results-page **导出已选方案** to export the selection.
7. **交互图表** reuses the real local 2D / 3D / parallel-coordinates component, showing the complete candidate background. The recommendation subset does not redefine global Pareto membership. Small-window audit and mapping panels scroll, rather than hiding weights or squeezing controls.

The local session remembers proposed/reviewed preferences across turns. New analysis sources invalidate old plans and normalization references. **重置本轮偏好** clears the current conversation/preferences and restores a visible local equal-weight starting point; it does not silently change the existing results-page selection. No conversation or API key is serialized into a project. The applied data-only plan may be included as ordinary local analysis/export provenance.

## Transparent soft defaults and trade-offs

The new preference prompt proposes, rather than silently applies:

- A first clear daylight-priority request: daylight weight 0.6, the remaining 0.4 shared among other objectives; a single objective gets 1. These numbers are editable assumptions, not a scientific calibration.
- A new energy/glare balance request: equal weights for the two confirmed named objectives, all secondary weights explicitly zero unless the user requests otherwise.
- A follow-up changes only the requested portion of the previous state. It never invents a hard numeric cutoff from “better” or “balanced”.

Every objective has a user-visible min/max direction. The engine computes dimensionless loss over a **fixed full candidate reference**, before hard constraints, Pareto/current-view scope, top-k or click selection:

- fixed_minmax: use finite full-column minimum/maximum; min direction uses (value−min)/(max−min), max direction reverses it.
- fixed_robust: use the full-column finite 5th/95th percentiles, with normalized values clipped to [0,1]. This is an explicit alternative, not a silent recalculation on the shortlist.
- The score is the weighted mean loss, with numerical weights normalized by their sum. Lower means preferred *under those assumptions*. Scores are not physical quantities and equal weights do not mean equal physical importance. Weighted sums do not prove uniqueness or enumerate every possible Pareto compromise.
- Constant targets contribute zero loss; their displayed weights are not silently reassigned. Rows missing a positive-weight metric or containing a non-finite value there are explicitly excluded. Missing zero-weight secondary objectives do not secretly filter a row. Non-finite plotted coordinates are displayed as missing, never fabricated finite values.
- Hard numeric constraints exclude non-finite values, including for “!=”. No feasible rows means an empty recommendation table with an explanation, not relaxed conditions or invented candidates.
- Ties are displayed in deterministic typed-ID JSON-text order, not presented as a unique preference. No Pareto flags or original metric values are rewritten.
- A local fingerprint detects changed candidate numeric data under the frozen reference and demands a new review. Row reordering alone does not change scores or normalization.

## Privacy and provider boundary

The preset uses `https://api.deepseek.com/chat/completions`, model `deepseek-flash`, with an editable model-name field. The implementation permits **only that official HTTPS endpoint**, no arbitrary host. It uses a single POST, JSON-object response format, temperature 0, thinking disabled and no streaming, with TLS validation, a 20-second default timeout bounded to 1–60 seconds, a 64 KiB response limit, no redirects and no automatic retries. No new SDK or dependency is required. Availability and pricing are not asserted by offline testing.

Only the current question, numeric/boolean schema and aggregate ranges, confirmed objective mapping and the previous bounded preference state are transmitted. No candidate rows, IDs, source filename/path, training rows, string data-column contents or project files are included. Local current-view IDs and candidate fingerprints are never in the request body. Users should not type row data, secrets or other private content into their own outgoing question. The exact preview excludes the Authorization header, which is the only place the key is used.

The API key is password-masked, stays in page/request-thread memory for the session, has an explicit clear button, and is cleared on page/main-window shutdown. It is not written into logs, settings, project metadata, conversation, preview or disk. API fees may apply. Cancellation makes the UI responsive and invalidates the in-flight job ID; a late result is discarded. An already transmitted request cannot be recalled, may complete and incur a charge, and remains bounded by the network timeout. Provider error bodies and arbitrary transport exception messages are not echoed.

## Strict data-only contract and compatibility

`preference_engine.py` provides `ObjectiveSpec`, `build_preference_preview`, `preference_request_body`, `validate_preference_plan`, `request_preference_plan`, `make_normalization_reference` and `rank_candidates`. `RankingResult.frame` contains only original rows/columns/values. Auxiliary rank, score and per-objective normalized losses live separately in `RankingResult.metrics`.

A preference plan has exactly version, intent, weights, constraints, scope, normalization, limit, needs_clarification, explanation and assumptions. It requires one finite bounded weight for **every** supplied objective, including zeros; at least one is positive. Limits are 1–100; there are at most 20 objectives and 12 AND constraints. No unknown columns, IDs, unknown operators, arbitrary formulas, JSON duplicate keys, NaN/Infinity, extra fields or executable expressions are accepted. Column names, aliases, query and explanatory strings are untrusted data, not instructions or code. All response text is rendered as plain text.

`llm_filter.py` retains its original `build_request_preview`, `request_body`, `validate_plan`, `apply_plan` and `request_plan` APIs unchanged for old projects/tests. Its conservative legacy numerical-filter grammar still asks for clarification on vague balancing without explicit criteria; the main window now routes users to the dedicated preference page and richer strict grammar. Lazy preference-function exports are also available from `llm_filter.py`. The old `NaturalLanguageFilterDialog` remains for compatibility but is no longer the primary workflow.

## Verification and limitations

`tests/test_preference_engine.py` checks both common Chinese phrases using mocked JSON, multi-turn state, direction/unit-aware balance, fixed reference, hard constraints, zero-span/constant/non-finite handling, empty results, ties, original-row invariance, schema/injection rejection, exact outbound body, key redaction and official-host restrictions. `tests/test_decision_page.py` exercises actual Qt controls, explicit Preview/Send/Apply separation, cancellation/late-result protection, mapping, local editing, bidirectional ID linkage and repeated current-view Apply. Existing legacy filtering and UI tests remain relevant.

`qa/preference_decision_smoke.py` generates clearly synthetic screenshots at 1440×960 and 1100×760 plus a JSON evidence summary. These are offscreen native-widget/comparison-chart checks. **No live DeepSeek call was tested** because no key was supplied. Offscreen testing does not prove a GPU-rendered interactive 3D scene was visually correct; the existing real interactive plot is reused rather than replaced by a fake 3D image.

Official API references: [DeepSeek overview](https://api-docs.deepseek.com/zh-cn/) and [Chat Completion](https://api-docs.deepseek.com/api/create-chat-completion/).
