# ConversationContextBuilder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` for inline execution and `superpowers:test-driven-development` for every behavior change. Do not commit because the user explicitly prohibited commits.

**Goal:** Build one provider-neutral conversation context pipeline that preserves relevant user/assistant history, tool exchanges, attachment derivations, and retrieved memories under an observable token budget, then route desktop normal and streaming chat through it.

**Architecture:** Add a focused `jarvis.conversation_context` package containing contracts, token estimation, selection, compaction, and telemetry. Extend `ModelMessage` with provider-neutral tool-message fields and serialize them explicitly at the OpenAI-compatible boundary. `DesktopRuntimeService` supplies immutable history snapshots and keeps attachment-enriched model text in a private per-message ledger while the builder remains stateless and independently testable.

**Tech Stack:** Python 3.12, Pydantic 2, existing EventBus, pytest, httpx MockTransport, existing PySide/desktop runtime fixtures.

**Repository constraints:** Preserve all pre-existing dirty-tree work. No commit, push, reset, clean, destructive checkout, `.env` write, secret output, or mass formatting.

---

## File map

- Create `src/jarvis/conversation_context/__init__.py`: public context contracts and builder exports.
- Create `src/jarvis/conversation_context/models.py`: context fragments, build input/result, stats, and overflow error.
- Create `src/jarvis/conversation_context/tokens.py`: injectable token-estimator protocol and conservative default.
- Create `src/jarvis/conversation_context/builder.py`: ordering, critical selection, deduplication, compaction, budgeting, and telemetry.
- Modify `src/jarvis/models_runtime/base.py`: provider-neutral assistant tool-call and tool-result message fields.
- Modify `src/jarvis/models_runtime/openai_compatible.py`: exact OpenAI-compatible message serialization.
- Modify `src/jarvis/desktop_runtime/chat.py`: finalize every normal model request through the builder.
- Modify `src/jarvis/desktop_runtime/service.py`: supply history snapshots, retain private enriched attachment context, and finalize streaming requests.
- Modify `src/jarvis/config.py`: explicit context limits and safety margins with conservative defaults.
- Create `tests/test_conversation_context_builder.py`: unit and regression coverage for ordering, budgeting, compaction, privacy, attachments, tools, and memory.
- Create `tests/test_conversation_context_desktop_e2e.py`: 20-turn desktop provider-capture scenarios for normal and streaming flows.
- Modify `tests/test_openai_compatible_http_integration.py`: provider-boundary tool-call/tool-result payload coverage.

---

### Task 1: Provider-neutral conversation and tool-message contracts

**Files:**

- Modify: `src/jarvis/models_runtime/base.py`
- Modify: `src/jarvis/models_runtime/openai_compatible.py`
- Modify: `tests/test_openai_compatible_http_integration.py`

- [ ] **Step 1: Write failing message-serialization tests**

Add a MockTransport test that constructs this sequence and captures `/chat/completions`:

```python
messages = [
    ModelMessage(role="user", content="calcula 781 x 94"),
    ModelMessage(
        role="assistant",
        content="",
        tool_calls=[
            ModelToolCall(
                id="call-calculator-1",
                name="calculator.evaluate",
                arguments={"expression": "781 * 94"},
            )
        ],
    ),
    ModelMessage(
        role="tool",
        content='{"value": 73414}',
        name="calculator.evaluate",
        tool_call_id="call-calculator-1",
    ),
]
```

Assert the HTTP payload contains OpenAI-compatible assistant `tool_calls` with `type=function`, JSON-string arguments, and a tool message with matching `tool_call_id`. Assert internal `metadata` is absent from the HTTP payload.

- [ ] **Step 2: Run the directed test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_openai_compatible_http_integration.py -k tool_message -vv -p no:cacheprovider
```

Expected: construction fails because `ModelMessage` does not accept `tool_calls`, `name`, `tool_call_id`, or `metadata`.

- [ ] **Step 3: Extend the neutral message contract**

Move `ModelToolCall` above `ModelMessage` and implement:

```python
class ModelMessage(JarvisBaseModel):
    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)
```

Keep existing two-field construction compatible.

- [ ] **Step 4: Serialize messages explicitly at the provider boundary**

Add a helper in `openai_compatible.py` equivalent to:

```python
def _provider_message(message: ModelMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": (
                        call.arguments
                        if isinstance(call.arguments, str)
                        else json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return payload
```

Use it in `_request_payload` instead of `model_dump()`.

- [ ] **Step 5: Verify GREEN and regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_openai_compatible_http_integration.py tests/test_nvidia_nim_provider.py tests/test_nemotron_local_provider.py -q -p no:cacheprovider
```

Expected: all provider protocol tests pass and no secret value appears in captured output.

- [ ] **Step 6: Audit the task diff without committing**

Run:

```powershell
git diff --check -- src/jarvis/models_runtime/base.py src/jarvis/models_runtime/openai_compatible.py tests/test_openai_compatible_http_integration.py
```

Expected: exit 0. Do not stage or commit.

---

### Task 2: Context contracts and conservative token estimation

**Files:**

- Create: `src/jarvis/conversation_context/__init__.py`
- Create: `src/jarvis/conversation_context/models.py`
- Create: `src/jarvis/conversation_context/tokens.py`
- Create: `tests/test_conversation_context_builder.py`

- [ ] **Step 1: Write failing contract and estimator tests**

Cover deterministic estimates, Unicode, empty messages, tool arguments, and validation of impossible budgets:

```python
def test_conservative_estimator_counts_text_and_tool_payload() -> None:
    estimator = ConservativeTokenEstimator()
    plain = ModelMessage(role="user", content="hola mundo")
    tool = ModelMessage(
        role="assistant",
        tool_calls=[ModelToolCall(id="c1", name="calculator.evaluate", arguments={"expression": "27*83"})],
    )
    assert estimator.estimate_message(plain) > 0
    assert estimator.estimate_message(tool) > estimator.estimate_message(ModelMessage(role="assistant"))


def test_context_budget_rejects_output_reservation_that_consumes_window() -> None:
    with pytest.raises(ContextBudgetError, match="reserve"):
        ConversationContextRequest(
            system_message=ModelMessage(role="system", content="policy"),
            current_message=ModelMessage(role="user", content="hola"),
            max_context_tokens=128,
            reserve_output_tokens=128,
            safety_margin_tokens=16,
        )
```

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py -k "estimator or budget" -vv -p no:cacheprovider
```

Expected: import failure for the missing package.

- [ ] **Step 3: Implement focused contracts**

Define:

```python
class ContextSource(StrEnum):
    HISTORY = "history"
    TOOL = "tool"
    ATTACHMENT = "attachment"
    MEMORY = "memory"
    CURRENT = "current"


class ContextFragment(JarvisBaseModel):
    fragment_id: str
    source: ContextSource
    content: str
    score: float = 0.0
    critical: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class ConversationContextRequest(JarvisBaseModel):
    system_message: ModelMessage
    history: list[ModelMessage] = Field(default_factory=list)
    tool_observations: list[ModelMessage] = Field(default_factory=list)
    attachment_context: list[ContextFragment] = Field(default_factory=list)
    memory_candidates: list[ContextFragment] = Field(default_factory=list)
    current_message: ModelMessage
    max_context_tokens: int
    reserve_output_tokens: int
    safety_margin_tokens: int = 256
    correlation_id: str | None = None


class ConversationContextStats(JarvisBaseModel):
    history_message_count: int
    tool_observation_count: int
    attachment_context_count: int
    memory_candidate_count: int
    memory_selected_count: int
    estimated_context_tokens: int
    context_compacted: bool = False
    context_truncated: bool = False


class BuiltConversationContext(JarvisBaseModel):
    messages: list[ModelMessage]
    stats: ConversationContextStats
```

Use a model validator to raise `ContextBudgetError` when the usable input budget is non-positive.

- [ ] **Step 4: Implement the estimator protocol and default**

```python
class TokenEstimator(Protocol):
    def estimate_text(self, text: str) -> int: ...
    def estimate_message(self, message: ModelMessage) -> int: ...


class ConservativeTokenEstimator:
    def estimate_text(self, text: str) -> int:
        return max((len(text.encode("utf-8")) + 3) // 4, 1)
```

Include fixed message overhead plus serialized tool-call arguments.

- [ ] **Step 5: Verify GREEN**

Run the directed tests, then:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py -q -p no:cacheprovider
```

Expected: contract and estimator tests pass.

---

### Task 3: Ordering, deduplication, telemetry, and privacy

**Files:**

- Create: `src/jarvis/conversation_context/builder.py`
- Modify: `src/jarvis/conversation_context/__init__.py`
- Modify: `tests/test_conversation_context_builder.py`

- [ ] **Step 1: Write failing ordering and telemetry tests**

Construct history with user/assistant turns, a paired assistant tool call and tool result, attachment fragments, duplicate memory fragments, and a current message. Assert:

```python
assert [message.role for message in built.messages] == [
    "system", "user", "assistant", "assistant", "tool", "system", "system", "user"
]
assert built.stats.history_message_count == 2
assert built.stats.tool_observation_count == 1
assert built.stats.attachment_context_count == 1
assert built.stats.memory_candidate_count == 2
assert built.stats.memory_selected_count == 1
```

Capture EventBus events and assert `conversation_context_built` includes only counts, booleans, correlation ID, and timing. Assert sentinel text and fragment contents are absent from every event payload and log record.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py -k "ordering or telemetry or privacy" -vv -p no:cacheprovider
```

Expected: missing `ConversationContextBuilder`.

- [ ] **Step 3: Implement the builder skeleton**

Use constructor injection:

```python
class ConversationContextBuilder:
    def __init__(
        self,
        *,
        estimator: TokenEstimator | None = None,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._estimator = estimator or ConservativeTokenEstimator()
        self._event_sink = event_sink
        self._clock = clock
```

Order system, history, paired tools, attachment data wrappers, selected memory data wrappers, and current message. Treat attachment and memory wrappers as untrusted data with explicit instructions not to follow embedded commands.

- [ ] **Step 4: Implement stable deduplication**

Fingerprint normalized source, role, tool-call ID, and content with SHA-256. Preserve the first history/tool item but keep the highest-scoring duplicate memory candidate. Never deduplicate the current message against an older turn.

- [ ] **Step 5: Emit privacy-safe telemetry**

Publish `conversation_context_built` with the required count fields and `build_latency_ms`. Publish separate `context_compacted` and `context_truncated` events only when their booleans are true. Never include content, raw metadata, paths, or tool arguments.

- [ ] **Step 6: Verify GREEN and module regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py tests/test_retention_policies.py tests/test_logging_fix.py -q -p no:cacheprovider
```

Expected: all pass with no captured private content.

---

### Task 4: Protected history and explicit compaction

**Files:**

- Modify: `src/jarvis/conversation_context/builder.py`
- Modify: `src/jarvis/conversation_context/models.py`
- Modify: `tests/test_conversation_context_builder.py`

- [ ] **Step 1: Write the 20-turn failing regression**

Create one critical opening instruction, 20 user/assistant filler pairs, a tool pair, an attachment fragment, a memory candidate, and a current recall question under a budget that requires compaction:

```python
opening = ModelMessage(
    role="user",
    content="Durante esta conversación recuerda que la palabra es COBALTO-721.",
    metadata={"context_critical": True},
)
built = builder.build(
    ConversationContextRequest(
        system_message=ModelMessage(role="system", content="Jarvis policy"),
        history=[opening, *twenty_turns],
        tool_observations=tool_exchange,
        attachment_context=[attachment],
        memory_candidates=[memory],
        current_message=ModelMessage(role="user", content="¿Cuál era la palabra?"),
        max_context_tokens=900,
        reserve_output_tokens=160,
        safety_margin_tokens=64,
    )
)
assert any("COBALTO-721" in message.content for message in built.messages)
assert built.stats.context_compacted is True
assert built.stats.context_truncated is False
```

Add cases for an unresolved confirmation, broken tool pair, oversized protected content, and a low budget that requires explicit truncation.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py -k "twenty or compaction or protected or tool_pair" -vv -p no:cacheprovider
```

Expected: builder exceeds the budget or drops the opening sentinel.

- [ ] **Step 3: Implement critical detection and coherent selection**

Critical if explicitly marked, current, unresolved confirmation, attachment referenced by the current turn, or part of a required tool pair. Also recognize explicit conversation-memory phrases (`recuerda`, `remember`, `no olvides`) as a conservative fallback.

Select recent history in complete user/assistant turn groups. Select assistant tool call and matching tool result atomically by call ID.

- [ ] **Step 4: Implement safe compaction**

Represent omitted non-critical groups in one deterministic system message containing bounded, escaped excerpts and counts. The compacted message is untrusted conversation data, not new policy. Do not summarize or alter critical content. If an excerpt itself is shortened, set `context_truncated=True` and emit the event.

Raise `ContextOverflowError` when system + current + all protected items cannot fit; include counts and token estimates but no content in the exception details.

- [ ] **Step 5: Verify RED-GREEN and budget invariants**

Run the directed tests, then the complete builder test file. Assert for every successful build:

```python
assert built.stats.estimated_context_tokens <= request.usable_input_tokens
```

- [ ] **Step 6: Property-style edge coverage**

Parametrize empty strings, Unicode, emojis, long unbroken URLs, duplicated chunks, malformed tool arguments, empty tool results, and 500 generated turns. Assert deterministic output and no silent budget overflow.

---

### Task 5: Apply context to requests, attachments, and memory candidates

**Files:**

- Modify: `src/jarvis/conversation_context/builder.py`
- Modify: `src/jarvis/desktop_runtime/chat.py`
- Modify: `src/jarvis/config.py`
- Modify: `tests/test_conversation_context_builder.py`
- Modify: `tests/test_attachment_model_request.py`

- [ ] **Step 1: Write failing request-finalization tests**

Test `build_model_request(request, history=..., memory_candidates=...)` using a request already enriched by `enrich_model_request`. Assert attachment text remains present, history precedes current, native media bytes remain only in `request.media`, and context metadata contains counts but no content.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py tests/test_attachment_model_request.py -k "finalize or history or attachment" -vv -p no:cacheprovider
```

Expected: request finalization API is absent.

- [ ] **Step 3: Add explicit settings**

Add conservative fields without provider hardcodes:

```python
conversation_context_max_tokens: int = 16_384
conversation_context_safety_margin_tokens: int = 512
conversation_context_recent_turns: int = 8
conversation_context_memory_top_k: int = 5
conversation_context_attachment_char_budget: int = 12_000
```

Provider capability limits may lower `conversation_context_max_tokens`; output budget always comes from the current generation profile.

- [ ] **Step 4: Implement request finalization**

Add a builder method that takes the existing request's system/current messages after attachment enrichment, builds context, then returns `request.model_copy(update={"messages": built.messages, "metadata": safe_stats})`. Keep `prompt` as the current task prompt for compatibility; providers use `messages` as authoritative.

- [ ] **Step 5: Add memory-fragment adaptation**

Implement a pure adapter from `RetrievedContext.chunks` to `ContextFragment` using chunk ID, score, and content. Do not log source paths or content. Tests use temporary semantic-memory fixtures and verify candidate/selected counts and relevance order.

- [ ] **Step 6: Verify module and attachment regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py tests/test_attachment_model_request.py tests/test_multimodal_pipeline.py tests/test_memory_semantic.py -q -p no:cacheprovider
```

Expected: all executable tests pass; missing optional dependencies are reported separately, not converted into pass.

---

### Task 6: Desktop normal-chat integration and private context ledger

**Files:**

- Modify: `src/jarvis/desktop_runtime/chat.py`
- Modify: `src/jarvis/desktop_runtime/service.py`
- Create: `tests/test_conversation_context_desktop_e2e.py`
- Modify: `tests/test_desktop_runtime_routing.py`

- [ ] **Step 1: Write the failing 20-turn desktop E2E test**

Build the real desktop runtime with a temporary data directory, embeddings disabled, voice replaced by no-op, and `infer_model` replaced by a provider capture that returns `Entendido.`. Send the sentinel, 20 filler messages, and the recall question. Assert the last captured request contains user and assistant history plus `COBALTO-721`, while temporary persistent memory has zero sentinel records.

- [ ] **Step 2: Write a failing attachment follow-up test**

Add a temporary text attachment on the first request, capture the enriched first model request, then send a follow-up without reattaching. Assert the safe derived attachment context survives through the private model-context ledger, but attachment bytes and derived text do not appear in logs or public activity events.

- [ ] **Step 3: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_desktop_e2e.py -vv -p no:cacheprovider
```

Expected: the last request contains only system + current and the follow-up loses attachment context.

- [ ] **Step 4: Inject one builder into DesktopChatEngine**

Construct it in `DesktopRuntimeService.__init__` with `jarvis_app.event_bus.publish`. Add `history: list[ModelMessage]` and `memory_candidates` parameters to `handle`, `_build_model_request`, and request finalization. Preserve deterministic confirmation and safety paths.

- [ ] **Step 5: Add a private per-message model-context ledger**

Maintain:

```python
self._model_context_by_message_id: dict[str, ModelMessage] = {}
```

Before appending the current UI message, take an immutable history snapshot that maps prior user messages to enriched private content and prior assistant messages to their actual text. Exclude the desktop boot banner from model history.

After attachment transformation and context finalization, store only the enriched current `ModelMessage` under its UI message ID. Never put this private override into `DesktopShellState`, logs, EventBus, or persisted memory.

- [ ] **Step 6: Retrieve memory candidates separately**

Use the semantic service only when enabled. Convert results through the pure adapter, catch declared retrieval/degradation errors, and pass candidate counts to the builder. The COBALTO test keeps embeddings disabled and asserts recall still works.

- [ ] **Step 7: Verify GREEN and desktop module regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_desktop_e2e.py tests/test_desktop_runtime_routing.py tests/test_chat_engine.py tests/test_desktop_app.py -q -p no:cacheprovider
```

Expected: all pass and deterministic non-model routes retain prior behavior until Phase 4 deliberately converts them to tools.

---

### Task 7: Streaming context, cancellation, and concurrency

**Files:**

- Modify: `src/jarvis/desktop_runtime/service.py`
- Modify: `src/jarvis/desktop_runtime/chat.py`
- Modify: `tests/test_conversation_context_desktop_e2e.py`
- Modify: `tests/test_backpressure.py`

- [ ] **Step 1: Write failing streaming tests**

Capture requests sent through `runtime_service.stream_model`. Assert the streaming request contains the same history as non-streaming. Add cancellation during stream, duplicate correlation ID, two rapid requests, and attachment retry cases. Assert cancelled partial assistant text is retained with status `cancelled` but is not treated as a completed authoritative answer.

- [ ] **Step 2: Run and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_desktop_e2e.py -k streaming -vv -p no:cacheprovider
```

Expected: prepared streaming requests contain only the current message.

- [ ] **Step 3: Finalize streaming requests through the builder**

Capture the private history snapshot before appending the streaming placeholder. Apply attachment enrichment first, builder finalization second, and provider streaming last. Store the enriched user message only after successful finalization.

- [ ] **Step 4: Keep ledger updates atomic**

Use the existing request lock for request construction and the state lock for ledger mutation. Deduplicate by correlation ID and message ID. Do not hold the state lock during provider I/O or semantic retrieval.

- [ ] **Step 5: Verify GREEN and concurrency regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_desktop_e2e.py tests/test_backpressure.py tests/test_desktop_app.py -q -p no:cacheprovider
```

Expected: no duplicate turns, no missing attachment context, and clean cancellation.

---

### Task 8: Phase-1 verification gate

**Files:**

- Verify all Phase-1 files; no commit

- [ ] **Step 1: Run directed acceptance scenarios**

Run the builder and desktop E2E tests with `-vv`. Record collected/passed/failed/skipped counts and prove the COBALTO, tool-pair, attachment, memory-disabled, compaction, and streaming assertions.

- [ ] **Step 2: Run affected module suites**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_context_builder.py tests/test_conversation_context_desktop_e2e.py tests/test_chat_engine.py tests/test_desktop_runtime_routing.py tests/test_desktop_app.py tests/test_attachment_model_request.py tests/test_multimodal_pipeline.py tests/test_memory_semantic.py tests/test_openai_compatible_http_integration.py tests/test_nvidia_nim_provider.py tests/test_nemotron_local_provider.py -q -p no:cacheprovider
```

- [ ] **Step 3: Run static validation for touched files**

```powershell
.\.venv\Scripts\python.exe -m ruff check src/jarvis/conversation_context src/jarvis/models_runtime/base.py src/jarvis/models_runtime/openai_compatible.py src/jarvis/desktop_runtime/chat.py src/jarvis/desktop_runtime/service.py tests/test_conversation_context_builder.py tests/test_conversation_context_desktop_e2e.py
.\.venv\Scripts\python.exe -m compileall -q src/jarvis/conversation_context src/jarvis/models_runtime src/jarvis/desktop_runtime
git diff --check
```

- [ ] **Step 4: Run the complete regression suite with durable output**

Use a PowerShell transcript or pytest JUnit XML inside ignored `runtime/validation/` so the result cannot be lost when the PTY closes:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml runtime/validation/phase1-pytest.xml
```

If `pypdf` or another declared dependency is absent, first run `pip check`, verify the active interpreter, and classify it as `BLOCKED_BY_DEPENDENCY` until installed. Do not alter tests to hide it.

- [ ] **Step 5: Reproduce and repair any new failure before Phase 2**

For each failure: preserve its full traceback, identify the boundary and root cause, add or refine one regression test, watch it fail, implement one fix, rerun the directed test, affected module, integration group, and complete suite.

- [ ] **Step 6: Audit Git preservation**

```powershell
git status --short
git diff --check
git diff --stat
```

Compare against the captured initial state and list only Phase-1 files changed by this work. Confirm no staged changes, commits, pushes, resets, cleans, `.env` writes, secret output, or pre-existing reversions.

---

## Phase-1 completion criteria

Phase 1 is complete only when:

- COBALTO-721 survives at least 20 turns with persistent memory disabled;
- the same protected fact survives a tool-call/result pair, attachment context, and actual compaction;
- normal and streaming desktop requests use the builder;
- assistant messages are present in later requests;
- tool-call/result IDs remain paired and serialize correctly;
- attachment-derived text persists privately for follow-up without persisting bytes;
- memory candidate and selected counts are observable without content leakage;
- every truncation or overflow is explicit;
- directed, module, integration, and full-regression evidence is fresh;
- NVIDIA API/AUTO/LOCAL provider tests remain green or are honestly classified;
- no prohibited Git or secret action occurred.
