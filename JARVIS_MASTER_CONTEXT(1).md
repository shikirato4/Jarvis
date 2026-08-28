# JARVIS Master Context

Updated: 2026-08-16

This is the current architectural context for the repository. It is not a historical audit and does not replace test evidence or `git diff`. No `JARVIS_AUDIT_2026-08-15.md` existed when this context was created, so none was fabricated.

## System map

- Language/runtime: Python 3.12+, Pydantic 2, FastAPI, Typer and PySide6.
- Composition root: `src/jarvis/bootstrap.py` builds the dependency graph; `JarvisApplication.start/stop` owns lifecycle.
- Entry points: `python -m jarvis`, `python -m jarvis serve`, `python -m jarvis.desktop`, console scripts `jarvis` and `jarvis-desktop`.
- Persistence: SQLAlchemy with local SQLite by default. Memory, activity, automation, conversation and policy state remain local and provider-neutral.
- Model layer: `ModelService`, `ProviderRegistry`, neutral `ModelCatalog`/`ModelRouter`, and `ProviderCoordinator`.
- Primary providers: `NvidiaNimProvider` and `NemotronOmniLocalProvider`, both normalized through OpenAI-compatible contracts.
- Retired model integrations: removed from active source, UI, CLI, settings and tests; the hybrid modes have no hidden third-provider fallback.
- Local model target: `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`; no automatic checkpoint or quantization substitution.
- Tools/actions: always execute locally through registries, safety policy, confirmation and receipts. A model tool call is a proposal, never authority to bypass controls.
- Multimodal: request-scoped attachments plus OCR, STT, video extraction and document parsing. Raw storage paths/Base64 never enter normal model/history serialization.
- Interfaces: desktop UI, CLI and local HTTP API share the runtime facade.
- Optional local systems preserved: voice/TTS/STT, vision/OCR/capture, image generation, Windows automation, Word COM, Unity, research, writing, science, autonomy, Code Agent and repo learning.

## Hybrid model invariants

`selectedMode`, `requestProvider`, `activeProvider`, `providerStatus`, `capabilities` and `fallbackReason` are separate contracts.

- AUTO: authenticated NVIDIA health is checked, not generic internet access. NVIDIA is preferred; local NVFP4 is used for missing/invalid key, rate limit, timeout/network or `5xx` when the request can be handled safely. Recovery applies to the next request.
- NVIDIA API: strict remote provider, explicit failure, no silent local fallback.
- LOCAL: strict configured local endpoint, no call to NVIDIA.
- The circuit opens after three consecutive cloud failures; `429` respects `Retry-After`.
- A response never switches providers mid-stream. One local retry is permitted only before output/external effects and only with compatible modality transport. Request/tool/file IDs prevent duplicate effects.
- If both providers fail, the combined diagnostic remains visible.

## Capability migration inventory

| Existing function | Responsible component | Previous model dependency | NVIDIA Nemotron | Local NVFP4 | Migration improvement | Regression evidence | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| General chat/streaming | desktop chat, `ModelService` | retired provider selection | Yes, normalized API/stream | Yes, OpenAI-compatible endpoint | Strict modes, visible provider, circuit/failover | coordinator, provider, chat and desktop tests | Preserved/improved |
| Reasoning/planning | cognition orchestrator, autonomy | model catalog plus heuristics | Reasoning advertised/normalized | Endpoint capability conservative | Same local policies; structured plan output retained | model runtime, autonomy and compatibility suite | Preserved |
| Tool calling | model contracts and local registries | provider-specific/plain JSON | Normalized tool call proposals | Supported when endpoint advertises contract | No tool execution in provider; effect IDs prevent replay | NVIDIA tool tests and stream fallback tests | Improved |
| Structured output | `ModelRequest`/providers/API | parser-specific output | JSON response format forwarded | OpenAI-compatible forwarding | API no longer drops `response_format` | provider and hybrid API tests | Improved |
| Memory/conversation | memory services and desktop runtime | provider-neutral context | Bounded context only | Local context | No complete memory upload; mode change preserves state | existing memory/chat plus desktop tests | Preserved |
| Voice/STT/TTS | `voice_runtime` | independent local providers | Derived transcript can be supplied | Entire route local | Timestamp segments retained; no voice feature removed | voice and media adapter tests | Preserved/improved |
| Vision/OCR | `vision_runtime`, image processor | independent OCR/vision | Native image transport verified | OCR/vision local path | Strict real OCR adapter; no mock fallback in production | vision, attachment and adapter tests | Preserved/improved |
| Video understanding | multimodal processor | no unified route | Local-derived until native wire contract verified | FFmpeg frames/audio/STT | Timestamped bounded representation | media adapter/pipeline tests | New/additive |
| Document reading | indexing/research plus attachment processor | local readers | Derived text only; raw documents not claimed native | Local PDF/DOCX/text parsing | Page refs, chunks and context budget | document/corruption/security tests | Preserved/improved |
| File attachments | desktop composer and `attachments/` | fragmented/manual paths | Explicit native image or derived content | local derived/native endpoint route | `+`, multi-select/drop/paste/chips/retry and protected tokens | attachment/UI/pipeline tests | New/additive |
| Web research | Brave provider plus chat synthesis | previous model drafted results | Assigned Nemotron drafts | Assigned local Nemotron drafts | Stale model claims removed; sources survive synthesis failures | chat/research/desktop routing suite | Preserved/improved |
| Code Agent/Git/repo learning | code-agent runtime | optional LLM adapter | Through model service when configured | Through local endpoint when configured | Permissions, patch review and confirmations unchanged | existing code-agent suite/full regression | Preserved |
| Automations/system/Unity | actions/tools/runtime services | provider-neutral | Model can propose only | Model can propose only | Provider boundary cannot bypass safety | existing integration suite/full regression | Preserved |
| Local image generation | image runtime/Diffusers | independent SDXL | Not replaced | remains local | Chat model migration does not reduce generation | existing desktop/image tests | Preserved |
| Secret handling | `security_runtime/secrets.py` | env/config strings | keyring first, `.env` fallback | no key needed | status-only UI/API, validity tracking, redaction | secret/provider/API tests | Improved |
| Logging lifecycle | `jarvis_logging.py` | global shutdown | independent | independent | only Jarvis-owned handlers close | logging regression tests | Fixed |

## Verified model capabilities and conservative transport

NVIDIA's model material identifies text, images, audio and video as inputs; text output; reasoning; JSON/structured responses; tool calling; word timestamps; and a context limit of 262,144 tokens. It is English-focused. The current NVIDIA OpenAI-compatible adapter asserts native binary transport only for images because that wire path is verified in this implementation. Audio/video use local extraction/transcription until their API payload contract is verified. PDF/DOCX are processed locally and sent as bounded derived text.

The local provider publishes conservative capabilities because an arbitrary configured endpoint may expose less than the checkpoint. It validates `/models` and rejects a server that does not expose the exact target.

## Local runtime and current host

The local runtime is a registry of backend adapters. `LocalRuntimeService` validates requested features against adapter capabilities before building a command; vLLM currently declares CPU offload, GPU memory utilization, max model length and tensor parallel flags. A future compatible backend can register without changing `ModelService`.

The audited host reports approximately 13.9 GiB RAM, PyTorch 2.11 CPU-only, no CUDA, no usable NVFP4 GPU detection and no vLLM. NVIDIA NVFP4 requires Blackwell compute capability 10.0 or newer according to the implementation's verified requirement, and the official checkpoint repository contains approximately 22.4 GB. `launch_supported` is therefore false. This does not block an endpoint on another compatible host; diagnostics explicitly distinguish a remote local-LLM endpoint from host compatibility.

## Privacy and secret boundary

- Put the real key only in `NVIDIA_API_KEY` in ignored `.env`, or use the masked desktop field backed by OS keyring.
- Never put a key in `.env.example`, README, logs, history, frontend storage or API responses.
- LOCAL never invokes `NvidiaNimProvider`.
- AUTO sends only necessary conversation context and explicitly attached/native or locally derived file content.
- Attachments are not promoted to permanent memory without an explicit policy.
- Path traversal, MIME spoofing, executable/active content, corruption, oversize input and likely secrets are rejected before provider processing.

## Operational commands

```powershell
python -m pip install -e ".[dev,desktop]"
python -m pip check
python -m ruff check --no-cache .
python -m pytest -p no:cacheprovider -q
python -m build --no-isolation
python -m jarvis doctor
python -m jarvis serve
python -m jarvis.desktop
```

Use `RUN_NVIDIA_LIVE_TESTS=true` only when an external live check is intentionally authorized and `NVIDIA_API_KEY` exists. Normal tests use mocks and never require or print a real key.

## Known external limitations

- This host cannot execute the target NVFP4 checkpoint; a compatible backend/hardware or external endpoint is required.
- NVIDIA live inference requires a valid external NVIDIA key/service and is intentionally skipped by default.
- Native NVIDIA audio/video transport is not claimed until the request schema is verified; local derived processing remains the safe route.
- OCR/audio/video fallbacks require installed Tesseract, FFmpeg/FFprobe and an STT backend.
- Processes launched by this audit cannot create/update some workspace files (`.pytest_cache` and the existing `egg-info`). Tests therefore use `-p no:cacheprovider`; build and smoke startup were repeated against writable temporary directories using the same source. The Windows visual-inspection plugin also fails during bootstrap because its runtime does not export a requested internal module; Qt windowed tests remain successful. These are execution-environment constraints, not hidden by changing application behavior.

## Verification snapshot (2026-08-16)

- Baseline: `632 passed, 2 skipped`.
- First closing full suite: `720 passed, 3 failed, 3 skipped` in 1171.55 seconds; the three failures were corrected at their causes.
- Directed final hybrid regression: `72 passed` in 39.80 seconds.
- Final full suite after implementation: `724 passed, 3 skipped` in 1869.88 seconds.
- Ruff lint: all checks passed. Forty-three new Python files passed the formatter check; inherited files were not mass-formatted.
- Dependency resolution: `python -m pip check` reported no broken requirements.
- Distribution: wheel and source distribution built successfully from a clean writable copy of the final source tree.
- Artifact install: the final wheel installed into an isolated target, imported successfully and retained both exact model identifiers.
- API smoke: clean startup/shutdown and HTTP 200 from `/health`, `/llm/status` and `/llm/local-diagnostic`.
- Desktop smoke: three PySide6 tests on the Windows platform showed the window and verified the three-mode selector and multimodal composer interactions.
- Live NVIDIA test: intentionally skipped without `RUN_NVIDIA_LIVE_TESTS=true` and an externally valid credential.
- Static typing: no mypy/pyright dependency or configuration exists in the repository; `compileall` and strict Pydantic contract tests passed, but no nonexistent type-check command is reported as successful.

See `README.md`, `docs/architecture.md`, `docs/modules/models_runtime.md` and `docs/multimodal_attachments.md` for user-facing configuration and component details.
