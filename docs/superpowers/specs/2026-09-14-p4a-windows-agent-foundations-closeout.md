# P4A Windows Agent Foundations — Architecture and P4A.1 Closeout

## Scope and result

P4A establishes bounded, semantic Windows inspection and reversible window control. P4A.1 repairs environment-tool selection and window-state observation, then closes the controlled live-smoke matrix. Both phases are PASS. Arbitrary keyboard, mouse, screen/vision, browser, process termination, registry, service, installation, and privilege operations remain outside this phase.

## Repository audit and architecture

The repository already contained `ui_automation` Win32 primitives, the system application catalog, the Capability Registry, `AgentController`, `SecurityGate`, `ActionExecutor`, and `RuntimeActivityStore`. P4A reuses those systems rather than introducing a parallel agent loop or security path.

`windows_runtime` is the canonical provider-neutral boundary. `NativeWindowsControlBackend` uses `ctypes` Win32 calls plus the existing optional `psutil` dependency; `InMemoryWindowsControlBackend` supplies deterministic tests. `WindowsRuntimeService` is injected by `bootstrap`, `WindowsModule` registers schemas/actions, and the legacy `WindowsDesktopAutomationBackend` delegates its active/list/focus operations to the canonical backend when injected. Every side effect still passes through `SecurityGate` and `ActionExecutor`, and produces normal runtime activity events.

## Capabilities

| Name | Purpose | Input | Observation / verification | Risk | Idempotent | Reversible |
|---|---|---|---|---|---|---|
| `windows.environment.inspect` | Current Windows edition/version/build/architecture/session | Empty | Structured OS facts, read from the host | SAFE | Yes | N/A |
| `windows.process.list` | Bounded process/application discovery | limit, interactive-only | Structured process references | SAFE | Yes | N/A |
| `windows.window.list` | Bounded semantic application-window inventory | limit | Stable `WindowRef` entries and state | SAFE | Yes | N/A |
| `windows.window.foreground` | Current foreground application | Empty | Current `WindowRef` | SAFE | Yes | N/A |
| `windows.window.focus` | Focus one safe semantic target | `WindowTarget`, timeout | Foreground HWND re-read | SAFE | Yes | Yes |
| `windows.window.minimize` | Minimize one resolved target | `WindowTarget`, timeout | `minimized=true` re-read | SAFE | Yes | Yes |
| `windows.window.maximize` | Maximize one resolved target | `WindowTarget`, timeout | `maximized=true` re-read | SAFE | Yes | Yes |
| `windows.window.restore` | Return minimized or maximized target to normal size | `WindowTarget`, timeout | both state flags re-read as false | SAFE | Yes | N/A |
| `windows.application.launch` | Launch from a fixed safe catalog | semantic application name, timeout | one stable new target for at least one second | SENSITIVE / confirmation | No | Cleanup is application-specific |

The launch catalog is deliberately fixed; it does not accept arbitrary paths or shell strings.

## Window identity, classification, and errors

`WindowRef.runtime_id` combines PID and HWND. Follow-up resolution validates both fields and reclassifies the HWND before use. A stale exact identity fails; semantic rebinding is permitted only when additional selectors identify exactly one current safe candidate. Multiple legitimate matches return `ambiguous_target` rather than guessing.

Classification keeps existence/targetability, human-list eligibility, visibility, foreground, minimized, and maximized as distinct concepts. A legitimate minimized application remains listable and targetable. Invalid handles, empty-title windows, non-minimized hidden windows, helper classes/processes, tool windows, cloaked windows, and owned helpers remain suppressed. `window_diagnostics()` exposes raw local facts and suppression reasons for debugging, but is not registered as a model capability and is not injected into normal context.

Normalized errors include target-not-found, ambiguous, stale/closed, permission, unsupported operation, launch/focus failure, Win32 error, and timeout. Normal model observations receive safe structured error data rather than Win32 traces.

## P4A.1 root-cause evidence

### Environment selection

The expected `windows.environment.inspect` schema was present. The first failing layer was detailed-schema ranking: matching the shared `windows` namespace gave unrelated sibling tools too much relevance, diluting the operation-level signal. In the captured failing live result Qwen emitted no tool call and produced an ungrounded inspection claim. The fix excludes the shared namespace from the semantic haystack, weights operation-name tokens, retains a small namespace hint, compares tokens rather than substrings, and prunes weak siblings once a strong operation match exists. The environment description now distinguishes inspection of this machine now from explanatory how-to requests. No prompt-specific routing was added.

The local-Qwen development matrix passed five action paraphrases with exactly `windows.environment.inspect` and two information/how-to prompts with no tool execution. CUDA, context 8192, and parallel 1 were retained.

### Window observation and Notepad diagnosis

The real Notepad failure was not caused by a suppression predicate. Immediately after launch its HWND was raw, targetable, list-eligible, and resolvable; at the later focus stage no Notepad HWND existed in raw enumeration. An isolated longevity check showed the modern Notepad launcher handing off to another process/window and session restore behavior. Therefore the original “filtered after transition” diagnosis was false for that live instance. No unrelated document content was read or modified, and the exact observed instance was closed.

The general observation contract nevertheless had to stop equating hidden with nonexistent. The pure classifier preserves minimized user windows while retaining helper protections, and the resolver uses safe target candidates rather than the compact display filter. Launch verification also requires a new stable runtime identity, making modern process handoff fail safely rather than silently binding an unrelated window.

Paint was selected for the final live lifecycle because it opens a blank, harmless catalog window and avoids Notepad session restoration. Live evidence also exposed two independent Win32 details: foreground activation required temporary thread-input attachment, and restoring a window that was maximized before minimization may first return it to maximized state. Focus now uses the Windows thread-input protocol and still verifies foreground; restore explicitly normalizes both minimized and maximized flags within one timeout budget.

## Controlled live evidence

The final safe target was `win:22364:0x807e6` (`mspaint.exe`, class `MSPaintApp`). No Paint instance or old harness existed before the run.

| Stage | HWND stable | Inventory | Targetable | Visible | Minimized | Maximized | Verified |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal/launched | Yes | Yes | Yes | Yes | No | No | Yes |
| focus | Yes | Yes | Yes | Yes | No | No | Yes |
| minimize | Yes | Yes | Yes | Yes | Yes | No | Yes |
| restore | Yes | Yes | Yes | Yes | No | No | Yes |
| maximize | Yes | Yes | Yes | Yes | No | Yes | Yes |
| restore | Yes | Yes | Yes | Yes | No | No | Yes |

The real Qwen → AgentController → SecurityGate → ActionExecutor → native-backend matrix passed environment inspection, enumeration, foreground inspection, focus, minimize, restore, maximize, second restore, launch confirmation/execution, and exact-window cleanup. Each expected control action executed once. Runtime activity receipts were recorded. Final cleanup reported true, with zero remaining Paint processes and zero smoke harnesses.

Exact live state changes were: launch one blank Paint window, focus it, minimize it, restore it, maximize it, restore it, and close that same PID/HWND. No unrelated application was manipulated.

## Verification

- Windows deterministic suite: 16 passed.
- Core Windows/UI/agent/prompt-policy/activity regressions: 78 passed.
- Calculator, Files, Deep Research, OutputCompletion, provider, memory, and desktop-runtime regressions: 121 passed.
- P3.3R/web/research plus file/calculator regressions: 80 passed.
- Total focused pytest results in the final verification: 279 passed, one unrelated deprecation warning.
- Local-Qwen environment dev matrix: 7/7 passed.
- Ruff on all P4A/P4A.1 Python changes: passed.
- `git diff --check`: passed (existing line-ending notices only).
- GGUF SHA-256 before/after: `7E918AECA06C52BCB528EA6B04B4EC957E75EE8C0A73138854C0DFCF371EA429`.
- Production runtime: CUDA, context 8192, parallel 1; unchanged.
- No training or cloud work was performed. Brave, frozen B3, and B4A/B4A.2 were untouched.

## Known limitations and next boundary

Foreground focus remains subject to Windows desktop/session policy and always fails closed if foreground verification does not succeed. Semantic rebinding never resolves ambiguity. The normal inventory intentionally omits hidden/internal HWNDs. Modern packaged applications can recreate or hand off their window identity; catalog launch requires a stable unique new candidate, while subsequent stale references require unambiguous semantic selectors.

P4B may build advanced semantic window/application control on this boundary. It should not bypass the classifier, resolver, SecurityGate, ActionExecutor, verification, or activity receipts.
