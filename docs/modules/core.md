# Core

Responsibility: runtime contracts, mode policy, metacommand parsing, state management, event dispatch, and safety validation.

Key files:
- `src/jarvis/core/errors.py`
- `src/jarvis/core/metacommands.py`
- `src/jarvis/core/modes.py`
- `src/jarvis/core/state.py`
- `src/jarvis/core/modules.py`

Failure points:
- `EventBus` subscribers must not break the main flow.
- `ensure_within_roots()` is the minimum boundary for safe local operations.
- `ModeManager` must stay aligned with actual registered capabilities.
- `RuntimeStateManager` must remain thread-safe and bounded.
