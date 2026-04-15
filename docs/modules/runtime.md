# Runtime

Responsibility: assemble the full application, expose a stable runtime facade, and keep startup and shutdown deterministic.

Key files:
- `src/jarvis/bootstrap.py`
- `src/jarvis/services/runtime.py`

Failure points:
- bad configuration paths
- registry collisions
- startup order regressions
- runtime facade calls before `start()`
