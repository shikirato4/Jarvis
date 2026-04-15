# Routing

Responsibility: route top-level requests into metacommands, tools, direct actions, or orchestration.

Key files:
- `src/jarvis/routing/models.py`
- `src/jarvis/routing/task_router.py`

Failure points:
- malformed metacommands
- mode violations
- unsafe intent routing
- missing capability registrations
