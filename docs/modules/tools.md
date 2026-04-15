# Tools

Responsibility: expose low-level callable capabilities behind typed contracts.

Key files:
- `src/jarvis/tools/models.py`
- `src/jarvis/tools/registry.py`
- `src/jarvis/tools/builtin.py`

Failure points:
- duplicate tool names
- invalid payload schemas
- tool mode restrictions
- adapter drift between tool wrappers and action contracts
