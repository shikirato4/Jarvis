# Actions

Responsabilidad: encapsular capacidades como acciones validadas y reversibles.

Puntos críticos:
- Cada acción define contrato de entrada con Pydantic.
- `ActionRouter` registra actividad persistente para trazabilidad.
- `execute_plan()` aplica rollback en orden inverso cuando un flujo falla.
