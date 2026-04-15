# Automation

Responsabilidad: ejecutar acciones recurrentes y mantener su estado persistente.

Puntos críticos:
- Cada automatización apunta a una acción registrada; no hay jobs huérfanos.
- El scheduler actual usa intervalos recurrentes y actualiza `last_run_at` y `next_run_at`.
- Fallos de ejecución se registran en logs y no derriban el runtime.
