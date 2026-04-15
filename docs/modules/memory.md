# Memory

Responsabilidad: persistir memoria, actividad operativa y estado de automatizaciones.

Puntos críticos:
- SQLite permite un runtime local simple y portable.
- `MemoryService` separa vistas de dominio del detalle ORM.
- La actividad queda registrada aunque una acción falle, lo que preserva auditabilidad.
