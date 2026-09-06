from __future__ import annotations

from dataclasses import dataclass

from jarvis.runtime_activity import RuntimeActivityEvent, RuntimeActivitySnapshot


@dataclass(frozen=True, slots=True)
class ThinkingProjection:
    status: str
    lines: tuple[str, ...]
    safe_summary: str | None


def project_runtime_progress(snapshot: RuntimeActivitySnapshot) -> ThinkingProjection:
    """Present the canonical activity stream without arguments, output, or CoT."""

    lines: list[str] = []
    summary: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    status = "active"
    model_requests = 0

    for event in snapshot.events:
        key = (event.event_name, event.call_id or "", event.action or "")
        if key in seen:
            continue
        seen.add(key)
        line, summary_line = _present_event(event, model_requests=model_requests)
        if event.event_name == "agent_controller.model_requested":
            model_requests += 1
        if line:
            lines.append(line)
        if summary_line:
            summary.append(summary_line)
        if event.event_name in {
            "agent_controller.completed",
        }:
            status = "completed"
        elif event.event_name in {
            "agent_controller.failed",
            "agent_controller.interrupted",
            "agent_controller.output_truncated",
            "agent_controller.timed_out",
            "agent_controller.step_limit",
        }:
            status = "failed"
        elif event.event_name in {
            "agent_controller.blocked",
            "action_executor.blocked",
        }:
            status = "blocked"
        elif event.event_name in {
            "agent_controller.cancelled",
        }:
            status = "cancelled"
        elif event.event_name in {
            "agent_controller.waiting_confirmation",
            "action_executor.waiting_confirmation",
        }:
            status = "waiting"

    safe_summary = "\n".join(f"• {item}" for item in summary[-6:]) or None
    return ThinkingProjection(
        status=status,
        lines=tuple(lines[-8:]),
        safe_summary=safe_summary,
    )


def _present_event(
    event: RuntimeActivityEvent,
    *,
    model_requests: int,
) -> tuple[str | None, str | None]:
    name = event.event_name
    tool = _tool_label(event.action)
    if name == "agent_controller.started":
        return "Analizando la solicitud…", "Identifiqué que la solicitud requería una ejecución observable."
    if name == "agent_controller.model_requested" and model_requests == 0:
        return "Seleccionando la siguiente capacidad…", None
    if name == "agent_controller.tool_requested":
        return f"Seleccionando {tool}…", f"Seleccioné {tool} a partir del registro de capacidades disponible."
    if name == "action_executor.started":
        return f"Ejecutando {tool}…", None
    if name == "action_executor.succeeded":
        return f"{tool} finalizó correctamente.", f"{tool} devolvió una observación satisfactoria."
    if name == "action_executor.failed":
        return f"{tool} falló.", f"{tool} devolvió un fallo real."
    if name == "action_executor.blocked":
        return "SecurityGate bloqueó la acción.", "SecurityGate bloqueó la acción solicitada."
    if name == "action_executor.waiting_confirmation":
        return "La acción requiere confirmación.", "La acción quedó pendiente de confirmación explícita."
    if name == "agent_controller.tool_observed":
        return "Observación recibida.", None
    if name == "agent_controller.continuation_started":
        return "Preparando continuación con la observación…", "Usé la observación para preparar la continuación."
    if name == "model.auto_continuation_started":
        return "Continuando respuesta…", "Completé la respuesta mediante una continuación textual acotada."
    if name == "model.auto_continuation_blocked":
        return "No queda contexto suficiente para continuar.", "La continuación se detuvo por el límite total o de contexto."
    if name == "model.auto_continuation_failed":
        return "La continuación no pudo completarse.", "La continuación textual terminó con un fallo estructurado."
    if name == "agent_controller.tool_selection_retry":
        return "Reintentando la selección de capacidad…", "El primer intento no seleccionó la capacidad requerida."
    if name == "agent_controller.output_truncated":
        return "La respuesta alcanzó el límite de salida.", "La respuesta alcanzó el límite de salida y quedó marcada como truncada."
    if name == "agent_controller.interrupted":
        return "La conexión con el proveedor se interrumpió.", "La finalización del proveedor se interrumpió."
    if name == "agent_controller.failed":
        if event.outcome_code == "MODEL_CONTINUATION_FAILURE":
            return "Falló la continuación del modelo.", "El modelo no produjo una respuesta final válida después de la observación."
        return "No se pudo finalizar la respuesta.", "La ejecución terminó con un fallo estructurado."
    if name == "agent_controller.blocked":
        return "La ejecución fue bloqueada.", "La ejecución terminó bloqueada."
    if name == "agent_controller.cancelled":
        return "La ejecución fue cancelada.", "La ejecución fue cancelada."
    if name == "agent_controller.timed_out":
        return "La ejecución agotó el tiempo.", "La ejecución agotó el tiempo disponible."
    if name == "agent_controller.completed":
        return "Preparando respuesta…", "Verifiqué el estado final antes de responder."
    return None, None


def _tool_label(action: str | None) -> str:
    namespace = str(action or "herramienta").split(".", 1)[0].casefold()
    return {
        "calculator": "Cálculo",
        "files": "Archivos",
        "terminal": "Terminal",
        "operations": "Terminal",
        "web": "Búsqueda web",
    }.get(namespace, "la herramienta")
