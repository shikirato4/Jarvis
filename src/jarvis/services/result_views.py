from __future__ import annotations

from typing import Any

from jarvis.identity import sanitize_assistant_identity


def summarize_model_response(payload: dict[str, Any]) -> str:
    content = str(payload.get("content") or "").strip()
    if content:
        return sanitize_assistant_identity(content)
    return "El modelo local no devolvio contenido."


def summarize_research_task(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "unknown")
    report = payload.get("report") or {}
    if report.get("short_summary"):
        return str(report["short_summary"])
    if report.get("detailed_summary"):
        return str(report["detailed_summary"])
    findings = payload.get("findings") or []
    if findings:
        first = findings[0]
        summary = first.get("summary") or first.get("claim") or "hallazgo disponible"
        return f"Research {status}: {summary}"
    last_error = payload.get("last_error")
    if last_error:
        return f"Research {status}: {last_error}"
    task_id = payload.get("task_id")
    if task_id:
        return f"Research {status}. Task `{task_id}`."
    return f"Research {status}."


def summarize_research_report(payload: dict[str, Any]) -> str:
    if payload.get("short_summary"):
        return str(payload["short_summary"])
    if payload.get("detailed_summary"):
        return str(payload["detailed_summary"])
    key_points = payload.get("key_points") or []
    if key_points:
        return "\n".join(f"- {point}" for point in key_points[:5])
    return "No hay un resumen de research disponible."


def summarize_science_result(payload: dict[str, Any]) -> str:
    operation = payload.get("operation") or "science"
    result = payload.get("result") or {}
    explanation = str(payload.get("explanation") or "").strip()
    if "derivative" in result:
        return f"Derivada lista: {result['derivative']}."
    if "integral" in result:
        return f"Integral lista: {result['integral']}."
    if "required_average_force_N" in result:
        return f"Escape estimado: v={result.get('escape_velocity_m_s'):.3f} m/s, F={result['required_average_force_N']:.3f} N."
    if "combined_proper_time_seconds" in result:
        return f"Dilatacion temporal estimada: tiempo propio {result['combined_proper_time_seconds']:.6f} s."
    if "proper_time_seconds" in result:
        return f"Dilatacion temporal estimada: tiempo propio {result['proper_time_seconds']:.6f} s."
    if "range_m" in result:
        return f"Simulacion {operation}: alcance {result['range_m']:.3f} m, altura maxima {result['max_height_m']:.3f} m."
    if "impact_velocity_m_s" in result:
        return f"Caida libre simulada: impacto a {result['impact_velocity_m_s']:.3f} m/s."
    if "max_radius_m" in result:
        return f"Orbita simulada: radio entre {result['min_radius_m']:.3f} m y {result['max_radius_m']:.3f} m."
    if explanation:
        return explanation
    return f"Resultado cientifico listo para {operation}."


def summarize_security_result(payload: dict[str, Any]) -> str:
    category = payload.get("category") or "security"
    findings = payload.get("findings") or []
    metadata = payload.get("metadata") or {}
    if category == "password":
        tier = metadata.get("tier", "unknown")
        entropy = metadata.get("entropy_bits")
        if entropy is not None:
            return f"Password {tier}: entropia estimada {entropy} bits."
        return f"Password {tier}."
    if findings:
        highest = findings[0]
        title = highest.get("title") or "hallazgo"
        return f"Analisis de seguridad: {len(findings)} hallazgo(s). Primero: {title}."
    explanation = payload.get("explanation")
    if explanation:
        return str(explanation)
    return f"Resultado de {category} listo."


def summarize_writing_receipt(payload: dict[str, Any], *, active_title: str | None = None) -> str:
    ui_receipt = (payload.get("verification_summary") or {}).get("ui") or {}
    integration = (ui_receipt.get("data") or {}).get("integration")
    if ui_receipt.get("confirmation_required"):
        return "Esta aplicacion requiere permiso. Deseas continuar?"
    if integration == "word_com" and payload.get("written_text"):
        return "He continuado el texto en Word."
    target = active_title or payload.get("window_title") or "ventana activa"
    if payload.get("written_text"):
        return f"He continuado tu texto en {target}."
    generated = str(payload.get("generated_text") or "").strip()
    if generated:
        preview = generated.replace("\n", " ")
        return f"Prepare una continuacion para {target}: {preview[:220]}"
    message = payload.get("message")
    if message:
        return str(message)
    return f"No hubo salida visible de escritura para {target}."


def summarize_system_search(payload: dict[str, Any]) -> str:
    matches = payload.get("matches") or []
    if not matches:
        return "No encontre coincidencias del sistema para esa busqueda."
    first = matches[0]
    label = first.get("display_name") or first.get("path") or first.get("identifier") or "resultado"
    return f"Encontre {len(matches)} coincidencia(s) del sistema. La primera es {label}."


def summarize_system_operation(payload: dict[str, Any]) -> str:
    target = payload.get("resolved_target") or {}
    label = target.get("display_name") or target.get("path") or target.get("identifier") or "destino solicitado"
    status = payload.get("status") or "processed"
    if payload.get("confirmation_required"):
        return "Esta aplicacion requiere permiso. Deseas continuar?"
    if status in {"opened", "launched"}:
        return f"He abierto {label}."
    if status == "confirmation_required":
        return f"Necesito confirmacion para abrir {label}."
    return f"Operacion del sistema {status}: {label}."


def summarize_ops_status(payload: dict[str, Any]) -> str:
    aggregate = payload.get("aggregate_status", "unknown")
    degraded = ", ".join(payload.get("degraded_dependencies", [])) or "none"
    return f"Estado operativo: {aggregate}. Dependencias degradadas: {degraded}."


def summarize_autonomy_view(payload: dict[str, Any]) -> str:
    if not payload:
        return "No hay misiones activas."
    goal = payload.get("goal") or payload.get("mission_id") or "mision"
    status = payload.get("status") or "unknown"
    step = payload.get("active_step_id") or payload.get("pending_approval_step_id")
    if step:
        return f"Mision {status}: {goal}. Paso actual: {step}."
    return f"Mision {status}: {goal}."


def summarize_error(prompt: str, error: str) -> str:
    lowered = prompt.casefold()
    detail = error.strip()
    lowered_detail = detail.casefold()
    visual_terms = ("pantalla", "screen", "escritorio", "desktop", "ventana", "window", "ocr", "captura", "visible")
    visual_verbs = ("ves", "ver", "lee", "leer", "describe", "analiza", "que hay", "que ves", "dice")
    looks_visual = any(term in lowered for term in visual_terms) and any(term in lowered for term in visual_verbs)
    if detail in {
        "Word no tiene un documento activo.",
        "La integracion COM de Word no estuvo disponible.",
        "No pude usar la integracion nativa de Word; probando ruta alternativa.",
    }:
        return detail
    if "circuit breaker is open" in detail:
        return "El proveedor local necesario esta temporalmente bloqueado por fallos recientes. Usa reset breaker o espera la ventana de recuperacion."
    if "timed out" in lowered_detail or "readtimeout" in lowered_detail:
        return "El modelo principal no respondio a tiempo. Intenta de nuevo en unos segundos."
    if lowered_detail.startswith("general conversation requires provider") and "request failed" in lowered_detail:
        return "No pude obtener una respuesta del modelo principal."
    if "no model candidates available" in detail or "no registered providers available" in detail:
        return "No hay un modelo local disponible para responder esa solicitud."
    if "not resolved" in detail or "not_found" in detail:
        if lowered.startswith("abre ") or lowered.startswith("open "):
            return "No pude encontrar esa aplicacion."
        return "No pude resolver ese destino en el sistema. Prueba con una ruta completa o un nombre mas especifico."
    if "window not found" in detail or "was not found" in detail:
        if looks_visual:
            return "No encontre contexto visual suficiente en la ventana solicitada."
        return "No pude encontrar esa ventana o aplicacion."
    if "could not be focused" in detail:
        return "No pude enfocar la ventana objetivo."
    if looks_visual and (
        "screen capture" in lowered_detail
        or "capture" in lowered_detail
        or "ocr" in lowered_detail
        or "vision" in lowered_detail
        or "screenshot" in lowered_detail
    ):
        return "No pude obtener una captura visual en este momento."
    if looks_visual:
        return f"No pude completar la observacion visual: {detail}"
    if "insufficient writing context" in detail:
        return "No hay contexto suficiente para continuar el texto con seguridad. Abre el documento correcto o agrega mas texto antes de pedir continuacion."
    if detail in {
        "La lectura del contexto excedio el tiempo permitido.",
        "La generacion tardo demasiado.",
        "La escritura en Word excedio el tiempo permitido.",
    }:
        return detail
    if "not allowed" in detail:
        return "Esa accion no esta permitida en el modo actual."
    if "contin" in lowered or "escribe" in lowered:
        return f"No pude completar la operacion de escritura: {detail}"
    if "abre" in lowered or "open" in lowered or "sistema" in lowered or "system" in lowered:
        return f"No pude completar la operacion del sistema: {detail}"
    if "investiga" in lowered or "research" in lowered:
        if "watchdog timeout" in lowered_detail or "operation deadline exceeded" in lowered_detail or "timed out" in lowered_detail:
            return (
                "No pude terminar la investigacion a tiempo. Puede que la busqueda o la redaccion local hayan tardado demasiado. "
                "Puedo intentarlo otra vez con menos fuentes o darte solo las fuentes encontradas."
            )
        return "No pude completar la investigacion en este momento. Intentalo de nuevo con una consulta mas especifica."
    if "science" in lowered or "calcula" in lowered or "simula" in lowered or "deriv" in lowered or "integr" in lowered:
        return f"No pude completar la operacion cientifica: {detail}"
    if "seguridad" in lowered or "vulnerabilidad" in lowered or "contrase" in lowered:
        return f"No pude completar la operacion de seguridad: {detail}"
    return f"No pude completar la solicitud: {detail}"
