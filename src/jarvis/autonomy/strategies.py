from __future__ import annotations

from .base import MissionStepKind


DEFAULT_AUTONOMY_STRATEGY = "balanced"


def classify_goal(goal: str, payload: dict[str, object]) -> tuple[str, list[MissionStepKind]]:
    lowered = goal.casefold()
    sequence = [MissionStepKind.OBSERVE, MissionStepKind.REASON]
    if payload.get("research_query") or any(token in lowered for token in ("investiga", "research", "analiza", "compare", "compara")):
        sequence.append(MissionStepKind.ACTION)
    if payload.get("writing_prompt") or any(token in lowered for token in ("continúa", "sigue donde", "escribe como", "continúa la escena", "writing")):
        sequence.append(MissionStepKind.ACTION)
    if payload.get("collection_name") or "memoria" in lowered or "memory" in lowered:
        sequence.append(MissionStepKind.RETRIEVE)
    if any(token in lowered for token in ("pantalla", "screen", "ocr", "vision", "ventana")):
        sequence.append(MissionStepKind.VISION)
    if any(token in lowered for token in ("escribe", "write", "click", "ventana", "window", "hotkey")):
        sequence.append(MissionStepKind.UI)
    if any(token in lowered for token in ("voz", "voice", "speak", "dicta")):
        sequence.append(MissionStepKind.VOICE)
    sequence.extend([MissionStepKind.VERIFY, MissionStepKind.REFLECT])
    return DEFAULT_AUTONOMY_STRATEGY, sequence
