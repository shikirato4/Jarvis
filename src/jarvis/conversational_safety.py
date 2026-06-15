from __future__ import annotations

from enum import StrEnum


class ConversationRisk(StrEnum):
    BENIGN_REPO = "benign_repo"
    MALWARE_OFFENSIVE = "malware_offensive"
    MALWARE_DEFENSIVE = "malware_defensive"
    UNKNOWN_RISK = "unknown_risk"


_MALWARE_TERMS = (
    "rat",
    "android rat",
    "remote access trojan",
    "troyano de acceso remoto",
    "stealer",
    "keylogger",
    "botnet",
    "payload",
    "c2",
    "command and control",
    "persistence",
    "persistencia",
    "evasion",
    "evadir",
    "bypass",
)

_OFFENSIVE_TERMS = (
    "explica el codigo",
    "explicame el codigo",
    "explica cÃ³digo",
    "explicame cÃ³digo",
    "como funciona",
    "cÃ³mo funciona",
    "como compilar",
    "cÃ³mo compilar",
    "como ejecutar",
    "cÃ³mo ejecutar",
    "como usar",
    "cÃ³mo usar",
    "como controlar",
    "cÃ³mo controlar",
    "desplegar",
    "modificar",
    "payload",
    "c2",
    "evadir",
    "bypass",
)

_DEFENSIVE_TERMS = (
    "proteger",
    "defens",
    "mitigar",
    "mitigacion",
    "mitigaciÃ³n",
    "senales de infeccion",
    "seÃ±ales de infeccion",
    "seÃ±ales de infecciÃ³n",
    "permisos sospechosos",
    "aislar",
    "analisis estatico",
    "analisis estÃ¡tico",
    "sin ejecutarlo",
    "riesgos",
    "detectar",
)


def classify_conversation_risk(text: str) -> ConversationRisk:
    lowered = (text or "").casefold()
    has_malware = any(term in lowered for term in _MALWARE_TERMS)
    if not has_malware:
        return ConversationRisk.BENIGN_REPO
    if any(term in lowered for term in _DEFENSIVE_TERMS):
        return ConversationRisk.MALWARE_DEFENSIVE
    if any(term in lowered for term in _OFFENSIVE_TERMS) or "github.com" in lowered:
        return ConversationRisk.MALWARE_OFFENSIVE
    return ConversationRisk.UNKNOWN_RISK


def malware_safe_refusal() -> str:
    return (
        "No puedo ayudarte a operar, modificar, compilar, ejecutar o explicar ese RAT de forma que facilite su uso.\n\n"
        "Si tu objetivo es defensivo, si puedo ayudarte con:\n"
        "- una revision de riesgos de alto nivel;\n"
        "- senales de infeccion en Android;\n"
        "- permisos sospechosos;\n"
        "- como aislar un dispositivo o APK sospechoso sin ejecutarlo;\n"
        "- mitigacion y buenas practicas de proteccion;\n"
        "- analisis estatico no operativo y seguro.\n\n"
        "Por seguridad no voy a clonar, ejecutar ni indexar ese repositorio como learning normal."
    )
