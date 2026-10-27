"""Minero de dataset para el QLoRA de JARVIS.

Construye ejemplos de entrenamiento (JSONL formato chat) para que el
Qwen3.5-9B local aprenda NATIVAMENTE el formato de tool-calls de JARVIS
en español, en vez de depender de parches de prompt.

Dos fuentes:
  1. REAL: activity_records de jarvis.db (acciones ejecutadas con su
     payload y resultado — la verdad de cómo usas JARVIS).
  2. SINTETICA: el registro de acciones (nombres, descripciones, aliases,
     schemas) → pares instruccion-espanol → tool-call correcto. Ensenia
     el FORMATO con la verdad de tierra del propio registro.

Uso:
  python scripts/minero_dataset_qlora.py -o dataset_qlora.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DB_DEFAULT = Path("C:/Users/shiki/AppData/Local/Jarvis/runtime/jarvis.db")

PLANTILLAS_INSTRUCCION = [
    "{desc}",
    "Jarvis, {desc_min}",
    "Por favor {desc_min}",
    "Necesito que {desc_min}",
    "Oye, {desc_min}",
    "{alias}",
    "Jarvis, {alias}",
    "Hazlo: {alias}",
    "Quiero {desc_min}",
    "{alias}, por favor",
]


def minar_reales(db: Path) -> list[dict]:
    """Acciones reales ejecutadas -> ejemplos con el payload historico."""
    if not db.is_file():
        print(f"[real] sin db en {db}")
        return []
    con = sqlite3.connect(str(db))
    con.row_factory = sqlite3.Row
    filas = con.execute(
        "SELECT action_name, status, payload_json, result_json, created_at "
        "FROM activity_records WHERE status = 'success' ORDER BY created_at"
    ).fetchall()
    con.close()
    ejemplos = []
    for fila in filas:
        try:
            payload = json.loads(fila["payload_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if not payload:
            continue
        ejemplos.append(
            {
                "source": "real",
                "action": fila["action_name"],
                "payload": payload,
                "created_at": fila["created_at"],
            }
        )
    print(f"[real] {len(ejemplos)} acciones completadas extraidas")
    return ejemplos


def _catalego_acciones() -> list[dict]:
    """Bootea la app real (in_memory) y enumera el registro de acciones."""
    import os
    import tempfile
    from pathlib import Path as _P

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from jarvis.bootstrap import build_application
    from jarvis.config import Settings

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            _env_file=None,
            data_dir=_P(tmp) / "runtime",
            workspace_root=_P(tmp),
            image_model_kind="qwen_turbo",
            ui_backend_kind="in_memory",
            system_backend_kind="in_memory",
            voice_tts_provider_default="in_memory",
            voice_tts_provider_fallback_order=("in_memory",),
        )
        app = build_application(settings)
        try:
            acciones = []
            for d in app.action_registry.list_actions():
                desc = (d.description or d.name).strip().rstrip(".")
                aliases = list(getattr(d, "semantic_aliases", None) or [])
                acciones.append(
                    {"name": d.name, "desc": desc, "aliases": aliases}
                )
            return acciones
        finally:
            app.stop()


def _sintetizar_una(accion: dict, rng: random.Random) -> dict | None:
    desc = accion["desc"]
    alias = rng.choice(accion["aliases"]) if accion["aliases"] else None
    plantilla = rng.choice(PLANTILLAS_INSTRUCCION)
    try:
        instruccion = plantilla.format(
            desc=desc,
            desc_min=desc[0].lower() + desc[1:],
            alias=alias or desc,
        )
    except (KeyError, IndexError):
        return None
    return {
        "source": "sintetico",
        "action": accion["name"],
        "instruccion": instruccion,
        "payload": {},
    }


def minar_sinteticos(por_accion: int = 3, seed: int = 42) -> list[dict]:
    acciones = _catalego_acciones()
    print(f"[sintetico] {len(acciones)} acciones en el registro")
    rng = random.Random(seed)  # noqa: S311 - reproducible, no cripto
    ejemplos = []
    for accion in acciones:
        for _ in range(por_accion):
            ejemplo = _sintetizar_una(accion, rng)
            if ejemplo:
                ejemplos.append(ejemplo)
    print(f"[sintetico] {len(ejemplos)} ejemplos generados")
    return ejemplos


def a_formato_chat(ejemplo: dict, con_sistema: bool = True) -> dict:
    """Convierte un ejemplo minado al formato chat de QLoRA."""
    messages: list[dict] = []
    if con_sistema:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Eres JARVIS. Respondes en espanol y usas herramientas "
                    "mediante el formato de acciones disponible. Cuando una "
                    "accion aplica, respondes con el tool-call exacto."
                ),
            }
        )
    messages.append(
        {"role": "user", "content": ejemplo.get("instruccion") or ejemplo["action"]}
    )
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": ejemplo["action"],
                        "arguments": json.dumps(
                            ejemplo.get("payload") or {}, ensure_ascii=False
                        ),
                    },
                }
            ],
        }
    )
    return {"messages": messages, "source": ejemplo.get("source", "?")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--salida", default="dataset_qlora.jsonl")
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--por-accion", type=int, default=3)
    args = parser.parse_args()

    reales = minar_reales(Path(args.db))
    sinteticos = minar_sinteticos(por_accion=args.por_accion)

    ejemplos = [a_formato_chat(e) for e in reales + sinteticos]
    salida = Path(args.salida)
    with salida.open("w", encoding="utf-8") as fh:
        for ejemplo in ejemplos:
            fh.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")
    print(f"[listo] {len(ejemplos)} ejemplos -> {salida}")
    reales_n = sum(1 for e in ejemplos if e["source"] == "real")
    print(f"   {reales_n} reales + {len(ejemplos) - reales_n} sinteticos")


if __name__ == "__main__":
    main()
