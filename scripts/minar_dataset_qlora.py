"""Minero de dataset QLoRA para el cerebro de JARVIS (Qwen3.5-9B).

Genera ejemplos de entrenamiento (formato mensajes/tool-calls, estilo
llama.cpp+--jinja / OpenAI) para que el modelo local aprenda el ciclo
completo de herramienta de JARVIS en español:

  system (manifiesto real de capacidades) ->
  user (frase en español) ->
  assistant (tool_call correcto con payload válido) ->
  tool (resultado simulado) ->
  assistant (respuesta natural)

Dos fuentes:
  A) catálogo REAL de acciones del registro (128 acciones): frases
     variadas por familia con payloads válidos.
  B) jarvis.db (activity_records): estadística de uso para reporte.

Uso:
  python scripts/minar_dataset_qlora.py [--out data/qlora] [--por-accion 10]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

random.seed(42)

# ---------------------------------------------------------------- instancias
SITIOS = [
    ("youtube", "https://youtube.com"),
    ("github", "https://github.com"),
    ("wikipedia", "https://wikipedia.org"),
    ("chat gpt", "https://chat.openai.com"),
    ("el correo", "https://mail.google.com"),
    ("reddit", "https://reddit.com"),
    ("stack overflow", "https://stackoverflow.com"),
    ("hugging face", "https://huggingface.co"),
]
CONSULTAS = [
    "modelos locales de voz",
    "la receta de los chilaquiles",
    "precio del dolar hoy",
    "como funciona la cuantizacion GGUF",
    "noticias de inteligencia artificial",
    "que es un LoRA en machine learning",
    "mejores laptops del ano",
    "el clima en Monterrey",
]
IMAGENES = [
    "un robot mexicano sombrero de colores",
    "un poster con el texto JARVIS OS en neón",
    "un gato astronauta estilo acuarela",
    "una infografía del café de Veracruz",
    "un paisaje cyberpunk con la luna llena",
]
ARCHIVOS = ["reporte.pdf", "notas.txt", "presupuesto.xlsx", "foto.jpg", "datos.csv"]


def _r(list_, i):
    return list_[i % len(list_)]


# ------------------------------------------------------------- frases por familia
# (plantilla_user, nombre_accion, factory_payload, respuesta_final)
def frases_familia(familia: str, defs: dict) -> list[dict]:
    out: list[dict] = []

    def add(action, user, payload, final):
        out.append({"action": action, "user": user, "payload": payload, "final": final})

    if familia == "browser":
        for i, (nombre, url) in enumerate(SITIOS):
            for plantilla in (f"abre {nombre}", f"entrar a {nombre}", f"vamos a {nombre} en el navegador"):
                add(
                    "browser.navigate",
                    plantilla,
                    {"url": url},
                    f"Listo, abrí {nombre} en el navegador integrado.",
                )
        for i, (nombre, _) in enumerate(SITIOS):
            add("browser.find", f"busca '{nombre}' en la página", {"text": nombre}, "Encontré las coincidencias y te las marqué.")
        add("browser.back", "regresa a la página anterior", {}, "Ya regresó a la página anterior.")
        add("browser.back", "ponte atrás", {}, "Hecho, volví atrás.")
        for i, consulta in enumerate(CONSULTAS):
            add("browser.find", f"en la página busca {consulta}", {"text": consulta}, f"Listo, marqué dónde aparece '{consulta}'.")
    elif familia == "web":
        for i, q in enumerate(CONSULTAS):
            for plantilla in (f"busca {q}", f"investiga {q}", f"averigua {q} en internet", f"qué sabes de {q} en la web"):
                add("web.search", plantilla, {"query": q, "max_results": 5}, f"Busqué '{q}' y te resumo lo que encontré.")
    elif familia == "voice_runtime" or familia == "voice":
        add("voice.live.start", "activa el modo live", {}, "Modo live activo: ya puedes hablarme directo por el micrófono.")
        add("voice.live.start", "quiero hablar contigo en tiempo real", {}, "Listo, modo live prendido. Háblame nomás.")
        add("voice.live.stop", "detén el modo live", {}, "Modo live apagado, de regreso al chat.")
        add("voice.live.stop", "bájale del live", {}, "Hecho, salí del modo live.")
    elif familia == "faceguard":
        add("faceguard.ban", "banea esa cara", {}, "Cara capturada y agregada a la lista prohibida.")
        add("faceguard.ban", "bloquea esa cara de la cámara", {}, "Listo, banneé esa cara con fotos frescas de la webcam.")
        add("faceguard.status", "cómo va el guardián de caras", {}, "Te doy el estado del guardián.")
        add("faceguard.remove", "desbanea esa cara", {}, "Cara removida de la lista prohibida.")
        add("faceguard.monitor.start", "prende el guardián de caras", {}, "Monitor del guardián activo.")
        add("faceguard.monitor.stop", "apaga el guardián de caras", {}, "Monitor detenido.")
    elif familia == "memory":
        add("memory.save", "recuerda que me llamo Shiki", {"text": "El usuario se llama Shiki"}, "Guardado: ya te tengo presente como Shiki.")
        add("memory.save", "apunta que mi laptop tiene 8GB de VRAM", {"text": "La laptop del usuario tiene 8GB de VRAM"}, "Hecho, apuntado en mi memoria.")
        add("memory.lookup", "qué recuerdas de mí", {"query": "datos del usuario", "limit": 5}, "Te comparto lo que tengo guardado de ti.")
        add("memory.lookup", "qué sabes de mi laptop", {"query": "laptop", "limit": 5}, "Esto es lo que recuerdo de tu laptop.")
    elif familia == "files":
        for i, nombre in enumerate(ARCHIVOS):
            add("files.search", f"busca el archivo {nombre}", {"query": nombre, "max_results": 10}, f"Encontré coincidencias para '{nombre}'.")
            add("files.read_text", f"lee el archivo {nombre}", {"path": nombre}, f"Leí {nombre} y te lo resumo.")
        add("files.search", "encuéntrame documentos recientes", {"query": "*", "max_results": 20}, "Aquí están tus documentos más recientes.")
    elif familia == "system":
        add("system.status", "cómo estás", {}, "Todo en línea por acá: te doy el estado del sistema.")
        add("system.status", "dame un diagnóstico", {}, "Corriendo el diagnóstico general…")
    elif familia == "windows":
        add("windows.workstation.lock", "bloquea la pantalla", {}, "Pantalla bloqueada. Nos vemos al desbloquear.")
        add("windows.workstation.lock", "cierra sesión ahora", {}, "Sesión bloqueada.")
        add("windows.screenshot", "toma un screenshot", {}, "Captura de pantalla tomada y guardada.")
        add("windows.clipboard.set", "copia esto al portapapeles: hola mundo", {"text": "hola mundo"}, "Copiado al portapapeles.")
    elif familia == "terminal":
        add("terminal.run", "corre el comando dir", {"command": "dir"}, "Corrí el comando, aquí va la salida.")
        add("terminal.run", "ejecuta python --version en la terminal", {"command": "python --version"}, "Ya lo ejecuté: te muestro la versión.")
    return out


def frases_genericas(definicion, indice: int) -> list[dict]:
    """Fallback: frases desde descripcion + aliases + payload mínimo válido."""
    nombre = definicion.name
    aliases = list(getattr(definicion, "semantic_aliases", None) or ())
    descripcion = (getattr(definicion, "description", "") or "").strip().rstrip(".")
    payload = payload_minimo(definicion)
    frases = []
    for i, frase in enumerate(aliases[:4]):
        frases.append({"action": nombre, "user": frase, "payload": payload, "final": f"Hecho: {descripcion.lower()}."})
    plantillas = [
        f"usa la accion {nombre}",
        f"ejecuta {nombre}",
        f"quiero que {descripcion.lower()}",
    ]
    for plantilla in plantillas[: 2]:
        frases.append({"action": nombre, "user": plantilla, "payload": payload, "final": f"Listo, {descripcion.lower()}."})
    return frases


_EJEMPLOS = {
    str: "ejemplo",
    int: 1,
    float: 1.5,
    bool: True,
}


def payload_minimo(definicion) -> dict:
    model = getattr(definicion, "payload_model", None)
    if model is None:
        return {}
    payload = {}
    for nombre_campo, campo in model.model_fields.items():
        if campo.is_required():
            anot = campo.annotation
            valor = _EJEMPLOS.get(anot, "valor")
            payload[nombre_campo] = valor
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/qlora")
    parser.add_argument("--por-accion", type=int, default=10)
    args = parser.parse_args()

    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from jarvis.config import Settings
    from jarvis.bootstrap import build_application

    print("[1/4] arrancando JARVIS para leer el registro real de acciones…")
    settings = Settings()
    app = build_application(settings)
    try:
        defs = app.action_registry.list_actions()
        print(f"    {len(defs)} acciones reales en el registro")

        # system prompt real: manifiesto de capacidades (como en produccion)
        from jarvis.agent_runtime.prompt_policy import AgenticPromptPolicy

        tools = [
            {"name": d.name, "description": getattr(d, "description", "") or ""}
            for d in defs
        ]
        system_prompt = (
            "Eres JARVIS, el asistente local de Shiki. Respondes en español informal. "
            "Cuando una peticion necesita una herramienta, llamas a la accion exacta del "
            "registro y despues respondes al usuario con el resultado.\n\n"
            + AgenticPromptPolicy.capability_context(tools)
        )

        print("[2/4] generando ejemplos por familia…")
        ejemplos = []
        por_familia = {}
        vistos = set()
        for d in defs:
            familia = d.name.split(".")[0]
            base = frases_familia(familia, None)
            propios = [f for f in base if f["action"] == d.name]
            if propios:
                candidatas = propios
            else:
                candidatas = frases_genericas(d, 0)
            tomadas = candidatas[: args.por_accion]
            for f in tomadas:
                clave = (f["user"], f["action"])
                if clave in vistos:
                    continue
                vistos.add(clave)
                ejemplos.append(f)
                por_familia[familia] = por_familia.get(familia, 0) + 1
        print(f"    {len(ejemplos)} ejemplos; familias: {por_familia}")

        print("[3/4] estadistica real de uso (jarvis.db)…")
        try:
            import sqlite3

            db = sqlite3.connect(Path(os.environ.get("LOCALAPPDATA", "")) / "Jarvis/runtime/jarvis.db")
            usadas = db.execute(
                "SELECT action_name, COUNT(*) FROM activity_records GROUP BY action_name ORDER BY 2 DESC"
            ).fetchall()
            print(f"    {len(usadas)} acciones con uso real registrado (top: {usadas[:3]})")
        except Exception as exc:  # noqa: BLE001
            print(f"    (sin estadistica: {exc})")

        print("[4/4] escribiendo JSONL…")
        random.shuffle(ejemplos)
        corte = max(1, int(len(ejemplos) * 0.9))
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for nombre_parte, lote in (("train", ejemplos[:corte]), ("eval", ejemplos[corte:])):
            path = out_dir / f"{nombre_parte}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for f in lote:
                    llamada_id = f"call_{abs(hash(f['user'])) % 10**8}"
                    ejemplo = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f["user"]},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": llamada_id,
                                        "type": "function",
                                        "function": {
                                            "name": f["action"],
                                            "arguments": json.dumps(f["payload"], ensure_ascii=False),
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": llamada_id,
                                "content": json.dumps({"status": "success", "message": "ok"}, ensure_ascii=False),
                            },
                            {"role": "assistant", "content": f["final"]},
                        ]
                    }
                    fh.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")
            print(f"    {path}: {len(lote)} ejemplos")
        print("MINERIA COMPLETA")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
