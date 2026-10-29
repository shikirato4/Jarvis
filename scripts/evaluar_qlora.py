"""Evalua el adapter QLoRA de JARVIS con prompts fuera del dataset.

Compara base sola vs base+adapter para ver si el entrenamiento enseno
el formato de tool-calls en espanol con el manifiesto real.

  .venv-qlora/Scripts/python.exe scripts/evaluar_qlora.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "models" / "llm" / "qwen3.5-9b-base"
ADAPTER = REPO / "models" / "llm" / "qlora-jarvis-adapter"

# Prompts NUEVOS (no estan en el dataset): deben disparar una tool-call.
CASOS = [
    (
        "abre youtube y busca tutoriales de blender",
        "browser.open / browser.find",
    ),
    (
        "que hora es",
        "tool de hora o respuesta directa",
    ),
    (
        "hazme una imagen de un gato astronauta",
        "generacion de imagen (qwen)",
    ),
]


def main() -> None:
    from unsloth import FastLanguageModel

    system = ""
    for linea in (REPO / "data" / "qlora" / "train.jsonl").open(encoding="utf-8"):
        if linea.strip():
            import json

            system = json.loads(linea)["messages"][0]["content"]
            break
    print(f"[system] {system[:120]}…\n")

    print("=" * 70)
    print("BASE SIN ADAPTER")
    print("=" * 70)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        offload_layers="auto",
    )
    FastLanguageModel.for_inference(model)
    _generar(model, tokenizer, system, etiqueta="base")

    del model
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 70)
    print("BASE + ADAPTER QLORA")
    print("=" * 70)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER),
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
        offload_layers="auto",
    )
    FastLanguageModel.for_inference(model)
    _generar(model, tokenizer, system, etiqueta="adapter")
    print("EVALUACION COMPLETA")


def _generar(model, tokenizer, system: str, etiqueta: str) -> None:
    import json as _json

    for prompt, esperado in CASOS:
        mensajes = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        texto = tokenizer.apply_chat_template(
            mensajes, tokenize=False, add_generation_prompt=True
        )
        entradas = tokenizer(texto, return_tensors="pt").to(model.device)
        salida = model.generate(
            **entradas,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        respuesta = tokenizer.decode(
            salida[0][entradas["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"\n[{etiqueta}] PROMPT: {prompt}")
        print(f"  esperado: {esperado}")
        print(f"  salida  : {respuesta.strip()[:400]}")
        # heuristica: la respuesta trae una tool-call en formato JARVIS?
        trae_tool = any(
            marca in respuesta for marca in ("tool_call", "<tool", '"name"', "browser.")
        )
        print(f"  tool-call: {'SI' if trae_tool else 'NO'}")


if __name__ == "__main__":
    sys.exit(main())
