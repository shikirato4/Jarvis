"""Entrenamiento QLoRA de Qwen3.5-9B para JARVIS.

Aprende el formato de tool-calls de JARVIS en español con el manifiesto
real de capacidades. Rank 16 conservador: no romper el base; el adapter
queda separado para rollback.

Corre con el venv aislado:
  .venv-qlora/Scripts/python.exe scripts/entrenar_qlora.py

Requisitos: dataset en data/qlora/{train,eval}.jsonl, base bf16 en
models/llm/qwen3.5-9b-base/, ~7GB VRAM libres.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "qlora"
BASE = REPO / "models" / "llm" / "qwen3.5-9b-base"
SALIDA = REPO / "models" / "llm" / "qlora-jarvis-adapter"


def cargar_jsonl(ruta: Path) -> list[dict]:
    filas = []
    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                filas.append(json.loads(linea))
    return filas


def main() -> None:
    from unsloth import FastLanguageModel  # noqa: F401 - inicializa cuda
    from unsloth.chat_templates import train_on_responses_only  # noqa: F401

    max_seq = 1024  # ejemplos miden <=918 tokens; 6.2GB VRAM libres
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE),
        max_seq_length=max_seq,
        dtype=None,
        load_in_4bit=True,
        # 4.5GB pesos + 2.4GB embeddings sin cuantizar > 6.8GB libres:
        # las capas que no caben viven en RAM y fluyen por PCIe.
        offload_layers="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # dataset -> prompts con la plantilla de chat del propio Qwen3.5
    from datasets import Dataset

    def formatear(fila: dict) -> dict:
        texto = tokenizer.apply_chat_template(
            fila["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": texto}

    train_raw = [formatear(f) for f in cargar_jsonl(DATA / "train.jsonl")]
    eval_raw = [formatear(f) for f in cargar_jsonl(DATA / "eval.jsonl")]
    train_ds = Dataset.from_list(train_raw)
    eval_ds = Dataset.from_list(eval_raw)
    print(f"[datos] train={len(train_ds)} eval={len(eval_ds)}")

    from trl import SFTTrainer, SFTConfig

    config = SFTConfig(
        output_dir=str(SALIDA),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=40,
        save_total_limit=2,
        max_seq_length=max_seq,
        dataset_text_field="text",
        packing=False,
        seed=42,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=config,
    )

    # entrenar solo sobre la respuesta del asistente (no el system largo)
    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("[modo] train_on_responses_only activado")
    except Exception as exc:  # noqa: BLE001 - plantillas varian
        print(f"[modo] sin responses_only: {exc}")

    print("[entrenamiento] iniciando…")
    trainer.train()

    model.save_pretrained(str(SALIDA))
    tokenizer.save_pretrained(str(SALIDA))
    print(f"[listo] adapter en {SALIDA}")
    print("ENTRENAMIENTO COMPLETO")


if __name__ == "__main__":
    sys.exit(main())
