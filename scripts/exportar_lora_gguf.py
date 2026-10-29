"""Exporta el adapter QLoRA a LoRA-GGUF para usarlo en llama-server.

Convierte el adapter PEFT (models/llm/qlora-jarvis-adapter) al formato
GGUF de LoRA con el conversor oficial de llama.cpp, sin fusionar los
pesos (el adapter pesa ~60MB en vez de 18GB).

  .venv-qlora/Scripts/python.exe scripts/exportar_lora_gguf.py

Luego se arranca con:
  llama-server -m models/llm/qwen3.5-9b/Qwen3.5-9B-IQ4_XS.gguf \\
      --lora models/llm/qlora-jarvis-adapter-gguf/jarvis-lora.gguf
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADAPTER = REPO / "models" / "llm" / "qlora-jarvis-adapter"
SALIDA = REPO / "models" / "llm" / "qlora-jarvis-adapter-gguf"
CONVERSOR = REPO / "scripts" / "llamacpp_tools" / "convert_lora_to_gguf.py"


def _descargar_conversor(destino: Path) -> bool:
    """Baja convert_lora_to_gguf.py del repo oficial de llama.cpp (MIT)."""
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/"
        "convert_lora_to_gguf.py"
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            destino.write_bytes(resp.read())
        print(f"[conversor] descargado en {destino}")
        return True
    except OSError as exc:
        print(f"[error] no pude descargar el conversor: {exc}", file=sys.stderr)
        return False


def main() -> int:
    if not ADAPTER.exists():
        print(f"[error] falta el adapter: {ADAPTER}", file=sys.stderr)
        return 1
    if not CONVERSOR.exists() and not _descargar_conversor(CONVERSOR):
        return 1

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "jarvis-lora.gguf"
    cmd = [
        sys.executable,
        str(CONVERSOR),
        str(ADAPTER),
        "--outfile",
        str(destino),
        "--outtype",
        "f16",
    ]
    # el conversor acepta --base-model-id (id HF), no una ruta local;
    # la metadata no es imprescindible para cargar el LoRA.
    print("[export] " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    print(res.stdout[-2000:])
    if res.returncode != 0:
        print(res.stderr[-3000:], file=sys.stderr)
        return res.returncode

    if destino.exists():
        mb = destino.stat().st_size / 1024 / 1024
        print(f"[listo] {destino} ({mb:.1f} MB)")
        print("EXPORT LORA GGUF COMPLETO")
    else:
        print("[error] no se genero el archivo", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
