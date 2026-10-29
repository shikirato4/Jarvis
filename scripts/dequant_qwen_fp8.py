"""Dequant en streaming del transformer FP8 de Unsloth (formato torchao) a BF16 plano.

Lee tensor por tensor (RAM ~300MB pico), dequantiza con la escala por fila y
escribe un safetensors estandar que diffusers carga sin exotic deps.
"""
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open

SRC = Path("models/qwen_image_turbo/transformer/diffusion_pytorch_model.safetensors")
DST = Path("models/qwen_image_turbo/transformer/diffusion_pytorch_model.rebuilt.safetensors")


def main() -> None:
    header: dict = {}
    with safe_open(SRC, framework="pt", device="cpu") as f:
        header = f.metadata if isinstance(f.metadata, dict) else {}
        keys = list(f.keys())
        # Paso 1: inventariar tensores de salida
        out_meta: dict[str, dict] = {}
        for k in keys:
            sl = f.get_slice(k)
            shape = tuple(sl.get_shape())
            dtype = sl.get_dtype()
            if k.endswith("_weight_qdata"):
                base = k[: -len("_weight_qdata")]
                # 'X._weight_qdata' -> 'X.weight' (la clave YA termina en punto)
                out_meta[base + "weight"] = {"shape": shape, "dtype": "BF16"}
            elif k.endswith("_weight_scale"):
                continue  # absorbida en el dequant
            else:
                dt = "BF16" if dtype == "BF16" else ("F32" if dtype == "F32" else None)
                if dt is None:
                    raise SystemExit(f"dtype inesperado {dtype} en {k}")
                out_meta[k] = {"shape": shape, "dtype": dt}
        n_out = len(out_meta)
        print(f"tensores de salida: {n_out}", flush=True)

        # Paso 2: construir el header JSON con offsets
        data_start_placeholder = 0
        items: dict[str, dict] = {}
        offset = 0
        for name, meta in out_meta.items():
            elem = 2 if meta["dtype"] == "BF16" else 4
            nbytes = elem
            for dim in meta["shape"]:
                nbytes *= dim
            items[name] = {"dtype": meta["dtype"], "shape": list(meta["shape"]), "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        total_data = offset
        json_header = json.dumps(items, separators=(",", ":")).encode("utf-8")
        # 8 bytes de longitud + header alineado (sin necesidad, pero seguro)
        with open(DST, "wb") as out:
            out.write(struct.pack("<Q", len(json_header)))
            out.write(json_header)
            assert out.tell() == 8 + len(json_header)

            # Paso 3: escribir tensores en orden de offset
            written = 0
            for name, meta in out_meta.items():
                if name.endswith("weight") and name[: -len("weight")] + "_weight_qdata" in keys:
                    base = name[: -len("weight")]
                    qdata = f.get_tensor(base + "_weight_qdata")
                    scale = f.get_tensor(base + "_weight_scale")
                    w = (qdata.to(torch.float32) * scale).to(torch.bfloat16)
                else:
                    t = f.get_tensor(name)
                    w = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
                if w.dtype == torch.bfloat16:
                    buf = w.view(torch.int16).numpy().tobytes()
                else:
                    buf = w.numpy().tobytes()
                expected = items[name]["data_offsets"][1] - items[name]["data_offsets"][0]
                assert len(buf) == expected, f"{name}: {len(buf)} != {expected}"
                out.write(buf)
                written += 1
                if written % 50 == 0:
                    print(f"  {written}/{n_out}", flush=True)
            assert written == n_out
            assert out.tell() == 8 + len(json_header) + total_data
    print(f"OK -> {DST} ({DST.stat().st_size / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
