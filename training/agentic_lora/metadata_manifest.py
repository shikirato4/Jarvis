from __future__ import annotations

import hashlib
import json
from pathlib import Path

REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path) -> dict[str, object]:
    metadata = root / "base_metadata"
    return {
        "model_id": "Qwen/Qwen3.5-9B",
        "revision": REVISION,
        "source": "huggingface",
        "architecture": "Qwen3_5ForConditionalGeneration (text_config qwen3_5_text)",
        "training_target": "Qwen3_5ForCausalLM text backbone, pending B3/B4 architecture smoke test",
        "parameter_count": 8_950_000_000,
        "config": {
            "model_type": "qwen3_5",
            "text_model_type": "qwen3_5_text",
            "vocab_size": 248320,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "layers": 32,
            "attention_heads": 16,
            "key_value_heads": 4,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
            "dtype": "bfloat16",
        },
        "tokenizer": {
            "class": "Qwen2Tokenizer",
            "vocab_size": 248320,
            "eos_token_id": 248044,
            "im_start_id": 248045,
            "im_end_id": 248046,
            "tool_call_id": 248058,
            "tool_response_id": 248066,
            "chat_template_file": "chat_template.jinja",
            "production_chat_template_file": "production_chat_template.jinja",
        },
        "metadata_files": {path.name: sha256(path) for path in sorted(metadata.glob("*")) if path.is_file()},
        "full_bf16_checkpoint_downloaded_locally": False,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    manifest = build_manifest(args.root)
    out = args.root / "base_metadata" / "identity_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)
