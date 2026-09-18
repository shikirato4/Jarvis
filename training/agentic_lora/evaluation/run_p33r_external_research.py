from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.desktop_runtime.service import DesktopRuntimeService
from jarvis.web_search import ExaSearchProvider, WebProviderRouter, YouSearchProvider

from .run_baseline import _managed_runtime


def run(output: Path) -> dict[str, Any]:
    manager, owned = _managed_runtime(Path.cwd())
    app = None
    try:
        with tempfile.TemporaryDirectory(prefix="jarvis-p33r-research-") as raw_root:
            workspace = Path(raw_root).resolve()
            settings = Settings(
                data_dir=workspace / "runtime",
                workspace_root=workspace,
                research_allowed_roots=(workspace,),
                embeddings_enabled=False,
                ui_backend_kind="in_memory",
                voice_audio_input_backend_default="in_memory",
                voice_audio_output_backend_default="in_memory",
                voice_tts_provider_default="in_memory",
                llm_mode="local",
                local_runtime_profile_id="qwen35-9b-iq4-xs",
                local_llm_base_url="http://127.0.0.1:8081/v1",
                local_runtime_cuda_base_url="http://127.0.0.1:8081/v1",
                local_runtime_context_override=None,
                local_runtime_context_size=8192,
                local_runtime_parallel=1,
                local_runtime_autostart=False,
                local_runtime_stop_on_shutdown=False,
                pre_lora_agentic_profile="full",
            )
            providers = [
                YouSearchProvider(settings=settings),
                ExaSearchProvider(settings=settings),
            ]
            configured = {
                provider.provider: bool(
                    provider.status().configured and provider.status().enabled
                )
                for provider in providers
            }
            if not any(configured.values()):
                report = {
                    "phase": "P3.3R",
                    "status": "blocked_no_configured_provider",
                    "configured": configured,
                    "brave_in_router": False,
                    "passed": False,
                }
            else:
                router = WebProviderRouter(providers)
                app = build_application(settings)
                app.web_research_orchestrator.provider = router
                app.start()
                started = perf_counter()
                response = DesktopRuntimeService(app).send_chat(
                    (
                        "/deepresearch Compara las recomendaciones oficiales "
                        "actuales para crear entornos virtuales de Python y para "
                        "declarar dependencias de un proyecto. Resume diferencias "
                        "prácticas y cita al menos dos fuentes útiles."
                    ),
                    metadata={"search_mode": "deep"},
                )
                latency = round(perf_counter() - started, 3)
                raw = response.raw_result
                research = raw.get("research") or {}
                sources = research.get("sources") or []
                citation_validation = raw.get("citation_validation") or {}
                used = dict(router.request_counts)
                synthesis = raw.get("synthesis") or {}
                report = {
                    "phase": "P3.3R",
                    "status": raw.get("status"),
                    "configured": configured,
                    "providers_used": used,
                    "brave_in_router": False,
                    "brave_requested": bool(used.get("brave")),
                    "source_count": len(sources),
                    "source_ids": [item.get("citation_id") for item in sources],
                    "source_domains": [item.get("domain") for item in sources],
                    "search_rounds": (research.get("metrics") or {}).get(
                        "search_rounds"
                    ),
                    "duplicate_queries": (research.get("metrics") or {}).get(
                        "duplicate_queries"
                    ),
                    "stop_reason": research.get("stop_reason"),
                    "citation_valid": citation_validation.get("valid"),
                    "invalid_ids": citation_validation.get("invalid_ids") or [],
                    "synthesis_status": synthesis.get("final_status"),
                    "synthesis_attempts": len(synthesis.get("attempts") or []),
                    "continuations": sum(
                        int(item.get("continuations") or 0)
                        for item in synthesis.get("attempts") or []
                    ),
                    "visible_answer_present": bool(response.message.content.strip()),
                    "source_only_fallback": raw.get("status")
                    in {"partial", "partial_validation"},
                    "latency_seconds": latency,
                    "training_started": False,
                    "cloud_model_used": False,
                }
                report["passed"] = bool(
                    raw.get("status") in {"completed", "completed_with_repairs"}
                    and len(sources) >= 2
                    and citation_validation.get("valid") is True
                    and not report["invalid_ids"]
                    and not report["brave_requested"]
                    and report["visible_answer_present"]
                    and not report["source_only_fallback"]
                )
                app.stop()
                app = None

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return report
    finally:
        if app is not None:
            app.stop()
        if owned:
            manager.stop()
        manager.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real external non-Brave Deep Research smoke"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data/p33r_external_research.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
