from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from jarvis.config import Settings
from jarvis.models_runtime.base import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderKind,
)
from jarvis.models_runtime.output_completion import OutputCompletionPolicy
from jarvis.web_search import (
    SearchMode,
    WebResearchOrchestrator,
    WebSearchHit,
    WebSearchResponse,
)


class DeterministicResearchProvider:
    provider = "deterministic_fixture"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> WebSearchResponse:
        self.calls.append(query)
        topic = (
            "Alpha and Beta comparison, definition, context, development, causes, "
            "effects, persistence, recurrence and treatment evidence."
        )
        hits = [
            WebSearchHit(
                title="Primary Alpha report",
                url="https://alpha.example/report",
                snippet=topic,
                content=topic + " Alpha is supported by primary measurements.",
                provider=self.provider,
                rank=1,
            ),
            WebSearchHit(
                title="Independent Beta analysis",
                url="https://beta.example/analysis",
                snippet=topic,
                content=topic + " Beta independently corroborates the comparison.",
                provider=self.provider,
                rank=2,
            ),
        ]
        return WebSearchResponse(
            status="ok",
            provider=self.provider,
            query=query,
            hits=hits[:max_results],
        )


def _response(content: str, finish_reason: str, tokens: int) -> ModelResponse:
    return ModelResponse(
        provider_name="deterministic_local",
        provider_kind=ProviderKind.LOCAL,
        logical_model="general_assistant",
        model_name="Qwen3.5-9B-IQ4_XS",
        content=content,
        latency_ms=1.0,
        usage=ModelUsage(output_tokens=tokens),
        metadata={
            "finish_reason": finish_reason,
            "stream_closed_normally": True,
            "incomplete": finish_reason == "length",
            "completion_tokens": tokens,
        },
    )


def _legacy_research_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: dict[str, Any] = {"available": True, "arms": {}}
    for arm in ("control", "compact", "full"):
        group = [
            row
            for row in rows
            if row.get("arm") == arm and row.get("category") == "research"
        ]
        if not group:
            continue
        missing = sum(
            len(row.get("actual_sequence", []))
            < len(row.get("expected_sequence", []))
            for row in group
        )
        extra = sum(
            len(row.get("actual_sequence", []))
            > len(row.get("expected_sequence", []))
            for row in group
        )
        task_success = sum(
            bool(row.get("first_tool_correct"))
            and bool(row.get("argument_valid"))
            and bool(row.get("final_nonempty"))
            for row in group
        )
        output["arms"][arm] = {
            "cases": len(group),
            "exact_sequence_rate": sum(
                bool(row.get("tool_selection")) for row in group
            )
            / len(group),
            "task_success_rate": task_success / len(group),
            "missing_required_search_cases": missing,
            "extra_search_cases": extra,
        }
    return output


def run(output: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jarvis-p33-diagnostics-") as raw:
        root = Path(raw)
        settings = Settings(
            _env_file=None,
            data_dir=root / "runtime",
            workspace_root=root,
            research_allowed_roots=(root,),
            search_deep_max_queries=4,
            search_deep_max_pages=4,
            search_max_research_rounds=2,
            local_runtime_context_size=8192,
        )
        provider = DeterministicResearchProvider()
        research = WebResearchOrchestrator(provider, settings)
        session = research.run(
            "/deepresearch compare Alpha y Beta con causas y efectos",
            mode=SearchMode.DEEP,
        )
        source_map_before = {
            citation.citation_id: citation.source_id
            for citation in session.citations
        }
        prompt_one = research.build_synthesis_prompt(
            session,
            max_context_tokens=2_600,
        )
        prompt_two = research.build_synthesis_prompt(
            session,
            max_context_tokens=2_600,
        )
        answer = (
            "Alpha cuenta con evidencia primaria para la comparación. [S1]\n\n"
            "Beta aporta corroboración independiente. [S2]"
        )
        validation = research.validate_answer(answer, session)
        rendered = research.render_citations(validation.answer, session)
        numeric_repair = research.normalize_citation_syntax(
            "Dato conocido [1]. Dato desconocido [99].",
            session,
        )

        completion = OutputCompletionPolicy(settings)
        request = completion.prepare_request(
            ModelRequest(
                task_type="assistant",
                logical_model="general_assistant",
                messages=[
                    ModelMessage(role="user", content="Escribe una respuesta completa")
                ],
                metadata={
                    "generation_profile": "detailed",
                    "adaptive_output_budget": True,
                    "auto_continue_text": True,
                    "output_context_tokens": 8192,
                },
            )
        )
        continuation_requests: list[ModelRequest] = []

        def continue_once(next_request: ModelRequest) -> ModelResponse:
            continuation_requests.append(next_request)
            return _response(" termina sin repetir el inicio.", "stop", 40)

        completed = completion.complete_response(
            request,
            _response("La respuesta comienza y", "length", 100),
            infer_next=continue_once,
        )
        source_map_after = {
            citation.citation_id: citation.source_id
            for citation in session.citations
        }
        report: dict[str, Any] = {
            "phase": "P3.3",
            "deterministic_research": {
                "task_success": bool(session.evidence and validation.valid),
                "status": session.status,
                "stop_reason": session.stop_reason,
                "search_rounds": session.metrics.search_rounds,
                "searches": session.metrics.queries_count,
                "duplicate_queries": len(session.queries)
                - len({item.casefold() for item in session.queries}),
                "sources": len(session.citations),
                "source_ids": source_map_before,
                "source_ids_stable": source_map_before == source_map_after,
                "synthesis_prompt_stable": prompt_one == prompt_two,
                "citation_valid": validation.valid,
                "parsed_citation_ids": validation.cited_ids,
                "invalid_ids": validation.invalid_ids,
                "unknown_numeric_id_not_promoted": (
                    "[S99]" not in numeric_repair and "[99]" in numeric_repair
                ),
                "rendered_verified_links": all(
                    citation.url in rendered for citation in session.citations
                ),
            },
            "completion": {
                "owner": "ModelService.OutputCompletionPolicy",
                "request_class": completed.metadata.get("output_request_class"),
                "initial_budget": request.max_tokens,
                "finish_reasons": completed.metadata.get("segment_finish_reasons"),
                "segments": completed.metadata.get("model_request_count"),
                "continuations": completed.metadata.get("auto_continuation_count"),
                "one_logical_answer": completed.content
                == "La respuesta comienza y termina sin repetir el inicio.",
                "continuation_tools_removed": bool(
                    continuation_requests
                    and not continuation_requests[0].tools
                    and continuation_requests[0].tool_choice is None
                ),
                "final_finish_reason": completed.metadata.get("finish_reason"),
            },
            "legacy_prompt_research": _legacy_research_metrics(
                Path("training/agentic_lora/data/prelora2_prompt_abc_results.jsonl")
            ),
            "root_causes": {
                "deep_research_source_only_fallback": (
                    "legacy explicit synthesis cap disabled automatic continuation; "
                    "citation/fallback handling saw an incomplete or failed synthesis"
                ),
                "simple_information_length": (
                    "agent_initial was explicitly excluded from textual continuation"
                ),
                "research_sequence_50_percent": (
                    "exact-call-count metric conflated one sufficient search, one "
                    "missing search, and one bounded extra-search loop"
                ),
            },
            "completion_matrix": [
                {"path": "short/normal/long/extended chat", "owner": "ModelService", "continuation": "bounded textual", "explicit_cap": "authoritative"},
                {"path": "agent initial text", "owner": "ModelService", "continuation": "bounded textual; schemas removed", "explicit_cap": "authoritative"},
                {"path": "structured tool call", "owner": "AgentController", "continuation": "never textual; one structured retry", "explicit_cap": "authoritative"},
                {"path": "post-tool final", "owner": "ModelService", "continuation": "bounded textual; no tools", "explicit_cap": "authoritative"},
                {"path": "research planning/search", "owner": "WebResearchOrchestrator", "continuation": "bounded rounds/gain/sufficiency", "explicit_cap": "query/page/round limits"},
                {"path": "Deep Research synthesis/repair", "owner": "ModelService then citation validator", "continuation": "before validation; max 2 plus max 3 repair attempts", "explicit_cap": "research ceiling"},
                {"path": "desktop streaming", "owner": "ModelService", "continuation": "single logical stream", "explicit_cap": "authoritative"},
                {"path": "non-streaming runtime service", "owner": "ModelService", "continuation": "single logical response", "explicit_cap": "authoritative"},
            ],
            "brave_touched": False,
            "training_started": False,
            "cloud_used": False,
            "cost_usd": 0,
        }
        report["passed"] = all(
            (
                report["deterministic_research"]["task_success"],
                report["deterministic_research"]["source_ids_stable"],
                report["deterministic_research"]["citation_valid"],
                report["deterministic_research"]["unknown_numeric_id_not_promoted"],
                report["completion"]["one_logical_answer"],
                report["completion"]["continuation_tools_removed"],
                report["completion"]["final_finish_reason"] == "stop",
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="P3.3 deterministic diagnostics")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "data/p33_diagnostics.json",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
