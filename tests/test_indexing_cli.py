from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from jarvis.bootstrap import build_application
from jarvis.cli import app
from jarvis.config import Settings
from jarvis.memory_semantic.base import EmbeddingProviderHealth, EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class DeterministicEmbeddingProvider:
    provider_name = "indexing_cli_provider"
    provider_kind = "local"

    def health_check(self) -> EmbeddingProviderHealth:
        return EmbeddingProviderHealth(provider_name=self.provider_name, healthy=True)

    def embed(self, request: EmbeddingRequest, *, model_name: str, timeout_seconds: float | None) -> EmbeddingResponse:
        vectors = [EmbeddingVector(index=index, text=text, values=[2.0, 1.0], dimensions=2) for index, text in enumerate(request.texts)]
        return EmbeddingResponse(
            provider_name=self.provider_name,
            provider_kind=self.provider_kind,
            logical_model=request.logical_model or model_name,
            model_name=model_name,
            vectors=vectors,
            latency_ms=1.0,
        )


def test_indexing_cli_commands(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "cli.md").write_text("Jarvis CLI indexing", encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "runtime",
        workspace_root=tmp_path,
        research_allowed_roots=(tmp_path,),
        ollama_enabled=False,
        embedding_provider_default="indexing_cli_provider",
        embedding_provider_fallback_order=("indexing_cli_provider",),
        indexing_auto_sync_on_start=False,
        ui_backend_kind="in_memory",
    )

    def build() -> object:
        instance = build_application(settings)
        instance.embedding_provider_registry.register(DeterministicEmbeddingProvider())
        return instance

    runner = CliRunner()
    monkeypatch.setattr("jarvis.cli.build_application", build)
    status = runner.invoke(app, ["index", "status"])
    run = runner.invoke(app, ["index", "run", "--sources", "workspace"])
    add_source = runner.invoke(app, ["index", "add-source", "--id", "docs", "--kind", "user_documents", "--name", "Docs", "--root", str(tmp_path)])
    reindex = runner.invoke(app, ["index", "reindex", "--sources", "workspace"])
    assert status.exit_code == 0
    assert run.exit_code == 0
    assert add_source.exit_code == 0
    assert reindex.exit_code == 0
    assert '"job_id"' in run.stdout
