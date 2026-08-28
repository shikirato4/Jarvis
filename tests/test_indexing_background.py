from __future__ import annotations

from threading import Event
from time import perf_counter

from jarvis.bootstrap import build_application
from jarvis.config import Settings
from jarvis.indexing_runtime import IndexRunRequest
from jarvis.indexing_runtime.models import IndexingTrigger


def _application(tmp_path, *, auto_sync: bool):
    return build_application(
        Settings(
            data_dir=tmp_path / "runtime",
            workspace_root=tmp_path,
            research_allowed_roots=(tmp_path,),
            embeddings_enabled=False,
            indexing_auto_sync_on_start=auto_sync,
            indexing_auto_index_research=True,
            ui_backend_kind="in_memory",
        )
    )


def test_startup_indexing_runs_in_background_without_blocking_start(tmp_path) -> None:
    application = _application(tmp_path, auto_sync=True)
    service = application.indexing_runtime_service
    entered = Event()
    release = Event()

    def _blocking_run(**_kwargs):
        entered.set()
        release.wait(timeout=5)
        return None

    service._pipeline.run = _blocking_run  # type: ignore[method-assign]  # noqa: SLF001

    started = perf_counter()
    service.start()
    elapsed = perf_counter() - started
    try:
        assert elapsed < 0.5
        assert entered.wait(timeout=1)
        assert service.status()["background_active"] >= 1
    finally:
        release.set()
        service.stop()


def test_background_indexing_can_pause_resume_and_cancel_queued_jobs(tmp_path) -> None:
    application = _application(tmp_path, auto_sync=False)
    service = application.indexing_runtime_service
    entered = Event()
    release = Event()

    def _blocking_run(**_kwargs):
        entered.set()
        release.wait(timeout=5)
        return None

    service._pipeline.run = _blocking_run  # type: ignore[method-assign]  # noqa: SLF001
    service.start()
    try:
        first = service.run_async(
            IndexRunRequest(
                source_ids=("workspace",),
                trigger=IndexingTrigger.EVENT,
                requested_by="first",
            )
        )
        assert entered.wait(timeout=1)
        second = service.run_async(
            IndexRunRequest(
                source_ids=("research_results",),
                trigger=IndexingTrigger.EVENT,
                requested_by="second",
            )
        )

        paused = service.pause()
        cancelled = service.cancel_pending()

        assert paused["paused"] is True
        assert cancelled["cancelled"] >= 1
        assert second.cancelled()
        assert first.running()

        resumed = service.resume()
        assert resumed["paused"] is False
        assert (
            service.status()["chat_priority_policy"]
            == "foreground_requests_bypass_background_queue"
        )
    finally:
        release.set()
        service.stop()
