from __future__ import annotations

from collections import Counter

from .models import IndexJob, IndexJobStatus, IndexSource, IndexSourceState, IndexStatus


class IndexingMonitor:
    def build_status(self, *, enabled: bool, sources: list[IndexSource], jobs: list[IndexJob], stats: dict[str, int], degradation_policy: str, last_reconcile_at=None) -> IndexStatus:
        counters = Counter()
        counters.update({"sources": len(sources), "jobs": len(jobs)})
        active = next((job for job in jobs if job.status == IndexJobStatus.RUNNING), None)
        return IndexStatus(
            enabled=enabled,
            active_job_id=active.job_id if active else None,
            total_sources=len(sources),
            enabled_sources=sum(1 for source in sources if source.enabled),
            active_snapshots=stats.get("snapshots", 0),
            pending_jobs=sum(1 for job in jobs if job.status in {IndexJobStatus.PENDING, IndexJobStatus.RUNNING}),
            failed_jobs=sum(1 for job in jobs if job.status == IndexJobStatus.FAILED),
            out_of_sync_sources=sum(1 for source in sources if source.state in {IndexSourceState.DEGRADED, IndexSourceState.ERROR}),
            total_documents=stats.get("documents", 0),
            total_chunks=stats.get("chunks", 0),
            last_reconcile_at=last_reconcile_at,
            sources=[
                {
                    "source_id": source.source_id,
                    "kind": source.source_kind.value,
                    "state": source.state.value,
                    "last_snapshot_id": source.last_snapshot_id,
                    "last_successful_index_at": source.last_successful_index_at.isoformat() if source.last_successful_index_at else None,
                }
                for source in sources
            ],
            jobs=[
                {
                    "job_id": job.job_id,
                    "type": job.job_type.value,
                    "status": job.status.value,
                    "progress_total": job.progress_total,
                    "progress_completed": job.progress_completed,
                    "progress_failed": job.progress_failed,
                }
                for job in jobs
            ],
            counters=dict(counters),
            degradation_policy=degradation_policy,
        )
