import json

from jarvis.local_model_ab import ResumableABStore


def test_store_persists_immediately_and_resumes_completed_cases(tmp_path):
    store = ResumableABStore(tmp_path / "run", {"schema": "a31c"})
    store.append({"case_id": "a", "status": "completed", "reasoning_content": "secret"})
    store.append({"case_id": "b", "status": "interrupted"})

    assert store.completed_case_ids() == {"a"}
    assert [case["case_id"] for case in store.pending(
        [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
    )] == ["b", "c"]
    raw = (tmp_path / "run" / "results.jsonl").read_text(encoding="utf-8")
    assert "reasoning_content" not in raw
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"] == {"completed": 1, "interrupted": 1}
