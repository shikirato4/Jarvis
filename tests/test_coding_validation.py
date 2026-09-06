from jarvis.coding_validation import execute_solution, extract_python_solution, inspect_safety


def test_extracts_single_python_block_without_prose():
    assert extract_python_solution("Here it is:\n```python\ndef f(x):\n return x\n```\n") == "def f(x):\n return x"


def test_rejects_unsafe_import_and_times_out_bounded_code(tmp_path):
    safe, reason = inspect_safety("import os\ndef f(): return 1")
    assert not safe and reason == "UNSAFE_GENERATED_CODE"
    result = execute_solution("def f():\n while True: pass", "assert f()", timeout_seconds=0.1)
    assert result["timeout"] is True


def test_executes_solution_and_serializes_assertion_counts():
    result = execute_solution("def f(x): return x * 2", "\n".join([
        "checks = [(f(2), 4), (f(0), 0), (f(-1), -2)]",
        "print(f'{sum(actual == expected for actual, expected in checks)}/{len(checks)}')",
    ]))
    assert result["passed"] == 3 and result["total"] == 3
