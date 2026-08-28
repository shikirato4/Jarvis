from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from jarvis.action_executor import ActionExecutor
from jarvis.actions.registry import ActionRegistry
from jarvis.actions.router import ActionRouter
from jarvis.agent_runtime import AgentController, AgentRunRequest
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.data_analyzer import DataAnalysisError, DataAnalyzer
from jarvis.models_runtime.base import ModelResponse, ModelToolCall, ProviderKind
from jarvis.modules.data_module import DataModule


def _xlsx_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>category</t></si><si><t>value</t></si>
              <si><t>A</t></si><si><t>B</t></si>
            </sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
                <row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>10</v></c></row>
                <row r="3"><c r="A3" t="s"><v>3</v></c><c r="B3"><v>25</v></c></row>
              </sheetData>
            </worksheet>""",
        )
    return output.getvalue()


def test_csv_analysis_filter_sort_aggregate_and_anomalies() -> None:
    analyzer = DataAnalyzer()
    table = analyzer.load_bytes(
        "ventas.csv",
        (
            "category;value;date;note\n"
            "A;10;2026-01-01;ok\n"
            "A;12;2026-01-02;\n"
            "B;11;2026-01-03;ok\n"
            "B;1000;2026-01-04;outlier\n"
        ).encode("utf-8"),
    )

    description = analyzer.describe(table.table_id)
    filtered = analyzer.filter(table.table_id, column="value", operator="gt", value=11)
    sorted_rows = analyzer.sort(table.table_id, column="value", descending=True)
    grouped = analyzer.aggregate(
        table.table_id,
        group_by="category",
        column="value",
        operation="mean",
    )
    anomalies = analyzer.anomalies(table.table_id, column="value")

    assert description["row_count"] == 4
    assert description["columns"]["value"]["type"] == "number"
    assert description["columns"]["note"]["null_count"] == 1
    assert [row["value"] for row in filtered] == [12, 1000]
    assert sorted_rows[0]["value"] == 1000
    assert grouped == {"A": 11.0, "B": 505.5}
    assert anomalies == [{"row_index": 3, "value": 1000.0}]


def test_tsv_json_and_xlsx_are_loaded_without_active_content() -> None:
    analyzer = DataAnalyzer()
    tsv = analyzer.load_bytes("datos.tsv", b"name\tcount\nuno\t1\ndos\t2\n")
    json_table = analyzer.load_bytes(
        "datos.json",
        b'[{"name":"uno","count":1},{"name":"dos","count":2}]',
    )
    xlsx = analyzer.load_bytes("datos.xlsx", _xlsx_bytes())

    assert tsv.rows == json_table.rows
    assert xlsx.rows == [
        {"category": "A", "value": 10},
        {"category": "B", "value": 25},
    ]


def test_corrupt_empty_and_macro_xlsx_are_rejected() -> None:
    analyzer = DataAnalyzer()
    with pytest.raises(DataAnalysisError, match="empty"):
        analyzer.load_bytes("empty.csv", b"")
    with pytest.raises(DataAnalysisError, match="JSON"):
        analyzer.load_bytes("bad.json", b"{not-json")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("xl/vbaProject.bin", b"active-content")
        archive.writestr("xl/worksheets/sheet1.xml", b"<worksheet/>")
    with pytest.raises(DataAnalysisError, match="active content"):
        analyzer.load_bytes("macro.xlsx", output.getvalue())


def test_duplicate_headers_and_mixed_columns_are_explicit() -> None:
    analyzer = DataAnalyzer()
    table = analyzer.load_bytes(
        "mixed.csv",
        b"value,value,blank\n1,text,\n2,3,\n",
    )
    description = analyzer.describe(table.table_id)

    assert table.columns == ["value", "value_2", "blank"]
    assert description["columns"]["value_2"]["type"] == "mixed"
    assert description["columns"]["blank"]["null_count"] == 2


def test_data_agent_can_describe_then_find_anomaly_before_final_answer() -> None:
    analyzer = DataAnalyzer()
    table = analyzer.load_bytes(
        "agent.csv",
        b"value\n10\n11\n12\n1000\n",
    )
    registry = ActionRegistry()
    DataModule(analyzer).register_actions(registry)
    events = EventBus()
    memory = type("Memory", (), {"record_activity": lambda self, **payload: None})()
    executor = ActionExecutor(
        ActionRouter(registry, Settings(_env_file=None), memory, object(), events)
    )
    requests = []

    def response(content="", tool_calls=None):
        return ModelResponse(
            provider_name="nemotron-test",
            provider_kind=ProviderKind.LOCAL,
            logical_model="general_assistant",
            model_name="nemotron-test",
            content=content,
            tool_calls=tool_calls or [],
            latency_ms=1.0,
        )

    replies = iter(
        [
            response(
                tool_calls=[
                    ModelToolCall(
                        id="data-describe",
                        name="data.describe",
                        arguments={"table_id": table.table_id},
                    )
                ]
            ),
            response(
                tool_calls=[
                    ModelToolCall(
                        id="data-anomaly",
                        name="data.anomalies",
                        arguments={"table_id": table.table_id, "column": "value"},
                    )
                ]
            ),
            response(content="La fila 4 contiene el valor anómalo 1000."),
        ]
    )

    def infer(request):
        requests.append(request)
        return next(replies)

    result = AgentController(
        infer_model=infer,
        action_executor=executor,
    ).run(AgentRunRequest(user_message="describe y encuentra anomalías", max_steps=3))

    assert result.final_answer == "La fila 4 contiene el valor anómalo 1000."
    assert [item.action for item in result.observations] == [
        "data.describe",
        "data.anomalies",
    ]
    assert requests[-1].messages[-1].role == "tool"
    assert "1000" in requests[-1].messages[-1].content


def test_plot_reports_real_artifact_from_measurable_data() -> None:
    calls = []

    def plot_writer(**payload):
        calls.append(payload)
        return str(Path("safe-output") / "chart.png")

    analyzer = DataAnalyzer(output_dir=Path("safe-output"), plot_writer=plot_writer)
    table = analyzer.load_bytes("plot.csv", b"x,y\n1,10\n2,20\n")

    artifact = analyzer.plot(
        table.table_id,
        x_column="x",
        y_column="y",
        kind="line",
    )

    assert artifact == str(Path("safe-output") / "chart.png")
    assert calls[0]["x"] == [1, 2]
    assert calls[0]["y"] == [10, 20]
    assert calls[0]["kind"] == "line"
