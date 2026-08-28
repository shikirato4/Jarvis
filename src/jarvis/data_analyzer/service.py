from __future__ import annotations

import csv
import io
import json
import math
import statistics
import zipfile
from datetime import date, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any
from uuid import uuid4
from xml.etree import ElementTree

from .models import DataAnalysisError, DataTable


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class DataAnalyzer:
    def __init__(
        self,
        *,
        max_bytes: int = 25_000_000,
        max_rows: int = 100_000,
        output_dir: Path | None = None,
        plot_writer: Callable[..., str] | None = None,
    ) -> None:
        self._max_bytes = max(max_bytes, 1)
        self._max_rows = max(max_rows, 1)
        self._tables: dict[str, DataTable] = {}
        self._output_dir = output_dir
        self._plot_writer = plot_writer or self._write_plot

    def load_path(self, path: str | Path) -> DataTable:
        source = Path(path)
        try:
            data = source.read_bytes()
        except OSError as exc:
            raise DataAnalysisError("data file could not be read") from exc
        return self.load_bytes(source.name, data)

    def load_bytes(self, source_name: str, data: bytes) -> DataTable:
        if not data:
            raise DataAnalysisError("data file is empty")
        if len(data) > self._max_bytes:
            raise DataAnalysisError("data file exceeds the configured size budget")
        suffix = Path(source_name).suffix.casefold()
        if suffix in {".xlsx", ".xlsm"} or data.startswith(b"PK\x03\x04"):
            columns, rows, metadata = self._parse_xlsx(data)
        elif suffix == ".json" or data.lstrip().startswith((b"[", b"{")):
            columns, rows, metadata = self._parse_json(data)
        else:
            columns, rows, metadata = self._parse_delimited(data, suffix)
        if len(rows) > self._max_rows:
            raise DataAnalysisError("data table exceeds the configured row budget")
        table = DataTable(
            table_id=f"table-{uuid4().hex}",
            source_name=Path(source_name).name,
            columns=columns,
            rows=rows,
            metadata=metadata,
        )
        self._tables[table.table_id] = table
        return table

    def get(self, table_id: str) -> DataTable:
        try:
            return self._tables[table_id]
        except KeyError as exc:
            raise DataAnalysisError("unknown data table") from exc

    def describe(self, table_id: str) -> dict[str, Any]:
        table = self.get(table_id)
        columns: dict[str, dict[str, Any]] = {}
        for name in table.columns:
            values = [row.get(name) for row in table.rows]
            present = [value for value in values if value is not None]
            kind = self._column_type(present)
            details: dict[str, Any] = {
                "type": kind,
                "null_count": len(values) - len(present),
                "unique_count": len({self._hashable(value) for value in present}),
            }
            numeric = [float(value) for value in present if self._is_number(value)]
            if kind == "number" and numeric:
                details["min"] = min(numeric)
                details["max"] = max(numeric)
                details["mean"] = statistics.fmean(numeric)
                details["median"] = statistics.median(numeric)
            columns[name] = details
        return {
            "table_id": table.table_id,
            "source_name": table.source_name,
            "row_count": len(table.rows),
            "column_count": len(table.columns),
            "columns": columns,
            "metadata": table.metadata,
        }

    def filter(
        self,
        table_id: str,
        *,
        column: str,
        operator: str,
        value: Any,
    ) -> list[dict[str, Any]]:
        table = self.get(table_id)
        self._require_column(table, column)
        operations = {
            "eq": lambda left: left == value,
            "ne": lambda left: left != value,
            "gt": lambda left: left is not None and left > value,
            "ge": lambda left: left is not None and left >= value,
            "lt": lambda left: left is not None and left < value,
            "le": lambda left: left is not None and left <= value,
            "contains": lambda left: value.casefold() in str(left).casefold(),
            "is_null": lambda left: left is None,
            "not_null": lambda left: left is not None,
        }
        predicate = operations.get(operator)
        if predicate is None:
            raise DataAnalysisError("unsupported filter operator")
        try:
            return [row.copy() for row in table.rows if predicate(row.get(column))]
        except TypeError as exc:
            raise DataAnalysisError("filter value is incompatible with the column") from exc

    def sort(
        self,
        table_id: str,
        *,
        column: str,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        table = self.get(table_id)
        self._require_column(table, column)
        present = [row for row in table.rows if row.get(column) is not None]
        missing = [row for row in table.rows if row.get(column) is None]
        try:
            ordered = sorted(present, key=lambda row: row[column], reverse=descending)
        except TypeError as exc:
            raise DataAnalysisError("mixed column cannot be sorted safely") from exc
        return [row.copy() for row in [*ordered, *missing]]

    def aggregate(
        self,
        table_id: str,
        *,
        group_by: str,
        column: str,
        operation: str,
    ) -> dict[str, float | int | None]:
        table = self.get(table_id)
        self._require_column(table, group_by)
        self._require_column(table, column)
        groups: dict[str, list[Any]] = {}
        for row in table.rows:
            key = str(row.get(group_by))
            value = row.get(column)
            if value is not None:
                groups.setdefault(key, []).append(value)
        output: dict[str, float | int | None] = {}
        for key, values in groups.items():
            if operation == "count":
                output[key] = len(values)
                continue
            numeric = [float(value) for value in values if self._is_number(value)]
            if len(numeric) != len(values) or not numeric:
                raise DataAnalysisError("aggregate requires a numeric column")
            if operation == "sum":
                output[key] = sum(numeric)
            elif operation == "mean":
                output[key] = statistics.fmean(numeric)
            elif operation == "min":
                output[key] = min(numeric)
            elif operation == "max":
                output[key] = max(numeric)
            else:
                raise DataAnalysisError("unsupported aggregate operation")
        return output

    def anomalies(self, table_id: str, *, column: str) -> list[dict[str, Any]]:
        table = self.get(table_id)
        self._require_column(table, column)
        indexed = [
            (index, float(row[column]))
            for index, row in enumerate(table.rows)
            if self._is_number(row.get(column))
        ]
        if len(indexed) < 4:
            return []
        values = [value for _, value in indexed]
        median = statistics.median(values)
        median_deviation = statistics.median(
            abs(value - median) for value in values
        )
        if median_deviation > 0:
            is_anomaly = lambda value: (  # noqa: E731 - local statistical predicate.
                0.6745 * abs(value - median) / median_deviation > 3.5
            )
        else:
            ordered = sorted(values)
            lower, upper = self._quartiles(ordered)
            iqr = upper - lower
            low_fence = lower - 1.5 * iqr
            high_fence = upper + 1.5 * iqr
            is_anomaly = lambda value: value < low_fence or value > high_fence  # noqa: E731
        return [
            {"row_index": index, "value": value}
            for index, value in indexed
            if is_anomaly(value)
        ]

    def plot(
        self,
        table_id: str,
        *,
        x_column: str,
        y_column: str,
        kind: str = "line",
    ) -> str:
        table = self.get(table_id)
        self._require_column(table, x_column)
        self._require_column(table, y_column)
        if kind not in {"line", "bar", "scatter"}:
            raise DataAnalysisError("unsupported plot kind")
        pairs = [
            (row.get(x_column), row.get(y_column))
            for row in table.rows
            if row.get(x_column) is not None and row.get(y_column) is not None
        ]
        if not pairs:
            raise DataAnalysisError("plot has no complete data points")
        if self._output_dir is None:
            raise DataAnalysisError("data plot output directory is not configured")
        try:
            return self._plot_writer(
                output_dir=self._output_dir,
                x=[pair[0] for pair in pairs],
                y=[pair[1] for pair in pairs],
                x_label=x_column,
                y_label=y_column,
                kind=kind,
            )
        except DataAnalysisError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataAnalysisError("data plot could not be generated") from exc

    @classmethod
    def _parse_delimited(
        cls,
        data: bytes,
        suffix: str,
    ) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
        text, encoding = cls._decode(data)
        sample = text[:8_192]
        delimiter = "\t" if suffix == ".tsv" else None
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        try:
            raw_rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        except csv.Error as exc:
            raise DataAnalysisError("delimited data is corrupt") from exc
        if not raw_rows or not any(cell.strip() for cell in raw_rows[0]):
            raise DataAnalysisError("delimited data has no header")
        columns = cls._unique_headers(raw_rows[0])
        rows = [
            {
                column: cls._coerce(raw[index] if index < len(raw) else "")
                for index, column in enumerate(columns)
            }
            for raw in raw_rows[1:]
            if any(cell.strip() for cell in raw)
        ]
        return columns, rows, {"format": "delimited", "delimiter": delimiter, "encoding": encoding}

    @classmethod
    def _parse_json(
        cls,
        data: bytes,
    ) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
        text, encoding = cls._decode(data)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DataAnalysisError("JSON data is corrupt") from exc
        if isinstance(payload, dict):
            candidates = [value for value in payload.values() if isinstance(value, list)]
            if len(candidates) != 1:
                raise DataAnalysisError("JSON must contain one record array")
            payload = candidates[0]
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise DataAnalysisError("JSON must be an array of objects")
        columns: list[str] = []
        for item in payload:
            for key in item:
                name = str(key)
                if name not in columns:
                    columns.append(name)
        rows = [{column: item.get(column) for column in columns} for item in payload]
        return columns, rows, {"format": "json", "encoding": encoding}

    @classmethod
    def _parse_xlsx(
        cls,
        data: bytes,
    ) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise DataAnalysisError("XLSX container is corrupt") from exc
        with archive:
            names = set(archive.namelist())
            if any(name.casefold().endswith("vbaproject.bin") for name in names):
                raise DataAnalysisError("XLSX active content is not allowed")
            try:
                sheet_xml = archive.read("xl/worksheets/sheet1.xml")
            except KeyError as exc:
                raise DataAnalysisError("XLSX has no readable first worksheet") from exc
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                try:
                    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                    shared = ["".join(node.itertext()) for node in root.findall(f"{_XLSX_NS}si")]
                except ElementTree.ParseError as exc:
                    raise DataAnalysisError("XLSX shared strings are corrupt") from exc
            try:
                root = ElementTree.fromstring(sheet_xml)
            except ElementTree.ParseError as exc:
                raise DataAnalysisError("XLSX worksheet XML is corrupt") from exc
            matrix: list[list[Any]] = []
            formula_count = 0
            for row in root.findall(f".//{_XLSX_NS}row"):
                cells: list[Any] = []
                for cell in row.findall(f"{_XLSX_NS}c"):
                    while len(cells) < cls._column_index(cell.get("r") or "A1"):
                        cells.append(None)
                    if cell.find(f"{_XLSX_NS}f") is not None:
                        formula_count += 1
                    cell_type = cell.get("t")
                    value_node = cell.find(f"{_XLSX_NS}v")
                    inline = cell.find(f"{_XLSX_NS}is")
                    raw = value_node.text if value_node is not None else ("".join(inline.itertext()) if inline is not None else "")
                    if cell_type == "s" and raw:
                        try:
                            value: Any = shared[int(raw)]
                        except (ValueError, IndexError) as exc:
                            raise DataAnalysisError("XLSX shared string index is invalid") from exc
                    else:
                        value = cls._coerce(raw)
                    cells.append(value)
                matrix.append(cells)
        if not matrix:
            raise DataAnalysisError("XLSX worksheet is empty")
        columns = cls._unique_headers([str(value or "") for value in matrix[0]])
        rows = [
            {column: raw[index] if index < len(raw) else None for index, column in enumerate(columns)}
            for raw in matrix[1:]
            if any(value is not None for value in raw)
        ]
        return columns, rows, {"format": "xlsx", "formula_count": formula_count, "macros_executed": False}

    @staticmethod
    def _decode(data: bytes) -> tuple[str, str]:
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError:
                continue
        raise DataAnalysisError("data encoding is unsupported")

    @staticmethod
    def _unique_headers(raw: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        headers: list[str] = []
        for index, value in enumerate(raw, start=1):
            base = value.strip() or f"column_{index}"
            counts[base] = counts.get(base, 0) + 1
            headers.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
        return headers

    @staticmethod
    def _coerce(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized or normalized.casefold() in {"null", "none", "na", "n/a"}:
            return None
        lowered = normalized.casefold()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            if normalized.lstrip("+-").isdigit():
                return int(normalized)
            number = float(normalized)
            if math.isfinite(number):
                return number
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return normalized

    @staticmethod
    def _column_type(values: list[Any]) -> str:
        kinds = set()
        for value in values:
            if isinstance(value, bool):
                kinds.add("boolean")
            elif DataAnalyzer._is_number(value):
                kinds.add("number")
            elif isinstance(value, (date, datetime)):
                kinds.add("date")
            else:
                kinds.add("text")
        if not kinds:
            return "empty"
        return next(iter(kinds)) if len(kinds) == 1 else "mixed"

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _hashable(value: Any) -> Any:
        try:
            hash(value)
        except TypeError:
            return json.dumps(value, sort_keys=True, default=str)
        return value

    @staticmethod
    def _require_column(table: DataTable, column: str) -> None:
        if column not in table.columns:
            raise DataAnalysisError("unknown data column")

    @staticmethod
    def _quartiles(values: list[float]) -> tuple[float, float]:
        midpoint = len(values) // 2
        lower_half = values[:midpoint]
        upper_half = values[midpoint + (len(values) % 2) :]
        return statistics.median(lower_half), statistics.median(upper_half)

    @staticmethod
    def _column_index(reference: str) -> int:
        letters = "".join(character for character in reference if character.isalpha()).upper()
        index = 0
        for character in letters:
            index = index * 26 + (ord(character) - ord("A") + 1)
        return max(index - 1, 0)

    @staticmethod
    def _write_plot(
        *,
        output_dir: Path,
        x: list[Any],
        y: list[Any],
        x_label: str,
        y_label: str,
        kind: str,
    ) -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"data_{kind}_{uuid4().hex[:10]}.png"
        figure, axis = plt.subplots(figsize=(8, 4.5))
        try:
            if kind == "line":
                axis.plot(x, y)
            elif kind == "bar":
                axis.bar(x, y)
            else:
                axis.scatter(x, y)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            figure.tight_layout()
            figure.savefig(path, dpi=140)
        finally:
            plt.close(figure)
        return str(path)
