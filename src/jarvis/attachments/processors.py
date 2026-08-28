from __future__ import annotations

import csv
import io
import json
from typing import Any, Protocol
from xml.etree import ElementTree
import zipfile

from .models import AttachmentCategory, AttachmentDescriptor


class AttachmentProcessorUnavailableError(RuntimeError):
    pass


class AttachmentProcessor(Protocol):
    categories: tuple[AttachmentCategory, ...]

    def process(
        self, descriptor: AttachmentDescriptor, data: bytes
    ) -> dict[str, Any]: ...


class ProcessorRegistry:
    def __init__(self) -> None:
        self._processors: dict[AttachmentCategory, AttachmentProcessor] = {}

    def register(self, processor: AttachmentProcessor) -> None:
        for category in processor.categories:
            self._processors[category] = processor

    def get(self, category: AttachmentCategory) -> AttachmentProcessor | None:
        return self._processors.get(category)


class TextDocumentProcessor:
    categories = (
        AttachmentCategory.TEXT,
        AttachmentCategory.CODE,
        AttachmentCategory.DOCUMENT,
    )

    def __init__(self, *, context_char_budget: int = 4_000) -> None:
        self._context_char_budget = max(context_char_budget, 1)

    def process(self, descriptor: AttachmentDescriptor, data: bytes) -> dict[str, Any]:
        page_references: list[dict[str, Any]] = []
        if (
            descriptor.mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            text = _extract_docx(data)
        elif (
            descriptor.mime_type
            == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ):
            text = _extract_xlsx(data)
        elif descriptor.mime_type == "application/pdf":
            text, page_references = _extract_pdf(data)
        elif descriptor.mime_type == "application/json":
            payload = json.loads(data.decode(_descriptor_encoding(descriptor)))
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        elif descriptor.mime_type in {"text/csv", "text/tab-separated-values"}:
            delimiter = "\t" if descriptor.mime_type == "text/tab-separated-values" else ","
            text = _extract_delimited(data, encoding=_descriptor_encoding(descriptor), delimiter=delimiter)
        else:
            text = data.decode(_descriptor_encoding(descriptor))
        normalized = text.replace("\x00", "").strip()
        chunks = _chunk_text(normalized, self._context_char_budget)
        return {
            "text": normalized,
            "chunks": chunks,
            "pages": page_references,
            "context_char_budget": self._context_char_budget,
            "truncated": sum(len(chunk["text"]) for chunk in chunks) < len(normalized),
        }


class ImageProcessor:
    categories = (AttachmentCategory.IMAGE,)

    def __init__(self, *, ocr=None) -> None:
        self._ocr = ocr

    def process(self, descriptor: AttachmentDescriptor, data: bytes) -> dict[str, Any]:
        if self._ocr is None:
            raise AttachmentProcessorUnavailableError(
                "No hay una ruta OCR local configurada para procesar esta imagen."
            )
        result = self._ocr(data, descriptor.mime_type)
        if not isinstance(result, dict):
            raise AttachmentProcessorUnavailableError(
                "El OCR local devolvió un resultado incompatible."
            )
        return result


class AudioProcessor:
    categories = (AttachmentCategory.AUDIO,)

    def __init__(self, *, transcriber=None) -> None:
        self._transcriber = transcriber

    def process(self, descriptor: AttachmentDescriptor, data: bytes) -> dict[str, Any]:
        if self._transcriber is None:
            raise AttachmentProcessorUnavailableError(
                "No hay una ruta de transcripción local configurada para este audio."
            )
        result = self._transcriber(data, descriptor.mime_type)
        if not isinstance(result, dict):
            raise AttachmentProcessorUnavailableError(
                "La transcripción local devolvió un resultado incompatible."
            )
        result.setdefault("segments", [])
        return result


class VideoProcessor:
    categories = (AttachmentCategory.VIDEO,)

    def __init__(self, *, analyzer=None) -> None:
        self._analyzer = analyzer

    def process(self, descriptor: AttachmentDescriptor, data: bytes) -> dict[str, Any]:
        if self._analyzer is None:
            raise AttachmentProcessorUnavailableError(
                "No hay una ruta local de video configurada para extraer fotogramas, audio y transcripción."
            )
        result = self._analyzer(data, descriptor.mime_type)
        if not isinstance(result, dict):
            raise AttachmentProcessorUnavailableError(
                "El analizador de video devolvió un resultado incompatible."
            )
        result.setdefault("frames", [])
        result.setdefault("transcript", [])
        return result


def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise AttachmentProcessorUnavailableError(
            "El documento DOCX no se puede extraer."
        ) from exc
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise AttachmentProcessorUnavailableError(
            "El XML del DOCX está corrupto."
        ) from exc
    return " ".join(text.strip() for text in root.itertext() if text.strip())


def _extract_pdf(data: bytes) -> tuple[str, list[dict[str, Any]]]:
    try:
        import pypdf
    except ImportError as exc:
        raise AttachmentProcessorUnavailableError(
            "La extracción PDF requiere la dependencia opcional pypdf; el OCR para escaneos permanece disponible por separado."
        ) from exc
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except (pypdf.errors.PdfReadError, OSError, ValueError) as exc:
        raise AttachmentProcessorUnavailableError(
            "No fue posible extraer el PDF."
        ) from exc
    references = [
        {"page": index + 1, "text": page.strip()}
        for index, page in enumerate(pages)
        if page.strip()
    ]
    return "\n\n".join(item["text"] for item in references), references


def _extract_delimited(data: bytes, *, encoding: str, delimiter: str) -> str:
    text = data.decode(encoding)
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    return "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)


def _descriptor_encoding(descriptor: AttachmentDescriptor) -> str:
    return str(descriptor.metadata.get("encoding") or "utf-8-sig")


def _extract_xlsx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if any(name.casefold().endswith("vbaproject.bin") for name in names):
                raise AttachmentProcessorUnavailableError(
                    "El XLSX contiene macros o contenido activo."
                )
            shared: list[str] = []
            if "xl/sharedStrings.xml" in names:
                shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = [
                    "".join(node.text or "" for node in item.iter() if _local_name(node.tag) == "t")
                    for item in shared_root.iter()
                    if _local_name(item.tag) == "si"
                ]
            rows: list[str] = []
            for sheet_name in sorted(
                name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml")
            ):
                root = ElementTree.fromstring(archive.read(sheet_name))
                for row in root.iter():
                    if _local_name(row.tag) != "row":
                        continue
                    values: list[str] = []
                    for cell in row:
                        if _local_name(cell.tag) != "c":
                            continue
                        cell_type = cell.attrib.get("t")
                        inline = "".join(
                            node.text or "" for node in cell.iter() if _local_name(node.tag) == "t"
                        )
                        raw = next(
                            (node.text or "" for node in cell if _local_name(node.tag) == "v"),
                            "",
                        )
                        if cell_type == "s" and raw.isdigit() and int(raw) < len(shared):
                            value = shared[int(raw)]
                        else:
                            value = inline or raw
                        values.append(value.strip())
                    if values:
                        rows.append(" | ".join(values))
            return "\n".join(rows)
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise AttachmentProcessorUnavailableError(
            "El documento XLSX no se puede extraer."
        ) from exc


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _chunk_text(text: str, budget: int) -> list[dict[str, Any]]:
    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(text), budget)):
        end = min(start + budget, len(text))
        chunks.append(
            {
                "index": index,
                "start_char": start,
                "end_char": end,
                "text": text[start:end],
            }
        )
    return chunks
