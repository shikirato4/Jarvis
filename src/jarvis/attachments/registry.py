from __future__ import annotations

import io
import json
from pathlib import PurePath, PureWindowsPath
import re
import zipfile

from .models import AttachmentCategory, DetectedAttachmentType


class AttachmentSecurityError(ValueError):
    pass


class AttachmentTooLargeError(ValueError):
    pass


_EXTENSION_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".json": "application/json",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".cs",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".sql",
    ".sh",
    ".ps1",
}
_SENSITIVE_NAME = re.compile(
    r"(^\.env(?:\.|$)|\.(?:pem|key|pfx|kdbx)$|secret|credential|token)", re.IGNORECASE
)
_SENSITIVE_CONTENT = re.compile(
    r"(?im)(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|^\s*(?:nvidia_api_key|api_key|password|secret|token)\s*[:=]\s*\S+)"
)


class AttachmentTypeRegistry:
    def validate_name(self, name: str) -> str:
        normalized = name.strip()
        posix = PurePath(normalized)
        windows = PureWindowsPath(normalized)
        if (
            not normalized
            or posix.is_absolute()
            or windows.is_absolute()
            or posix.name != normalized
            or windows.name != normalized
            or normalized in {".", ".."}
        ):
            raise AttachmentSecurityError(
                "El nombre del archivo contiene una ruta no permitida."
            )
        if _SENSITIVE_NAME.search(normalized):
            raise AttachmentSecurityError(
                "El archivo parece contener un secreto o credencial y no se adjuntó."
            )
        safe = re.sub(r"[^\w.()\- ]", "_", normalized, flags=re.UNICODE).strip(" .")
        if not safe:
            raise AttachmentSecurityError("El nombre del archivo no es seguro.")
        return safe[:180]

    def detect(
        self,
        name: str,
        data: bytes,
        *,
        claimed_mime: str | None = None,
    ) -> DetectedAttachmentType:
        safe_name = self.validate_name(name)
        if not data:
            raise AttachmentSecurityError("El archivo está vacío y no se adjuntó.")
        self._reject_sensitive_content(data)
        detected = self._detect_content(safe_name, data)
        expected = _EXTENSION_MIME.get(PurePath(safe_name).suffix.casefold())
        if claimed_mime and not _mime_matches(claimed_mime, detected.mime_type):
            raise AttachmentSecurityError(
                f"El MIME declarado ({claimed_mime}) no coincide con el contenido detectado ({detected.mime_type})."
            )
        if expected and not _mime_matches(expected, detected.mime_type):
            raise AttachmentSecurityError(
                f"La extensión declara MIME {expected}, pero el contenido detectado es {detected.mime_type}."
            )
        return detected

    def _detect_content(self, name: str, data: bytes) -> DetectedAttachmentType:
        suffix = PurePath(name).suffix.casefold()
        lowered = data[:4096].lstrip().lower()
        if data.startswith(b"MZ") or data.startswith(b"\x7fELF"):
            raise AttachmentSecurityError(
                "El contenido ejecutable no está permitido como adjunto."
            )
        if lowered.startswith((b"<html", b"<!doctype html", b"<script", b"<svg")):
            raise AttachmentSecurityError(
                "El contenido activo HTML/SVG no está permitido como adjunto."
            )
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return _detected(AttachmentCategory.IMAGE, "image/png")
        if data.startswith(b"\xff\xd8\xff"):
            return _detected(AttachmentCategory.IMAGE, "image/jpeg")
        if data.startswith((b"GIF87a", b"GIF89a")):
            return _detected(AttachmentCategory.IMAGE, "image/gif")
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return _detected(AttachmentCategory.IMAGE, "image/webp")
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
            return self._detected_text(suffix, text, "utf-16")
        if len(data) >= 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            if suffix == ".m4a" or brand in {b"M4A ", b"M4B ", b"M4P "}:
                return _detected(AttachmentCategory.AUDIO, "audio/mp4")
            if suffix == ".mov" or brand == b"qt  ":
                return _detected(AttachmentCategory.VIDEO, "video/quicktime")
            return _detected(AttachmentCategory.VIDEO, "video/mp4")
        if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
            return _detected(AttachmentCategory.AUDIO, "audio/wav")
        if data.startswith(b"ID3") or (
            len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
        ):
            return _detected(AttachmentCategory.AUDIO, "audio/mpeg")
        if data.startswith(b"fLaC"):
            return _detected(AttachmentCategory.AUDIO, "audio/flac")
        if data.startswith(b"OggS"):
            return _detected(AttachmentCategory.AUDIO, "audio/ogg")
        if data.startswith(b"\x1aE\xdf\xa3"):
            mime = "video/webm" if suffix == ".webm" else "video/x-matroska"
            return _detected(AttachmentCategory.VIDEO, mime)
        if data.startswith(b"%PDF"):
            if b"%%EOF" not in data[-1024:]:
                raise AttachmentSecurityError(
                    "PDF corrupto: no se encontró el marcador de cierre."
                )
            return _detected(AttachmentCategory.DOCUMENT, "application/pdf")
        if data.startswith(b"PK\x03\x04"):
            return self._detect_zip_document(data)
        decoded = _decode_text(data)
        if decoded is None:
            return _detected(
                AttachmentCategory.OTHER, "application/octet-stream", modality="binary"
            )
        text, encoding = decoded
        return self._detected_text(suffix, text, encoding)

    @staticmethod
    def _detected_text(
        suffix: str,
        text: str,
        encoding: str,
    ) -> DetectedAttachmentType:
        if suffix in _CODE_EXTENSIONS:
            return _detected(AttachmentCategory.CODE, "text/x-source", modality="text", metadata={"encoding": encoding})
        if suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                raise AttachmentSecurityError("JSON corrupto o inválido.") from exc
            return _detected(
                AttachmentCategory.DOCUMENT, "application/json", modality="text", metadata={"encoding": encoding}
            )
        if suffix == ".csv":
            return _detected(AttachmentCategory.DOCUMENT, "text/csv", modality="text", metadata={"encoding": encoding})
        if suffix == ".tsv":
            return _detected(AttachmentCategory.DOCUMENT, "text/tab-separated-values", modality="text", metadata={"encoding": encoding})
        if suffix in {".md", ".markdown"}:
            return _detected(AttachmentCategory.TEXT, "text/markdown", modality="text", metadata={"encoding": encoding})
        return _detected(AttachmentCategory.TEXT, "text/plain", modality="text", metadata={"encoding": encoding})

    def _detect_zip_document(self, data: bytes) -> DetectedAttachmentType:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = archive.namelist()
                if any(_unsafe_archive_name(name) for name in names):
                    raise AttachmentSecurityError(
                        "DOCX inseguro: contiene rutas fuera del documento."
                    )
                if any(name.casefold().endswith("vbaproject.bin") for name in names):
                    raise AttachmentSecurityError(
                        "El documento contiene macros o contenido activo y no se adjuntó."
                    )
                if "[Content_Types].xml" in names and "word/document.xml" in names:
                    return _detected(
                        AttachmentCategory.DOCUMENT,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        modality="text",
                    )
                if "[Content_Types].xml" in names and "xl/workbook.xml" in names:
                    return _detected(
                        AttachmentCategory.DOCUMENT,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        modality="text",
                    )
        except zipfile.BadZipFile as exc:
            raise AttachmentSecurityError(
                "DOCX corrupto: el contenedor ZIP no se puede leer."
            ) from exc
        return _detected(AttachmentCategory.OTHER, "application/zip", modality="binary")

    @staticmethod
    def _reject_sensitive_content(data: bytes) -> None:
        sample = data[:65_536].decode("utf-8", errors="ignore")
        if _SENSITIVE_CONTENT.search(sample):
            raise AttachmentSecurityError(
                "El contenido parece incluir un secreto o credencial y no se almacenó."
            )


def _detected(
    category: AttachmentCategory,
    mime_type: str,
    *,
    modality: str | None = None,
    metadata: dict | None = None,
) -> DetectedAttachmentType:
    return DetectedAttachmentType(
        category=category,
        mime_type=mime_type,
        required_modality=modality or category.value,
        metadata=metadata or {},
    )


def _mime_matches(expected: str, detected: str) -> bool:
    aliases = {
        "audio/mp3": "audio/mpeg",
        "text/x-markdown": "text/markdown",
        "application/x-pdf": "application/pdf",
    }
    return aliases.get(expected.casefold(), expected.casefold()) == aliases.get(
        detected.casefold(), detected.casefold()
    )


def _unsafe_archive_name(name: str) -> bool:
    posix = PurePath(name)
    windows = PureWindowsPath(name)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    )


def _decode_text(data: bytes) -> tuple[str, str] | None:
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(text):
            return text, encoding
    try:
        text = data.decode("cp1252")
    except UnicodeDecodeError:
        return None
    if _looks_like_text(text):
        return text, "cp1252"
    return None


def _looks_like_text(text: str) -> bool:
    if not text:
        return False
    controls = sum(
        1
        for char in text
        if ord(char) < 32 and char not in {"\n", "\r", "\t", "\f"}
    )
    return controls / len(text) <= 0.02
