from __future__ import annotations

import gzip
from html.parser import HTMLParser
import io
import ipaddress
import re
import socket
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .models import FetchedSource, SourceCandidate


_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}
_BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata",
}
_TEXT_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "application/pdf",
)


class UnsafeUrlError(ValueError):
    pass


@dataclass(frozen=True)
class RawPage:
    url: str
    content_type: str
    body: bytes
    headers: dict[str, str]


def normalize_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold().removeprefix("www.")
    if not scheme or not hostname:
        return ""
    port = parsed.port
    netloc = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_PARAMETERS
        ),
        doseq=True,
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def validate_public_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple]] | None = None,
) -> str:
    original = urlsplit((url or "").strip())
    if original.username or original.password:
        raise UnsafeUrlError("URLs containing credentials are forbidden")
    normalized = normalize_url(url)
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("only http and https URLs are allowed")
    host = parsed.hostname or ""
    if host.casefold() in _BLOCKED_HOSTNAMES or host.casefold().endswith(".local"):
        raise UnsafeUrlError("local and metadata hosts are forbidden")
    try:
        literal_ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not literal_ip.is_global:
        raise UnsafeUrlError("private, loopback, link-local and reserved addresses are forbidden")
    try:
        resolve = resolver or socket.getaddrinfo
        addresses = resolve(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise UnsafeUrlError("URL host could not be resolved") from exc
    if not addresses:
        raise UnsafeUrlError("URL host did not resolve")
    for item in addresses:
        address = item[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeUrlError("URL resolved to an invalid address") from exc
        if not ip.is_global:
            raise UnsafeUrlError("private, loopback, link-local and reserved addresses are forbidden")
    return normalized


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, validator: Callable[[str], str]) -> None:
        super().__init__()
        self._validator = validator

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        self._validator(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_transport(
    url: str,
    timeout_seconds: float,
    max_bytes: int,
    *,
    validator: Callable[[str], str],
) -> RawPage:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.8",
            "Accept-Encoding": "gzip",
            "User-Agent": "Jarvis-ResearchReader/1.0",
        },
    )
    opener = build_opener(_SafeRedirectHandler(validator))
    with opener.open(request, timeout=timeout_seconds) as response:  # noqa: S310 - validated public URL and every redirect.
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("page exceeds configured byte budget")
        headers = {str(key).casefold(): str(value) for key, value in response.headers.items()}
        if "gzip" in headers.get("content-encoding", "").casefold():
            body = gzip.decompress(body)
            if len(body) > max_bytes:
                raise ValueError("decompressed page exceeds configured byte budget")
        return RawPage(
            url=response.geturl(),
            content_type=headers.get("content-type", "application/octet-stream").split(";", 1)[0].casefold(),
            body=body,
            headers=headers,
        )


class _MainContentParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "aside", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self.metadata: dict[str, str] = {}
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            values = {str(key).casefold(): value for key, value in attrs if value}
            key = str(values.get("property") or values.get("name") or "").casefold()
            content = str(values.get("content") or "").strip()
            aliases = {
                "author": "author",
                "article:author": "author",
                "byl": "author",
                "og:site_name": "organization",
                "application-name": "organization",
                "article:published_time": "published_at",
                "date": "published_at",
                "datepublished": "published_at",
                "article:modified_time": "updated_at",
                "datemodified": "updated_at",
            }
            if key in aliases and content:
                self.metadata.setdefault(aliases[key], content)
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "article", "section", "br"}:
            self._parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "article", "section"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        compact = " ".join(data.split())
        if not compact:
            return
        if self._in_title:
            self.title = f"{self.title} {compact}".strip()
        self._parts.append(compact)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in " ".join(self._parts).splitlines()]
        return "\n".join(line for line in lines if line)


class SourceFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 12.0,
        max_bytes: int = 2_000_000,
        cache_ttl_seconds: float = 600.0,
        resolver: Callable[..., list[tuple]] | None = None,
        transport: Callable[[str, float, int], RawPage] | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._cache_ttl_seconds = cache_ttl_seconds
        self._resolver = resolver
        self._transport = transport
        self._cache: dict[str, tuple[float, FetchedSource]] = {}

    def fetch(self, candidate: SourceCandidate) -> FetchedSource:
        try:
            safe_url = validate_public_url(candidate.url, resolver=self._resolver)
        except UnsafeUrlError as exc:
            return self._failed(candidate, "blocked", str(exc))
        cached = self._cache.get(safe_url)
        if cached and time.monotonic() - cached[0] <= self._cache_ttl_seconds:
            return cached[1].model_copy(deep=True)
        try:
            raw = (
                self._transport(safe_url, self._timeout_seconds, self._max_bytes)
                if self._transport is not None
                else _default_transport(
                    safe_url,
                    self._timeout_seconds,
                    self._max_bytes,
                    validator=lambda value: validate_public_url(value, resolver=self._resolver),
                )
            )
            final_url = validate_public_url(raw.url, resolver=self._resolver)
            content_type = raw.content_type.split(";", 1)[0].casefold()
            if not any(content_type.startswith(item) for item in _TEXT_CONTENT_TYPES):
                return self._failed(candidate, "unsupported", f"unsupported content type: {content_type}")
            if content_type == "application/pdf" or raw.body.startswith(b"%PDF"):
                text, pages, extracted = self._extract_pdf_with_metadata(raw.body)
                title = extracted.get("title") or candidate.title
                links: list[str] = []
            elif content_type in {"text/html", "application/xhtml+xml"}:
                text, title, links, extracted = self._extract_html(raw.body, raw.headers, final_url)
                pages = []
            else:
                text = self._decode(raw.body, raw.headers)
                title = candidate.title
                links = []
                pages = []
                extracted = {}
            if not text.strip():
                return self._failed(candidate, "empty", "page contained no extractable text")
            fetched = FetchedSource(
                source_id=candidate.source_id,
                url=candidate.url,
                final_url=final_url,
                title=(title or candidate.title)[:300],
                domain=urlsplit(final_url).hostname or candidate.domain,
                content_type=content_type,
                text=text,
                links=links[:100],
                published_at=extracted.get("published_at") or candidate.published_at,
                updated_at=extracted.get("updated_at") or raw.headers.get("last-modified"),
                page_references=pages,
                provider=candidate.provider,
                author=extracted.get("author") or candidate.metadata.get("author"),
                organization=extracted.get("organization"),
            )
            self._cache[safe_url] = (time.monotonic(), fetched.model_copy(deep=True))
            return fetched
        except TimeoutError:
            return self._failed(candidate, "timeout", "page request timed out")
        except Exception as exc:  # noqa: BLE001 - isolated untrusted document boundary.
            return self._failed(candidate, "failed", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _decode(body: bytes, headers: dict[str, str]) -> str:
        match = re.search(r"charset=([\w-]+)", headers.get("content-type", ""), re.I)
        encoding = match.group(1) if match else "utf-8"
        return body.decode(encoding, errors="replace")

    def _extract_html(self, body: bytes, headers: dict[str, str], base_url: str) -> tuple[str, str, list[str], dict[str, str]]:
        parser = _MainContentParser()
        parser.feed(self._decode(body, headers))
        links = [urljoin(base_url, item) for item in parser.links if item.startswith(("http://", "https://", "/"))]
        return parser.text(), parser.title, links, parser.metadata

    @staticmethod
    def _extract_pdf(body: bytes) -> tuple[str, list[int]]:
        text, pages, _metadata = SourceFetcher._extract_pdf_with_metadata(body)
        return text, pages

    @staticmethod
    def _extract_pdf_with_metadata(body: bytes) -> tuple[str, list[int], dict[str, str]]:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(body))
        parts: list[str] = []
        pages: list[int] = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"[Page {index}]\n{text}")
                pages.append(index)
        raw_metadata = getattr(reader, "metadata", None) or {}
        metadata = {
            key: str(value).strip()
            for key, value in {
                "title": raw_metadata.get("/Title"),
                "author": raw_metadata.get("/Author"),
                "organization": raw_metadata.get("/Company") or raw_metadata.get("/Creator"),
                "published_at": raw_metadata.get("/CreationDate"),
                "updated_at": raw_metadata.get("/ModDate"),
            }.items()
            if value
        }
        return "\n\n".join(parts), pages, metadata

    @staticmethod
    def _failed(candidate: SourceCandidate, status: str, error: str) -> FetchedSource:
        return FetchedSource(
            source_id=candidate.source_id,
            url=candidate.url,
            final_url=candidate.normalized_url or candidate.url,
            title=candidate.title,
            domain=candidate.domain,
            content_type="",
            status=status,
            error=error,
        )
