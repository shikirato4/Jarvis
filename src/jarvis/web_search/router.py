from __future__ import annotations

from .models import WebSearchHit, WebSearchResponse


_WEB_TRIGGERS = (
    "hoy",
    "actual",
    "actualmente",
    "ultimo",
    "ultima",
    "reciente",
    "noticia",
    "noticias",
    "precio",
    "disponibilidad",
    "version reciente",
    "nueva version",
    "busca",
    "buscalo",
    "internet",
    "fuentes",
    "citas",
    "2026",
    "que paso",
    "quien es actualmente",
    "cuanto cuesta",
    "donde comprar",
    "esta disponible",
    "documentacion actual",
)

_LOCAL_TRIGGERS = (
    "sin internet",
    "offline",
    "archivo",
    ".env",
    "token",
    "password",
    "pin",
    "patch",
    "aplica",
    "edita",
    "codigo local",
    "este proyecto",
)


def should_use_web_search(text: str, *, mode: str = "auto") -> bool:
    lowered = (text or "").casefold()
    if mode == "offline" or mode == "disabled":
        return False
    if any(marker in lowered for marker in _LOCAL_TRIGGERS):
        return False
    if mode == "online":
        return True
    return any(marker in lowered for marker in _WEB_TRIGGERS)


def build_grounded_web_prompt(user_query: str, response: WebSearchResponse, *, max_sources: int = 3, snippet_chars: int = 500) -> str:
    selected_hits = select_synthesis_hits(response.hits, max_sources=max_sources, snippet_chars=snippet_chars)
    sources = []
    for index, hit in enumerate(selected_hits, start=1):
        line = f"{index}. {hit.title} ({hit.source})\nURL: {hit.url}\nResumen: {hit.snippet}"
        sources.append(line)
    source_block = "\n\n".join(sources) if sources else "Sin fuentes disponibles."
    source_count = len(response.hits)
    used_count = len(selected_hits)
    count_line = source_count_message(source_count, used_count)
    return (
        "Responde como Jarvis. Usa las fuentes web solo como contexto de referencia. "
        "No digas que eres ChatGPT, OpenAI ni Gemini. No inventes datos no cubiertos por las fuentes. "
        "Formato requerido: 'Busque en la web.', luego el conteo real de fuentes, luego 'Resumen:' y luego 'Fuentes:'. "
        "No inventes conteos de fuentes.\n\n"
        f"Pregunta del usuario:\n{user_query}\n\n"
        f"Conteo real:\n{count_line}\n\n"
        f"Fuentes encontradas por Brave Search:\n{source_block}"
    )


def source_count_message(source_count: int, used_count: int | None = None) -> str:
    source_count = max(0, int(source_count or 0))
    if used_count is None:
        used_count = source_count
    used_count = max(0, min(int(used_count or 0), source_count))
    if source_count == 0:
        return "No encontre fuentes confiables para esta busqueda."
    noun = "fuente" if source_count == 1 else "fuentes"
    if used_count and used_count < source_count:
        return f"Encontre {source_count} {noun} y use {used_count} para redactar la respuesta."
    return f"Encontre {source_count} {noun}."


def select_synthesis_hits(hits: list[WebSearchHit], *, max_sources: int = 3, snippet_chars: int = 500) -> list[WebSearchHit]:
    seen_domains: set[str] = set()
    selected: list[WebSearchHit] = []
    limit = min(max(1, int(max_sources or 3)), 5)
    snippet_limit = min(max(120, int(snippet_chars or 500)), 1000)
    ranked = sorted(hits, key=_hit_score, reverse=True)
    for hit in ranked:
        domain = (hit.source or "").casefold() or hit.url.casefold()
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        selected.append(
            WebSearchHit(
                title=hit.title[:180],
                url=hit.url[:500],
                snippet=hit.snippet[:snippet_limit],
                source=hit.source[:120],
                provider=hit.provider,
                rank=len(selected) + 1,
                published_at=hit.published_at,
                metadata={},
            )
        )
        if len(selected) >= limit:
            break
    return selected


def _hit_score(hit: WebSearchHit) -> tuple[int, int, int, int]:
    snippet_len = len(hit.snippet or "")
    title_len = len(hit.title or "")
    has_source = 1 if hit.source else 0
    rank_bonus = max(0, 20 - int(hit.rank or 0))
    return (min(snippet_len, 500), has_source, rank_bonus, min(title_len, 160))
