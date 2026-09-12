from __future__ import annotations

import re
import unicodedata

from .models import ResearchDepth, ResearchIntent, ResearchRequirement, SearchMode, SourceScope


def _fold(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char))


def parse_research_intent(question: str, requested: SearchMode | str = SearchMode.AUTO) -> ResearchIntent:
    original = " ".join(question.strip().split())
    text = re.sub(r"^/deepresearch\b", "", original, flags=re.I).strip()
    folded = _fold(text)
    file_explicit = any(term in folded for term in ("mis archivos", "mis documentos", "archivo local", "archivos locales", "mi carpeta", "mis notas", "solo archivos", "unicamente archivos"))
    web_explicit = any(term in folded for term in ("internet", "en la web", "web publica", "solo web", "unicamente web", "con brave", "con you", "con exa", "searxng", "informacion actual"))
    if file_explicit and web_explicit:
        scope = SourceScope.FILES_AND_WEB
    elif file_explicit:
        scope = SourceScope.FILES
    elif web_explicit:
        scope = SourceScope.WEB
    else:
        scope = SourceScope.AUTO

    mode = SearchMode(requested)
    explicit_deep = original.casefold().startswith("/deepresearch") or mode is SearchMode.DEEP
    if explicit_deep:
        depth = ResearchDepth.DEEP
    elif mode in {SearchMode.QUICK, SearchMode.NORMAL} or scope is not SourceScope.AUTO:
        depth = ResearchDepth.NORMAL
    else:
        depth = ResearchDepth.NONE

    wants_apa = bool(re.search(r"\b(apa|bibliograf(?:ia|ía)|referencias?)\b", folded))
    wants_report = bool(re.search(r"\b(informe|report|reporte|secciones|detallad[oa])\b", folded))
    requested_format = "apa_report" if wants_apa and wants_report else "apa" if wants_apa else "report" if wants_report else "answer"

    # Presentation/citation directives are deliberately removed before planning.
    # Substantive clauses remain and become requirements.
    topic = re.sub(r"\b(?:y\s+)?(?:al final(?: de todo)?\s+)?(?:dame|incluye|agrega|pon)?\s*(?:las?\s+)?(?:fuentes?|referencias?|bibliograf(?:ia|ía)|citas?)(?:\s+citadas?)?(?:\s+en\s+formato\s+apa)?\b", " ", text, flags=re.I)
    topic = re.sub(r"\b(?:dame|haz|redacta|prepara)\s+(?:un\s+)?(?:informe|reporte)(?:\s+detallado)?\b", " ", topic, flags=re.I)
    topic = re.sub(r"\bformato\s+apa\b|\bapa\b", " ", topic, flags=re.I)
    topic = " ".join(topic.split()).strip(" ,.;:-") or text

    concepts: list[str] = []
    rules = (
        (r"\b(que es|qué es|contexto|definicion|definición)\b", "context and definition"),
        (r"\b(desarrolla|desarrollo|evoluciona|funciona|mecanismo|proceso)\b", "development or mechanism"),
        (r"\b(causas?|factores?|riesgos?|por que se da|por qué se da|origen)\b", "causes and risk factors"),
        (r"\b(efectos?|consecuencias?|impactos?)\b", "effects and consequences"),
        (r"\b(no se quita|persiste|persistencia|cronico|crónico)\b", "persistence and why it may not resolve"),
        (r"\b(recurre|recurrencia|recaida|recaída)\b", "recurrence"),
        (r"\b(resistencia|tratamiento|terapia)\b", "treatment response and resistance"),
        (r"\b(compara|comparacion|comparación|versus| vs )\b", "comparison"),
    )
    for pattern, description in rules:
        if re.search(pattern, folded):
            concepts.append(description)
    if depth is ResearchDepth.DEEP:
        if not any(item.startswith("context") for item in concepts):
            concepts.insert(0, "context and definition")
        concepts.append("evidence-based conclusion")
    if not concepts:
        concepts = [topic]
    requirements = [ResearchRequirement(requirement_id=f"R{index}", description=description) for index, description in enumerate(dict.fromkeys(concepts), 1)]
    return ResearchIntent(original_request=original, topic=topic, depth=depth, source_scope=scope, requested_format=requested_format, requirements=requirements)


def resolve_auto_scope(intent: ResearchIntent, *, web_needed: bool) -> SourceScope:
    if intent.source_scope is not SourceScope.AUTO:
        return intent.source_scope
    return SourceScope.WEB if web_needed or intent.depth is ResearchDepth.DEEP else SourceScope.AUTO


def public_web_topic(intent: ResearchIntent) -> str:
    """Remove local-source routing phrases before a mixed-scope Web request.

    File excerpts are never accepted as input here; this only transforms the
    original public request text.
    """

    topic = intent.topic
    topic = re.sub(
        r"\b(?:busca(?:r)?\s+)?(?:en\s+)?(?:mis\s+)?(?:archivos|documentos|notas|carpeta)(?:\s+local(?:es)?)?(?:\s+sobre)?\b",
        " ",
        topic,
        flags=re.I,
    )
    topic = re.sub(r"\b(?:solo|unicamente|únicamente)\s+(?:archivos|web)\b", " ", topic, flags=re.I)
    return " ".join(topic.split()).strip(" ,.;:-") or intent.topic
