from __future__ import annotations

import re
import unicodedata

from .models import (
    ResearchDepth,
    ResearchIntent,
    ResearchRequirement,
    ResponseDepth,
    ResponseLengthRequest,
    SearchMode,
    SourceScope,
)


def _fold(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char))


def parse_response_length(text: str, *, deep: bool = False) -> ResponseLengthRequest:
    """Preserve the user's requested answer depth independently of search depth."""

    folded = _fold(" ".join((text or "").split()))
    word_match = re.search(
        r"\b(?:aproximadamente|aprox\.?|unas?|alrededor de|about|approximately)?\s*"
        r"([1-9]\d{0,3}(?:[.,]\d{3})?)\s*(?:palabras|words)\b",
        folded,
    )
    if word_match:
        requested = int(word_match.group(1).replace(".", "").replace(",", ""))
        requested = max(100, min(requested, 4_000))
        return ResponseLengthRequest(
            depth=(
                ResponseDepth.EXHAUSTIVE
                if requested >= 2_400
                else ResponseDepth.EXTENSIVE
                if requested >= 1_200
                else ResponseDepth.DETAILED
            ),
            target_words=requested,
            minimum_words=max(80, int(requested * 0.75)),
            maximum_words=int(requested * 1.25),
            explicit_word_count=True,
            matched_instruction=word_match.group(0),
        )
    if re.search(r"\b(brevemente|breve|conciso|en pocas palabras|briefly|short answer)\b", folded):
        return ResponseLengthRequest(
            depth=ResponseDepth.BRIEF,
            target_words=120,
            minimum_words=0,
            maximum_words=220,
            matched_instruction="brief",
        )
    if re.search(r"\b(exhaustiv|exhaustively|thorough(?:ly)?|muy a fondo)\w*\b", folded):
        return ResponseLengthRequest(
            depth=ResponseDepth.EXHAUSTIVE,
            target_words=2_400,
            minimum_words=1_600,
            maximum_words=3_200,
            matched_instruction="exhaustive",
        )
    if re.search(
        r"\b(extensamente|extenso|extensa|informe completo|reporte completo|"
        r"comprehensive|in depth|long[- ]form)\b",
        folded,
    ):
        return ResponseLengthRequest(
            depth=ResponseDepth.EXTENSIVE,
            target_words=1_600,
            minimum_words=1_000,
            maximum_words=2_300,
            matched_instruction="extensive",
        )
    if re.search(r"\b(detalladamente|detallado|detallada|a fondo|detailed)\b", folded):
        return ResponseLengthRequest(
            depth=ResponseDepth.DETAILED,
            target_words=900,
            minimum_words=550,
            maximum_words=1_400,
            matched_instruction="detailed",
        )
    # Answer depth stays independent of search depth: a /deepresearch run
    # without explicit length markers keeps the standard response flow.
    return ResponseLengthRequest()


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
    requested_format = (
        "apa_report"
        if wants_apa and wants_report
        else "apa"
        if wants_apa
        else "report"
        if wants_report or explicit_deep
        else "answer"
    )
    requested_length = parse_response_length(original, deep=explicit_deep)

    # Presentation/citation directives are deliberately removed before planning.
    # Substantive clauses remain and become requirements.
    topic = re.sub(
        r"\b(?:utilizando|usando|con)\s+(?:las\s+)?(?:fuentes?|referencias?)\s+"
        r"(?:que\s+)?(?:encuentres?|recuperadas?|disponibles?)\b",
        " ",
        text,
        flags=re.I,
    )
    topic = re.sub(r"\b(?:y\s+)?(?:al final(?: de todo)?\s+)?(?:dame|incluye|agrega|pon)?\s*(?:las?\s+)?(?:fuentes?|referencias?|bibliograf(?:ia|ía)|citas?)(?:\s+citadas?)?(?:\s+en\s+formato\s+apa)?\b", " ", topic, flags=re.I)
    topic = re.sub(r"\b(?:dame|haz|redacta|prepara)\s+(?:un\s+)?(?:informe|reporte)(?:\s+detallado)?\b", " ", topic, flags=re.I)
    topic = re.sub(
        r"\b(?:de\s+)?(?:aproximadamente|aprox\.?|unas?|alrededor\s+de)?\s*"
        r"[1-9]\d{0,3}(?:[.,]\d{3})?\s*(?:palabras|words)\s+(?:sobre\s+)?",
        " ",
        topic,
        flags=re.I,
    )
    topic = re.sub(r"\bformato\s+apa\b|\bapa\b", " ", topic, flags=re.I)
    topic = " ".join(topic.split()).strip(" ,.;:-") or text

    concepts: list[str] = []
    rules = (
        (r"\b(que es|qué es|contexto|definicion|definición)\b", "context and definition"),
        (r"\b(desarrolla|desarrollo|evoluciona|funciona|mecanismo|proceso)\b", "development or mechanism"),
        (r"\b(causas?|factores?|riesgos?|por que se da|por qué se da|origen)\b", "causes and risk factors"),
        (r"\b(efectos?|consecuencias?|impactos?)\b", "effects and consequences"),
        (
            r"\b(mitigacion|mitigación|descarboniz|reduc(?:ir|cion|ción)\s+emisiones|"
            r"estrategias?\s+climaticas?|climate mitigation)\b",
            "mitigation strategies",
        ),
        (r"\b(no se quita|persiste|persistencia|cronico|crónico)\b", "persistence and why it may not resolve"),
        (r"\b(recurre|recurrencia|recaida|recaída)\b", "recurrence"),
        (r"\b(resistencia|tratamiento|terapia)\b", "treatment response and resistance"),
        (r"\b(compara|comparacion|comparación|versus| vs )\b", "comparison"),
        (r"\b(aplicaciones?|usos?|applications?|use cases?)\b", "requested applications and use cases"),
        (r"\b(ejemplos?|examples?)\b", "examples supporting the requested explanation"),
        (r"\b(derechos humanos|violaciones? de derechos|human rights)\b", "documented human-rights violations and legal context"),
        (r"\b(maltrato|trato inhumano|condiciones de detencion|condiciones de detención|mistreatment|detention conditions)\b", "reported mistreatment and detention conditions"),
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
    return ResearchIntent(
        original_request=original,
        topic=topic,
        depth=depth,
        source_scope=scope,
        requested_format=requested_format,
        requested_length=requested_length,
        requirements=requirements,
    )


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
