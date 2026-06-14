from __future__ import annotations

from jarvis.code_agent_runtime.search.models import SearchDocument


class SearchRanker:
    SOURCE_BOOSTS = {
        "learned_pattern": 5.0,
        "repo_snippet": 4.0,
        "readme_summary": 3.0,
        "repo_metadata": 2.5,
    }

    def score(self, document: SearchDocument, *, terms: list[str], skill_ids: list[str], base_score: float = 0.0) -> tuple[float, list[str]]:
        haystack = " ".join(
            [
                document.title,
                document.body,
                document.path,
                " ".join(document.tags),
                " ".join(document.skills),
                document.language,
                document.framework,
            ]
        ).casefold()
        score = base_score + self.SOURCE_BOOSTS.get(document.source_type, 1.0)
        reasons: list[str] = []
        matched = [term for term in terms if term in haystack][:8]
        if matched:
            score += sum(3.0 if len(term) >= 5 else 1.5 for term in matched)
            reasons.append(f"matched terms: {', '.join(matched)}")
        matched_skills = [skill for skill in skill_ids if skill in document.skills or skill in document.tags]
        if matched_skills:
            score += 7.0 + len(matched_skills)
            reasons.append(f"matched skills: {', '.join(matched_skills)}")
        if document.language and any(document.language.casefold() in term for term in terms):
            score += 2.0
            reasons.append(f"language: {document.language}")
        if document.framework and any(document.framework.casefold() in term for term in terms):
            score += 2.0
            reasons.append(f"framework: {document.framework}")
        if document.confidence:
            score += min(max(document.confidence, 0.0), 1.0) * 4.0
            reasons.append(f"confidence: {document.confidence:.2f}")
        if document.license and document.license.casefold() != "unknown":
            score += 1.5
            reasons.append(f"license: {document.license}")
        if len(document.body) <= 900:
            score += 1.0
            reasons.append("small snippet")
        return score, reasons or ["metadata match"]
