from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from difflib import SequenceMatcher
from time import perf_counter
from typing import Any

from jarvis.config import Settings

from .base import ModelMessage, ModelRequest, ModelResponse, StreamChunk


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    request: ModelRequest | None
    reason: str | None
    remaining_total_tokens: int


@dataclass(frozen=True, slots=True)
class RollingContinuationState:
    """Bounded state derived exclusively from public request/answer text."""

    original_user_intent: str
    public_summary: str
    recent_tail: str
    structural_state: str
    visible_tokens_estimated: int
    completed_headings: tuple[str, ...]


class OutputCompletionPolicy:
    """Canonical adaptive budget and bounded textual continuation policy."""

    max_auto_continuations = 2
    extended_max_auto_continuations = 8
    extended_continuation_tokens = 2500
    extended_visible_soft_target = 16_000
    extended_visible_hard_ceiling = 24_000
    rolling_summary_max_tokens = 768
    rolling_recent_tail_tokens = 1000
    rolling_original_intent_max_tokens = 1024
    rolling_context_reserve_tokens = 1024
    minimum_continuation_tokens = 128

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._context_tokens = max(
            int(getattr(settings, "local_runtime_context_size", 8192)), 512
        )

    def prepare_request(self, request: ModelRequest) -> ModelRequest:
        metadata = dict(request.metadata)
        adaptive = request.max_tokens is None or bool(
            metadata.get("adaptive_output_budget")
        )
        request_class = self._request_class(request)
        if (
            request_class == "EXTENDED_LONG_FORM"
            and not metadata.get("reasoning_preference_explicit")
            and str(
                metadata.get("reasoning_preference_requested") or "auto"
            ).casefold()
            == "auto"
        ):
            metadata["reasoning_preference"] = "off"
            metadata["educational_auto_reasoning_reduced"] = True
        if not adaptive:
            metadata["disable_auto_continuation"] = True
            metadata["output_budget"] = {
                "owner": "caller",
                "request_class": "EXPLICIT_CALLER_CAP",
                "max_tokens": request.max_tokens,
            }
            return request.model_copy(update={"metadata": metadata})

        context_tokens = self._request_context_tokens(request)
        estimated_input = estimate_request_tokens(request)
        safety_margin = max(
            int(
                getattr(
                    self._settings,
                    "conversation_context_safety_margin_tokens",
                    256,
                )
            ),
            128,
        )
        followup_reserve = (
            1024
            if request_class == "AGENT_INITIAL"
            else self._extended_context_reserve(context_tokens)
            if request_class == "EXTENDED_LONG_FORM"
            else 0
        )
        available = max(
            context_tokens - estimated_input - safety_margin - followup_reserve,
            32,
        )
        target, minimum = self._target_for(request_class, context_tokens)
        ceiling = metadata.get("output_budget_ceiling")
        if isinstance(ceiling, int) and ceiling > 0:
            target = min(target, ceiling)
        max_safe_fraction = (
            0.55 if request_class == "EXTENDED_LONG_FORM" else 0.35
        )
        max_safe = max(int(context_tokens * max_safe_fraction), minimum)
        budget = min(target, max_safe, available)
        if available >= minimum:
            budget = max(budget, minimum)
        budget = max(int(budget), 32)
        metadata["output_budget"] = {
            "owner": "jarvis_adaptive",
            "request_class": request_class,
            "context_tokens": context_tokens,
            "estimated_input_tokens": estimated_input,
            "followup_reserve_tokens": followup_reserve,
            "max_tokens": budget,
        }
        return request.model_copy(
            update={"max_tokens": budget, "metadata": metadata}
        )

    def complete_response(
        self,
        request: ModelRequest,
        first_response: ModelResponse,
        *,
        infer_next: Callable[[ModelRequest], ModelResponse],
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> ModelResponse:
        response = first_response
        accumulated = response.content
        segment_reasons = [response.metadata.get("finish_reason")]
        segment_durations = [round(response.latency_ms / 1000, 3)]
        segment_tokens = [self._response_tokens(response, request)]
        segment_budgets = [int(request.max_tokens or 0)]
        segment_visible_tokens = [estimated_tokens_from_text(response.content)]
        segment_reasoning_chunks = [
            int(response.metadata.get("reasoning_chunks") or 0)
        ]
        rolling_context_tokens = [estimate_request_tokens(request)]
        overlap_removed = 0
        attempts = 0
        loop_detection_count = 0
        loop_correction_attempted = False
        corrective_next = False
        context_blocked = False
        continuation_failed = False
        stop_reason: str | None = None
        total_limit = self.total_response_limit(request)

        while self._should_continue(request, response, attempts):
            decision = self.continuation_request(
                request,
                accumulated,
                generated_tokens=self._logical_generated_tokens(
                    request,
                    accumulated,
                    segment_tokens,
                ),
                attempt=attempts + 1,
                total_limit=total_limit,
                corrective=corrective_next,
            )
            if decision.request is None:
                context_blocked = decision.reason == "context_exhausted"
                stop_reason = decision.reason
                self._publish(
                    event_sink,
                    "model.auto_continuation_blocked",
                    request,
                    attempt=attempts + 1,
                    reason=decision.reason,
                )
                break
            corrective_next = False
            attempts += 1
            self._publish(
                event_sink,
                "model.auto_continuation_started",
                request,
                attempt=attempts,
            )
            started_at = perf_counter()
            try:
                next_response = infer_next(decision.request)
            except Exception as exc:  # noqa: BLE001
                continuation_failed = True
                stop_reason = "continuation_failure"
                self._publish(
                    event_sink,
                    "model.auto_continuation_failed",
                    request,
                    attempt=attempts,
                    reason=type(exc).__name__,
                )
                break
            elapsed = perf_counter() - started_at
            if next_response.tool_calls:
                continuation_failed = True
                stop_reason = "unexpected_tool_calls"
                self._publish(
                    event_sink,
                    "model.auto_continuation_failed",
                    request,
                    attempt=attempts,
                    reason="unexpected_tool_calls",
                )
                break
            segment_reasons.append(next_response.metadata.get("finish_reason"))
            segment_durations.append(round(elapsed, 3))
            segment_tokens.append(
                self._response_tokens(next_response, decision.request)
            )
            segment_budgets.append(int(decision.request.max_tokens or 0))
            segment_reasoning_chunks.append(
                int(next_response.metadata.get("reasoning_chunks") or 0)
            )
            rolling_context_tokens.append(
                int(
                    decision.request.metadata.get("rolling_context_tokens")
                    or estimate_request_tokens(decision.request)
                )
            )
            repeated = (
                self._request_class(request) == "EXTENDED_LONG_FORM"
                and self.detect_repetition_loop(
                    accumulated,
                    next_response.content,
                )
            )
            if repeated:
                loop_detection_count += 1
                segment_visible_tokens.append(0)
                self._publish(
                    event_sink,
                    "model.auto_continuation_loop_detected",
                    request,
                    attempt=attempts,
                    corrective=not loop_correction_attempted,
                )
                response = next_response.model_copy(
                    update={
                        "content": accumulated,
                        "metadata": {
                            **next_response.metadata,
                            "finish_reason": "length",
                            "incomplete": True,
                        },
                    }
                )
                if loop_correction_attempted:
                    stop_reason = "repetition_loop"
                    break
                loop_correction_attempted = True
                corrective_next = True
                continue
            accumulated, removed = merge_continuation_text(
                accumulated, next_response.content
            )
            overlap_removed += removed
            segment_visible_tokens.append(
                estimated_tokens_from_text(next_response.content[removed:])
            )
            response = next_response.model_copy(update={"content": accumulated})
            self._publish(
                event_sink,
                "model.auto_continuation_completed",
                request,
                attempt=attempts,
                finish_reason=next_response.metadata.get("finish_reason"),
                deduplicated_overlap_chars=removed,
            )

        if (
            response.metadata.get("finish_reason") == "length"
            and attempts >= self._max_continuations_for(request)
            and stop_reason is None
        ):
            stop_reason = "max_continuations_reached"

        return self._with_completion_telemetry(
            response,
            content=accumulated,
            attempts=attempts,
            segment_reasons=segment_reasons,
            segment_durations=segment_durations,
            segment_tokens=segment_tokens,
            segment_budgets=segment_budgets,
            segment_visible_tokens=segment_visible_tokens,
            segment_reasoning_chunks=segment_reasoning_chunks,
            rolling_context_tokens=rolling_context_tokens,
            loop_detection_count=loop_detection_count,
            loop_correction_attempted=loop_correction_attempted,
            request_class=self._request_class(request),
            total_limit=total_limit,
            overlap_removed=overlap_removed,
            context_blocked=context_blocked,
            continuation_failed=continuation_failed,
            stop_reason=stop_reason,
        )

    def stream_completed(
        self,
        request: ModelRequest,
        *,
        stream_once: Callable[[ModelRequest], Iterator[StreamChunk]],
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> Iterator[StreamChunk]:
        current_request = request
        accumulated = ""
        attempts = 0
        segment_reasons: list[str | None] = []
        segment_durations: list[float] = []
        segment_tokens: list[int] = []
        segment_budgets: list[int] = []
        segment_visible_tokens: list[int] = []
        segment_reasoning_chunks: list[int] = []
        rolling_context_tokens: list[int] = []
        overlap_removed = 0
        loop_detection_count = 0
        loop_correction_attempted = False
        corrective_next = False
        context_blocked = False
        continuation_failed = False
        stop_reason: str | None = None
        total_limit = self.total_response_limit(request)
        previous_terminal: StreamChunk | None = None

        while True:
            started_at = perf_counter()
            segment_text = ""
            terminal: StreamChunk | None = None
            buffered = attempts > 0
            try:
                for chunk in stream_once(current_request):
                    if chunk.done:
                        terminal = chunk
                        continue
                    if chunk.text:
                        segment_text += chunk.text
                        if not buffered:
                            yield chunk
                        continue
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                if not buffered or previous_terminal is None:
                    raise
                continuation_failed = True
                stop_reason = "continuation_failure"
                terminal = previous_terminal
                self._publish(
                    event_sink,
                    "model.auto_continuation_failed",
                    request,
                    attempt=attempts,
                    reason=type(exc).__name__,
                )
                break
            elapsed = perf_counter() - started_at
            if terminal is None:
                terminal = StreamChunk(
                    done=True,
                    metadata={
                        "finish_reason": None,
                        "stream_closed_normally": False,
                        "incomplete": True,
                    },
                )

            repeated = bool(
                buffered
                and self._request_class(request) == "EXTENDED_LONG_FORM"
                and self.detect_repetition_loop(accumulated, segment_text)
            )
            removed = 0
            novel = segment_text
            if repeated:
                loop_detection_count += 1
                novel = ""
                terminal = terminal.model_copy(
                    update={
                        "metadata": {
                            **terminal.metadata,
                            "finish_reason": "length",
                            "incomplete": True,
                        }
                    }
                )
                self._publish(
                    event_sink,
                    "model.auto_continuation_loop_detected",
                    request,
                    attempt=attempts,
                    corrective=not loop_correction_attempted,
                )
            elif buffered:
                merged, removed = merge_continuation_text(
                    accumulated, segment_text
                )
                novel = merged[len(accumulated) :]
                accumulated = merged
                overlap_removed += removed
                if novel:
                    yield StreamChunk(
                        text=novel,
                        metadata={
                            "provider": terminal.metadata.get("provider")
                            or terminal.metadata.get("active_provider"),
                            "model": terminal.metadata.get("model")
                            or terminal.metadata.get("active_model"),
                            "auto_continuation_segment": attempts,
                        },
                    )
            else:
                accumulated = segment_text

            finish_reason = terminal.metadata.get("finish_reason")
            segment_reasons.append(finish_reason)
            segment_durations.append(round(elapsed, 3))
            segment_tokens.append(
                self._chunk_tokens(terminal, segment_text, current_request)
            )
            segment_budgets.append(int(current_request.max_tokens or 0))
            segment_visible_tokens.append(estimated_tokens_from_text(novel))
            segment_reasoning_chunks.append(
                int(terminal.metadata.get("reasoning_chunks") or 0)
            )
            rolling_context_tokens.append(
                int(
                    current_request.metadata.get("rolling_context_tokens")
                    or estimate_request_tokens(current_request)
                )
            )
            if buffered and not repeated:
                self._publish(
                    event_sink,
                    "model.auto_continuation_completed",
                    request,
                    attempt=attempts,
                    finish_reason=finish_reason,
                    deduplicated_overlap_chars=removed,
                )
            response = ModelResponse(
                provider_name=str(
                    terminal.metadata.get("provider")
                    or terminal.metadata.get("active_provider")
                    or "model_provider"
                ),
                provider_kind=self._provider_kind(terminal),
                logical_model=request.logical_model or "general_assistant",
                model_name=str(
                    terminal.metadata.get("model")
                    or terminal.metadata.get("active_model")
                    or request.logical_model
                    or "model"
                ),
                content=accumulated,
                tool_calls=terminal.tool_calls,
                latency_ms=elapsed * 1000,
                metadata=dict(terminal.metadata),
            )
            if repeated:
                if loop_correction_attempted:
                    stop_reason = "repetition_loop"
                    break
                loop_correction_attempted = True
                corrective_next = True
            if not self._should_continue(request, response, attempts):
                break
            decision = self.continuation_request(
                request,
                accumulated,
                generated_tokens=self._logical_generated_tokens(
                    request,
                    accumulated,
                    segment_tokens,
                ),
                attempt=attempts + 1,
                total_limit=total_limit,
                corrective=corrective_next,
            )
            if decision.request is None:
                context_blocked = decision.reason == "context_exhausted"
                stop_reason = decision.reason
                self._publish(
                    event_sink,
                    "model.auto_continuation_blocked",
                    request,
                    attempt=attempts + 1,
                    reason=decision.reason,
                )
                break
            corrective_next = False
            attempts += 1
            self._publish(
                event_sink,
                "model.auto_continuation_started",
                request,
                attempt=attempts,
            )
            current_request = decision.request
            previous_terminal = terminal

        if (
            terminal.metadata.get("finish_reason") == "length"
            and attempts >= self._max_continuations_for(request)
            and stop_reason is None
        ):
            stop_reason = "max_continuations_reached"

        metadata = dict(terminal.metadata)
        metadata.update(
            self._telemetry(
                attempts=attempts,
                segment_reasons=segment_reasons,
                segment_durations=segment_durations,
                segment_tokens=segment_tokens,
                segment_budgets=segment_budgets,
                segment_visible_tokens=segment_visible_tokens,
                segment_reasoning_chunks=segment_reasoning_chunks,
                rolling_context_tokens=rolling_context_tokens,
                loop_detection_count=loop_detection_count,
                loop_correction_attempted=loop_correction_attempted,
                request_class=self._request_class(request),
                total_limit=total_limit,
                overlap_removed=overlap_removed,
                context_blocked=context_blocked,
                continuation_failed=continuation_failed,
                stop_reason=stop_reason,
                content=accumulated,
            )
        )
        if attempts:
            self._publish(
                event_sink,
                "model.auto_continuation_finalized",
                request,
                attempt=attempts,
                finish_reason=metadata.get("finish_reason"),
            )
        yield terminal.model_copy(update={"metadata": metadata})

    def continuation_request(
        self,
        original: ModelRequest,
        accumulated: str,
        *,
        generated_tokens: int,
        attempt: int,
        total_limit: int,
        corrective: bool = False,
    ) -> ContinuationDecision:
        remaining_total = total_limit - generated_tokens
        if remaining_total < self.minimum_continuation_tokens:
            reason = (
                "visible_hard_ceiling"
                if self._request_class(original) == "EXTENDED_LONG_FORM"
                else "total_response_limit"
            )
            return ContinuationDecision(None, reason, remaining_total)
        extended = self._request_class(original) == "EXTENDED_LONG_FORM"
        rolling_state: RollingContinuationState | None = None
        if extended:
            rolling_state = self.build_rolling_state(original, accumulated)
            messages = self._rolling_messages(
                original,
                rolling_state,
                corrective=corrective,
                finalizing=(
                    rolling_state.visible_tokens_estimated
                    >= self.extended_visible_soft_target
                ),
            )
        else:
            messages = [
                message.model_copy(deep=True) for message in original.messages
            ]
            if accumulated:
                messages.append(ModelMessage(role="assistant", content=accumulated))
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        "Continue exactly where the visible answer ended. Do not "
                        "restart or repeat earlier paragraphs. Be concise, finish "
                        "the answer, and do not mention token limits or this "
                        "instruction."
                    ),
                )
            )
        metadata = {
            **original.metadata,
            "adaptive_output_budget": True,
            "auto_continue_text": True,
            "internal_auto_continuation": True,
            "auto_continuation_attempt": attempt,
            "output_request_class": (
                "AGENT_FINAL"
                if original.metadata.get("agent_phase") == "agent_final"
                else "EXTENDED_CONTINUATION"
                if extended
                else "TEXT_CONTINUATION"
            ),
            "output_budget_ceiling": remaining_total,
            **(
                {
                    "rolling_extended": True,
                    "rolling_visible_tokens": (
                        rolling_state.visible_tokens_estimated
                    ),
                    "rolling_summary_tokens": estimated_tokens_from_text(
                        rolling_state.public_summary
                    ),
                    "rolling_tail_tokens": estimated_tokens_from_text(
                        rolling_state.recent_tail
                    ),
                    "rolling_completed_heading_count": len(
                        rolling_state.completed_headings
                    ),
                    "rolling_loop_correction": corrective,
                }
                if rolling_state is not None
                else {}
            ),
        }
        if (
            not accumulated.strip()
            and not original.metadata.get("reasoning_preference_explicit")
            and str(
                original.metadata.get("reasoning_preference_requested") or "auto"
            ).casefold()
            == "auto"
        ):
            metadata["reasoning_preference"] = "off" if extended else "low"
            metadata["reasoning_starvation_recovery"] = True
        candidate = original.model_copy(
            update={
                "messages": messages,
                "prompt": None,
                "tools": [],
                "tool_choice": None,
                "max_tokens": None,
                "metadata": metadata,
            }
        )
        prepared = self.prepare_request(candidate)
        context_tokens = self._request_context_tokens(prepared)
        context_reserve = (
            self.rolling_context_reserve_tokens if extended else 128
        )
        available = (
            context_tokens - estimate_request_tokens(prepared) - context_reserve
        )
        if available < self.minimum_continuation_tokens:
            return ContinuationDecision(None, "context_exhausted", remaining_total)
        budget = min(
            int(prepared.max_tokens or 0), remaining_total, max(available, 0)
        )
        if budget < self.minimum_continuation_tokens:
            return ContinuationDecision(None, "context_exhausted", remaining_total)
        prepared_metadata = dict(prepared.metadata)
        prepared_metadata["output_budget_ceiling"] = budget
        prepared_metadata["continuation_remaining_context_tokens"] = available
        prepared_metadata["continuation_total_budget_remaining"] = remaining_total
        prepared_metadata["rolling_context_tokens"] = estimate_request_tokens(
            prepared
        )
        prepared = prepared.model_copy(
            update={"max_tokens": budget, "metadata": prepared_metadata}
        )
        return ContinuationDecision(prepared, None, remaining_total)

    def build_rolling_state(
        self,
        original: ModelRequest,
        public_text: str,
    ) -> RollingContinuationState:
        """Build compact continuation state from public text only."""

        headings = tuple(
            match.group(1).strip()
            for match in re.finditer(
                r"(?m)^#{1,6}\s+(.+?)\s*$",
                public_text,
            )
        )
        completed = headings[:-1] if headings else ()
        excerpts: list[str] = []
        for paragraph in re.split(r"\n\s*\n", public_text):
            cleaned = " ".join(paragraph.split())
            if not cleaned or cleaned.startswith("#"):
                continue
            if len(cleaned) < 40:
                continue
            sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
            excerpts.append(sentence[:280])
            if len(excerpts) >= 8:
                break
        summary_parts: list[str] = []
        if completed:
            summary_parts.append(
                "Completed public sections: " + " | ".join(completed[-12:])
            )
        if headings:
            summary_parts.append("Current public section: " + headings[-1])
        if excerpts:
            summary_parts.append(
                "Public points already stated: " + " | ".join(excerpts)
            )
        public_summary = self._bounded_text(
            "\n".join(summary_parts) or "No completed public section yet.",
            self.rolling_summary_max_tokens,
        )
        recent_tail = self._recent_public_tail(
            public_text,
            self.rolling_recent_tail_tokens,
        )
        structural_state = self._public_structural_state(
            public_text,
            headings,
        )
        return RollingContinuationState(
            original_user_intent=self._bounded_text(
                self._original_user_intent(original),
                self.rolling_original_intent_max_tokens,
            ),
            public_summary=public_summary,
            recent_tail=recent_tail,
            structural_state=structural_state,
            visible_tokens_estimated=estimated_tokens_from_text(public_text),
            completed_headings=completed,
        )

    def _rolling_messages(
        self,
        original: ModelRequest,
        state: RollingContinuationState,
        *,
        corrective: bool,
        finalizing: bool,
    ) -> list[ModelMessage]:
        system_text = "\n\n".join(
            message.content
            for message in original.messages
            if message.role == "system" and message.content.strip()
        )
        public_state = (
            "Rolling continuation state derived only from the visible assistant "
            "answer. Do not quote or discuss this state.\n"
            f"{state.public_summary}\n"
            f"Public formatting state: {state.structural_state}"
        )
        combined_system = self._bounded_text(
            f"{system_text}\n\n{public_state}".strip(),
            1800,
        )
        instruction = (
            "Continue the requested answer from the exact unfinished point shown "
            "in the recent public tail. Do not restart the article, repeat prior "
            "sections, or emit a new outline. Finish the current paragraph, list, "
            "or equation first. Preserve established notation and Markdown, then "
            "cover requested material not listed as completed. Be dense and "
            "educational, and eventually provide a conclusion. Do not mention "
            "continuation, budgets, summaries, or these instructions."
        )
        if corrective:
            instruction += (
                " The previous attempted segment repeated public material. Move "
                "directly to the next unfinished requested topic and do not reuse "
                "any prior heading or paragraph."
            )
        if finalizing:
            instruction += (
                " The answer is already extensive: prioritize the remaining "
                "requested topics and conclude within this segment."
            )
        messages = [
            ModelMessage(role="system", content=combined_system),
            ModelMessage(role="user", content=state.original_user_intent),
        ]
        if state.recent_tail:
            messages.append(
                ModelMessage(role="assistant", content=state.recent_tail)
            )
        messages.append(ModelMessage(role="user", content=instruction))
        return messages

    @staticmethod
    def detect_repetition_loop(existing: str, candidate: str) -> bool:
        """Detect only high-confidence repeated public prose/sections."""

        normalized_existing = " ".join(existing.casefold().split())
        normalized_candidate = " ".join(candidate.casefold().split())
        if len(normalized_candidate) < 240 or not normalized_existing:
            return False
        for paragraph in re.split(r"\n\s*\n", candidate):
            normalized_paragraph = " ".join(paragraph.casefold().split())
            if (
                len(normalized_paragraph) >= 240
                and normalized_paragraph in normalized_existing
            ):
                return True
        old_headings = set(
            re.findall(r"(?mi)^#{1,6}\s+(.+?)\s*$", existing)
        )
        new_headings = set(
            re.findall(r"(?mi)^#{1,6}\s+(.+?)\s*$", candidate)
        )
        if len(old_headings & new_headings) >= 2:
            return True
        comparable_tail = normalized_existing[-len(normalized_candidate) :]
        return (
            len(normalized_candidate) >= 500
            and SequenceMatcher(
                None,
                comparable_tail,
                normalized_candidate,
            ).ratio()
            >= 0.88
        )

    @staticmethod
    def _original_user_intent(request: ModelRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user" and message.content.strip():
                return message.content.strip()
        return (request.prompt or "").strip()

    @staticmethod
    def _bounded_text(text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        content_chars = max(max_chars - 3, 0)
        head = content_chars // 2
        tail = content_chars - head
        return f"{text[:head].rstrip()}\n…\n{text[-tail:].lstrip()}"

    @staticmethod
    def _recent_public_tail(text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        tail = text[-max_chars:]
        boundary = tail.find("\n\n")
        if 0 <= boundary <= 500:
            tail = tail[boundary + 2 :]
        return tail

    @staticmethod
    def _public_structural_state(
        text: str,
        headings: tuple[str, ...],
    ) -> str:
        states = [
            f"current_heading={headings[-1] if headings else 'none'}",
            f"open_code_fence={text.count('```') % 2 == 1}",
            f"open_display_math={text.count('$$') % 2 == 1}",
            f"open_bracket_math={text.count(r'\[') > text.count(r'\]')}",
        ]
        list_matches = re.findall(r"(?m)^\s*((?:[-*+] |\d+[.)] )).+$", text)
        states.append(
            f"last_list_marker={list_matches[-1].strip() if list_matches else 'none'}"
        )
        return "; ".join(states)

    def total_response_limit(self, request: ModelRequest) -> int:
        context_tokens = self._request_context_tokens(request)
        request_class = self._request_class(request)
        initial = int(request.max_tokens or 0)
        if request_class == "EXTENDED_LONG_FORM":
            return self.extended_visible_hard_ceiling
        if request_class == "SHORT_CHAT":
            return min(max(initial * 2, 512), int(context_tokens * 0.2))
        if request_class in {"LONG_FORM_CHAT", "AGENT_FINAL"}:
            return min(max(initial * 2, 3072), int(context_tokens * 0.45))
        return min(max(initial * 3, 1536), int(context_tokens * 0.35))

    def _should_continue(
        self,
        request: ModelRequest,
        response: ModelResponse,
        attempts: int,
    ) -> bool:
        return bool(
            request.metadata.get("auto_continue_text")
            and not request.metadata.get("disable_auto_continuation")
            and response.metadata.get("stream_closed_normally") is True
            and response.metadata.get("finish_reason") == "length"
            and not response.tool_calls
            and request.metadata.get("agent_phase")
            not in {"structured_tool_call", "agent_initial"}
            and attempts < self._max_continuations_for(request)
        )

    def _request_class(self, request: ModelRequest) -> str:
        explicit = str(request.metadata.get("output_request_class") or "").upper()
        if explicit:
            return explicit
        phase = str(request.metadata.get("agent_phase") or "").casefold()
        if phase == "agent_final":
            return "AGENT_FINAL"
        if phase == "structured_tool_call":
            return "STRUCTURED_TOOL_CALL"
        if request.task_type == "agent" or request.tools:
            return "AGENT_INITIAL"
        profile = str(request.metadata.get("generation_profile") or "").casefold()
        if profile == "fast":
            return "SHORT_CHAT"
        if profile == "detailed":
            return "LONG_FORM_CHAT"
        if profile == "extended":
            return "EXTENDED_LONG_FORM"
        return "NORMAL_CHAT"

    def _target_for(self, request_class: str, context_tokens: int) -> tuple[int, int]:
        default = int(getattr(self._settings, "chat_default_max_tokens", 500))
        detailed = int(getattr(self._settings, "chat_detailed_max_tokens", 1200))
        fast = int(getattr(self._settings, "chat_fast_max_tokens", 160))
        if request_class == "SHORT_CHAT":
            return fast, min(fast, 96)
        if request_class == "LONG_FORM_CHAT":
            return max(detailed, min(2048, context_tokens // 4)), 768
        if request_class == "EXTENDED_LONG_FORM":
            return max(detailed, min(4096, context_tokens // 2)), 2048
        if request_class == "AGENT_INITIAL":
            return max(768, min(1024, context_tokens // 8)), 768
        if request_class == "STRUCTURED_TOOL_CALL":
            return max(768, min(1024, context_tokens // 8)), 512
        if request_class == "AGENT_FINAL":
            return max(detailed, min(1536, context_tokens // 5)), 768
        if request_class == "TEXT_CONTINUATION":
            return min(1024, context_tokens // 8), 256
        if request_class == "EXTENDED_CONTINUATION":
            return min(
                self.extended_continuation_tokens,
                int(context_tokens * 0.4),
            ), 768
        return max(default, min(768, context_tokens // 10)), 384

    def _request_context_tokens(self, request: ModelRequest) -> int:
        value = request.metadata.get("output_context_tokens")
        if isinstance(value, int) and value > 0:
            return value
        return self._context_tokens

    @staticmethod
    def _response_tokens(
        response: ModelResponse,
        request: ModelRequest,
    ) -> int:
        reported = response.metadata.get("completion_tokens")
        if reported is None:
            reported = response.usage.output_tokens
        if reported:
            return int(reported)
        if response.metadata.get("finish_reason") == "length":
            return int(
                response.metadata.get("requested_max_tokens")
                or request.max_tokens
                or estimated_tokens_from_text(response.content)
            )
        return estimated_tokens_from_text(response.content)

    @staticmethod
    def _chunk_tokens(
        chunk: StreamChunk,
        text: str,
        request: ModelRequest,
    ) -> int:
        reported = chunk.metadata.get("completion_tokens")
        if reported:
            return int(reported)
        if chunk.metadata.get("finish_reason") == "length":
            return int(
                chunk.metadata.get("requested_max_tokens")
                or request.max_tokens
                or estimated_tokens_from_text(text)
            )
        return estimated_tokens_from_text(text)

    def _max_continuations_for(self, request: ModelRequest) -> int:
        if self._request_class(request) == "EXTENDED_LONG_FORM":
            return self.extended_max_auto_continuations
        return self.max_auto_continuations

    def _logical_generated_tokens(
        self,
        request: ModelRequest,
        accumulated: str,
        segment_tokens: list[int],
    ) -> int:
        if self._request_class(request) == "EXTENDED_LONG_FORM":
            return estimated_tokens_from_text(accumulated)
        return sum(segment_tokens)

    @staticmethod
    def _extended_context_reserve(context_tokens: int) -> int:
        return max(1536, int(context_tokens * 0.18))

    @staticmethod
    def _provider_kind(chunk: StreamChunk):
        from .base import ProviderKind

        provider = str(
            chunk.metadata.get("provider")
            or chunk.metadata.get("active_provider")
            or ""
        )
        return ProviderKind.REMOTE if provider == "nvidia_nim" else ProviderKind.LOCAL

    def _with_completion_telemetry(
        self,
        response: ModelResponse,
        **values: Any,
    ) -> ModelResponse:
        metadata = dict(response.metadata)
        metadata.update(self._telemetry(**values))
        usage = response.usage.model_copy(
            update={
                "output_tokens": sum(values["segment_tokens"]),
                "total_tokens": (
                    (response.usage.input_tokens or 0)
                    + sum(values["segment_tokens"])
                ),
            }
        )
        return response.model_copy(
            update={
                "metadata": metadata,
                "usage": usage,
                "latency_ms": sum(values["segment_durations"]) * 1000,
            }
        )

    def _telemetry(
        self,
        *,
        attempts: int,
        segment_reasons: list[str | None],
        segment_durations: list[float],
        segment_tokens: list[int],
        segment_budgets: list[int],
        segment_visible_tokens: list[int],
        segment_reasoning_chunks: list[int],
        rolling_context_tokens: list[int],
        loop_detection_count: int,
        loop_correction_attempted: bool,
        request_class: str,
        total_limit: int,
        overlap_removed: int,
        context_blocked: bool,
        continuation_failed: bool,
        stop_reason: str | None,
        content: str,
    ) -> dict[str, Any]:
        visible_tokens = estimated_tokens_from_text(content)
        extended = request_class == "EXTENDED_LONG_FORM"
        return {
            "auto_continuation_count": attempts,
            "model_request_count": len(segment_reasons),
            "segment_finish_reasons": segment_reasons,
            "segment_durations_seconds": segment_durations,
            "segment_completion_tokens": segment_tokens,
            "segment_visible_tokens_estimated": segment_visible_tokens,
            "segment_reasoning_chunks": segment_reasoning_chunks,
            "segment_max_tokens": segment_budgets,
            "output_request_class": request_class,
            "total_completion_tokens": sum(segment_tokens),
            "provider_completion_tokens": sum(segment_tokens),
            "visible_response_tokens_estimated": visible_tokens,
            "total_response_token_limit": total_limit,
            "visible_response_soft_target": (
                self.extended_visible_soft_target if extended else None
            ),
            "visible_response_hard_ceiling": (
                self.extended_visible_hard_ceiling if extended else None
            ),
            "logical_response_remaining": max(total_limit - visible_tokens, 0),
            "rolling_context_tokens": rolling_context_tokens,
            "rolling_continuation_count": attempts if extended else 0,
            "loop_detection_count": loop_detection_count,
            "loop_correction_attempted": loop_correction_attempted,
            "deduplicated_overlap_chars": overlap_removed,
            "context_prevented_continuation": context_blocked,
            "auto_continuation_failed": continuation_failed,
            "continuation_stop_reason": stop_reason,
            "provider_text_length": len(content),
            "original_finish_reason": segment_reasons[0] if segment_reasons else None,
        }

    @staticmethod
    def _publish(
        sink: Callable[[str, dict[str, Any]], None] | None,
        event_name: str,
        request: ModelRequest,
        **payload: Any,
    ) -> None:
        if sink is None:
            return
        sink(
            event_name,
            {
                "run_id": request.correlation_id,
                "correlation_id": request.correlation_id,
                **payload,
            },
        )


def estimate_request_tokens(request: ModelRequest) -> int:
    chars = len(request.prompt or "")
    for message in request.messages:
        chars += len(message.role) + len(message.content) + len(message.name or "")
        for call in message.tool_calls:
            chars += len(call.name) + len(json.dumps(call.arguments, ensure_ascii=False))
    if request.tools:
        tools = request.tools
        if request.task_type == "agent" and len(tools) > 8:
            # The production agent/provider contract sends at most eight
            # relevant schemas to compact-context local models. Estimate the
            # eight largest schemas so budgeting remains conservative without
            # charging for registry entries that never reach the provider.
            serialized = sorted(
                (json.dumps(tool, ensure_ascii=False) for tool in tools),
                key=len,
                reverse=True,
            )[:8]
            chars += sum(len(item) for item in serialized)
        else:
            chars += len(json.dumps(tools, ensure_ascii=False))
    return estimated_tokens_from_text("x" * chars)


def estimated_tokens_from_text(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def merge_continuation_text(existing: str, continuation: str) -> tuple[str, int]:
    """Append a segment while removing only an exact, confident overlap."""

    if not existing:
        return continuation, 0
    if not continuation:
        return existing, 0
    maximum = min(len(existing), len(continuation), 2048)
    overlap = 0
    for size in range(maximum, 7, -1):
        if existing[-size:] != continuation[:size]:
            continue
        before = existing[-size - 1] if len(existing) > size else ""
        after = continuation[size] if len(continuation) > size else ""
        if before and after and before.isalnum() and after.isalnum():
            continue
        overlap = size
        break
    return existing + continuation[overlap:], overlap
