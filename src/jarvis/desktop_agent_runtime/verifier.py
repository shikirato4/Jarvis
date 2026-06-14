from __future__ import annotations

from pathlib import Path

from .models import DesktopAgentStep, DesktopAgentVerificationResult, DesktopVerificationStatus, DesktopWorldState


class DesktopAgentVerifier:
    def verify(self, world: DesktopWorldState, step: DesktopAgentStep, action_result: dict) -> DesktopAgentVerificationResult:
        expectation = step.verification
        missing: list[str] = []
        matched: list[str] = []
        observed = {
            "active_window_title": world.active_window.title if world.active_window else None,
            "active_process_name": world.active_window.process_name if world.active_window else None,
            "visible_text": world.visible_text[:400],
            "selection_text": world.selection_text[:200],
            "clipboard_text": world.clipboard_text[:200],
            "context_signals": world.context_signals,
            "detected_targets": [target.label for target in world.detected_targets if target.label],
            "active_path": world.active_path,
            "action_result": action_result,
        }
        expected = expectation.model_dump(mode="json")

        if expectation.action_success_required and not action_result.get("success", action_result.get("ok", True)):
            missing.append("action_success")

        if expectation.search_results_min is not None:
            matches = action_result.get("matches") or world.last_result.get("matches") or []
            if len(matches) < expectation.search_results_min:
                missing.append(f"search_results_min:{expectation.search_results_min}")
            else:
                matched.append("search_results")

        active_title = (world.active_window.title if world.active_window else "").casefold()
        active_process = (world.active_window.process_name if world.active_window and world.active_window.process_name else "").casefold()
        if expectation.active_window_contains:
            token = expectation.active_window_contains.casefold()
            if token not in active_title:
                missing.append(f"active_window_contains:{expectation.active_window_contains}")
            else:
                matched.append(f"active_window_contains:{expectation.active_window_contains}")
        if expectation.process_name_contains:
            token = expectation.process_name_contains.casefold()
            if token not in active_process:
                missing.append(f"process_name_contains:{expectation.process_name_contains}")
            else:
                matched.append(f"process_name_contains:{expectation.process_name_contains}")

        visible_text = world.visible_text.casefold()
        selection_text = world.selection_text.casefold()
        clipboard_text = world.clipboard_text.casefold()
        target_labels = [(target.label or "").casefold() for target in world.detected_targets]
        action_label = str(action_result.get("matched_label") or "").casefold()
        signal_set = {signal.casefold() for signal in world.context_signals}

        for token in expectation.visible_text_contains:
            lowered = token.casefold()
            if lowered not in visible_text:
                missing.append(f"visible_text_contains:{token}")
            else:
                matched.append(f"visible_text_contains:{token}")
        for token in expectation.visible_text_not_contains:
            lowered = token.casefold()
            if lowered in visible_text:
                missing.append(f"visible_text_not_contains:{token}")
            else:
                matched.append(f"visible_text_not_contains:{token}")
        for token in expectation.selection_contains:
            lowered = token.casefold()
            if lowered not in selection_text:
                missing.append(f"selection_contains:{token}")
            else:
                matched.append(f"selection_contains:{token}")
        for token in expectation.clipboard_contains:
            lowered = token.casefold()
            if lowered not in clipboard_text:
                missing.append(f"clipboard_contains:{token}")
            else:
                matched.append(f"clipboard_contains:{token}")
        for token in expectation.expected_targets:
            lowered = token.casefold()
            if not any(lowered in label for label in target_labels) and lowered not in action_label:
                missing.append(f"expected_target:{token}")
            else:
                matched.append(f"expected_target:{token}")
        for signal in expectation.required_context_signals:
            lowered = signal.casefold()
            if lowered not in signal_set:
                missing.append(f"context_signal:{signal}")
            else:
                matched.append(f"context_signal:{signal}")
        for signal in expectation.forbidden_context_signals:
            lowered = signal.casefold()
            if lowered in signal_set:
                missing.append(f"forbidden_context_signal:{signal}")
            else:
                matched.append(f"forbidden_context_signal:{signal}")

        path_value = self._observed_path(world, action_result)
        if expectation.path_exists is not None:
            exists = bool(path_value and path_value.exists())
            if exists != expectation.path_exists:
                missing.append(f"path_exists:{expectation.path_exists}")
            else:
                matched.append(f"path_exists:{expectation.path_exists}")
        if expectation.file_exists is not None:
            is_file = bool(path_value and path_value.exists() and path_value.is_file())
            if is_file != expectation.file_exists:
                missing.append(f"file_exists:{expectation.file_exists}")
            else:
                matched.append(f"file_exists:{expectation.file_exists}")
        if expectation.folder_exists is not None:
            is_dir = bool(path_value and path_value.exists() and path_value.is_dir())
            if is_dir != expectation.folder_exists:
                missing.append(f"folder_exists:{expectation.folder_exists}")
            else:
                matched.append(f"folder_exists:{expectation.folder_exists}")
        if expectation.path_kind:
            expected_kind = expectation.path_kind.casefold()
            actual_kind = "folder" if path_value and path_value.exists() and path_value.is_dir() else "file" if path_value and path_value.exists() else None
            if actual_kind != expected_kind:
                missing.append(f"path_kind:{expectation.path_kind}")
            else:
                matched.append(f"path_kind:{expectation.path_kind}")
        if expectation.path_contains:
            lowered = expectation.path_contains.casefold()
            if path_value is None or lowered not in str(path_value).casefold():
                missing.append(f"path_contains:{expectation.path_contains}")
            else:
                matched.append(f"path_contains:{expectation.path_contains}")
        for key, value in expectation.action_data_contains.items():
            if action_result.get("data", {}).get(key) != value:
                missing.append(f"action_data_contains:{key}")
            else:
                matched.append(f"action_data_contains:{key}")

        if missing:
            status = DesktopVerificationStatus.PARTIAL if matched else DesktopVerificationStatus.FAILED
            note = f"Verification {status.value}: faltan {', '.join(missing[:4])}."
        else:
            status = DesktopVerificationStatus.PASSED
            note = "Verification passed."
        step.verification_status = status
        world.verification_status = status
        world.verify_count += 1
        return DesktopAgentVerificationResult(
            status=status,
            note=note,
            observed=observed,
            expected=expected,
            missing=missing,
            matched=matched,
        )

    @staticmethod
    def _observed_path(world: DesktopWorldState, action_result: dict) -> Path | None:
        resolved_target = action_result.get("resolved_target") or {}
        action_data = action_result.get("data") or {}
        candidates = [
            resolved_target.get("path"),
            action_data.get("destination_path"),
            action_data.get("created_path"),
            action_data.get("opened_path"),
            action_data.get("revealed_path"),
            world.active_path,
            world.target_path,
        ]
        for candidate in candidates:
            if candidate:
                return Path(str(candidate)).expanduser().resolve(strict=False)
        return None
