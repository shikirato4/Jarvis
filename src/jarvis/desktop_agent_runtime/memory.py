from __future__ import annotations

from .models import DesktopAgentActionRecord, DesktopAgentObservation, DesktopWorldState


class DesktopAgentMemoryManager:
    def append_observation(self, world: DesktopWorldState, observation: DesktopAgentObservation) -> DesktopWorldState:
        world.recent_observations = [*world.recent_observations[-9:], observation]
        world.memory.last_observation_summary = observation.summary
        world.last_observation_summary = observation.summary
        return world

    def append_action(self, world: DesktopWorldState, action: DesktopAgentActionRecord) -> DesktopWorldState:
        world.recent_actions = [*world.recent_actions[-9:], action]
        if action.step_id not in world.memory.attempted_steps:
            world.memory.attempted_steps.append(action.step_id)
        return world

    def note_opened_application(self, world: DesktopWorldState, application: str) -> DesktopWorldState:
        folded = application.casefold()
        if all(item.casefold() != folded for item in world.memory.opened_applications):
            world.memory.opened_applications.append(application)
        world.memory.target_application = application
        world.target_application = application
        return world

    def note_fallback(self, world: DesktopWorldState, fallback: str) -> DesktopWorldState:
        world.memory.attempted_fallbacks.append(fallback)
        return world

    def note_recovery_attempt(self, world: DesktopWorldState, step_id: str, strategy: str) -> DesktopWorldState:
        attempts = world.memory.recovery_attempts_by_step.setdefault(step_id, [])
        if strategy not in attempts:
            attempts.append(strategy)
        world.last_recovery_strategy = strategy
        return self.note_fallback(world, f"{step_id}:{strategy}")

    def has_recovery_attempt(self, world: DesktopWorldState, step_id: str, strategy: str) -> bool:
        return strategy in world.memory.recovery_attempts_by_step.get(step_id, [])

    def note_expectation(self, world: DesktopWorldState, expected: dict) -> DesktopWorldState:
        world.memory.last_expected_observation = expected
        world.expected_next_state = expected
        return world

    def note_step_completed(self, world: DesktopWorldState, step_id: str, *, strategy: str | None = None) -> DesktopWorldState:
        if step_id not in world.completed_steps:
            world.completed_steps.append(step_id)
        if step_id not in world.memory.completed_steps:
            world.memory.completed_steps.append(step_id)
        world.memory.last_completed_step = step_id
        world.memory.mission_position = f"completed:{step_id}"
        if strategy and strategy not in world.memory.successful_strategies:
            world.memory.successful_strategies.append(strategy)
        return world

    def note_strategy(self, world: DesktopWorldState, strategy: str) -> DesktopWorldState:
        world.memory.last_strategy = strategy
        return world

    def note_target_window(self, world: DesktopWorldState, target_window: str | None) -> DesktopWorldState:
        world.target_window_title = target_window
        world.memory.target_window_title = target_window
        return world

    def note_error(self, world: DesktopWorldState, error: str) -> DesktopWorldState:
        world.memory.last_error = error
        world.last_error = error
        if error not in world.memory.failed_verifications:
            world.memory.failed_verifications.append(error)
        return world

    def note_mission_position(self, world: DesktopWorldState, position: str) -> DesktopWorldState:
        world.memory.mission_position = position
        return world
