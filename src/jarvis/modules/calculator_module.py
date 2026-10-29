from __future__ import annotations

from pydantic import BaseModel, Field

from jarvis.actions.models import ActionResult
from jarvis.actions.registry import ActionContext, ActionDefinition, ActionRegistry
from jarvis.calculator_runtime import evaluate_expression
from jarvis.core.capabilities import CapabilityDescriptor, CapabilityRegistry
from jarvis.core.modes import ExecutionMode


class CalculatorPayload(BaseModel):
    expression: str = Field(min_length=1, max_length=512)


def _evaluate(_context: ActionContext, payload: CalculatorPayload) -> ActionResult:
    value = evaluate_expression(payload.expression)
    return ActionResult(
        message="calculator evaluation completed",
        data={"expression": payload.expression, "value": value},
    )


def build_calculator_action() -> ActionDefinition:
    return ActionDefinition(
        name="calculator.evaluate",
        description=(
            "Evaluate a bounded arithmetic expression with precedence, decimals, "
            "percentages, powers, roots, and constants."
        ),
        payload_model=CalculatorPayload,
        handler=_evaluate,
        tags=("calculator", "mathematics"),
        idempotent=True,
    )


class CalculatorModule:
    name = "calculator"
    description = "Safe arithmetic calculator available only as an agent tool."

    def register_actions(self, registry: ActionRegistry) -> None:
        registry.register(build_calculator_action())

    def register_capabilities(self, registry: CapabilityRegistry) -> None:
        registry.register(
            CapabilityDescriptor(
                name="calculator.runtime",
                module_name=self.name,
                intent="calculator",
                description="Bounded arithmetic evaluation for model tool calls.",
                action_names=("calculator.evaluate",),
                keywords=("calcula", "porcentaje", "raiz", "raíz", "multiplica"),
                mode_policy=(
                    ExecutionMode.ASSIST,
                    ExecutionMode.RESEARCH,
                    ExecutionMode.OPERATOR,
                    ExecutionMode.AUTOMATION,
                ),
                task_type="analysis",
            )
        )

    def start(self) -> None:
        return

    def stop(self) -> None:
        return
