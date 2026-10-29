from __future__ import annotations

from dataclasses import dataclass, field

from jarvis.runtime_activity import RuntimeActivitySnapshot, RuntimeActivityState


@dataclass(frozen=True, slots=True)
class ActivityGraphSceneNode:
    component: str
    state: RuntimeActivityState
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ActivityGraphSceneEdge:
    source: str
    target: str


@dataclass(slots=True)
class ActivityGraphScene:
    nodes: list[ActivityGraphSceneNode] = field(default_factory=list)
    edges: list[ActivityGraphSceneEdge] = field(default_factory=list)


def build_activity_graph_scene(
    snapshot: RuntimeActivitySnapshot,
    *,
    width: int,
    height: int,
) -> ActivityGraphScene:
    if not snapshot.nodes:
        return ActivityGraphScene()
    width = max(width, 320)
    height = max(height, 240)
    priorities = {
        "neural_router": 0,
        "nemotron": 1,
        "agent_controller": 2,
        "action_executor": 3,
    }
    ordered = sorted(
        snapshot.nodes,
        key=lambda item: (priorities.get(item.component, 10), item.component),
    )
    core = [item for item in ordered if item.component in priorities]
    tools = [item for item in ordered if item.component not in priorities]
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(core):
        positions[node.component] = (
            width * (index + 1) / (len(core) + 1),
            height * 0.28,
        )
    for index, node in enumerate(tools):
        columns = min(max(len(tools), 1), 4)
        row = index // columns
        column = index % columns
        rows = (len(tools) + columns - 1) // columns
        positions[node.component] = (
            width * (column + 1) / (columns + 1),
            height * (0.62 + (0.26 * row / max(rows - 1, 1))),
        )
    scene_nodes = [
        ActivityGraphSceneNode(
            component=node.component,
            state=node.state,
            x=positions[node.component][0],
            y=positions[node.component][1],
        )
        for node in ordered
    ]
    present = set(positions)
    scene_edges = [
        ActivityGraphSceneEdge(source=edge.source, target=edge.target)
        for edge in snapshot.edges
        if edge.source in present and edge.target in present
    ]
    return ActivityGraphScene(nodes=scene_nodes, edges=scene_edges)


try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QPainter, QPen
    from PySide6.QtWidgets import QWidget
except Exception:  # noqa: BLE001
    QWidget = None


if QWidget is not None:

    class ActivityGraphWidget(QWidget):
        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._snapshot = RuntimeActivitySnapshot()
            self.setMinimumSize(420, 300)
            self.setAccessibleName("Actividad real del runtime de JARVIS")

        def set_snapshot(self, snapshot: RuntimeActivitySnapshot) -> None:
            self._snapshot = snapshot.model_copy(deep=True)
            self.update()

        def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API.
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#071015"))
            scene = build_activity_graph_scene(
                self._snapshot,
                width=self.width(),
                height=self.height(),
            )
            if not scene.nodes:
                painter.setPen(QColor("#6d8f91"))
                painter.drawText(
                    self.rect(),
                    Qt.AlignCenter,
                    "Sin actividad real del runtime",
                )
                return
            positions = {
                node.component: QPointF(node.x, node.y) for node in scene.nodes
            }
            painter.setPen(QPen(QColor("#236b70"), 2.0))
            for edge in scene.edges:
                painter.drawLine(positions[edge.source], positions[edge.target])
            colors = {
                RuntimeActivityState.IDLE: "#52656a",
                RuntimeActivityState.QUEUED: "#637d8a",
                RuntimeActivityState.ACTIVE: "#25d9c8",
                RuntimeActivityState.WAITING: "#e5b949",
                RuntimeActivityState.SUCCESS: "#58d68d",
                RuntimeActivityState.FAILED: "#e05b67",
                RuntimeActivityState.BLOCKED: "#d47a42",
                RuntimeActivityState.CANCELLED: "#8b7ea8",
            }
            painter.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
            for node in scene.nodes:
                center = positions[node.component]
                painter.setPen(QPen(QColor(colors[node.state]), 3.0))
                painter.setBrush(QColor("#0b1d22"))
                painter.drawEllipse(center, 25, 25)
                painter.setPen(QColor("#d8ffff"))
                label = {
                    "nemotron": "MODEL",
                    "model_provider": "MODEL",
                }.get(node.component, node.component.replace("_", " ").upper())
                painter.drawText(
                    QRectF(center.x() - 75, center.y() + 29, 150, 34),
                    Qt.AlignHCenter | Qt.AlignTop,
                    label,
                )

else:

    class ActivityGraphWidget:  # pragma: no cover - dependency fallback.
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required for ActivityGraphWidget")

