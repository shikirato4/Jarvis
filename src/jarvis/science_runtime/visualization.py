from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def create_plot(*, output_dir: Path, name: str, x, y, xlabel: str, ylabel: str) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_{uuid4().hex[:8]}.png"
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x, y, linewidth=2.0)
    axis.set_title(name.replace("_", " ").title())
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)
