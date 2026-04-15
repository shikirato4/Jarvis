from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from jarvis.core.errors import ActionValidationError

from .base import ScienceResult, ScienceSimulationRequest
from .solver import EARTH_GRAVITY, G
from .visualization import create_plot

try:
    from scipy.integrate import solve_ivp
except Exception:  # noqa: BLE001
    solve_ivp = None


def run_simulation(request: ScienceSimulationRequest, *, output_dir) -> ScienceResult:
    simulation_type = (request.simulation_type or _infer_simulation_type(request.query or "")).casefold()
    handlers: dict[str, Callable[[ScienceSimulationRequest, object], ScienceResult]] = {
        "free_fall": _simulate_free_fall,
        "projectile_motion": _simulate_projectile_motion,
        "orbit": _simulate_orbit,
        "exponential_growth": _simulate_exponential_growth,
        "iterative_map": _simulate_iterative_map,
    }
    handler = handlers.get(simulation_type)
    if handler is None:
        raise ActionValidationError(
            f"science simulation '{simulation_type}' is not supported",
            details={"supported_simulations": sorted(handlers)},
        )
    return handler(request, output_dir)


def _simulate_free_fall(request: ScienceSimulationRequest, output_dir) -> ScienceResult:
    params = request.parameters
    duration = float(request.duration or params.get("duration", 10.0))
    dt = float(request.time_step or params.get("time_step", 0.1))
    height = float(params.get("initial_height", 100.0))
    velocity = float(params.get("initial_velocity", 0.0))
    gravity = float(params.get("gravity", EARTH_GRAVITY))
    time = np.arange(0.0, duration + dt, dt)
    position = height + velocity * time - 0.5 * gravity * np.square(time)
    impact_index = int(np.argmax(position <= 0.0)) if np.any(position <= 0.0) else len(position) - 1
    if np.any(position <= 0.0):
        time = time[: impact_index + 1]
        position = position[: impact_index + 1]
        position[-1] = 0.0
    velocity_series = velocity - gravity * time
    artifacts = _plot_if_requested(request, output_dir, "free_fall", time, position, "Tiempo (s)", "Altura (m)")
    return ScienceResult(
        kind="simulate",
        domain="simulation",
        operation="free_fall",
        explanation="Se integro la caída libre con cinemática clásica y gravedad constante.",
        formulas=["y(t)=y0+v0 t-(1/2) g t^2", "v(t)=v0-g t"],
        assumptions=["Sin rozamiento del aire.", "Gravedad uniforme durante todo el trayecto."],
        result={
            "duration_seconds": float(time[-1]),
            "impact_velocity_m_s": float(velocity_series[-1]),
            "max_speed_m_s": float(np.max(np.abs(velocity_series))),
            "final_height_m": float(position[-1]),
        },
        table=_sample_table({"time_s": time, "height_m": position, "velocity_m_s": velocity_series}),
        artifacts=artifacts,
    )


def _simulate_projectile_motion(request: ScienceSimulationRequest, output_dir) -> ScienceResult:
    params = request.parameters
    duration = float(request.duration or params.get("duration", 20.0))
    dt = float(request.time_step or params.get("time_step", 0.05))
    speed = float(params.get("initial_speed", 50.0))
    angle_deg = float(params.get("angle_degrees", 45.0))
    gravity = float(params.get("gravity", EARTH_GRAVITY))
    angle = math.radians(angle_deg)
    vx = speed * math.cos(angle)
    vy = speed * math.sin(angle)
    time = np.arange(0.0, duration + dt, dt)
    x = vx * time
    y = vy * time - 0.5 * gravity * np.square(time)
    landing_index = int(np.argmax(y <= 0.0)) if np.any(y <= 0.0) else len(y) - 1
    if np.any(y <= 0.0):
        x = x[: landing_index + 1]
        y = y[: landing_index + 1]
        time = time[: landing_index + 1]
        y[-1] = 0.0
    artifacts = _plot_if_requested(request, output_dir, "projectile_motion", x, y, "Distancia (m)", "Altura (m)")
    return ScienceResult(
        kind="simulate",
        domain="simulation",
        operation="projectile_motion",
        explanation="Se simulo el tiro parabólico con velocidad inicial y gravedad constante.",
        formulas=["x(t)=v0 cos(theta) t", "y(t)=v0 sin(theta) t-(1/2) g t^2"],
        assumptions=["Sin resistencia aerodinámica."],
        result={
            "flight_time_seconds": float(time[-1]),
            "range_m": float(x[-1]),
            "max_height_m": float(np.max(y)),
        },
        table=_sample_table({"time_s": time, "x_m": x, "y_m": y}),
        artifacts=artifacts,
    )


def _simulate_orbit(request: ScienceSimulationRequest, output_dir) -> ScienceResult:
    params = request.parameters
    duration = float(request.duration or params.get("duration", 5400.0))
    dt = float(request.time_step or params.get("time_step", 10.0))
    central_mass = float(params.get("central_mass", 5.972e24))
    initial_position = np.array(params.get("initial_position", [7_000_000.0, 0.0]), dtype=float)
    initial_velocity = np.array(params.get("initial_velocity", [0.0, 7_500.0]), dtype=float)

    def dynamics(_, state):
        x, y, vx, vy = state
        radius = math.hypot(x, y)
        factor = -G * central_mass / max(radius**3, 1.0)
        return [vx, vy, factor * x, factor * y]

    samples = max(2, min(request.max_points, int(duration / dt) + 1))
    evaluation_times = np.linspace(0.0, duration, samples)
    initial_state = np.concatenate([initial_position, initial_velocity])
    if solve_ivp is not None:
        solution = solve_ivp(dynamics, (0.0, duration), initial_state, t_eval=evaluation_times, rtol=1e-8, atol=1e-8)
        state = solution.y
        time = solution.t
    else:
        time, state = _integrate_rk4(dynamics, initial_state, duration=duration, dt=dt)
    x = np.array(state[0])
    y = np.array(state[1])
    vx = np.array(state[2])
    vy = np.array(state[3])
    radius = np.sqrt(x**2 + y**2)
    speed = np.sqrt(vx**2 + vy**2)
    artifacts = _plot_if_requested(request, output_dir, "orbit", x, y, "x (m)", "y (m)")
    return ScienceResult(
        kind="simulate",
        domain="simulation",
        operation="orbit",
        explanation="Se integraron las ecuaciones orbitales newtonianas en 2D.",
        formulas=["r'' = -GM r / |r|^3"],
        assumptions=[
            "Dos cuerpos con masa de la nave despreciable.",
            "Campo central newtoniano sin perturbaciones.",
            "Se uso scipy.solve_ivp si estaba disponible; en caso contrario RK4 propio.",
        ],
        result={
            "duration_seconds": float(time[-1]),
            "min_radius_m": float(np.min(radius)),
            "max_radius_m": float(np.max(radius)),
            "min_speed_m_s": float(np.min(speed)),
            "max_speed_m_s": float(np.max(speed)),
        },
        table=_sample_table({"time_s": time, "x_m": x, "y_m": y, "speed_m_s": speed}),
        artifacts=artifacts,
    )


def _simulate_exponential_growth(request: ScienceSimulationRequest, output_dir) -> ScienceResult:
    params = request.parameters
    duration = float(request.duration or params.get("duration", 10.0))
    dt = float(request.time_step or params.get("time_step", 0.2))
    initial_value = float(params.get("initial_value", 1.0))
    growth_rate = float(params.get("growth_rate", 0.2))
    time = np.arange(0.0, duration + dt, dt)
    values = initial_value * np.exp(growth_rate * time)
    artifacts = _plot_if_requested(request, output_dir, "exponential_growth", time, values, "Tiempo (s)", "Valor")
    return ScienceResult(
        kind="simulate",
        domain="simulation",
        operation="exponential_growth",
        explanation="Se evaluo un crecimiento exponencial continuo.",
        formulas=["N(t)=N0 e^(r t)"],
        assumptions=["La tasa de crecimiento se mantuvo constante."],
        result={
            "initial_value": initial_value,
            "growth_rate": growth_rate,
            "final_value": float(values[-1]),
        },
        table=_sample_table({"time_s": time, "value": values}),
        artifacts=artifacts,
    )


def _simulate_iterative_map(request: ScienceSimulationRequest, output_dir) -> ScienceResult:
    params = request.parameters
    rate = float(params.get("rate", 3.7))
    initial_value = float(params.get("initial_value", 0.25))
    iterations = int(params.get("iterations", 60))
    values = np.zeros(iterations + 1)
    values[0] = initial_value
    for index in range(iterations):
        values[index + 1] = rate * values[index] * (1.0 - values[index])
    steps = np.arange(0, iterations + 1, dtype=float)
    artifacts = _plot_if_requested(request, output_dir, "iterative_map", steps, values, "Iteracion", "x_n")
    return ScienceResult(
        kind="simulate",
        domain="simulation",
        operation="iterative_map",
        explanation="Se genero una trayectoria del mapa logístico.",
        formulas=["x_(n+1)=r x_n (1-x_n)"],
        assumptions=["Sistema discreto no lineal con parametro constante."],
        result={
            "rate": rate,
            "iterations": iterations,
            "final_value": float(values[-1]),
        },
        table=_sample_table({"iteration": steps, "value": values}),
        artifacts=artifacts,
    )


def _sample_table(columns: dict[str, np.ndarray], rows: int = 12) -> list[dict[str, float | int | str]]:
    length = len(next(iter(columns.values())))
    if length <= rows:
        indices = list(range(length))
    else:
        indices = np.linspace(0, length - 1, rows, dtype=int).tolist()
    table: list[dict[str, float | int | str]] = []
    for index in indices:
        row: dict[str, float | int | str] = {}
        for name, values in columns.items():
            row[name] = round(float(values[index]), 6)
        table.append(row)
    return table


def _plot_if_requested(request: ScienceSimulationRequest, output_dir, name: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str) -> list[str]:
    if not request.generate_plot:
        return []
    artifact = create_plot(output_dir=output_dir, name=name, x=x, y=y, xlabel=xlabel, ylabel=ylabel)
    return [artifact] if artifact else []


def _infer_simulation_type(query: str) -> str:
    lowered = query.casefold()
    if "caida" in lowered or "free fall" in lowered:
        return "free_fall"
    if "orbita" in lowered or "orbit" in lowered:
        return "orbit"
    if "exponencial" in lowered or "growth" in lowered:
        return "exponential_growth"
    if "iterativo" in lowered or "logistic" in lowered:
        return "iterative_map"
    return "projectile_motion"


def _integrate_rk4(dynamics, initial_state, *, duration: float, dt: float):
    steps = max(2, int(duration / dt) + 1)
    time = np.linspace(0.0, duration, steps)
    state = np.zeros((len(initial_state), steps))
    state[:, 0] = initial_state
    for index in range(steps - 1):
        current = state[:, index]
        step = time[index + 1] - time[index]
        k1 = np.array(dynamics(time[index], current))
        k2 = np.array(dynamics(time[index] + step / 2.0, current + step * k1 / 2.0))
        k3 = np.array(dynamics(time[index] + step / 2.0, current + step * k2 / 2.0))
        k4 = np.array(dynamics(time[index] + step, current + step * k3))
        state[:, index + 1] = current + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return time, state
