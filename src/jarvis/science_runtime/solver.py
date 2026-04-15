from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import sympy as sp

from jarvis.core.errors import ActionValidationError

from .base import ScienceResult, ScienceSolveRequest

G = 6.67430e-11
C = 299_792_458.0
EARTH_GRAVITY = 9.80665


def solve_problem(request: ScienceSolveRequest) -> ScienceResult:
    operation = (request.operation or _infer_operation(request.query)).casefold()
    handlers: dict[str, Callable[[ScienceSolveRequest], ScienceResult]] = {
        "differentiate": _solve_derivative,
        "integrate": _solve_integral,
        "simplify": _solve_simplify,
        "equation_system": _solve_equation_system,
        "matrix": _solve_matrix,
        "probability": _solve_probability,
        "black_hole_escape": _solve_black_hole_escape,
        "time_dilation": _solve_time_dilation,
    }
    handler = handlers.get(operation)
    if handler is None:
        raise ActionValidationError(
            f"science operation '{operation}' is not supported",
            details={"supported_operations": sorted(handlers)},
        )
    return handler(request)


def _solve_derivative(request: ScienceSolveRequest) -> ScienceResult:
    expression_text = str(request.parameters.get("expression") or _extract_expression(request.query))
    variable_text = str(request.parameters.get("variable") or "x")
    order = int(request.parameters.get("order", 1))
    variable = sp.Symbol(variable_text)
    expression = sp.sympify(expression_text)
    derivative = sp.diff(expression, variable, order)
    return ScienceResult(
        kind="solve",
        domain="mathematics",
        operation="differentiate",
        explanation="Se calculo la derivada simbolica con SymPy.",
        formulas=[f"d^{order}/d{variable_text}^{order} ({sp.sstr(expression)})"],
        assumptions=["La expresion se interpreto como algebra simbólica exacta."],
        result={
            "expression": sp.sstr(expression),
            "variable": variable_text,
            "order": order,
            "derivative": sp.sstr(derivative),
            "latex": sp.latex(derivative),
        },
    )


def _solve_integral(request: ScienceSolveRequest) -> ScienceResult:
    expression_text = str(request.parameters.get("expression") or _extract_expression(request.query))
    variable_text = str(request.parameters.get("variable") or "x")
    lower = request.parameters.get("lower")
    upper = request.parameters.get("upper")
    variable = sp.Symbol(variable_text)
    expression = sp.sympify(expression_text)
    if lower is not None and upper is not None:
        integral = sp.integrate(expression, (variable, lower, upper))
        formulas = [f"Integral[{lower},{upper}] {sp.sstr(expression)} d{variable_text}"]
        assumptions = ["Se resolvio una integral definida exacta cuando fue posible."]
        result = {
            "expression": sp.sstr(expression),
            "variable": variable_text,
            "lower": float(lower),
            "upper": float(upper),
            "integral": sp.sstr(sp.simplify(integral)),
            "numeric": float(sp.N(integral)),
            "latex": sp.latex(integral),
        }
    else:
        integral = sp.integrate(expression, variable)
        formulas = [f"Integral {sp.sstr(expression)} d{variable_text}"]
        assumptions = ["Se devolvio una antiderivada simbólica."]
        result = {
            "expression": sp.sstr(expression),
            "variable": variable_text,
            "integral": sp.sstr(integral),
            "latex": sp.latex(integral),
        }
    return ScienceResult(
        kind="solve",
        domain="mathematics",
        operation="integrate",
        explanation="Se calculo la integral con SymPy.",
        formulas=formulas,
        assumptions=assumptions,
        result=result,
    )


def _solve_simplify(request: ScienceSolveRequest) -> ScienceResult:
    expression_text = str(request.parameters.get("expression") or _extract_expression(request.query))
    expression = sp.sympify(expression_text)
    simplified = sp.simplify(expression)
    return ScienceResult(
        kind="solve",
        domain="mathematics",
        operation="simplify",
        explanation="Se simplifico la expresion simbólica.",
        formulas=[sp.sstr(expression)],
        assumptions=["La simplificación siguió las reglas algebraicas estándar de SymPy."],
        result={
            "expression": sp.sstr(expression),
            "simplified": sp.sstr(simplified),
            "latex": sp.latex(simplified),
        },
    )


def _solve_equation_system(request: ScienceSolveRequest) -> ScienceResult:
    equations_raw = request.parameters.get("equations")
    variables_raw = request.parameters.get("variables")
    if not equations_raw or not variables_raw:
        raise ActionValidationError("equation_system requires 'equations' and 'variables'")
    variables = [sp.Symbol(str(item)) for item in variables_raw]
    equations = [_parse_equation(str(item)) for item in equations_raw]
    solution = sp.solve(equations, variables, dict=True)
    return ScienceResult(
        kind="solve",
        domain="mathematics",
        operation="equation_system",
        explanation="Se resolvio el sistema de ecuaciones de forma simbólica.",
        formulas=[sp.sstr(item) for item in equations],
        assumptions=["El sistema se resolvio con álgebra simbólica exacta cuando fue posible."],
        result={
            "variables": [str(item) for item in variables],
            "solutions": [{str(key): sp.sstr(value) for key, value in item.items()} for item in solution],
        },
        warnings=[] if solution else ["No se encontro una solucion cerrada para el sistema dado."],
    )


def _solve_matrix(request: ScienceSolveRequest) -> ScienceResult:
    matrix_data = request.parameters.get("matrix")
    if matrix_data is None:
        raise ActionValidationError("matrix operation requires 'matrix'")
    matrix = sp.Matrix(matrix_data)
    determinant = matrix.det()
    inverse = matrix.inv() if matrix.det() != 0 else None
    eigenvalues = matrix.eigenvals()
    return ScienceResult(
        kind="solve",
        domain="mathematics",
        operation="matrix",
        explanation="Se analizaron propiedades matriciales exactas.",
        formulas=["det(A)", "A^-1", "eig(A)"],
        assumptions=["La matriz se interpreto sobre numeros reales/simbólicos de SymPy."],
        result={
            "matrix": [[sp.sstr(value) for value in row] for row in matrix.tolist()],
            "determinant": sp.sstr(determinant),
            "inverse": None if inverse is None else [[sp.sstr(value) for value in row] for row in inverse.tolist()],
            "eigenvalues": {sp.sstr(key): int(value) for key, value in eigenvalues.items()},
        },
        warnings=[] if inverse is not None else ["La matriz es singular; no tiene inversa."],
    )


def _solve_probability(request: ScienceSolveRequest) -> ScienceResult:
    trials = int(request.parameters.get("trials", 0))
    probability = float(request.parameters.get("success_probability", 0.0))
    successes = int(request.parameters.get("successes", 0))
    if not 0.0 <= probability <= 1.0:
        raise ActionValidationError("success_probability must be between 0 and 1")
    if trials < 0 or successes < 0 or successes > trials:
        raise ActionValidationError("invalid binomial parameters")
    coefficient = math.comb(trials, successes)
    value = coefficient * (probability**successes) * ((1 - probability) ** (trials - successes))
    return ScienceResult(
        kind="solve",
        domain="mathematics",
        operation="probability",
        explanation="Se evaluo la probabilidad binomial exacta.",
        formulas=["P(X=k)=C(n,k) p^k (1-p)^(n-k)"],
        assumptions=["Se asumieron ensayos independientes con probabilidad de exito constante."],
        result={
            "distribution": "binomial",
            "trials": trials,
            "success_probability": probability,
            "successes": successes,
            "probability": value,
        },
    )


def _solve_black_hole_escape(request: ScienceSolveRequest) -> ScienceResult:
    params = request.parameters
    black_hole_mass = _coerce_si(params.get("black_hole_mass", 10), params.get("black_hole_mass_unit", "solar_mass"))
    schwarzschild_radius = (2 * G * black_hole_mass) / (C**2)
    radius = params.get("radius")
    assumptions: list[str] = []
    warnings: list[str] = []
    if radius is None:
        radius_value = 3.0 * schwarzschild_radius
        assumptions.append("Como no se dio radio, se asumio una posicion inicial a 3 radios de Schwarzschild.")
    else:
        radius_value = _coerce_si(radius, params.get("radius_unit", "m"))
    if radius_value <= schwarzschild_radius:
        raise ActionValidationError(
            "the starting radius is inside the event horizon",
            details={"schwarzschild_radius_m": schwarzschild_radius},
        )
    probe_mass = _coerce_si(params.get("probe_mass", 1.0), params.get("probe_mass_unit", "kg"))
    burn_time = float(params.get("burn_time", 1.0))
    assumptions.append(f"Para convertir velocidad de escape en fuerza se asumio una aceleracion uniforme en {burn_time:g} s.")
    escape_velocity = math.sqrt((2 * G * black_hole_mass) / radius_value)
    required_acceleration = escape_velocity / max(burn_time, 1e-9)
    required_force = probe_mass * required_acceleration
    if escape_velocity > 0.3 * C:
        warnings.append("La velocidad de escape es relativista; la formula newtoniana es una estimacion optimista.")
    return ScienceResult(
        kind="solve",
        domain="physics",
        operation="black_hole_escape",
        explanation="Se estimo la velocidad de escape y la fuerza media necesaria para acelerar una sonda fuera del potencial local.",
        formulas=[
            "r_s = 2GM/c^2",
            "v_escape = sqrt(2GM/r)",
            "F_promedio = m * v_escape / delta_t",
        ],
        assumptions=assumptions,
        result={
            "black_hole_mass_kg": black_hole_mass,
            "radius_m": radius_value,
            "schwarzschild_radius_m": schwarzschild_radius,
            "probe_mass_kg": probe_mass,
            "burn_time_s": burn_time,
            "escape_velocity_m_s": escape_velocity,
            "escape_velocity_fraction_c": escape_velocity / C,
            "required_average_force_N": required_force,
        },
        warnings=warnings,
    )


def _solve_time_dilation(request: ScienceSolveRequest) -> ScienceResult:
    params = request.parameters
    assumptions: list[str] = []
    result: dict[str, Any] = {}
    formulas: list[str] = []
    interval = float(params.get("coordinate_time_seconds", 1.0))
    if "velocity" in params or "velocity_fraction_c" in params:
        velocity = (
            float(params.get("velocity_fraction_c", 0.0)) * C
            if "velocity_fraction_c" in params
            else _coerce_si(params["velocity"], params.get("velocity_unit", "m/s"))
        )
        beta = velocity / C
        if abs(beta) >= 1.0:
            raise ActionValidationError("velocity must be below the speed of light")
        gamma = 1.0 / math.sqrt(1.0 - beta**2)
        proper_time = interval / gamma
        formulas.append("gamma = 1/sqrt(1-v^2/c^2)")
        result.update(
            {
                "special_relativity_gamma": gamma,
                "coordinate_time_seconds": interval,
                "proper_time_seconds": proper_time,
            }
        )
    if "central_mass" in params:
        central_mass = _coerce_si(params["central_mass"], params.get("central_mass_unit", "kg"))
        radius = _coerce_si(params.get("radius", 0.0), params.get("radius_unit", "m"))
        if radius <= 0:
            raise ActionValidationError("radius must be positive for gravitational time dilation")
        schwarzschild_radius = (2 * G * central_mass) / (C**2)
        if radius <= schwarzschild_radius:
            raise ActionValidationError("radius must be outside the event horizon")
        gravitational_factor = math.sqrt(1.0 - (schwarzschild_radius / radius))
        formulas.append("d_tau = d_t * sqrt(1-r_s/r)")
        result.update(
            {
                "gravitational_factor": gravitational_factor,
                "gravitational_proper_time_seconds": interval * gravitational_factor,
                "schwarzschild_radius_m": schwarzschild_radius,
                "radius_m": radius,
            }
        )
        assumptions.append("Se uso la aproximacion de Schwarzschild para un campo esfericamente simetrico no rotante.")
    if not result:
        raise ActionValidationError(
            "time_dilation requires velocity/velocity_fraction_c or central_mass with radius",
        )
    if "special_relativity_gamma" in result and "gravitational_factor" in result:
        combined = result["coordinate_time_seconds"] * result["gravitational_factor"] / result["special_relativity_gamma"]
        result["combined_proper_time_seconds"] = combined
        assumptions.append("Se multiplico el factor gravitacional y el factor cinemático como estimacion compuesta.")
    return ScienceResult(
        kind="solve",
        domain="physics",
        operation="time_dilation",
        explanation="Se estimo la dilatacion temporal con relatividad especial y/o gravitacional segun los datos aportados.",
        formulas=formulas,
        assumptions=assumptions,
        result=result,
    )


def _infer_operation(query: str) -> str:
    lowered = query.casefold()
    if "deriv" in lowered:
        return "differentiate"
    if "integr" in lowered:
        return "integrate"
    if "simplif" in lowered:
        return "simplify"
    if "matriz" in lowered or "matrix" in lowered:
        return "matrix"
    if "probab" in lowered or "binomial" in lowered:
        return "probability"
    if "agujero negro" in lowered or "black hole" in lowered:
        return "black_hole_escape"
    if "dilat" in lowered or "time dilation" in lowered or "relativ" in lowered:
        return "time_dilation"
    if "sistema" in lowered or "equation" in lowered:
        return "equation_system"
    return "simplify"


def _extract_expression(query: str) -> str:
    separators = (" de ", " of ", ":")
    lowered = query.casefold()
    for separator in separators:
        if separator in lowered:
            index = lowered.rfind(separator)
            candidate = query[index + len(separator) :].strip()
            if candidate:
                return candidate
    for prefix in ("calcula", "resuelve", "simplifica", "deriva", "integra"):
        if lowered.startswith(prefix):
            candidate = query[len(prefix) :].strip()
            if candidate:
                return candidate
    return query


def _parse_equation(raw: str) -> sp.Expr:
    if "=" not in raw:
        return sp.sympify(raw)
    left, right = raw.split("=", 1)
    return sp.Eq(sp.sympify(left), sp.sympify(right))


def _coerce_si(value: Any, unit: str | None = None) -> float:
    if isinstance(value, dict):
        return _coerce_si(value.get("value"), str(value.get("unit") or unit or ""))
    number = float(value)
    normalized = (unit or "").strip().casefold()
    conversions = {
        "": 1.0,
        "m": 1.0,
        "km": 1_000.0,
        "cm": 0.01,
        "mm": 0.001,
        "s": 1.0,
        "min": 60.0,
        "h": 3_600.0,
        "kg": 1.0,
        "g": 0.001,
        "m/s": 1.0,
        "km/s": 1_000.0,
        "solar_mass": 1.98847e30,
    }
    if normalized not in conversions:
        raise ActionValidationError(f"unsupported unit '{unit}'", details={"supported_units": sorted(conversions)})
    return number * conversions[normalized]
