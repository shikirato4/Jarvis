from __future__ import annotations

import ast
import math
import re
import socket
import tomllib
from pathlib import Path

from jarvis.core.errors import ActionValidationError, SafetyViolationError

from .base import SecurityAnalyzeRequest, SecurityFinding, SecurityPasswordCheckRequest, SecurityResult

_DANGEROUS_PATTERNS = (
    "hackear cuentas",
    "romper contrase",
    "deface",
    "malware",
    "ransomware",
    "phishing kit",
    "robar credenciales",
    "atacar sistemas externos",
)


def analyze_security(request: SecurityAnalyzeRequest, *, workspace_root: Path) -> SecurityResult:
    _block_dangerous_request(request.query or "")
    audit_kind = _infer_audit_kind(request)
    if audit_kind == "local_ports":
        return _scan_local_ports(request)
    if audit_kind == "secrets":
        target = Path(request.path).expanduser().resolve() if request.path else workspace_root
        return _scan_secrets(target, request.max_findings)
    if audit_kind == "dependencies":
        target = Path(request.path).expanduser().resolve() if request.path else workspace_root
        return _audit_dependencies(target)
    if request.code:
        findings = _analyze_python_code(request.code, "<inline>")
        return _build_code_result(findings, source="<inline>")
    if request.path:
        target = Path(request.path).expanduser().resolve()
        if target.is_dir():
            findings = _analyze_directory(target, request.max_findings)
            return _build_code_result(findings, source=str(target))
        if target.suffix.lower() == ".py":
            findings = _analyze_python_code(target.read_text(encoding="utf-8"), str(target))
            return _build_code_result(findings, source=str(target))
        raise ActionValidationError("security analyze currently supports Python files or directories")
    if request.include_workspace:
        findings = _analyze_directory(workspace_root, request.max_findings)
        return _build_code_result(findings, source=str(workspace_root))
    topic = request.query or "owasp top 10"
    return _teach_topic(topic)


def check_password(request: SecurityPasswordCheckRequest) -> SecurityResult:
    password = request.password
    categories = 0
    if re.search(r"[a-z]", password):
        categories += 1
    if re.search(r"[A-Z]", password):
        categories += 1
    if re.search(r"\d", password):
        categories += 1
    if re.search(r"[^A-Za-z0-9]", password):
        categories += 1
    charset = [26, 26, 10, 33]
    effective_space = sum(value for index, value in enumerate(charset) if index < categories) or 1
    entropy = len(password) * math.log2(effective_space)
    penalties = 0.0
    lowered = password.casefold()
    if len(password) < 12:
        penalties += 18.0
    if re.search(r"(.)\1\1", password):
        penalties += 10.0
    if any(token in lowered for token in ("password", "admin", "jarvis", "1234", "qwerty")):
        penalties += 25.0
    if re.search(r"(0123|1234|abcd|qwer)", lowered):
        penalties += 18.0
    adjusted_entropy = max(entropy - penalties, 0.0)
    guesses = 2**adjusted_entropy
    guesses_per_second = 10_000_000_000
    seconds = guesses / guesses_per_second
    if adjusted_entropy < 40:
        tier = "weak"
        recommendations = [
            "Usa al menos 14 caracteres.",
            "Mezcla mayúsculas, minúsculas, números y símbolos.",
            "Evita palabras comunes, patrones de teclado y datos personales.",
        ]
    elif adjusted_entropy < 60:
        tier = "moderate"
        recommendations = [
            "Aumenta la longitud o usa una passphrase aleatoria.",
            "Evita reutilizar la contraseña en otros servicios.",
        ]
    else:
        tier = "strong"
        recommendations = ["Mantén esta contraseña en un gestor y no la reutilices."]
    return SecurityResult(
        category="password",
        explanation="La contraseña se evaluó mediante longitud, diversidad de caracteres y penalizaciones por patrones comunes; no se intentó romperla.",
        findings=[
            SecurityFinding(
                rule_id="password.strength",
                severity="high" if tier == "weak" else "medium" if tier == "moderate" else "low",
                title=f"Password {tier}",
                message=f"Entropía ajustada estimada: {adjusted_entropy:.2f} bits.",
                recommendation=" ".join(recommendations),
                references=["NIST SP 800-63B"],
            )
        ],
        recommendations=recommendations,
        metadata={
            "tier": tier,
            "length": len(password),
            "entropy_bits": round(adjusted_entropy, 2),
            "estimated_offline_crack_seconds": seconds,
        },
    )


def _analyze_directory(root: Path, max_findings: int) -> list[SecurityFinding]:
    findings: list[SecurityFinding] = []
    for path in root.rglob("*.py"):
        try:
            findings.extend(_analyze_python_code(path.read_text(encoding="utf-8"), str(path)))
        except OSError:
            continue
        if len(findings) >= max_findings:
            return findings[:max_findings]
    return findings[:max_findings]


def _analyze_python_code(code: str, path: str) -> list[SecurityFinding]:
    tree = ast.parse(code, filename=path)
    findings: list[SecurityFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"eval", "exec"}:
                findings.append(
                    _finding("PY001", "critical", "Dynamic code execution", "Se detectó eval/exec, con riesgo alto de RCE.", "Elimina la ejecución dinámica o restringe con parsers seguros.", path, node.lineno, ["OWASP A03:2021"])
                )
            if name.startswith("subprocess.") and any(keyword.arg == "shell" and _is_true(keyword.value) for keyword in node.keywords):
                findings.append(
                    _finding("PY002", "high", "subprocess shell=True", "Invocar subprocess con shell=True expone a command injection.", "Usa listas de argumentos y shell=False.", path, node.lineno, ["OWASP A03:2021"])
                )
            if name in {"pickle.load", "pickle.loads"}:
                findings.append(
                    _finding("PY003", "high", "Unsafe deserialization", "pickle puede ejecutar código al deserializar datos no confiables.", "Usa formatos seguros como JSON o valida la procedencia del blob.", path, node.lineno, ["OWASP A08:2021"])
                )
            if name == "yaml.load":
                findings.append(
                    _finding("PY004", "high", "Unsafe YAML loading", "yaml.load sin SafeLoader puede instanciar objetos peligrosos.", "Usa yaml.safe_load.", path, node.lineno, ["OWASP A08:2021"])
                )
            if name in {"hashlib.md5", "hashlib.sha1"}:
                findings.append(
                    _finding("PY005", "medium", "Weak hash primitive", "MD5/SHA1 no son adecuados para integridad o contraseñas modernas.", "Usa SHA-256/SHA-3 o Argon2/bcrypt/scrypt para contraseñas.", path, node.lineno, ["NIST SP 800-131A"])
                )
            if name.startswith("requests.") and any(keyword.arg == "verify" and not _is_true(keyword.value) for keyword in node.keywords):
                findings.append(
                    _finding("PY006", "high", "TLS verification disabled", "La llamada HTTP desactiva la validación TLS.", "Mantén verify=True y usa CA confiables.", path, node.lineno, ["OWASP A02:2021"])
                )
            if name == "tempfile.mktemp":
                findings.append(
                    _finding("PY007", "medium", "Insecure temporary file", "tempfile.mktemp es vulnerable a race conditions.", "Usa NamedTemporaryFile o mkstemp.", path, node.lineno, ["CWE-377"])
                )
            if name.endswith(".execute") and node.args and isinstance(node.args[0], (ast.JoinedStr, ast.BinOp)):
                findings.append(
                    _finding("PY008", "high", "Potential SQL injection", "Se construye una sentencia SQL mediante interpolación.", "Usa consultas parametrizadas.", path, node.lineno, ["OWASP A03:2021"])
                )
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and re.search(r"(password|secret|token|api[_-]?key)", target.id, re.IGNORECASE):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        findings.append(
                            _finding("PY009", "high", "Hardcoded secret", "Hay un secreto hardcodeado en el código fuente.", "Mueve el secreto a variables de entorno o un gestor dedicado.", path, node.lineno, ["OWASP A02:2021"])
                        )
    return findings


def _scan_local_ports(request: SecurityAnalyzeRequest) -> SecurityResult:
    if request.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SafetyViolationError(
            "local port scanning is restricted to the current host",
            details={"allowed_hosts": ["127.0.0.1", "localhost", "::1"]},
        )
    ports = request.ports or [22, 80, 443, 8000, 8080, 11434]
    open_ports: list[int] = []
    findings: list[SecurityFinding] = []
    timeout_seconds = max(request.timeout_ms, 50) / 1000.0
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_seconds)
            is_open = sock.connect_ex(("127.0.0.1", int(port))) == 0
        if is_open:
            open_ports.append(int(port))
            severity = "medium" if port in {22, 80, 443} else "low"
            findings.append(
                SecurityFinding(
                    rule_id="NET001",
                    severity=severity,
                    title=f"Puerto local abierto: {port}",
                    message=f"El puerto {port} responde en localhost.",
                    recommendation="Confirma que el servicio expuesto sea esperado y esté endurecido.",
                    references=["CIS Controls 12"],
                )
            )
    return SecurityResult(
        category="local_ports",
        explanation="Se realizó una comprobación TCP básica sobre el propio host.",
        findings=findings,
        recommendations=[
            "Cierra los puertos que no necesites.",
            "Documenta qué servicio escucha en cada puerto abierto.",
        ],
        warnings=["El escaneo está limitado a localhost y a conectividad TCP básica; no hace fingerprinting ni explotación."],
        metadata={"host": request.host, "ports_checked": ports, "open_ports": open_ports},
    )


def _scan_secrets(target: Path, max_findings: int) -> SecurityResult:
    root = target if target.is_dir() else target.parent
    patterns = (
        (r"(?i)api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]", "SECRET001", "Possible API key"),
        (r"(?i)secret\s*[:=]\s*['\"][^'\"]{8,}['\"]", "SECRET002", "Possible secret"),
        (r"(?i)token\s*[:=]\s*['\"][^'\"]{8,}['\"]", "SECRET003", "Possible token"),
        (r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----", "SECRET004", "Private key material"),
    )
    findings: list[SecurityFinding] = []
    files = [target] if target.is_file() else list(root.rglob("*"))
    for path in files:
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", "node_modules", ".pytest_cache"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, rule_id, title in patterns:
                if re.search(pattern, line):
                    findings.append(
                        SecurityFinding(
                            rule_id=rule_id,
                            severity="high",
                            title=title,
                            message="Se detectó una cadena con alta probabilidad de secreto embebido.",
                            recommendation="Mueve el secreto a variables de entorno o un gestor y rota la credencial si es real.",
                            file_path=str(path),
                            line=line_number,
                            references=["OWASP A02:2021"],
                        )
                    )
                    if len(findings) >= max_findings:
                        return _build_secrets_result(findings, str(target))
    return _build_secrets_result(findings, str(target))


def _build_secrets_result(findings: list[SecurityFinding], source: str) -> SecurityResult:
    return SecurityResult(
        category="secrets",
        explanation="Se inspeccionaron archivos locales en busca de patrones de secretos expuestos.",
        findings=findings,
        recommendations=[
            "Elimina secretos del repositorio y rota las credenciales afectadas.",
            "Añade escaneo de secretos en CI antes de mergear cambios.",
        ],
        warnings=["El escaneo usa patrones; puede omitir secretos ofuscados o marcar cadenas benignas."],
        metadata={"source": source, "finding_count": len(findings)},
    )


def _audit_dependencies(target: Path) -> SecurityResult:
    project_root = target if target.is_dir() else target.parent
    pyproject = project_root / "pyproject.toml"
    requirements = project_root / "requirements.txt"
    deps: list[str] = []
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        deps.extend(data.get("project", {}).get("dependencies", []))
    elif requirements.exists():
        deps.extend(
            line.strip()
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    else:
        return SecurityResult(
            category="dependencies",
            explanation="No se encontró `pyproject.toml` ni `requirements.txt` para auditar dependencias.",
            recommendations=["Indica una ruta de proyecto válida o añade manifiestos de dependencias."],
            metadata={"source": str(project_root)},
        )
    findings: list[SecurityFinding] = []
    for dep in deps:
        normalized = dep.casefold()
        if "django<" in normalized or "flask<" in normalized:
            findings.append(
                SecurityFinding(
                    rule_id="DEP001",
                    severity="medium",
                    title="Version range potentially stale",
                    message=f"La dependencia `{dep}` parece fijar una versión antigua o muy acotada.",
                    recommendation="Revisa changelog, CVEs y actualiza a una rama soportada.",
                    references=["OWASP A06:2021"],
                )
            )
        if "pyyaml" in normalized:
            findings.append(
                SecurityFinding(
                    rule_id="DEP002",
                    severity="low",
                    title="Dependency requires safe usage review",
                    message="PyYAML requiere revisar que el proyecto use `safe_load`.",
                    recommendation="Verifica el uso de `yaml.safe_load` y añade tests de regresión.",
                    references=["OWASP A08:2021"],
                )
            )
    return SecurityResult(
        category="dependencies",
        explanation="Se realizó una auditoría básica del manifiesto de dependencias del proyecto.",
        findings=findings,
        recommendations=[
            "Complementa esto con un escáner de CVEs como pip-audit en CI.",
            "Mantén un proceso de actualización periódica de dependencias.",
        ],
        warnings=["La auditoría actual es básica y heurística; no consulta bases de CVEs en red."],
        metadata={"source": str(project_root), "dependency_count": len(deps), "dependencies": deps},
    )


def _teach_topic(topic: str) -> SecurityResult:
    lowered = topic.casefold()
    catalog = {
        "owasp": (
            "OWASP Top 10 resume las categorías de riesgo más frecuentes en aplicaciones web modernas.",
            [
                "Controla inyecciones, autenticación rota, fallos criptográficos y SSRF.",
                "Integra validación de entrada, autorización por defecto denegada y registro de eventos.",
                "Automatiza análisis estático, dependencias y tests de seguridad en CI.",
            ],
        ),
        "xss": (
            "XSS ocurre cuando datos no confiables llegan al navegador sin escaping/contextual encoding adecuado.",
            [
                "Escapa salida según contexto HTML, atributo, JS o URL.",
                "Usa CSP y plantillas seguras por defecto.",
                "Evita innerHTML salvo sanitización estricta.",
            ],
        ),
        "sqli": (
            "La inyección SQL aparece cuando la consulta mezcla código y datos sin parametrización.",
            [
                "Usa parámetros enlazados en todas las consultas.",
                "Restringe permisos de la cuenta de base de datos.",
                "Registra errores sin exponer detalles al cliente.",
            ],
        ),
        "criptograf": (
            "La seguridad criptográfica depende tanto del algoritmo como del manejo de claves, nonces y almacenamiento.",
            [
                "Para contraseñas usa Argon2id, bcrypt o scrypt.",
                "Para cifrado simétrico usa AES-GCM o ChaCha20-Poly1305.",
                "Nunca reutilices nonces ni hardcodees claves.",
            ],
        ),
        "red": (
            "Una postura defensiva en red prioriza segmentación, mínimos privilegios y visibilidad.",
            [
                "Cierra puertos innecesarios y documenta la superficie expuesta.",
                "Aplica TLS, autenticación fuerte y logging centralizado.",
                "Audita servicios locales y el propio workspace antes de tocar terceros.",
            ],
        ),
    }
    for key, (explanation, recommendations) in catalog.items():
        if key in lowered:
            return SecurityResult(
                category="education",
                explanation=explanation,
                recommendations=recommendations,
                warnings=["Contenido con fines educativos y defensivos; no se proporcionan pasos de explotación operativa."],
            )
    return SecurityResult(
        category="education",
        explanation="El runtime de seguridad enseña conceptos defensivos, OWASP, criptografía básica y análisis estático local.",
        recommendations=[
            "Pide un tema concreto como OWASP Top 10, XSS, SQLi, criptografía básica o redes.",
            "Usa security analyze con código o rutas para una auditoría local real.",
        ],
    )


def _build_code_result(findings: list[SecurityFinding], *, source: str) -> SecurityResult:
    recommendations = []
    if any(item.severity in {"critical", "high"} for item in findings):
        recommendations.append("Corrige primero los hallazgos críticos/altos y añade tests de regresión de seguridad.")
    if not findings:
        recommendations.append("No se detectaron patrones inseguros de las reglas implementadas; sigue faltando revisión manual.")
    return SecurityResult(
        category="code_analysis",
        explanation="Se ejecutó un análisis estático local sobre Python para detectar patrones de riesgo comunes.",
        findings=findings,
        recommendations=recommendations,
        warnings=["El análisis es heurístico: puede omitir vulnerabilidades lógicas o reportar falsos positivos."],
        metadata={"source": source, "finding_count": len(findings)},
    )


def _finding(rule_id: str, severity: str, title: str, message: str, recommendation: str, path: str, line: int, references: list[str]) -> SecurityFinding:
    return SecurityFinding(
        rule_id=rule_id,
        severity=severity,
        title=title,
        message=message,
        recommendation=recommendation,
        file_path=path,
        line=line,
        references=references,
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_true(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value) is True


def _block_dangerous_request(text: str) -> None:
    lowered = text.casefold()
    if any(pattern in lowered for pattern in _DANGEROUS_PATTERNS):
        raise SafetyViolationError(
            "security runtime only supports defensive analysis and ethical education",
            details={"safe_alternative": "Usa análisis de código local, auditoría del workspace, escaneo de secretos o revisión defensiva de contraseñas."},
        )


def _infer_audit_kind(request: SecurityAnalyzeRequest) -> str | None:
    if request.audit_kind is not None:
        return request.audit_kind
    lowered = (request.query or "").casefold()
    if "puerto" in lowered or "port" in lowered or "localhost" in lowered:
        return "local_ports"
    if "secret" in lowered or "token" in lowered or "api key" in lowered:
        return "secrets"
    if "dependenc" in lowered or "dependencia" in lowered or "pyproject" in lowered or "requirements" in lowered:
        return "dependencies"
    return None
