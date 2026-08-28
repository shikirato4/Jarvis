# Nemotron Hybrid Multimodal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before reporting success.

**Goal:** Implementar la migración aditiva de JARVIS a NVIDIA Nemotron 3 Nano Omni con modos AUTO/NVIDIA API/LOCAL, diagnóstico y launcher local extensibles, secretos protegidos y adjuntos multimodales en el chat, conservando toda capacidad anterior.

**Architecture:** Ampliar los contratos actuales de `models_runtime` con proveedores OpenAI-compatible y un coordinador híbrido que fija el proveedor por solicitud. Mantener secretos, runtime local, adjuntos y UI como módulos separados conectados mediante servicios neutrales. Los proveedores heredados quedan disponibles solo por selección explícita y el pipeline local sigue siendo la autoridad para herramientas, permisos, memoria y modalidades no nativas.

**Tech Stack:** Python 3.11+, Pydantic/Pydantic Settings, httpx, PySide6, FastAPI, SQLAlchemy/SQLite, pytest, pytest-qt, respx, Ruff y PyInstaller.

**Restricción de integración:** El usuario prohibió commits y push. Cada paso se ejecutará en el árbol actual preservando los cambios preexistentes y sin reformateo masivo.

---

### Task 1: Cierre robusto del logging

**Files:**
- Modify: `src/jarvis/jarvis_logging.py`
- Modify: `tests/test_jarvis_logging.py`

- [ ] Añadir una prueba con handlers cuyos `flush` y `close` lanzan `OSError(9)` y comprobar que `shutdown_logging()` desacopla todos los handlers sin propagar el fallo.
- [ ] Ejecutar `pytest tests/test_jarvis_logging.py -q -p no:cacheprovider` y observar el fallo esperado.
- [ ] Separar el manejo de `flush` y `close`, capturando únicamente errores de stream cerrado (`OSError`/`ValueError`) y evitando una segunda llamada desprotegida.
- [ ] Repetir la prueba hasta verde y ejecutar los tests de bootstrap/aplicación relacionados.

### Task 2: Contratos neutrales, capacidades y configuración persistente

**Files:**
- Create: `src/jarvis/models_runtime/hybrid.py`
- Modify: `src/jarvis/models_runtime/base.py`
- Modify: `src/jarvis/config.py`
- Modify: `src/jarvis/persistent_config.py`
- Modify: `src/jarvis/models_runtime/__init__.py`
- Create: `tests/test_hybrid_contracts.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_persistent_config.py`

- [ ] Escribir pruebas de enumeraciones, serialización segura, capacidades verificadas, modo predeterminado `auto`, aliases de variables sin prefijo y migración no destructiva de valores heredados.
- [ ] Ejecutar las pruebas y confirmar que fallan por contratos/campos inexistentes.
- [ ] Implementar `LLMMode`, estados operativos, `ProviderCapabilities`, `ProviderStatus`, `ProviderSelection` y contexto/ledger idempotente de solicitud.
- [ ] Añadir configuración exacta remota/local y persistencia de `selected_mode` sin eliminar claves heredadas.
- [ ] Ejecutar pruebas específicas y Ruff en los archivos modificados.

### Task 3: Almacén seguro y redacción de secretos

**Files:**
- Create: `src/jarvis/security_runtime/__init__.py`
- Create: `src/jarvis/security_runtime/secrets.py`
- Modify: `pyproject.toml`
- Create: `tests/test_secret_store.py`

- [ ] Probar resolución keyring→entorno, guardar/reemplazar/eliminar, respuesta solo booleana/estado, y redacción en errores/repr/logs.
- [ ] Observar fallos por módulo ausente.
- [ ] Implementar protocolo de almacén, adaptadores keyring y entorno, estado seguro y composición; `keyring` será opcional y su ausencia tendrá diagnóstico explícito.
- [ ] Ejecutar pruebas y verificar que ninguna clave centinela aparece en la salida capturada.

### Task 4: Proveedores NVIDIA y Nemotron local OpenAI-compatible

**Files:**
- Create: `src/jarvis/models_runtime/openai_compatible.py`
- Create: `src/jarvis/models_runtime/nvidia_nim.py`
- Create: `src/jarvis/models_runtime/nemotron_local.py`
- Modify: `src/jarvis/models_runtime/catalog.py`
- Modify: `src/jarvis/models_runtime/registry.py`
- Create: `tests/test_nvidia_nim_provider.py`
- Create: `tests/test_nemotron_local_provider.py`
- Create: `tests/test_nvidia_live.py`

- [ ] Probar requests y streaming normalizados, salud autenticada, ausencia/invalidación de clave, `401/403`, `429` y `Retry-After`, timeout/red, `5xx`, formatos/capacidades y mensajes sin secretos.
- [ ] Probar que el proveedor local usa únicamente `LOCAL_LLM_BASE_URL`, exige el modelo exacto y produce una guía de arranque cuando el endpoint no responde.
- [ ] Ejecutar las pruebas para confirmar los fallos de importación/contrato.
- [ ] Implementar cliente común inyectable y excepciones tipadas; implementar ambos proveedores y sus entradas de catálogo.
- [ ] Mantener el test real marcado y omitido salvo `RUN_NVIDIA_LIVE_TESTS=true`.
- [ ] Ejecutar tests específicos, lint y comprobación de imports.

### Task 5: Coordinador híbrido, circuit breaker e idempotencia

**Files:**
- Create: `src/jarvis/models_runtime/coordinator.py`
- Modify: `src/jarvis/models_runtime/service.py`
- Modify: `src/jarvis/models_runtime/router.py`
- Create: `tests/test_provider_coordinator.py`
- Modify: `tests/test_model_service.py`

- [ ] Probar AUTO saludable, clave ausente/inválida, red/timeout/5xx, 429, tres fallos, cooldown, recuperación y ambos proveedores caídos.
- [ ] Probar modos estrictos, persistencia/cambio sin reinicio, cero invocaciones NVIDIA en LOCAL y ausencia de fallback GPT-OSS/Ollama.
- [ ] Probar que cada solicitud fija `request_provider`, que el streaming solo reintenta una vez antes de tokens/efectos y que IDs de archivo/herramienta no se duplican.
- [ ] Confirmar la línea roja.
- [ ] Implementar circuit breaker con reloj inyectable, selección y snapshot observable; conectar `ModelService` sin acoplar consumidores al proveedor.
- [ ] Ejecutar tests del runtime de modelos completos.

### Task 6: Diagnóstico de hardware y launcher extensible

**Files:**
- Create: `src/jarvis/local_runtime/__init__.py`
- Create: `src/jarvis/local_runtime/base.py`
- Create: `src/jarvis/local_runtime/diagnostics.py`
- Create: `src/jarvis/local_runtime/registry.py`
- Create: `src/jarvis/local_runtime/vllm_backend.py`
- Create: `src/jarvis/local_runtime/service.py`
- Create: `tests/test_local_runtime.py`

- [ ] Probar inventario de RAM/GPU/VRAM/driver/CUDA/backend, falta de permisos, PyTorch CPU-only, host incompatible y endpoint en otra máquina.
- [ ] Probar perfiles de estabilidad/rendimiento y que `cpu_offload_gb` solo se emite para adaptadores que declaran soporte.
- [ ] Probar rechazo explícito de cambio de modelo/cuantización y registro de un backend futuro sin modificar `ModelService`.
- [ ] Implementar modelos, sondas inyectables, registro y adaptador vLLM declarativo sin descargar ni lanzar automáticamente el modelo.
- [ ] Ejecutar tests y guardar el diagnóstico actual como evidencia, sin afirmar compatibilidad NVFP4.

### Task 7: Registro y almacenamiento seguro de adjuntos

**Files:**
- Create: `src/jarvis/attachments/__init__.py`
- Create: `src/jarvis/attachments/models.py`
- Create: `src/jarvis/attachments/registry.py`
- Create: `src/jarvis/attachments/storage.py`
- Create: `src/jarvis/attachments/service.py`
- Create: `tests/test_attachments.py`

- [ ] Probar selección múltiple, nombres seguros, magic bytes/MIME, SHA-256, tokens opacos, alcance por solicitud y ausencia de Base64/rutas absolutas en serialización pública.
- [ ] Probar traversal, MIME falso, ejecutables/contenido activo, corrupción, tamaño, archivos con indicios de secretos, cancelar/eliminar/reintentar y deduplicación.
- [ ] Observar los fallos antes de implementar.
- [ ] Implementar registro extensible y almacenamiento por solicitud con límites configurables y eliminación recuperable del descriptor pendiente.
- [ ] Ejecutar tests de seguridad y ciclo de vida.

### Task 8: Procesamiento multimodal unificado

**Files:**
- Create: `src/jarvis/attachments/processors.py`
- Create: `src/jarvis/attachments/pipeline.py`
- Modify: módulos de `vision`, `voice` e `indexing` solo donde sea necesario para adaptadores neutrales
- Create: `tests/test_multimodal_pipeline.py`

- [ ] Probar imágenes múltiples, OCR, audio/transcripción, video derivado, PDF/DOCX/TXT/Markdown/CSV/JSON/código, páginas/fragmentos y presupuesto de contexto.
- [ ] Probar que una modalidad nativa recibe solo el adjunto autorizado y que una no nativa recibe derivados; comprobar cero tráfico NVIDIA en LOCAL y no reprocesar el mismo hash.
- [ ] Implementar procesadores adaptadores que reutilicen las funciones existentes y devuelvan resultados estructurados, con errores claros si una dependencia opcional falta.
- [ ] Ejecutar tests multimodales y regresiones de visión/voz/indexación.

### Task 9: Servicios internos y superficie de estado segura

**Files:**
- Modify: `src/jarvis/bootstrap.py`
- Modify: `src/jarvis/desktop/service.py`
- Modify: `src/jarvis/api.py`
- Modify: pruebas de bootstrap, desktop service y API

- [ ] Probar obtener/cambiar modo, estado/proveedor/capacidades, guardar/eliminar/probar clave sin devolverla y preparar/enviar adjuntos con IDs.
- [ ] Confirmar fallos de métodos/endpoints ausentes.
- [ ] Cablear servicios mediante dependencias explícitas y respuestas seguras; no inicializar NVIDIA en LOCAL.
- [ ] Ejecutar suites de API, bootstrap y escritorio.

### Task 10: Selector, Ajustes NVIDIA y botón `+` en PySide

**Files:**
- Modify: `src/jarvis/desktop.py`
- Modify: archivos de UI auxiliares existentes solo si la separación lo exige
- Modify: `tests/test_desktop.py`
- Create: `tests/test_desktop_hybrid_ui.py`
- Create: `tests/test_desktop_attachments_ui.py`

- [ ] Probar selector y persistencia, cambio para la próxima solicitud, badges de proveedor/modelo/estado/latencia/fallback/capacidades y conservación de conversación/archivos pendientes.
- [ ] Probar campo enmascarado, mostrar/ocultar, guardar/reemplazar/eliminar/probar y limpieza del valor sensible.
- [ ] Probar apertura/cierre y teclado del `+`, etiqueta accesible, multiselección, drop/paste, tarjetas, progreso, cancelar/eliminar/reintentar y texto con archivos en anchos de escritorio/compactos.
- [ ] Ejecutar tests offscreen para observar fallos.
- [ ] Implementar los controles respetando el lenguaje visual existente y sin texto hardcodeado de GPT-OSS.
- [ ] Ejecutar tests de UI y una prueba visual/funcional real con backend y frontend iniciados.

### Task 11: Variables, documentación y contexto maestro

**Files:**
- Modify: `.env.example`
- Verify: `.gitignore`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/modules/models_runtime.md`
- Create: `JARVIS_MASTER_CONTEXT(1).md`

- [ ] Actualizar `.env.example` con `JARVIS_LLM_MODE`, valores exactos de modelos/endpoints y perfiles `auto`, dejando `NVIDIA_API_KEY` vacía/con marcador seguro.
- [ ] Verificar con `git check-ignore` que `.env` está ignorado y buscar secretos rastreados sin imprimir valores.
- [ ] Documentar causas/correcciones, funciones conservadas/adaptadas/nuevas, modos, failover, privacidad, clave, endpoint/launcher/offload, hardware, botón `+`, formatos, pruebas, limitaciones, troubleshooting y “Cambios recientes” de 2026-08-16.
- [ ] Crear el contexto maestro actual y actualizar arquitectura; no crear ni reescribir una auditoría histórica inexistente.
- [ ] Comparar el diff funcional contra README mediante un inventario de archivos y secciones.

### Task 12: Verificación integral y auditoría final del diff

**Files:**
- Verify all modified files; no commit/push

- [ ] Ejecutar Ruff format únicamente en archivos Python nuevos/modificados de esta implementación y revisar el diff para evitar reformatos heredados.
- [ ] Ejecutar `ruff check --no-cache` sobre `src` y `tests`.
- [ ] Ejecutar la batería completa `pytest -q -p no:cacheprovider`; registrar por separado el problema ambiental de cache solo si persiste.
- [ ] Ejecutar tests unitarios, integración/UI y el live test solo si la bandera explícita y clave están disponibles.
- [ ] Ejecutar `pip check`, build limpio sdist/wheel, smoke de bootstrap, backend y frontend.
- [ ] Buscar imports rotos, marcadores incompletos, secretos y rutas absolutas serializadas.
- [ ] Revisar `git status --short`, `git diff --check` y el diff completo; conservar los cambios preexistentes y no hacer commit/push.
- [ ] Aplicar `superpowers:verification-before-completion` y redactar el informe final solo con resultados recién observados.
