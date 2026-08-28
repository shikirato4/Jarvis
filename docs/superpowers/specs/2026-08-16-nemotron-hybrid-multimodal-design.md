# Diseño híbrido Nemotron y compositor multimodal

**Fecha:** 2026-08-16  
**Estado:** aprobado para implementación  
**Opción elegida:** A — ampliar las abstracciones existentes de `models_runtime` y mantener la aplicación monolítica modular.

## Objetivo

Migrar el modelo principal de JARVIS a NVIDIA Nemotron 3 Nano Omni sin reducir las funciones actuales. JARVIS podrá elegir entre NVIDIA API y un endpoint local OpenAI-compatible que sirva exactamente `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`. El modo, proveedor real, salud, capacidades y causa de fallback permanecerán separados y observables.

La arquitectura no prometerá que NVFP4 funciona en el equipo actual. Permitirá utilizarlo posteriormente en hardware compatible, en otra máquina de la red o mediante un backend futuro registrado sin cambiar `ModelService`.

## Restricciones confirmadas

- La migración es aditiva: memoria, herramientas, voz, visión, automatizaciones, permisos, lectura de archivos y razonamiento permanecen disponibles.
- `gpt_oss` y Ollama se conservan como compatibilidad explícita durante la migración; nunca son fallback oculto de `AUTO`, `NVIDIA API` o `LOCAL`.
- El modelo y la cuantización locales no se sustituyen silenciosamente.
- La clave NVIDIA nunca se devuelve al frontend, conversación, logs o telemetría.
- `LOCAL` no realiza tráfico hacia NVIDIA.
- Un proveedor queda fijado por solicitud; cambiar el selector solo afecta la siguiente solicitud.
- Los archivos se guardan como referencias locales protegidas, nunca como Base64 dentro del historial.
- Las herramientas continúan ejecutándose localmente y bajo los controles de permisos existentes.

## Inventario de capacidades y estrategia de conservación

| Función actual | Componente responsable | Dependencia del modelo anterior | NVIDIA API | Nemotron local NVFP4 | Mejora y pruebas | Estado objetivo |
| --- | --- | --- | --- | --- | --- | --- |
| Chat con streaming | `models_runtime.ModelService`, escritorio y API | Proveedor Ollama/GPT-OSS | Compatible mediante chat completions OpenAI | Compatible mediante endpoint OpenAI-compatible | Normalizar eventos, fijar proveedor por solicitud y probar fallo antes/después del primer token | Conservada y reforzada |
| Razonamiento | Catálogo, prompts y proveedor | Capacidades declaradas de GPT-OSS | Declarada por el endpoint oficial; se registra explícitamente | Se registra según endpoint/modelo configurado | Exponer capacidad y probar que no se asume para proveedores desconocidos | Mejorada |
| Tool calling | `tools`, políticas y bucle del agente | Esquema de herramientas del proveedor | Declarada; las acciones siguen siendo locales | Depende del servidor OpenAI-compatible | Identificadores idempotentes y pruebas de no duplicación tras failover | Conservada y reforzada |
| Memoria y conversación | `memory`, SQLite y orquestación | Neutral en su mayor parte | Se envía solo el contexto mínimo | Permanece local | Probar persistencia y ausencia de secretos/archivos completos | Conservada |
| Visión y OCR | Módulos de visión/OCR | Pipeline local separado | Imágenes nativas verificadas | Solo si el endpoint anuncia visión; de lo contrario pipeline local | Registro de capacidades, varias imágenes y derivación OCR | Conservada y extendida |
| Voz, STT y TTS | Módulos de voz | Motores locales y servicios configurables | Audio de entrada verificado; TTS actual permanece | Audio nativo solo si se declara; transcripción local como ruta neutral | Marcas de tiempo y pruebas de fallback de modalidad | Conservada |
| Video | Procesamiento local existente/parcial | No depende directamente del LLM | Video de entrada verificado dentro de límites del servicio | Solo si el endpoint lo declara | Pipeline neutral de audio, fotogramas y resumen derivado | Extendida |
| Documentos e indexación | Ingesta, lectores e índice | Texto enviado al modelo | PDF bruto no es entrada nativa; se envía contenido/páginas relevantes | Texto/imágenes según capacidades | Extracción segura, fragmentación y referencias a páginas | Conservada y extendida |
| Automatizaciones | Registro de acciones y servicios | Propuestas del modelo | Compatible con tool calling, sujeto a permisos | Compatible si el endpoint ofrece tool calling | Ledger de efectos y confirmación si una respuesta queda incompleta | Conservada |
| Permisos y seguridad | Políticas y confirmaciones locales | Neutral al proveedor | El remoto no puede evitarlas | El local no puede evitarlas | Pruebas de no duplicación y de alcance por solicitud | Conservada |
| Lectura de archivos/código | Herramientas e indexación | Neutral | Solo derivados necesarios | Procesamiento completamente local en modo LOCAL | Registro extensible MIME, magic bytes y límites | Mejorada |
| Generación de imágenes | Módulo de imágenes existente | Backend separado | No se atribuye al nuevo modelo | No se atribuye al nuevo modelo | Mantener backend existente y prueba de compatibilidad | Conservada |
| CLI y API | Typer/FastAPI | Configuración del modelo anterior | Estados normalizados | Estados normalizados | Endpoints/servicios sin secretos y pruebas de contrato | Conservada y extendida |

## Arquitectura

### Contratos neutrales

`models_runtime` incorpora tipos estables para:

- `LLMMode`: `auto`, `nvidia_api`, `local`.
- `ProviderOperationalState`: `cloud_healthy`, `cloud_unavailable`, `cloud_rate_limited`, `cloud_invalid_key`, `local_active`, `both_unavailable`, `recovering`.
- `ProviderCapabilities`: modalidades, streaming, razonamiento, salidas estructuradas, herramientas, límites y formatos verificados.
- `ProviderStatus`: salud, latencia, estado, última comprobación y mensaje seguro.
- `ProviderSelection`: `selected_mode`, `active_provider`, `request_provider`, `fallback_reason` y capacidades efectivas.
- `RequestExecutionContext`: identificadores de turno/solicitud/archivo/llamada y registro de efectos externos.

Estos contratos impiden que la UI o la memoria dependan de NVIDIA, vLLM, Ollama u otro proveedor concreto.

### Proveedores

`NvidiaNimProvider` usa `NVIDIA_BASE_URL`, el modelo remoto exacto configurado y una clave obtenida a través del almacén de secretos. La prueba de salud es autenticada y clasifica `401/403`, `429`, timeout/red y `5xx`. Sus errores no contienen cabeceras ni credenciales.

`NemotronOmniLocalProvider` usa `LOCAL_LLM_BASE_URL` y exige el modelo local exacto configurado. Puede apuntar a `127.0.0.1`, otra máquina o un backend futuro. Nunca cambia modelo ni cuantización y no conoce la clave NVIDIA.

Los proveedores existentes `OllamaProvider` y `GptOssProvider` siguen registrables mediante selección de compatibilidad explícita, fuera de los tres modos nuevos.

### Selección, circuit breaker y recuperación

Un coordinador selecciona una sola vez al inicio de cada solicitud:

- `AUTO`: prueba NVIDIA de forma autenticada cuando su circuito permite intentos; si no está disponible, selecciona Nemotron local. La recuperación se aplica en la solicitud siguiente.
- `NVIDIA API`: selecciona exclusivamente NVIDIA y devuelve error seguro si falla.
- `LOCAL`: selecciona exclusivamente el endpoint local y no construye ni invoca un cliente NVIDIA.

Tras tres fallos consecutivos el circuito remoto se abre. `Retry-After` controla la reapertura en `429`; el resto usa un enfriamiento configurable. Un reloj inyectable hace deterministas las pruebas. Un éxito restablece el circuito.

Si el streaming falla antes de emitir contenido y no hay efectos externos, AUTO puede repetir una sola vez con el proveedor local. Si ya hubo contenido o un efecto, la respuesta queda marcada incompleta y exige confirmación; las llamadas de herramienta conservan identificadores idempotentes.

### Secretos

El backend expone únicamente `configured`, `validity` y `source`. El orden de resolución es:

1. almacén seguro del sistema operativo mediante `keyring`, cuando está disponible;
2. variable `NVIDIA_API_KEY` en el proceso o archivo `.env` local.

Guardar, reemplazar o eliminar desde Ajustes opera sobre el almacén seguro. La clave solo vive temporalmente en el campo enmascarado y se limpia tras la operación. Nunca se usa `localStorage`.

### Runtime local, diagnóstico y offload

`local_runtime` define un protocolo de adaptador, un registro y un servicio de lanzamiento. Cada adaptador declara sus opciones compatibles; el perfil de offload solo se traduce a argumentos cuando el backend declara soporte real. `ModelService` únicamente habla con el endpoint, por lo que añadir un adaptador futuro no lo modifica.

El diagnóstico informa por separado:

- alcance del endpoint;
- GPU, VRAM y estado reportado;
- RAM;
- driver, CUDA y backend;
- soporte conocido de NVFP4;
- requisitos faltantes;
- parámetros de contexto/paralelismo/offload aplicables.

Un endpoint remoto local puede estar sano aunque el anfitrión de la UI no pueda ejecutar el modelo. No se infiere compatibilidad local a partir de que el endpoint responda.

### Adjuntos y pipeline multimodal

`attachments` contiene:

- registro extensible de categorías, firmas y formatos;
- almacenamiento protegido por solicitud;
- descriptores neutrales con ID, nombre seguro, MIME detectado, tamaño, SHA-256, token opaco, estado, metadatos, derivados y modalidad requerida;
- procesadores que reutilizan OCR, visión, transcripción e indexación existentes;
- control de tamaño, magic bytes, ZIP/DOCX, corrupción, path traversal, contenido activo y posibles secretos.

El historial solo conserva el descriptor y la referencia opaca autorizada para la solicitud. Los documentos grandes se fragmentan y se recuperan por relevancia; los originales no se envían completos. En `LOCAL`, todos los procesadores y el LLM local permanecen fuera de NVIDIA. En remoto solo se envían los adjuntos seleccionados o derivados mínimos necesarios.

### Interfaz

El compositor PySide incorpora un botón `+` con nombre accesible `Adjuntar archivos`, menú por categoría, selección múltiple, arrastrar/soltar, pegado, tarjetas previas, progreso, cancelar, eliminar y reintentar. El estado de archivos pendientes vive fuera del selector de proveedor para que cambiar de modo no los borre.

El selector ofrece `AUTO`, `NVIDIA API` y `LOCAL`, persiste la selección y muestra proveedor/modelo real, estado, latencia, fallback, clave configurada y capacidades. Ajustes incluye el campo seguro y acciones de guardar, eliminar y probar conexión.

## Persistencia y compatibilidad

La configuración persistente añade el modo sin eliminar las claves heredadas. Los valores históricos se migran mediante adaptadores explícitos y reversibles. Las conversaciones y bases existentes no se reescriben. Cualquier cambio de esquema usa creación aditiva y valores predeterminados seguros.

## Estrategia de pruebas

La implementación sigue TDD. Las pruebas normales usan clientes, relojes, almacenes y backends falsos deterministas. Las pruebas reales NVIDIA solo se ejecutan si `RUN_NVIDIA_LIVE_TESTS=true` y jamás imprimen la clave.

Se cubren selección/persistencia de modos, salud y errores HTTP, `Retry-After`, circuit breaker, recuperación, cero tráfico remoto en LOCAL, ambos proveedores caídos, streaming e idempotencia; secretos; diagnóstico/launcher/offload; seguridad y ciclo de vida de adjuntos; modalidades y derivados; controles de UI y compatibilidad con memoria, conversación, herramientas y funciones anteriores.

## Evidencia y limitaciones conocidas de este entorno

La implementación no depende de que el anfitrión actual ejecute NVFP4. En la línea base, PyTorch es CPU-only, CUDA no está disponible para el proceso y no fue posible obtener una lectura utilizable de `nvidia-smi`; por ello el funcionamiento local NVFP4 no se declara verificado. El proveedor y sus pruebas se validan contra endpoints simulados, mientras que el diagnóstico conserva las razones exactas para un servidor real.

La documentación histórica de auditoría no se inventará. Se creará un contexto maestro actual y se actualizarán README y arquitectura con hechos verificados y comandos reproducibles.
