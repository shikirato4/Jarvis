# Capability Modules

Responsabilidad: exponer capacidades concretas del sistema sin acoplar la superficie API/CLI al detalle de cada integración.

Módulos iniciales:
- `memory`: persistencia y búsqueda.
- `research`: búsqueda local sobre archivos.
- `writer`: generación de artefactos Markdown con rollback.
- `vision`: inspección estructural de imágenes.
- `voice`: inspección estructural de audio.
- `operations`: comandos locales con allowlist.
