---
name: windows-desktop
description: Operaciones de escritorio Windows con confirmacion y verificacion
triggers: abre, lanza, ejecuta programa, abre notepad, escritorio, abre la app, carpeta
version: 1
---
## Cuándo usar
Cuando el usuario pida abrir aplicaciones, archivos o carpetas del escritorio,
o realizar operaciones sobre ventanas.

## Procedimiento
1. Resuelve el objetivo con las herramientas del sistema (catálogo de
   aplicaciones, resolución de rutas) antes de ejecutar; no adivines rutas.
2. Toda apertura de aplicación pasa por el flujo de confirmación: si la
   puerta pide confirmación, espera el "sí" explícito del usuario.
3. Ejecuta con la herramienta designada (`system.open_target`) en lugar de
   comandos de terminal equivalentes; la herramienta aplica las salvaguardas.
4. Después de ejecutar, verifica el resultado con una observación real
   (ventana activa, proceso) antes de afirmar éxito. Nunca reportes
   "abierto" sin evidencia.
5. Si el objetivo no se encuentra en el catálogo, dilo y propone la
   alternativa más cercana en lugar de forzar una ruta.
