---
name: web-operate
description: Procedimiento para operar sitios web reales en el navegador embebido de JARVIS
triggers: abre, abrir, entra a, navegador, navegador web, browser, entra en la pagina, ve a, navega
version: 1
---
## Cuándo usar
Cuando el usuario pide ABRIR o ENTRAR a un sitio y OPERARLO: buscar algo,
llenar un formulario, dar clic, ver contenido. No es para leer artículos
estáticos (eso es research-web con web.fetch_page); es para hacer cosas
que requieren un navegador real.

## Procedimiento
1. `browser.open` con la URL. El snapshot devuelve título, texto y la lista
   de elementos interactivos con índices `[ref]` (rol + nombre).
2. LOCALIZA el objetivo en la lista antes de actuar: busca por rol
   (textbox, button, link) y por nombre parcial. Nunca inventes un [ref]:
   debe salir del último snapshot.
3. Para BUSCADORES: `browser.type` en el textbox del search, luego
   `browser.press_key` con Enter. NO clikees el botón Buscar: el dropdown
   de sugerencias suele taparlo y el clic se pierde.
4. Después de CADA acción (click, type, Enter) el snapshot regresa el
   estado nuevo: verifica que la URL/título cambió a lo esperado antes del
   siguiente paso. Si no cambió, `browser.read` y reconsidera.
5. `browser.screenshot` para confirmación visual cuando el texto no
   basta (layout, videos, resultados visuales) o antes de pasos delicados.
6. Reporta al usuario QUÉ encontraste y DÓNDE quedó, con la URL final.

## Reglas de seguridad
- NUNCA escribas contraseñas, tarjetas ni datos bancarios con
  `browser.type`. Dile al usuario que los teclee él mismo.
- Confirma con el usuario antes de: comprar, borrar, enviar formularios
  irreversibles o aceptar términos.
- Solo http/https. Si el usuario pide algo raro (file://), explícale el
  rechazo.

## Errores comunes
- `[ref]` fuera de rango → la página cambió: re-lee con `browser.read`.
- Click no navega → puede haber un overlay: prueba Escape y luego Enter,
  o localiza el elemento otra vez.
- Título vacío o "Loading..." → la página aún carga: `browser.read` de nuevo.
