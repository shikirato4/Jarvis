---
name: infografia
description: Generar infografias e imagenes con texto legible pasando TODA la informacion estructurada al modelo de imagen
triggers: infografia, infografía, poster, diagrama visual, imagen con texto, imagen con datos, resumen visual, presentacion visual
version: 1
---
## Cuándo usar
Cuando el usuario pide una infografía, póster, diagrama con texto, o
cualquier imagen que deba contener datos, títulos o cifras legibles.

## Regla de oro
El modelo de imagen SOLO ve el prompt que le mandas — no conoce la
conversación. Antes de generar, RECOGE tú la información: pídesela al
usuario, sácala de la conversación, o investígala. Luego pásala COMPLETA
en el prompt de la imagen.

## Procedimiento
1. Reúne el contenido: título principal, 3-6 puntos o cifras clave,
   conclusión, y cualquier fecha/fuente. Si falta información, pídela
   explícitamente antes de generar (una infografía vacía es inútil).
2. Escribe el prompt en un bloque estructurado con el TEXTO EXACTO que
   debe aparecer en la imagen, entre comillas, por ejemplo:

   Infografía sobre "Producción de café en México".
   Título grande: "CAFÉ DE MÉXICO".
   Cifra destacada: "12.3 millones de sacos (2025)".
   Puntos con icono: "60% Veracruz", "20% Chiapas", "12% Oaxaca".
   Texto del pie: "Fuente: SIAP 2025".
   Estilo: flat design moderno, paleta terracota y crema, layout
   vertical limpio, tipografía sans-serif bold, sin texto adicional.

3. Genera con la acción de imagen (image.generate). Revisa el
   resultado: si el texto salió mal, reintenta con menos texto por
   elemento y descripciones más cortas.
4. Entrega la ruta del archivo y ofrece ajustes.

## Notas
- Los modelos de imagen fallan con más de ~8 elementos de texto:
  si hay más datos, prioriza los 5-6 más importantes.
- Cifras largas y palabras compuestas se deforman: usa números
  redondos y frases cortas cuando puedas.
- Para estilos: sé específico (flat, 3D, isométrico, acuarela,
  minimalista) y fija la paleta con nombres de color.

## Garantia total de letras (híbrido)

Si necesitas CERO galimatías (paneles del modelo inventando texto),
usa el generador híbrido: el modelo pinta la plantilla SIN TEXTO y
PIL escribe las letras exactas con tipografía real.

```
python scripts/infografia_hibrida.py ejemplo -o poster.png
```

Spec JSON: {titulo, subtitulo, paneles:[{titulo, puntos:[...]}],
pie, fuente, estilo}. Verificado: transcripción 11/11 exacta.
