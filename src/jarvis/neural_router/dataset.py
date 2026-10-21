from __future__ import annotations

import random
from collections import defaultdict

from .models import DatasetSplit, RouterExample


_EXAMPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "chat": (
        ("hola cómo estás", ""), ("explícame una idea", ""), ("conversemos sobre filosofía", ""),
        ("ayúdame a pensar", ""), ("qué opinas de este tema", ""), ("resume este concepto", ""),
    ),
    "calculator": (
        ("cuánto es 27 por 83", "calculator.evaluate"), ("calcula 781 multiplicado por 94", "calculator.evaluate"),
        ("resuelve 15 más 19", "calculator.evaluate"), ("raíz cuadrada de 144", "calculator.evaluate"),
        ("qué porcentaje es 20 de 80", "calculator.evaluate"), ("evalúa dos elevado a diez", "calculator.evaluate"),
    ),
    "web": (
        ("busca las últimas noticias", "web.search"), ("qué ocurrió hoy en tecnología", "web.search"),
        ("encuentra fuentes actuales", "web.search"), ("consulta novedades recientes", "web.search"),
        ("investiga en la web este producto", "web.search"), ("busca información pública actualizada", "web.search"),
    ),
    "data": (
        ("analiza este csv", "data.describe"), ("encuentra anomalías en la tabla", "data.anomalies"),
        ("agrupa los datos por categoría", "data.aggregate"), ("ordena este xlsx", "data.sort"),
        ("filtra las filas del json", "data.filter"), ("genera una gráfica con estos datos", "data.plot"),
    ),
    "image": (
        ("genera una imagen de un robot", "image.generate"), ("crea una ilustración futurista", "image.generate"),
        ("dibuja un paisaje nocturno", "image.generate"), ("haz una foto sintética", "image.generate"),
        ("produce arte conceptual", "image.generate"), ("crea un póster visual", "image.generate"),
    ),
    "vision": (
        ("qué aparece en esta foto", "vision.inspect_image"), ("describe la imagen adjunta", "vision.inspect_image"),
        ("lee el texto de la pantalla", "vision.extract_text"), ("qué ves en mi escritorio", "vision.describe_screen"),
        ("analiza esta fotografía", "vision.inspect_image"), ("localiza el botón visible", "vision.locate_element"),
    ),
    "code": (
        ("corrige este bug de python", "code.inspect"), ("modifica la función del repositorio", "code.change"),
        ("ejecuta los tests del proyecto", "code.test"), ("revisa este código", "code.inspect"),
        ("encuentra la clase en el código", "code.search"), ("explica la arquitectura del repo", "code.inspect"),
    ),
    "files": (
        ("abre este archivo", "system.open_path"), ("busca el documento en disco", "system.search_resources"),
        ("muestra el archivo en la carpeta", "system.reveal_target"), ("lee el documento local", "research.workspace_search"),
        ("encuentra un pdf llamado informe", "system.search_resources"), ("localiza la carpeta proyectos", "system.search_resources"),
    ),
    "windows": (
        ("abre spotify", "system.launch_application"), ("cambia a la ventana de word", "interface.focus_window"),
        ("haz click en guardar", "interface.click_mouse"), ("escribe en la aplicación activa", "interface.write_text"),
        ("cierra la ventana actual", "desktop_agent.run_mission"), ("abre notepad", "system.launch_application"),
    ),
    "git": (
        ("muéstrame el diff de git", "git.diff"), ("revisa el estado del repositorio", "git.status"),
        ("busca cambios sin commit", "git.status"), ("analiza las ramas git", "git.branches"),
        ("actualiza el índice del repo", "git.refresh"), ("explica este parche git", "git.diff"),
    ),
    "memory": (
        ("recuerda que mi color es azul", "memory.remember"), ("qué recuerdas de mi proyecto", "memory.search"),
        ("guarda este dato para después", "memory.remember"), ("busca en tu memoria", "memory.search"),
        ("olvida este recuerdo de prueba", "memory.update"), ("recupera lo que te dije ayer", "memory.search"),
    ),
    "science": (
        ("calcula la derivada de x cuadrada", "science.solve_problem"), ("simula una órbita", "science.run_simulation"),
        ("resuelve esta integral", "science.solve_problem"), ("estima la caída libre", "science.solve_problem"),
        ("modela crecimiento exponencial", "science.run_simulation"), ("calcula una matriz", "science.solve_problem"),
    ),
    "audio": (
        ("transcribe este audio", "audio.transcribe"), ("qué se escucha en el mp3", "audio.transcribe"),
        ("lee en voz alta esta respuesta", "voice.speak"), ("analiza el clip de voz", "audio.analyze"),
        ("convierte wav a texto", "audio.transcribe"), ("detén la voz", "voice.cancel"),
    ),
    "video": (
        ("analiza este video", "video.analyze"), ("extrae fotogramas del mp4", "video.frames"),
        ("dime la duración del video", "video.metadata"), ("transcribe el audio del video", "video.transcribe"),
        ("qué ocurre en este clip", "video.analyze"), ("muestrea escenas del video", "video.frames"),
    ),
    "agent": (
        ("haz esta tarea de varios pasos", "autonomy.start_mission"), ("investiga y luego crea un informe", "research.run_task"),
        ("abre el archivo analízalo y grafica", "desktop_agent.run_mission"), ("planifica y ejecuta todo", "autonomy.start_mission"),
        ("usa varias herramientas para resolverlo", "autonomy.start_mission"), ("completa esta misión por tu cuenta", "autonomy.start_mission"),
    ),
}


def build_controlled_dataset() -> list[RouterExample]:
    return [
        RouterExample(text=text, intent=intent, tools=((tool,) if tool else ()))
        for intent, examples in _EXAMPLES.items()
        for text, tool in examples
    ]


def stratified_split(
    examples: list[RouterExample],
    *,
    seed: int = 42,
) -> DatasetSplit:
    grouped: dict[str, list[RouterExample]] = defaultdict(list)
    for example in examples:
        grouped[example.intent].append(example)
    train: list[RouterExample] = []
    validation: list[RouterExample] = []
    test: list[RouterExample] = []
    randomizer = random.Random(seed)  # noqa: S311  # seeded shuffle for ML dataset splits, not crypto
    for label in sorted(grouped):
        items = list(grouped[label])
        if len(items) < 3:
            raise ValueError("each router class requires at least three examples")
        randomizer.shuffle(items)
        test.append(items.pop())
        validation.append(items.pop())
        train.extend(items)
    randomizer.shuffle(train)
    randomizer.shuffle(validation)
    randomizer.shuffle(test)
    return DatasetSplit(train=train, validation=validation, test=test)

