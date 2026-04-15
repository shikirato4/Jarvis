# Architecture

The current Jarvis codebase is centered on a local runtime kernel with explicit boundaries:

- `bootstrap.py` wires all dependencies in one place.
- `core/` contains policy and runtime contracts that other layers depend on.
- `actions/`, `tools/`, and `models_runtime/` are separate execution surfaces.
- `routing/` is the first entry point for top-level work.
- `services/runtime.py` is the stable facade for external callers.
- `core/capabilities.py` is the central registry for intents, actions, tools, and planning metadata.
- `models_runtime/service.py` is the only layer that talks to providers.

This split matters because the desktop shell, local automations, background agents, and future multimodal providers should all talk to the same core without duplicating runtime logic.
