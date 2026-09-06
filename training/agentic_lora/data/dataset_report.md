# JARVIS Agentic LoRA pilot dataset report

- Schema: jarvis.agentic_lora.v1
- Generator: agentic-pilot-v1
- Seed: 20260831
- Total: 1200
- Splits: {'train': 800, 'validation': 120, 'heldout': 200, 'hard_regression': 80}
- Languages: {'es-MX': 720, 'en-US': 420, 'mixed': 60}
- Real/fictitious/control: {'real': 300, 'fictitious': 720, 'control': 180}
- Categories: {'action_vs_information': 123, 'argument_extraction': 63, 'capability_absent': 60, 'capability_present': 66, 'continuation': 63, 'correct_tool_selection': 63, 'empty_observation': 3, 'false_success': 4, 'fictitious_tool': 674, 'general_explanation': 60, 'irrelevant_distractors': 60, 'missing_required_tool': 63, 'multi_tool': 66, 'no_false_capability_denial': 60, 'no_false_success': 60, 'non_tool_control': 226, 'observation_grounding': 63, 'output_concision': 60, 'parameter_ambiguity': 60, 'real_tool': 300, 'safe_partial_completion': 60, 'scope_honesty': 3, 'security_confirmation': 66, 'security_denied': 3, 'tool_failure': 62, 'tool_loop': 3, 'tool_timeout': 3, 'wrong_argument': 3}
- Estimated length tokens: average 288.13, maximum 568
- Held-out families: aquarium, drone_sim, energy_grid_sim, museum, observatory, spaceship
- Near-duplicate warnings: 0
- Validation failures: 0
