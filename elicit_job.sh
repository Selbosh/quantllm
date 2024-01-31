#!/usr/bin/bash

# Psychology
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment psychology --llm_role expert
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment psychology --llm_role nonexpert
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment psychology --llm_role conference
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment psychology --llm_role expert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment psychology --llm_role nonexpert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment psychology --llm_role conference --shelf

# Weather
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment weather --llm_role expert
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment weather --llm_role nonexpert
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment weather --llm_role conference
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment weather --llm_role expert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment weather --llm_role nonexpert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model llama --experiment weather --llm_role conference --shelf

# Crowdfunding
# Should only take one or two prompts