#!/usr/bin/bash
MIXTRAL=mistral/8x7B/mixtral-8x7b-instruct-v0.1.Q5_K_M
LLAMA_70B=llama-2-70b-chat.Q5_K_M
MISTRAL_7B=mistralai/Mistral-7B-Instruct-v0.2
GPT_4=gpt-4
LLAMA_13=meta-llama/Llama-2-13b-chat-hf

MODEL=$MISTRAL_7B

# Psychology
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role expert
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role nonexpert
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role conference
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role expert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role nonexpert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role conference --shelf
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role expert --roulette
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role nonexpert --roulette
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment psychology --llm_role conference --roulette

# Weather
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role expert
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role nonexpert
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role conference
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role expert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role nonexpert --shelf
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role conference --shelf
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role expert --roulette
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role nonexpert --roulette
poetry run python scripts/elicitation/experiment.py --llm_model $MODEL --experiment weather --llm_role conference --roulette

# Crowdfunding
# Should only take one or two prompts