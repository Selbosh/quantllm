# llms-for-quantitative-knowledge-retrieval

## Installation

Install [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/) for environment management. In the `pyproject.toml`, see line `python = ...` for the acceptable Python version, e.g., `">=3.9,<3.10"`, and do the following from the root directory of the project 
```
pyenv install <python version>
pyenv local <python version>
poetry env use <python version>
poetry install
```
e.g., `<python version>` is equal to 3.9.17

For missing packages, use `poetry add <package name>` (see [poetry docs](https://python-poetry.org/)).

## Get data

You can get OpenML-CC18 Curated Classification benchmark datasets and download them locally. The downloaded data will be stored in `/data/openml`. For each dataset, the following files will be downloaded.
- `X.csv` : the feature matrix
- `y.csv` : the classification labels
- `X_categories.json` : a list of categorical variables in the features
- `y_categories.json` : a list of class in `y.csv`
- `description.txt` : the description of the dataset written in OpenML
- `details.json` : a meta data of the dataset

```bash
poetry run python scripts/get-datasets.py 
```

## Missing values imputation

### Preprocess

In the preprocessing step, you will split the original OpenML datasets into train and test subsets, and generate missing values.
Please get OpenML datasets and store them in `/data/openml` in advance.
The splitted complete datasets will be stored in `/data/working/complete`, and the incomplete datasets (datasets with "real" missing values) will be stored in `/data/working/incomplete`.
For the complete datasets, the code artificially generates missing values based on missingness patterns (MCAR, MAR, MNAR).

```bash
poetry run python scripts/imputation/preprocess.py
```

```bash
poetry run python scripts/imputation/preprocess.py
  [--n_corrupted_rows_train N_CORRUPTED_ROWS_TRAIN] [--n_corrupted_rows_test N_CORRUPTED_ROWS_TEST] 
  [--n_corrupted_columns N_CORRUPTED_COLUMNS] [--test_size TEST_SIZE]
  [--seed SEED] [--debug]

required arguments:
  (none)

optional arguments:
  --n_corrupted_rows_train  the default value is 120
  --n_corrupted_rows_test   the default value is 30
  --n_corrupted_columns     the default value is 6. the code will generate max 6 corrupted columns.
  --test_size               the default value is 0.2. fraction of testing subsets for train test split.
  --seed                    default value: 42
  --debug                   display some additional logs to the terminal
```

### LLM Imputer

#### Setups for using LLMs

You can use OpenAI API, or other APIs compatible with OpenAI API, e.g. [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [vLLM](https://github.com/vllm-project/vllm).


Instructions for each model are the following:

- OpenAI API
  Please set your API key in `/.env` as
  ```
  OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
  ```
- Other OpenAI API compatible APIs
  Please set a base URL to the inference server in `/.env` as
  ```
  CUSTOM_INFERENCE_SERVER_URL="YOUR_CUSTOM_INFERENCE_SERVER_URL"
  ```
  If an API key is required, please set it in `/.env` as
  ```
  CUSTOM_API_KEY="YOUR_CUSTOM_API_KEY"
  ```

#### Prompt engineering

To edit prompts, edit `/scripts/imputation/prompts.json`.

```json
{
  "expert_prompt_initialization": {
    "system_prompt": "...",
    "user_prompt_prefix": "...",
    "user_prompt_suffix": "..."
  },
  "non_expert_prompt": "...",
  "data_imputation": {
    "system_prompt_suffix": "...",
    "user_prompt_prefix": "..."
  }
}
```

For each row with missing values, two types of requests to LLMs will be done.

1. **Expert prompt initialization**: Ask LLMs to make prompts for LLMs to act like experts. System prompt: `system_prompt`. User prompt: `user_prompt_prefix + dataset_description + user_prompt_suffix`. `dataset_description` is a description of the dataset downloaded from OpenML.
2. **Data Imputation**: Using the expert prompt, ask LLMs to guess a missing value. System prompt: `epi_prompt` + `system_prompt_suffix`. User prompt: `

(Note)
There may be multiple missing values in the target row. This will be done by repeating step 2 for each missing value in the target row. (Other missing values are hidden)

### Experiment

#### Note

- Please run `generate-missing-values.py` (see above) in advance. Corrupted datasets (datasets with missing values) and the log file (`log.csv`) must be stored in `/data/working/complete` and `/data/working/incomplete`.
- Evaluation is currently unavailable. Needs update.

#### Imputation methods

You can test multiple missing values imputation methods for the generated incomplete datasets.
The following methods are available:
- Mean/Mode (impute numerical values with mean and categorical values with mode)
- K-nearest neighbors
- Random Forest
- LLMs

For example, if you want to impute with Mean/Mode method, run the following command.
```bash
poetry run python scripts/imputation/experiment.py --method meanmode
```

#### LLM Imputer

For LLMs, you can test several models. OpenAI GPT models are available, and also other models OpenAI API compatible models are available.

To select a model, set the `--llm_model` option. The default model is `gpt-4`. For OpenAI GPT models, please use official model names, e.g. `gpt-3.5-turbo`. For OpenAI API compatible models, you can freely set a model name, but please note that `--llm_model` option is required.

```bash
poetry run python scripts/imputation/experiment.py --method llm --llm_model gpt-3.5-turbo
```

You can also ask whether you want LLMs to behave like an expert or not. The default role is `expert`.
```bash
poetry run python scripts/imputation/experiment.py --method llm --llm_model gpt-4 --llm_role nonexpert
```

#### Specifying datasets

If you want to run experiments for a specific dataset, please give the OpenML ID, missingness. For example,
```bash
poetry run python scripts/imputation/experiment.py --method meanmode --openml_id 31 --missingness MCAR
```

#### Downstream task

You can also evaluate downstream tasks by adding the downstream flag.
```bash
poetry run python scripts/imputation/experiment.py --method meanmode --downstream
```

#### Arguments
```bash
poetry run python scripts/imputation/experiment.py
  [--method {meanmode, knn, rf, llm}] [--evaluate] [--downstream]
  [--openml_id OPENML_ID] [--missingness {MCAR, MAR, MNAR}] [--dataset {['incomplete', 'complete'], incomplete, complete}]
  [--llm_model LLM_MODEL] [--llm_role {expert, nonexpert}]
  [--debug]

required arguments:
  --method                  select a imputation method you want to apply (default value: meanmode)

optional arguments:
  --evaluate                calculate RMSE or Macro F1
  --downstream              evaluate downstream tasks
  --openml_id               specify a target openml id
  --missingness             specify a missingness pattern (MCAR or MAR or MNAR)
  --dataset                 specify a dataset type. complete or incomplete.
  --llm_model               specify a llm model. the default is gpt-4
  --llm_role                select whether the llm to be an expert or not.
  --seed                    default value: 42
  --debug                   display some additional logs to the terminal
```

### Modify LLM Imputer

If you want to modify imputation method using LLMs, please edit `/scripts/imputation/modules/llmimputer.py`.

## Prior elicitation

### LLM elicitor

Setup for LLM APIs is the same as for the LLM imputer. See above.

#### Prompt engineering

To edit prompts, edit `/scripts/elicitation/prompts.json`.

### Modify LLM elicitor

If you want to modify the elictation method using LLMs, please edit `/scripts/elicitation/modules/llmelicitor.py`.
