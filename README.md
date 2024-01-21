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

You can use OpenAI API, Llama 2 or Mistral AI API via LangChain.
You can also use other LLMs that behave like OpenAI API, e.g. LM Studio.

Instructions for each model are the following:

- OpenAI API (via LangChain)
  Please set your API key in `/.env`.
  ```
  OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
  ```
- Mistral AI API (via LangChain)
  Please set your API key in `/.env`.
  ```
  MISTRALAI_API_KEY="YOUR_MISTRALAI_API_KEY"
  ```
- Llama 2 (via LangChain)
  First, please follow [the official instructions](https://python.langchain.com/docs/integrations/chat/llama2_chat).
  If you want to use HuggingFaceTextGenInference, please add a URL of your inference server to `/.env`.
  ```
  LLAMA2_INFERENCE_SERVER_URL="YOUR_LLAMA2_INFERENCE_SERVER_URL"
  ```
  If you want to use LlamaCPP, please add a path to `gguf` file to `/.env`.
  ```
  LLAMA_MODEL_PATH="YOUR_LLAMA_MODEL_PATH"
  ```
- LM Studio (or other custom APIs)
  If you want to use LM Studio, please start the local inference server.
  Please set a base URL to `/.env`.
  ```
  LMSTUDIO_INFERENCE_SERVER_URL="http://localhost:1234/v1"
  ```

#### Prompt engineering

To edit prompts, edit `/data/working/prompts.json`.

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

1. **Expert prompt initialization**: AAsk LLMs to make prompts for LLMs to act like experts. System prompt: `system_prompt`. User prompt: `user_prompt_prefix + dataset_description + user_prompt_suffix`. `dataset_description` is a description of the dataset downloaded from OpenML.
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

For LLMs, you can test several models. Mistral AI (via LangChain), Llama 2 (via LangChain) and LM Studio APIs are also available in addition to OpenAI.

To select a model, set the `--llm_model` option. The default model is `gpt-4`. For LM Studio, please add the name of the model as well (e.g. `--llm_model lmstudio-llama-2-7b`).

| Model | Argument (`*`: wildcard regex) |
| ---- | ---- |
| GPT (e.g. gpt-4, gpt-3.5-turbo) | `--llm_model gpt*` |
| Mistral AI | `--llm_model mistral*` |
| Llama 2 | `--llm_model llama*` |
| LM Studio | `--llm_model lmstudio-*` |

```bash
poetry run python scripts/imputation/experiment.py --method llm --llm_model mistral
```

You can also ask whether you want LLMs to behave like an expert or not. The default role is `expert`.
```bash
poetry run python scripts/imputation/experiment.py --method llm --llm_role nonexpert
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

Also, to set hyperparameters for llamaCpp, please edit the following codes in the file.
```python
llm = llamacpp.LlamaCpp(
    model_path=os.getenv("LLAMA_MODEL_PATH"),
    temperature=temperature,
    max_tokens=max_tokens,
    streaming=False,
)
```

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.opendfki.de/sergred/llms-for-quantitative-knowledge-retrieval.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.opendfki.de/sergred/llms-for-quantitative-knowledge-retrieval/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

