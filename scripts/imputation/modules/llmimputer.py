import os

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tiktoken
import re
import json

import openai
from pathlib import Path


class LLMImputer():
    def __init__(self, na_value=np.nan, prompts: dict = {}, model: str = 'gpt-4', role: str = 'expert', X_categories: dict = {}, dataset_description: str = "", log_filepath: Path = None, debug: bool = False):
        '''
        Args:
            - `na_value`: The value to be replaced with the imputation. Default is `np.nan`.
            - `X_categories`: A dictionary of categorical features and their categories.
            - `dataset_description`: A description of the dataset in a markdown format.
            - `model`: The model to be used for the imputation. Default is `gpt-4`.
            - `debug`: If `True`, print debug messages. Default is `False`.
        '''
        self.na_value = na_value
        self.prompts = prompts
        self.model = model
        self.role = role
        self.X_categories = X_categories
        self.dataset_description = dataset_description
        self.log_filepath = log_filepath

        self.n_input_tokens_tiktoken = 0 if self.model.startswith("gpt") else None
        self.n_tokens = {"n_input_tokens": 0, "n_input_tokens_tiktoken": 0, "n_output_tokens": 0, "n_total_tokens": 0}
        self.debug = debug
        self.log = {
            "model": self.model,
            "role": self.role,
            "na_value": f'{self.na_value}',
            "prompts": self.prompts,
            "n_requests": {
                "epi": 0,
                "di": 0
            },
            "n_tokens": {
                "epi": {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "n_total_tokens": 0,
                    "n_input_tokens_tiktoken": 0
                },
                "di": {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "n_total_tokens": 0,
                    "n_input_tokens_tiktoken": 0
                },
                "total": {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "n_total_tokens": 0,
                    "n_input_tokens_tiktoken": 0
                }
            }
        }
        self.__save_log()

    def __save_log(self):
        """
        Save the log to a file.

        Args:
            - `log_filepath`: The path to the log file.
        """
        if self.log_filepath is None:
            return
        with open(self.log_filepath, "w") as f:
            json.dump(self.log, f, indent=2)

    def fit_transform(self, X: pd.DataFrame):
        X_copy = X.copy()

        epi_prompt = ""
        if self.role == "expert":
            epi_prompt = self.__expert_prompt_initialization(self.dataset_description)

        dataset_variable_description = self.__generate_dataset_variables_description(X_copy)

        # The imputation module will be called for each rows with missing values
        # Rows with no missing values will be skipped
        X_copy = X_copy.apply(lambda x: self.__data_imputation(x, dataset_variable_description, epi_prompt) if x.isna().sum() > 0 else x, axis=1)

        if self.debug:
            print(f"Number of tokens: {self.n_tokens}")

        self.__save_log()

        return X_copy

    def fetch_log(self):
        return self.log

    def __num_tokens_from_messages(self, system_prompt, user_prompt, model="gpt-4"):
        """
        Return the number of tokens used by a list of messages.
        Basically, this function is a copy of the sample code from OpenAI.

        Args:
            - `system_prompt`: The system prompt or context for the conversation.
            - `user_prompt`: The user's input or question.
            - `model`: The model to be used (e.g., 'gpt-4').
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
            }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.__num_tokens_from_messages(system_prompt=system_prompt, user_prompt=user_prompt, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.__num_tokens_from_messages(system_prompt=system_prompt, user_prompt=user_prompt, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def __generate_dataset_variables_description(self, X: pd.DataFrame):
        """
        Generate dataset variables description from dataset

        Args:
            - `X`: The dataset to be described.
        """
        X_columns = X.columns
        X_categorical_columns = self.X_categories.keys()
        X_missing_columns = X_columns[X.isna().any()].tolist()

        dataset_variables_description = {}

        for column in X_columns:
            if column in X_categorical_columns:
                role = "Feature"
                variable_type = "categorical"
                description = ""
                category_list = self.X_categories[column]
                candidates = f'candidates are {category_list}'
                missing_values = "no"
            else:
                role = "Feature"
                variable_type = "numerical"
                description = ""
                candidates = ""
                missing_values = "no"

            if column in X_missing_columns:
                missing_values = "yes"

            description = {
                "role": role,
                "variable_type": variable_type,
                "description": description,
                "candidates": candidates,
                "missing_values": missing_values
            }
            dataset_variables_description[column] = description

        return dataset_variables_description

    def __chat_api_call(self, model, messages, temperature=0.2, max_tokens=256, frequency_penalty=0.0):
        """
        Make an API call to OpenAI's GPT-4.

        Args:
            - `model`: The model to be used (e.g., 'gpt-4').
            - `system_prompt`: The system prompt or context for the conversation.
            - `user_prompt`: The user's input or question.
            - `temperature`: Controls randomness (lower is more deterministic).
            - `max_tokens`: Maximum length of the response.
            - `frequency_penalty`: Discourages repetition.

        Returns:
        - The content of the response from GPT-4.
        """
        load_dotenv()

        if self.debug:
            print(f"- Starting API call to {model}")

        response = None

        # Save the number of tokens used for the API call
        n_tokens = {
            "n_input_tokens": 0,
            "n_output_tokens": 0,
            "n_total_tokens": 0
        }

        if model.startswith("gpt"):
            client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            client = openai.OpenAI(
                base_url=os.getenv("CUSTOM_INFERENCE_SERVER_URL"), 
                api_key="not-needed"
            )

        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature
        )

        response = chat.choices[0].message.content
        n_tokens["n_input_tokens"] = chat.usage.prompt_tokens
        n_tokens["n_output_tokens"] = chat.usage.completion_tokens
        n_tokens["n_total_tokens"] = chat.usage.total_tokens

        if self.debug:
            print(f"- Response: {response}")
            print(f"- Number of tokens: {n_tokens}")

        return (response, n_tokens)

    def __expert_prompt_initialization(self, dataset_description: str):
        """
        Expert Prompt Initialization (EPI) module
        """
        if self.debug:
            print("Starting Expert Prompt Initialization (EPI) module...")
        # First API Call to generate the Expert Prompt Initialization (epi)
        system_prompt = self.prompts["expert_prompt_initialization"]["system_prompt"]
        user_prompt_prefix = self.prompts["expert_prompt_initialization"]["user_prompt_prefix"]
        user_prompt_suffix = self.prompts["expert_prompt_initialization"]["user_prompt_suffix"]
        user_prompt = user_prompt_prefix + dataset_description + user_prompt_suffix
        if 'mistral' in self.model:
            messages = [
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        epi_max_tokens = 2048

        if self.model.startswith("gpt"):
            n_input_tokens_tiktoken = self.__num_tokens_from_messages(system_prompt, user_prompt, self.model)
            if self.debug:
                print(f"- Tiktoken: {n_input_tokens_tiktoken} tokens")
            self.n_input_tokens_tiktoken += n_input_tokens_tiktoken

        epi_prompt, ept_n_tokens = self.__chat_api_call(self.model, messages, max_tokens=epi_max_tokens)
        self.log["prompts"]["expert_prompt"] = epi_prompt
        self.log["n_tokens"]["epi"] = ept_n_tokens
        self.log["n_tokens"]["total"]["n_input_tokens"] = self.log["n_tokens"]["epi"]["n_input_tokens"] + self.log["n_tokens"]["di"]["n_input_tokens"]
        self.log["n_tokens"]["total"]["n_output_tokens"] = self.log["n_tokens"]["epi"]["n_output_tokens"] + self.log["n_tokens"]["di"]["n_output_tokens"]
        self.log["n_tokens"]["total"]["n_total_tokens"] = self.log["n_tokens"]["epi"]["n_total_tokens"] + self.log["n_tokens"]["di"]["n_total_tokens"]
        self.log["n_requests"]["epi"] += 1

        if self.debug:
            print("Finished Expert Prompt Initialization (EPI) module.")
            print(f"- EPI Prompt: {epi_prompt}")

        if epi_prompt is None:
            raise ValueError("The Expert Prompt Initialization (epi) module returned None.")

        self.__save_log()

        return epi_prompt

    def __data_imputation(self, X_row: pd.Series, dataset_variables_description: dict, expert_prompt: str | None):
        """
        Data Imputation (DI) module
        """
        # Second API Call to generate the Data Imputation (di)
        if self.role == "expert":
            system_prompt = expert_prompt + self.prompts["data_imputation"]["system_prompt_suffix"]
        else:
            system_prompt = self.prompts["non_expert_prompt"] + self.prompts["data_imputation"]["system_prompt"]
        user_prompt_prefix = self.prompts["data_imputation"]["user_prompt_prefix"]
        user_prompt_infix = self.prompts["data_imputation"]["user_prompt_infix"]
        user_prompt_suffix = self.prompts["data_imputation"]["user_prompt_suffix"]

        # run imputation for each target column which has missing values
        def __impute(target_column):
            dataset_row = ""
            for column in list(X_row.index):
                if column != target_column and pd.isna(X_row.loc[column]):
                    continue
                if column == target_column:
                    value = "<missing>"
                    variable_type = dataset_variables_description[column]["variable_type"]
                    candidates = dataset_variables_description[column]["candidates"]
                    dataset_row += f'The {column} is {value} ({variable_type} variable {", ".join(candidates) if variable_type == "categorical" else ""}). '
                else:
                    value = X_row.loc[column]
                    dataset_row += f'The {column} is {value}. '

            user_prompt = user_prompt_prefix + user_prompt_infix + dataset_row + user_prompt_suffix
            if 'mistral' in self.model:
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

            di_response, di_n_tokens = self.__chat_api_call(self.model, messages, max_tokens=148, temperature=0.0)
            self.log["n_tokens"]["di"]["n_input_tokens"] += di_n_tokens["n_input_tokens"]
            self.log["n_tokens"]["di"]["n_output_tokens"] += di_n_tokens["n_output_tokens"]
            self.log["n_tokens"]["di"]["n_total_tokens"] += di_n_tokens["n_total_tokens"]
            self.log["n_tokens"]["total"]["n_input_tokens"] = self.log["n_tokens"]["epi"]["n_input_tokens"] + self.log["n_tokens"]["di"]["n_input_tokens"]
            self.log["n_tokens"]["total"]["n_output_tokens"] = self.log["n_tokens"]["epi"]["n_output_tokens"] + self.log["n_tokens"]["di"]["n_output_tokens"]
            self.log["n_tokens"]["total"]["n_total_tokens"] = self.log["n_tokens"]["epi"]["n_total_tokens"] + self.log["n_tokens"]["di"]["n_total_tokens"]
            self.log["n_requests"]["di"] += 1

            di = __parser(di_response, dataset_variables_description, target_column)

            if di is None:
                if 'mistral' in self.model:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
                        {"role": "assistant", "content": di_response},
                        {"role": "user", "content": "Parse the result and only provide single value in a JSON format.\nRESPONSE FORMAT: {\"output\": value}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": di_response},
                        {"role": "user", "content": f"{user_prompt_suffix}"}
                    ]
                di_response, di_n_tokens = self.__chat_api_call(self.model, messages, max_tokens=148, temperature=0.0)
                self.log["n_tokens"]["di"]["n_input_tokens"] += di_n_tokens["n_input_tokens"]
                self.log["n_tokens"]["di"]["n_output_tokens"] += di_n_tokens["n_output_tokens"]
                self.log["n_tokens"]["di"]["n_total_tokens"] += di_n_tokens["n_total_tokens"]
                self.log["n_tokens"]["total"]["n_input_tokens"] = self.log["n_tokens"]["epi"]["n_input_tokens"] + self.log["n_tokens"]["di"]["n_input_tokens"]
                self.log["n_tokens"]["total"]["n_output_tokens"] = self.log["n_tokens"]["epi"]["n_output_tokens"] + self.log["n_tokens"]["di"]["n_output_tokens"]
                self.log["n_tokens"]["total"]["n_total_tokens"] = self.log["n_tokens"]["epi"]["n_total_tokens"] + self.log["n_tokens"]["di"]["n_total_tokens"]
                self.log["n_requests"]["di"] += 1
                di = __parser(di_response, dataset_variables_description, target_column)

            if self.debug:
                print(f"- Imputed value: {di}")

            self.__save_log()

            return (target_column, di)

        def __parser(di_response, dataset_variables_description, target_column):
            regex_1 = r'"output":\s?["\']?(.*)["\']?\s?}?'
            regex_2 = r':\s?["\']?(.*)["\']?\s?}?'
            re1_result = re.findall(regex_1, di_response)
            re2_result = re.findall(regex_2, di_response)
            if len(re1_result) > 0:
                di = re1_result[0]
            elif len(re2_result) > 0:
                di = re2_result[0]
            else:
                di = None
            if di is None:
                return None
            di = di.replace(":", "").replace('"', "").replace("'", "").replace("{", "").replace("}", "").replace("output", "").strip()
            di = re.sub(r'\s#.*', '', di)
            if dataset_variables_description[target_column]["variable_type"] == "numerical":
                try:
                    di = float(di)
                except Exception as e:
                    if self.debug:
                        print(f"- Error: {e}")
                    return None
            return di

        target_columns = list(X_row[X_row.isna()].index)
        responses = [__impute(target_column) for target_column in target_columns]
        for target_column, di in responses:
            X_row.loc[target_column] = di

        return X_row
