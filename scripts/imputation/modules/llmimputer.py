import os

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tiktoken
import re

import openai
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.llms import llamacpp, huggingface_text_gen_inference
from langchain_experimental.chat_models import Llama2Chat


class LLMImputer():
    def __init__(self, na_value=np.nan, prompts: dict = {}, X_categories: dict = {}, dataset_description: str = "", model: str = 'gpt-4', role: str = 'expert', debug: bool = False):
        '''
        Args:
            - `na_value`: The value to be replaced with the imputation. Default is `np.nan`.
            - `X_categories`: A dictionary of categorical features and their categories.
            - `dataset_description`: A description of the dataset in a markdown format.
            - `model`: The model to be used for the imputation. Default is `gpt-4`.
            - `debug`: If `True`, print debug messages. Default is `False`.
        '''
        self.na_value = na_value
        self.X_categories = X_categories
        self.dataset_description = dataset_description
        self.model = model
        self.role = role
        self.num_tokens = 0 if self.model.startswith("gpt") else None
        self.debug = debug
        self.log = {}
        self.rate_limit_per_minute = 500
        self.prompts = prompts
        self.log["model"] = self.model

    def fit_transform(self, X: pd.DataFrame):
        X_copy = X.copy()

        epi_prompt = ""
        if self.role == "expert":
            epi_prompt = self.__expert_prompt_initialization(self.dataset_description)
            self.log["epi_prompt"] = epi_prompt

        dataset_variable_description = self.__generate_dataset_variables_description(X_copy)

        # The imputation module will be called for each rows with missing values
        # Rows with no missing values will be skipped
        X_copy = X_copy.apply(lambda x: self.__data_imputation(x, dataset_variable_description, epi_prompt) if x.isna().sum() > 0 else x, axis=1)

        if self.model.startswith("gpt") and self.debug:
            print(f"Total number of tokens used: {self.num_tokens}")

        self.log["num_tokens"] = self.num_tokens

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
                candidates = f"range from {X[column].min()} to { X[column].max()}"
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

    def __chat_api_call(self, model, system_prompt, user_prompt, temperature=0.2, max_tokens=256, frequency_penalty=0.0):
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

        # Construct the messages for the API call
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = None
        if model.startswith("gpt"):
            chat = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                n=1
            )
            ai_message = chat(messages)
            response = ai_message.content
        elif model.startswith("mistral"):
            chat = ChatMistralAI(
                api_key=os.getenv("MISTRALAI_API_KEY"),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ai_message = chat(messages)
            response = ai_message.content
        elif model.startswith("llama"):
            # Default model is the LlamaCpp model
            if os.getenv("LLAMA2_INFERENCE_SERVER_URL") is None and os.getenv("LLAMA_MODEL_PATH") is None:
                raise ValueError("Please set the environment variables LLAMA_MODEL_PATH or LLAMA2_INFERENCE_SERVER_URL.")
            if os.getenv("LLAMA2_INFERENCE_SERVER_URL") is not None:
                llm = huggingface_text_gen_inference.HuggingFaceTextGenInference(
                    inference_server_url=os.getenv("LLAMA2_INFERENCE_SERVER_URL"),
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                )
            else:
                llm = llamacpp.LlamaCpp(
                    model_path=os.getenv("LLAMA_MODEL_PATH"),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n_ctx=2048,
                    n_gpu_layers=2,
                    repeat_penalty=frequency_penalty,
                    streaming=False,
                    verbose=self.debug,
                )
            llama_model = Llama2Chat(llm=llm)
            llama_messages = [
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template('{text}'),
            ]
            prompt_template = ChatPromptTemplate.from_messages(llama_messages)
            chain = LLMChain(llm=llama_model, prompt=prompt_template)
            response = chain.invoke({'text': user_prompt})['text']
        elif model.startswith("lmstudio"):
            base_url = os.getenv("LMSTUDIO_INFERENCE_SERVER_URL")
            if base_url is None:
                raise ValueError("Please set the environment variable LMSTUDIO_INFERENCE_SERVER_URL.")
            client = openai.OpenAI(base_url=base_url, api_key="not-needed")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                response_format='{type: "json_object"}',
                n=1,
            )
            response = completion.choices[0].message.content

        if self.debug:
            print(f"- Response: {response}")
        return response

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

        epi_max_tokens = 2048

        if self.model.startswith("gpt"):
            num_tokens = self.__num_tokens_from_messages(system_prompt, user_prompt, self.model)
            if self.debug:
                print(f"- Number of tokens for EPI: {num_tokens}")
            self.num_tokens += num_tokens

        epi_prompt = self.__chat_api_call(self.model, system_prompt, user_prompt, max_tokens=epi_max_tokens)

        if self.debug:
            print("Finished Expert Prompt Initialization (EPI) module.")
            print(f"- EPI Prompt: {epi_prompt}")

        if epi_prompt is None:
            raise ValueError("The Expert Prompt Initialization (epi) module returned None.")

        return epi_prompt

    def __data_imputation(self, X_row: pd.Series, dataset_variables_description: dict, epi_prompt: str):
        """
        Data Imputation (DI) module
        """
        # Second API Call to generate the Data Imputation (di)
        system_prompt_prefix = epi_prompt if self.role == "expert" else self.prompts["non_expert_prompt"]
        system_prompt = system_prompt_prefix + self.prompts["data_imputation"]["system_prompt_suffix"]
        user_prompt_prefix = self.prompts["data_imputation"]["user_prompt_prefix"]
        user_prompt_suffix = self.prompts["data_imputation"]["user_prompt_suffix"]

        # run imputation for each target column which has missing values
        def __impute(target_column):
            dataset_row = ""
            for column in list(X_row.index):
                if column != target_column and pd.isna(X_row.loc[column]):
                    continue
                value = "<missing>" if column == target_column else X_row.loc[column]
                variable_type = dataset_variables_description[column]["variable_type"]
                candidates = dataset_variables_description[column]["candidates"]
                dataset_row += f'The {column} is {value} ({variable_type} variable, {candidates}). '

            user_prompt = user_prompt_prefix + user_prompt_suffix + dataset_row
            di_max_tokens = 148  # 256
            
            if self.model.startswith("gpt"):
                num_tokens = self.__num_tokens_from_messages(system_prompt, user_prompt, self.model)
                self.num_tokens += num_tokens
                if self.debug:
                    print(f"- Number of tokens for DI: {num_tokens}")
            di = self.__chat_api_call(self.model, system_prompt, user_prompt, max_tokens=di_max_tokens)

            # find the imputed value from the response
            # the imputed value is in ("{imputed value}") format
            # Use regex to find the imputed value
            # re_result = re.search(r'"(.*)"', di)
            # re_result = re.search(r'"(.*)"', di)
            # re1_result = re.search(r':\s(.*)', di)
            re1_result = re.findall(r':\s"?(.*)"?}?', di)
            if len(re1_result) > 0:
                di = re1_result[0]
            else:
                re2_result = re.findall('"([^"]*)"', di)
                di = re2_result[0] if re2_result else di
            di = di.strip(":").strip('"').strip("'").strip("{").strip("}").strip().strip('"')
            if di is None:
                raise ValueError("The Data Imputation (di) module returned None.")

            # convert to float if the value is numerical
            if dataset_variables_description[target_column]["variable_type"] == "numerical":
                # Check if compatible with float
                try:
                    di = float(di)
                except Exception as e:
                    if self.debug:
                        print(f"- Error: {e}")
                    di = 0.0

            if self.debug:
                print(f"- Imputed value: {di}")

            return (target_column, di)

        target_columns = list(X_row[X_row.isna()].index)
        responses = [__impute(target_column) for target_column in target_columns]
        for target_column, di in responses:
            X_row.loc[target_column] = di

        return X_row
