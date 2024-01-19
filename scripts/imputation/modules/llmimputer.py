import os

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tiktoken

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
    def __init__(self, na_value=np.nan, X_categories: dict = {}, dataset_description: str = "", model: str = 'gpt-4', role: str = 'expert', debug: bool = False):
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

        if self.llm_model.startswith("gpt") and self.debug:
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
                variable_type = "Categorical"
                description = ""
                category_list = self.X_categories[column]
                candidates = f'categories: {category_list}'
                missing_values = "no"
            else:
                role = "Feature"
                variable_type = "Numerical"
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
                    streaming=False,
                )
            llama_model = Llama2Chat(llm=llm)
            llama_messages = [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name='chat_history'),
                HumanMessagePromptTemplate.from_template('{text}'),
            ]
            prompt_template = ChatPromptTemplate.from_messages(llama_messages)
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
            chain = LLMChain(llm=llama_model, prompt=prompt_template, memory=memory)
            response = chain.run(text=user_prompt)
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
        # First API Call to generate the Expert Prompt Initialization (epi)
        system_prompt = (
                            "I am going to give you a description of a dataset. "
                            "Please read it and then tell me which hypothetical "
                            "persona would be the best domain expert on the content "
                            "of the data set if I had questions about specific variables, "
                            "attributes or properties.\r\n"
                            "I don't need a data scientist or machine learning expert, "
                            "and I don't have questions about the analysis of the data"
                            "but about specific attributes and values.\r\n"
                            "Please do not give me a list. Just give me a detailed description of a "
                            "(single) person who really knows a lot about the field in which the dataset was generated.\r\n"
                            "Do not use your knowledge about the author of the data record as a guide. "
                            "Do not mention the dataset or anything about it. Do not explain what you do. "
                            "Just give the description and be concise. No Intro like 'An expert would be'."
                        )
        user_prompt_prefix = (
                            "Here is the description of the dataset:\r\n\r\n"
                        )
        user_prompt_suffix = (
                            "\r\n\r\n\r\n\r\n"
                            "Remember: Do not mention the dataset in your description. "
                            "Don\'t explain what you do. Just give me a concise description "
                            "of a hypthetical person, that would be an expert on this.\r\n"
                            "Formulate this as an instruction like \"You are an ...\"."
                        )
        user_prompt = user_prompt_prefix + dataset_description + user_prompt_suffix
        epi_max_tokens = 2048

        if self.model.startswith("gpt"):
            num_tokens = self.__num_tokens_from_messages(system_prompt, user_prompt, self.model)
            if self.debug:
                print(f"- Number of tokens for EPI: {num_tokens}")
            self.num_tokens += num_tokens

        epi_prompt = self.__chat_api_call(self.model, system_prompt, user_prompt, max_tokens=epi_max_tokens)

        if epi_prompt is None:
            raise ValueError("The Expert Prompt Initialization (epi) module returned None.")

        return epi_prompt

    def __data_imputation(self, X_row: pd.Series, dataset_variables_description: dict, epi_prompt: str):
        """
        Data Imputation (DI) module
        """
        # Second API Call to generate the Data Imputation (di)
        system_prompt = epi_prompt + "\r\n\r\n###\r\n\r\n"
        user_prompt_prefix = (
                                "THE PROBLEM: We would like to analyze a data set, "
                                "but unfortunately this data set has some missing values."
                                "\r\n\r\n###\r\n\r\n"
                            )
        user_prompt_suffix = (
                                "YOUR TASK: "
                                "Please use your years of experience and the knowledge you have acquired "
                                "in the course of your work to provide an estimate of what value the missing value "
                                "(marked as <missing>) in the following row of the dataset would most likely have."
                                "\r\n\r\n"
                                "IMPORTANT: "
                                "Please do not provide any explanation or clarification. "
                                "Only answer with the respective value in quotation marks (\"\")."
                                "\r\n\r\n"
                                "Here is a set of values from that row, along with their data type and the column description:"
                                "\r\n\r\n"
                            )

        # run imputation for each target column which has missing values
        def __impute(target_column):
            dataset_row = ""
            for column in list(X_row.index):
                if column != target_column and pd.isna(X_row.loc[column]):
                    continue
                value = "<missing>" if column == target_column else X_row.loc[column]
                variable_type = dataset_variables_description[column]["variable_type"]
                candidates = dataset_variables_description[column]["candidates"]
                dataset_row += f'The {column} is {value} ([Description] variable_type: {variable_type}, {candidates}). '

            user_prompt = user_prompt_prefix + user_prompt_suffix + dataset_row
            di_max_tokens = 256
            
            if self.model.startswith("gpt"):
                num_tokens = self.__num_tokens_from_messages(system_prompt, user_prompt, self.model)
                self.num_tokens += num_tokens
                if self.debug:
                    print(f"- Number of tokens for DI: {num_tokens}")
            di = self.__chat_api_call(self.model, system_prompt, user_prompt, max_tokens=di_max_tokens)

            if di is None:
                raise ValueError("The Data Imputation (di) module returned None.")

            di = di.strip('"').strip("'").strip()

            # convert to float if the value is numerical
            if dataset_variables_description[target_column]["variable_type"] == "Numerical":
                di = float(di)

            if self.debug:
                print(f"- Imputed value: {di}")

            return (target_column, di)

        target_columns = list(X_row[X_row.isna()].index)
        responses = [__impute(target_column) for target_column in target_columns]
        for target_column, di in responses:
            X_row.loc[target_column] = di

        return X_row
