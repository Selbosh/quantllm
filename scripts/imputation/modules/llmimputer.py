import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import time
from openai import OpenAI


class LLMImputer():
    def __init__(self, na_value=np.nan, X_categories: dict = {}, dataset_description: str = None):
        self.na_value = na_value
        self.X_categories = X_categories
        self.dataset_description = dataset_description
        
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


    def generate_dataset_variables_description(self, X: pd.DataFrame):
        """
        Generate dataset variables description from dataset
        """
        X_original_columns = X.columns
        X_categorical_columns = self.X_categories.keys()
        X_numerical_columns = list(set(X_original_columns) - set(X_categorical_columns))
        X_missing_columns = X_original_columns[X.isna().any()].tolist()

        dataset_variables_description = "Variable Name\tRole\tType\tDescription\tCandidates\tMissing Values\r\n"

        for column in X_original_columns:
            if column in X_numerical_columns:
                role = "Feature"
                variable_type = "Numerical"
                description = ""
                candidates = f"range: from {X[column].min()} to { X[column].max()}"
                missing_values = "no"
            elif column in X_categorical_columns:
                role = "Feature"
                variable_type = "Categorical"
                description = ""
                category_list = list(filter(lambda x: not pd.isna(x), X[column].unique().tolist()))
                candidates = f'categories: {category_list}'
                missing_values = "no"
            else:
                role = ""
                variable_type = ""
                description = ""
                candidates = ""
                missing_values = ""

            if column in X_missing_columns:
                missing_values = "yes"

            dataset_variables_description += f"{column}\t{role}\t{variable_type}\t{description}\t{candidates}\t{missing_values}\r\n"

        return dataset_variables_description


    def fit_transform(self, X: pd.DataFrame):
        X_copy = X.copy()

        epi_prompt = self.expert_prompt_initialization(self.dataset_description)
        dataset_variable_description = self.generate_dataset_variables_description(X_copy)

        # The imputation module will be called for each rows with missing values
        # Rows with no missing values will be skipped
        X_copy = X_copy.apply(lambda x: self.data_imputation(x, dataset_variable_description, epi_prompt) if x.isna().sum() > 0 else x, axis=1)

        return X_copy


    def __gpt_api_call__(self, model, system_prompt, user_prompt, temperature=0.2, max_tokens=256, frequency_penalty=0.0):
        """
        Make an API call to OpenAI's GPT-4.

        Parameters:
        - model: The model to be used (e.g., 'gpt-4').
        - system_prompt: The system prompt or context for the conversation.
        - user_prompt: The user's input or question.
        - temperature: Controls randomness (lower is more deterministic).
        - max_tokens: Maximum length of the response.
        - frequency_penalty: Discourages repetition.

        Returns:
        - The content of the response from GPT-4.
        """

        client = OpenAI(api_key=self.OPENAI_API_KEY)

        # Construct the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Perform the API call
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty
            )
            # Return the response content
            return response.choices[0].message.content
        except Exception as e:
            if e.code == 429:
                print("Too many requests. Waiting 2 seconds...")
                time.sleep(2)
                self.__gpt_api_call__(model, system_prompt, user_prompt, temperature, max_tokens, frequency_penalty)


    def expert_prompt_initialization(self, dataset_description: str):
        """
        Expert Prompt Initialization (EPI) module
        """

        # First API Call to generate the Expert Prompt Initialization (epi)
        model = "gpt-4"
        system_prompt = \
            """
            I am going to give you a description of a dataset. Please read it and then tell me which hypothetical persona would be the best domain expert on the content of the data set if I had questions about specific variables, attributes or properties.\r\nI don't need a data scientist or machine learning expert, and I don't have questions about the analysis of the data, but about specific attributes and values.\r\nPlease do not give me a list. Just give me a detailed description of a (single) person who really knows a lot about the field in which the dataset was generated.\r\nDo not use your knowledge about the author of the data record as a guide. Do not mention the dataset or anything about it. Do not explain what you do. Just give the description and be concise. No Intro like 'An expert would be'.
            """
        user_prompt_prefix = \
            """
            Here is the description of the dataset:\r\n\r\n
            """
        user_prompt_suffix = \
            """
            \r\n\r\n\r\n\r\nRemember: Do not mention the dataset in your description. 
            Don\'t explain what you do. Just give me a concise description of a hypthetical person, that would be an expert on this.\r\n
            Formulate this as an instruction like \"You are an ...\".
            """
        user_prompt = user_prompt_prefix + dataset_description + user_prompt_suffix
        epi_max_tokens = 2048

        epi_prompt = self.__gpt_api_call__(model, system_prompt, user_prompt, max_tokens=epi_max_tokens)
        
        if epi_prompt == None or epi_prompt == "":
            self.expert_prompt_initialization(dataset_description)
            return

        return epi_prompt


    def data_imputation(self, X_row: pd.Series, dataset_variables_description: str, epi_prompt: str):
        """
        Data Imputation (DI) module

        ToDo: Extract dataset variables description and dataset row from the dataset
        """

        # Dataset values 2
        # dataset_variables_description = "Variable Name	Role	Type	Demographic	Description	Units	Missing Values\r\nAttribute1	Feature	Categorical		Status of existing checking account		no\r\nAttribute2	Feature	Integer		Duration	months	no\r\nAttribute3	Feature	Categorical		Credit history		no\r\nAttribute4	Feature	Categorical		Purpose		no\r\nAttribute5	Feature	Integer		Credit amount		no\r\nAttribute6	Feature	Categorical		Savings account/bonds		no\r\nAttribute7	Feature	Categorical	Other	Present employment since		no\r\nAttribute8	Feature	Integer		Installment rate in percentage of disposable income		no\r\nAttribute9	Feature	Categorical	Marital Status	Personal status and sex		no\r\nAttribute10	Feature	Categorical		Other debtors / guarantors		no\r\nAttribute11	Feature	Integer		Present residence since		no\r\nAttribute12	Feature	Categorical		Property		no\r\nAttribute13	Feature	Integer	Age	Age	years	no\r\nAttribute14	Feature	Categorical		Other installment plans		no\r\nAttribute15	Feature	Categorical	Other	Housing		no\r\nAttribute16	Feature	Integer		Number of existing credits at this bank		no\r\nAttribute17	Feature	Categorical	Occupation	Job		no\r\nAttribute18	Feature	Integer		Number of people being liable to provide maintenance for		no\r\nAttribute19	Feature	Binary		Telephone		no\r\nAttribute20	Feature	Binary	Other	foreign worker		no\r\nclass	Target	Binary		1 = Good, 2 = Bad		no"
        # missing value in the row must be marked as "<missing>"!
        # dataset_row = "4  12   4  21   1   4   3   3   1  42   3   1   2   1   1   0   0   1   0   <missing>   1   0   1   0   1"

        dataset_row = X_row.to_string(index=False).replace("nan", "<missing>").split("\n")
        dataset_row = [item.strip() for item in dataset_row]
        dataset_row = "\t".join(dataset_row)

        # Second API Call to generate the Data Imputation (di)
        model = "gpt-4"
        system_prompt = epi_prompt + "\r\n\r\n###\r\n\r\nThe Problem: We would like to analyze a data set, but unfortunately this data set has some missing values."
        user_prompt_prefix = \
            """
            The Problem: We would like to analyze a data set, but unfortunately this data set has some missing values.\r\n\r\n
            Here is the description of the variables in the dataset as a table:\r\n\r\n
            """
        user_prompt_suffix = \
            """
            Your Task: Please use your years of experience and the knowledge you have acquired in the course of your work to provide an estimate of 
            what value the missing entry (marked as <missing>) in a column would most likely have.\r\n\r\n
            IMPORTANT: Please do not provide any explanation or clarification. Only answer with the respective value in quotation marks (\"\").\r\n\r\n
            The row with the missing value is:\r\n
            """
        user_prompt = user_prompt_prefix + dataset_variables_description + user_prompt_suffix + dataset_row
        di_max_tokens = 256

        di = self.__gpt_api_call__(model, system_prompt, user_prompt, max_tokens=di_max_tokens)
        print(di)

        if di == None or di == "":
            self.data_imputation(X_row, dataset_variables_description, epi_prompt)
            return

        if di.isnumeric():
            di = float(di)

        X_row = X_row.fillna(di)

        return X_row
