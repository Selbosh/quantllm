import openai
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd

class LLMElicitor:
    def __init__(self,
                 prompts: dict = {},
                 method: str = 'shelf',
                 format: str = 'dist',
                 context: str = '',
                 log_filepath: Path = None,
                 debug: bool = False):
        self.prompts = prompts
        self.method = method
        self.format = format
        self.context = context
        self.log_filepath = log_filepath
        self.debug = debug
        
        self.log = {
            "model": self.model,
            "role": self.role,
            "prompts": self.prompts,
            "n_requests": {
                "epi": 0,
                "elicit": 0
            },
            "n_tokens": {
                key: {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "n_total_tokens": 0,
                    "n_input_tokens_tiktoken": 0
                } for key in ['epi', 'elicit', 'total']
            }
        }
        self.__save_log()
    
    def __save_log(self):
        """
        Save the log to a file.
        
        Args
            - `log_filepath`: The path to the log file.
        """
        if self.log_filepath is None:
            return
        with open(self.log_filepath, "w") as f:
            json.dump(self.log, f, indent=2)
            
    def elicit_prior(self, X: pd.DataFrame):
        """
        Elicit a prior distribution from the LLM.
        """
        epi_prompt = ""
        if self.role == "expert":
            epi_prompt = self.__expert_prompt_initialization(self.dataset_description)
        
    def fetch_log(self):
        return self.log
    
    def __expert_prompt_initialization(self, dataset_description: str):
        """
        Expert Prompt Initialization (EPI) module
        """
        if self.debug:
            print("Starting Expert Prompt Initialization (EPI) module...")
        # First API Call to generate the Expert Prompt Initialization (epi)
        system_prompt = self.prompts['expert_prompt_initialization']['system_prompt']
        user_prompt_prefix = self.prompts['expert_prompt_initialization']['user_prompt_prefix']
        user_prompt_suffix = self.prompts['expert_prompt_initialization']['user_prompt_suffix']
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
    
    def __parser(llm_response, family, target_param):
        """
        For extracting numeric values from natural language responses.
        """
        pass
            
    def __chat_api_call(self, model, messages, temperature=0.2, max_tokens=256, frequency_penalty=0.0):
        load_dotenv()
        
        if self.debug:
            print(f"- Starting API call to {model}")
            
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
