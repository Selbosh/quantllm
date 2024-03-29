import openai
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import re
from collections.abc import Iterable
import tiktoken

class LLMElicitor:
    def __init__(self,
                 prompts: dict = {},
                 model: str = 'gpt-4',
                 role: str = 'expert',
                 expert_prompt: str | None = "",
                 shelf: bool = False,
                 roulette: bool = False,
                 log_filepath: Path = None,
                 debug: bool = False):
        self.prompts = prompts
        self.expert_prompt = expert_prompt
        self.model = model
        self.role = role
        self.shelf = shelf
        self.roulette = roulette
        self.log_filepath = log_filepath
        self.debug = debug
        self.n_input_tokens_tiktoken = 0 if self.model.startswith("gpt") else None
        self.log = {
            "model": self.model,
            "role": self.role,
            "expert_prompt": self.expert_prompt,
            "shelf": self.shelf,
            "roulette": self.roulette,
            "prompts": self.prompts,
            "n_requests": {
                "epi": 0,
                "pi": 0
            },
            "n_tokens": {
                key: {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "n_total_tokens": 0,
                    "n_input_tokens_tiktoken": 0
                } for key in ['epi', 'pi', 'total']
            }
        }
        self.__save_log()
        
    def elicit(self, target_quantity: str, target_distribution: str | None = None, expert_prompt: str | None = None):
        return self.__prior_elicitation(target_quantity, target_distribution, expert_prompt)
    
    def __prior_elicitation(self, target_quantity: str, target_distribution: str | None = None, expert_prompt: str | None = None):
        """
        Prior elicitation module
        
        Args:
            - `target_quantity`: Text description of phenomenon to be estimated.
            - `target_distribution`: (Optional) parametrized distribution to return.
        """
        # Who is the expert?
        if self.role == 'expert':
            system_prompt = str(expert_prompt or self.expert_prompt or '')
        elif self.role == 'conference':
            system_prompt = self.prompts['elicitation_framework']['conference']
        else:
            system_prompt = self.prompts['non_expert_prompt']

        # What elicitation method will be used?
        # (Decision conferencing is treated as an expert role, rather than a method)
        if self.shelf:
            system_prompt += self.prompts['elicitation_framework']['shelf']
        if self.roulette:
            system_prompt += self.prompts['elicitation_framework']['roulette']
        if not (self.shelf or self.roulette):
            system_prompt += self.prompts['elicitation_framework']['direct']
        system_prompt += self.prompts['prior_elicitation']['system_prompt_suffix']
        user_prompt_prefix = self.prompts['prior_elicitation']['user_prompt_prefix']
        user_prompt_infix = self.prompts['prior_elicitation']['user_prompt_infix']
        if target_distribution is None:
            # Unconstrained selection of parametric distribution (could be tricky to evaluate)
            target_distribution = 'any'
        target_keylist = self.prompts['prior_elicitation']['user_prompt_suffix']['distribution'][target_distribution]
        target_json_str = generate_target_json(target_keylist)
        user_prompt_suffix = self.prompts['prior_elicitation']['user_prompt_suffix']['suffix'] + target_json_str
        user_prompt = user_prompt_prefix + user_prompt_infix + target_quantity + user_prompt_suffix
        
        if 'mistral' in self.model:
            messages = [
                {'role': 'user', 'content': f"{system_prompt}\n\n{user_prompt}"}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        
        # Call the LLM.
        pi_response, pi_n_tokens = self.__chat_api_call(self.model, messages, max_tokens=256, temperature=0.0)
        self.__update_n_tokens(pi_n_tokens) # could maybe move this to inside __chat_api_call
        
        # Parse the result.
        parsed_response = self.__parser(pi_response, target_keylist)
        if parsed_response is None:
            # If the __parser fails, ask the LLM to parse it for us by repeating the user_prompt_suffix. Then run it through the parser again.
            # This is done by passing the LLM's response to role `assistant`, i.e. chat history. Then asking just for formatting as a follow-up.
            # But we should try with regex first since it will be faster than sending another API call.
            if 'mistral' in self.model:
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}" },
                    {"role": "assistant", "content": pi_response },
                    {"role": "user", "content": self.prompts['prior_elicitation']['user_prompt_suffix']['retry'] + target_json_str},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": pi_response},
                    {"role": "user", "content": user_prompt_suffix}
                ]
            pi_response, pi_n_tokens = self.__chat_api_call(self.model, messages, max_tokens=148, temperature=0.0)
            self.__update_n_tokens(pi_n_tokens)
            parsed_response = self.__parser(pi_response, target_keylist) # NB could still fail a second time
        
        if self.debug or parsed_response is None:
            print(f'- LLM response: {pi_response}')
            print(f"- Elicited value: {parsed_response}")
            
        self.__save_log()
        return (target_distribution, parsed_response)
    
    def __update_n_tokens(self, n_tokens):
        for tt in ['input', 'output', 'total']:
            token_type = f'n_{tt}_tokens'
            self.log['n_tokens']['pi'][token_type] += n_tokens[token_type]
            self.log['n_tokens']['total'][token_type] = self.log['n_tokens']['epi'][token_type] + self.log['n_tokens']['pi'][token_type]
        self.log['n_requests']['pi'] += 1
        
    def __parser(self, pi_response: str, target_keys: list[str]) -> dict | None:
        """
        Attempts to parse a JSON object from LLM text output.
        If no (valid) object is found, or values are missing, returns None.
        """
        regex_json = re.compile(r'\{[^{}]*\}')
        re_match = re.search(regex_json, pi_response)
        data = None
        if re_match:
            json_text = re_match.group()
            try:
                dictionary = json.loads(json_text)
                if is_flat_dict(dictionary) and validate_keys(dictionary, target_keys):
                    data = {key: float(dictionary[key]) for key in target_keys} # drop extra items
                elif not is_flat_dict(dictionary):
                    print(f'Dictionary is not flat: {dictionary}')
                else:
                    print(f'Failed to match keys of {dictionary} to targets: {target_keys}')
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM JSON output: {e}\n\n{json_text}")
        else:
            print(f"No valid JSON object found in LLM output:\n{pi_response}")
        return data
    
    def fetch_log(self):
        return self.log
    
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
    
    def expert_prompt_initialization(self, task_description: str):
        """
        Expert prompt initialization (EPI) module
        """
        if self.debug:
            print("Starting Expert Prompt Initialization (EPI) module...")
        # First API Call to generate the Expert Prompt Initialization (epi)
        system_prompt = self.prompts["expert_prompt_initialization"]["system_prompt"]
        user_prompt_prefix = self.prompts["expert_prompt_initialization"]["user_prompt_prefix"]
        user_prompt_suffix = self.prompts["expert_prompt_initialization"]["user_prompt_suffix"]
        user_prompt = user_prompt_prefix + task_description + user_prompt_suffix
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
        self.log["prompts"]["expert_prompt"] = self.expert_prompt = epi_prompt
        self.log["n_tokens"]["epi"] = ept_n_tokens
        self.log["n_tokens"]["total"]["n_input_tokens"] = self.log["n_tokens"]["epi"]["n_input_tokens"] + self.log["n_tokens"]["pi"]["n_input_tokens"]
        self.log["n_tokens"]["total"]["n_output_tokens"] = self.log["n_tokens"]["epi"]["n_output_tokens"] + self.log["n_tokens"]["pi"]["n_output_tokens"]
        self.log["n_tokens"]["total"]["n_total_tokens"] = self.log["n_tokens"]["epi"]["n_total_tokens"] + self.log["n_tokens"]["pi"]["n_total_tokens"]
        self.log["n_requests"]["epi"] += 1

        if self.debug:
            print("Finished Expert Prompt Initialization (EPI) module.")
            print(f"- EPI Prompt: {epi_prompt}")

        if epi_prompt is None:
            raise ValueError("The Expert Prompt Initialization (epi) module returned None.")

        self.__save_log()

        return epi_prompt
    
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
            
def validate_keys(dictionary: dict, expected_keys: list[str] | dict):
    return all(key in dictionary for key in expected_keys)

def is_flat_dict(dictionary: dict) -> bool:
    if not isinstance(dictionary, dict):
        return False
    is_iterable = [isinstance(val, Iterable) and not isinstance(val, str) for val in dictionary.values()]
    return not any(is_iterable)

def generate_target_json(keylist: list[str]) -> str:
    dictionary = {key: 'numeric_value' + str(idx) for idx, key in enumerate(keylist)}
    return json.dumps(dictionary)
