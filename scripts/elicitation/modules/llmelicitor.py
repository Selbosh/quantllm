import openai
import json
from pathlib import Path
from dotenv import load_dotenv
import os

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
