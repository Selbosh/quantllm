import os
import openai
from dotenv import load_dotenv

def main():
   load_dotenv()
   base_url = os.getenv("LMSTUDIO_INFERENCE_SERVER_URL")
   if base_url is None:
      raise ValueError("Please set the environment variable LMSTUDIO_INFERENCE_SERVER_URL.")
   
   system_prompt="You are an expert developmental psychologist."
   user_prompt="Give me a standard normal distribution in a single line of Stan code."

   client = openai.OpenAI(base_url=base_url, api_key="not-needed")

   chat = client.chat.completions.create(
      model="tinyllama-1.1b",
      messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                frequency_penalty=0.0,
                max_tokens=1024,
                n=1,
                temperature=0.2
            )
   return chat.choices[0].message.content

if __name__ == "__main__":
    print(main())