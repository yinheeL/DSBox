import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "your api"
class LLM:
    def __init__(self, model='gpt-4o-mini'):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model

    def generate(self, messages, temperature):
        ret = self.client.chat.completions.create(
            messages=messages,
            temperature=temperature,
            model=self.model,
        )
        return ret.choices[0].message.content