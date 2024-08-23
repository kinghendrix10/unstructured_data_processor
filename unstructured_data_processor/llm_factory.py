# unstructured_data_processor/llm_factory.py
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

class LLMFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        if model_name.startswith('llama'):
            return Groq(model=model_name, **kwargs)
        elif model_name.startswith('gpt'):
            return OpenAI(model=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
