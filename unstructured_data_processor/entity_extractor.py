# unstructured_data_processor/entity_extractor.py
import json
from typing import List, Dict, Any
from .rate_limiter import RateLimiter

class EntityExtractor:
    def __init__(self, llm, rate_limiter: RateLimiter):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.custom_prompt = None
        self.entity_types = ["Person", "Organization", "Location", "Event", "Concept"]

    def set_custom_prompt(self, custom_prompt: str):
        self.custom_prompt = custom_prompt

    def set_entity_types(self, entity_types: List[str]):
        self.entity_types = entity_types

    def set_llm(self, llm):
        self.llm = llm

    def generate_prompt(self, text: str) -> str:
        if self.custom_prompt:
            return self.custom_prompt.format(text=text, entity_types=", ".join(self.entity_types))
        return f"""
        Analyze the following text and extract entities. Identify entities of the following types: {", ".join(self.entity_types)}.
        For each entity, provide an ID, type, name, and any relevant metadata.
        
        Text: {text}
        
        Output the result as a JSON array of entities.
        """

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        prompt = self.generate_prompt(text)
        response = await self.rate_limiter.execute(self.llm.acomplete, prompt)
        
        # Update token count if available
        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
            self.rate_limiter.update_token_count(response.usage.total_tokens)
        
        # Extract JSON from response
        start = response.text.find('[')
        end = response.text.rfind(']') + 1
        if start != -1 and end != -1:
            json_str = response.text[start:end]
            try:
                entities = json.loads(json_str)
                return entities
            except json.JSONDecodeError:
                print("Error: Invalid JSON in entity extraction response")
                return []
        else:
            print("Error: No valid JSON array found in entity extraction response")
            return []
