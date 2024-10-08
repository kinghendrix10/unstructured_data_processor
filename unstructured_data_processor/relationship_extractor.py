# unstructured_data_processor/relationship_extractor.py
import json
from typing import List, Dict, Any
from .rate_limiter import RateLimiter

class RelationshipExtractor:
    def __init__(self, llm, rate_limiter: RateLimiter):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.custom_prompt = None
        self.relationship_types = ["works_for", "located_in", "part_of", "affiliated_with"]

    def set_custom_prompt(self, custom_prompt: str):
        self.custom_prompt = custom_prompt

    def set_relationship_types(self, relationship_types: List[str]):
        self.relationship_types = relationship_types

    def set_llm(self, llm):
        self.llm = llm

    def generate_prompt(self, text: str, entities: List[Dict[str, Any]]) -> str:
        entity_str = "\n".join([f"{e['id']}: {e['name']} ({e['type']})" for e in entities])
        if self.custom_prompt:
            return self.custom_prompt.format(text=text, entities=entity_str, relationship_types=", ".join(self.relationship_types))
        return f"""
        Analyze the following text and extract relationships between the given entities. 
        Consider relationship types such as: {", ".join(self.relationship_types)}.
        
        Text: {text}
        
        Entities:
        {entity_str}
        
        For each relationship, provide a source entity ID, target entity ID, relationship type, and any relevant metadata.
        Output the result as a JSON array of relationships.
        """

    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompt = self.generate_prompt(text, entities)
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
                relationships = json.loads(json_str)
                return relationships
            except json.JSONDecodeError:
                print("Error: Invalid JSON in relationship extraction response")
                return []
        else:
            print("Error: No valid JSON array found in relationship extraction response")
            return []
