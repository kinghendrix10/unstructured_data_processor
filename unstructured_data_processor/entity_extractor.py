# unstructured_data_processor/entity_extractor.py
import json
import logging
from typing import List, Dict, Any
from .rate_limiter import RateLimiter

class EntityExtractor:
    def __init__(self, llm, rate_limiter: RateLimiter):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.custom_prompt = None
        self.entity_types = ["Person", "Organization", "Location", "Event", "Concept"]
        self.entity_id_counter = {}

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
        Analyze the following text and extract entities.
        Identify entities of the following types: {", ".join(self.entity_types)}.
        For each entity, provide a type, name, and any relevant metadata.

        Text: {text}

        Output the result as a JSON array of entities. Do not assign IDs to the entities.
        """

    def generate_unique_id(self, entity_type: str, entity_name: str) -> str:
        base_id = f"{entity_type.upper()}_{entity_name.replace(' ', '_').upper()}"
        if base_id not in self.entity_id_counter:
            self.entity_id_counter[base_id] = 0
            return base_id
        else:
            self.entity_id_counter[base_id] += 1
            return f"{base_id}_{self.entity_id_counter[base_id]}"

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        try:
            prompt = self.generate_prompt(text)
            response = await self.rate_limiter.execute(self.llm.acomplete, prompt)
            
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                self.rate_limiter.update_token_count(response.usage.total_tokens)
            
            start = response.text.find('[')
            end = response.text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response.text[start:end]
                try:
                    entities = json.loads(json_str)
                    if not isinstance(entities, list):
                        raise ValueError("Extracted entities are not in list format")
                    # Assign unique IDs to entities
                    for entity in entities:
                        entity['id'] = self.generate_unique_id(entity['type'], entity['name'])
                    return entities
                except json.JSONDecodeError:
                    logging.error("Error: Invalid JSON in entity extraction response")
            else:
                logging.error("Error: No valid JSON array found in entity extraction response")
        except Exception as e:
            logging.error(f"Error in entity extraction: {e}")
        
        return []  # Return an empty list if anything goes wrong
