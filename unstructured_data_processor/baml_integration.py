# baml_integration.py
import json
from typing import List, Dict, Any
from baml_client import baml as b
from baml_client.types import Entity

class EntityExtractor:
    def __init__(self, llm, rate_limiter):
        self.llm = llm
        self.rate_limiter = rate_limiter
        self.custom_prompt = None
        self.entity_types = ["Person", "Organization", "Location", "Event", "Concept"]
        self.entity_id_counter = 0
        self.entity_id_map = {}

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
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Your task is to identify the entities and relations requested with the user prompt from a given text. You must generate the output in a JSON format containing a list with JSON objects.

Analyze the following text and extract entities. Identify entities of the following types: {", ".join(self.entity_types)}.
For each entity, provide an ID, type, name, and any relevant metadata such as relationships, attributes, and context.

Text: {text}

Example:
Text: "John Doe, a software engineer at OpenAI, lives in San Francisco."
Output: [
    {{"id": "PERSON_0", "type": "Person", "name": "John Doe", "metadata": {{"occupation": "software engineer", "organization": "OpenAI", "location": "San Francisco"}}}},
    {{"id": "ORGANIZATION_1", "type": "Organization", "name": "OpenAI", "metadata": {{}}}},
    {{"id": "LOCATION_2", "type": "Location", "name": "San Francisco", "metadata": {{}}}}
]

Now, analyze the following text and extract entities:
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
                for entity in entities:
                    entity_id = self._generate_unique_entity_id(entity)
                    entity['id'] = entity_id
                return entities
            except json.JSONDecodeError:
                print("Error: Invalid JSON in entity extraction response")
                return []
        else:
            print("Error: No valid JSON array found in entity extraction response")
            return []

    def _generate_unique_entity_id(self, entity: Dict[str, Any]) -> str:
        entity_key = (entity['type'], entity['name'])
        if entity_key in self.entity_id_map:
            return self.entity_id_map[entity_key]
        else:
            unique_id = f"{entity['type'].upper()}_{self.entity_id_counter}"
            self.entity_id_counter += 1
            self.entity_id_map[entity_key] = unique_id
            return unique_id

class BAMLRelationshipExtractor:
    @baml.function
    def extract_relationships(self, text: str, entities):
        entity_descriptions = "\n".join([f"{e['id']}: {e['name']} ({e['type']})" for e in entities])
        prompt = f"""
        Extract relationships between entities. Consider types: works_for, located_in, part_of, affiliated_with.
        Text: {text}
        Entities: {entity_descriptions}
        Output JSON format with fields: source, target, type, metadata.
        """
        return baml.run(prompt)
