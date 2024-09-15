# baml_integration.py
import baml_client as b

class BAMLEntityExtractor:
    @baml.function
    def extract_entities(self, text: str):
        prompt = f"""
        Extract entities of types: Person, Organization, Location, Event, and Concept from the text.
        Text: {text}
        Output JSON format with fields: id, type, name, metadata.
        """
        return b.run(prompt)

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
        return b.run(prompt)
