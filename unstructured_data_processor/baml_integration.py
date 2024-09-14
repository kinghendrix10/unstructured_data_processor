#unstructured_data_processor/baml_integration.py
import baml

class BAMLEntityExtractor:
    @baml.function
    def extract_entities(text: str):
        """
        Extract entities from the given text using BAML.
        """
        prompt = f"""
        You are tasked with extracting entities from the following text. Identify entities of the types: Person, Organization, Location, Event, and Concept.
        
        Text: {text}
        
        Output the entities in JSON format with fields: id, type, name, and metadata.
        """
        return baml.run(prompt)

class BAMLRelationshipExtractor:
    @baml.function
    def extract_relationships(text: str, entities):
        """
        Extract relationships between entities using BAML.
        """
        entity_descriptions = "\n".join([f"{e['id']}: {e['name']} ({e['type']})" for e in entities])
        prompt = f"""
        Analyze the text and extract relationships between the provided entities. Consider relationship types such as: works_for, located_in, part_of, affiliated_with.
        
        Text: {text}
        
        Entities:
        {entity_descriptions}
        
        Output the relationships in JSON format with fields: source, target, type, and metadata.
        """
        return baml.run(prompt)
