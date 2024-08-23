# unstructured_data_processor/data_pipeline.py
import asyncio
import json
import re
from typing import List, Dict, Any, Tuple
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from .rate_limiter import RateLimiter

class UnstructuredDataProcessor:
    def __init__(self, llm, embedding_model="BAAI/bge-small-en-v1.5", chunk_size=1024, chunk_overlap=100, 
                 rate_limit: int = 60, time_period: int = 60, max_tokens: int = None):
        self.llm = llm
        Settings.embed_model = FastEmbedEmbedding(model_name=embedding_model)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        self.rate_limiter = RateLimiter(rate_limit, time_period, max_tokens)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'[\xa0\n]', ' ', text)
        text = re.sub(r'([ ]{2,})', ' ', text)
        text = re.sub(r'[-\u2013]', ' ', text)
        return text.strip().rstrip(".,:;")

    @staticmethod
    def filter_content(text: str) -> str:
        text = re.sub(r'Copyright © \d{4}.*', '', text)
        text = re.sub(r'All rights reserved\.', '', text)
        return text

    @staticmethod
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return ' '.join(sentences[:max_sentences])

    def preprocess_documents(self, directory: str) -> List[Document]:
        documents = SimpleDirectoryReader(input_dir=directory).load_data()
        processed_docs = []
        for doc in documents[:20]:
            cleaned_text = self.clean_text(doc.text)
            filtered_text = self.filter_content(cleaned_text)
            summarized_text = self.summarize_text(filtered_text, max_sentences=5)
            processed_docs.append(Document(text=summarized_text, metadata=doc.metadata))

        splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(processed_docs)
        return nodes

    @staticmethod
    def generate_baseline_prompt() -> str:
        return """
        You are an AI reading assistant specialized in restructuring unstructured data into a specified format. Your task is to analyze the given input and extract entities, their relationships, types, and relevant metadata. Follow these instructions carefully:

        1. Input Analysis:
           * Carefully read and understand the provided unstructured data.
           * Identify the main topics, themes, or domains present in the data.

        2. Entity Extraction:
           * Identify all distinct entities mentioned in the text.
           * Assign a unique identifier to each entity (e.g., E1, E2, E3).
           * Determine the type of each entity (e.g., Person, Organization, Location, Event, Concept).
           * Ensure entity descriptions are concise, preferably less than four words.

        3. Relationship Identification:
           * Identify relationships between the extracted entities.
           * Assign a descriptive label to each relationship (e.g., "works_for", "located_in", "part_of", etc).
           * Note the direction of the relationship if applicable.

        4. Metadata Extraction:
           * For each entity and relationship, extract any relevant metadata.
           * Metadata may include dates, quantities, qualifiers, or any additional contextual information.

        5. Output Formatting:
           * Present the restructured data in the following JSON format:

        {
          "entities": [
            {
              "id": "E1",
              "type": "Person",
              "name": "John Doe",
              "metadata": {
                "age": 35,
                "occupation": "Software Engineer"
              }
            }
          ],
          "relationships": [
            {
              "source": "E1",
              "target": "E2",
              "type": "works_for",
              "metadata": {
                "start_date": "2020-01-15"
              }
            }
          ]
        }

        IMPORTANT: Ensure that your response is a single, valid JSON object. Do not include any text before or after the JSON structure.
        """

    @staticmethod
    def clean_json_string(json_str: str) -> str:
        last_brace = json_str.rfind('}')
        if last_brace != -1:
            json_str = json_str[:last_brace+1]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str

    async def extract_structured_data(self, text: str) -> Tuple[Dict[str, Any], bool]:
        prompt = self.generate_baseline_prompt() + f"\n\nText: {text}"
        
        try:
            response = await self.rate_limiter.execute(self.llm.acomplete, prompt)
        except Exception as e:
            print(f"Error: Rate limit or token limit exceeded. {str(e)}")
            return {"entities": [], "relationships": []}, False

        # Update token count (assuming the LLM provides this information)
        if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
            self.rate_limiter.update_token_count(response.usage.total_tokens)

        start = response.text.find('{')
        end = response.text.rfind('}')

        if start != -1 and end != -1:
            json_str = response.text[start:end+1]
            json_str = self.clean_json_string(json_str)
            try:
                structured_data = json.loads(json_str)
                return structured_data, True
            except json.JSONDecodeError as e:
                print(f"Error: JSON decoding failed. {str(e)}")
                return {"entities": [], "relationships": []}, False
        else:
            print("Error: No valid JSON structure found in the output.")
            return {"entities": [], "relationships": []}, False

    async def process_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_data = {"entities": [], "relationships": []}

        async def process_node(node):
            summary = node.text
            structured_data, success = await self.extract_structured_data(summary)
            return structured_data, success, node.metadata.get('file_name', 'unknown')

        tasks = [process_node(node) for node in nodes]
        results = await asyncio.gather(*tasks)

        for structured_data, success, filename in results:
            if success:
                all_data["entities"].extend(structured_data.get("entities", []))
                all_data["relationships"].extend(structured_data.get("relationships", []))
                print(f"Successfully processed: {filename}")
            else:
                print(f"Failed to process: {filename}")

        return all_data

    @staticmethod
    def finalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        unique_entities = {entity["id"]: entity for entity in data["entities"]}
        unique_relationships = {}
        for rel in data["relationships"]:
            key = (rel["source"], rel["target"], rel["type"])
            if key not in unique_relationships:
                unique_relationships[key] = rel

        return {
            "entities": list(unique_entities.values()),
            "relationships": list(unique_relationships.values())
        }

    async def restructure_documents(self, input_directory: str) -> Dict[str, Any]:
        nodes = self.preprocess_documents(input_directory)
        all_data = await self.process_nodes(nodes)
        final_data = self.finalize_data(all_data)
        return final_data
