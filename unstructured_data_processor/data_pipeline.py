# unstructured_data_processor/data_pipeline.py

import asyncio
import json
import os
from typing import List, Dict, Any, Callable
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from .rate_limiter import RateLimiter
from .preprocessor import Preprocessor
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .output_formatter import OutputFormatter
from .llm_factory import LLMFactory
import logging

class UnstructuredDataProcessor:
    def __init__(self, llm, embedding_model="BAAI/bge-small-en-v1.5", chunk_size=1024, chunk_overlap=100, 
                 rate_limit: int = 60, time_period: int = 60, max_tokens: int = None):
        self.llm = llm
        Settings.embed_model = FastEmbedEmbedding(model_name=embedding_model)
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        self.rate_limiter = RateLimiter(rate_limit, time_period, max_tokens)
        self.preprocessor = Preprocessor()
        self.entity_extractor = EntityExtractor(self.llm, self.rate_limiter)
        self.relationship_extractor = RelationshipExtractor(self.llm, self.rate_limiter)
        self.output_formatter = OutputFormatter()
        self.pipeline_steps = []
        self.batch_size = 10
        self.max_retries = 3

    def set_custom_prompt_entity(self, custom_prompt: str):
        self.entity_extractor.set_custom_prompt(custom_prompt)

    def set_custom_prompt_relationship(self, custom_prompt: str):
        self.relationship_extractor.set_custom_prompt(custom_prompt)

    def set_entity_types(self, entity_types: List[str]):
        self.entity_extractor.set_entity_types(entity_types)

    def set_relationship_types(self, relationship_types: List[str]):
        self.relationship_extractor.set_relationship_types(relationship_types)

    def add_preprocessing_step(self, step_function: Callable[[str], str]):
        self.preprocessor.add_preprocessing_step(step_function)

    def set_document_parser(self, parser: Callable[[str], List[str]]):
        self.preprocessor.set_document_parser(parser)

    def update_rate_limit(self, new_rate: int, new_period: int):
        self.rate_limiter.update_settings(new_rate, new_period)

    def set_output_format(self, format: str):
        self.output_formatter.set_output_format(format)

    def set_logging_config(self, log_level: int, log_format: str):
        logging.basicConfig(level=log_level, format=log_format)

    def set_llm_model(self, model_name: str, **kwargs):
        self.llm = LLMFactory.get_model(model_name, **kwargs)
        self.entity_extractor.set_llm(self.llm)
        self.relationship_extractor.set_llm(self.llm)

    def set_batch_size(self, size: int):
        self.batch_size = size

    def set_max_retries(self, retries: int):
        self.max_retries = retries

    def enable_progress_tracking(self, callback: Callable[[int, int], None]):
        self.progress_callback = callback

    def add_pipeline_step(self, step: Callable[[Any], Any], position: int = -1):
        if position == -1:
            self.pipeline_steps.append(step)
        else:
            self.pipeline_steps.insert(position, step)

    async def process_document(self, document: Document) -> Dict[str, Any]:
        try:
            processed_text = self.preprocessor.preprocess_text(document.text)
            
            entities = await self.entity_extractor.extract_entities(processed_text)
            if not isinstance(entities, list):
                logging.warning(f"Unexpected entity extraction result: {entities}")
                entities = []

            relationships = await self.relationship_extractor.extract_relationships(processed_text, entities)
            if not isinstance(relationships, list):
                logging.warning(f"Unexpected relationship extraction result: {relationships}")
                relationships = []
            
            # Add document metadata to each entity and relationship
            for entity in entities:
                if isinstance(entity, dict):
                    entity['document_metadata'] = document.metadata
                else:
                    logging.warning(f"Unexpected entity format: {entity}")

            for relationship in relationships:
                if isinstance(relationship, dict):
                    relationship['document_metadata'] = document.metadata
                else:
                    logging.warning(f"Unexpected relationship format: {relationship}")
            
            return {
                "entities": entities,
                "relationships": relationships,
                "document_metadata": document.metadata
            }
        except Exception as e:
            logging.error(f"Error processing document: {e}")
            return {
                "entities": [],
                "relationships": [],
                "document_metadata": document.metadata
            }

    async def process_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        results = []

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            if hasattr(self, 'progress_callback'):
                self.progress_callback(i + len(batch), len(documents))

        return results

    async def _process_batch(self, documents: List[Document]) -> List[Dict[str, Any]]:
        tasks = [self.process_document(doc) for doc in documents]
        return await asyncio.gather(*tasks)

    def merge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_entities = []
        all_relationships = []
        all_document_metadata = []
        
        for result in results:
            all_entities.extend(result.get("entities", []))
            all_relationships.extend(result.get("relationships", []))
            all_document_metadata.append(result.get("document_metadata", {}))

        # Merge entities with the same ID
        merged_entities = {}
        for entity in all_entities:
            if entity["id"] not in merged_entities:
                merged_entities[entity["id"]] = entity
                merged_entities[entity["id"]]["document_metadata"] = set()
            # Merge metadata
            merged_entities[entity["id"]]["metadata"].update(entity.get("metadata", {}))
            # Add document metadata
            if isinstance(entity.get("document_metadata"), dict):
                merged_entities[entity["id"]]["document_metadata"].add(
                    frozenset(entity["document_metadata"].items())
                )
            else:
                logging.warning(f"Unexpected document_metadata format for entity {entity['id']}")

        # Remove duplicate relationships while preserving document metadata
        unique_relationships = {}
        for r in all_relationships:
            key = (r["source"], r["target"], r["type"])
            if key not in unique_relationships:
                unique_relationships[key] = r
                unique_relationships[key]["document_metadata"] = set()
            # Add document metadata
            if isinstance(r.get("document_metadata"), dict):
                unique_relationships[key]["document_metadata"].add(
                    frozenset(r["document_metadata"].items())
                )
            else:
                logging.warning(f"Unexpected document_metadata format for relationship {key}")

        # Convert sets back to lists of dicts for JSON serialization
        for entity in merged_entities.values():
            entity["document_metadata"] = [dict(metadata) for metadata in entity["document_metadata"]]
        for relationship in unique_relationships.values():
            relationship["document_metadata"] = [dict(metadata) for metadata in relationship["document_metadata"]]

        return {
            "entities": list(merged_entities.values()),
            "relationships": list(unique_relationships.values()),
            "document_metadata": all_document_metadata
        }

    def _finalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.pipeline_steps:
            data = step(data)
        return data

    async def restructure_documents(self, input_directory: str) -> Dict[str, Any]:
        if not os.path.exists(input_directory):
            logging.error(f"Input directory does not exist: {input_directory}")
            return {"entities": [], "relationships": [], "document_metadata": []}

        try:
            # Load documents from the directory
            documents = SimpleDirectoryReader(input_dir=input_directory).load_data()
            
            if not documents:
                logging.warning(f"No documents found in directory: {input_directory}")
                return {"entities": [], "relationships": [], "document_metadata": []}

            # Process documents in batches
            results = await self.process_documents(documents)

            # Merge results from all documents
            merged_data = self.merge_results(results)
            final_data = self._finalize_data(merged_data)

            return self.output_formatter.format_output(final_data)

        except Exception as e:
            logging.error(f"Error processing documents: {e}")
            return {"entities": [], "relationships": [], "document_metadata": []}
