# unstructured_data_processor/data_pipeline.py
import asyncio
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

    def set_custom_prompt(self, custom_prompt: str):
        self.entity_extractor.set_custom_prompt(custom_prompt)
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

    async def process_documents(self, input_directory: str) -> Dict[str, Any]:
        documents = SimpleDirectoryReader(input_dir=input_directory).load_data()
        processed_docs = []
        for doc in documents:
            processed_text = self.preprocessor.preprocess_text(doc.text)
            processed_docs.append(Document(text=processed_text, metadata=doc.metadata))

        splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(processed_docs)

        all_data = {"entities": [], "relationships": []}
        for i in range(0, len(nodes), self.batch_size):
            batch = nodes[i:i+self.batch_size]
            batch_data = await self._process_batch(batch)
            all_data["entities"].extend(batch_data["entities"])
            all_data["relationships"].extend(batch_data["relationships"])
            if hasattr(self, 'progress_callback'):
                self.progress_callback(i + len(batch), len(nodes))

        final_data = self._finalize_data(all_data)
        return self.output_formatter.format_output(final_data)

    async def _process_batch(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_data = {"entities": [], "relationships": []}
        for node in nodes:
            for _ in range(self.max_retries):
                try:
                    entities = await self.entity_extractor.extract_entities(node.text)
                    relationships = await self.relationship_extractor.extract_relationships(node.text, entities)
                    batch_data["entities"].extend(entities)
                    batch_data["relationships"].extend(relationships)
                    break
                except Exception as e:
                    logging.error(f"Error processing node: {e}")
                    if _ == self.max_retries - 1:
                        logging.error(f"Max retries reached for node. Skipping.")
        return batch_data

    def _finalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.pipeline_steps:
            data = step(data)
        return data

    async def restructure_documents(self, input_directory: str) -> Dict[str, Any]:
        data = await self.process_documents(input_directory)
        # Ensure we're returning a dictionary, not a formatted string
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # If it's not valid JSON, return the original structure
                return {"entities": [], "relationships": []}
        return data
