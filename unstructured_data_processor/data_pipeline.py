# unstructured_data_processor/data_pipeline.py
import asyncio
from typing import List, Dict, Any, Callable, Union, Optional
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from .rate_limiter import RateLimiter
from .preprocessor import Preprocessor
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .output_formatter import OutputFormatter
from .llm_factory import LLMFactory
from .directory_reader import DirectoryReader
from .utils import setup_logging, progress_callback, run_with_retry, chunk_list
import logging
import json
from pathlib import Path

class UnstructuredDataProcessor:
    def __init__(self, 
                 llm: Any, 
                 embedding_model: str = "BAAI/bge-small-en-v1.5", 
                 chunk_size: int = 1024, 
                 chunk_overlap: int = 100, 
                 rate_limit: int = 60, 
                 time_period: int = 60, 
                 max_tokens: Optional[int] = None, 
                 verbose: bool = False):
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
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self):
        log_level = logging.DEBUG if self.verbose else logging.INFO
        setup_logging(log_level)

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

    def set_llm_model(self, model_name: str, **kwargs):
        self.llm = LLMFactory.get_model(model_name, **kwargs)
        self.entity_extractor.set_llm(self.llm)
        self.relationship_extractor.set_llm(self.llm)

    def set_batch_size(self, size: int):
        self.batch_size = size

    def set_max_retries(self, retries: int):
        self.max_retries = retries

    def enable_progress_tracking(self, callback: Callable[[int, int], None] = progress_callback):
        self.progress_callback = callback

    def add_pipeline_step(self, step: Callable[[Any], Any], position: int = -1):
        if position == -1:
            self.pipeline_steps.append(step)
        else:
            self.pipeline_steps.insert(position, step)

    async def process_data(self, input_data: Union[str, List[str]]) -> Dict[str, Any]:
        if isinstance(input_data, str):
            if Path(input_data).is_dir():
                return await self._process_directory(input_data)
            else:
                return await self._process_file(input_data)
        elif isinstance(input_data, list):
            return await self._process_urls(input_data)
        else:
            raise ValueError("Input must be a file path, directory path, or list of URLs")

    async def _process_directory(self, input_directory: str) -> Dict[str, Any]:
        reader = DirectoryReader(input_dir=input_directory, recursive=True, max_workers=4)
        documents = reader.load_data()
        return await self._process_documents(documents)

    async def _process_file(self, file_path: str) -> Dict[str, Any]:
        reader = DirectoryReader(input_files=[file_path], max_workers=1)
        documents = reader.load_data()
        return await self._process_documents(documents)

    async def _process_urls(self, urls: List[str]) -> Dict[str, Any]:
        documents = []
        for url in urls:
            parsed_content = self.preprocessor.parse_url(url)
            for content in parsed_content:
                processed_text = self.preprocessor.preprocess_text(content)
                documents.append({"text": processed_text, "metadata": {"source": url}})
        return await self._process_documents(documents)

    async def _process_file(self, file_path: str) -> Dict[str, Any]:
        # Create a temporary directory to hold the single file
        temp_dir = os.path.dirname(file_path)
        
        # Initialize DirectoryReader with the directory containing the file
        reader = DirectoryReader(input_dir=temp_dir, recursive=False, max_workers=1)
        
        # Filter to only process the specific file
        reader.input_files = [file_path]
        
        documents = reader.load_data()
        return await self._process_documents(documents)

    async def process_data(self, input_data: Union[str, List[str]], output_dir: str = None, max_pages: int = 10) -> Dict[str, Any]:
        preprocessed_data = await self.preprocessor.process_input(input_data, output_dir, max_pages)
        documents = [Document(text=item['text'], metadata=item['metadata']) for item in preprocessed_data]
        
        splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        all_data = {"entities": [], "relationships": []}
        entity_id_map = {}
        
        for i, batch in enumerate(chunk_list(nodes, self.batch_size)):
            batch_data = await run_with_retry(self._process_batch, self.max_retries, 1.0, batch, entity_id_map)
            all_data["entities"].extend(batch_data["entities"])
            all_data["relationships"].extend(batch_data["relationships"])
            
            if hasattr(self, 'progress_callback'):
                self.progress_callback((i + 1) * self.batch_size, len(nodes))

        final_data = self._finalize_data(all_data)
        return self.output_formatter.format_output(final_data)

    async def _process_batch(self, nodes: List[Document], entity_id_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        batch_data = {"entities": [], "relationships": []}
        for node in nodes:
            try:
                entities = await self.entity_extractor.extract_entities(node.text)
                relationships = await self.relationship_extractor.extract_relationships(node.text, entities)
                
                # Ensure unique entity IDs and merge metadata for duplicates
                for entity in entities:
                    if entity['id'] in entity_id_map:
                        existing_entity = entity_id_map[entity['id']]
                        existing_entity['metadata'].update(entity['metadata'])
                    else:
                        entity_id_map[entity['id']] = entity
                        batch_data["entities"].append(entity)
                
                batch_data["relationships"].extend(relationships)
            except Exception as e:
                logging.error(f"Error processing node: {str(e)}")
        
        return batch_data

    def _finalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for step in self.pipeline_steps:
            data = step(data)
        return data

    async def restructure_documents(self, input_directory: Optional[str] = None, website_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        all_data = {"entities": [], "relationships": []}
        
        if input_directory:
            data = await self.process_data(input_directory)
            all_data["entities"].extend(data["entities"])
            all_data["relationships"].extend(data["relationships"])

        if website_urls:
            data = await self.process_data(website_urls)
            all_data["entities"].extend(data["entities"])
            all_data["relationships"].extend(data["relationships"])

        return all_data
