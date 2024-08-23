# data_pipeline.py
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

      # ... (previous methods remain unchanged)

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

    # ... (rest of the methods remain unchanged)
