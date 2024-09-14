# unstructured_data_processor/preprocessor.py
import re
from typing import List, Callable, Dict, Any, Set
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse

class Preprocessor:
    def __init__(self):
        self.preprocessing_steps = [
            self.clean_text,
            self.filter_content,
            self.summarize_text
        ]

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

    def add_preprocessing_step(self, step_function: Callable[[str], str]):
        self.preprocessing_steps.append(step_function)

    def preprocess_text(self, text: str) -> str:
        for step in self.preprocessing_steps:
            text = step(text)
        return text

    async def process_website(self, url: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        try:
            documents = await self.crawl_website(url, max_pages)
            return documents
        except Exception as e:
            logging.error(f"Error processing website {url}: {str(e)}")
            return []  # Return an empty list instead of None

    async def crawl_website(self, start_url: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        visited: Set[str] = set()
        to_visit: List[str] = [start_url]
        documents: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession() as session:
            while to_visit and len(documents) < max_pages:
                url = to_visit.pop(0)
                if url not in visited:
                    visited.add(url)
                    try:
                        async with session.get(url, timeout=10) as response:  # Add timeout
                            if response.status == 200:
                                content = await response.text()
                                soup = BeautifulSoup(content, 'html.parser')
                                text = soup.get_text()
                                processed_text = self.preprocess_text(text)
                                documents.append({"text": processed_text, "metadata": {"source": url}})

                                for link in soup.find_all('a', href=True):
                                    full_url = urljoin(url, link['href'])
                                    if urlparse(full_url).netloc == urlparse(start_url).netloc:
                                        to_visit.append(full_url)
                            else:
                                logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    except Exception as e:
                        logging.error(f"Error processing {url}: {str(e)}")

        return documents

    async def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        documents = []
        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            soup = BeautifulSoup(content, 'html.parser')
                            text = soup.get_text()
                            processed_text = self.preprocess_text(text)
                            documents.append({"text": processed_text, "metadata": {"source": url}})
                        else:
                            logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
                except Exception as e:
                    logging.error(f"Error processing URL {url}: {str(e)}")
        return documents
