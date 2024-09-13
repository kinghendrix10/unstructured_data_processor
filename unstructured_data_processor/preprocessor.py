# unstructured_data_processor/preprocessor.py
import re
from typing import List, Callable, Dict, Any, Set
import pandas as pd
import docx
from bs4 import BeautifulSoup
import aiohttp
import asyncio
import os
from pathlib import Path
from urllib.parse import urljoin, urlparse
import chardet

class Preprocessor:
    def __init__(self):
        self.preprocessing_steps = [
            self.clean_text,
            self.filter_content,
            self.summarize_text
        ]
        self.document_parser = None

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

    def set_document_parser(self, parser: Callable[[str], List[str]]):
        self.document_parser = parser

    def preprocess_text(self, text: str) -> str:
        for step in self.preprocessing_steps:
            text = step(text)
        return text

    async def process_input(self, input_data: Any, output_dir: str = None, max_pages: int = 10) -> List[Dict[str, Any]]:
        if isinstance(input_data, str):
            if input_data.startswith('http://') or input_data.startswith('https://'):
                # if not output_dir:
                #     output_dir = "crawled_websites"
                # os.makedirs(output_dir, exist_ok=True)
                return await self.process_website(input_data, output_dir, max_pages)
            elif Path(input_data).is_dir():
                return self.process_directory(input_data)
            else:
                return self.process_file(input_data)
        elif isinstance(input_data, list):
            if all(url.startswith('http://') or url.startswith('https://') for url in input_data):
                return await self.process_urls(input_data)
            else:
                raise ValueError("All items in the list must be URLs")
        else:
            raise ValueError("Input must be a file path, directory path, URL, or list of URLs")

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.xlsx':
            content = self._parse_excel(file_path)
        elif file_extension == '.docx':
            content = self._parse_docx(file_path)
        elif file_extension == '.txt':
            content = self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        processed_text = self.preprocess_text(content)
        return [{"text": processed_text, "metadata": {"source": file_path}}]

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    documents.extend(self.process_file(file_path))
                except ValueError:
                    print(f"Skipping unsupported file: {file_path}")
        return documents

    async def process_website(self, url: str, output_dir: str, max_pages: int = 10) -> List[Dict[str, Any]]:
        saved_files = await self.crawl_website(url, output_dir, max_pages)
        documents = []
        for file in saved_files:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            processed_text = self.preprocess_text(content)
            documents.append({"text": processed_text, "metadata": {"source": file}})
        return documents

    async def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        documents = []
        for url in urls:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            processed_text = self.preprocess_text(text)
            documents.append({"text": processed_text, "metadata": {"source": url}})
        return documents

    async def crawl_website(self, start_url: str, output_dir: str, max_pages: int = 10) -> List[str]:
        visited: Set[str] = set()
        to_visit: List[str] = [start_url]
        saved_files: List[str] = []

        async with aiohttp.ClientSession() as session:
            while to_visit and len(saved_files) < max_pages:
                url = to_visit.pop(0)
                if url not in visited:
                    visited.add(url)
                    try:
                        filepath = await self.save_webpage(url, output_dir, session)
                        saved_files.append(filepath)

                        async with session.get(url) as response:
                            content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        for link in soup.find_all('a', href=True):
                            full_url = urljoin(url, link['href'])
                            if urlparse(full_url).netloc == urlparse(start_url).netloc:
                                to_visit.append(full_url)
                    except Exception as e:
                        print(f"Error processing {url}: {str(e)}")

        return saved_files

    async def save_webpage(self, url: str, output_dir: str, session: aiohttp.ClientSession) -> str:
        async with session.get(url) as response:
            content = await response.text()

        soup = BeautifulSoup(content, 'html.parser')
        title = soup.title.string if soup.title else url.split('/')[-1]
        filename = "".join(c if c.isalnum() else "_" for c in title) + ".html"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        
        print(f"Saved: {filepath}")
        return filepath

    def _parse_excel(self, file_path: str) -> str:
        df = pd.read_excel(file_path)
        return df.to_string(index=False)

    def _parse_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])

    def _parse_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except Exception as e:
            print(f"Error parsing text file {file_path}: {e}")
            return ""
