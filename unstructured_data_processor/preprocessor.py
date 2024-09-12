# unstructured_data_processor/preprocessor.py
import re
from typing import List, Callable
import pandas as pd
import docx
import PyPDF2
from bs4 import BeautifulSoup
import requests
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

    def parse_document(self, document: str) -> List[str]:
        if self.document_parser:
            return self.document_parser(document)
        return [document]  # Default behavior: treat entire document as one chunk

    def parse_documents(self, input_path: str) -> List[str]:
        if input_path.endswith('.xlsx'):
            return self._parse_excel(input_path)
        elif input_path.endswith('.docx'):
            return self._parse_docx(input_path)
        elif input_path.endswith('.pdf'):
            return self._parse_pdf(input_path)
        elif input_path.endswith('.txt'):
            return self._parse_txt(input_path)
        else:
            raise ValueError(f"Unsupported document format: {input_path}")
            
    def parse_url(self, url: str) -> List[str]:
        url_pattern = re.compile(
            r'^(https?|ftp)://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        if re.match(url_pattern, url):
            return self._parse_url(url)
        else:
            raise ValueError(f"Invalid url: {url}")

    def _parse_excel(self, file_path: str) -> List[str]:
        df = pd.read_excel(file_path)
        return df.to_string(index=False).split('\n')

    def _parse_docx(self, file_path: str) -> List[str]:
        doc = docx.Document(file_path)
        return [para.text for para in doc.paragraphs]

    def _parse_pdf(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return text
        except Exception as e:
            print(f"Error parsing PDF file {file_path}: {e}")
            return []

    def _parse_txt(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']
            
            with open(file_path, 'r', encoding=encoding) as file:
                return file.readlines()
        except Exception as e:
            print(f"Error parsing text file {file_path}: {e}")
            return []

    def _parse_url(self, url: str) -> List[str]:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return [soup.get_text()]
