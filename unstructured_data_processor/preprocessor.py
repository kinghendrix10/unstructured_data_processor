# unstructured_data_processor/preprocessor.py
import re
from typing import List, Callable

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
