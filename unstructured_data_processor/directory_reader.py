#unstructured_data_processor/directory_reader.py

import os
import concurrent.futures
from typing import List, Dict, Any, Generator, Optional
import chardet
import PyPDF2
from pathlib import Path

class DirectoryReader:
    def __init__(
        self, 
        input_dir: str, 
        recursive: bool = False, 
        max_workers: int = 1,
        file_types: Optional[List[str]] = None
    ):
        self.input_dir = input_dir
        self.recursive = recursive
        self.max_workers = max_workers
        self.file_types = file_types or ['.txt', '.pdf', '.docx', '.xlsx']

    def _get_file_paths(self) -> List[str]:
        file_paths = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.file_types):
                    file_paths.append(os.path.join(root, file))
            if not self.recursive:
                break
        return file_paths

    def _load_file(self, file_path: str) -> Dict[str, Any]:
        try:
            if file_path.lower().endswith('.pdf'):
                return self._load_pdf(file_path)
            else:
                return self._load_text_file(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return {"text": "", "metadata": {}}

    def _load_pdf(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = [page.extract_text() for page in reader.pages]
            content = "\n".join(text)
            metadata = self._extract_metadata(file_path)
            return {"text": content, "metadata": metadata}
        except Exception as e:
            print(f"Error loading PDF file {file_path}: {e}")
            return {"text": "", "metadata": {}}

    def _load_text_file(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            
            metadata = self._extract_metadata(file_path)
            return {"text": content, "metadata": metadata}
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return {"text": "", "metadata": {}}

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        file_stat = os.stat(file_path)
        return {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": file_stat.st_size,
            "file_extension": Path(file_path).suffix,
            "creation_time": file_stat.st_ctime,
            "modification_time": file_stat.st_mtime
        }

    def load_data(self) -> List[Dict[str, Any]]:
        file_paths = self._get_file_paths()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return [result for result in executor.map(self._load_file, file_paths) if result["text"]]

    def iter_data(self) -> Generator[Dict[str, Any], None, None]:
        file_paths = self._get_file_paths()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for result in executor.map(self._load_file, file_paths):
                if result["text"]:
                    yield result
