import os
import concurrent.futures
from typing import List, Dict, Any, Generator
from .preprocessor import Preprocessor

class DirectoryReader:
    def __init__(self, input_dir: str, recursive: bool = False, max_workers: int = 1):
        self.input_dir = input_dir
        self.recursive = recursive
        self.max_workers = max_workers
        self.preprocessor = Preprocessor()

    def _get_file_paths(self) -> List[str]:
        file_paths = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_paths.append(os.path.join(root, file))
            if not self.recursive:
                break
        return file_paths

    def _load_file(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            metadata = self._extract_metadata(file_path)
            return {"text": content, "metadata": metadata}
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return {"text": "", "metadata": {}}

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        return {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_extension": os.path.splitext(file_path)[1]
        }

    def load_data(self) -> List[Dict[str, Any]]:
        file_paths = self._get_file_paths()
        data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._load_file, file_paths))
            for result in results:
                if result["text"]:
                    data.append(result)
        return data

    def iter_data(self) -> Generator[Dict[str, Any], None, None]:
        file_paths = self._get_file_paths()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for result in executor.map(self._load_file, file_paths):
                if result["text"]:
                    yield result
