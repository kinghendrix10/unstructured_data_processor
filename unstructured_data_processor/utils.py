# unstructured_data_processor/utils.py
import logging

def setup_logging(log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s'):
    logging.basicConfig(level=log_level, format=log_format)

def progress_callback(current: int, total: int):
    print(f"Progress: {current}/{total} ({current/total*100:.2f}%)")
