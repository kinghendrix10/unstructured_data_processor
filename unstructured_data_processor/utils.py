# unstructured_data_processor/utils.py
import logging
from typing import Callable
import asyncio

def setup_logging(log_level: int = logging.INFO, log_format: str = '%(asctime)s - %(levelname)s - %(message)s'):
    logging.basicConfig(level=log_level, format=log_format)

def progress_callback(current: int, total: int):
    print(f"Progress: {current}/{total} ({current/total*100:.2f}%)")

async def run_with_retry(func: Callable, max_retries: int = 3, delay: float = 1.0, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

def chunk_list(lst: list, chunk_size: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def flatten_list(lst: list) -> list:
    """Flatten a nested list."""
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]
