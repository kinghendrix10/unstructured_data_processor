# unstructured_data_processor/rate_limiter.py
import asyncio
import time
from typing import Callable, Any

class RateLimiter:
    def __init__(self, rate_limit: int, time_period: int = 60, max_tokens: int = None):
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.max_tokens = max_tokens
        self.tokens_used = 0
        self.call_times = []

    async def wait(self):
        current_time = time.time()
        self.call_times = [t for t in self.call_times if current_time - t < self.time_period]

        if len(self.call_times) >= self.rate_limit:
            wait_time = self.time_period - (current_time - self.call_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.call_times.append(time.time())

    def update_token_count(self, tokens: int):
        self.tokens_used += tokens
        if self.max_tokens and self.tokens_used >= self.max_tokens:
            raise Exception("Token limit exceeded")

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        await self.wait()
        result = await func(*args, **kwargs)
        return result
