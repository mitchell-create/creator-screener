from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def map_async(
    items: list[Any],
    func: Callable[..., Coroutine],
    concurrency: int = 4,
    desc: str = "Processing",
) -> list[Any]:
    """Run an async function over items with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Any] = [None] * len(items)

    async def _run(idx: int, item: Any) -> None:
        async with semaphore:
            try:
                results[idx] = await func(item)
            except Exception as e:
                logger.warning(f"{desc} failed for item {idx}: {e}")
                results[idx] = None

    tasks = [_run(i, item) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)

    completed = sum(1 for r in results if r is not None)
    logger.info(f"{desc}: {completed}/{len(items)} completed")
    return results


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    """Split a list into chunks of specified size."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
