from typing import Any, Protocol

from .models import LayoutImageResponse


class LayoutProcessor(Protocol):
    async def __call__(self, folder_path: str, **kwargs: Any) -> list[LayoutImageResponse]:
        ...