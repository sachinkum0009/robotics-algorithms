from abc import ABC, abstractmethod

from typing import Any


class BaseController(ABC):
    @abstractmethod
    def control(self, *args: Any, **kwargs: Any) -> Any:
        pass
