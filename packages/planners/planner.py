from abc import ABC, abstractmethod
from typing import Any


class BasePlanner(ABC):
    @abstractmethod
    def plan(self, *args: Any, **kwargs: Any) -> Any:
        pass
