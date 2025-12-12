from abc import ABC, abstractmethod
from typing import Any


class BaseVSLAM(ABC):
    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        pass
