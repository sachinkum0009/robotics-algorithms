from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseVSLAM(ABC):
    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        pass