from abc import ABC, abstractmethod
from typing import Any


class BaseFilter(ABC):
    """Abstract base class for all filter estimators."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Fit the filter to the data.

        Parameters
        ----------
        *args : Any
            Positional arguments for fitting the filter.
        **kwargs : Any
            Keyword arguments for fitting the filter.

        Returns:
        -------
        BaseFilter
            The fitted filter instance.
        """
        pass