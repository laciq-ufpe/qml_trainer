from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        ...

    @abstractmethod
    def to_label(self, raw: np.ndarray) -> np.ndarray:
        ...
