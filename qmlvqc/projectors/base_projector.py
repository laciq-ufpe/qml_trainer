from abc import ABC, abstractmethod

import numpy as np


class DimensionalityProjector(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, n_qubits: int) -> None:
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...
