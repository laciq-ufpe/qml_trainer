from abc import ABC, abstractmethod

import numpy as np


class Encoding(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, n_qubits: int) -> None:
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def apply(self, x: np.ndarray, wires) -> None:
        ...


class JaxEncoding(Encoding):
    """Marker base class for JAX-traceable encodings."""
