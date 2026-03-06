from abc import ABC, abstractmethod

import numpy as np
import pennylane as qml

from .projectors import DimensionalityProjector, PCAProjector


class Encoding(ABC):
    @abstractmethod
    def fit(self, X, n_qubits: int) -> None:
        ...

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        ...

    @abstractmethod
    def apply(self, x, wires) -> None:
        ...


class PhaseEncoding(Encoding):
    def __init__(self, projector: DimensionalityProjector = None):
        self._projector = projector if projector is not None else PCAProjector()

    def fit(self, X, n_qubits: int) -> None:
        self._projector.fit(X, n_qubits)

    def transform(self, X) -> np.ndarray:
        return self._projector.transform(X)

    def apply(self, x, wires) -> None:
        for w in wires:
            qml.Hadamard(wires=w)
        qml.AngleEmbedding(x, wires=wires, rotation='Z')


class AngleEncoding(Encoding):
    def __init__(self, projector: DimensionalityProjector = None):
        self._projector = projector if projector is not None else PCAProjector()

    def fit(self, X, n_qubits: int) -> None:
        self._projector.fit(X, n_qubits)

    def transform(self, X) -> np.ndarray:
        return self._projector.transform(X)

    def apply(self, x, wires) -> None:
        qml.AngleEmbedding(x, wires=wires, rotation='Z')
