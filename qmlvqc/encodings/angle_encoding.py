import numpy as np
import pennylane as qml

from ..projectors import DimensionalityProjector, PCAProjector
from .base_encoding import Encoding


class AngleEncoding(Encoding):
    def __init__(self, projector: DimensionalityProjector = None):
        self._projector = projector if projector is not None else PCAProjector()

    def fit(self, X: np.ndarray, n_qubits: int) -> None:
        self._projector.fit(X, n_qubits)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._projector.transform(X)

    def apply(self, x: np.ndarray, wires) -> None:
        qml.AngleEmbedding(x, wires=wires, rotation='Z')
