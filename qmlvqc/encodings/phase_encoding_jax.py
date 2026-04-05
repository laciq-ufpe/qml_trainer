import numpy as np
import pennylane as qml

from ..projectors import DimensionalityProjector, PCAProjector
from .base_encoding import JaxEncoding


class PhaseEncodingJax(JaxEncoding):
    """
    JAX+GPU-compatible variant of PhaseEncoding.

    Uses qml.broadcast instead of a Python loop for Hadamard gates so the
    circuit is JAX-traceable and compatible with jax.jit and qml.batch_params.
    Only compatible with VQCJax.
    """

    def __init__(self, projector: DimensionalityProjector = None):
        self._projector = projector if projector is not None else PCAProjector()

    def fit(self, X: np.ndarray, n_qubits: int) -> None:
        self._projector.fit(X, n_qubits)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._projector.transform(X)

    def apply(self, x: np.ndarray, wires) -> None:
        for wire in wires:
            qml.Hadamard(wires=wire)
        qml.AngleEmbedding(x, wires=wires, rotation='Z')
