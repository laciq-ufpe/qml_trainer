import numpy as np
import pennylane as qml

from ..architectures import Architecture
from ..encodings import Encoding
from .base_executor import BaseExecutor


class PennylaneExecutor(BaseExecutor):

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def execute(
        self,
        encoding: Encoding,
        architecture: Architecture,
        x: np.ndarray,
        weights,
        bias,
    ):
        @qml.qnode(self.dev)
        def circuit():
            encoding.apply(x, wires=range(self.n_qubits))
            architecture.apply(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        return circuit() + bias
