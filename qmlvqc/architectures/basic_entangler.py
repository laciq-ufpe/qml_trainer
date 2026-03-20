import pennylane as qml

from .base_architecture import Architecture


class BasicEntangler(Architecture):
    def __init__(self, n_layers: int = 2):
        self.n_layers = n_layers

    def weight_shape(self, n_qubits: int) -> tuple:
        return qml.BasicEntanglerLayers.shape(self.n_layers, n_qubits)

    def apply(self, weights, wires) -> None:
        qml.BasicEntanglerLayers(weights, wires=wires)
