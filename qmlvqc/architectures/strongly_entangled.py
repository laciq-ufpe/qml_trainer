import pennylane as qml

from .base_architecture import Architecture, JaxArchitecture


class StronglyEntangled(JaxArchitecture):
    def __init__(self, n_layers: int = 2):
        self.n_layers = n_layers

    def weight_shape(self, n_qubits: int) -> tuple:
        return qml.StronglyEntanglingLayers.shape(self.n_layers, n_qubits)

    def apply(self, weights, wires) -> None:
        qml.StronglyEntanglingLayers(weights, wires=wires)
