from abc import ABC, abstractmethod

import numpy as np
import pennylane as qml


class Architecture(ABC):
    @abstractmethod
    def weight_shape(self, n_qubits: int) -> tuple:
        ...

    @abstractmethod
    def apply(self, weights, wires) -> None:
        ...


class StronglyEntangled(Architecture):
    def __init__(self, n_layers=2):
        self.n_layers = n_layers

    def weight_shape(self, n_qubits):
        return qml.StronglyEntanglingLayers.shape(self.n_layers, n_qubits)

    def apply(self, weights, wires):
        qml.StronglyEntanglingLayers(weights, wires=wires)
