import pennylane as qml


class StronglyEntangled:
    def __init__(self, n_layers=2):
        self.n_layers = n_layers

    def weight_shape(self, n_qubits):
        return qml.StronglyEntanglingLayers.shape(self.n_layers, n_qubits)

    def apply(self, weights, wires):
        qml.StronglyEntanglingLayers(weights, wires=wires)
