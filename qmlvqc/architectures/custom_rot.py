import pennylane as qml

from .base_architecture import Architecture


class CustomRot(Architecture):
    def weight_shape(self, n_qubits: int) -> tuple:
        # 2 Rot gates per qubit, each with 3 learnable params (φ, θ, ω)
        return (n_qubits, 2, 3)

    def apply(self, weights, wires) -> None:
        for i, wire in enumerate(wires):
            qml.Rot(weights[i, 0, 0], weights[i, 0, 1], weights[i, 0, 2], wires=wire)
            qml.Hadamard(wires=wire)
            qml.Rot(weights[i, 1, 0], weights[i, 1, 1], weights[i, 1, 2], wires=wire)
