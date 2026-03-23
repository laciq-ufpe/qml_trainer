import pennylane as qml

from .base_architecture import JaxArchitecture


class CustomRotJax(JaxArchitecture):
    """
    JAX+GPU-compatible variant of CustomRot.

    Circuit per qubit: Rot(φ,θ,ω) → H → Rot(φ,θ,ω)

    Uses qml.broadcast instead of a Python loop so the circuit is
    JAX-traceable and compatible with jax.jit and qml.batch_params.
    Only compatible with VQCJax + BCEJax.
    """

    def weight_shape(self, n_qubits: int) -> tuple:
        # 2 Rot gates per qubit, each with 3 learnable params (φ, θ, ω)
        return (n_qubits, 2, 3)

    def apply(self, weights, wires) -> None:
        for i, wire in enumerate(wires):
            qml.Rot(weights[i, 0, 0], weights[i, 0, 1], weights[i, 0, 2], wires=wire)
        for wire in wires:
            qml.Hadamard(wires=wire)
        for i, wire in enumerate(wires):
            qml.Rot(weights[i, 1, 0], weights[i, 1, 1], weights[i, 1, 2], wires=wire)
