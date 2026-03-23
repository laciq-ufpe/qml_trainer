from abc import ABC, abstractmethod


class Architecture(ABC):
    @abstractmethod
    def weight_shape(self, n_qubits: int) -> tuple:
        ...

    @abstractmethod
    def apply(self, weights, wires) -> None:
        ...


class JaxArchitecture(Architecture):
    """Marker base class for JAX-traceable architectures."""
