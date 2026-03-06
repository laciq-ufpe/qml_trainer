from .vqc import VQC
from .architectures import Architecture, StronglyEntangled
from .losses import Loss, BCE
from .encodings import Encoding, AngleEncoding, PhaseEncoding
from .projectors import DimensionalityProjector, PCAProjector
from .data import load_wisconsin

__all__ = [
    "VQC",
    "Architecture",
    "StronglyEntangled",
    "Loss",
    "BCE",
    "Encoding",
    "AngleEncoding",
    "PhaseEncoding",
    "DimensionalityProjector",
    "PCAProjector",
    "load_wisconsin",
]
