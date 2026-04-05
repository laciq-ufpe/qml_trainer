from .trainers import Trainer, VQC, VQCJax
from .architectures import Architecture, StronglyEntangled, BasicEntangler
from .losses import Loss, BCE, BCEJax
from .encodings import Encoding, AngleEncoding, PhaseEncoding, PhaseEncodingJax
from .projectors import DimensionalityProjector, PCAProjector
from .data import load_wisconsin

__all__ = [
    "Trainer",
    "VQC",
    "VQCJax",
    "Architecture",
    "StronglyEntangled",
    "BasicEntangler",
    "Loss",
    "BCE",
    "BCEJax",
    "Encoding",
    "AngleEncoding",
    "PhaseEncoding",
    "PhaseEncodingJax",
    "DimensionalityProjector",
    "PCAProjector",
    "load_wisconsin",
]
