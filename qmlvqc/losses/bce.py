import numpy as np
import pennylane.numpy as pnp

from .base_loss import Loss


class BCE(Loss):
    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        probs = (predictions + 1.0) / 2.0
        probs = pnp.clip(probs, self.epsilon, 1 - self.epsilon)
        return -pnp.mean(targets * pnp.log(probs) + (1 - targets) * pnp.log(1 - probs))

    def to_label(self, raw: np.ndarray) -> np.ndarray:
        return (raw >= 0.0).astype(int)
