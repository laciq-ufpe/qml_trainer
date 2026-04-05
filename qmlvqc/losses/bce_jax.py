import jax.numpy as jnp
import numpy as np

from .base_loss import JaxLoss


class BCEJax(JaxLoss):
    """JAX-backed BCE loss. Only compatible with VQCJax. Do NOT use with VQC."""

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def __call__(self, predictions, targets, mask=None):
        if not isinstance(predictions, jnp.ndarray):
            raise TypeError("BCEJax expects JAX arrays. Use BCE with VQC.")
        probs = jnp.clip((predictions + 1.0) / 2.0, self.epsilon, 1 - self.epsilon)
        per_sample = -(targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs))
        if mask is not None:
            return jnp.sum(per_sample * mask) / jnp.sum(mask)
        return jnp.mean(per_sample)

    def to_label(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw) >= 0.0).astype(int)
