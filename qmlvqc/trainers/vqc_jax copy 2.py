import logging
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pennylane as qml

from ..architectures.base_architecture import JaxArchitecture
from ..encodings import AngleEncoding
from ..encodings.base_encoding import JaxEncoding
from ..losses.base_loss import JaxLoss
from .base_trainer import Trainer

logger = logging.getLogger(__name__)


class VQCJax(Trainer):
    """
    VQC trainer using JAX autodiff + Optax optimization.

    Requires JaxArchitecture, JaxEncoding, and JaxLoss — all must be
    broadcast-based and JAX-traceable. The circuit uses adjoint differentiation
    and the entire update step (batched forward, loss, backward, optimizer) is
    compiled into a single XLA program via jax.jit.
    """

    def __init__(self, n_qubits, architecture, loss, encoding=None, seed=0):
        if not isinstance(architecture, JaxArchitecture):
            raise TypeError(
                f"{type(architecture).__name__} is not a JaxArchitecture. "
                "Use a broadcast-based architecture (e.g. CustomRotJax) with VQCJax."
            )
        if not isinstance(loss, JaxLoss):
            raise TypeError(
                f"{type(loss).__name__} is not a JaxLoss. "
                "Use a JAX-compatible loss (e.g. BCEJax) with VQCJax."
            )
        encoding = encoding if encoding is not None else AngleEncoding()
        if not isinstance(encoding, JaxEncoding):
            raise TypeError(
                f"{type(encoding).__name__} is not a JaxEncoding. "
                "Use a broadcast-based encoding (e.g. AngleEncoding, PhaseEncodingJax) with VQCJax."
            )
        self.n_qubits = n_qubits
        self.architecture = architecture
        self.loss = loss
        self.encoding = encoding
        self._seed = seed
        self._dev = qml.device("lightning.qubit", wires=n_qubits)
        self._weights = None
        self._bias = None
        self.history_: list[float] = []

    def _build_circuit(self):
        @qml.qnode(self._dev, interface="jax", diff_method="adjoint")
        def circuit(x, weights):
            self.encoding.apply(x, wires=range(self.n_qubits))
            self.architecture.apply(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        # vmap only — jit is applied over the full update_step so the entire
        # forward pass, loss, adjoint backward, and optimizer update compile
        # as a single XLA program.
        return jax.vmap(circuit, in_axes=(0, None))

    def fit(self, X_train, y_train, epochs=50, lr=0.01, batch_size=16):
        t_fit_start = time.perf_counter()

        logger.info("Applying encoding (%s)...", type(self.encoding).__name__)
        self.encoding.fit(X_train, self.n_qubits)
        X_enc = self.encoding.transform(X_train)

        # 🔥 ADICIONE AQUI
        X_enc = jnp.array(X_enc)
        y_train = jnp.array(y_train)

        weight_shape = self.architecture.weight_shape(self.n_qubits)
        key = jax.random.PRNGKey(self._seed)
        key, w_key = jax.random.split(key)
        self._weights = jax.random.uniform(w_key, shape=weight_shape, minval=0.0, maxval=2 * jnp.pi)
        self._bias = jnp.array(0.0)

        logger.info("Building update_step (jit+vmap+adjoint) — first call will trigger XLA compilation...")
        batched_circuit = self._build_circuit()
        optimizer = optax.adam(learning_rate=lr)
        opt_state = optimizer.init((self._weights, self._bias))

        def cost(params, X_batch, y_batch, mask):
            w, b = params
            preds = batched_circuit(X_batch, w) + b
            return self.loss(preds, y_batch, mask)

        @jax.jit
        def update_step(params, opt_state, X_batch, y_batch, mask):
            print(" Compilando")
            loss_val, grads = jax.value_and_grad(cost)(params, X_batch, y_batch, mask)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        params = (self._weights, self._bias)
        n_samples = len(X_enc)
        self.history_ = []

        logger.info(
            "Starting training loop | epochs=%d  lr=%g  batch_size=%d  n_samples=%d",
            epochs, lr, batch_size, n_samples,
        )
        t_train_start = time.perf_counter()

        for epoch in range(epochs):
            t_epoch_start = time.perf_counter()
            perm = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
            X_shuffled = X_enc[perm]
            y_shuffled = y_train[perm]
            epoch_loss, n_batches = 0.0, 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_b = X_shuffled[start:end]
                y_b = y_shuffled[start:end]

                real = X_b.shape[0]
                pad = batch_size - real

                # 🔥 usar jnp.pad (não numpy)
                X_b = jnp.pad(X_b, ((0, pad), (0, 0)))
                y_b = jnp.pad(y_b, (0, pad))

                # 🔥 máscara JAX-friendly (SEM lista Python)
                mask = (jnp.arange(batch_size) < real).astype(jnp.float32)

                # 🔥 NÃO recriar jnp.array aqui
                X_batch = X_b
                y_batch = y_b
                params, opt_state, loss_val = update_step(params, opt_state, X_batch, y_batch, mask)
                epoch_loss += float(loss_val)
                n_batches += 1

            mean_loss = epoch_loss / n_batches
            self.history_.append(mean_loss)
            epoch_time = time.perf_counter() - t_epoch_start
            print(f"Epoch {epoch + 1:3d}/{epochs} | loss: {mean_loss:.4f} | time: {epoch_time:.2f}s")

        self._weights, self._bias = params
        training_time = time.perf_counter() - t_train_start
        total_time = time.perf_counter() - t_fit_start
        logger.info("Training complete | training time: %.2fs | total fit time: %.2fs", training_time, total_time)

        self._circuit = jax.jit(batched_circuit)

    def predict(self, X):
        X_enc = jnp.array(self.encoding.transform(X))
        raw = np.array(self._circuit(X_enc, self._weights) + self._bias)
        return self.loss.to_label(raw)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y.astype(int)))
