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

jax.config.update("jax_log_compiles", True)

logger = logging.getLogger(__name__)

DIAGNOSE_RECOMPILATION = True


def create_batches(X, y, batch_size):
    n = X.shape[0]
    pad = (batch_size - n % batch_size) % batch_size
    X = jnp.pad(X, ((0, pad), (0, 0)))
    y = jnp.pad(y, (0, pad))
    X = X.reshape(-1, batch_size, X.shape[1])
    y = y.reshape(-1, batch_size)
    return X, y


#  Elimina weak_type de todos os arrays do pytree
# antes da primeira chamada ao JIT. O opt_state do Optax (incluindo o
# contador 'count' do Adam) é inicializado com weak_type=True, o que faz
# o JAX tratar cada chamada como uma assinatura de tipos diferente e recompilar.
def _cast_to_concrete(tree):
    def cast(x):
        if not isinstance(x, jnp.ndarray):
            return x
        if x.dtype in (jnp.float32, jnp.float64):
            return x.astype(jnp.float32)
        if x.dtype in (jnp.int32, jnp.int64):
            return x.astype(jnp.int32)
        return x
    return jax.tree_util.tree_map(cast, tree)


class _CompileCounter(logging.Handler):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.last_msg = ""

    def emit(self, record):
        msg = record.getMessage()
        if "Compiling jit(update_step)" in msg:
            self.count += 1
            self.last_msg = msg
            print(f"\nRECOMPILAÇÃO REAL #{self.count}: {msg}\n")


class VQCJax(Trainer):

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

        return jax.vmap(circuit, in_axes=(0, None))

    def fit(self, X_train, y_train, epochs=50, lr=0.01, batch_size=16):
        t_fit_start = time.perf_counter()

        compile_counter = _CompileCounter()
        if DIAGNOSE_RECOMPILATION:
            logging.getLogger("jax._src.interpreters.pxla").addHandler(compile_counter)

        logger.info("Applying encoding (%s)...", type(self.encoding).__name__)
        self.encoding.fit(X_train, self.n_qubits)
        X_enc = self.encoding.transform(X_train)

        # dtype explícito evita weak_type nos arrays de entrada
        X_enc   = jnp.array(X_enc,   dtype=jnp.float32)
        y_train = jnp.array(y_train, dtype=jnp.float32)

        weight_shape = self.architecture.weight_shape(self.n_qubits)
        key = jax.random.PRNGKey(self._seed)
        key, w_key = jax.random.split(key)

        # dtype e minval/maxval explícitos evitam weak_type nos pesos
        self._weights = jax.random.uniform(
            w_key,
            shape=weight_shape,
            dtype=jnp.float32,
            minval=jnp.float32(0.0),
            maxval=jnp.float32(2 * jnp.pi),
        )

        # dtype explícito evita weak_type no bias (principal causa de recompilação)
        self._bias = jnp.array(0.0, dtype=jnp.float32)

        logger.info("Building update_step (jit+vmap+adjoint) — first call will trigger XLA compilation...")
        batched_circuit = self._build_circuit()
        optimizer = optax.adam(learning_rate=lr)

        params    = (self._weights, self._bias)
        opt_state = optimizer.init(params)

        # elimina weak_type do opt_state (ex: contador 'count' do Adam)
        # e dos params antes da primeira chamada JIT
        opt_state = _cast_to_concrete(opt_state)
        params    = _cast_to_concrete(params)

        # máscara criada uma única vez fora do loop para que
        # shape e dtype nunca variem entre chamadas ao JIT
        mask = jnp.ones(batch_size, dtype=jnp.float32)

        def cost(params, X_batch, y_batch, mask):
            w, b = params
            preds = batched_circuit(X_batch, w) + b
            return self.loss(preds, y_batch, mask)

        @jax.jit
        def update_step(params, opt_state, X_batch, y_batch, mask):
            loss_val, grads = jax.value_and_grad(cost)(params, X_batch, y_batch, mask)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

        n_samples     = len(X_enc)
        self.history_ = []

        logger.info(
            "Starting training loop | epochs=%d  lr=%g  batch_size=%d  n_samples=%d",
            epochs, lr, batch_size, n_samples,
        )

        _compiles_before_loop = compile_counter.count
        t_train_start = time.perf_counter()

        for epoch in range(epochs):
            t_epoch_start = time.perf_counter()

            perm       = jax.random.permutation(jax.random.PRNGKey(epoch), n_samples)
            X_shuffled = X_enc[perm]
            y_shuffled = y_train[perm]

            X_batches, y_batches = create_batches(X_shuffled, y_shuffled, batch_size)

            epoch_loss, n_batches = 0.0, 0

            for i in range(X_batches.shape[0]):
                X_batch = X_batches[i]
                y_batch = y_batches[i]

                params, opt_state, loss_val = update_step(
                    params, opt_state, X_batch, y_batch, mask
                )

                loss_val.block_until_ready()

                epoch_loss += float(loss_val)
                n_batches  += 1

            mean_loss  = epoch_loss / n_batches
            self.history_.append(mean_loss)
            epoch_time = time.perf_counter() - t_epoch_start

            recompiles = compile_counter.count - _compiles_before_loop
            print(
                f"Epoch {epoch + 1:3d}/{epochs}"
                f" | loss: {mean_loss:.4f}"
                f" | time: {epoch_time:.2f}s"
                f" | recompilações reais: {recompiles}"
            )

        self._weights, self._bias = params
        training_time = time.perf_counter() - t_train_start
        total_time    = time.perf_counter() - t_fit_start

        if DIAGNOSE_RECOMPILATION:
            recompiles = compile_counter.count - _compiles_before_loop
            print(f"\n{'='*60}")
            print(f"  Total de recompilações de update_step durante treino: {recompiles}")
            if recompiles <= 1:
                print("  ✅ Apenas a compilação inicial — JIT cache está funcionando!")
            else:
                print("  ⚠️  Mais de 1 compilação — ainda há recompilação real acontecendo.")
            print(f"{'='*60}\n")
            logging.getLogger("jax._src.interpreters.pxla").removeHandler(compile_counter)

        logger.info(
            "Training complete | training time: %.2fs | total fit time: %.2fs",
            training_time, total_time,
        )

        self._circuit = jax.jit(batched_circuit)

    def predict(self, X):
        X_enc = jnp.array(self.encoding.transform(X), dtype=jnp.float32)
        raw   = np.array(self._circuit(X_enc, self._weights) + self._bias)
        return self.loss.to_label(raw)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y.astype(int)))