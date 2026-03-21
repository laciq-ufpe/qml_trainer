from qmlvqc.architectures import CustomRot
from qmlvqc import BCE, PhaseEncoding, PennylaneExecutor, VQC
from qmlvqc.data import load_wisconsin

X_train, X_test, y_train, y_test = load_wisconsin()

executor = PennylaneExecutor(n_qubits=8)

model = VQC(
    n_qubits=8,
    architecture=CustomRot(),
    loss=BCE(),
    encoding=PhaseEncoding(),
    executor=executor,
)

model.fit(X_train, y_train, epochs=10, lr=1e-3, batch_size=32)

print("Train acc:", model.score(X_train, y_train))
print("Test acc: ", model.score(X_test, y_test))
