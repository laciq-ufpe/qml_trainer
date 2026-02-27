from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np


def load_wisconsin(test_size=0.2, random_state=42):
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features.values
    targets = dataset.data.targets
    y = (targets['Diagnosis'] == 'M').astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
