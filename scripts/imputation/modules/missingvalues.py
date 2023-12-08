import numpy as np
from jenga.corruptions.generic import MissingValues


class MissingValuesGenerator:
    def __init__(self, na_value=np.nan, logs=False):
        self.na_value = na_value
        self.logs = logs

    def generate(self, X, n_corrupted_rows: int, columns: [str], missingness: str):
        if n_corrupted_rows >= len(X):
            n_corrupted_rows = len(X)

        fraction = n_corrupted_rows / len(X)

        corruptions = []
        for column in columns:
            corruption = MissingValues(
                column=column, fraction=fraction,
                missingness=missingness, na_value=self.na_value
            )
            corruptions.append(corruption)

        X_corrupted = X.copy(deep=True)
        for corruption in corruptions:
            X_corrupted = corruption.transform(X_corrupted)

        return X_corrupted
