import numpy as np
from jenga.corruptions.generic import MissingValues


class MissingValuesGenerator:
    def __init__(self, na_value=np.nan, logs=False):
        self.na_value = na_value
        self.logs = logs

    def generate(self, X, n_corrupted_rows: int, column, missingness: str):
        if n_corrupted_rows >= len(X):
            n_corrupted_rows = len(X)

        fraction = n_corrupted_rows / len(X)

        corruption = MissingValues(
            column=column, fraction=fraction,
            missingness=missingness, na_value=self.na_value
        )

        X_corrupted = corruption.transform(X)

        return X_corrupted
