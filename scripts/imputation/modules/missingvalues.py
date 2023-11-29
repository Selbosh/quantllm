from pathlib import Path
from tqdm import tqdm
import numpy as np
from jenga.corruptions.generic import MissingValues
import pandas as pd


class MissingValuesGenerator:
    def __init__(self, na_value=np.nan, logs=False):
        self.na_value = na_value
        self.logs = logs

    def generate(self, X, n_corrupted_rows: int, column, missingness: str, save_dir: Path):
        if save_dir:
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if n_corrupted_rows >= len(X):
            n_corrupted_rows = len(X)

        fraction = n_corrupted_rows / len(X)
        columns = X.columns

        categorical_columns = [
            column for column in X.columns
            if pd.api.types.is_categorical_dtype(X[column])
        ]

        numerical_columns = [
            column for column in X.columns
            if pd.api.types.is_numeric_dtype(X[column]) and column not in categorical_columns
        ]

        if self.logs:
            with open(self.save_dir / 'corruptions.csv', 'w') as f:
                f.write('column,n_corrupted_rows,missingness,na_value\n')

        for column in tqdm(columns):
            corruption = MissingValues(
                column=column, fraction=fraction,
                missingness=missingness, na_value=self.na_value
            )

            X_corrupted = corruption.transform(X)
            
            if self.save_dir:
                X_corrupted.to_csv(
                    self.save_dir / f'X_{column}-{n_corrupted_rows}-{missingness}.csv',
                    index=False
                )

            if self.logs:
                with open(self.save_dir / 'corruptions.csv', 'a') as f:
                    f.write(f"{column},{n_corrupted_rows},\
                        {missingness},{self.na_value}\n")
