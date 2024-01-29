import numpy as np
import pandas as pd


class MissingValues:
    '''
    This class is based on the MissingValues class from Jenga.
    https://github.com/schelterlabs/jenga
    '''
    def __init__(self, n_corrupted_rows: int, n_corrupted_columns: int, na_value=np.nan, missingness='MCAR', seed=42):
        '''
        This class is based on the MissingValues class from Jenga.
        https://github.com/schelterlabs/jenga

        Corruptions for structured data

        Input:
            - column:    column to perturb, string
            - n_corrupted_rows:   numbers of rows to corrupt, integer between 0 and len(data)
            - na_value:   value
            - missingness:   sampling mechanism for corruptions, string in ['MCAR', 'MAR', 'MNAR']
        '''
        self.n_corrupted_rows = n_corrupted_rows
        self.n_corrupted_columns = n_corrupted_columns
        self.sampling = missingness
        self.na_value = na_value

        np.random.seed(seed)

    def sample_rows(self, data, column):
        if self.n_corrupted_rows >= len(data):
            rows = data.index
        # Completely At Random
        elif self.sampling.endswith('CAR'):
            rows = np.random.permutation(data.index)[:int(self.n_corrupted_rows)]
        elif self.sampling.endswith('NAR') or self.sampling.endswith('AR'):
            n_values_to_discard = int(min(self.n_corrupted_rows, len(data)))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

            # Not At Random
            if self.sampling.endswith('NAR'):
                # pick a random percentile of values in this column
                rows = data[column].sort_values().iloc[perc_idx].index

            # At Random
            elif self.sampling.endswith('AR'):
                depends_on_col = np.random.choice(list(set(data.columns) - {column}))
                # pick a random percentile of values in other column
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows

    def sample_columns(self, data):
        return np.random.choice(data.columns, self.n_corrupted_columns, replace=False)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_corrupted = X.copy(deep=True)

        columns = self.sample_columns(X_corrupted)
        for column in columns:
            rows = self.sample_rows(X_corrupted, column)
            X_corrupted.loc[rows, [column]] = self.na_value

        if len(X_corrupted[X_corrupted.isna().any(axis=1)]) > self.n_corrupted_rows:
            n_revert_rows = len(X_corrupted[X_corrupted.isna().any(axis=1)]) - self.n_corrupted_rows
            revert_rows = np.random.choice(X_corrupted[X_corrupted.isna().any(axis=1)].index, n_revert_rows, replace=False)
            X_corrupted.loc[revert_rows, :] = X.loc[revert_rows, :]

        return X_corrupted