import numpy as np
import pandas as pd


class MissingValues:
    '''
    This class is based on the MissingValues class from Jenga.
    https://github.com/schelterlabs/jenga
    '''
    def __init__(self, column, n_corrupted_rows, na_value=np.nan, missingness='MCAR', seed=42):
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
        self.column = column
        self.n_corrupted_rows = n_corrupted_rows
        self.sampling = missingness
        self.na_value = na_value

    def sample_rows(self, data):
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
                rows = data[self.column].sort_values().iloc[perc_idx].index

            # At Random
            elif self.sampling.endswith('AR'):
                depends_on_col = np.random.choice(list(set(data.columns) - {self.column}))
                # pick a random percentile of values in other column
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows

    def transform(self, X):
        X_corrupted = X.copy(deep=True)
        rows = self.sample_rows(X_corrupted)
        X_corrupted.loc[rows, [self.column]] = self.na_value
        return X_corrupted