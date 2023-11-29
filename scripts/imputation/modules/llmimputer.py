import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LLMImputer(BaseEstimator, TransformerMixin):
    def __init__(self, na_value=np.nan):
        self.na_value = na_value

    def fit(self, X, y=None):
        # Nothing to do here. Just for compatibility with scikit-learn.
        return self

    def transform(self, X: pd.DataFrame):
        X_copy = X.copy()
        
        # The imputation module will be called for each rows with missing values
        # Rows with no missing values will be skipped
        X_copy = X_copy.apply(lambda x: self.llm(x) if x.isna().sum() > 0 else x, axis=1)
        
        return X_copy

    def llm(self, x):
        """LLM module
        - Kai will implement this?

        Args:
            x (pd.series): A row (sample) to be imputed

        Returns:
            pd.series: The imputed row
        """
        column_name = x.name
        estimated_value = 0
        x = x.fillna(estimated_value)
        return x
