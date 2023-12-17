from sklearn.metrics import mean_squared_error, f1_score
import numpy as np
import pandas as pd


class ImputationEvaluator:
    def __init__(self):
        return

    def evaluate(self, X_original: pd.DataFrame, X_incomplete: pd.DataFrame, X_imputed: pd.DataFrame):
        self.X_original = X_original.copy()
        self.X_incomplete = X_incomplete.copy()
        self.X_imputed = X_imputed.copy()

        rmse_results = self.__rmse()
        macro_f1_results = self.__macro_f1()

        return rmse_results, macro_f1_results

    def __rmse(self):
        X_numerical_columns = self.X_imputed.select_dtypes(include=np.number).columns
        X_original_numerical = self.X_original[X_numerical_columns]
        X_imputed_numerical = self.X_imputed[X_numerical_columns]

        rmse_results = {}
        for column in X_numerical_columns:
            if self.X_incomplete[column].isna().any():
                rmse = mean_squared_error(X_original_numerical, X_imputed_numerical, squared=False)
                rmse_results[column] = rmse

        return rmse_results

    def __macro_f1(self):
        X_categorical_columns = self.X_imputed.select_dtypes(exclude=np.number).columns
        X_original_categorical = self.X_original[X_categorical_columns]
        X_imputed_categorical = self.X_imputed[X_categorical_columns]

        macro_f1_results = {}
        for column in X_categorical_columns:
            if self.X_incomplete[column].isna().any():
                macro_f1 = f1_score(X_original_categorical[column], X_imputed_categorical[column], average='macro')
                macro_f1_results[column] = macro_f1

        return macro_f1_results
