from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error
import numpy as np
import pandas as pd


class ImputationEvaluator:
    def __init__(self, X_complete: pd.DataFrame, X_incomplete: pd.DataFrame, X_imputed: pd.DataFrame, X_categories: dict):
        self.X_complete = X_complete.copy()
        self.X_incomplete = X_incomplete.copy()
        self.X_imputed = X_imputed.copy()
        self.X_categories = X_categories.copy()
        return

    def evaluate(self):
        rmse_results = self.__rmse__()
        macro_f1_results = self.__macro_f1__()

        return rmse_results, macro_f1_results

    def __rmse__(self):
        X_numerical_columns = list(set(self.X_complete.columns) - set(self.X_categories.keys()))
        X_missing_index = self.X_incomplete[X_numerical_columns].isna().any(axis=1)
        X_complete_numerical = self.X_complete[X_numerical_columns].loc[X_missing_index]
        X_imputed_numerical = self.X_imputed[X_numerical_columns].loc[X_missing_index]

        rmse_results = {}
        for column in X_numerical_columns:
            if self.X_incomplete[column].isna().any():
                mse = np.mean((X_complete_numerical[column].to_numpy() - X_imputed_numerical[column].to_numpy()) ** 2)
                var = np.var(X_complete_numerical[column].to_numpy())
                rmse_results[column] = np.sqrt(mse/var) if var != 0.0 else 0.0

        return rmse_results

    def __macro_f1__(self):
        X_categorical_columns = self.X_categories.keys()
        X_missing_index = self.X_incomplete[X_categorical_columns].isna().any(axis=1)
        X_complete_categorical = self.X_complete[X_categorical_columns].loc[X_missing_index]
        X_imputed_categorical = self.X_imputed[X_categorical_columns].loc[X_missing_index]

        macro_f1_results = {}
        for column in X_categorical_columns:
            categories = self.X_categories[column]
            if self.X_incomplete[column].isna().any():
                macro_f1 = f1_score(X_complete_categorical[column], X_imputed_categorical[column], labels=categories, average='macro')
                macro_f1_results[column] = macro_f1

        return macro_f1_results
