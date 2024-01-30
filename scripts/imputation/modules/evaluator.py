from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd


class ImputationEvaluator:
    def __init__(self, X_complete: pd.DataFrame, X_incomplete: pd.DataFrame, X_imputed: pd.DataFrame, X_categories: dict):
        self.X_complete = X_complete.copy()
        self.X_incomplete = X_incomplete.copy()
        self.X_imputed = X_imputed.copy()
        self.X_categories = X_categories.copy()
        return

    def evaluate(self, column: str):
        X_categorical_columns = self.X_categories.keys()
        rmse_result = None
        macro_f1_result = None
        if column in X_categorical_columns:
            macro_f1_result = self.__macro_f1__(column)
        else:
            rmse_result = self.__rmse__(column)
        return rmse_result, macro_f1_result

    def __rmse__(self, column: str):
        X_missing_index = self.X_incomplete[self.X_incomplete[column].isnull()].index
        X_complete_numerical = self.X_complete[column].loc[X_missing_index].to_numpy().reshape(-1, 1)
        X_imputed_numerical = self.X_imputed[column].loc[X_missing_index].to_numpy().reshape(-1, 1)
        diff = X_complete_numerical - X_imputed_numerical
        diff = diff[~np.isnan(diff)]
        rmse = np.sqrt(np.sum(diff ** 2) / len(diff))
        max_min_range = np.abs(np.max(X_complete_numerical) - np.min(X_complete_numerical))
        return rmse / max_min_range if max_min_range != 0 else None

    def __macro_f1__(self, column: str):
        X_missing_index = self.X_incomplete[self.X_incomplete[column].isnull()].index
        X_complete_categorical = self.X_complete[column].loc[X_missing_index].str.lower()
        X_imputed_categorical = self.X_imputed[column].loc[X_missing_index].str.lower()
        categories = list(map(lambda x: x.lower(), self.X_categories[column]))
        encoder = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=-1)
        X_complete_categorical = encoder.fit_transform(X_complete_categorical.to_numpy().reshape(-1, 1))
        X_imputed_categorical = encoder.transform(X_imputed_categorical.to_numpy().reshape(-1, 1))
        macro_f1 = f1_score(X_complete_categorical, X_imputed_categorical, average='macro')
        return macro_f1
