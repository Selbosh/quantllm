from sklearn.metrics import mean_squared_error, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ImputationEvaluator:
    def __init__(self, original_df: pd.DataFrame, corrupted_df: pd.DataFrame, imputed_df: pd.DataFrame):
        self.original_df = original_df
        self.corrupted_df = corrupted_df
        self.imputed_df = imputed_df

    def calculate_rmse(self):
        numerical_cols = self.original_df.select_dtypes(include=['float', 'int']).columns
        rmse_scores = []
        for col in numerical_cols:
            original_values = self.original_df[col]
            imputed_values = self.imputed_df[col]
            rmse = np.sqrt(mean_squared_error(original_values, imputed_values))
            rmse_scores.append(rmse)
        return rmse_scores

    def calculate_macro_f1(self):
        categorical_cols = self.original_df.select_dtypes(include=['object']).columns
        f1_scores = []
        for col in categorical_cols:
            original_values = self.original_df[col]
            imputed_values = self.imputed_df[col]
            f1 = f1_score(original_values, imputed_values, average='macro')
            f1_scores.append(f1)
        return f1_scores
