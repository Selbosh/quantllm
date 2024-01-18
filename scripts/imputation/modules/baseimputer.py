import numpy as np
import pandas as pd
from missforest.missforest import MissForest
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


class MeanModeImputer():
    def __init__(self, na_value=np.nan, X_categories: dict = {}):
        self.na_value = na_value
        self.X_categories = X_categories

    def fit_transform(self, X: pd.DataFrame):
        X_original_columns = X.columns
        X_categorical_columns = self.X_categories.keys()
        X_numerical_columns = list(set(X_original_columns) - set(X_categorical_columns))
        X_numerical, X_categorical = X[X_numerical_columns], X[X_categorical_columns]

        if X_numerical.shape[1] > 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_numerical_imputed = imputer.fit_transform(X_numerical)
            X_numerical_imputed = pd.DataFrame(X_numerical_imputed, columns=X_numerical_columns)

        if X_categorical.shape[1] > 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_categorical_imputed = imputer.fit_transform(X_categorical)
            X_categorical_imputed = pd.DataFrame(X_categorical_imputed, columns=X_categorical_columns)

        if X_numerical.shape[1] > 0 and X_categorical.shape[1] > 0:
            X_imputed = pd.merge(X_numerical_imputed, X_categorical_imputed, left_index=True, right_index=True)
        elif X_numerical.shape[1] > 0:
            X_imputed = X_numerical_imputed
        else:
            X_imputed = X_categorical_imputed

        return X_imputed.reindex(columns=X_original_columns)


class KNNImputer():
    def __init__(self, na_value=np.nan, n_jobs: int | None = None, X_categories: dict = {}):
        self.na_value = na_value
        self.n_jobs = n_jobs
        self.X_categories = X_categories

    def fit_transform(self, X: pd.DataFrame):
        X_original_columns = X.columns
        X_categorical_columns = self.X_categories.keys()
        X_numerical_columns = list(set(X_original_columns) - set(X_categorical_columns))
        X_numerical, X_categorical = X[X_numerical_columns], X[X_categorical_columns]

        if X_numerical.shape[1] > 0:
            imputer = KNNImputer(neighbors=5, weights='distance', n_jobs=self.n_jobs)
            X_numerical_imputed = imputer.fit_transform(X_numerical)
            X_numerical_imputed = pd.DataFrame(X_numerical_imputed, columns=X_numerical_columns)

        if X_categorical.shape[1] > 0:
            knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=self.n_jobs)
            imputer = IterativeImputer(estimator=knn, missing_values=np.nan, max_iter=10, random_state=42)
            encoder = OrdinalEncoder()
            X_categorical = encoder.fit_transform(X_categorical)
            X_categorical_imputed = imputer.fit_transform(X_categorical)
            X_categorical_imputed = encoder.inverse_transform(X_categorical_imputed)
            X_categorical_imputed = pd.DataFrame(X_categorical_imputed, columns=X_categorical_columns)

        if X_numerical.shape[1] > 0 and X_categorical.shape[1] > 0:
            X_imputed = pd.merge(X_numerical_imputed, X_categorical_imputed, left_index=True, right_index=True).reindex(columns=X_original_columns)
        elif X_numerical.shape[1] > 0:
            X_imputed = X_numerical_imputed
        else:
            X_imputed = X_categorical_imputed

        return X_imputed.reindex(columns=X_original_columns)


class RandomForestImputer():
    def __init__(self, na_value=np.nan, n_jobs: int | None = None, X_categories: dict = {}):
        self.na_value = na_value
        self.n_jobs = n_jobs
        self.X_categories = X_categories

    def fit_transform(self, X: pd.DataFrame):
        # Original paper: "MissForest—non-parametric missing value imputation for mixed-type data"
        # https://academic.oup.com/bioinformatics/article/28/1/112/219101
        
        X_categorical_columns = self.X_categories.keys()

        clf = RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs)
        rgr = RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs)
        mf = MissForest(clf, rgr)
        
        if len(X_categorical_columns) > 0:
            X_imputed = mf.fit_transform(X, X_categorical_columns)
        else:
            X_imputed = mf.fit_transform(X)

        return X_imputed