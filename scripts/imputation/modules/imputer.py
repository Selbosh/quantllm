import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder


class MeanModeImputer():
    def __init__(self, na_value=np.nan):
        self.na_value = na_value

    def fit_transform(self, X: pd.DataFrame):
        X_numerical, X_categorical = X.select_dtypes(include=np.number), X.select_dtypes(exclude=np.number)
        X_original_columns, X_numerical_columns, X_categorical_columns = X.columns, X_numerical.columns, X_categorical.columns

        if X_numerical.shape[1] > 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_numerical_imputed = imputer.fit_transform(X_numerical.to_numpy())
            X_numerical_imputed = pd.DataFrame(X_numerical_imputed, columns=X_numerical_columns)

        if X_categorical.shape[1] > 0:
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X_categorical_imputed = imputer.fit_transform(X_categorical.to_numpy())
            X_categorical_imputed = pd.DataFrame(X_categorical_imputed, columns=X_categorical_columns)

        if X_numerical.shape[1] > 0 and X_categorical.shape[1] > 0:
            X_imputed = pd.merge(X_numerical_imputed, X_categorical_imputed, left_index=True, right_index=True).reindex(columns=X_original_columns)
        elif X_numerical.shape[1] > 0:
            X_imputed = X_numerical_imputed
        else:
            X_imputed = X_categorical_imputed

        return X_imputed


class KNNImputer():
    def __init__(self, na_value=np.nan, n_jobs: int | None = None):
        self.na_value = na_value
        self.n_jobs = n_jobs

    def fit_transform(self, X: pd.DataFrame):
        X_numerical, X_categorical = X.select_dtypes(include=np.number), X.select_dtypes(exclude=np.number)
        X_original_columns, X_numerical_columns, X_categorical_columns = X.columns, X_numerical.columns, X_categorical.columns

        if X_numerical.shape[1] > 0:
            knn = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=self.n_jobs)
            imputer = IterativeImputer(estimator=knn, missing_values=np.nan, max_iter=10, random_state=42)
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

        return X_imputed


class RandomForestImputer():
    def __init__(self, na_value=np.nan, n_jobs: int | None = None):
        self.na_value = na_value
        self.n_jobs = n_jobs

    def fit_transform(self, X: pd.DataFrame):
        # Original paper: "MissForest—non-parametric missing value imputation for mixed-type data"
        # https://academic.oup.com/bioinformatics/article/28/1/112/219101

        X_numerical, X_categorical = X.select_dtypes(include=np.number), X.select_dtypes(exclude=np.number)
        X_original_columns, X_numerical_columns, X_categorical_columns = X.columns, X_numerical.columns, X_categorical.columns

        if X_numerical.shape[1] > 0:
            rf = RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs)
            imputer = IterativeImputer(estimator=rf, missing_values=np.nan, max_iter=10, random_state=42)
            X_numerical_imputed = imputer.fit_transform(X_numerical)
            X_numerical_imputed = pd.DataFrame(X_numerical_imputed, columns=X_numerical_columns)

        if X_categorical.shape[1] > 0:
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            imputer = IterativeImputer(estimator=rf, missing_values=np.nan, max_iter=10, random_state=42)
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

        return X_imputed
