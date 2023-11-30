from pathlib import Path
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from missforest.missforest import MissForest
from modules.llmimputer import LLMImputer


def mean_imputation(X):
    print(f'size of X: {X.shape}')
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed


def mode_imputation(X):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)
    return X_imputed


def knn_imputation(X):
    # TODO: (Enhancement) Hyperparameter tuning
    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed


def rf_imputation(X):
    # Original paper: "MissForest—non-parametric missing value imputation for mixed-type data"
    # https://academic.oup.com/bioinformatics/article/28/1/112/219101
    # Original implementation: https://github.com/yuenshingyan/MissForest

    # FIX: (Bug) MissForest displays 
    # "ValueError: Found array with 0 sample(s) (shape=(0, 15)) while a minimum of 1 is required by RandomForestRegressor."

    clf = RandomForestClassifier(n_jobs=-1)
    rgr = RandomForestRegressor(n_jobs=-1)
    mf = MissForest(clf, rgr)
    X_imputed = mf.fit_transform(X)
    return X_imputed


def llm_imputation(X):
    imputer = LLMImputer()
    X_imputed = imputer.fit_transform(X)
    return X_imputed


def evaluate(X_original, X_imputed):
    # TODO: Support macro F1 score for categorical features imputation

    # calculate RMSE
    rmse = np.sqrt(np.mean((X_original - X_imputed)**2))

    return rmse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', type=str, default='mean')
    args = argparser.parse_args()

    # load complete data
    complete_path = Path('../../data/openml/6/X.csv')
    X_complete = pd.read_csv(complete_path)

    # load incomplete data
    incomplete_path = Path('../../data/working/generated-missing-values/6/X_high-100-MAR.csv')
    X_incomplete = pd.read_csv(incomplete_path)

    n_missing_values = X_incomplete.isna().sum().sum()
    print(f'Number of missing values: {n_missing_values}')

    # Split into train and test (8:2)
    X_complete_train, X_complete_test = train_test_split(X_complete, test_size=0.2, shuffle=False)
    X_incomplete_train, X_incomplete_test = train_test_split(X_incomplete, test_size=0.2, shuffle=False)

    # Impute missing values
    # For non ML-based imputation, we can use fit methods on test data
    if args.mode == 'mean':
        X_incomplete_test_imputed = mean_imputation(X_incomplete_test)
    elif args.mode == 'mode':
        X_incomplete_test_imputed = mode_imputation(X_incomplete_test)
    elif args.mode == 'knn':
        X_incomplete_test_imputed = knn_imputation(X_incomplete_test)
    elif args.mode == 'rf':
        X_incomplete_test_imputed = rf_imputation(X_incomplete_test)
    elif args.mode == 'llm':
        X_incomplete_test_imputed = llm_imputation(X_incomplete_test)

    print(f'Imputation mode: {args.mode}')

    rmse = evaluate(X_complete_test, X_incomplete_test_imputed)
    print(f'RMSE: {rmse}')

    return


if __name__ == '__main__':
    main()