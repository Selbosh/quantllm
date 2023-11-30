from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from missforest.missforest import MissForest
from modules.llmimputer import LLMImputer


def mean_imputation(X):
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

    result_path = Path('../../data/output/experiment/imputation/imputation_result.csv')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not result_path.exists():
        with open(result_path, 'w') as f:
            f.write('timestamp, imputation_mode, openml_id, missing_column, n_missing_values, missingness, rmse\n')

    # load complete data
    incomplete_list_path = Path('../../data/working/incomplete/logs.csv')
    incomplete_list = pd.read_csv(incomplete_list_path)

    for openml_id in incomplete_list['openml_id']:
        complete_file_path = Path(f'../../data/openml/{openml_id}/X.csv')
        X_complete = pd.read_csv(complete_file_path)

        incomplete_dir_path = Path(f'../../data/working/incomplete/{openml_id}')
        for incomplete_file_path in incomplete_dir_path.glob('*.csv'):
            # load incomplete data
            X_incomplete = pd.read_csv(incomplete_file_path)

            n_missing_values = X_incomplete.isna().sum().sum()
            
            missing_column = X_incomplete.columns[X_incomplete.isna().any()].tolist()[0]
            missingness = incomplete_file_path.stem.split('-')[-1]
            
            # Impute missing values
            # For non ML-based imputation, we can use fit methods on test data
            if args.mode == 'mean':
                X_incomplete_test_imputed = mean_imputation(X_incomplete)
            elif args.mode == 'mode':
                X_incomplete_test_imputed = mode_imputation(X_incomplete)
            elif args.mode == 'knn':
                X_incomplete_test_imputed = knn_imputation(X_incomplete)
            elif args.mode == 'rf':
                X_incomplete_test_imputed = rf_imputation(X_incomplete)
            elif args.mode == 'llm':
                X_incomplete_test_imputed = llm_imputation(X_incomplete)

            rmse = evaluate(X_complete, X_incomplete_test_imputed)
            
            print(f'Imputation mode: {args.mode}')
            print(f'incomplete_file_path: {incomplete_file_path}')
            print(f'RMSE: {rmse}')
            
            with open(result_path, 'a') as f:
                f.write(f'{timestamp}, {args.mode}, {openml_id}, {missing_column}, {n_missing_values}, {missingness}, {rmse}\n')

    return


if __name__ == '__main__':
    main()