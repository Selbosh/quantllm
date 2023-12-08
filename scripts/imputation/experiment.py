from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from modules.llmimputer import LLMImputer

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, mean_squared_error


def meanmode_imputation(X):
    X_numerical = X.select_dtypes(include=np.number)
    X_categorical = X.select_dtypes(exclude=np.number)
    original_columns = X.columns
    numerical_columns = X_numerical.columns
    categorical_columns = X_categorical.columns
    concatenated_columns = numerical_columns.append(categorical_columns)
    X_numerical_imputed = np.empty((X.shape[0], 0))
    X_categorical_imputed = np.empty((X.shape[0], 0))

    if X_numerical.shape[1] > 0:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_numerical_imputed = imputer.fit_transform(X_numerical)

    if X_categorical.shape[1] > 0:
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        X_categorical_imputed = imputer.fit_transform(X_categorical)

    X_imputed = np.concatenate((X_numerical_imputed, X_categorical_imputed), axis=1)
    X_imputed = pd.DataFrame(X_imputed, columns=concatenated_columns).reindex(columns=original_columns)

    return X_imputed


def knn_imputation(X):
    # TODO: (Enhancement) Hyperparameter tuning
    X_numerical = X.select_dtypes(include=np.number)
    X_categorical = X.select_dtypes(exclude=np.number)
    original_columns = X.columns
    categorical_columns = X_categorical.columns

    if X_numerical.shape[1] > 0:
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1)
        imputer = IterativeImputer(estimator=knn, missing_values=np.nan, max_iter=10, random_state=42)
        X_numerical_imputed = imputer.fit_transform(X_numerical)

    if X_categorical.shape[1] > 0:
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        imputer = IterativeImputer(estimator=knn, missing_values=np.nan, max_iter=10, random_state=42)
        encoder = OrdinalEncoder()
        X_categorical = encoder.fit_transform(X_categorical)
        X_categorical_imputed = imputer.fit_transform(X_categorical)
        X_categorical_imputed = pd.DataFrame(X_categorical_imputed, columns=categorical_columns)
        X_categorical_imputed = encoder.inverse_transform(X_categorical_imputed)

    X_imputed = np.concatenate((X_numerical_imputed, X_categorical_imputed), axis=1)
    X_imputed = pd.DataFrame(X_imputed, columns=original_columns)

    return X_imputed

def rf_imputation(X):
    # Original paper: "MissForest—non-parametric missing value imputation for mixed-type data"
    # https://academic.oup.com/bioinformatics/article/28/1/112/219101

    X_numerical = X.select_dtypes(include=np.number)
    X_categorical = X.select_dtypes(exclude=np.number)
    original_columns = X.columns
    categorical_columns = X_categorical.columns

    if X_numerical.shape[1] > 0:
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        imputer = IterativeImputer(estimator=rf, missing_values=np.nan, max_iter=10, random_state=42)
        X_numerical_imputed = imputer.fit_transform(X_numerical)

    if X_categorical.shape[1] > 0:
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        imputer = IterativeImputer(estimator=rf, missing_values=np.nan, max_iter=10, random_state=42)
        encoder = OrdinalEncoder()
        X_categorical = encoder.fit_transform(X_categorical)
        X_categorical_imputed = imputer.fit_transform(X_categorical)
        X_categorical_imputed = pd.DataFrame(X_categorical_imputed, columns=categorical_columns)
        X_categorical_imputed = encoder.inverse_transform(X_categorical_imputed)

    X_imputed = np.concatenate((X_numerical_imputed, X_categorical_imputed), axis=1)
    X_imputed = pd.DataFrame(X_imputed, columns=original_columns)

    return X_imputed


def llm_imputation(X, description=None):
    X_numerical = X.select_dtypes(include=np.number)
    X_categorical = X.select_dtypes(exclude=np.number)
    original_columns = X.columns
    numerical_columns = X_numerical.columns
    categorical_columns = X_categorical.columns

    imputer = LLMImputer()
    X_imputed = imputer.fit_transform(X)

    return X_imputed


def evaluate(X_original, X_incomplete, X_imputed):
    # TODO: Support macro F1 score for categorical features imputation
    X_numerical = X_imputed.select_dtypes(include=np.number)
    X_categorical = X_imputed.select_dtypes(exclude=np.number)
    numerical_columns = X_numerical.columns
    categorical_columns = X_categorical.columns

    X_original_numerical = X_original[numerical_columns]
    X_original_categorical = X_original[categorical_columns]
    X_imputed_numerical = X_imputed[numerical_columns]
    X_imputed_categorical = X_imputed[categorical_columns]

    # calculate RMSE for numerical features
    rmse_results = {}
    for column in numerical_columns:
        if X_incomplete[column].isna().any():
            rmse = mean_squared_error(X_original_numerical, X_imputed_numerical, squared=False)
            rmse_results[column] = rmse

    # calculate macro F1 score for categorical features
    macro_f1_results = {}
    for column in categorical_columns:
        if X_incomplete[column].isna().any():
            macro_f1 = f1_score(X_original_categorical[column], X_imputed_categorical[column], average='macro')
            macro_f1_results[column] = macro_f1

    return rmse_results, macro_f1_results


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, default='meanmode')
    argparser.add_argument('--debug', action='store_true')
    args = argparser.parse_args()

    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_dir = Path(__file__).parents[2] / f'data/output/imputation/{args.method}'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'results.csv'

    with open(results_file, 'w') as f:
        f.write('timestamp, imputation_method, openml_id, missing_column, n_missing_values, missingness, rmse, macro_f1\n')

    # load complete data
    incomplete_dir = Path(__file__).parents[2] / 'data/working/incomplete'
    incomplete_log_file = incomplete_dir / 'logs.csv'
    incomplete_log = pd.read_csv(incomplete_log_file)

    openml_dir = Path(__file__).parents[2] / 'data/openml'

    for openml_id in incomplete_log['openml_id']:
        original_file = openml_dir / f'{openml_id}/X.csv'
        X_original = pd.read_csv(original_file)

        incomplete_id_dir = incomplete_dir / f'{openml_id}'
        for incomplete_file in incomplete_id_dir.glob('*.csv'):
            X_incomplete = pd.read_csv(incomplete_file)

            n_missing_values = X_incomplete.isna().sum().sum()
            missing_columns = X_incomplete.columns[X_incomplete.isna().any()].tolist()
            missingness = incomplete_file.stem.split('-')[-1]

            # Impute missing values
            # For non ML-based imputation, we can use fit methods on test data
            if args.method == 'meanmode':
                X_imputed = meanmode_imputation(X_incomplete)
            elif args.method == 'knn':
                X_imputed = knn_imputation(X_incomplete)
            elif args.method == 'rf':
                X_imputed = rf_imputation(X_incomplete)
            elif args.method == 'llm':
                description_file = openml_dir / f'{openml_id}/description.txt'
                description = description_file.read_text()
                X_imputed = llm_imputation(X_incomplete, description=description)
            
            # Save imputed data
            imputed_dir = output_dir / f'imputed_data/{openml_id}'
            imputed_dir.mkdir(parents=True, exist_ok=True)
            imputed_file = imputed_dir / f'{incomplete_file.stem}.csv'
            X_imputed.to_csv(imputed_file, index=False)

            # Evaluate imputation
            rmse, macro_f1 = evaluate(X_original, X_incomplete, X_imputed)

            if args.debug:
                print(f'RMSE: {rmse}')
                print(f'Macro F1 score: {macro_f1}')

            with open(results_file, 'a') as f:
                f.write(f'{timestamp}, {args.method}, {openml_id}, {missing_columns}, {n_missing_values}, {missingness}, {rmse}, {macro_f1}\n')

    return


if __name__ == '__main__':
    main()
