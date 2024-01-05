from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import json

from modules.imputer import MeanModeImputer, KNNImputer, RandomForestImputer, MissForestImputer
from modules.llmimputer import LLMImputer
from modules.evaluator import ImputationEvaluator


def experiment(args: argparse.Namespace, openml_id: int, X_original_filepath: Path, X_incomplete_filepath: Path, X_imputed_filepath: Path, X_categories_filepath: Path, results_filepath: Path):
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if args.debug:
        print(f'Imputing OpenML Id: {openml_id}')

    # Load data
    X_original = pd.read_csv(X_original_filepath, header=0)
    X_incomplete = pd.read_csv(X_incomplete_filepath, header=0)

    with open(X_categories_filepath, 'r') as f:
        X_categories = json.load(f)

    n_missing_values = X_incomplete.isna().sum().sum()
    missing_columns = X_incomplete.columns[X_incomplete.isna().any()].tolist()
    missing_columns = {column: ("categorical" if column in X_categories.keys() else "numerical") for column in missing_columns}
    missingness = X_incomplete_filepath.stem.split('-')[-1]

    # Impute missing values
    # For non ML-based imputation, we can use fit methods on test data
    if args.method == 'meanmode':
        imputer = MeanModeImputer(X_categories=X_categories)
    elif args.method == 'knn':
        imputer = KNNImputer(n_jobs=-1, X_categories=X_categories)
    elif args.method == 'rf':
        imputer = MissForestImputer(n_jobs=-1, X_categories=X_categories)
    elif args.method == 'llm':
        openml_dirpath = data_dirpath / 'openml'
        description_file = openml_dirpath / f'{openml_id}/description.txt'
        description = description_file.read_text()
        imputer = LLMImputer(dataset_description=description)

    # Run imputation
    X_imputed = imputer.fit_transform(X_incomplete)

    # Save imputed data
    X_imputed.to_csv(X_imputed_filepath, index=False)

    # Evaluate imputation
    if args.evaluate:
        evaluator = ImputationEvaluator(X_original, X_incomplete, X_imputed, X_categories)
        rmse, macro_f1 = evaluator.evaluate()

        if args.debug:
            print(f' - RMSE: {rmse}')
            print(f' - Macro F1 score: {macro_f1}')
            print('')

        with open(results_filepath, 'a') as f:
            missing_columns, rmse, macro_f1 = f'\"{missing_columns}\"', f'\"{rmse}\"', f'\"{macro_f1}\"'
            f.write(f'{timestamp},{args.method},{openml_id},{missing_columns},{n_missing_values},{missingness},{rmse},{macro_f1}\n')

    return


def main():
    # Arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, default='meanmode')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--openml_id', type=int, default=None)
    argparser.add_argument('--X_incomplete_filename', type=str, default=None)
    argparser.add_argument('--evaluate', action='store_true')
    args = argparser.parse_args()
    
    data_dirpath = Path(__file__).parents[2] / 'data'

    output_dirpath = data_dirpath / f'output/imputation/{args.method}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results_filepath = output_dirpath / f'results_{timestamp}.csv'

    if args.evaluate:
        with open(results_filepath, 'w') as f:
            f.write('timestamp,method,openml_id,missing_column,n_missing_values,missingness,rmse,macro_f1\n')

    # load incomplete data
    incomplete_dirpath = data_dirpath / 'working/incomplete'
    incomplete_log_filepath = incomplete_dirpath / 'logs.csv'

    # path to original (complete) opanml datasets
    openml_dirpath = data_dirpath / 'openml'

    # run experiment for a specific dataset
    if args.openml_id is not None and args.X_incomplete_filename is not None:
        openml_id = args.openml_id

        X_original_filepath = openml_dirpath / f'{openml_id}/X.csv'
        X_incomplete_filepath = incomplete_dirpath / f'{openml_id}/{args.X_incomplete_filename}'
        X_imputed_dirpath = output_dirpath / f'imputed_data/{openml_id}'
        X_imputed_dirpath.mkdir(parents=True, exist_ok=True)
        X_imputed_filepath = X_imputed_dirpath / args.X_incomplete_filename
        X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'

        experiment(args, openml_id, X_original_filepath, X_incomplete_filepath, X_imputed_filepath, X_categories_filepath, results_filepath)
        return

    # run experiment for each dataset
    with open(incomplete_log_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('openml_id'):
            continue
        elif args.openml_id is not None and args.openml_id != int(line.split(',')[0]):
            continue
        else:
            items = line.split(',')
            openml_id, n_missing_values, missing_column_name, missingness = int(items[0]), int(items[1]), items[2], items[4].strip()

            X_original_filepath = openml_dirpath / f'{openml_id}/X.csv'
            X_incomplete_filepath = incomplete_dirpath / f'{openml_id}/X_{missing_column_name}_{n_missing_values}-{missingness}.csv'
            X_imputed_dirpath = output_dirpath / f'imputed_data/{openml_id}'
            X_imputed_dirpath.mkdir(parents=True, exist_ok=True)
            X_imputed_filepath = X_imputed_dirpath / f'X_{missing_column_name}_{n_missing_values}-{missingness}.csv'
            X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'

            experiment(args, openml_id, X_original_filepath, X_incomplete_filepath, X_imputed_filepath, X_categories_filepath, results_filepath)

    return


if __name__ == '__main__':
    main()
