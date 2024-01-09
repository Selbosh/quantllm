from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import json

from modules.imputer import MeanModeImputer, KNNImputer, MissForestImputer
from modules.llmimputer import LLMImputer
from modules.evaluator import ImputationEvaluator


def experiment(args: argparse.Namespace, openml_id: int, train_or_test: str, X_complete_filepath: Path, X_incomplete_filepath: Path, X_imputed_filepath: Path, X_categories_filepath: Path, results_filepath: Path):
    '''
    This function runs the imputation experiment for a specific dataset.
    
    Args:
        - args:    argparse.Namespace object
        - openml_id:    OpenML dataset id, integer
        - X_complete_filepath:    path to complete data, pathlib.Path object
        - X_incomplete_filepath:    path to incomplete data, pathlib.Path object
        - X_imputed_filepath:    path to save imputed data, pathlib.Path object
        - X_categories_filepath:    path to save X_categories.json, pathlib.Path object
        - results_filepath:    path to save results, pathlib.Path object
    '''
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if args.debug:
        print(f'Imputing OpenML Id: {openml_id}, Train/Test: {train_or_test}, Method: {args.method}')

    # Load data
    X_complete = pd.read_csv(X_complete_filepath, header=0)
    X_incomplete = pd.read_csv(X_incomplete_filepath, header=0)

    with open(X_categories_filepath, 'r') as f:
        X_categories = json.load(f)

    n_missing_values = X_incomplete.isna().sum().sum()
    missing_columns = X_incomplete.columns[X_incomplete.isna().any()].tolist()
    missing_columns = {column: ('categorical' if column in X_categories.keys() else 'numerical') for column in missing_columns}
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
        data_dirpath = Path(__file__).parents[2] / 'data'
        openml_dirpath = data_dirpath / 'openml'
        description_file = openml_dirpath / f'{openml_id}/description.txt'
        description = description_file.read_text()
        imputer = LLMImputer(X_categories=X_categories, dataset_description=description)

    # Run imputation
    X_imputed = imputer.fit_transform(X_incomplete)

    # Save imputed data
    X_imputed.to_csv(X_imputed_filepath, index=False)

    # Evaluate imputation
    if args.evaluate:
        evaluator = ImputationEvaluator(X_complete, X_incomplete, X_imputed, X_categories)
        rmse, macro_f1 = evaluator.evaluate()

        if args.debug:
            print(f' - RMSE: {rmse}')
            print(f' - Macro F1 score: {macro_f1}')
            print('')

        with open(results_filepath, 'a') as f:
            missing_columns, rmse, macro_f1 = f'\"{missing_columns}\"', f'\"{rmse}\"', f'\"{macro_f1}\"'
            f.write(f'{timestamp},{args.method},{openml_id},{train_or_test},{missing_columns},{n_missing_values},{missingness},{rmse},{macro_f1}\n')

    return


def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, default='meanmode')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--openml_id', type=int, default=None)
    argparser.add_argument('--X_incomplete_filename', type=str, default=None)
    argparser.add_argument('--evaluate', action='store_true')
    return argparser.parse_args()


def main():
    args = config_args()
    
    data_dirpath = Path(__file__).parents[2] / 'data'

    # Set path to save imputed data and results
    output_dirpath = data_dirpath / f'output/imputation/{args.method}'
    output_dirpath.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results_filepath = output_dirpath / f'results_{timestamp}.csv'
        with open(results_filepath, 'w') as f:
            f.write('timestamp,method,openml_id,train_or_test,missing_column,n_missing_values,missingness,rmse,macro_f1\n')

    openml_dirpath = data_dirpath / 'openml'
    working_dirpath = data_dirpath / 'working'
    incomplete_dirpath = working_dirpath / 'incomplete'
    incomplete_log_filepath = incomplete_dirpath / 'logs.csv'
    complete_dirpath = working_dirpath / 'complete'

    if args.openml_id is None and args.X_incomplete_filename is not None:
        raise ValueError('When --X_incomplete_filename is specified, --openml_id must also be specified.')
    
    if args.openml_id is not None and args.X_incomplete_filename is not None:
        train_or_test = args.X_incomplete_filename.split('_')[1]
        X_complete_filepath = complete_dirpath / f'{args.openml_id}/X_{train_or_test}.csv'
        X_incomplete_filepath = incomplete_dirpath / f'{args.openml_id}/{args.X_incomplete_filename}'
        X_imputed_dirpath = output_dirpath / f'imputed_data/{args.openml_id}'
        X_imputed_dirpath.mkdir(parents=True, exist_ok=True)
        X_imputed_filepath = X_imputed_dirpath / f'{args.X_incomplete_filename}'
        X_categories_filepath = openml_dirpath / f'{args.openml_id}/X_categories.json'

        experiment(
            args, args.openml_id, train_or_test, X_complete_filepath, X_incomplete_filepath, X_imputed_filepath, X_categories_filepath, results_filepath
        )
        return

    with open(incomplete_log_filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('openml_id'):
            continue
        else:
            items = line.split(',')
            openml_id, train_or_test, n_missing_values, missing_column_name, missingness = int(items[0]), items[1], int(items[2]), items[3], items[5].strip()

            X_complete_filepath = complete_dirpath / f'{openml_id}/X_{train_or_test}.csv'
            X_incomplete_filepath = incomplete_dirpath / f'{openml_id}/X_{train_or_test}_{missing_column_name}_{n_missing_values}-{missingness}.csv'
            
            X_imputed_dirpath = output_dirpath / f'imputed_data/{openml_id}'
            X_imputed_dirpath.mkdir(parents=True, exist_ok=True)
            
            X_imputed_filepath = X_imputed_dirpath / f'X_{train_or_test}_{missing_column_name}_{n_missing_values}-{missingness}.csv'
            X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'

            if args.openml_id is not None:
                if args.openml_id == openml_id:
                    experiment(
                        args, openml_id, train_or_test, X_complete_filepath, X_incomplete_filepath, X_imputed_filepath, X_categories_filepath, results_filepath
                    )
            else:
                experiment(
                    args, openml_id, train_or_test, X_complete_filepath, X_incomplete_filepath, X_imputed_filepath, X_categories_filepath, results_filepath
                )
    return


if __name__ == '__main__':
    main()
