from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import json
from tqdm import tqdm

from modules.baseimputer import MeanModeImputer, KNNImputer, RandomForestImputer
from modules.llmimputer import LLMImputer
from modules.evaluator import ImputationEvaluator

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, default='meanmode')
    argparser.add_argument('--evaluate', action='store_true')
    argparser.add_argument('--downstream', action='store_true')
    argparser.add_argument('--openml_id', type=int, default=None)
    argparser.add_argument('--missingness', type=str, default=None)
    argparser.add_argument('--dataset', nargs='*', type=str, default=['incomplete', 'complete'])
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()


def experiment(args: argparse.Namespace, timestamp: str, dataset_type: str, openml_dirpath: Path, input_dirpath: Path, output_dirpath: Path):
    imputation_results_filepath = None
    if args.evaluate:
        imputation_results_filepath = output_dirpath / f'imputation_{timestamp}.csv'
        with open(imputation_results_filepath, 'w') as f:
            f.write('timestamp,method,openml_id,train_or_test,missing_column_name,missing_column_type,n_missing_values,missingness,rmse,macro_f1\n')

    downstream_results_filepath = None
    if args.downstream:
        downstream_results_filepath = output_dirpath / f'downstream_{timestamp}.csv'
        with open(downstream_results_filepath, 'w') as f:
            f.write('timestamp,method,openml_id,missing_column_name,missing_column_type,missingness,score\n')

    groundtruth_dirpath = None
    corrupted_dirpath = input_dirpath
    imputed_dirpath = output_dirpath / 'imputed'
    if dataset_type == 'complete':
        groundtruth_dirpath = input_dirpath / 'original'
        corrupted_dirpath = input_dirpath / 'corrupted'

    logs = pd.read_csv(input_dirpath / 'logs.csv', header=0).drop('train_or_test', axis=1)

    for log in tqdm(logs.itertuples(), total=len(logs)):
        openml_id, missingness = log.openml_id, log.missingness

        if args.openml_id is not None and args.openml_id != openml_id:
            continue
        elif args.missingness is not None and args.missingness != missingness:
            continue

        X_groundtruth_train_filepath = None
        X_groundtruth_test_filepath = None
        y_train_filepath = corrupted_dirpath / f'{openml_id}/y_train.csv'
        y_test_filepath = corrupted_dirpath / f'{openml_id}/y_test.csv'
        X_corrupted_train_filepath = corrupted_dirpath / f'{openml_id}/X_train.csv'
        X_corrupted_test_filepath = corrupted_dirpath / f'{openml_id}/X_test.csv'
        X_imputed_train_filepath = imputed_dirpath / f'{openml_id}/X_train.csv'
        X_imputed_test_filepath = imputed_dirpath / f'{openml_id}/X_test.csv'
        X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'

        if dataset_type == 'complete':
            X_groundtruth_train_filepath = groundtruth_dirpath / f'{openml_id}/X_train.csv'
            X_groundtruth_test_filepath = groundtruth_dirpath / f'{openml_id}/X_test.csv'
            y_train_filepath = groundtruth_dirpath / f'{openml_id}/y_train.csv'
            y_test_filepath = groundtruth_dirpath / f'{openml_id}/y_test.csv'
            X_corrupted_train_filepath = corrupted_dirpath / f'{openml_id}/{missingness}/X_train.csv'
            X_corrupted_test_filepath = corrupted_dirpath / f'{openml_id}/{missingness}/X_test.csv'
            X_imputed_train_filepath = imputed_dirpath / f'{openml_id}/{missingness}/X_train.csv'
            X_imputed_test_filepath = imputed_dirpath / f'{openml_id}/{missingness}/X_test.csv'

        X_imputed_train_filepath.parent.mkdir(parents=True, exist_ok=True)
        X_imputed_test_filepath.parent.mkdir(parents=True, exist_ok=True)

        imputation_experiment(
            args=args, timestamp=timestamp, openml_id=openml_id, train_or_test='train', missingness=missingness,
            X_groundtruth_filepath=X_groundtruth_train_filepath, X_corrupted_filepath=X_corrupted_train_filepath,
            X_imputed_filepath=X_imputed_train_filepath, X_categories_filepath=X_categories_filepath,
            results_filepath=imputation_results_filepath
        )

        imputation_experiment(
            args=args, timestamp=timestamp, openml_id=openml_id, train_or_test='test', missingness=missingness,
            X_groundtruth_filepath=X_groundtruth_test_filepath, X_corrupted_filepath=X_corrupted_test_filepath,
            X_imputed_filepath=X_imputed_test_filepath, X_categories_filepath=X_categories_filepath,
            results_filepath=imputation_results_filepath
        )
        
        if args.downstream:
            downstream_experiment(
                args=args, timestamp=timestamp, openml_id=openml_id, missingness=missingness,
                X_train_filepath=X_imputed_train_filepath, X_test_filepath=X_imputed_test_filepath,
                y_train_filepath=y_train_filepath, y_test_filepath=y_test_filepath,
                results_filepath=downstream_results_filepath, X_incomplete_filepath=X_corrupted_train_filepath
            )

    return


def imputation_experiment(args: argparse.Namespace, timestamp: str, openml_id: int, train_or_test: str, missingness: str | None, X_groundtruth_filepath: Path | None, X_corrupted_filepath: Path, X_imputed_filepath: Path, X_categories_filepath: Path, results_filepath: Path | None):
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

    if args.debug:
        print(f'Imputing OpenML Id: {openml_id}, Train/Test: {train_or_test}, Method: {args.method}')

    with open(X_categories_filepath, 'r') as f:
        X_categories = json.load(f)

    # Load data
    X_groundtruth = None
    if X_groundtruth_filepath is not None:
        X_groundtruth = pd.read_csv(X_groundtruth_filepath, header=0, dtype={column: str for column in X_categories.keys()})
    X_corrupted = pd.read_csv(X_corrupted_filepath, header=0, dtype={column: str for column in X_categories.keys()})

    n_missing_values = X_corrupted.isna().sum().sum()
    missing_columns = X_corrupted.columns[X_corrupted.isna().any()].tolist()
    missing_columns = {column: ('categorical' if column in X_categories.keys() else 'numerical') for column in missing_columns}

    # Impute missing values
    # For non ML-based imputation, we can use fit methods on test data
    if args.method == 'meanmode':
        imputer = MeanModeImputer(X_categories=X_categories)
    elif args.method == 'knn':
        imputer = KNNImputer(n_jobs=-1, X_categories=X_categories)
    elif args.method == 'rf':
        imputer = RandomForestImputer(n_jobs=-1, X_categories=X_categories)
    elif args.method == 'llm':
        data_dirpath = Path(__file__).parents[2] / 'data'
        openml_dirpath = data_dirpath / 'openml'
        description_file = openml_dirpath / f'{openml_id}/description.txt'
        description = description_file.read_text()
        imputer = LLMImputer(X_categories=X_categories, dataset_description=description, model='gpt-4')

    # Run imputation
    X_imputed = imputer.fit_transform(X_corrupted)

    # Save imputed data
    X_imputed.to_csv(X_imputed_filepath, index=False)

    # Evaluate imputation
    if args.evaluate and X_groundtruth is not None:
        evaluator = ImputationEvaluator(X_groundtruth, X_corrupted, X_imputed, X_categories)
        rmse, macro_f1 = evaluator.evaluate()
        rmse = list(rmse.values())[0] if rmse != {} else None
        macro_f1 = list(macro_f1.values())[0] if macro_f1 != {} else None

        if args.debug:
            print(f' - RMSE: {rmse}')
            print(f' - Macro F1 score: {macro_f1}')
            print('')

        with open(results_filepath, 'a') as f:
            # missing_columns = f'\"{missing_columns}\"'
            f.write(f'{timestamp},{args.method},{openml_id},{train_or_test},{list(missing_columns.keys())[0]},{list(missing_columns.values())[0]},{n_missing_values},{missingness},{rmse},{macro_f1}\n')

    return


def downstream_experiment(args: argparse.Namespace, timestamp: str, openml_id: int, missingness: str, X_train_filepath: Path, X_test_filepath: Path, y_train_filepath: Path, y_test_filepath: Path, results_filepath: Path, X_incomplete_filepath: Path, X_imputed_filepath: Path, X_categories_filepath: Path):
    if args.debug:
        print(f'Imputing OpenML Id: {openml_id}')

    with open(X_categories_filepath, 'r') as f:
        X_categories = json.load(f)

    # Load data
    X_train = pd.read_csv(X_train_filepath, header=0, dtype={column: str for column in X_categories.keys()})
    X_test = pd.read_csv(X_test_filepath, header=0, dtype={column: str for column in X_categories.keys()})
    y_train = pd.read_csv(y_train_filepath, header=0, keep_default_na=False, na_values=[''])
    y_test = pd.read_csv(y_test_filepath, header=0, keep_default_na=False, na_values=[''])
    X_incomplete = pd.read_csv(X_incomplete_filepath, header=0)

    categories = [X_categories[column] for column in X_train.columns if column in X_categories.keys()]

    missing_columns = X_incomplete.columns[X_incomplete.isna().any()].tolist()
    missing_columns = {column: ('categorical' if column in X_categories.keys() else 'numerical') for column in missing_columns}

    # encode categorical features
    categorical_features = X_categories.keys()
    if len(categorical_features) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=categories)
        encoder.fit(X_train[categorical_features])
        X_train = pd.concat([X_train.drop(categorical_features, axis=1), pd.DataFrame(encoder.transform(X_train[categorical_features]))], axis=1)
        X_test = pd.concat([X_test.drop(categorical_features, axis=1), pd.DataFrame(encoder.transform(X_test[categorical_features]))], axis=1)
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

    # Train
    clf = RandomForestClassifier(random_state=args.seed)
    clf.fit(X_train, y_train.values.ravel())

    # Test
    score = clf.score(X_test, y_test.values.ravel())
    
    with open(results_filepath, 'a') as f:
        f.write(f'{timestamp},{args.method},{openml_id},{list(missing_columns.keys())[0]},{list(missing_columns.values())[0]},{missingness},{score}\n')


def main():
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    args = config_args()

    data_dirpath = Path(__file__).parents[2] / 'data'
    openml_dirpath = data_dirpath / 'openml'
    output_dirpath = data_dirpath / f'output/imputation/{args.method}'
    output_dirpath.mkdir(parents=True, exist_ok=True)

    if 'complete' in args.dataset:
        input_dirpath = data_dirpath / 'working/complete'
        experiment(
            args=args, timestamp=timestamp, dataset_type='complete', openml_dirpath=openml_dirpath, input_dirpath=input_dirpath, output_dirpath=output_dirpath
        )

    if 'incomplete' in args.dataset:
        input_dirpath = data_dirpath / 'working/incomplete'
        experiment(
            args=args, timestamp=timestamp, dataset_type='incomplete', openml_dirpath=openml_dirpath, input_dirpath=input_dirpath, output_dirpath=output_dirpath
        )


if __name__ == '__main__':
    main()
