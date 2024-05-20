from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import json
from tqdm import tqdm

from modules.evaluator import ImputationEvaluator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder


def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, default='meanmode',
                           choices=['meanmode', 'knn', 'rf', 'llm'])
    argparser.add_argument('--downstream', action='store_true')
    argparser.add_argument('--downstreambaseline', type=str, default=None,
                           choices=['train_complete', 'train_incomplete'])
    argparser.add_argument('--openml_id', type=int, default=None)
    argparser.add_argument('--missingness', type=str, default=None,
                           choices=['MCAR', 'MAR', 'MNAR'])
    argparser.add_argument('--dataset', nargs='*', type=str, 
                           default=['incomplete', 'complete'])
    argparser.add_argument('--llm_model', type=str, default='gpt-4')
    argparser.add_argument('--llm_role', type=str, default='expert',
                           choices=['expert', 'nonexpert'])
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()


def evaluation(args: argparse.Namespace, 
               timestamp: str,
               dataset_type: str,
               openml_dirpath: Path,
               input_dirpath: Path,
               output_dirpath: Path):
    imputation_results_filepath = output_dirpath / f'imputation_{dataset_type}_{timestamp}.csv'
    with open(imputation_results_filepath, 'w') as f:
        f.write('timestamp,method,openml_id,missingness,missing_column_name,missing_column_type,n_missing_values,rmse,macro_f1\n')

    downstream_results_filepath = output_dirpath / f'downstream_{dataset_type}_{timestamp}.csv'
    if args.downstream:
        with open(downstream_results_filepath, 'w') as f:
            f.write('timestamp,method,openml_id,missingness,accuracy,macro_f1\n')

    complete_dirpath = None
    corrupted_dirpath = input_dirpath
    imputed_dirpath = output_dirpath / 'imputed'
    if dataset_type == 'complete':
        complete_dirpath = input_dirpath / 'original'
        corrupted_dirpath = input_dirpath / 'corrupted'

    logs = pd.read_csv(input_dirpath / 'logs.csv', header=0).loc[:, ['openml_id', 'missingness']].drop_duplicates()
    if args.openml_id is not None:
        logs = logs[logs.openml_id == args.openml_id].drop_duplicates()
    if args.missingness is not None:
        logs = logs[logs.missingness == args.missingness].drop_duplicates()

    for log in tqdm(logs.itertuples(), total=len(logs)):
        openml_id, missingness = log.openml_id, log.missingness

        X_complete_train_filepath = None
        X_complete_test_filepath = None
        y_train_filepath = corrupted_dirpath / f'{openml_id}/y_train.csv'
        y_test_filepath = corrupted_dirpath / f'{openml_id}/y_test.csv'
        X_corrupted_train_filepath = corrupted_dirpath / f'{openml_id}/X_train.csv'
        X_corrupted_test_filepath = corrupted_dirpath / f'{openml_id}/X_test.csv'
        X_imputed_train_filepath = imputed_dirpath / f'{openml_id}/X_train.csv'
        X_imputed_test_filepath = imputed_dirpath / f'{openml_id}/X_test.csv'
        X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'

        if dataset_type == 'complete':
            X_complete_train_filepath = complete_dirpath / f'{openml_id}/X_train.csv'
            X_complete_test_filepath = complete_dirpath / f'{openml_id}/X_test.csv'
            y_train_filepath = complete_dirpath / f'{openml_id}/y_train.csv'
            y_test_filepath = complete_dirpath / f'{openml_id}/y_test.csv'
            X_corrupted_train_filepath = corrupted_dirpath / f'{openml_id}/{missingness}/X_train.csv'
            X_corrupted_test_filepath = corrupted_dirpath / f'{openml_id}/{missingness}/X_test.csv'
            X_imputed_train_filepath = imputed_dirpath / f'{openml_id}/{missingness}/X_train.csv'
            X_imputed_test_filepath = imputed_dirpath / f'{openml_id}/{missingness}/X_test.csv'

            if X_imputed_train_filepath.exists() is False or X_imputed_test_filepath.exists() is False:
                continue

            try:
                with open(X_categories_filepath, 'r') as f:
                    X_categories = json.load(f)
                X_groundtruth_train = pd.read_csv(X_complete_train_filepath, header=0, dtype={column: str for column in X_categories.keys()})
                X_groundtruth_test = pd.read_csv(X_complete_test_filepath, header=0, dtype={column: str for column in X_categories.keys()})
                X_groundtruth = pd.concat([X_groundtruth_train, X_groundtruth_test], axis=0)

                X_corrupted_train = pd.read_csv(X_corrupted_train_filepath, header=0, dtype={column: str for column in X_categories.keys()})
                X_corrupted_test = pd.read_csv(X_corrupted_test_filepath, header=0, dtype={column: str for column in X_categories.keys()})
                X_corrupted = pd.concat([X_corrupted_train, X_corrupted_test], axis=0)

                X_imputed_train = pd.read_csv(X_imputed_train_filepath, header=0, dtype={column: str for column in X_categories.keys()})
                X_imputed_test = pd.read_csv(X_imputed_test_filepath, header=0, dtype={column: str for column in X_categories.keys()})
                X_imputed = pd.concat([X_imputed_train, X_imputed_test], axis=0)

                imputation_evaluation(
                    args=args, timestamp=timestamp, openml_id=openml_id, missingness=missingness,
                    X_groundtruth=X_groundtruth, X_corrupted=X_corrupted, X_imputed=X_imputed, X_categories=X_categories,
                    results_filepath=imputation_results_filepath
                )
            except Exception as e:
                print(f'[ERROR] OpenML Id: {openml_id}, Train/Test: train, Method: {args.method}\n----\n{e}\n----')

        if args.downstream:
            try:
                downstream_evaluation(
                    args=args, timestamp=timestamp, openml_id=openml_id, missingness=missingness,
                    X_train_filepath=X_imputed_train_filepath, X_test_filepath=X_imputed_test_filepath,
                    y_train_filepath=y_train_filepath, y_test_filepath=y_test_filepath,
                    results_filepath=downstream_results_filepath, X_incomplete_filepath=X_corrupted_train_filepath,
                    X_categories_filepath=X_categories_filepath
                )
            except Exception as e:
                print(f'Error in OpenML Id: {openml_id}, Train/Test: train, Method: {args.method}')
                print(e)

    return


def imputation_evaluation(args: argparse.Namespace, 
                          timestamp: str, 
                          openml_id: int, 
                          missingness: str | None, 
                          X_groundtruth: pd.DataFrame, 
                          X_corrupted: pd.DataFrame, 
                          X_imputed: pd.DataFrame, 
                          X_categories: dict, 
                          results_filepath: Path | None):
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
    print(f'# Imputation evaluation OpenML Id: {openml_id}, Missingness: {missingness}, Method: {args.method}')

    missing_columns = X_corrupted.columns[X_corrupted.isna().any()].tolist()
    missing_columns = {column: ('categorical' if column in X_categories.keys() else 'numerical') for column in missing_columns}

    # Evaluate imputation
    for missing_column, missing_column_type in missing_columns.items():
        evaluator = ImputationEvaluator(X_groundtruth, X_corrupted, X_imputed, X_categories)
        rmse, macro_f1 = evaluator.evaluate(missing_column)
        n_missing_values = X_corrupted[missing_column].isna().sum()
        if args.debug:
            print(f' - RMSE: {rmse}')
            print(f' - Macro F1 score: {macro_f1}')
            print('')
        with open(results_filepath, 'a') as f:
            f.write(f'{timestamp},{args.method},{openml_id},{missingness},{missing_column},{missing_column_type},{n_missing_values},{rmse},{macro_f1}\n')
    return


def downstream_evaluation(args: argparse.Namespace,
                          timestamp: str,
                          openml_id: int,
                          missingness: str,
                          X_train_filepath: Path,
                          X_test_filepath: Path,
                          y_train_filepath: Path,
                          y_test_filepath: Path,
                          results_filepath: Path,
                          X_incomplete_filepath: Path,
                          X_categories_filepath: Path):
    if args.debug:
        print(f'Imputing OpenML Id: {openml_id}')

    with open(X_categories_filepath, 'r') as f:
        X_categories = json.load(f)

    # Load data
    X_train = pd.read_csv(X_train_filepath, header=0, dtype={column: str for column in X_categories.keys()})
    X_test = pd.read_csv(X_test_filepath, header=0, dtype={column: str for column in X_categories.keys()})
    y_train = pd.read_csv(y_train_filepath, header=0, keep_default_na=False, na_values=[''])
    y_test = pd.read_csv(y_test_filepath, header=0, keep_default_na=False, na_values=[''])
    X_incomplete = pd.read_csv(X_incomplete_filepath, header=0) if X_incomplete_filepath is not None else None

    categories = [X_categories[column] for column in X_train.columns if column in X_categories.keys()]

    if X_incomplete is not None:
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
    y_pred = clf.predict(X_test)
    # predict_proba = clf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    with open(results_filepath, 'a') as f:
        f.write(f'{timestamp},{args.method},{openml_id},{missingness},{acc},{macro_f1}\n')


def downstream_baseline_evaluation(args: argparse.Namespace,
                                   timestamp: str,
                                   dataset_type: str,
                                   openml_dirpath: Path,
                                   input_dirpath: Path,
                                   output_dirpath: Path):
    downstream_results_filepath = output_dirpath / f'downstream_{dataset_type}_{timestamp}.csv'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    if args.downstream:
        with open(downstream_results_filepath, 'w') as f:
            f.write('timestamp,method,openml_id,missingness,accuracy,macro_f1\n')

    logs = pd.read_csv(input_dirpath / 'logs.csv', header=0).loc[:, ['openml_id', 'missingness']].drop_duplicates()
    if args.openml_id is not None:
        logs = logs[logs.openml_id == args.openml_id].drop_duplicates()
    if args.missingness is not None:
        logs = logs[logs.missingness == args.missingness].drop_duplicates()

    for log in tqdm(logs.itertuples(), total=len(logs)):
        openml_id, missingness = log.openml_id, log.missingness

        X_train_filepath = input_dirpath / f'{openml_id}/X_train.csv'
        X_test_filepath = input_dirpath / f'{openml_id}/X_test.csv'
        y_train_filepath = input_dirpath / f'{openml_id}/y_train.csv'
        y_test_filepath = input_dirpath / f'{openml_id}/y_test.csv'
        X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'

        if dataset_type == 'complete':
            complete_dirpath = input_dirpath / 'original'
            incomplete_dirpath = input_dirpath / 'corrupted'
            if args.downstreambaseline == 'train_complete':
                X_train_filepath = complete_dirpath / f'{openml_id}/X_train.csv'
            else:
                X_train_filepath = incomplete_dirpath / f'{openml_id}/{missingness}/X_train.csv'
            X_test_filepath = incomplete_dirpath / f'{openml_id}/{missingness}/X_test.csv'
            y_train_filepath = complete_dirpath / f'{openml_id}/y_train.csv'
            y_test_filepath = complete_dirpath / f'{openml_id}/y_test.csv'

        downstream_evaluation(
            args=args, timestamp=timestamp, openml_id=openml_id, missingness=missingness,
            X_train_filepath=X_train_filepath, X_test_filepath=X_test_filepath,
            y_train_filepath=y_train_filepath, y_test_filepath=y_test_filepath,
            results_filepath=downstream_results_filepath, X_incomplete_filepath=None,
            X_categories_filepath=X_categories_filepath
        )

    return


def main():
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    args = config_args()

    data_dirpath = Path(__file__).parents[2] / 'data'
    openml_dirpath = data_dirpath / 'openml'
    imputation_dirpath = data_dirpath / 'output/imputation'

    output_dirpath = imputation_dirpath / args.method
    if args.method == 'llm':
        output_dirpath = output_dirpath / f'{args.llm_role}/{args.llm_model}'

    if args.downstreambaseline == 'train_complete':
        output_dirpath = imputation_dirpath / 'baseline/train_complete'
    if args.downstreambaseline == 'train_incomplete':
        output_dirpath = imputation_dirpath / 'baseline/train_incomplete'
    output_dirpath.mkdir(parents=True, exist_ok=True)

    if 'complete' in args.dataset:
        input_dirpath = data_dirpath / 'working/complete'
        if args.downstreambaseline:
            downstream_baseline_evaluation(
                args=args,
                timestamp=timestamp,
                dataset_type='complete',
                openml_dirpath=openml_dirpath,
                input_dirpath=input_dirpath,
                output_dirpath=output_dirpath
            )
        else:
            evaluation(
                args=args,
                timestamp=timestamp,
                dataset_type='complete',
                openml_dirpath=openml_dirpath,
                input_dirpath=input_dirpath,
                output_dirpath=output_dirpath
            )

    if 'incomplete' in args.dataset:
        input_dirpath = data_dirpath / 'working/incomplete'
        evaluation(
            args=args,
            timestamp=timestamp,
            dataset_type='incomplete',
            openml_dirpath=openml_dirpath,
            input_dirpath=input_dirpath,
            output_dirpath=output_dirpath
        )

    return


if __name__ == '__main__':
    main()
