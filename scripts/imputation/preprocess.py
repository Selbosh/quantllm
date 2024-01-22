import json
import math

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from modules.missingvalues import MissingValues


def config_args():
    """
    Configure command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_corrupted_rows_train', type=int, default=40)
    argparser.add_argument('--n_corrupted_rows_test', type=int, default=10)
    argparser.add_argument('--n_corrupted_columns', type=int, default=3)
    argparser.add_argument('--test_size', type=float, default=0.2)
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()


def load_dataset_list(openml_dir: Path):
    """
    Load the list of datasets.

    Args:
        openml_dir (Path): The path to the OpenML datasets.

    Returns:
        pd.DataFrame: The list of datasets.
    """
    dataset_list_file = openml_dir / 'openml-datasets-CC18.csv'
    return pd.read_csv(dataset_list_file)


def dataset_extraction(args: argparse.Namespace, dataset_list: pd.DataFrame, generate: bool):
    """
    Extracts datasets based on specified criteria.

    Extraction criteria:
        - The number of missing values is 0.

    Args:
        args (argparse.Namespace): The command line arguments.
        dataset_list (pd.DataFrame): The list of datasets.

    Returns:
        pd.DataFrame: The extracted datasets.
    """
    candidate_datasets = dataset_list.copy()

    # Extract datasets that have no missing values
    if generate:
        candidate_datasets = candidate_datasets[candidate_datasets['NumberOfMissingValues'] == 0.0]
    else:
        candidate_datasets = candidate_datasets[candidate_datasets['NumberOfMissingValues'] > 0.0]

    return candidate_datasets


def generate_missing_values(args: argparse.Namespace, X: pd.DataFrame, train_or_test: str, openml_id: int, categorical_columns: list, output_dirpath: Path, log_filepath: Path):
    """
    Generate and save incomplete datasets.

    Args:
        args (argparse.Namespace): The command line arguments.
        X (pd.DataFrame): The dataset.
        train_or_test (str): The type of dataset. 'train' or 'test'.
        target_columns (list): The list of columns to be corrupted.
        target_columns_type (list): The list of column types to be corrupted.
        openml_id (int): The OpenML ID of the dataset.
        incomplete_dirpath (Path): The path to save incomplete datasets.
        log_filepath (Path): The path to save the log of incomplete datasets.
    """
    if args.debug:
        print(f'Generating {train_or_test} dataset')
    n_corrupted_rows = args.n_corrupted_rows_train if train_or_test == 'train' else args.n_corrupted_rows_test
    n_corrupted_columns = min(args.n_corrupted_columns, len(X.columns))
    for missingness in tqdm(['MCAR', 'MAR', 'MNAR']):
        corruption = MissingValues(
            n_corrupted_rows=n_corrupted_rows, n_corrupted_columns=n_corrupted_columns, 
            missingness=missingness, seed=args.seed
        )
        X_corrupted = corruption.transform(X)

        corrupted_columns = X_corrupted.columns[X_corrupted.isnull().any()].tolist()
        n_corrupted_columns = len(corrupted_columns)
        n_corrupted_categorical_columns = len([column for column in corrupted_columns if column in categorical_columns])
        n_corrupted_numerical_columns = n_corrupted_columns - n_corrupted_categorical_columns

        if args.debug:
            # Test if the number of corrupted rows and columns are correct
            print(f'Number of corrupted rows: {X_corrupted.shape[0] - X_corrupted.dropna().shape[0]}')
            print(f'Number of corrupted columns (categorical/numerical): {n_corrupted_columns} ({n_corrupted_categorical_columns}/{n_corrupted_numerical_columns})')

        corrupted_filepath = output_dirpath / f'{openml_id}/{missingness}/X_{train_or_test}.csv'
        corrupted_filepath.parent.mkdir(parents=True, exist_ok=True)
        X_corrupted.to_csv(corrupted_filepath, index=False)

        with open(log_filepath, 'a') as f:
            f.write(f'{openml_id},{missingness},{train_or_test},{n_corrupted_rows},{n_corrupted_columns},{n_corrupted_categorical_columns},{n_corrupted_numerical_columns}\n')


def preprocess(args: argparse.Namespace, input_dirpath: Path, output_dirpath: Path, generate: bool):
    """
    Create data for

    Args:
        args (argparse.Namespace): The command line arguments.
        selected_datasets (pd.DataFrame): The list of selected datasets.
        openml_dirpath (Path): The path to the OpenML datasets.
        complete_dirpath (Path): The path to save splitted complete datasets.
        incomplete_dirpath (Path): The path to save incomplete datasets.
        log_filepath (Path): The path to save the log of incomplete datasets.
    """
    original_dirpath = output_dirpath / 'original' if generate else output_dirpath
    corrupted_dirpath = output_dirpath / 'corrupted' if generate else None
    original_dirpath.mkdir(parents=True, exist_ok=True)
    corrupted_dirpath.mkdir(parents=True, exist_ok=True) if generate else None

    log_filepath = output_dirpath / 'logs.csv'
    if not log_filepath.exists():
        with open(log_filepath, 'w') as f:
            f.write('openml_id,missingness,train_or_test,n_corrupted_rows,n_corrupted_columns,n_corrupted_categorical_columns,n_corrupted_numerical_columns\n')

    dataset_list = load_dataset_list(input_dirpath)
    target_datasets = dataset_extraction(args=args, dataset_list=dataset_list, generate=generate)
    if len(target_datasets) == 0:
        ValueError('No dataset was found that meets the conditions')

    for openml_id in target_datasets['did']:
        print(f'# Processing OpenML ID {openml_id}')

        Path(original_dirpath / str(openml_id)).mkdir(parents=True, exist_ok=True)

        X = pd.read_csv(input_dirpath / f'{openml_id}/X.csv')
        y = pd.read_csv(input_dirpath / f'{openml_id}/y.csv', keep_default_na=False, na_values=[''])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        # Fetch lists of categorical and numerical columns
        X_categories_filepath = input_dirpath / f'{openml_id}/X_categories.json'
        with open(X_categories_filepath, 'r') as f:
            X_categories = json.load(f)
        categorical_columns = list(X_categories.keys())
        numerical_columns = [column for column in X.columns.tolist() if column not in categorical_columns]

        if generate:
            # Convert categorical columns to string
            X_train[categorical_columns] = X_train[categorical_columns].astype(str)
            X_test[categorical_columns] = X_test[categorical_columns].astype(str)

        X_train.to_csv(original_dirpath / f'{openml_id}/X_train.csv', index=False)
        X_test.to_csv(original_dirpath / f'{openml_id}/X_test.csv', index=False)
        y_train.to_csv(original_dirpath / f'{openml_id}/y_train.csv', index=False)
        y_test.to_csv(original_dirpath / f'{openml_id}/y_test.csv', index=False)

        if generate:
            Path(corrupted_dirpath / str(openml_id)).mkdir(parents=True, exist_ok=True)

            generate_missing_values(
                args=args, X=X_train, train_or_test='train', openml_id=openml_id, 
                categorical_columns=categorical_columns, output_dirpath=corrupted_dirpath, 
                log_filepath=log_filepath
            )
            generate_missing_values(
                args=args, X=X_test, train_or_test='test', openml_id=openml_id, 
                categorical_columns=categorical_columns, output_dirpath=corrupted_dirpath, 
                log_filepath=log_filepath
            )
        else:
            n_corrupted_rows = X_train.shape[0] - X_train.dropna().shape[0]
            corrupted_columns = X_train.columns[X_train.isnull().any()].tolist()
            n_corrupted_columns = len(corrupted_columns)
            n_corrupted_categorical_columns = len([column for column in corrupted_columns if column in categorical_columns])
            n_corrupted_numerical_columns = n_corrupted_columns - n_corrupted_categorical_columns
            with open(log_filepath, 'a') as f:
                f.write(f'{openml_id},,train,{n_corrupted_rows},{n_corrupted_columns},{n_corrupted_categorical_columns},{n_corrupted_numerical_columns}\n')

            n_corrupted_rows = X_test.shape[0] - X_test.dropna().shape[0]
            corrupted_columns = X_train.columns[X_train.isnull().any()].tolist()
            n_corrupted_columns = len(corrupted_columns)
            n_corrupted_categorical_columns = len([column for column in corrupted_columns if column in categorical_columns])
            n_corrupted_numerical_columns = n_corrupted_columns - n_corrupted_categorical_columns
            with open(log_filepath, 'a') as f:
                f.write(f'{openml_id},,test,{n_corrupted_rows},{n_corrupted_columns},{n_corrupted_categorical_columns},{n_corrupted_numerical_columns}\n')

    logs = pd.read_csv(log_filepath, header=0)
    logs = logs.drop_duplicates()
    logs.to_csv(log_filepath, index=False)


def main():
    args = config_args()

    np.random.seed(args.seed)

    data_dirpath = Path(__file__).parents[2] / 'data'
    openml_dirpath = data_dirpath / 'openml'
    working_dirpath = data_dirpath / 'working'

    # Split complete datasets into train and test, and generate missing values for train and test subsets.
    preprocess(
        args=args, 
        input_dirpath=openml_dirpath, output_dirpath=working_dirpath / 'complete', 
        generate=True
    )

    # Split incomplete datasets into train and test.
    preprocess(
        args=args, 
        input_dirpath=openml_dirpath, output_dirpath=working_dirpath / 'incomplete', 
        generate=False
    )


if __name__ == "__main__":
    main()
