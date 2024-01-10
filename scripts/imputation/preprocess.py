from pathlib import Path
from tqdm import tqdm
import argparse
import json
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from modules.missingvalues import MissingValues


def load_dataset_list(openml_dir: Path):
    '''
    Load the list of datasets.

    Args:
        openml_dir (Path): The path to the OpenML datasets.

    Returns:
        pd.DataFrame: The list of datasets.
    '''
    dataset_list_file = openml_dir / 'openml-datasets-CC18.csv'
    return pd.read_csv(dataset_list_file)


def dataset_extraction(args: argparse.Namespace, dataset_list: pd.DataFrame, extraction_log_filepath: Path):
    '''
    Extracts datasets based on specified criteria.

    Extraction criteria:
        - The number of missing values is 0.
        - The number of samples is less than 50,000.

    Args:
        args (argparse.Namespace): The command line arguments.
        dataset_list (pd.DataFrame): The list of datasets.
        extraction_log_filepath (Path): The filepath to the extraction log.

    Returns:
        pd.DataFrame: The extracted datasets.
    '''
    candidate_datasets = dataset_list.copy()

    # Extract datasets that have no missing values
    with open(extraction_log_filepath, 'a') as f:
        removed_datasets = candidate_datasets[candidate_datasets['NumberOfMissingValues'] > 0.0]
        f.write(','.join(removed_datasets.columns.tolist()) + '\n')
        for _, row in removed_datasets.iterrows():
            f.write(','.join(map(str, row.tolist())) + '\n')
    candidate_datasets = candidate_datasets[candidate_datasets['NumberOfMissingValues'] == 0.0]

    # Extract datasets which samples are less than 50,000
    with open(extraction_log_filepath, 'a') as f:
        removed_datasets = candidate_datasets[candidate_datasets['NumberOfInstances'] > 50000.0]
        f.write(','.join(removed_datasets.columns.tolist()) + '\n')
        for _, row in removed_datasets.iterrows():
            f.write(','.join(map(str, row.tolist())) + '\n')
    candidate_datasets = candidate_datasets[candidate_datasets['NumberOfInstances'] < 50000.0]

    if 'categorical' in args.column_type and 'numerical' in args.column_type and args.n_corrupted_columns == 1:
        return candidate_datasets

    if 'categorical' in args.column_type:
        # Extract datasets that have at least one categorical column
        # Note: 'NumberOfSymbolicFeatures' includes the target column
        candidate_datasets = candidate_datasets[(candidate_datasets['NumberOfSymbolicFeatures'] - 1.0) > 0.0]

    if 'numerical' in args.column_type:
        # Extract datasets that have at least one numerical column
        candidate_datasets = candidate_datasets[candidate_datasets['NumberOfNumericFeatures'] > 0.0]

    return candidate_datasets


def process_datasets(args: argparse.Namespace, selected_datasets: pd.DataFrame, openml_dirpath: Path, complete_dirpath: Path, incomplete_dirpath: Path, log_filepath: Path):
    '''
    Generate missing values for all selected datasets.

    Args:
        args (argparse.Namespace): The command line arguments.
        selected_datasets (pd.DataFrame): The list of selected datasets.
        openml_dirpath (Path): The path to the OpenML datasets.
        complete_dirpath (Path): The path to save splitted complete datasets.
        incomplete_dirpath (Path): The path to save incomplete datasets.
        log_filepath (Path): The path to save the log of incomplete datasets.
    '''
    for openml_id in selected_datasets['did']:
        print(f'Generating missing values for OpenML ID: {openml_id}')

        Path(complete_dirpath / str(openml_id)).mkdir(parents=True, exist_ok=True)
        Path(incomplete_dirpath / str(openml_id)).mkdir(parents=True, exist_ok=True)

        X_original_filepath = openml_dirpath / f'{openml_id}/X.csv'
        y_original_filepath = openml_dirpath / f'{openml_id}/y.csv'
        X_original = pd.read_csv(X_original_filepath)
        y_original = pd.read_csv(y_original_filepath, keep_default_na=False, na_values=[''])

        # Split dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X_original, y_original, test_size=args.test_size, random_state=args.seed
        )

        # Fetch lists of categorical and numerical columns
        X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'
        with open(X_categories_filepath, 'r') as f:
            X_categories = json.load(f)
        X_categorical_columns = list(X_categories.keys())
        X_numerical_columns = [column for column in X_original.columns.tolist() if column not in X_categorical_columns]

        X_train[X_categorical_columns] = X_train[X_categorical_columns].astype(str)
        X_test[X_categorical_columns] = X_test[X_categorical_columns].astype(str)

        X_train.to_csv(complete_dirpath / f'{openml_id}/X_train.csv', index=False)
        X_test.to_csv(complete_dirpath / f'{openml_id}/X_test.csv', index=False)
        y_train.to_csv(complete_dirpath / f'{openml_id}/y_train.csv', index=False)
        y_test.to_csv(complete_dirpath / f'{openml_id}/y_test.csv', index=False)

        target_column = select_column(args, openml_id, X_categorical_columns, X_numerical_columns)

        target_column_type = 'categorical' if target_column in X_categorical_columns else 'numerical'

        generate_and_save_incomplete_datasets(
            args, X_train, 'train', target_column, target_column_type, openml_id, incomplete_dirpath, log_filepath
        )
        generate_and_save_incomplete_datasets(
            args, X_test, 'test', target_column, target_column_type, openml_id, incomplete_dirpath, log_filepath
        )


def select_column(args: argparse.Namespace, openml_id: int, X_categorical_columns: list, X_numerical_columns: list):
    '''
    Select columns to be corrupted.

    Args:
        args (argparse.Namespace): The command line arguments.
        openml_id (int): The OpenML ID of the dataset.
        X_categorical_columns (list): The list of categorical columns.
        X_numerical_columns (list): The list of numerical columns.

    Returns:
        list: The list of columns to be corrupted.
    '''
    columns = []
    if args.column_type == ['categorical']:
        if len(X_categorical_columns) == 0 and args.debug:
            print(f'OpenML ID {openml_id} has no categorical columns')
        columns = X_categorical_columns
    elif args.column_type == ['numerical']:
        if len(X_numerical_columns) == 0 and args.debug:
            print(f'OpenML ID {openml_id} has no numerical columns')
            return []
        columns = X_numerical_columns
    else:
        columns = X_categorical_columns + X_numerical_columns

    return np.random.choice(columns, size=None, replace=False)


def generate_and_save_incomplete_datasets(args: argparse.Namespace, X: pd.DataFrame, train_or_test: str, target_column: str, target_column_type: str, openml_id: int, incomplete_dirpath: Path, log_filepath: Path):
    '''
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
    '''
    n_corrupted_rows = args.n_corrupted_rows_train if train_or_test == 'train' else args.n_corrupted_rows_test
    for missingness in tqdm(['MNAR', 'MAR', 'MCAR']):
        corruption = MissingValues(column=target_column, n_corrupted_rows=n_corrupted_rows, missingness=missingness, seed=args.seed)
        X_incomplete = corruption.transform(X)

        incomplete_filepath = incomplete_dirpath / f'{openml_id}/{missingness}/X_{train_or_test}.csv'
        incomplete_filepath.parent.mkdir(parents=True, exist_ok=True)
        X_incomplete.to_csv(incomplete_filepath, index=False)
        with open(log_filepath, 'a') as f:
            f.write(f'{openml_id},{train_or_test},{n_corrupted_rows},{target_column},{target_column_type},{missingness}\n')


def config_args():
    '''
    Configure command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_corrupted_rows_train', type=int, default=120)
    argparser.add_argument('--n_corrupted_rows_test', type=int, default=30)
    argparser.add_argument('--n_corrupted_columns', type=int, default=1)
    argparser.add_argument('--column_type', nargs='*', type=str, default=['categorical', 'numerical'])
    argparser.add_argument('--test_size', type=float, default=0.2)
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()


def main():
    args = config_args()

    # Set random seed. This is used to select datasets and columns to be corrupted.
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    data_dirpath = Path(__file__).parents[2] / 'data'

    # Load dataset list (default: /data/openml/openml-datasets-CC18.csv)
    openml_dirpath = data_dirpath / 'openml'
    dataset_list = load_dataset_list(openml_dirpath)
    
    # Set paths to save incomplete datasets and logs
    working_dirpath = data_dirpath / 'working'
    complete_dirpath = working_dirpath / 'complete'
    incomplete_dirpath = working_dirpath / 'incomplete'
    complete_dirpath.mkdir(parents=True, exist_ok=True)
    incomplete_dirpath.mkdir(parents=True, exist_ok=True)
    log_filepath = incomplete_dirpath / 'logs.csv' # log of incomplete datasets
    if not log_filepath.exists():
        with open(log_filepath, 'w') as f:
            f.write('openml_id,train_or_test,NumberOfInstancesWithMissingValues,missing_column_name,missing_column_type,missingness\n')
    extraction_log_filepath = incomplete_dirpath / 'extraction_logs.csv' # this log is used to check which datasets are removed by the extraction process
    if not extraction_log_filepath.exists():
        with open(extraction_log_filepath, 'w') as f:
            f.write('did,name,version,uploader,status,format,MajorityClassSize,MaxNominalAttDistinctValues,MinorityClassSize,NumberOfClasses,NumberOfFeatures,NumberOfInstances,NumberOfInstancesWithMissingValues,NumberOfMissingValues,NumberOfNumericFeatures,NumberOfSymbolicFeatures\n')

    # List datasets that meet the conditions
    selected_datasets = dataset_extraction(args, dataset_list, extraction_log_filepath)
    if len(selected_datasets) == 0:
        ValueError('No dataset was found that meets the conditions')

    # Generate missing values for all selected datasets
    process_datasets(args, selected_datasets, openml_dirpath, complete_dirpath, incomplete_dirpath, log_filepath)

    logs = pd.read_csv(log_filepath, header=0)
    logs = logs.drop_duplicates()
    logs.to_csv(log_filepath, index=False)


if __name__ == "__main__":
    main()
