from pathlib import Path
from tqdm import tqdm
import argparse
import json

import pandas as pd
import numpy as np

from modules.missingvalues import MissingValuesGenerator


def load_dataset_list(openml_dir: Path):
    dataset_list_file = openml_dir / 'openml-datasets-CC18.csv'
    dataset_list = pd.read_csv(dataset_list_file)
    return dataset_list


def dataset_extraction(args: argparse.Namespace, dataset_list: pd.DataFrame, extraction_log_filepath: Path):
    candidate_datasets = dataset_list.copy()
    
    # Extract datasets that have no missing values
    with open(extraction_log_filepath, 'a') as f:
        removed_datasets = candidate_datasets[candidate_datasets['NumberOfMissingValues'] > 0.0]
        f.write(','.join(removed_datasets.columns.tolist()) + '\n')
        for _, row in removed_datasets.iterrows():
            f.write(','.join(map(str, row.tolist())) + '\n')

    candidate_datasets = candidate_datasets[candidate_datasets['NumberOfMissingValues'] == 0.0]
    
    # Extract datasets which samples are less than 50000
    with open(extraction_log_filepath, 'a') as f:
        removed_datasets = candidate_datasets[candidate_datasets['NumberOfInstances'] > 50000.0]
        f.write(','.join(removed_datasets.columns.tolist()) + '\n')
        for _, row in removed_datasets.iterrows():
            f.write(','.join(map(str, row.tolist())) + '\n')

    
    candidate_datasets = candidate_datasets[candidate_datasets['NumberOfInstances'] < 50000.0]
    
    if args.openml_id is not None:
        # If openml_id is specified, return the dataset with the given openml_id
        candidate_datasets = candidate_datasets[candidate_datasets['did'] == args.openml_id]
    
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


def main():
    # Load arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--openml_id', type=int, default=None)
    argparser.add_argument('--n_selected_datasets', type=int, default=-1)
    argparser.add_argument('--n_corrupted_rows', nargs='*', type=int, default=[50, 100, 150])
    argparser.add_argument('--n_corrupted_columns', type=int, default=1)
    argparser.add_argument('--column_type', nargs='*', type=str, default=['categorical', 'numerical'])
    argparser.add_argument('--seed', type=int, default=42)
    args = argparser.parse_args()

    np.random.seed(args.seed)
    
    data_dirpath = Path(__file__).parents[2] / 'data'

    # Load dataset list (default: /data/openml/openml-datasets-CC18.csv)
    openml_dirpath = data_dirpath / 'openml'
    dataset_list = load_dataset_list(openml_dirpath)
    
    incomplete_dirpath = data_dirpath / 'working/incomplete/'
    incomplete_dirpath.mkdir(parents=True, exist_ok=True)
    log_filepath = incomplete_dirpath / 'logs.csv'
    extraction_log_filepath = incomplete_dirpath / 'extraction_logs.csv'

    if not log_filepath.exists():
        with open(log_filepath, 'w') as f:
            f.write('openml_id,NumberOfInstancesWithMissingValues,missing_column_name,missing_column_type,missingness\n')
            
    if not extraction_log_filepath.exists():
        with open(extraction_log_filepath, 'w') as f:
            f.write('did,name,version,uploader,status,format,MajorityClassSize,MaxNominalAttDistinctValues,MinorityClassSize,NumberOfClasses,NumberOfFeatures,NumberOfInstances,NumberOfInstancesWithMissingValues,NumberOfMissingValues,NumberOfNumericFeatures,NumberOfSymbolicFeatures\n')

    # Filter dataset list
    dataset_list = dataset_extraction(args, dataset_list, extraction_log_filepath)
    if len(dataset_list) == 0:
        print('No dataset was found that meets the conditions')
        return

    if args.n_selected_datasets == -1:
        # If n_selected_datasets is not specified, use all datasets
        selected_datasets = dataset_list
    else:
        n_selected_datasets = int(args.n_selected_datasets) if args.openml_id is None else 1
        selected_datasets = dataset_list.sample(n_selected_datasets, random_state=args.seed)

    for openml_id in selected_datasets['did']:
        print(f'Generating missing values for OpenML ID: {openml_id}')

        # Load complete dataset
        X_original_filepath = openml_dirpath / f'{openml_id}/X.csv'
        X_original = pd.read_csv(X_original_filepath)
        
        # Load category dictionary
        X_categories_filepath = openml_dirpath / f'{openml_id}/X_categories.json'
        with open(X_categories_filepath, 'r') as f:
            X_categories = json.load(f)

        # create list of categorical and numerical columns (names)
        X_categorical_columns = list(X_categories.keys())
        X_numerical_columns = list(set(X_original.columns.tolist()) - set(X_categorical_columns))

        columns = X_original.columns.tolist()
        if args.column_type == ['categorical']:
            columns = X_categorical_columns
            if len(columns) == 0:
                print(f'OpenML ID {openml_id} has no categorical columns')
                continue
        elif args.column_type == ['numerical']:
            columns = X_numerical_columns
            if len(columns) == 0:
                print(f'OpenML ID {openml_id} has no numerical columns')
                continue

        n_corrupted_columns = min(args.n_corrupted_columns, len(columns))
        selected_columns = np.random.choice(columns, n_corrupted_columns)

        selected_columns_type = []
        for column in selected_columns:
            if column in X_categorical_columns:
                if selected_columns_type.count('categorical') == 0:
                    selected_columns_type.append('categorical')
            elif column in X_numerical_columns:
                if selected_columns_type.count('numerical') == 0:
                    selected_columns_type.append('numerical')

        generator = MissingValuesGenerator()

        for n_corrupted_rows in tqdm(args.n_corrupted_rows):
            for missingness in ['MNAR', 'MAR', 'MCAR']:
                X_incomplete = generator.generate(X_original, n_corrupted_rows, selected_columns, missingness)
                Path(incomplete_dirpath / f'{openml_id}').mkdir(parents=True, exist_ok=True)
                X_incomplete.to_csv(incomplete_dirpath / f'{openml_id}/X_{selected_columns}_{n_corrupted_rows}-{missingness}.csv', index=False)
                with open(log_filepath, 'a') as f:
                    f.write(f'{openml_id},{n_corrupted_rows},\"{selected_columns}\",\"{selected_columns_type}\",{missingness}\n')

    logs = pd.read_csv(log_filepath, header=0)
    logs = logs.drop_duplicates()
    logs.to_csv(log_filepath, index=False)


if __name__ == "__main__":
    main()
