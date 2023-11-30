from pathlib import Path
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np

from modules.missingvalues import MissingValuesGenerator


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_selected_datasets', type=int, default=10)
    argparser.add_argument('--n_corrupted_rows', nargs='*', type=int, default=[100, 300, 500])
    argparser.add_argument('--column_type', nargs='*', type=str, default=['categorical', 'numerical'])
    argparser.add_argument('--seed', type=int, default=42)
    args = argparser.parse_args()

    np.random.seed(args.seed)

    dataset_list_path = Path('../../data/openml-datasets-CC18.csv')
    dataset_list = pd.read_csv(dataset_list_path)

    complete_dataset_list = dataset_list[dataset_list['NumberOfMissingValues'] == 0.0]

    if 'categorical' in args.column_type:
        complete_dataset_list = complete_dataset_list[complete_dataset_list['NumberOfSymbolicFeatures'] > 0.0]
    if 'numerical' in args.column_type:
        complete_dataset_list = complete_dataset_list[complete_dataset_list['NumberOfNumericFeatures'] > 0.0]

    # Select a certain number of datasets at random
    selected_complete_datasets = complete_dataset_list.sample(args.n_selected_datasets, random_state=args.seed)
    print(selected_complete_datasets.head())

    log_path = Path('../../data/working/incomplete/logs.csv')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as f:
        f.write('openml_id, NumberOfInstancesWithMissingValues, column_name, column_type, missingness\n')

    for openml_id in tqdm(selected_complete_datasets['did']):
        # Load complete dataset
        X_complete_dataset_path = Path(f'../../data/openml/{openml_id}/X.csv')
        X_complete = pd.read_csv(X_complete_dataset_path)

        # Select one column at random
        columns = X_complete.columns
        categorical_columns = [
            column for column in columns
            if pd.api.types.is_categorical_dtype(X_complete[column])
        ]
        numerical_columns = [
            column for column in columns
            if pd.api.types.is_numeric_dtype(X_complete[column]) and column not in categorical_columns
        ]
        if args.column_type == ['categorical']:
            columns = categorical_columns
        elif args.column_type == ['numerical']:
            columns = numerical_columns
        selected_column = columns[np.random.randint(0, len(columns))]
        selected_column_type = 'categorical' if selected_column in categorical_columns else 'numerical'

        generator = MissingValuesGenerator()

        for n_corrupted_rows in args.n_corrupted_rows:
            for missingness in ['MNAR', 'MAR', 'MCAR']:
                X_incomplete = generator.generate(X_complete, n_corrupted_rows, selected_column, missingness)
                X_incomplete_path = Path(f'../../data/working/incomplete/{openml_id}/X_{selected_column}-{n_corrupted_rows}-{missingness}.csv')
                X_incomplete_path.parent.mkdir(parents=True, exist_ok=True)
                X_incomplete.to_csv(X_incomplete_path, index=False)
                
                with open(log_path, 'a') as f:
                    f.write(f'{openml_id}, {n_corrupted_rows}, {selected_column}, {selected_column_type}, {missingness}\n')


if __name__ == "__main__":
    main()
