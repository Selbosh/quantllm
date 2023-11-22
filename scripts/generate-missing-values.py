from pathlib import Path
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np

from jenga.corruptions.generic import MissingValues


original_data_dir = Path(__file__).parents[1] / 'data' / 'openml'


def generate_corupptions(openml_id, na_value=np.nan, logs=False):
    print(f'Generating corruptions for OpenML ID: {openml_id}')

    openml_id_dir = original_data_dir / f'{openml_id}'
    X_path = openml_id_dir / 'X.csv'
    X = pd.read_csv(X_path)
    
    save_dir = Path(__file__).parents[1] / 'data' / 'working' / 'generated-missing-values' / f'{openml_id}'
    save_dir.mkdir(parents=True, exist_ok=True)

    columns = X.columns

    if logs:
        with open(save_dir / f'corruptions.csv', 'w') as f:
            f.write('column,fraction,missingness,na_value\n')

    for column in tqdm(columns):
        for fraction in [0.01, 0.1, 0.3, 0.5]:
            for missingness in ['MNAR', 'MAR', 'MCAR']:
                corruption = MissingValues(column=column, fraction=fraction, missingness=missingness, na_value=na_value)

                X_corrupted = corruption.transform(X)
                X_corrupted.to_csv(save_dir / f'X_{column}-{fraction}-{missingness}.csv', index=False)
                
                if logs:
                    with open(save_dir / f'corruptions.csv', 'a') as f:
                        f.write(f"{column},{fraction},{missingness},{na_value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openml_id', type=int, help='OpenML ID of the dataset to generate corruptions for')
    parser.add_argument('--all', action='store_true', help='Generate corruptions for all datasets')
    parser.add_argument('--logs', action='store_true', help='Log corruption patterns')
    args = parser.parse_args()
    if not args.openml_id and not args.all:
        raise ValueError('Either --openml_id [openml_id: int > 0] or --all must be specified')

    if args.openml_id:
        generate_corupptions(args.openml_id, logs=args.logs)

    if args.all:
        dirs = list(original_data_dir.glob('*'))
        for dir in dirs:
            openml_id = dir.name
            generate_corupptions(openml_id, logs=args.logs)


if __name__ == "__main__":
    main()
