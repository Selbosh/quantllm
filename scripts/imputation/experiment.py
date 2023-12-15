from pathlib import Path
import argparse
import datetime as dt
import pandas as pd

from modules.imputer import MeanModeImputer, KNNImputer, RandomForestImputer
from modules.llmimputer import LLMImputer
from modules.evaluator import ImputationEvaluator


def experiment(args: argparse.Namespace, openml_id: int, X_original_file: Path, X_incomplete_file: Path, X_imputed_file: Path, results_file: Path):
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if args.debug:
        print(f'Imputing OpenML Id: {openml_id}')

    # Load data
    X_original = pd.read_csv(X_original_file, header=0)
    X_incomplete = pd.read_csv(X_incomplete_file, header=0)

    n_missing_values = X_incomplete.isna().sum().sum()
    missing_columns = X_incomplete.columns[X_incomplete.isna().any()].tolist()
    missingness = X_incomplete_file.stem.split('-')[-1]

    # Impute missing values
    # For non ML-based imputation, we can use fit methods on test data
    if args.method == 'meanmode':
        imputer = MeanModeImputer()
    elif args.method == 'knn':
        imputer = KNNImputer()
    elif args.method == 'rf':
        imputer = RandomForestImputer()
    elif args.method == 'llm':
        description_file = openml_dir / f'{openml_id}/description.txt'
        description = description_file.read_text()
        imputer = LLMImputer()

    # Run imputation
    X_imputed = imputer.fit_transform(X_incomplete)

    # Save imputed data
    X_imputed.to_csv(X_imputed_file, index=False)

    # Evaluate imputation
    evaluator = ImputationEvaluator()
    rmse, macro_f1 = evaluator.evaluate(X_original, X_incomplete, X_imputed)

    if args.debug:
        print(f'\tRMSE: {rmse}')
        print(f'\tMacro F1 score: {macro_f1}')

    with open(results_file, 'a') as f:
        missing_columns, rmse, macro_f1 = f'\"{missing_columns}\"', f'\"{rmse}\"', f'\"{macro_f1}\"'
        f.write(f'{timestamp}, {args.method}, {openml_id}, {missing_columns}, {n_missing_values}, {missingness}, {rmse}, {macro_f1}\n')

    return


def main():
    # Arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--method', type=str, default='meanmode')
    argparser.add_argument('--debug', action='store_true')
    args = argparser.parse_args()

    output_dir = Path(__file__).parents[2] / f'data/output/imputation/{args.method}'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'results.csv'
    with open(results_file, 'w') as f:
        f.write('timestamp, method, openml_id, missing_column, n_missing_values, missingness, rmse, macro_f1\n')

    # load incomplete data
    incomplete_dir = Path(__file__).parents[2] / 'data/working/incomplete'
    incomplete_log_file = incomplete_dir / 'logs.csv'
    incomplete_log = pd.read_csv(incomplete_log_file)

    # path to original (complete) opanml datasets
    openml_dir = Path(__file__).parents[2] / 'data/openml'

    # run experiment for each dataset
    for openml_id in incomplete_log['openml_id']:
        original_file = openml_dir / f'{openml_id}/X.csv'
        incomplete_id_dir = incomplete_dir / f'{openml_id}'

        for incomplete_file in incomplete_id_dir.glob('*.csv'):
            imputed_dir = output_dir / f'imputed_data/{openml_id}'
            imputed_dir.mkdir(parents=True, exist_ok=True)
            X_imputed_file = imputed_dir / f'{incomplete_file.stem}.csv'

            experiment(args, openml_id, original_file, incomplete_file, X_imputed_file, results_file)

    return


if __name__ == '__main__':
    main()
