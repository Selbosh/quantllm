from pathlib import Path
import argparse
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import json


def calculate_rank(x, method, metric):
    records = [{'score': x['meanmode'], 'method': 'meanmode'},
            {'score': x['knn'], 'method': 'knn'},
            {'score': x['rf'], 'method': 'rf'}]
    ranking = {'meanmode': 0, 'knn': 0, 'rf': 0}
    rec_sort = list(sorted(records, key=lambda r:r['score'], reverse=(metric == 'rmse')))
    rank = 1
    count = 0
    last_score = None
    for r in rec_sort:
        if last_score != r['score']:
            rank += count
            count = 0
            last_score = r['score']
        r['rank'] = rank
        count += 1
        ranking[r['method']] = rank

    return ranking[method]


def visualize_imputation(df: pd.DataFrame, metric: str, output_dirpath: Path):
    df_ranked = df.copy()
    
    df_ranked['rank_meanmode'] = df_ranked.apply(lambda x: calculate_rank(x, 'meanmode', metric), axis=1)
    df_ranked['rank_knn'] = df_ranked.apply(lambda x: calculate_rank(x, 'knn', metric), axis=1)
    df_ranked['rank_rf'] = df_ranked.apply(lambda x: calculate_rank(x, 'rf', metric), axis=1)

    df_ranked.to_csv(output_dirpath / f"{metric}_ranked.csv", index=False)

    df_ranked_melted = pd.melt(df_ranked, id_vars=['n_missing_values', 'missingness'],
                                value_vars=['rank_meanmode', 'rank_knn', 'rank_rf'],
                                var_name='method', value_name='rank')
    # modify values in method column
    # meanmode -> Mean/Mode, knn -> k-NN, rf -> Random Forest
    df_ranked_melted['method'] = df_ranked_melted['method'].str.replace('rank_meanmode', 'Mean/Mode')
    df_ranked_melted['method'] = df_ranked_melted['method'].str.replace('rank_knn', 'k-NN')
    df_ranked_melted['method'] = df_ranked_melted['method'].str.replace('rank_rf', 'Random Forest')

    df_ranked_melted.to_csv(output_dirpath / f"results_{metric}_ranked_melted.csv", index=False)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 6))
    plt.subplots_adjust(left=0.05, right=0.995, bottom=0.05, top=0.995)
    fig.delaxes(ax[3])

    for i, missingness in enumerate(['MCAR', 'MAR', 'MNAR']):
        df_ranked_melted_subset = df_ranked_melted[(df_ranked_melted['missingness'] == missingness)]
        sns.boxplot(x='n_missing_values', y='rank', hue='method', data=df_ranked_melted_subset, dodge=True, ax=ax[i])
        handler, label = ax[i].get_legend_handles_labels()
        # remove legends from subplots
        ax[i].get_legend().remove()

        ax[i].set_title(missingness)
        ax[i].set_xlabel('Number of missing values')
        ax[i].set_ylabel('Rank')
        ax[i].set_ylim(0.9, 3.1)
        ax[i].set_yticks([1, 2, 3])
        ax[i].grid(axis='y')
    title = f'Missing values imputation performance for {"categorical" if metric == "macro_f1" else "numerical"} variables'
    fig.suptitle(title)
    fig.legend(handler, label, loc='upper left', ncol=1, bbox_to_anchor=(0.76, 0.95))

    plt.tight_layout()
    plt.savefig(output_dirpath / f'boxplot_rank_{metric}.pdf')
    plt.close()
    
    return


def visualize_downstream(df: pd.DataFrame, output_dirpath: Path):
    return


def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()


def main():
    args = config_args()

    data_dirpath = Path(__file__).parents[2] / 'data'
    output_dirpath = data_dirpath / 'output/imputation'
    openml_dirpath = data_dirpath / 'openml'

    methods = ['meanmode', 'knn', 'rf']
    rmse_results = pd.DataFrame()
    macrof1_results = pd.DataFrame()
    downstream_results = pd.DataFrame()
    imputation_columns = ['openml_id', 'train_or_test', 'missing_column_name', 'missing_column_type', 'n_missing_values', 'missingness']
    downstream_columns = ['openml_id', 'missing_column_name', 'missing_column_type', 'missingness']

    for method in methods:
        method_output_dirpath = output_dirpath / method
        method_imputation_results_filepath = method_output_dirpath / f'imputation.csv'
        method_downstream_results_filepath = method_output_dirpath / f'downstream.csv'
    
        method_imputation_results = pd.read_csv(method_imputation_results_filepath, header=0)
        method_downstream_results = pd.read_csv(method_downstream_results_filepath, header=0)
        
        method_imputation_results = method_imputation_results.drop('timestamp', axis=1).drop('method', axis=1)
        
        rmse_results[imputation_columns] = method_imputation_results[imputation_columns].copy()
        rmse_results[method] = method_imputation_results['rmse'].copy()
        
        macrof1_results[imputation_columns] = method_imputation_results[imputation_columns].copy()
        macrof1_results[method] = method_imputation_results['macro_f1'].copy()
        
        downstream_results[downstream_columns] = method_downstream_results[downstream_columns].copy()
        downstream_results[method] = method_downstream_results['score'].copy()

    rmse_results = rmse_results.sort_values(by=['openml_id', 'missingness', 'missing_column_name']).dropna().reset_index(drop=True)
    macrof1_results = macrof1_results.sort_values(by=['openml_id', 'missingness', 'missing_column_name']).dropna().reset_index(drop=True)
    downstream_results = downstream_results.sort_values(by=['openml_id', 'missingness', 'missing_column_name']).dropna().reset_index(drop=True)
    
    visualize_imputation(rmse_results, 'rmse', output_dirpath)
    visualize_imputation(macrof1_results, 'macro_f1', output_dirpath)

    return


if __name__ == '__main__':
    main()
