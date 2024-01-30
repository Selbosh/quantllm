from modules.evaluator import evaluate_conjugate_prior
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import argparse

def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--format', type=str, default='stan', choices=['stan', 'pyro', 'histogram', 'roulette', 'json'])
    argparser.add_argument('--evaluate', action='store_true')
    argparser.add_argument('--llm_model', type=str, default='gpt-4')
    argparser.add_argument('--llm_role', type=str, default='expert', choices=['expert', 'nonexpert'])
    argparser.add_argument('--shelf', action='store_true')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()

def main():
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    args = config_args()
    
    data_dirpath = Path(__file__).parents[2] / 'data'
    cities_dirpath = data_dirpath / 'cities'
    output_dirpath = data_dirpath / f'output/imputation/{args.method}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    
    experiment(args=args,
               timestamp=timestamp,
               cities_dirpath=cities_dirpath,
            #    input_dirpath=input_dirpath,
               output_dirpath=output_dirpath)

def conjugacy_example():
    np.random.seed(42)
    train_normal = np.random.normal(170, 5, size=100)
    test_normal = np.random.normal(170, 5, size=200)
    prior_normal = {'mean': 170, 'precision': 1/10, 'alpha': 1, 'beta': 1}
    eval_normal = evaluate_conjugate_prior('norm', train_normal, test_normal, **prior_normal)
    eval_df = pd.DataFrame(eval_normal)
    plt.plot(eval_df['n'], eval_df['loss'])
    plt.xscale('log')
    plt.title('Normal inverse gamma prior')
    plt.xlabel('Sample size')
    plt.ylabel('Expected loss')
    plt.show()
    
    train_exp = np.random.exponential(5, size=100)
    test_exp = np.random.exponential(5, size=100)
    prior_exp = {'shape': 1, 'scale': 1}
    eval_exp = evaluate_conjugate_prior('exp', train_exp, test_exp, **prior_exp)
    eval_df = pd.DataFrame(eval_exp)
    plt.plot(eval_df['n'], eval_df['loss'])
    plt.xscale('log')
    plt.title('Gamma prior')
    plt.xlabel('Sample size')
    plt.ylabel('Expected loss')
    plt.show()   
    
if __name__ == "__main__":
    conjugacy_example()
    #main()