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
    
if __name__ == "__main__":
    main()