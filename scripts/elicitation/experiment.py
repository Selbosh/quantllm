from modules.evaluator import PriorEvaluator
from modules.llmelicitor import LLMElicitor
import pandas as pd
from pathlib import Path
import datetime as dt
import argparse
import json
from tqdm import tqdm

def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--llm_model', type=str, default='gpt-4')
    argparser.add_argument('--llm_role', type=str, default='expert', choices=['expert', 'nonexpert', 'conference'])
    argparser.add_argument('--shelf', action='store_true')
    argparser.add_argument('--roulette', action='store_true')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()

def experiment(args: argparse.Namespace,
               timestamp: str,
               input_dirpath: Path,
               output_dirpath: Path):
    elicitation_results_filepath = output_dirpath / f"elicitation_{timestamp}.csv"
    logs = pd.read_csv(input_dirpath / 'logs.csv', header=0).loc[:, ['city', 'target']].drop_duplicates()
    for log in tqdm(logs.itertuples(), total=len(logs)):
        city, target = log.city, log.target
        try:
            weather_elicitation_experiment(
                args=args, timestamp=timestamp,
                city=city, target=target,
                results_filepath=elicitation_results_filepath
            )
        except Exception as e:
            print(f'Error in {city}, Target: {target}')
            print(e)
    return

# def weather_elicitation_experiment(prompts: dict, args: argparse.Namespace, timestamp: str, city: str, target: str, results_filepath: Path):
#     """
#     Perform prior elicitation for a meteorological information.
    
#     Args:
#         - args: argparse.Namespace object
#         - city: Name of target location
#         - target: Either 'temperature' or 'precipitation'
#         - timestamp
#         - results_filepath: path to save results, pathlib.Path object
#     """
    
#     weather_prompt = prompts['weather'][target]
#     log_filepath = results_filepath / 
#     elicitor = LLMElicitor(prompts, args.llm_model, role=args.llm_role, shelf=args.shelf, roulette=args.roulette,
                           
#                            debug=True)

def main():
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args = config_args()
    
    # IO
    data_dirpath = Path(__file__).parents[2] / 'data'
    output_dirpath = data_dirpath / f'output/elicitation/{args.llm_role}'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    prompts_filepath = Path(__file__).parents[0] / 'prompts.json'
    prompts = json.loads(prompts_filepath.read_text())
    log_filepath = output_dirpath / 'logs.csv'
    elicitor = LLMElicitor(prompts=prompts,
                           model=args.llm_model,
                           role=args.llm_role,
                           shelf=args.shelf,
                           roulette=args.roulette,
                           log_filepath=log_filepath,
                           debug=args.debug)
    cities_filepath = data_dirpath / 'cities/city_names.txt'
    with open(cities_filepath) as f:
        city_names = f.read().splitlines()
    results = []
    for target in ['temperature', 'precipitation']:
        for city in city_names:
            target_prompt = prompts['weather'][target].format(city = city)
            dist, params = elicitor.elicit(target_prompt,
                                            target_distribution='normal_inverse_gamma' if target == 'temperature' else 'gamma')
            results.append({'city': city, 'target': target, 'dist': dist, 'params': params,
                            'model': args.llm_model, 'role': args.llm_role, 'shelf': args.shelf, 'roulette': args.roulette})
    with open(output_dirpath / 'weather.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(results)
    
if __name__ == "__main__":
    main()
