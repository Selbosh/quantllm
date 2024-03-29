from modules.evaluator import PriorEvaluator
from modules.llmelicitor import LLMElicitor
import pandas as pd
from pathlib import Path
import datetime as dt
import argparse
import json
import os



def config_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--experiment', type=str, default='weather', choices=['weather', 'psychology', 'behavioural', 'crowdfunding', 'taskmaster'])
    argparser.add_argument('--llm_model', type=str, default='gpt-4')
    argparser.add_argument('--llm_role', type=str, default='expert', choices=['expert', 'nonexpert', 'conference'])
    argparser.add_argument('--shelf', action='store_true')
    argparser.add_argument('--roulette', action='store_true')
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--seed', type=int, default=42)
    return argparser.parse_args()

def elicit_weather(args: argparse.Namespace,
                   timestamp: str,
                   elicited_filepath: Path):
    """
    Run the weather elicitation experiment.
    
    Args:
        - args: Command-line arguments
        - timestamp: As string
        - elicited_filepath: Path to save elicited values CSV file.
    """
    # Set up.
    data_dirpath = Path(__file__).parents[2] / 'data'
    #output_dirpath = data_dirpath / f'output/elicitation/{args.llm_role}'
    #output_dirpath.mkdir(parents=True, exist_ok=True)
    prompts_filepath = Path(__file__).parent / 'prompts.json'
    prompts = json.loads(prompts_filepath.read_text())
    log_filepath = elicited_filepath.parent / f'log_{timestamp}.json'
    elicitor = LLMElicitor(prompts=prompts,
                           model=args.llm_model,
                           role=args.llm_role,
                           expert_prompt="You are an expert in meteorology.",
                           shelf=args.shelf,
                           roulette=args.roulette,
                           log_filepath=log_filepath,
                           debug=args.debug)
    
    # Tasks.
    cities_filepath = data_dirpath / 'cities/city_names.txt'
    with open(cities_filepath) as f:
        city_names = f.read().splitlines()
    results = []
    for target in ['temperature', 'precipitation']:
        for city in city_names:
            target_prompt = prompts['weather'][target].format(city = city)
            dist, params = elicitor.elicit(target_prompt,
                                           target_distribution='normal_inverse_gamma' if target == 'temperature' else 'gamma')
            results.append({'field': city, 'target': target, 'dist': dist, 'params': params,
                            'model': args.llm_model, 'role': args.llm_role, 'shelf': args.shelf, 'roulette': args.roulette,
                            'timestamp': timestamp})
    # Saving results.
    json_str = json.dumps(results)
    with open(elicited_filepath, 'a') as file:
        file.write(json_str + '\n')
    #results_df.to_csv(elicited_filepath, index=False, mode='a', header=not os.path.exists(elicited_filepath))
    return results

def elicit_psychology(args: argparse.Namespace,
                      timestamp: str,
                      elicited_filepath: Path):
    prompts_filepath = Path(__file__).parent / 'prompts.json'
    prompts = json.loads(prompts_filepath.read_text())
    log_filepath = elicited_filepath.parent / f'log_{timestamp}.json'
    elicitor = LLMElicitor(prompts=prompts,
                           model=args.llm_model,
                           role=args.llm_role,
                           #expert_prompt="",
                           shelf=args.shelf,
                           roulette=args.roulette,
                           log_filepath=log_filepath,
                           debug=args.debug)
    results = []
    for subfield in ['social psychology', 'developmental psychology', 'cognitive neuroscience']:
        expert_prompt = f'You are an expert in {subfield}.' # only used if args.llm_role == 'expert'
        target_qty = f"Imagine what small-to-medium effect sizes in {subfield} look like. "
        "Which effect size would you expect as the most probable one to be found? "
        "Which range of values would you consider possible? "
        "Specifically, we are interested in: "
        for target in ['cohen', 'pearson']:
            target_prompt = target_qty + prompts['psychology'][target]
            target_distribution = "student_t" if target == 'cohen' else 'beta'
            dist, params = elicitor.elicit(target_prompt, target_distribution=target_distribution, expert_prompt=expert_prompt)
            results.append({'field': subfield, 'target': target, 'dist': dist, 'params': params,
                            'model': args.llm_model, 'role': args.llm_role, 'shelf': args.shelf, 'roulette': args.roulette,
                            'timestamp': timestamp})
    # Saving results.
    json_str = json.dumps(results)
    with open(elicited_filepath, 'a') as file:
        file.write(json_str + '\n')
    # results_df.to_csv(elicited_filepath, index=False, mode='a', header=not os.path.exists(elicited_filepath))
    return results

def elicit_any_task(args: argparse.Namespace,
                    timestamp: str,
                    tasks_filepath: Path,
                    elicited_filepath: Path):
    prompts_filepath = Path(__file__).parent / 'prompts.json'
    prompts = json.loads(prompts_filepath.read_text())
    tasks = pd.read_csv(tasks_filepath)
    results = []
    for i, tsk in tasks.iterrows():
        if args.debug:
            print(i)
            print(tsk)
        task_description = tsk.Task
        target_dist = tsk.Distribution
        tsi = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # avoid overwriting logs
        log_filepath = elicited_filepath.parent / f'log_{tsi}.json' 
        elicitor = LLMElicitor(prompts=prompts, model=args.llm_model, role='expert',
                               expert_prompt='', shelf=args.shelf, roulette=args.roulette,
                               log_filepath=log_filepath)
        expert_prompt = elicitor.expert_prompt_initialization(task_description)
        if args.debug:
            print(f'Expert prompt:\n{expert_prompt}')
        _, params = elicitor.elicit(task_description, target_dist, expert_prompt)
        results.append({'task_index': i, #'task': task_description,
                        'dist': target_dist, 'params': params,
                        'model': args.llm_model, #'expert_prompt': expert_prompt,
                        # timestamp in `results` refers to the overall job:
                        'shelf': args.shelf, 'roulette': args.roulette, 'timestamp': timestamp})
    # Saving results
    json_str = json.dumps(results)
    with open(elicited_filepath, 'a') as file:
        file.write(json_str + '\n')
    return results

def elicit_behavioural_sci(args: argparse.Namespace,
                           timestamp: str,
                           elicited_filepath: Path):
    pass
    
def main():
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    args = config_args()
    
    data_dirpath = Path(__file__).parents[2] / 'data'
    output_dirpath = data_dirpath / f'output/elicitation'
    output_dirpath.mkdir(parents=True, exist_ok=True)
    
    if args.experiment == 'weather':
        weather_output_filepath = output_dirpath / "weather.ndjson"
        elicit_weather(args, timestamp, weather_output_filepath)
    
    if args.experiment == 'psychology':
        psych_output_filepath = output_dirpath / "psychology.ndjson"
        elicit_psychology(args, timestamp, psych_output_filepath)
        
    if args.experiment == 'behavioural':
        raise NotImplementedError()
    
    if args.experiment == 'crowdfunding':
        raise NotImplementedError()
    
    if args.experiment == 'taskmaster':
        any_output_filepath = output_dirpath / 'anytask.ndjson'
        tasks_filepath = data_dirpath / 'tasks.csv'
        if not Path.exists(tasks_filepath):
            raise FileNotFoundError(tasks_filepath)
        elicit_any_task(args, timestamp, tasks_filepath, any_output_filepath)
    
    
if __name__ == "__main__":
    main()
