from modules.conjugate import NormalInverseGammaPrior, GammaExponentialPrior
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def evaluate_conjugate_prior(family: str, train_data, test_data,
                             sample_sizes: list[int] | range = None,
                             keep_params = False, **prior_params):
    """
    Calculate the posterior predictive loss for a given prior.
    
    Args:
        - `family`: Likelihood for the observations: 'norm' or 'exp'
        - `train_data`: Array of observations for fitting model
        - `test_data`: Data on which predictive loss will be evaluated
        - `sample_sizes`: Subsets of training data to fit on
        - `**prior_params`: Keyword args for `NormalInverseGammaPrior` or `GammaExponentialPrior`
        
    Returns:
        - A list of dictionaries containing expected log posterior predictive loss.
    """
    if family == 'norm':
        prior = NormalInverseGammaPrior(**prior_params)
    elif family == 'exp':
        prior = GammaExponentialPrior(**prior_params)
    else:
        raise NotImplementedError(f"Evaluation for family {family} not implemented")
    if sample_sizes is None:
        sample_sizes = range(1, len(train_data) + 1)
    if max(sample_sizes) > len(train_data):
        raise ValueError(f'sample size {max(sample_sizes)} larger than training data size {len(train_data)}')
    results = []
    for n in sample_sizes:
        if n < 1:
            print(f"Skipping sample size {n} < 1")
            continue
        subset = train_data[:n]
        prior.update_posterior(subset)
        loss = prior.expected_log_posterior_predictive_loss(test_data)
        results.append({
            'n': n,
            'family': family,
            'loss': loss,
            #'params': prior_params if keep_params else None,
        })
    return results

def main():
    np.random.seed(42)
    #data_dirpath = Path(__file__).parents[2] / 'data'
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
    main()