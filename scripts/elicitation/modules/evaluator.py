from conjugate import NormalInverseGammaPrior, GammaExponentialPrior
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PriorEvaluator:
    def __init__(self, family: str, train_data, test_data, **prior_params):
        """
        Args:
            - `family`: Likelihood for the observations: 'norm' or 'exp'
            - `train_data`: Array of observations for fitting model
            - `test_data`: Array of samples on which to evaluate model
            - `**prior_params`: Hyperparameters passed to `NormalInverseGammaPrior` or `GammaExponentialPrior`
        """
        if family == 'norm':
            self.prior = NormalInverseGammaPrior(**prior_params)
        elif family == 'exp':
            self.prior = GammaExponentialPrior(**prior_params)
        else:
            raise NotImplementedError(f'family {family} evaluator not implemented')
        self.prior_params = prior_params
        self.train_data = train_data
        self.test_data = test_data
        
    def compute_loss(self, n: int):
        max_n = len(self.train_data)
        if n > max_n:
            raise ValueError(f'Sample size is {n} larger than training set {max_n}')
        elif n < 1:
            raise ValueError(f'Sample size must be positive! Got n = {n}')
        subset = self.train_data[:n]
        self.prior.update_posterior(subset)
        loss = self.prior.expected_log_posterior_predictive_loss(self.test_data)
        return loss
    
    def evaluate(self, sample_sizes: list[int] | range = None):
        if sample_sizes is None:
            sample_sizes = range(1, len(self.train_data) + 1)
        results = [{'n': n, 'loss': self.compute_loss(n)} for n in sample_sizes]
        return results

def conjugacy_example():
    np.random.seed(42)
    train_normal = np.random.normal(170, 5, size=100)
    test_normal = np.random.normal(170, 5, size=200)
    prior_normal = {'mean': 170, 'precision': 1/10, 'alpha': 1, 'beta': 1}
    eval_normal = PriorEvaluator('norm', train_normal, test_normal, **prior_normal)
    eval_df = pd.DataFrame(eval_normal.evaluate())
    plt.plot(eval_df['n'], eval_df['loss'])
    plt.xscale('log')
    plt.title('Normal inverse gamma prior')
    plt.xlabel('Sample size')
    plt.ylabel('Expected loss')
    plt.show()
    
    train_exp = np.random.exponential(5, size=100)
    test_exp = np.random.exponential(5, size=100)
    prior_exp = {'shape': 1, 'scale': 1}
    eval_exp = PriorEvaluator('exp', train_exp, test_exp, **prior_exp)
    eval_df = pd.DataFrame(eval_exp.evaluate())
    plt.plot(eval_df['n'], eval_df['loss'])
    plt.xscale('log')
    plt.title('Gamma prior')
    plt.xlabel('Sample size')
    plt.ylabel('Expected loss')
    plt.show()  
    
if __name__ == "__main__":
    conjugacy_example()