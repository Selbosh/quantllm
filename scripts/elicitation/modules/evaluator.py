from modules.conjugate import NormalInverseGammaPrior, GammaExponentialPrior


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