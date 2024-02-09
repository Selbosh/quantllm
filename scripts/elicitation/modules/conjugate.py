import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
    
class NormalInverseGammaPrior:
    def __init__(self, mean, variance, shape, scale):
        self.prior_mean, self.prior_variance = mean, variance
        self.prior_shape, self.prior_scale = shape, scale
        self.prior = {'mu': stats.norm(mean, np.sqrt(variance)),
                      'sigma2': stats.invgamma(shape / 2, shape * variance / 2)}
        self.likelihood = stats.norm
        self.prior_pred = stats.t(2*shape, loc=mean, scale=scale*np.sqrt(variance)/(shape-1))
        
    def fit(self, x):
        """Update the conjugate posterior."""
        n = len(x)
        x_bar = np.mean(x)
        mean_0, prec_0 = self.prior_mean, 1 / self.prior_variance
        shape_0, scale_0 = self.prior_shape, self.prior_scale
        # Posterior distribution
        # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        prec_n = prec_0 + n
        var_n = 1 / prec_n
        mean_n = var_n * (prec_0 * mean_0 + n * x_bar)
        shape_n = shape_0 + n / 2
        x_sum_sq = np.sum((x - x_bar)**2)
        scale_n = scale_0 + 0.5 * x_sum_sq + (n * prec_0) / (prec_0 + n) * (x_bar - mean_0) ** 2 / 2
        #scale_n = scale_0 + 0.5 * (mean_0**2 * prec_0 + np.sum(x**2) - mean_n**2 * prec_n)
        self.posterior = {'mu': stats.norm(loc=mean_n, scale=np.sqrt(var_n)),
                          'sigma2': stats.invgamma(shape_n, scale=scale_n)}
        # Posterior predictive distribution
        df_pp, loc_pp, scale_pp = 2 * shape_n, mean_n, scale_n * (1 + var_n) / (shape_n)
        self.posterior_pred = stats.t(df=df_pp, loc=loc_pp, scale=np.sqrt(scale_pp))
        
    def log_pointwise_predictive_density(self, newdata):
        lppd = np.mean(self.posterior_pred.logpdf(newdata))
        return lppd
    
    lppd = log_pointwise_predictive_density

class GammaExponentialPrior:
    def __init__(self, shape, scale):
        self.prior_shape = shape
        self.prior_scale = scale
        self.prior = stats.gamma(shape, scale=scale)
        self.likelihood = stats.expon
        self.prior_pred = stats.lomax(c=1/scale, scale=shape)
    
    def fit(self, x):
        """Update the conjugate posterior."""
        n = len(x)
        # Posterior distribution
        shape_n = self.prior_shape + n
        beta_n = 1 / self.prior_scale + np.sum(x)
        scale_n = 1 / beta_n
        self.posterior = stats.gamma(shape_n, scale=scale_n)
        # Posterior predictive distribution
        # Lomax, aka Pareto II
        shape_lomax = beta_n
        scale_lomax = shape_n
        self.posterior_pred = stats.lomax(c=shape_lomax, scale=scale_lomax)
        
    def log_pointwise_predictive_density(self, newdata):
        lppd = np.mean(self.posterior_pred.logpdf(newdata))
        return lppd     
    
    lppd = log_pointwise_predictive_density

def log_mean(logxs): # log-sum-exp trick
    max_logx = np.max(logxs)
    return max_logx + np.log(np.mean(np.exp(logxs - max_logx)))

def fit_empirical_distribution(observed_data, plot=True):
    distributions = [stats.norm, stats.gamma, stats.expon, stats.t]
    # best_fit = {}
    best_kl_divergence = np.inf
    for distribution in distributions:
        # Fit the distribution to the data
        params = distribution.fit(observed_data)
        # Calculate the KL divergence
        nbins = 100
        x = np.linspace(min(observed_data), max(observed_data), nbins)
        observed_prob = np.histogram(observed_data, bins=nbins, density=True)[0]
        expected_prob = distribution.pdf(x, *params)
        kl_divergence = stats.entropy(observed_prob, expected_prob)
        # Update best fit if current distribution is better
        if kl_divergence < best_kl_divergence:
            best_kl_divergence = kl_divergence
            best_fit = {'distribution': distribution, 'params': params}
    if plot:
        plt.hist(observed_data, bins=30, density=True, alpha=0.6, color='g')
        x = np.linspace(min(observed_data), max(observed_data), 50)
        pdf_fitted = best_fit['distribution'].pdf(x, *best_fit['params'])
        plt.plot(x, pdf_fitted, 'r-', lw=2)
        plt.suptitle('Best Fit Distribution (KL Divergence)')
        plt.title(best_fit['distribution'].name)
        plt.show()
    return best_fit, best_kl_divergence

def main():
    true_mu, true_sigma = 170.0, 5.0
    observed_data = np.random.normal(true_mu, true_sigma, size=1000)
    priors = {
        'good': { 'params': (170, 5, 100, 1) },
        'bad (high)': { 'params': (200, 5, 1, 2) },
        'bad (low)': { 'params': (100, 5, 1, 2) },
        'vague': { 'params': (150, 20, .5, 10) },
        }
    for key, val in priors.items():
        priors[key]['model'] = NormalInverseGammaPrior(*val['params'])
        priors[key]['pred_mean'] = []
        priors[key]['pred_std'] = []
    sample_sizes = [2, 3, 4, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
    for n in sample_sizes:
        for key, val in priors.items():
            priors[key]['model'].fit(observed_data[:n])
            priors[key]['pred_mean'] += [priors[key]['model'].posterior_pred.mean()]
            #t_terms = priors[key]['model'].posterior_pred.kwds
            priors[key]['pred_std'] += [priors[key]['model'].posterior_pred.std()]
    priors['empirical'] = {'pred_mean': [np.mean(observed_data[:n]) for n in sample_sizes],
                           'pred_std': [np.std(observed_data[:n]) for n in sample_sizes]}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for key, val in priors.items():
        axs[0].plot(sample_sizes, priors[key]['pred_mean'], label=key, marker='o')
        axs[1].plot(sample_sizes, priors[key]['pred_std'], label=key, marker='o')
    axs[0].axhline(y = true_mu, color='lightblue', linestyle='--', label='True')
    axs[1].axhline(y = true_sigma, color='lightblue', linestyle='--', label='True')
    for ax in axs:
        ax.set_xlabel('sample size')
        ax.set_xscale('log')
        #ax.legend()
    axs[0].set_ylabel('mean')
    axs[0].set_title('Normal inverse gamma model')
    axs[1].set_ylabel('standard deviation')
    #axs[1].set_yscale('log')
    axs[0].legend()
    plt.tight_layout()
    plt.show()
    
    ####################################################################
    # Gamma prior for exponential likelihood
    # Prior:
    # k: shape
    # theta: scale
    # mean k*theta
    # var k*theta^2
    #--------------
    # Likelihood:
    # lambda: rate (1/scale)
    # mean 1/lambda
    # var 1/lambda^2
    #####################################################################
    true_rate = 1/5 # so mean is 5, var is 25
    observed_data = stats.expon(scale=1/true_rate).rvs(1000)
    priors = {
        'good': { 'params': (np.sqrt(5), np.sqrt(5)) },
        'vague': { 'params': (1, 20) },
        'bad': { 'params': (5, 1) },
        'other': { 'params': (1, 1) },
        'other2': { 'params': (0.5, 0.5) }
    }
    for key, val in priors.items():
        priors[key]['model'] = GammaExponentialPrior(*val['params'])
        priors[key]['pred_mean'] = []
        priors[key]['pred_std'] = []
    for n in sample_sizes:
        for key, val in priors.items():
            priors[key]['model'].fit(observed_data[:n])
            priors[key]['pred_mean'] += [priors[key]['model'].posterior_pred.mean()]
            #t_terms = priors[key]['model'].posterior_pred.kwds
            priors[key]['pred_std'] += [priors[key]['model'].posterior_pred.std()]
    priors['empirical'] = {'pred_mean': [1/np.mean(observed_data[:n]) for n in sample_sizes],
                           'pred_std': [1/np.std(observed_data[:n]) for n in sample_sizes]}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for key, val in priors.items():
        axs[0].plot(sample_sizes, priors[key]['pred_mean'], label=key, marker='o')
        axs[1].plot(sample_sizes, priors[key]['pred_std'], label=key, marker='o')
    axs[0].axhline(y = true_rate, color='lightblue', linestyle='--', label='True')
    axs[1].axhline(y = true_rate, color='lightblue', linestyle='--', label='True')
    for ax in axs:
        ax.set_xlabel('sample size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.legend()
    axs[0].set_ylabel('mean')
    axs[0].set_title('Gamma prior model')
    axs[1].set_ylabel('standard deviation')
    axs[0].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()