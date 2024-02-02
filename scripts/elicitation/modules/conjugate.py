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
        
    def fit(self, data):
        """Update the conjugate posterior."""
        n = len(data)
        x_bar = np.mean(data)
        mean, variance, shape, scale = self.prior_mean, self.prior_variance, self.prior_shape, self.prior_scale
        # Posterior distribution
        shape_n = shape + n / 2
        scale_n = 1 / (1 / scale + n / 2 * variance + (shape * n) / (2 * (shape + n)) * (x_bar - mean) ** 2)
        mean_n = (scale * mean + n * x_bar) / (scale + n)
        var_n = scale_n / shape_n
        self.posterior = {'mu': stats.norm(loc=mean_n, scale=np.sqrt(var_n)),
                          'sigma2': stats.invgamma(shape_n, scale=scale_n)}
        # Posterior predictive distribution
        df_pp, loc_pp, scale_pp = shape_n, mean_n, np.sqrt(scale_n * (1 + 1/ shape_n))
        self.posterior_pred = stats.t(df=df_pp, loc=loc_pp, scale=scale_pp)
        
    def log_pointwise_predictive_density(self, newdata, M=10000):
        """
        How well the model fits the data.
        
        http://www.stat.columbia.edu/~gelman/research/unpublished/waic_understand_final.pdf
        """
        mu_samples = self.posterior['mu'].rvs(M)
        sigma2_samples = self.posterior['sigma2'].rvs(M)
        # likelihood = lambda x: self.likelihood.pdf(x, loc=mu_samples, scale=np.sqrt(sigma2_samples))
        # lppd = [np.log(np.mean(likelihood(y))) for y in newdata]
        log_lik = lambda x: self.likelihood.logpdf(x, loc=mu_samples, scale=np.sqrt(sigma2_samples))
        lppd = [log_mean(log_lik(y)) for y in newdata]
        return np.sum(lppd)
    
    lppd = log_pointwise_predictive_density

class GammaExponentialPrior:
    def __init__(self, shape, scale):
        self.prior_shape = shape
        self.prior_scale = scale
        self.prior = stats.gamma(shape, scale=scale)
        self.likelihood = stats.expon
    
    def fit(self, data):
        """Update the conjugate posterior."""
        n = len(data)
        shape_n = self.prior_shape + n
        scale_n = 1 / (1 / self.prior_scale + np.sum(data))
        self.posterior = stats.gamma(shape_n, scale=scale_n)
        
    def log_pointwise_predictive_density(self, newdata, M=1000):
        """
        How well the model fits the data.
        
        http://www.stat.columbia.edu/~gelman/research/unpublished/waic_understand_final.pdf
        """
        theta_samples = self.posterior.rvs(M)
        log_lik = lambda x: self.likelihood.logpdf(x, scale=1/theta_samples)
        lppd = [log_mean(log_lik(y)) for y in newdata]
        return np.sum(lppd)
    
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
        

if __name__ == "__main__":
    mean, variance, shape, scale = 0, 1, 2, 2
    data = np.random.normal(mean, variance, size=200)
    test = np.random.normal(mean, variance, size=100)
    lppd_values = []
    for n in range(1, len(data) + 1):
        subset_data = data[:n]
        model = NormalInverseGammaPrior(mean, variance, shape, scale)
        model.fit(subset_data)
        lppd = model.lppd(test)
        lppd_values.append(lppd)
        
    plt.plot(range(1, len(data) + 1), lppd_values, marker='o')
    plt.xlabel('Number of data points')
    plt.ylabel('Expected log predictive density (ELPD)')
    plt.title('ELPD vs. number of data points')
    plt.suptitle('Gaussian random variable')
    plt.show()
    
    shape, scale = 2, 1
    data = np.random.exponential(2, size=200)
    test = [2.0] #np.random.exponential(2, size=100)
    lppd_values = []
    for n in range(1, len(data) + 1):
        subset_data = data[:n]
        model = GammaExponentialPrior(2, .001)
        model.fit(subset_data)
        lppd = model.lppd(test)
        lppd_values.append(lppd)
        
    plt.plot(range(1, len(data) + 1), lppd_values, marker='o')
    plt.xlabel('Number of data points')
    plt.ylabel('Expected log predictive density (ELPD)')
    plt.title('ELPD vs. number of data points')
    plt.suptitle('Exponential random variable')
    plt.show()
    
    
