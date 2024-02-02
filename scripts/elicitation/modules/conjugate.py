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
        x_sum_sq = np.sum((data - x_bar) ** 2)
        mean_0, prec_0 = self.prior_mean, 1 / self.prior_variance
        shape_0, scale_0 = self.prior_shape, self.prior_scale
        # Posterior distribution
        prec_n = prec_0 + n
        mean_n = (prec_0 * mean_0 + n * x_bar) / prec_n
        shape_n = shape_0 + n / 2
        scale_n = scale_0 + 0.5 * x_sum_sq + (prec_0 * n * (x_bar - mean_0) ** 2) / (2 * prec_n)
        var_n = 1 / prec_n
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
    np.random.seed(42)
    data = np.random.normal(5, 1, size=100)
    test = np.random.normal(5, 1, size=50)
    lppd = {'good': [], 'bad': []}
    n_range = range(1, len(data) + 1)
    for n in n_range:
        subset_data = data[:n]
        bad = NormalInverseGammaPrior(0, .5, 5, .5)
        good = NormalInverseGammaPrior(7.5, 1, 2, 1)
        bad.fit(subset_data)
        good.fit(subset_data)
        lppd['bad'] += [bad.lppd(test)]
        lppd['good'] += [good.lppd(test)]
    
    plt.plot(n_range, lppd['bad'], marker='o', label='Bad prior', color='blue')
    plt.plot(n_range, lppd['good'], marker='o', label='Good prior', color='green')
    plt.xlabel('Number of data points')
    plt.ylabel('Expected log predictive density (ELPD)')
    plt.title('ELPD vs. number of data points')
    plt.suptitle('Gaussian random variable')
    plt.legend()
    plt.show()
    
    data = np.random.exponential(2, size=100)
    test = np.random.exponential(2, size=50)
    lppd = {'good': [], 'bad': []}
    n_range = range(1, len(data) + 1)
    for n in n_range:
        subset_data = data[:n]
        bad = GammaExponentialPrior(20, .01)
        good = GammaExponentialPrior(2, 4)
        bad.fit(subset_data)
        good.fit(subset_data)
        lppd['bad'] += [bad.lppd(test)]
        lppd['good'] += [good.lppd(test)]
    
    plt.plot(n_range, lppd['bad'], marker='o', label='Bad prior', color='blue')
    plt.plot(n_range, lppd['good'], marker='o', label='Good prior', color='green')
    plt.xlabel('Number of data points')
    plt.ylabel('Expected log predictive density (ELPD)')
    plt.title('ELPD vs. number of data points')
    plt.suptitle('Exponential random variable')
    plt.legend()
    plt.show()
    
    
