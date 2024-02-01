import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class NormalInverseGammaPrior:
    """
    Normal distribution normal inverse gamma prior.
    """
    def __init__(self, mean, precision, shape, scale):
        alpha, beta = shape, scale
        self.prior_mean = self.post_mean = mean
        self.prior_precision = self.post_precision = precision
        self.prior_alpha = self.post_alpha = alpha
        self.prior_beta = self.post_beta = beta
        
    def update_posterior(self, data):
        n = len(data)
        x_bar = np.mean(data)
        
        post_precision = self.prior_precision + n
        post_mean = (self.prior_precision * self.prior_mean + n * x_bar) / post_precision
        post_alpha = self.prior_alpha + n / 2
        post_beta = self.prior_beta + 0.5 * np.sum((data - x_bar) ** 2) + (n * self.prior_precision) / (2 * post_precision)
        
        self.post_mean = post_mean
        self.post_precision = post_precision
        self.post_alpha = post_alpha
        self.post_beta = post_beta
    
    def log_posterior_predictive_density(self, new_data):
        df = 2 * self.post_alpha
        loc = self.post_mean
        scale = np.sqrt(self.post_beta * (1 + 1 / self.post_alpha) / self.post_alpha)
        return stats.t.logpdf(new_data, df=df, loc=loc, scale=scale)
    
    def expected_log_posterior_predictive_loss(self, new_data):
        log_pred_density = self.log_posterior_predictive_density(new_data)
        return np.mean(-log_pred_density)
    
class GammaExponentialPrior:
    """
    Exponential distribution with conjugate gamma prior.
    """
    def __init__(self, shape, scale):
        self.prior_shape = self.post_shape = shape
        self.prior_scale = self.post_scale = scale
    
    def update_posterior(self, data):
        n = len(data)
        post_shape = self.prior_shape + n
        post_scale = self.prior_scale + np.sum(data)
        self.shape_post = post_shape
        self.scale_post = post_scale
        
    def log_posterior_predictive_density(self, new_data):
        shape = self.shape_post
        scale = self.scale_post
        #return np.log(shape / scale * (1 + new_data / scale) ** -(shape + 1))
        return stats.lomax.logpdf(new_data, c=shape, scale=1/scale)
    
    def expected_log_posterior_predictive_loss(self, new_data):
        post_pred_scale = 1 / self.scale_post
        post_pred_samples = stats.expon.rvs(scale=post_pred_scale, size=1000)
        log_likelihoods = np.log(post_pred_samples)
        elppd = np.mean(log_likelihoods)
        return -elppd
        #log_pred_density = self.log_posterior_predictive_density(new_data)
        #return np.mean(-log_pred_density)

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
    nig = NormalInverseGammaPrior(2, 1, 1, 1)
    data = np.random.normal(2, 3, size=20)
    print(fit_empirical_distribution(data))
    nig.update_posterior(data)
    new_data = np.random.normal(3, 4, size=100)
    print("Expected predictive loss:", nig.expected_log_posterior_predictive_loss(new_data))
    
    gexp = GammaExponentialPrior(1, 1)
    data2 = np.random.exponential(2, size=20)
    print(fit_empirical_distribution(data2))
    gexp.update_posterior(data2)
    new_data2 = np.random.exponential(2, size=100)
    print("Expected predictive loss:", gexp.expected_log_posterior_predictive_loss(new_data2))
    
    
