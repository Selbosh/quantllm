import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
from pyro.nn import PyroModule
import numpy as np
import math

def generate_normal_data(mean, std_dev, n_samples):
    normal = dist.Normal(mean, std_dev)
    samples = normal.sample([n_samples])
    return samples

def generate_heights_data(n_samples):
    heights = generate_normal_data(3.5, 0.5, n_samples)
    return heights

class NormalInverseGamma(PyroModule):
    def __init__(self, mu, gamma, alpha, beta):
        super().__init__()
        self.mu = mu
        self.gamma = gamma # 1 / lambda
        self.alpha = alpha
        self.beta = beta

    def forward(self):
        var = pyro.sample("var", dist.InverseGamma(self.alpha, self.beta))
        mean = pyro.sample("mean", dist.Normal(
            self.mu, torch.sqrt(self.gamma * var)))
        return mean, var

    def pdf(self, x, var):
        mu = self.mu
        gamma = self.gamma
        alpha = self.alpha
        beta = self.beta
        exp_term = torch.exp(-(2 * gamma * beta * (x - mu)
                             ** 2) / 2 * gamma * var)
        norm_term = torch.sqrt(2 * torch.tensor(np.pi) * gamma * var)
        return (beta ** 2 / math.gamma(alpha)) * (1/var)**(alpha + 1) * (exp_term / norm_term)

def posterior_hyperparameters(prior, data):
    mu = prior.mu
    gamma = prior.gamma
    alpha = prior.alpha
    beta = prior.beta
    nu = 1/gamma
    n = len(data)

    nu_post = (beta + n)
    mu_post = (nu * mu + data.sum()) / nu_post
    alpha_post = alpha + n/2
    beta_post = beta + data.var()/2 + (n * nu * (data.mean() - mu) ** 2)/(2*nu_post)
    return mu_post, nu_post, alpha_post, beta_post

class PosteriorPredictive(PyroModule):
    def __init__(self, prior, data):
        super().__init__()
        self.prior = prior
        self.data = data
    def forward(self):
        mu_post, nu_post, alpha_post, beta_post = posterior_hyperparameters(self.prior, self.data)
        return dist.StudentT(2 * alpha_post, mu_post, (beta_post * (nu_post + 1)) / (alpha_post * nu_post))

if __name__ == "__main__":
    heights = generate_heights_data(1000)
    #plt.hist(heights, bins=20)
    #plt.show()
    inf_prior = NormalInverseGamma(3.5, 0.01, 4., 3.)
    uninf_prior = NormalInverseGamma(0., 1., 4., 3.)
    inf_posterior = PosteriorPredictive(inf_prior, heights)
    print(inf_posterior().log_prob(torch.tensor(3.5)))
    n = 100
    train_data = generate_heights_data(n)
    test_data = generate_heights_data(1)
    xs = torch.linspace(0, n, n + 1)
    
    # Uninformative
    ys_u = torch.tensor([])
    for x in xs[2:]:    
        prior = uninf_prior
        data = train_data[:int(x)]
        post_predictive = PosteriorPredictive(prior, data)
        ys_u = torch.cat([ys_u, post_predictive().log_prob(data).mean(0, keepdim=True)])
        
    # Informative
    ys_i = torch.tensor([])
    for x in xs[2:]:
        prior = inf_prior
        data = train_data[0:int(x)]
        post_predictive = PosteriorPredictive(prior, data)
        ys_i = torch.cat([ys_i, post_predictive().log_prob(data).mean(0, keepdim=True)])
    
    # Plots
    plt.plot(xs[2:], ys_i.exp(), label='Informed prior')
    plt.plot(xs[2:], ys_u.exp(), label='Uninformed prior')
    plt.xlabel("Training data size")
    plt.ylabel("Probability sum of test data");
    plt.legend()
    plt.show()
    
    plt.plot(xs[2:], (ys_i), label='Informed prior')
    plt.plot(xs[2:], (ys_u), label='Uninformed prior')
    plt.xlabel("Training data size")
    plt.ylabel("Log probability of test data");
    plt.legend()
    plt.show()
    
