import pandas as pd
from scipy.stats import beta


def beta_prior(x):
    mu = x.mean()
    var = x.std() ** 2
    a = mu * (mu * (1 - mu) / var - 1)
    b = (1 - mu) * (mu * (1 - mu) / var - 1)
    prior = beta(a, b)
    return prior


def beta_posteriors(data, prior):
    n = data['reviews']
    s = data['rating'] / 100 * n
    a_prior, b_prior = prior.args
    a_post = s + a_prior
    b_post = n - s + b_prior
    posts = [beta(a, b) for a, b in zip(a_post, b_post)]
    return posts


def weighted_scores(data):
    prior = beta_prior(data['rating'] / 100.)
    posteriors = beta_posteriors(data, prior)
    post_means = 100 * pd.Series([post.mean() for post in posteriors], index=data.index)
    return post_means
