import numpy as np
from scipy.stats import multivariate_normal as mv_norm


def get_proba(data, **kwargs):
    means = kwargs["means"]
    covs = kwargs["covs"]
    priors = kwargs["priors"]
    n_classes = kwargs["n_classes"]

    prob = np.zeros((data.shape[0], n_classes))
    for i in range(n_classes):
        prob[:, i] = mv_norm(means[i], covs[i]).pdf(data)
        prob[:, i] *= priors[i]

    prob /= np.sum(prob, axis=1, keepdims=True)

    return prob


def classifier(prob, **kwargs):
    confidence = np.max(prob, axis=1)
    pred = np.argmax(prob, axis=1)
    return pred, confidence


def selector(confidence, threshold=None):
    index = np.argsort(confidence)[::-1]
    if threshold is None:
        return index
    accepted_indices = index[confidence[index] <= threshold]
    return accepted_indices


def hari_selector():
    pass
