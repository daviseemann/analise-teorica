import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def generate_data(
    n_samples, means, covs, priors=None, seed=42, groups=None, **kwargs
) -> np.ndarray:
    """Gera um DataFrame com n-classes de dados gaussianos."""

    n_classes = len(means)

    if priors is None:
        priors = [1 / n_classes] * n_classes

    assert round(sum(priors), 2) == 1, "Os priors devem somar 1."
    assert (
        len(priors) == n_classes
    ), "O número de priors deve ser igual ao número de classes."

    class_samples = []
    for i in range(n_classes):
        class_pdf = multivariate_normal(means[i], covs[i])
        samples = class_pdf.rvs(int(n_samples * priors[i]), random_state=seed + 5 + i)
        class_samples.append(samples)

    data = np.vstack(class_samples)
    labels = np.concatenate(
        [np.array([i] * len(samples)) for i, samples in enumerate(class_samples)]
    )

    return data, labels
