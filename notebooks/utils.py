import numpy as np
import matplotlib.pyplot as plt


def gerar_gaussiana(mu=0, sigma=1, n=1000):
    """Gera dados sintéticos de uma distribuição gaussiana."""
    return np.random.normal(mu, sigma, n)


def plotar_histograma(dados, titulo="Distribuição"):
    """Plota histograma dos dados."""
    plt.figure(figsize=(10, 6))
    plt.hist(dados, bins=30, density=True, alpha=0.7, edgecolor="black")
    plt.title(titulo)
    plt.xlabel("Valor")
    plt.ylabel("Densidade")
    plt.grid(alpha=0.3)
    plt.show()
