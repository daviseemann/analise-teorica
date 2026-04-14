# Análise Teórica — Classificação Seletiva em Dados Sintéticos

Estudo experimental de classificação seletiva com foco em métricas equitativas (erro balanceado e worst-group error), aplicado a dados gaussianos 2D com desbalanceamento de classes.

Este repositório faz parte de uma pesquisa de mestrado sobre o tradeoff erro-rejeição em cenários de cauda longa, inspirada principalmente por Narasimhan et al. (ICLR, 2024).

## O que o projeto faz

O notebook `notebooks/classificador-2d.ipynb` implementa um pipeline completo de classificação seletiva em dados sintéticos:

1. **Geração de dados** — Duas classes gaussianas 2D com priors desbalanceados (20%/80%), usando `scipy.stats.multivariate_normal`.
2. **Classificador Bayesiano** — Cálculo da posteriori via regra de Bayes, com predição por argmax e score de confiança por max da posteriori.
3. **Curvas Risk-Coverage (RC)** — Ordenação por confiança e cálculo de três métricas de risco em função da cobertura:
   - **Erro padrão** — taxa de erro global cumulativa.
   - **Erro balanceado (BER)** — média das taxas de erro por classe.
   - **Worst-group error (WG)** — máximo das taxas de erro entre classes.

O módulo `notebooks/utils.py` contém funções auxiliares para geração e visualização de distribuições gaussianas.

## Como usar

```bash
git clone https://github.com/daviseemann/analise-teorica.git
cd analise-teorica
pip install -r requirements.txt
jupyter notebook notebooks/classificador-2d.ipynb
```

### Dependências

- Python 3.8+
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4
- Jupyter ≥ 1.0

## Referências

- Narasimhan, H. et al. _Learning to Reject Meets Long-Tail Learning._ ICLR, 2024.
- Chow, C. K. _On Optimum Recognition Error and Reject Tradeoff._ IEEE Trans. Inf. Theory, 1970.
- Galil, I. et al. _What Can We Learn from the Selective Prediction and Uncertainty Estimation Performance of 523 ImageNet Classifiers?_ ICLR, 2023.
- Franc, V. et al. _Optimal Strategies for Reject Option Classifiers._
