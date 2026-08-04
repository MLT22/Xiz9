import numpy as np


def pca_distribution_features(eig: np.ndarray) -> list:
    """Features invariantes à base — comparáveis entre imagens. Compartilhado
    entre FreqCorService e LuminanceService, que ambos derivam features de um PCA."""
    eig_sorted = np.sort(eig)[::-1]
    norm_eig = eig_sorted / (eig_sorted.sum() + 1e-12)
    eigen_entropy = float(-np.sum(norm_eig * np.log(norm_eig + 1e-12)))
    top5_ratio = float(eig_sorted[:5].sum() / (eig_sorted[-5:].sum() + 1e-12))
    cumsum = np.cumsum(norm_eig)
    n_50 = int(np.searchsorted(cumsum, 0.50) + 1)
    n_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    n_95 = int(np.searchsorted(cumsum, 0.95) + 1)
    k = np.arange(1, len(eig_sorted) + 1)
    slope = float(np.polyfit(np.log(k), np.log(eig_sorted + 1e-12), 1)[0])
    return [eigen_entropy, top5_ratio, n_50, n_90, n_95, slope]
