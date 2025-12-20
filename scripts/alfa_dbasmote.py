import numpy as np
from sklearn.neighbors import NearestNeighbors

def AlphaDistanceDBASMOTE(X_min, X_maj, beta=1.0, m=5, random_state=None):
    """
    αDistance Borderline-ADASYN-SMOTE según Feng & Li (2021).
    
    Parámetros:
    - X_min: ndarray de forma (n_min, n_features), muestras de la clase minoritaria
    - X_maj: ndarray de forma (n_maj, n_features), muestras de la clase mayoritaria
    - beta: proporción deseada de balanceo (entre 0 y 1)
    - m: número de vecinos
    - random_state: semilla aleatoria

    Retorna:
    - muestras_sinteticas: ndarray de forma (n_sinteticas, n_features)
    """
    rng = np.random.default_rng(seed=random_state)
    X_total = np.vstack([X_min, X_maj])
    y_min = np.ones(len(X_min))  # Clase 1
    y_maj = np.zeros(len(X_maj)) # Clase 0
    y_total = np.concatenate([y_min, y_maj])
    
    # Vecinos sobre todo el dataset
    nn = NearestNeighbors(n_neighbors=m+1)  # +1 para excluirse a sí mismo
    nn.fit(X_total)

    muestras_peligrosas = []
    distancias_peligrosas = []

    for idx, p_i in enumerate(X_min):
        distancias, indices = nn.kneighbors([p_i], return_distance=True)
        vecinos = indices[0][1:]  # excluir a sí mismo
        dists = distancias[0][1:]

        # Clasificar vecinos
        alphas_min = []
        alphas_maj = []
        delta_i = 0  # cantidad de vecinos mayoritarios

        for i, j in enumerate(vecinos):
            p_j = X_total[j]
            clase_j = y_total[j]
            alpha_j = 1.0 / (np.linalg.norm(p_i - p_j) + 1e-8)

            if clase_j == 1:
                alphas_min.append(alpha_j)
            else:
                alphas_maj.append(alpha_j)
                delta_i += 1

        alpha_p = sum(alphas_min)
        alpha_n = sum(alphas_maj)

        if alpha_n > alpha_p:
            muestras_peligrosas.append(idx)
            distancias_peligrosas.append(delta_i / m)

    # Total de sintéticos
    N = len(X_maj)
    n = len(X_min)
    G = int((N - n) * beta)

    if G <= 0 or not muestras_peligrosas:
        return np.empty((0, X_min.shape[1]))

    # Normalizar pesos r_i
    r_hat = np.array(distancias_peligrosas) / sum(distancias_peligrosas)
    g_i = (r_hat * G).astype(int)

    # Generación sintética
    muestras_sinteticas = []

    # Solo entre vecinos minoritarios
    nn_min = NearestNeighbors(n_neighbors=m)
    nn_min.fit(X_min)

    for i, gi in zip(muestras_peligrosas, g_i):
        p_i = X_min[i]
        _, vecinos_min = nn_min.kneighbors([p_i], return_distance=True)
        vecinos_min = vecinos_min[0]

        for _ in range(gi):
            z_idx = rng.choice(vecinos_min)
            p_z = X_min[z_idx]
            lmbda = rng.random()
            s = p_i + lmbda * (p_z - p_i)
            muestras_sinteticas.append(s)

    return np.array(muestras_sinteticas)
