
import numpy as np
from sklearn.neighbors import NearestNeighbors

def ARADASYN(X_min, X_maj, k=5, random_state=None):
    """
    AR-ADASYN: Angle Radius-Adaptive Synthetic Sampling (según Park & Kim, 2024)

    Parámetros:
    - X_min: ndarray, muestras minoritarias
    - X_maj: ndarray, muestras mayoritarias
    - k: número de vecinos
    - random_state: semilla

    Retorna:
    - X_syn: ndarray, muestras sintéticas generadas
    """
    rng = np.random.default_rng(random_state)
    X_total = np.vstack([X_min, X_maj])
    y_total = np.array([1]*len(X_min) + [0]*len(X_maj))

    n_min = len(X_min)
    n_maj = len(X_maj)
    nsyn = n_maj - n_min

    # Paso 1: calcular pesos adaptativos
    neigh_all = NearestNeighbors(n_neighbors=k+1)
    neigh_all.fit(X_total)
    weights = []

    for x_i in X_min:
        _, indices = neigh_all.kneighbors([x_i])
        vecinos = indices[0][1:]  # excluirse a sí mismo
        count_maj = sum(y_total[j] == 0 for j in vecinos)
        w_i = count_maj / k
        weights.append(w_i)

    weights = np.array(weights)
    w_hat = weights / np.sum(weights)
    g_vect = np.round(w_hat * nsyn).astype(int)

    # Paso 2 y 3: generar muestras sintéticas por cada x_i
    neigh_min = NearestNeighbors(n_neighbors=k+1)
    neigh_min.fit(X_min)

    X_syn = []

    for i, x_i in enumerate(X_min):
        g_i = g_vect[i]
        if g_i == 0:
            continue

        _, indices_min = neigh_min.kneighbors([x_i])
        vecinos_min = indices_min[0][1:]

        if len(vecinos_min) < 2:
            continue

        for _ in range(g_i):
            idx1, idx2 = rng.choice(vecinos_min, size=2, replace=False)
            x_nn1 = X_min[idx1]
            x_nn2 = X_min[idx2]

            v1 = x_nn1 - x_i
            v2 = x_nn2 - x_i

            # radio
            r = max(np.linalg.norm(v1), np.linalg.norm(v2))

            # ángulo
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            theta_prime = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            theta = min(theta_prime, np.pi - theta_prime)

            # generar ángulo y radio aleatorio
            alpha = rng.uniform(0, theta)
            beta = rng.uniform(0, r)

            # dirección rotada
            dir_vec = v1 / (np.linalg.norm(v1) + 1e-8)
            rand_vec = rng.normal(size=x_i.shape)
            rand_vec -= rand_vec.dot(dir_vec) * dir_vec
            rand_vec /= np.linalg.norm(rand_vec) + 1e-8

            rot_vec = np.cos(alpha) * dir_vec + np.sin(alpha) * rand_vec
            x_syn = x_i + beta * rot_vec
            X_syn.append(x_syn)

    return np.array(X_syn)
