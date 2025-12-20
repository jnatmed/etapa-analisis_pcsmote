
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AlphaDistanceARADASYN:
    """
    Técnica híbrida αDistance + AR-ADASYN.
    - Fase 1: Detección de muestras peligrosas con αDistance Borderline-SMOTE
              según pesos de distancia inversa.
    - Fase 2: Generación de sintéticos con rotación angular y radial (AR-ADASYN).
    """

    def __init__(self, m=5, beta=1.0, random_state=None):
        self.m = m  # número de vecinos
        self.beta = beta  # proporción deseada
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def fit_resample(self, X_min, X_maj):
        """
        Ejecuta el algoritmo híbrido sobre datos desbalanceados.

        Parámetros:
            X_min: ndarray con muestras minoritarias
            X_maj: ndarray con muestras mayoritarias

        Retorna:
            X_syn: nuevas muestras sintéticas generadas
        """

        # Unir conjunto completo para análisis de vecinos
        X_total = np.vstack([X_min, X_maj])
        y_total = np.array([1] * len(X_min) + [0] * len(X_maj))

        # ======================
        # FASE 1: αDistance Borderline
        # ======================
        nn_total = NearestNeighbors(n_neighbors=self.m + 1)
        nn_total.fit(X_total)

        muestras_peligrosas = []
        r_i_list = []

        for idx, x_i in enumerate(X_min):
            distancias, indices = nn_total.kneighbors([x_i])
            vecinos = indices[0][1:]  # ignorar self
            dists = distancias[0][1:]

            alpha_min = 0.0
            alpha_maj = 0.0
            delta_i = 0

            for j in vecinos:
                x_j = X_total[j]
                clase_j = y_total[j]
                alpha_j = 1.0 / (np.linalg.norm(x_i - x_j) + 1e-8)
                if clase_j == 1:
                    alpha_min += alpha_j
                else:
                    alpha_maj += alpha_j
                    delta_i += 1

            if alpha_maj > alpha_min:
                muestras_peligrosas.append(idx)
                r_i_list.append(delta_i / self.m)

        if not muestras_peligrosas:
            return np.empty((0, X_min.shape[1]))

        # Cálculo proporcional de cantidad sintética total
        G = int((len(X_maj) - len(X_min)) * self.beta)
        r_hat = np.array(r_i_list) / np.sum(r_i_list)
        g_i = (r_hat * G).astype(int)

        # ======================
        # FASE 2: AR-ADASYN
        # ======================
        nn_min = NearestNeighbors(n_neighbors=self.m + 1)
        nn_min.fit(X_min)
        X_syn = []

        for i, g in zip(muestras_peligrosas, g_i):
            x_i = X_min[i]
            _, indices_min = nn_min.kneighbors([x_i])
            vecinos_min = indices_min[0][1:]

            if len(vecinos_min) < 2:
                continue

            for _ in range(g):
                idx1, idx2 = self.rng.choice(vecinos_min, size=2, replace=False)
                x_nn1 = X_min[idx1]
                x_nn2 = X_min[idx2]

                # Vectores desde x_i hacia sus vecinos
                v1 = x_nn1 - x_i
                v2 = x_nn2 - x_i

                # Distancia y ángulo máximo
                r = max(np.linalg.norm(v1), np.linalg.norm(v2))
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                theta_prime = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                theta = min(theta_prime, np.pi - theta_prime)

                # Selección aleatoria dentro del sector definido
                alpha = self.rng.uniform(0, theta)
                beta = self.rng.uniform(0, r)

                dir_vec = v1 / (np.linalg.norm(v1) + 1e-8)
                rand_vec = self.rng.normal(size=x_i.shape)
                rand_vec -= rand_vec.dot(dir_vec) * dir_vec  # ortogonal
                rand_vec /= np.linalg.norm(rand_vec) + 1e-8

                # Vector rotado
                rot_vec = np.cos(alpha) * dir_vec + np.sin(alpha) * rand_vec
                x_syn = x_i + beta * rot_vec
                X_syn.append(x_syn)

        return np.vstack([X_min, X_maj, X_syn]), np.array([1]*len(X_min) + [0]*len(X_maj) + [1]*len(X_syn))

