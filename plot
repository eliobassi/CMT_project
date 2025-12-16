
# -*- coding: utf-8 -*-
# Plot: log(r) vs P (style "NDVI Sensitivity to NO2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# Fichier contenant des colonnes 'P' et 'r_hat' (r estimé historiquement)
path_hist = 'data/r_estimates_by_year.csv'  # <<< à remplacer
dfh = pd.read_csv(path_hist).dropna(subset=['P','r_hat'])

# Filtre r_hat > 0 pour log
dfh = dfh[dfh['r_hat'] > 0].copy()
dfh['log_r'] = np.log(dfh['r_hat'])

# Ajustement linéaire log(r) ~ α * P + β
x = dfh['P'].values
y = dfh['log_r'].values
A = np.vstack([x, np.ones_like(x)]).T
alpha, beta = np.linalg.lstsq(A, y, rcond=None)[0]
y_fit = alpha * x + beta

# Plot
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.scatter(x, y, s=28, color='royalblue', alpha=0.7, edgecolor='white', linewidth=0.6,
           label='log(r) observé')
# Ligne lissée triée pour l’esthétique
idx = np.argsort(x)
ax.plot(x[idx], y_fit[idx], color='crimson', lw=2.0,
        label=f'Droite ajustée (α={alpha:.3g}, β={beta:.3g})')

ax.set_title('NDVI Sensitivity to P\nlog(r) vs P')
ax.set_xlabel('P (pollution)')
ax.set_ylabel('log(r)')
ax.legend(loc='best', frameon=True)
ax.grid(True, ls='--', alpha=0.35)
plt.tight_layout()
plt.show()

# fig.savefig('results/logr_vs_P.png', dpi=300, bbox_inches='tight')
