"""Paired per-engine Wilcoxon tests on RAW trajectory RMSE (Table II, middle block of the paper).

Usage:  python scripts/raw_trajectory_tests.py            (run from the repository root)
Reads results/predictions/*.npz written by PGDBN_Kaggle.ipynb.
"""
import numpy as np, pandas as pd
from scipy import stats

PRED = 'results/predictions'
FDS = ['FD001', 'FD002', 'FD003', 'FD004']
BASELINES = ['MLP', '1D-CNN', 'TCN', 'LSTM', 'BiLSTM-Attn', 'Transformer', 'XGBoost', 'LightGBM']
SEEDS = [0, 1, 2]

def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i])); adj[i] = run
    return adj

P = np.load(f'{PRED}/predictions.npz')
rows = []
for fd in FDS:
    G = np.load(f'{PRED}/ground_truth_{fd}.npz')
    y, U, engines = G['y_traj'], G['units_traj'], G['units_last']
    def per_engine_rmse(model):  # per-engine trajectory RMSE, averaged over seeds
        return np.mean([[np.sqrt(np.mean((P[f'{fd}|{model}|{s}|traj'][U == u] - y[U == u]) ** 2)) for u in engines]
                        for s in SEEDS], axis=0)
    pg = per_engine_rmse('PG-DBN'); pvals, effects = [], []
    for b in BASELINES:
        d = pg - per_engine_rmse(b); d = d[d != 0]
        pvals.append(stats.wilcoxon(d).pvalue)
        rk = stats.rankdata(np.abs(d)); effects.append((rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum())
    for b, p, ph, r in zip(BASELINES, pvals, holm(pvals), effects):
        verdict = ('PG-DBN better' if r < 0 else 'PG-DBN worse') if ph < 0.05 else 'no significant difference'
        rows.append(dict(subset=fd, baseline=b, p=round(p, 5), p_holm=round(ph, 5), rank_biserial=round(r, 3), result=verdict))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv('results/tables/stats_trajectory_raw.csv', index=False)
print('\nSaved results/tables/stats_trajectory_raw.csv')
