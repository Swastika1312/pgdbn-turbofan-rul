# PG-DBN: Monotonic Remaining Useful Life Prediction of Turbofan Engines

Code, results and trained models for the paper

> **PG-DBN: A Physics-Guided Dual-Branch Network for Monotonic Remaining Useful Life Prediction of Turbofan Engines**
> S. Chakraborty, S. Ghosh, P. Karmakar, S. Dash, S. Sur. Submitted to IEEE MPCON.

PG-DBN predicts the remaining useful life (RUL) of turbofan engines on NASA C-MAPSS. It fuses a two-layer LSTM with a trend branch over per-sensor degradation slopes (dS/dt), rolling means and dispersion, is trained with a pairwise monotonicity penalty, and is deployed behind a Causal Operational Physics Constraint (COPC), a zero-look-ahead running minimum that guarantees a non-increasing RUL. The study compares nine models on all four subsets with trajectory-level metrics and paired per-engine significance tests.

## Key results (mean over 3 seeds)

| Subset | PG-DBN RMSE | Best baseline RMSE | PG-DBN raw MVR | MVR after COPC |
|---|---|---|---|---|
| FD001 | 12.54 ± 0.72 | 12.39 (LightGBM) | 37.1% | 0.00% |
| FD002 | **13.51 ± 0.06** | 13.78 (XGBoost) | 37.6% | 0.00% |
| FD003 | 12.86 ± 0.62 | 12.07 (MLP) | 29.4% | 0.00% |
| FD004 | **14.75 ± 0.10** | 15.06 (Transformer) | 29.5% | 0.00% |

* The eight baselines (MLP, 1D-CNN, TCN, LSTM, BiLSTM-Attn, Transformer, XGBoost, LightGBM) predict an RUL increase in 19.6%–44.2% of consecutive cycles.
* PG-DBN shares the best Friedman rank with XGBoost and is never significantly less accurate than any baseline at the last window (Wilcoxon, Holm-corrected).
* COPC removes every violation for any model but raises trajectory RMSE by 2.7–8.4 cycles. After COPC is applied to all nine models, PG-DBN is the only model in the top three on all four subsets.

Full numbers: `results/tables/` and `results/SUMMARY.txt`.

## Reproduce

The whole study is one notebook, `PGDBN_Kaggle.ipynb` (about 36 minutes on a Kaggle NVIDIA T4 GPU).

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and upload `PGDBN_Kaggle.ipynb`.
2. **Add Input** → search `nasa-cmaps` (by behrad3d) → add. The C-MAPSS data is not included in this repository.
3. **Settings → Accelerator → GPU T4 x2.**
4. **Save Version → Save & Run All.** Outputs are written to `pgdbn_outputs.zip`.

Set `QUICK = True` in the configuration cell for a ~5-minute smoke test. The notebook also runs locally (`pip install -r requirements.txt`) if the C-MAPSS text files are placed under `./data`.

The paired tests on raw trajectory RMSE (Table II, middle block) are computed from the saved predictions:

```bash
python scripts/raw_trajectory_tests.py
```

## Repository contents

| Path | Contents |
|---|---|
| `PGDBN_Kaggle.ipynb` | Data pipeline, 8 baselines, PG-DBN, COPC, ablation, statistical tests, figures |
| `scripts/raw_trajectory_tests.py` | Wilcoxon tests on raw trajectory RMSE from saved predictions |
| `results/tables/` | All result tables (CSV and LaTeX) and test outcomes |
| `results/per_seed/` | Every metric for every model, subset and seed |
| `results/predictions/` | Raw last-window and full-trajectory predictions of every model and seed, plus ground truth |
| `results/models/` | Trained PG-DBN weights (seed 0, one per subset) |
| `results/figures/` | Figures produced by the run |
| `results/run_config.json` | Exact configuration and library versions of the reported run |

## Citation

If you use this code, please cite the paper (citation details will be added after publication).

## License

MIT License; see `LICENSE`. The NASA C-MAPSS dataset is subject to its own terms.
