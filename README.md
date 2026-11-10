# PG-DBN: Monotonic Remaining Useful Life Prediction of Turbofan Engines

Code, results and trained models for the paper

> **PG-DBN: A Physics-Guided Dual-Branch Network for Monotonic Remaining Useful Life Prediction of Turbofan Engines**
> S. Chakraborty, S. Ghosh, P. Karmakar, S. Dash, S. Sur. Submitted to IEEE MPCON.

This repository reproduces every table and figure in the paper. The model combines an LSTM with a branch built on sensor-trend features, adds a training penalty against rising RUL, and passes its output through a running minimum so the reported RUL never goes up. 

## Main numbers

| Subset | PG-DBN RMSE | Best baseline RMSE | PG-DBN raw MVR | MVR after COPC |
|---|---|---|---|---|
| FD001 | 12.54 ± 0.72 | 12.39 (LightGBM) | 37.1% | 0.00% |
| FD002 | **13.51 ± 0.06** | 13.78 (XGBoost) | 37.6% | 0.00% |
| FD003 | 12.86 ± 0.62 | 12.07 (MLP) | 29.4% | 0.00% |
| FD004 | **14.75 ± 0.10** | 15.06 (Transformer) | 29.5% | 0.00% |

See the paper for the full comparison with eight baselines, significance tests and the ablation.

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
