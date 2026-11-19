# BigData Workshop 2026 revision experiments

Additional experiments run for the revision of the BigData 2026 workshop paper
("Conditional Denoising by Score-Matching for Return Covariance Forecasting").
Both folders are self-contained drivers that sit on top of the code in `src/`;
they do not modify the CDSM package itself.

| Folder | What it does | Detailed notes (Japanese) |
|---|---|---|
| `seed_table3/` | Retrains CDSM with all λ = 0 (Table III, λ0 row) for training seeds 42 / 12 / 22 / 32 on Exchange, FX29, Industry49 and iShares14, and reports mean ± std of test `cov_fro` across seeds. | `seed_table3/PLAN.ja.md` |
| `dcc_garch_long_window/` | DCC-GARCH baseline with longer estimation windows (context, 250, 500, expanding) and fixed vs. MLE (a, b), evaluated on the same test windows as Table II; `--split val` selects the setting on validation windows. | `dcc_garch_long_window/RUN.ja.md` |

## seed_table3

```bash
# once, on one machine: rebuild the λ=0 configs used in the paper (notebooks/paper.ipynb)
python seed_experiment.py make-base --repo <CDSM repo root>
# train + sample (resumes where it stopped); see run_*.bat for the per-machine split
python seed_experiment.py run --repo <CDSM repo root> --machine <name> --data-root ~/.gluonts/datasets \
    --epoch-mc 3 --epoch-steps 300 --sample-repeats 3
# aggregate results/values_*.csv into mean ± std
python seed_experiment.py summarize
```

- Checkpoint used for sampling: `best-matrix_corr_fro.ckpt`. Final sampling: 3 repeats × (1000 EM steps, 5 MC).
- With torch >= 2.6 set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (checkpoints contain omegaconf objects).
- `base_configs/` holds the generated train/sample configs (see `MADE_BY_INDEPENDENCE.txt`, `manifest.json`).
- `run_*.bat` / `*.ps1` are the exact launchers used on the lab's Windows machines (paths are machine specific).
- Per-run metrics are appended to `results/values_<machine>.csv`; logs go to `logs/` (not tracked).

## dcc_garch_long_window

```bash
pip install -r requirements.txt
python dcc_long_window.py --data-root ~/.gluonts/datasets \
    --dataset exchange fx29 industry49 ishares14 --lookbacks context 250 500 expanding --ab both
python dcc_long_window.py ... --split val   # validation windows, for choosing lookback / (a, b)
```

`results/` contains the summaries of the runs used for the revision:
`test_20261117/` (test windows), `val_20261117/` (validation windows) and
`cloud_ishares14_context/` (sanity check: iShares context/fixed = 4.439, matching Table II's 4.44).
