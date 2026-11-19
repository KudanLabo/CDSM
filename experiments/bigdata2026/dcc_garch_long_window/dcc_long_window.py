#!/usr/bin/env python
"""DCC-GARCH baseline with longer estimation windows (reviewer #1, comment 2).

The paper's Table II fits DCC-GARCH only on the 30/45-day context window.
This script re-evaluates DCC-GARCH on exactly the same test windows, but
estimates the model on a longer history that ends at the same point
(the end of the context window):

    context   : the context window only (reproduces the original setting)
    <int>     : the last <int> observations (e.g. 250, 500)
    expanding : all observations available up to the end of the context window

Data loading, standardisation, windowing and the val/test split replicate
``cfdiff.dataloaders.conditional_gluonts.ConditionalGluonTSJsonDatamodule``
of https://github.com/KudanLabo/CDSM so that the test windows are identical
to those used for CDSM and the other baselines. The forecast recursion is the
same as ``baselines.arch_baselines.dcc_garch_covariance_forecast``.

The only dependencies are numpy, pandas, scipy and arch, so it runs on Windows
without the CDSM training environment (torch / lightning / hydra).

Example
    python dcc_long_window.py --dataset ishares14 --lookbacks context 250 500 expanding --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from arch.univariate import arch_model
except ImportError:  # pragma: no cover
    sys.exit("The `arch` package is required: pip install arch")


# ---------------------------------------------------------------------------
# Dataset settings used for Table II of the paper (same as notebooks/paper.ipynb)
# ---------------------------------------------------------------------------
DATASETS: Dict[str, dict] = {
    # name        data dir under ~/.gluonts/datasets, (l_H, l_T), val_ratio
    "exchange": {"dir": "exchange_rate_clean", "context_len": 30, "pred_len": 15, "val_ratio": 0.3},
    "fx29": {"dir": "fx30_ecb", "context_len": 45, "pred_len": 15, "val_ratio": 0.3},
    "industry49": {"dir": "industry49_clean", "context_len": 30, "pred_len": 15, "val_ratio": 0.3},
    "ishares14": {"dir": "ishares14_clean", "context_len": 30, "pred_len": 15, "val_ratio": 0.3},
}

EPS = 1e-12


# ---------------------------------------------------------------------------
# Data loading: mirrors cfdiff.dataloaders.conditional_gluonts
# ---------------------------------------------------------------------------
def _resolve_split_file(root: Path, split: str) -> Optional[Path]:
    for cand in (
        root / f"{split}.json",
        root / split / f"{split}.json",
        root / split / "data.json",
        root / split / "data.json.gz",
    ):
        if cand.exists():
            return cand
    return None


def _load_gluonts_like(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    series: List[List[float]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                series.append([float(v) for v in json.loads(line)["target"]])
    if not series:
        raise ValueError(f"Empty series in {path}")
    min_len = min(len(s) for s in series)
    return np.stack([s[-min_len:] for s in series], axis=1).astype(np.float32)


@dataclass
class TestWindow:
    index: int  # index within the test set (same as baselines.cmd.run window_index)
    end_ctx: int  # end (exclusive) of the context window in the full series
    target_start: int
    target_end: int


def load_test_windows(
    data_dir: Path, context_len: int, pred_len: int, val_ratio: float, stride: int = 1, split: str = "test"
) -> Tuple[np.ndarray, List[TestWindow], int]:
    """Return the standardised full series (train ++ test), the test windows and len(train)."""
    train_path = _resolve_split_file(data_dir, "train")
    test_path = _resolve_split_file(data_dir, "test")
    if train_path is None or test_path is None:
        raise FileNotFoundError(f"train/test split not found under {data_dir}")
    Xtr = _load_gluonts_like(train_path)
    Xt = _load_gluonts_like(test_path)
    if Xtr.shape[1] != Xt.shape[1]:
        raise ValueError("train/test asset counts mismatch")
    if not (np.isfinite(Xtr).all() and np.isfinite(Xt).all()):
        raise ValueError("Data contains non-finite values (the paper runs use nan_policy=raise).")

    # Standardise with training statistics (float32, as in the datamodule)
    mu = np.nanmean(Xtr, axis=0)
    sigma = np.nanstd(Xtr, axis=0, ddof=1)
    sigma[sigma < 1e-8] = 1.0
    Xtr = (Xtr - mu) / sigma
    Xt = (Xt - mu) / sigma

    L = context_len + pred_len
    # stride=1 -> one window per start position; align_end adds nothing
    starts = list(range(0, Xt.shape[0] - L + 1, stride))
    if stride > 1 and starts[-1] != Xt.shape[0] - L:
        starts.append(Xt.shape[0] - L)
    total_eval = len(starts)

    if val_ratio < 0:
        test_starts = starts  # test windows = all windows of the test split
    else:
        val_count = 1 if val_ratio <= 0 else int(round(total_eval * val_ratio))
        val_count = max(2 if total_eval >= 2 else 1, val_count)
        val_count = min(val_count, total_eval)
        gap = max(0, (L + stride - 1) // stride - 1)
        test_start = min(total_eval, val_count + gap)
        if test_start >= total_eval:
            raise ValueError("No test windows remain after the val/test gap.")
        test_starts = starts[test_start:] if split == "test" else starts[:val_count]

    n_tr = Xtr.shape[0]
    full = np.concatenate([Xtr, Xt], axis=0).astype(np.float64)
    windows = [
        TestWindow(
            index=i,
            end_ctx=n_tr + s + context_len,
            target_start=n_tr + s + context_len,
            target_end=n_tr + s + L,
        )
        for i, s in enumerate(test_starts)
    ]
    return full, windows, n_tr


# ---------------------------------------------------------------------------
# DCC-GARCH(1,1)
# ---------------------------------------------------------------------------
def _dcc_negloglik(params: np.ndarray, resid: np.ndarray, Q_bar: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    Qt = Q_bar.copy()
    ll = 0.0
    for t in range(1, resid.shape[0]):
        e = resid[t - 1]
        Qt = (1 - a - b) * Q_bar + a * np.outer(e, e) + b * Qt
        d = 1.0 / np.sqrt(np.diag(Qt) + EPS)
        Rt = Qt * np.outer(d, d)
        sign, logdet = np.linalg.slogdet(Rt)
        if sign <= 0:
            return 1e10
        try:
            sol = np.linalg.solve(Rt, resid[t])
        except np.linalg.LinAlgError:
            return 1e10
        # the e'e term of the DCC likelihood is constant in (a, b) and omitted
        ll += 0.5 * (logdet + float(resid[t] @ sol))
    return ll


def dcc_garch_forecast(
    hist: np.ndarray,
    horizon: int,
    estimate_ab: bool,
    a: float = 0.05,
    b: float = 0.9,
) -> Tuple[np.ndarray, dict]:
    """DCC(1,1) covariance forecast averaged over ``horizon`` steps.

    Univariate step and forecast recursion follow
    baselines.arch_baselines.dcc_garch_covariance_forecast. The (a, b) MLE uses
    slogdet/solve instead of det/inv, which underflows for many assets.
    """
    T, N = hist.shape
    vols, resids = [], []
    n_fallback = 0
    for j in range(N):
        series = hist[:, j]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(series, p=1, q=1, dist="normal", mean="Zero").fit(disp="off")
            var = res.forecast(horizon=horizon, reindex=False).variance.values[-1]
            vols.append(np.sqrt(np.clip(var, EPS, None)))
            resids.append(np.asarray(res.resid / res.conditional_volatility))
        except Exception:
            n_fallback += 1
            sd = float(np.std(series, ddof=1)) + EPS
            vols.append(np.repeat(sd, horizon))
            resids.append(series / sd)
    resid = np.stack(resids, axis=1)
    Q_bar = np.cov(resid, rowvar=False)

    opt_ok = None
    if estimate_ab:
        r = minimize(
            _dcc_negloglik,
            np.array([a, b]),
            args=(resid, Q_bar),
            method="SLSQP",
            bounds=[(1e-5, 0.999), (1e-5, 0.999)],
            constraints=[{"type": "ineq", "fun": lambda x: 0.998 - x[0] - x[1]}],
        )
        opt_ok = bool(r.success)
        if r.success:
            a, b = float(r.x[0]), float(r.x[1])

    Qt = Q_bar.copy()
    for t in range(1, T):
        e = resid[t - 1]
        Qt = (1 - a - b) * Q_bar + a * np.outer(e, e) + b * Qt
    shock = np.outer(resid[-1], resid[-1])

    avg_cov = np.zeros((N, N))
    Qf = Qt.copy()
    for step in range(horizon):
        if step == 0:
            Qf = (1 - a - b) * Q_bar + a * shock + b * Qf
        else:
            Qf = (1 - (a + b)) * Q_bar + (a + b) * Qf
        d = 1.0 / np.sqrt(np.diag(Qf) + EPS)
        Rt = Qf * np.outer(d, d)
        v = np.array([vv[step] if step < len(vv) else vv[-1] for vv in vols])
        avg_cov += (v[:, None] + EPS) * Rt * (v[None, :] + EPS)
    avg_cov /= horizon
    info = {"a": a, "b": b, "ab_opt_success": opt_ok, "garch_fallbacks": n_fallback}
    return avg_cov, info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _resolve_lookback(lb: str, context_len: int, w: TestWindow) -> Tuple[int, int]:
    """Return (start, end) of the estimation sample in the full series."""
    if lb == "context":
        n = context_len
    elif lb == "expanding":
        n = w.end_ctx
    else:
        n = int(lb)
    start = max(0, w.end_ctx - n)
    return start, w.end_ctx


def _eval_one(job: dict) -> dict:
    full = job["full"]
    w: TestWindow = job["window"]
    start, end = _resolve_lookback(job["lookback"], job["context_len"], w)
    hist = full[start:end]
    target = full[w.target_start : w.target_end]
    truth = np.cov(target, rowvar=False)
    t0 = time.perf_counter()
    est, info = dcc_garch_forecast(hist, job["pred_len"], estimate_ab=job["estimate_ab"])
    elapsed = time.perf_counter() - t0
    diff = est - truth
    return {
        "dataset": job["dataset"],
        "lookback": job["lookback"],
        "estimate_ab": job["estimate_ab"],
        "window_index": w.index,
        "n_obs": end - start,
        "matrix_cov_fro": float(np.linalg.norm(diff, ord="fro")),
        "matrix_corr_fro": float(np.linalg.norm(_corr(est) - _corr(truth), ord="fro")),
        "seconds": elapsed,
        **info,
    }


def _corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), EPS, None))
    c = cov / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0)


def run_dataset(
    name: str,
    data_root: Path,
    lookbacks: Sequence[str],
    ab_modes: Sequence[bool],
    workers: int,
    out_dir: Path,
    max_windows: Optional[int],
    overrides: dict,
) -> pd.DataFrame:
    cfg = {**DATASETS[name], **{k: v for k, v in overrides.items() if v is not None}}
    data_dir = Path(cfg.get("data_dir") or (data_root / cfg["dir"])).expanduser()
    full, windows, n_tr = load_test_windows(
        data_dir, cfg["context_len"], cfg["pred_len"], cfg["val_ratio"], split=overrides.get("split", "test")
    )
    if max_windows:
        windows = windows[:max_windows]
    print(
        f"[{name}] data={data_dir}  assets={full.shape[1]}  train_len={n_tr}  "
        f"test_len={full.shape[0] - n_tr}  (l_H,l_T)=({cfg['context_len']},{cfg['pred_len']})  "
        f"val_ratio={cfg['val_ratio']}  test_windows={len(windows)}",
        flush=True,
    )

    jobs = [
        {
            "full": full,
            "window": w,
            "lookback": lb,
            "estimate_ab": ab,
            "context_len": cfg["context_len"],
            "pred_len": cfg["pred_len"],
            "dataset": name,
        }
        for lb in lookbacks
        for ab in ab_modes
        for w in windows
    ]
    rows: List[dict] = []
    t0 = time.perf_counter()
    if workers <= 1:
        for k, job in enumerate(jobs, 1):
            rows.append(_eval_one(job))
            if k % 20 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, row in enumerate(ex.map(_eval_one, jobs, chunksize=4), 1):
                rows.append(row)
                if k % 20 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}_per_window.csv", index=False)
    with open(out_dir / f"{name}_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **cfg,
                "data_dir": str(data_dir),
                "lookbacks": list(lookbacks),
                "estimate_ab": list(ab_modes),
                "test_windows": len(windows),
                "first_window": asdict(windows[0]),
            },
            f,
            indent=2,
        )
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "lookback", "estimate_ab"])
    s = g.agg(
        windows=("matrix_cov_fro", "size"),
        n_obs_mean=("n_obs", "mean"),
        cov_fro_mean=("matrix_cov_fro", "mean"),
        cov_fro_std=("matrix_cov_fro", "std"),
        corr_fro_mean=("matrix_corr_fro", "mean"),
        a_mean=("a", "mean"),
        b_mean=("b", "mean"),
        garch_fallbacks=("garch_fallbacks", "sum"),
        sec_per_window=("seconds", "mean"),
    ).reset_index()
    return s


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument(
        "--data-root",
        default=os.environ.get("CDSM_DATA_ROOT", str(Path.home() / ".gluonts" / "datasets")),
        help="Directory that contains exchange_rate_clean, fx30_ecb, industry49_clean, ishares14_clean.",
    )
    ap.add_argument("--data-dir", default=None, help="Override the data directory (only with a single --dataset).")
    ap.add_argument("--lookbacks", nargs="+", default=["context", "250", "500", "expanding"])
    ap.add_argument(
        "--ab",
        choices=["fixed", "mle", "both"],
        default="both",
        help="DCC (a,b): fixed=(0.05,0.9) as in the paper, mle=estimated by QML, both=run both.",
    )
    ap.add_argument("--context-len", type=int, default=None, help="Override l_H")
    ap.add_argument("--pred-len", type=int, default=None, help="Override l_T")
    ap.add_argument("--val-ratio", type=float, default=None, help="Override val_ratio")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--max-windows", type=int, default=None, help="Evaluate only the first N test windows (smoke test).")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--split", choices=["test", "val"], default="test", help="Evaluate on test windows (default) or validation windows (for selecting the lookback / (a,b)).")
    args = ap.parse_args(argv)

    if args.data_dir and len(args.dataset) != 1:
        ap.error("--data-dir can only be used with a single --dataset")
    for lb in args.lookbacks:
        if lb not in {"context", "expanding"} and not lb.isdigit():
            ap.error(f"invalid lookback '{lb}'")
    ab_modes = {"fixed": [False], "mle": [True], "both": [False, True]}[args.ab]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / stamp
    overrides = {
        "context_len": args.context_len,
        "pred_len": args.pred_len,
        "val_ratio": args.val_ratio,
        "data_dir": args.data_dir,
        "split": args.split,
    }
    frames = []
    for name in args.dataset:
        try:
            frames.append(
                run_dataset(
                    name,
                    Path(args.data_root),
                    args.lookbacks,
                    ab_modes,
                    args.workers,
                    out_dir,
                    args.max_windows,
                    overrides,
                )
            )
        except FileNotFoundError as exc:
            print(f"[{name}] SKIPPED: {exc}", flush=True)
    if not frames:
        sys.exit("No dataset was evaluated.")
    summary = summarise(pd.concat(frames, ignore_index=True))
    summary.to_csv(out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
