#!/usr/bin/env python3
"""
Table III (CDSM, lambda=0) の学習 seed 間ばらつきを検証するための実験ドライバ。

サブコマンド
  audit     既存の seed42 / seed12,22,32 run を探し、実際の random_seed・checkpoint の一致・
            サンプル repeat 数・テスト指標を一覧化する（読み取りのみ）。
  export    seed42 run の train_config.yaml / sample_config.yaml を base_configs/ にコピーする
            （もう一方のマシンでも同じ設定で再学習できるようにするため）。
  run       base_configs から random_seed だけを変えて学習 + サンプリングを実行する。
            --machine independence|diligence でジョブを 2 台に振り分ける。
  evaluate  run の samples.pt から repeat ごとの ||ΔΣ||_F（test, trajectory 0）を計算して CSV に出す。
  summarize 各マシンの CSV を集約し、seed 間 mean ± std などを表と LaTeX にする。

依存: CDSM リポジトリの環境 (conda env cfdiff)。torch, numpy, pyyaml (omegaconf) を使う。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
BASE_CONFIG_DIR = HERE / "base_configs"
RESULTS_DIR = HERE / "results"
LOG_DIR = HERE / "logs"

# Table III の 4 データセット（make_ablation_context_table.py と同じパターン）
DATASETS = [
    {"key": "exchange", "label": "Exchange", "assets": 8, "patterns": ["FX8_30_15val0.3train0.8"]},
    {
        "key": "fx29",
        "label": "FX29",
        "assets": 29,
        "patterns": ["FX30_45_15val0.3train0.8", "FX30_ecb_45_15val0.3train0.8"],
    },
    {"key": "industry", "label": "Industry", "assets": 49, "patterns": ["ind49_30_15val0.3train0.8"]},
    {"key": "ishares", "label": "iShares14", "assets": 14, "patterns": ["ishares14_30_15_val0.3train0.8"]},
]
DOMAINS = [("time", "T"), ("fourier", "S")]
NEW_SEEDS = [42, 12, 22, 32]
ALL_KEYS = [d["key"] for d in DATASETS]

# Table III の出力が見つからないため、seed 42 も含めて 4 seed すべてを新しく学習する（2026-11-18 決定）。
# 各台の担当: (seed, データセット) の順に実行。seed 42 を先に回し、論文値との一致を早めに確認する。
# seed 42 は Independence と Diligence で 2 データセットずつ分担し、Audacity は後から seed 32 を担当する。
MACHINE_PLAN = {
    "independence": [(42, ["exchange", "fx29"]), (12, ALL_KEYS)],
    "diligence": [(42, ["industry", "ishares"]), (22, ALL_KEYS)],
    "audacity": [(32, ALL_KEYS)],
}

# データセットごとの設定（notebooks/paper.ipynb, reproduce_experiment.ipynb と同じ）
DATASET_SETTINGS = {
    "exchange": {"data_name": "exchange_rate_clean", "context_len": 30, "pred_len": 15, "val_ratio": 0.3},
    "fx29": {"data_name": "fx30_ecb", "context_len": 45, "pred_len": 15, "val_ratio": 0.3},
    "industry": {"data_name": "industry49_clean", "context_len": 30, "pred_len": 15, "val_ratio": 0.3},
    "ishares": {"data_name": "ishares14_clean", "context_len": 30, "pred_len": 15, "val_ratio": 0.3},
}
# サンプリングに使う checkpoint（paper.ipynb のサンプリングコマンドと同じ）
DEFAULT_CKPT_NAME = "checkpoints/best-matrix_corr_fro.ckpt"

# 論文 (1025_EBF_修正2.pdf) の Table III で seed42 再現チェックに使う値（CDSM+CA 行）
PAPER_CDSM_CA = {"exchange": 4.58, "fx29": 8.15, "industry": 8.42, "ishares": 3.30}


# ----------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def load_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


def save_yaml(obj: dict, path: Path) -> None:
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(obj), path)


def output_roots(repo: Path) -> List[Path]:
    return [repo / "outputs", repo / "outputs_local", repo / "outputs_seed"]


def find_runs(repo: Path, domain: str, patterns: Sequence[str]) -> Dict[int, List[Path]]:
    """seed -> run dirs (lambda0, conditional)。seed42 は suffix 無しの run。"""
    found: Dict[int, List[Path]] = {}
    for root in output_roots(repo):
        base = root / domain / "conditional"
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if not any(p in name for p in patterns):
                continue
            if not re.search(r"_lambda0(_|$)", name):
                continue
            m = re.search(r"(?:train_random_seed|_seed)(\d+)$", name)
            seed = int(m.group(1)) if m else 42
            found.setdefault(seed, []).append(d)
    for s in found:
        found[s].sort(key=lambda p: p.name)
    return found


def list_batches(run_dir: Path) -> List[Path]:
    hist = run_dir / "samples_history"
    if not hist.is_dir():
        return []
    batches = sorted([p for p in hist.iterdir() if p.is_dir() and p.name.startswith("batch-")])
    if batches:
        return batches
    # sample_repeats=1 のときは batch- フォルダが作られず samples_history/<時刻>-<tag>/samples.pt になる
    singles = sorted([p for p in hist.iterdir() if p.is_dir() and (p / "samples.pt").exists()])
    return [hist] if singles else []


def pick_batch(run_dir: Path) -> Optional[Path]:
    batches = list_batches(run_dir)
    if not batches:
        return None
    mc0 = [b for b in batches if b.name.endswith("-mc0")]
    return mc0[-1] if mc0 else batches[-1]


def repeat_dirs(batch: Path) -> List[Path]:
    dirs = [d for d in sorted(batch.iterdir()) if d.is_dir() and (d / "samples.pt").exists()]
    reps = [d for d in dirs if re.search(r"-r\d{2}(-|$)", d.name)]
    if reps:
        return reps
    return dirs[-1:]  # repeat 無し（sample_repeats=1）: 最新の 1 つ


def repeat_id(d: Path) -> str:
    m = re.search(r"-r(\d{2})(-|$)", d.name)
    return m.group(1) if m else "01"


def cfg_seed(run_dir: Path, fname: str) -> Optional[int]:
    p = run_dir / fname
    if not p.exists():
        return None
    try:
        return int(load_yaml(p).get("random_seed"))
    except Exception:
        return None


def ckpt_used(run_dir: Path) -> Optional[Path]:
    """sample_config.yaml の checkpoint_path を優先、無ければ model.ckpt。"""
    sc = run_dir / "sample_config.yaml"
    if sc.exists():
        cp = load_yaml(sc).get("checkpoint_path")
        if cp:
            name = Path(str(cp).replace("\\", "/")).name
            parent = Path(str(cp).replace("\\", "/")).parent.name
            for cand in [run_dir / parent / name, run_dir / name, run_dir / "checkpoints" / name]:
                if cand.exists():
                    return cand
    if (run_dir / "model.ckpt").exists():
        return run_dir / "model.ckpt"
    return None


# ----------------------------------------------------------------------------- metric
_AGG = None


def _try_repo_metric(repo: Path):
    """リポジトリの scripts/plot_correlation_mean.py があれば論文と同じ集計関数を使う。"""
    global _AGG
    if _AGG is not None:
        return _AGG if _AGG is not False else None
    scripts = repo / "scripts"
    if (scripts / "plot_correlation_mean.py").exists():
        sys.path.insert(0, str(scripts))
        sys.path.insert(0, str(repo / "src"))
        try:
            from plot_correlation_mean import aggregate_model  # type: ignore

            _AGG = aggregate_model
            log("metric: scripts/plot_correlation_mean.aggregate_model を使用（論文と同一）")
            return _AGG
        except Exception as exc:  # pragma: no cover
            log(f"metric: plot_correlation_mean の import に失敗 ({exc}); 内蔵実装を使用")
    else:
        log("metric: plot_correlation_mean.py が無いので内蔵実装を使用（seed42 の再現値で要確認）")
    _AGG = False
    return None


def _np(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def load_payload(samples_pt: Path, traj: int = 0) -> dict:
    import torch

    data = torch.load(samples_pt, map_location="cpu", weights_only=False)
    samples = _np(data["samples"])
    truth = _np(data["truth"])
    context = _np(data.get("context")) if data.get("context") is not None else None
    if samples.ndim == 4:
        if samples.shape[0] == truth.shape[0]:
            samples = samples[:, traj % samples.shape[1]]
        elif samples.shape[1] == truth.shape[0]:
            samples = samples[traj % samples.shape[0]]
    if truth.ndim == 4:
        truth = truth[:, 0]
    if context is not None and context.ndim == 4:
        context = context[:, 0]
    counts = data.get("window_stage_counts") or {}
    return {
        "pred": samples,
        "truth": truth,
        "context": context,
        "stage_counts": {k: int(v) for k, v in counts.items()},
    }


def stage_idx(counts: dict, n: int, stage: str) -> List[int]:
    if not counts:
        return list(range(n))
    v, t = int(counts.get("val", 0)), int(counts.get("test", 0))
    return {"val": list(range(v)), "test": list(range(v, v + t)), "all": list(range(v + t))}[stage]


def cov_fro(payload: dict, assets: int, stage: str, use_ca: bool, repo: Path) -> float:
    import numpy as np

    agg = _try_repo_metric(repo)
    if agg is not None:
        _l, _m, metrics = agg(
            ("time", "conditional"), payload, [], list(range(assets)), "matrix_cov_fro", stage, False,
            use_augmented_cov=use_ca,
        )
        return float(metrics["pred_minus_truth"])
    pred, truth, ctx = payload["pred"], payload["truth"], payload["context"]
    idx = stage_idx(payload["stage_counts"], pred.shape[0], stage)
    vals = []
    for i in idx:
        p = pred[i][:, :assets]
        t = truth[i][:, :assets]
        if use_ca and ctx is not None:
            p = np.concatenate([ctx[i][:, :assets], p], axis=0)
        d = np.cov(p, rowvar=False) - np.cov(t, rowvar=False)
        vals.append(float(np.linalg.norm(d, ord="fro")))
    return float(np.mean(vals))


# ----------------------------------------------------------------------------- audit
def cmd_audit(args) -> None:
    repo = Path(args.repo).resolve()
    rows = []
    for ds in DATASETS:
        for domain, tag in DOMAINS:
            runs = find_runs(repo, domain, ds["patterns"])
            base_hash = None
            for seed in sorted(runs, key=lambda s: (s != 42, s)):
                for run in runs[seed]:
                    ck = ckpt_used(run)
                    h = sha256(ck) if (ck and not args.no_hash) else ""
                    if seed == 42 and base_hash is None:
                        base_hash = h
                    batch = pick_batch(run)
                    reps = repeat_dirs(batch) if batch else []
                    row = {
                        "dataset": ds["label"],
                        "domain": tag,
                        "name_seed": seed,
                        "train_cfg_random_seed": cfg_seed(run, "train_config.yaml"),
                        "sample_cfg_random_seed": cfg_seed(run, "sample_config.yaml"),
                        "ckpt": ck.relative_to(run).as_posix() if ck else "",
                        "ckpt_sha16": h,
                        "same_ckpt_as_seed42": (h == base_hash) if (h and base_hash and seed != 42) else "",
                        "batch": batch.name if batch else "",
                        "n_repeats": len(reps),
                        "run_dir": str(run),
                    }
                    if args.metrics and reps:
                        for ca in (True, False):
                            vals = [cov_fro(load_payload(r / "samples.pt"), ds["assets"], "test", ca, repo) for r in reps]
                            row[f"test_covfro{'_CA' if ca else ''}"] = ";".join(f"{v:.6f}" for v in vals)
                    rows.append(row)
                    log(" | ".join(f"{k}={v}" for k, v in row.items() if k != "run_dir"))
    out = RESULTS_DIR / f"audit_{args.tag or platform.node()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if rows and k in rows[0] else 99)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {out}")
    if not rows:
        log("lambda0 の run が見つかりませんでした。--repo を確認してください。")


# ----------------------------------------------------------------------------- export
def cmd_export(args) -> None:
    repo = Path(args.repo).resolve()
    BASE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ds in DATASETS:
        for domain, tag in DOMAINS:
            runs = find_runs(repo, domain, ds["patterns"]).get(42, [])
            if not runs:
                log(f"[WARN] seed42 run が無い: {ds['label']} {domain}")
                continue
            run = runs[-1]
            key = f"{ds['key']}_{domain}"
            for fname in ("train_config.yaml", "sample_config.yaml"):
                src = run / fname
                if src.exists():
                    shutil.copy2(src, BASE_CONFIG_DIR / f"{key}_{fname}")
                else:
                    log(f"[WARN] {src} が無い")
            ck = ckpt_used(run)
            manifest[key] = {
                "run_dir": str(run),
                "ckpt_name": ck.relative_to(run).as_posix() if ck else "model.ckpt",
                "batch": pick_batch(run).name if pick_batch(run) else None,
            }
            log(f"exported {key} from {run.name}")
    (BASE_CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {BASE_CONFIG_DIR / 'manifest.json'}")



# ----------------------------------------------------------------------------- make-base
def _paper_overrides(domain: str, st: dict) -> List[str]:
    """paper.ipynb の学習コマンドを λ=0 にしたもの（best_metrics の @参照は compose 後に設定）。"""
    return [
        f"experiment={domain}",
        "experiment.datamodule.nan_policy=ffill",
        "experiment.datamodule.ffill_warm_start=false",
        "experiment.datamodule.batch_size=32",
        f"experiment.datamodule.val_ratio={st['val_ratio']}",
        "experiment.datamodule.num_workers=0",
        f"experiment.datamodule.context_len={st['context_len']}",
        f"experiment.datamodule.pred_len={st['pred_len']}",
        "experiment.datamodule.stride=1",
        "experiment.model.d_model=192",
        "experiment.model.num_layers=6",
        "experiment.model.n_head=8",
        "experiment.model.lr_scheduler_type=cosine_warmup",
        "+experiment.model.lr_scheduler_kwargs.interval=step",
        "experiment.model.lr=5e-4",
        "experiment.model.lambda_mean=0",
        "experiment.model.lambda_cov=0",
        "experiment.model.lambda_corr=0",
        "experiment.model.lambda_spectral=0",
        "experiment.model.lambda_sliding_cov=0",
        "experiment.sampler.sample_batch_size=64",
        "experiment.sampler.num_diffusion_steps=1000",
        "experiment.trainer.max_epochs=80",
        "callbacks.epoch_metrics.enabled=true",
        "callbacks.epoch_metrics.compute_fourier=true",
        "callbacks.epoch_metrics.num_mc_samples=5",
        "callbacks.epoch_metrics.max_batches=999",
        "callbacks.epoch_metrics.max_windows=999",
        "callbacks.epoch_metrics.sampler.num_diffusion_steps=1000",
        "callbacks.epoch_metrics.sampler.sample_batch_size=64",
        "experiment.model.likelihood_weighting=false",
        "experiment.model.t_sampling_mode=beta",
        "experiment.model.t_beta_alpha=2.0",
        "experiment.model.t_beta_beta=5.0",
        "experiment.model.t_importance_correction=true",
        "scheduler=vesde",
        "scheduler.sigma_min=5e-4",
        "scheduler.sigma_max=0.5",
        "random_seed=42",
    ]


def cmd_make_base(args) -> None:
    """Table III の seed42 run が見つからない場合に、paper.ipynb のコマンド（λ=0）から base_configs を作る。"""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    repo = Path(args.repo).resolve()
    cfg_dir = repo / "src" / "cfdiff" / "configs"
    best_yaml = cfg_dir / "callbacks" / "best_metrics_full.yaml"
    BASE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for ds in DATASETS:
        st = DATASET_SETTINGS[ds["key"]]
        for domain, _ in DOMAINS:
            key = f"{ds['key']}_{domain}"
            with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
                cfg = compose(config_name="config", overrides=_paper_overrides(domain, st))
            OmegaConf.set_struct(cfg, False)
            # TrainingRunner と同じ: experiment をルートにマージして取り除く
            cfg = OmegaConf.merge(cfg, cfg.experiment)
            cfg.pop("experiment", None)
            cfg.datamodule.data_dir = f"~/.gluonts/datasets/{st['data_name']}"
            cfg.callbacks.epoch_metrics.best_metrics = "@" + best_yaml.as_posix()
            cfg.run_mode = "train"
            cfg.output_dir = None
            cfg.checkpoint_path = None
            train = OmegaConf.to_container(cfg, resolve=False)
            save_yaml(train, BASE_CONFIG_DIR / f"{key}_train_config.yaml")
            # サンプリング（paper.ipynb: num_mc_samples=5, sample_repeats=5, save_mc_artifacts=true）
            samp = OmegaConf.to_container(cfg, resolve=False)
            samp["run_mode"] = "sample"
            samp["sampler"]["sample_repeats"] = 5
            samp["sampler"]["num_mc_samples"] = 5
            samp["sampler"]["save_mc_artifacts"] = True
            save_yaml(samp, BASE_CONFIG_DIR / f"{key}_sample_config.yaml")
            manifest[key] = {"run_dir": None, "ckpt_name": DEFAULT_CKPT_NAME, "batch": None,
                             "source": "paper.ipynb overrides with lambda=0 (make-base)"}
            log(f"made {key}")
    (BASE_CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {BASE_CONFIG_DIR / 'manifest.json'}")


# ----------------------------------------------------------------------------- run
def _local_data_dir(orig: str, data_root: Optional[str]) -> str:
    if not data_root:
        return orig
    name = Path(str(orig).replace("\\", "/")).name
    return str(Path(data_root).expanduser() / name)


def _fix_best_metrics_refs(cfg: dict, repo: Path) -> None:
    """best_metrics=@/Users/.../best_metrics_full.yaml のような別マシンの絶対パスを、このリポジトリのファイルに置き換える。"""
    em = (cfg.get("callbacks") or {}).get("epoch_metrics") or {}
    bm = em.get("best_metrics")
    items = bm if isinstance(bm, list) else [bm]
    fixed = []
    for it in items:
        if isinstance(it, str) and it.strip().startswith("@"):
            name = Path(it.strip()[1:].replace("\\", "/")).name
            hits = list((repo / "src").rglob(name))
            if hits:
                it = "@" + hits[0].resolve().as_posix()
            else:
                log(f"[WARN] best_metrics の参照 {name} がリポジトリ内に見つかりません")
        fixed.append(it)
    if bm is not None:
        em["best_metrics"] = fixed if isinstance(bm, list) else fixed[0]


def jobs_for(machine: Optional[str], datasets: Optional[List[str]], seeds: Optional[List[int]]) -> List[Tuple[dict, str, int]]:
    """--seeds / --datasets を指定すればそれを優先。指定が無ければ MACHINE_PLAN に従う。"""
    if seeds:
        plan = [(s, datasets or ALL_KEYS) for s in seeds]
    elif machine:
        plan = [(s, [k for k in keys if not datasets or k in datasets]) for s, keys in MACHINE_PLAN[machine]]
    else:
        plan = [(s, datasets or ALL_KEYS) for s in NEW_SEEDS]
    jobs = []
    for seed, keys in plan:
        for ds in DATASETS:
            if ds["key"] not in keys:
                continue
            for domain, _ in DOMAINS:
                jobs.append((ds, domain, seed))
    return jobs


def run_dir_for(repo: Path, ds: dict, domain: str, seed: int) -> Path:
    return repo / "outputs_seed" / domain / "conditional" / f"{ds['patterns'][-1]}_lambda0_seed{seed}"


def _invoke(repo: Path, module: str, cfg_path: Path, out_dir: Path, log_path: Path, dry: bool) -> None:
    cmd = [
        sys.executable, "-m", module,
        f"--config-path={cfg_path.parent.resolve().as_posix()}",
        f"--config-name={cfg_path.stem}",
        f"hydra.run.dir={(out_dir / 'hydra').as_posix()}",
    ]
    log("$ " + " ".join(cmd))
    if dry:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=str(repo), env=env, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{module} failed (exit {proc.returncode}); see {log_path}")


def cmd_run(args) -> None:
    repo = Path(args.repo).resolve()
    manifest_p = BASE_CONFIG_DIR / "manifest.json"
    if not manifest_p.exists():
        sys.exit("base_configs/manifest.json がありません。先に export を実行してください。")
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    jobs = jobs_for(args.machine, args.datasets, args.seeds)
    log(f"{len(jobs)} jobs on {args.machine or 'all'}: " + ", ".join(f"{d['key']}/{m}/s{s}" for d, m, s in jobs))
    for ds, domain, seed in jobs:
        key = f"{ds['key']}_{domain}"
        tcfg_p = BASE_CONFIG_DIR / f"{key}_train_config.yaml"
        scfg_p = BASE_CONFIG_DIR / f"{key}_sample_config.yaml"
        if not tcfg_p.exists():
            log(f"[SKIP] {tcfg_p.name} が無い")
            continue
        out = run_dir_for(repo, ds, domain, seed)
        done_flag = out / "SEED_EXPERIMENT_DONE"
        if done_flag.exists():
            log(f"[DONE] {out.name}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        gen = out / "_generated_configs"
        log_path = LOG_DIR / f"{args.machine or platform.node()}_{out.name}.log"
        ckpt_name = manifest.get(key, {}).get("ckpt_name", "model.ckpt")
        if args.disable_epoch_metrics:
            ckpt_name = "model.ckpt"  # 検証なし: 最終 epoch のモデルを使う
        ckpt = out / ckpt_name

        # ---- train
        if not ckpt.exists():
            t = load_yaml(tcfg_p)
            t["random_seed"] = seed
            t["output_dir"] = out.as_posix()
            t["run_mode"] = "train"
            t["checkpoint_path"] = None
            t["datamodule"]["data_dir"] = _local_data_dir(t["datamodule"]["data_dir"], args.data_root)
            if args.max_epochs:
                t["trainer"]["max_epochs"] = args.max_epochs
            em = (t.get("callbacks") or {}).get("epoch_metrics")
            if em is not None:
                if args.disable_epoch_metrics:
                    em["enabled"] = False
                if args.epoch_mc:
                    em["num_mc_samples"] = args.epoch_mc
                if args.epoch_steps:
                    em.setdefault("sampler", {})["num_diffusion_steps"] = args.epoch_steps
            _fix_best_metrics_refs(t, repo)
            save_yaml(t, gen / "train_seed.yaml")
            log(f"[TRAIN] {out.name}")
            _invoke(repo, "cfdiff.cmd.train", gen / "train_seed.yaml", out, log_path, args.dry_run)
            if not args.dry_run and not ckpt.exists():
                raise RuntimeError(f"学習後に {ckpt} がありません（base の checkpoint 名: {ckpt_name}）")
        else:
            log(f"[TRAIN skip] {ckpt} exists")

        # ---- sample
        if not list_batches(out) or not repeat_dirs(pick_batch(out)):
            s = load_yaml(scfg_p if scfg_p.exists() else tcfg_p)
            s["random_seed"] = seed
            s["output_dir"] = out.as_posix()
            s["run_mode"] = "sample"
            s["checkpoint_path"] = ckpt.as_posix()
            s["datamodule"]["data_dir"] = _local_data_dir(s["datamodule"]["data_dir"], args.data_root)
            if args.sample_repeats:
                s["sampler"]["sample_repeats"] = args.sample_repeats
            if args.final_mc:
                s["sampler"]["num_mc_samples"] = args.final_mc
            save_yaml(s, gen / "sample_seed.yaml")
            log(f"[SAMPLE] {out.name}")
            _invoke(repo, "cfdiff.cmd.sample", gen / "sample_seed.yaml", out, log_path, args.dry_run)
        if not args.dry_run:
            done_flag.write_text(dt.datetime.now().isoformat(), encoding="utf-8")
            # 1 job ごとに評価して結果を Box に残す（大きい samples.pt は Box に置かない）
            _evaluate_runs(repo, [(ds, domain, seed, out)], args.machine or platform.node())


# ----------------------------------------------------------------------------- evaluate
def _evaluate_runs(repo: Path, items, tag: str) -> Path:
    out_csv = RESULTS_DIR / f"values_{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if out_csv.exists():
        with out_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing.add((r["dataset"], r["domain"], r["seed"], r["repeat"], r["ca"]))
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["dataset", "domain", "seed", "repeat", "ca", "test_cov_fro", "run_dir", "batch"])
        for ds, domain, seed, run in items:
            batch = pick_batch(run)
            if batch is None:
                log(f"[WARN] samples_history が無い: {run}")
                continue
            for r in repeat_dirs(batch):
                payload = load_payload(r / "samples.pt")
                rep = repeat_id(r)
                for ca in (1, 0):
                    k = (ds["key"], domain, str(seed), rep, str(ca))
                    if k in existing:
                        continue
                    v = cov_fro(payload, ds["assets"], "test", bool(ca), repo)
                    w.writerow([ds["key"], domain, seed, rep, ca, f"{v:.8f}", str(run), batch.name])
                    f.flush()
                    log(f"{ds['key']}/{domain}/seed{seed}/r{rep} CA={ca}: {v:.5f}")
    return out_csv


def cmd_evaluate(args) -> None:
    repo = Path(args.repo).resolve()
    items = []
    keys = args.datasets or [d["key"] for d in DATASETS]
    for ds in DATASETS:
        if ds["key"] not in keys:
            continue
        for domain, _ in DOMAINS:
            runs = find_runs(repo, domain, ds["patterns"])
            for seed, dirs in runs.items():
                if args.seeds and seed not in args.seeds:
                    continue
                # outputs_seed（今回の再実行）を優先、無ければ最新
                pref = [d for d in dirs if "outputs_seed" in d.parts]
                items.append((ds, domain, seed, (pref or dirs)[-1]))
    p = _evaluate_runs(repo, items, args.tag or platform.node())
    log(f"wrote {p}")


# ----------------------------------------------------------------------------- summarize
def _fmt(x: float, sig: int = 3) -> str:
    if x == 0 or not math.isfinite(x):
        return f"{x}"
    d = sig - 1 - int(math.floor(math.log10(abs(x))))
    return f"{x:.{max(d, 0)}f}"


def cmd_summarize(args) -> None:
    import numpy as np

    vals: Dict[Tuple[str, str, int], Dict[int, List[float]]] = {}
    files = sorted(RESULTS_DIR.glob("values_*.csv"))
    if not files:
        sys.exit("results/values_*.csv がありません")
    seen = set()
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                k = (r["dataset"], r["domain"], int(r["seed"]), r["repeat"], int(r["ca"]))
                if k in seen:
                    continue
                seen.add(k)
                vals.setdefault((r["dataset"], r["domain"], int(r["ca"])), {}).setdefault(int(r["seed"]), []).append(
                    float(r["test_cov_fro"])
                )
    rows = []
    for ds in DATASETS:
        for domain, tag in DOMAINS:
            for ca in (1, 0):
                per_seed = vals.get((ds["key"], domain, ca), {})
                if not per_seed:
                    continue
                seeds = sorted(per_seed)
                means = np.array([np.mean(per_seed[s]) for s in seeds])
                within = np.array([np.std(per_seed[s]) for s in seeds])
                pooled = np.concatenate([per_seed[s] for s in seeds])
                rows.append({
                    "dataset": ds["label"], "model": f"CDSM-{tag}" + ("+CA" if ca else ""),
                    "seeds": "/".join(map(str, seeds)),
                    "n_values": len(pooled),
                    "seed42_mean": float(np.mean(per_seed[42])) if 42 in per_seed else float("nan"),
                    "paper_value": PAPER_CDSM_CA.get(ds["key"], float("nan")) if ca else float("nan"),
                    "pooled_mean": float(pooled.mean()),
                    "pooled_std": float(pooled.std()),
                    "across_seed_std(ddof=1)": float(means.std(ddof=1)) if len(means) > 1 else float("nan"),
                    "within_seed_std_mean": float(within.mean()),
                    "per_seed_means": ";".join(f"{s}:{m:.5f}" for s, m in zip(seeds, means)),
                })
    out = RESULTS_DIR / "summary_seed.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(" | ".join(f"{k}={(_fmt(v) if isinstance(v, float) else v)}" for k, v in r.items()))
    # LaTeX: Table III の CDSM 行を seed 間 mean ± std で置き換えるための行
    tex = RESULTS_DIR / "table3_cdsm_rows_seed.tex"
    with tex.open("w", encoding="utf-8") as f:
        f.write("% mean over seed means; ± = std of the per-seed means (ddof=1), test windows, trajectory 0\n")
        for model in ["CDSM-T+CA", "CDSM-T", "CDSM-S+CA", "CDSM-S"]:
            cells = [model]
            for ds in DATASETS:
                r = next((x for x in rows if x["dataset"] == ds["label"] and x["model"] == model), None)
                if r is None:
                    cells.append("--")
                    continue
                means = [float(t.split(":")[1]) for t in r["per_seed_means"].split(";")]
                cells.append(f"{_fmt(float(np.mean(means)))} $\\pm$ {_fmt(r['across_seed_std(ddof=1)'], 2)}")
            f.write(" & ".join(cells) + " \\\\\n")
    log(f"wrote {out} and {tex}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_repo(p):
        p.add_argument("--repo", required=True, help="CDSM リポジトリのルート（outputs/ がある場所）")

    p = sub.add_parser("audit"); add_repo(p)
    p.add_argument("--metrics", action="store_true", help="repeat ごとの test ||ΔΣ||_F も計算する（遅い）")
    p.add_argument("--no-hash", action="store_true", help="checkpoint の SHA256 を計算しない")
    p.add_argument("--tag", default=None)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("export"); add_repo(p); p.set_defaults(func=cmd_export)

    p = sub.add_parser("make-base", help="paper.ipynb の設定（λ=0）から base_configs を作る"); add_repo(p)
    p.set_defaults(func=cmd_make_base)

    p = sub.add_parser("run"); add_repo(p)
    p.add_argument("--machine", choices=list(MACHINE_PLAN), default=None)
    p.add_argument("--datasets", nargs="*", choices=[d["key"] for d in DATASETS], default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--data-root", default=None, help="データセットの親フォルダ（例: %%USERPROFILE%%\\.gluonts\\datasets）")
    p.add_argument("--max-epochs", type=int, default=None, help="動作確認用。本番では指定しない")
    p.add_argument("--sample-repeats", type=int, default=None, help="動作確認用。本番では指定しない")
    p.add_argument("--disable-epoch-metrics", action="store_true", help="学習中の検証サンプリングを止め、最終 epoch の model.ckpt でサンプリングする")
    p.add_argument("--epoch-mc", type=int, default=None, help="学習中の検証サンプリングの MC サンプル数（論文: 5）")
    p.add_argument("--epoch-steps", type=int, default=None, help="学習中の検証サンプリングの拡散ステップ数（論文: 1000）")
    p.add_argument("--final-mc", type=int, default=None, help="最終サンプリングの MC サンプル数（論文: 5。Table III は trajectory 0 のみ使用）")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("evaluate"); add_repo(p)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("summarize"); p.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
