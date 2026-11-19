# DCC-GARCH 追加実験 実行手順（R1-2 対応）

この手順は、別スレッドの Claude または著者が Windows 機（Independence / Diligence）で実験を実行するためのものです。上から順に進めてください。

## 0. 目的と背景

- 査読者 R1 の指摘：「DCC-GARCH を 30 または 45 観測だけで推定しており、推定が不安定でベースラインに不利」。
- 対応：論文 Table II と**同じテスト窓**で、推定に使う履歴だけを長くして DCC-GARCH を再評価する。
- 推定窓（`--lookbacks`）
  - `context`：コンテキスト窓のみ（論文の元設定。Table II の再現確認用）
  - `250`, `500`：コンテキスト窓の終端から遡って 250 / 500 観測
  - `expanding`：コンテキスト窓の終端までの全履歴（学習期間を含む）
- DCC の (a, b)（`--ab`）
  - `fixed`：a=0.05, b=0.9 固定（論文の元設定）
  - `mle`：準最尤推定
  - `both`：両方（既定）
- 評価指標：論文と同じ `matrix_cov_fro`（予測共分散とターゲット窓の標本共分散の Frobenius ノルム、学習期間の平均・標準偏差で標準化したリターン上）。参考に `matrix_corr_fro` も出力します。

データ読み込み・標準化・窓切り出し・val/test 分割は CDSM リポジトリ（https://github.com/KudanLabo/CDSM）の `ConditionalGluonTSJsonDatamodule` と同じ処理を numpy で再実装しています。DCC の予測式は `src/baselines/arch_baselines.py` の `dcc_garch_covariance_forecast` と同じで、(a, b) の最尤推定だけ数値安定な `slogdet` / `solve` を使っています。torch や hydra は不要です。

## 1. 動作確認済みの内容（2026-11-17、クラウド環境）

- iShares14（ℓH=30, ℓT=15, val_ratio=0.3、テスト窓 74）で `context` + `fixed` が **4.439** となり、論文 Table II の DCC-GARCH（iShares 4.44）を再現しました。
- 同じ窓で、リポジトリの `dcc_garch_covariance_forecast` と出力が数値誤差（2e-16）以内で一致しました。
- iShares14 の全設定の結果（参考値）：

| lookback | (a,b) | cov_fro 平均 | 標準偏差 |
| --- | --- | --- | --- |
| context (30) | fixed | 4.439 | 2.057 |
| context (30) | mle | 4.381 | 1.954 |
| 250 | fixed | 4.589 | 1.599 |
| 250 | mle | 4.610 | 1.493 |
| 500 | fixed | 4.939 | 1.397 |
| 500 | mle | 5.016 | 1.261 |
| expanding (~1008) | fixed | 4.448 | 1.538 |
| expanding (~1008) | mle | 4.474 | 1.426 |

iShares では推定窓を長くしても DCC-GARCH は改善せず、CDSM+CA（3.30）の優位は変わりませんでした。

## 2. 実行環境の準備（Windows）

このフォルダは Box 同期で Windows 機にも見えます。パスは通常次のとおりです（機種により異なる場合は Box フォルダの場所を確認）。

```
C:\Users\ando\Box\DraftMaterials\BigData2026\追加実験GARCH
```

Python 3.10 以上で、次をインストールします。既存の `baselines` / `baselines_arch` conda 環境があればそれを使って構いません。

```bat
conda create -n garch python=3.10 -y
conda activate garch
pip install -r requirements.txt
```

## 3. データの確認

スクリプトは既定で `%USERPROFILE%\.gluonts\datasets\` の下の次のフォルダを読みます（各フォルダに `train` と `test` の GluonTS 形式 JSON）。

| データ | フォルダ名 | (ℓH, ℓT) | val_ratio |
| --- | --- | --- | --- |
| exchange | exchange_rate_clean | (30, 15) | 0.3 |
| fx29 | fx30_ecb | (45, 15) | 0.3 |
| industry49 | industry49_clean | (30, 15) | 0.3 |
| ishares14 | ishares14_clean | (30, 15) | 0.3 |

```bat
dir %USERPROFILE%\.gluonts\datasets
```

フォルダが無い場合は、CDSM リポジトリのルートで `notebooks/paper.ipynb` と同じコマンドで作ります（ネットワークが必要なものがあります）。

```bat
python scripts\prepare_fx30_ecb_dataset.py --dst-dir %USERPROFILE%\.gluonts\datasets\fx30_ecb --nan-policy ffill --trim-start 2020-01-01 --trim-end 2025-12-01 --train-ratio 0.8 --returns log --prediction-length 15 --exclude-usd --overwrite
python scripts\prepare_industry49_dataset.py --dst-dir %USERPROFILE%\.gluonts\datasets\industry49_clean --nan-policy ffill --train-ratio 0.8 --trim-start 2018-01-01 --trim-end 2023-12-31 --log-returns --overwrite
python scripts\prepare_ishares_dataset.py --src-file ishares\df_ishares14x1102.csv --nan-policy ffill --dst-dir %USERPROFILE%\.gluonts\datasets\ishares14_clean --trim-start 2015-11-12 --trim-end 2020-01-31 --train-ratio 0.8 --returns log --overwrite
```

**exchange_rate_clean の作成コマンドはノートブックに残っていません。** 論文で使ったデータが既に `.gluonts\datasets\exchange_rate_clean` にある機械で実行してください。作り直す場合は `scripts\prepare_exchange_rate_dataset.py` を Table I の期間（1995-01-04〜1999-09-20）に合わせる必要があり、手順 4 の再現確認で一致することを確かめてから本実験に進んでください。

データが別の場所にある場合は `--data-root <親フォルダ>` または `--dataset <名前> --data-dir <フォルダ>` で指定します。

## 4. スモークテストと再現確認

```bat
cd /d C:\Users\ando\Box\DraftMaterials\BigData2026\追加実験GARCH
python dcc_long_window.py --dataset ishares14 --max-windows 5 --workers 4
python dcc_long_window.py --lookbacks context --ab fixed
```

2 行目の出力で、各データの `context` / `fixed` 行の `cov_fro_mean` が論文 Table II の DCC-GARCH の値と一致することを確認します（Exchange 6.38、iShares 4.44。FX29 と Industry は Table II の DCC-GARCH 行の値）。ずれる場合はデータか (ℓH, ℓT, val_ratio) が論文と異なるので、本実験に進まず著者に報告してください。各データのテスト窓数も出力されます。

## 5. 本実験

```bat
run_all.bat
```

または、2 台で分担する場合の例（Industry49 が最も重い）：

```bat
REM Independence
python dcc_long_window.py --dataset industry49 > run_industry49.log 2>&1
REM Diligence
python dcc_long_window.py --dataset exchange fx29 ishares14 > run_others.log 2>&1
```

所要時間の目安：iShares14 全設定で 8 並列 約 1 分 15 秒。Industry49（49 資産）の `expanding` + `mle` は 1 窓あたり数十秒かかる可能性があり、全体で 1 時間前後を見込んでください。`--workers` は既定で CPU コア数 − 1 です。

## 6. 出力

`results\<実行日時>\` に次が書き出されます（Box で同期されます）。

- `summary.csv`：データ × lookback × (a,b) ごとの窓数、平均観測数、`cov_fro_mean` / `cov_fro_std`、`corr_fro_mean`、推定された a, b の平均、GARCH 推定失敗によるフォールバック数、1 窓あたり秒数
- `<dataset>_per_window.csv`：窓ごとの値
- `<dataset>_config.json`：使った設定とデータパス

## 7. 実行後に報告すること

実行したスレッドは、次をまとめて著者に報告してください。

1. 手順 4 の再現確認の結果（各データで Table II の DCC-GARCH 値と一致したか）
2. `summary.csv` の内容を、データ × lookback × (a,b) の表にしたもの
3. 各データで `cov_fro_mean` が最小になった設定と、その値が Table II の順位を変えるか（CDSM+CA：Exchange 4.58、FX29 8.15、Industry 8.42、iShares 3.30。DeepVAR+CA、CAB などの値は論文 Table II を参照）
4. `garch_fallbacks` が 0 でない場合や `mle` の推定失敗があればその件数

論文側の修正方針（V-B の記述と Table II の差し替え、R1 への回答）は、査読対応スレッドの Claude Doc「BigData Workshop 査読対応：要約・回答案・修正文案」の項目 5 と Response to Reviewers に [TODO] として用意してあります。
