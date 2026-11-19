# Table III 標準偏差の検証実験（追加実験Seed）

査読 R2-2（「複数 run の mean ± std を示すこと」）に答えるための実験です。Table III の CDSM の行が、本当に「学習 seed 4 個 × サンプリング 5 回」のばらつきを表しているかを確かめます。そうなっていなければ、学習 seed を正しく変えて再学習し、seed 間の mean ± std を出し直します。

## 0. 背景と現状

- Table III は `scripts/make_ablation_context_table.py` で作っています。CDSM の行は、seed 42, 12, 22, 32 のそれぞれについて、test 窓で平均した ‖ΔΣ‖_F をサンプリング 5 回ぶん集め、最大 20 個の値に `np.std` をかけたものです。UCDSM の行は 1 run × 5 回だけです。
- 論文の std は 1e-4〜1e-2 で、学習 seed の違うモデル同士のばらつきにしては小さすぎます。
- seed 12/22/32 の run 自体は 2026-02-01〜03 に作られています（`scripts/plot_paper_figures.sh` にフォルダ名が残っています）。
- ただしリポジトリのコードには `train_random_seed` という設定キーがありません。学習（`pl.seed_everything(cfg.random_seed)`）とサンプリング（seed = `random_seed + repeat番号`）が使うのは `random_seed`（既定 42）だけです。seed run を `+train_random_seed=12` のような未使用キーやフォルダ名だけで区別していた場合、4 つの run は同じ seed で学習・サンプリングされたことになります。CPU 上の非決定性による差しか残らないので、std が極端に小さいことと辻褄が合います。

## 1. 仮説

| ID | 内容 | 監査での見え方 |
| --- | --- | --- |
| H1 | seed run は全部あるが、実際の `random_seed` は 42 のまま | `train_config.yaml` の `random_seed` = 42、checkpoint のハッシュが seed42 とほぼ同じ、per-seed 平均がほぼ一致 |
| H2 | seed run が見つからず、集計で黙ってスキップされた | audit で seed 12/22/32 の行が出ない、または `n_repeats` = 0 |
| H3 | seed は正しく変わっており、seed 間の差が本当に小さい | `random_seed` = 12/22/32、checkpoint が別物、per-seed 平均の差が 1e-3 程度 |

## 2. 手順

### Step 1. 監査（読み取りのみ、数分〜数十分）

> 2026-11-18 追記：Table III の出力は見つからなかったため、Step 1 は行わず `make-base` で seed 42 の設定を作り直しました（Step 3 参照）。

Table III の出力（`outputs\` または `outputs_local\`）を持っているマシンで実行します。

```bat
cd C:\Users\ando\Box\DraftMaterials\BigData2026\追加実験Seed
audit.bat C:\path\to\CDSM
```

- `results\audit_<PC名>.csv`：run ごとの `random_seed`（train/sample）、checkpoint の SHA256（先頭 16 桁）と seed42 との一致、サンプル repeat 数、repeat ごとの test ‖ΔΣ‖_F（CA あり/なし）
- `base_configs\`：seed42 run の `train_config.yaml` / `sample_config.yaml` と `manifest.json`。この設定から seed だけを変えて再学習するので、もう一方のマシンでも同じ設定で学習できます。

**判定**

- H3 なら再実行は不要です。Step 4（evaluate / summarize）だけ行い、キャプションを「4 学習 seed × 5 サンプル」と明記します。
- H1 / H2 なら Step 2 に進みます。

### Step 2. 動作確認（各マシン 10〜20 分）

```bat
python seed_experiment.py run --repo C:\path\to\CDSM --machine independence --data-root %USERPROFILE%\.gluonts\datasets --datasets ishares --seeds 99 --max-epochs 1 --sample-repeats 1
```

学習、サンプリング、`results\values_independence.csv` への書き込みまで通ることを確認したら、`outputs_seed\...seed99` を削除します。Diligence（と、合流時に Audacity）でも `--machine diligence --datasets exchange`（`--machine audacity`）で同じことを行います。

### Step 3. 本番（4 seed × 4 データセット × 2 ドメイン = 32 run）

**2026-11-18 の決定**：Table III の出力（seed 42 を含む）はどの PC にも見つかりませんでした。そこで `notebooks/paper.ipynb` の学習コマンドを λ=0 に置き換えて seed 42 の設定を作り直し（`seed_experiment.py make-base`）、seed 42 も含めた 4 seed をすべて新しく学習します。これで 4 seed とも同じコード、同じ設定、同じデータで学習したことになります。

- `make-base` は、Independence と同じコードのコピー（`C:\Users\ando\CDSM`）を使って 1 台で 1 回だけ実行します。できた `base_configs\` は Box 経由で他の台と共有されます。
- サンプリングには `checkpoints\best-matrix_corr_fro.ckpt` を使います（paper.ipynb のサンプリングコマンドと同じ）。DataLoader の `num_workers` は Windows で安全な 0 にしています（結果には影響しません）。

| マシン | 担当（実行順） | run 数 | 開始 |
| --- | --- | --- | --- |
| Independence | seed 42（Exchange, FX29）→ seed 12（全データセット） | 12 | 先行 |
| Diligence | seed 42（Industry, iShares14）→ seed 22（全データセット） | 12 | 先行 |
| Audacity | seed 32（全データセット） | 8 | オンラインになり次第 |

```bat
run_independence.bat    (Independence で)
run_diligence.bat       (Diligence で)
run_audacity.bat        (Audacity で)
```

- seed 42 を先に回すのは、論文の値（CDSM+CA：4.58 / 8.15 / 8.42 / 3.30）が再現できるかを早めに確かめるためです。大きくずれたら、ほかの seed に進む前に止めて設定を見直します。
- Independence と Diligence が終われば seed 42/12/22 の 3 seed がそろい、`summarize` で暫定の mean ± std を出せます。Audacity の seed 32 が入ったら、もう一度 `summarize` を実行して 4 seed に更新します。
- Audacity をオンラインにできない場合は、先に終わった台で `run_independence.bat --seeds 32`（または `run_diligence.bat --seeds 32`）を実行して引き継げます。同じ seed を 2 台で同時に回さないようにしてください。
- 学習中の検証サンプリング（毎 epoch、1000 ステップ × 5 サンプル）は論文と同じにしています。CPU ではここが最も重いので、動作確認で 1 epoch の時間を測り、全体の所要時間を見積もり直します。
- Audacity でも、同じコードのコピー、venv、`%USERPROFILE%\.gluonts\datasets` の 4 データセットが必要です。
- バッチの先頭にある `CDSM_REPO` を各マシンのリポジトリの場所に書き換えてから実行してください。
- もう一度実行すると、終わった run（`SEED_EXPERIMENT_DONE` があるもの）は飛ばして続きから再開します。
- 学習設定（epoch 数、学習中の検証サンプリング、checkpoint の選択基準）は seed42 と同じにしています。`--disable-epoch-metrics` を付けると速くなりますが、どの checkpoint が選ばれるかが変わるので本番では使いません。
- 大きなファイル（checkpoint、`samples.pt`）はリポジトリ側の `outputs_seed\` に保存し、Box には置きません。Box の `results\` に入るのは、run ごとの指標 CSV だけです。

### Step 4. 集計

各 run の指標は `run` が終わるたびに `results\values_<マシン名>.csv` に書き込まれるので、集計は Box 上の CSV だけでできます。

```bat
python seed_experiment.py summarize
```

- `results\summary_seed.csv`：データセット × モデル（CDSM-T/S、CA あり/なし）ごとに、seed 別の平均、seed 間の std（ddof=1）、seed 内（サンプリング）の std、全 20 値の pooled std、seed42 の値と論文の値
- `results\table3_cdsm_rows_seed.tex`：Table III の CDSM 行をそのまま置き換えられる LaTeX（mean = seed 平均の平均、± = seed 平均の std）

**seed42 の再現チェック**：`seed42_mean` が論文の値（CDSM+CA：Exchange 4.58、FX29 8.15、Industry 8.42、iShares 3.30）と合うことを確かめます。マシンにリポジトリの `scripts/plot_correlation_mean.py` があれば、論文と同じ集計関数を使います（公開リポジトリにはこのファイルがありません）。無い場合は内蔵の実装（窓ごとの `np.cov`、CA ではコンテキストと予測を連結、test 窓で平均）を使うので、値がずれたら連絡してください。

## 3. 結果の使い方（キャプションと R2-2）

- **報告する std**：「学習 seed 間の std（4 seed の平均値の std）」を主に使い、サンプリングによる std は補足とします。R2-2 が問うている「複数 run のばらつき」に当たるのは前者です。
- **キャプション案**：「Mean ± std over 4 training seeds (42, 12, 22, 32); each seed is averaged over 5 sampling runs on the test windows.」
- **H1 が確認された場合**：現行の Table III の ± は、実質的に 1 つのモデルのサンプリング 5 回ぶんのばらつきです。再実行の結果で差し替えます。差し替えが間に合わないときは、キャプションを「single trained model, 5 sampling runs」に直すのが誠実です。
- **UCDSM の行**：現在は 1 run × 5 サンプルです。CDSM と同じ基準で比べるなら UCDSM も seed を振る必要がありますが、今回の範囲には入れていません。必要なら fdiff 用に同じ仕組みを追加します（さらに 24 run）。

## 4. ファイル一覧

| ファイル | 内容 |
| --- | --- |
| `seed_experiment.py` | audit / export / make-base / run / evaluate / summarize |
| `audit.bat` | Step 1（監査と seed42 設定のエクスポート） |
| `run_independence.bat` / `run_diligence.bat` / `run_audacity.bat` | Step 3（各マシンの担当分） |
| `base_configs\` | seed42 の設定（export で作成） |
| `results\` | 監査 CSV、指標 CSV、集計表 |
| `logs\` | ドライバとジョブごとのログ |
