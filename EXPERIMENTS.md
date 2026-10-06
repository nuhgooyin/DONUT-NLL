# DONUT-NLL experiment log

Tracks every training/evaluation run worth remembering: what changed, where it ran, and what came
out. Read this before starting a new experiment, so you don't repeat one or reuse a name/`--name`
that's already taken. See `CLUSTER_RUNBOOK.md` for how to run things on each cluster.

**Conventions**
- One row per experiment in the summary table below, newest first. One `## Experiment NNNN` section
  per row with the full detail, using the template at the bottom of this file.
- **ID** = zero-padded running number (`0001`, `0002`, ...), also used as the `--name` prefix for
  training runs so checkpoints/logs/samples don't collide across clusters or agents:
  `ckpts/<ID>-<short-slug>/`.
- **Baseline** = the released `donut-nll` checkpoint (`epoch=29-step=228300.ckpt`), reproduced on the
  full Waymo validation set (44,097 scenes) on Killarney on 2026-09-23. Every later experiment's
  "vs baseline" column is a delta against this row, not against the paper directly.
- Record a row **as soon as a run starts**, with status `running`, so two experiments don't quietly
  reuse the same `--name` on different clusters. Fill in the rest when it finishes or fails.
- An experiment that fails, OOMs, or gives a null result still gets a row — those are often the most
  useful entries for someone about to try something similar.

## Current conclusions and defaults (as of 2026-09-28)

A summary of what the entries below established. Read this before planning a new experiment, and
keep it updated when an entry changes one of these. How-to details are in `CLUSTER_RUNBOOK.md`
§9–§11.

**Defaults to use**
- **Train with `--checkpoint_social`, batch 8 per GPU, fp32.** Training is identical, memory drops
  about 6×, and throughput is 3.3× that of batch 2 (0003). bf16 was slower and saved almost no
  memory (0001).
- **Use the window policy with `pi=mean`** (confidence averaged over the 3 horizons):
  - about +0.010 mAP on the released model, measured on data it wasn't tuned on (0004);
  - +0.007 to +0.010 on 0008's models, +0.002 on 0005–0007's;
  - never worse in any of 16 retrained runs.
  - It's not yet the default in the TraDiE code: use `TUNE_PI=mean` with the patched copy
    (`slurm/tradie_tune.patch`).
- **Distance policy:** `adam_lr 0.4` (or 40 restarts) gives about −0.002 minFDE (0004). Real but
  small.
- **Comparing training changes:** use 0008's setup (40% data, 10 epochs, 8×H200, ~10 h per run),
  with at least 2 runs per configuration. The 10% setup (0005–0007) is too noisy.

**Noise to expect** (full validation, 44,097 scenes). Compare any difference with these before
calling it real:

| Source | Window mAP | Window miss rate | Distance minFDE |
|---|---|---|---|
| Re-scoring the same samples | 0 (window policy is deterministic) | 0 | ~0.0006 |
| Independent sampling of the same checkpoint | ~0.003 | ~0.0005 | ~0.0006 |
| Different training runs, 10% setup (0005–0007) | SD ~0.01, spread up to 0.024 | up to 0.03 | SD ~0.1 |
| Different training runs, 40% setup (0008) | spread 0.001–0.012 | 0.003–0.006 | 0.006–0.04 |

Training isn't bit-reproducible run to run, because GPU summation order varies (0003).

**Where the models stand** (window mAP / miss rate / minFDE, full validation):
- **released model (paper setting):** 0.509 / 0.088 / 1.009;
- **40% data, 10 epochs (0008 baseline):** 0.432 / 0.118 / 1.187;
- **10% data, 10 epochs (0005):** about 0.267 / 0.249 / 1.98.

**Rejected or ineffective**
- `--num_modes 10`: not significantly better at 10% (p = 0.21, 4 vs 4) and slightly worse at 40%
  (0006–0008).
- bf16 mixed precision (0001).
- Changing the window policy's rectangle size or shape (0004).

**Open questions**
- Does `pi=mean` plus distance `lr 0.4` combine well?
- How do the gains from 6,000 samples split between the sample count and the learning rate at
  6,000 samples? (0004)
- Full training (all data, 30 epochs) with `--checkpoint_social` hasn't been run: about 72 h on
  8×H200.

## Summary

| ID | Date | Name/slug | Cluster | What changed | Data | Status | Soft mAP | mAP | Miss rate | minFDE | vs baseline |
|----|------|-----------|---------|--------------|------|--------|----------|-----|-----------|--------|-------------|
| 0008 | 2026-09-27 | `0008-baseline-sub040-r1`,`-r2`; `0008-modes10-sub040-r1`,`-r2` | TamIA 8×H200 (train), Vulcan (eval) | longer protocol: 40% of training data (194,613 scenes), 10 epochs (4× the steps of 0005/0006); baseline ×2 vs `--num_modes 10` ×2 | shards 0–399, 10 ep; eval full val | done | 0.4306† | 0.4280† | 0.1216† | 1.2059‡ | `num_modes 10` vs 0008 baseline (0.4318 / 0.1182 / 1.1870): **slightly worse** on all three (−0.004 mAP, +0.003 miss, +0.019 minFDE); run-to-run noise much smaller than at 10% |
| 0007 | 2026-09-27 | `0005-baseline-sub010-r3`,`-r4` (TamIA); `0006-modes10-sub010-r3`,`-r4` (Killarney) | TamIA + Killarney (train), Vulcan (eval) | 2 more runs each of 0005 / 0006 → 4 v 4 (clusters mixed; 0006 runs moved off Trillium after system cancellations). Pre-registered: window mAP, Welch t, real if p < 0.05 and diff > 2 SE | shards 0–99 (48,805), 10 ep; eval full val | done | 0.2788† | 0.2777† | 0.2440† | 1.9382‡ | 4-run means for `num_modes 10`; baseline 4-run means 0.2677 / 0.2670 / 0.2487 / 1.9844. Window mAP +0.0107, p = 0.21, diff/SE 1.41: **fails the rule, not significant** |
| 0006 | 2026-09-26 | `0006-modes10-sub010-r1`, `-r2` | TamIA (train), Vulcan (eval) | `--num_modes 10` (default 6); otherwise same as 0005 | train shards 0–99 (48,805), 10 ep; eval full val | done | 0.2835† | 0.2823† | 0.2412† | 1.9323‡ | vs 0005: window mAP +0.019 (both runs above both 0005 runs); miss −0.016, minFDE −0.084 within training noise; **inconclusive, n=2** |
| 0005 | 2026-09-26 | `0005-baseline-sub010-r1`, `-r2` | Trillium (train), Vulcan (eval) | baseline retrain, reduced protocol: batch 8/GPU × 4 H100, `--checkpoint_social`, 10 epochs, `--decay_epochs 10` | train shards 0–99 (48,805), 10 ep; eval full val | done | 0.2638† | 0.2631† | 0.2570† | 2.0167‡ | reduced-protocol reference for 0006 (released model: 0.5060 / 0.0888 / 1.0087); mean of 2 runs |
| 0004 | 2026-09-26 | policy tuning (released ckpt) | Vulcan | TraDiE policy settings only, no retraining: window `pi=mean`; distance `adam_lr 0.4`; 6k samples | val report half (22,116) | done | 0.5200† | 0.5176† | 0.0904† | 1.0172‡ | window mAP **+0.0097** (noise ±0.004); miss rate 0; minFDE −0.0022 (noise ±0.0006); report-half deltas, not full-val |
| 0003 | 2026-09-25 | `ckpt-social` (`--checkpoint_social`) | Killarney | activation checkpointing in `SocialAttention` (new flag, default off) | 990 train scenes | done | n/a | n/a | n/a | n/a | n/a: bit-identical training; 228-agent scene 73 → 12 GiB; batch 8/H100 now fits (9.7 scenes/s, 3.3× batch 2) |
| 0002 | 2026-09-25 | `profile-social-attn` | Killarney | diagnostic only, no training change | 12 train scenes (1-228 agents) | done | n/a | n/a | n/a | n/a | n/a — root-caused the batch-8 OOM to `SocialAttention` |
| 0001 | 2026-09-24 | `bench-precision-batchsize` | Killarney | fp32 vs bf16 × batch size {2,4,8}, no accuracy eval | 990 train scenes | done | n/a | n/a | n/a | n/a | n/a — speed/memory only, bf16 rejected |
| 0000 | 2026-09-23 | `donut-nll` (released ckpt) | Killarney | — (baseline / paper reproduction) | val, full 44,097 | done | 0.5090† | 0.5060† | 0.0888† | 1.0087‡ | — |

†mAP_rectangles (window) policy · ‡fde_adam (distance) policy — see conventions in the detail
section below on why naive/window/distance are reported separately, not blended.

<!-- Add new rows above this line, newest first. -->

---

## Experiments 0007 + 0008: is `--num_modes 10` really better? More runs, and a longer setup

**Date:** 2026-09-27/28 · **Who/what ran it:** Claude. Training on TamIA (4×H100 and 8×H200) and
Killarney (4×H100); evaluation on Vulcan · **Status:** done

**Hypothesis.** Same as 0006: more mixture components help the policy scores. 0005/0006 hinted at
+0.019 window mAP from 2 runs per configuration, inside large run-to-run noise.

**What changed from baseline/previous.**
- **0007** adds 2 more runs of each 0005/0006 configuration on the same reduced setup, for 4 per
  configuration. 0006's r3/r4 were meant for Trillium, but its system cancelled them twice (`CANCELLED
  by 0` and `NODE_FAIL` after about 2 minutes, on different nodes), so they ran on Killarney. The
  decision rule was fixed before any 0007 result was seen:
  - primary metric: window mAP;
  - Welch t-test on 4 vs 4 runs;
  - call it real if p < 0.05 *and* the difference is more than 2 × the pooled SE.
- **0008** uses a longer setup:
  - 40% of the training data, shards 0–399: 194,613 scenes, TamIA's own `data`;
  - 10 epochs, 30,410 optimizer steps (4× 0005/0006);
  - 8×H200 per run, batch 8 per GPU, 64 scenes per step with no gradient accumulation, the same
    effective batch as 0005/0006;
  - `--checkpoint_social`;
  - baseline ×2 vs `--num_modes 10` ×2.

  Before launching, `slurm/scan_scenes.py` scanned the data: most crowded scene 286 agents, 2 scenes
  above 228, p99.9 = 106. The worst possible batch fits in the H200's 141 GB. An 8-GPU test passed
  training and resume, at 56 scenes/s.

**Setup**
- Training jobs:
  - 0007: TamIA 491222 and 491223 (baseline r3/r4, 4 h 31 m each); Killarney 5715402 and 5715403
    (10-mode r3/r4, 5 h 00 m and 5 h 22 m);
  - 0008: TamIA 491234 and 491236 (baseline, 10 h 05 m each), 491235 and 491237 (10-mode, 10 h 21 m
    each).
- All completed 10 epochs.
- Evaluation: Vulcan, full validation (44,097 scenes), 3k samples, default policies and window
  `pi=mean`. Jobs 1209233–1209272, all completed. Scores are in `bench/eval-0007-0008-fullval.txt`.

**Results.** 0007: all 4 runs per configuration (0005/0006 r1–r4), means ± standard deviation.

| metric | baseline (4) | `num_modes 10` (4) | diff | diff/SE | Welch p |
|---|---|---|---|---|---|
| **window mAP (primary)** | 0.2670 ± 0.0092 | 0.2777 ± 0.0120 | +0.0107 | 1.41 | **0.21** |
| window Soft mAP | 0.2677 ± 0.0092 | 0.2788 ± 0.0122 | +0.0111 | 1.46 | 0.20 |
| window miss rate | 0.2487 ± 0.0153 | 0.2440 ± 0.0144 | −0.0047 | −0.45 | 0.67 |
| distance minFDE | 1.9844 ± 0.1135 | 1.9382 ± 0.0938 | −0.0462 | −0.63 | 0.55 |
| naive mAP | 0.1544 ± 0.0161 | 0.1421 ± 0.0157 | −0.0123 | −1.09 | 0.32 |

New runs (the rest are in 0005/0006):

| run | window Soft mAP | window mAP | window miss | distance minFDE | naive mAP |
|---|---|---|---|---|---|
| 0005 r3 | 0.2617 | 0.2612 | 0.2534 | 2.0612 | 0.1338 |
| 0005 r4 | 0.2814 | 0.2807 | 0.2274 | 1.8429 | 0.1702 |
| 0006 r3 | 0.2690 | 0.2680 | 0.2549 | 1.9923 | 0.1388 |
| 0006 r4 | 0.2791 | 0.2781 | 0.2387 | 1.8959 | 0.1442 |

0008, the longer setup:

| run | window Soft mAP | window mAP | window miss | distance minFDE | naive mAP | window mAP `pi=mean` |
|---|---|---|---|---|---|---|
| baseline r1 | 0.4402 | 0.4376 | 0.1153 | 1.1677 | 0.2941 | 0.4456 |
| baseline r2 | 0.4287 | 0.4261 | 0.1212 | 1.2063 | 0.2890 | 0.4346 |
| `num_modes 10` r1 | 0.4299 | 0.4273 | 0.1229 | 1.2089 | 0.2486 | 0.4343 |
| `num_modes 10` r2 | 0.4314 | 0.4287 | 0.1204 | 1.2028 | 0.2492 | 0.4387 |
| **baseline mean** | 0.4345 | 0.4318 | 0.1182 | 1.1870 | 0.2915 | 0.4401 |
| **`num_modes 10` mean** | 0.4306 | 0.4280 | 0.1216 | 1.2059 | 0.2489 | 0.4365 |
| released model (0000, Vulcan) | 0.5118 | 0.5087 | 0.0883 | 1.0093 | 0.3442 | — |

**vs baseline/previous.**
- **0007 fails the pre-registered rule.** With 4 runs each, `num_modes 10`'s window-mAP advantage
  shrank from +0.019 (2 runs each) to +0.011, with p = 0.21 and 1.4 SE. That's consistent with the
  earlier gap coming largely from one strong run (0006 r2). No secondary metric is close.
- **In 0008, `num_modes 10` is slightly worse** on every policy metric: window mAP −0.004, miss rate
  +0.003, minFDE +0.019. Its naive score is much worse (−0.043 mAP), as expected when a 10-component
  model is cut to its top 6 modes. With 2 runs per configuration this isn't a significant loss
  either, but there's no sign of a gain.
- **Noise shrinks a lot with more training.**
  - Window mAP: the spread between two identical runs is 0.0014–0.0115 in 0008, vs a standard
    deviation of about 0.01 across 4 runs and spreads up to 0.024 at 10% data.
  - minFDE: spreads of 0.006–0.039 in 0008, vs a standard deviation of about 0.1 at 10% data.
- **40% data and 10 epochs gets most of the way to the released model** (full data, 30 epochs):
  window mAP 0.43 vs 0.51, and minFDE 1.19 vs 1.01.
- **Window `pi=mean` (0004) helps more as models get better:** about +0.002 mAP at 10%, about +0.008
  in 0008 (+0.007 to +0.010 per run), and +0.010 on the released model. It never hurt in any of the
  16 runs.

**Conclusion.**
- **`--num_modes 10`: rejected.** It isn't significantly better at the reduced setup and is slightly
  worse at the longer one. The 0006 hint was noise.
- **For future comparisons, use the longer 0008 setup** (40% data, 10 epochs, 8×H200, ~10 h and ~80
  GPU-hours per run). The 10% setup is too noisy to detect effects smaller than about 0.02 mAP
  without many runs.
- **Adopt window `pi=mean` by default:** a free, consistent gain that grows with model quality.

**Artifacts.**
- **Final checkpoints** (`epoch=9-step=7630` for 0007, `epoch=9-step=30410` for 0008): on the Mac in
  `ckpts/<name>/`, and on Vulcan. All per-epoch checkpoints stay on the training cluster
  (TamIA/Killarney `ckpts/<name>/`).
- **Training logs:** `csv_logs/<name>/`.
- **Scores:** `bench/eval-0007-0008-fullval.txt`.
- **Scene scan:** `bench/scan-train040.tsv` on TamIA.
- **Evaluation outputs** on Vulcan in `samples/eval/<name>-*`. The eight prediction files are
  ~11 GB each, and Vulcan's scratch isn't backed up.

---

## Experiments 0005 + 0006: reduced-protocol retraining, baseline vs `--num_modes 10`

**Date:** 2026-09-26 · **Who/what ran it:** Claude (training on Trillium and TamIA, evaluation on
Vulcan) · **Status:** done

**Hypothesis.** The policies draw 3,000 samples from the model's mixture distribution and pick their
own 6 predictions, so the mixture doesn't have to be limited to 6 components. More components
(`--num_modes 10`) could represent the distribution of futures better, and the policy scores would
improve.

**What changed from baseline/previous.**
- **0005 (baseline):** default model settings, retrained under a reduced protocol so variants can be
  compared cheaply.
- **0006:** identical except `--num_modes 10`.
- **Both:** `--checkpoint_social` (0003), batch 8 per GPU × 4 H100 (32 scenes/step; `acc_batch_size`
  64 gives 2-step accumulation), `--max_epochs 10 --decay_epochs 10`, `--limit_val_batches 0.1`
  (validation loss during training only).
- **Supporting code:**
  - `train_donut_nll.py`: removed the line forcing `decay_epochs = 32`.
  - `slurm/train.sh`: 4-GPU DDP via `srun`, with fixes for checkpoint loading when resuming, and
    for the open-files limit.
  - `slurm/make_subset.py` and `data_manifests/train_shards000-099.txt`.
  - `sample_donut_nll.py`: when a model has more than 6 modes, the naive predictions keep its 6 most
    probable modes, since Waymo scoring allows 6.
  - `slurm/sample.sh`: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, so our own checkpoints load.

**Setup**
- **Training data:** the 48,805 scenes of Waymo training shards 0–99 (`data_sub010`, symlinks),
  about 10% of the training set. Same manifest on both clusters.
- **Training:**
  - 0005 on Trillium, jobs 992205 and 992206, 4 h 21 m each;
  - 0006 on TamIA, jobs 489779 and 489780, 4 h 48 m each;
  - all completed 10 epochs (`epoch=9-step=7630`).
- **Evaluation:** Vulcan (L40S), full validation of 44,097 scenes, 3k samples, pipeline as in runbook
  §8, policies with default settings. The window policy was also scored with `pi=mean` from 0004.
  The final jobs were:
  - scoring: 1195547 and 1195550 (0005), 1195583 and 1195700 (0006);
  - `pi=mean` scoring: 1195548, 1195551, 1195584 and 1195702.
- **Two independent training runs per configuration**, because training isn't run-to-run
  reproducible (0003).

**Results** (full validation, 44,097 scenes; window = `mAP_rectangles`, distance = `fde_adam`)

| run | window Soft mAP | window mAP | window miss | distance minFDE | naive mAP | window mAP, `pi=mean` |
|---|---|---|---|---|---|---|
| 0005 r1 | 0.2637 | 0.2631 | 0.2635 | 2.0896 | 0.1637 | 0.2657 |
| 0005 r2 | 0.2639 | 0.2631 | 0.2504 | 1.9438 | 0.1498 | 0.2657 |
| 0006 r1 | 0.2712 | 0.2702 | 0.2564 | 2.0363 | 0.1236 | 0.2710 |
| 0006 r2 | 0.2959 | 0.2945 | 0.2260 | 1.8283 | 0.1617 | 0.2975 |
| **0005 mean** | 0.2638 | 0.2631 | 0.2570 | 2.0167 | 0.1568 | 0.2657 |
| **0006 mean** | 0.2835 | 0.2823 | 0.2412 | 1.9323 | 0.1426 | 0.2843 |
| released model (0000, Vulcan) | 0.5118 | 0.5087 | 0.0883 | 1.0093 | 0.3442 | — |

The full score output is in `bench/eval-0005-0006-fullval.txt`.

**vs baseline/previous.**
- **The reduced protocol gives a much weaker model than the released one:** window mAP about 0.27
  vs 0.51, and minFDE about 2.0 vs 1.0. 10% of the data and 10 epochs is far from converged.
  0005/0006 are only meant to be compared with each other, never with the paper.
- **Training noise between identical runs is large at this stage.** Between two identical-config
  runs, window mAP differs by up to 0.024, miss rate by up to 0.030, and minFDE by up to 0.21. (The
  two 0005 runs landing on the same window mAP is a coincidence: their other metrics differ.)
- **0006 − 0005, means of 2 runs each:**
  - window mAP **+0.019**: both 0006 runs are above both 0005 runs;
  - window miss rate −0.016 and minFDE −0.084: both inside the run-to-run spread;
  - naive mAP −0.014, also inside the spread. Naive is expected to suffer somewhat, because a
    10-component mixture cut to its top 6 isn't what the model was trained to output.
- **0004's window `pi=mean` still helps on the retrained models,** but less: +0.0008 to +0.0030 mAP
  per run, vs +0.0097 on the released model. It never hurt.

**Conclusion.** *(Superseded by 0007/0008: with 4 runs each, the effect isn't significant, and with longer training `num_modes 10` is slightly worse.)* **Inconclusive, but promising for window mAP.**
- `--num_modes 10` looks better on window mAP: both of its runs beat both baseline runs, by +0.019
  on average.
- Most of that average comes from one strong 0006 run (0.2945). With only 2 runs per configuration
  and noise this large, it isn't established.
- Miss rate and minFDE differences are within the noise.

**Next steps** (not yet run):
- more runs per configuration (3–4 each), or
- a longer, less noisy protocol, since noise should shrink as models converge. For example: more
  data (TamIA has 40%; the other clusters have the full set) and more epochs, before committing
  to a full 30-epoch run.

**Artifacts.**
- **Checkpoints** (`epoch=9-step=7630.ckpt`, 108 MB each):
  - on the Mac, in `ckpts/0005-baseline-sub010-r{1,2}/` and `ckpts/0006-modes10-sub010-r{1,2}/`;
  - all 10 per-epoch checkpoints on Trillium and TamIA in `ckpts/<name>/`.
- **Training logs:** `csv_logs/<name>/` (Mac, Trillium and TamIA).
- **Evaluation outputs** on Vulcan, `samples/eval/<name>-val.pkl` (about 11 GB each), plus
  `samples/eval/<name>-{default,pimean}/`. Not backed up.
- **Scores:** `bench/eval-0005-0006-fullval.txt`.

---

## Experiment 0004: policy tuning on the released checkpoint

**Date:** 2026-09-26 · **Who/what ran it:** Claude subagent (Vulcan) · **Status:** done

**Hypothesis.** The TraDiE post-processing policies turn 3,000 samples per agent into 6 predictions
using hand-set parameters. Better settings, or more samples, could raise the scores with no
retraining.

**What changed from baseline/previous.** Nothing in the model. All changes are in a separate copy of
the policy code, `/scratch/nuhgoyin/TraDiE-policies-tune`, which applies `slurm/tradie_tune.patch`.
The patch adds environment knobs (`TUNE_PI`, `TUNE_RECT_SCALE`, `TUNE_WIDTH_RATIO`, `TUNE_ADAM_LR`,
`TUNE_ADAM_STEPS`, `TUNE_N_INIT`, `TUNE_POLICIES`), which default to the original behaviour. With the
defaults, it reproduces the original output exactly.

`sample_donut_nll.py` gained `--num_samples` (default 3000). New scripts: `slurm/split_val.py`,
`slurm/tune_split.sh`, `slurm/tune_eval.sh` and `slurm/sample_n.sh`.

**Setup**
- Cluster: Vulcan (L40S).
- **Split** of validation by `int(md5(scenario_id), 16)`:
  - report half: odd, 22,116 scenes;
  - tune half: even, 21,981 scenes;
  - screen quarter: `% 4 == 0`, 10,921 scenes, inside the tune half.
- **Method:** settings were screened on the screen quarter, and the best were confirmed on the report
  half. Only report-half numbers are claimed.
- **Cost:** 82.6 of the 100 L40S GPU-hour budget; 126 jobs, all completed.

**Results**

*Vulcan baseline check (full validation, default settings):*
- naive 0.3761 / 0.3442 / 0.1563 / 1.3132;
- window 0.5118 / 0.5087 / 0.0883;
- distance 1.0093.

That's within 0.0003 (naive) and 0.0028 (policies) of Killarney's row 0000.

*Noise bands:*
- The window policy is deterministic on fixed samples: identical reruns.
- Its noise between independent sampling draws is about ±0.004 mAP and ±0.0007 miss rate on the
  report half. This comes from the Vulcan vs Killarney full-validation gap, scaled for half size.
- Distance policy: ±0.0006 minFDE, both between reruns and between sampling draws.

*Screening (screen quarter, 3k samples; default window 0.5199 / 0.5170 / 0.0849, default distance
0.9813):*
- **Rectangle scale** 0.8–1.2 and **width ratio** 0.4/0.6: all worse on mAP or miss rate. The default
  rectangles already equal the metric's hit region.
- **`pi=mean`** (confidence = covered-sample count averaged over the 3 horizons, instead of the 8 s
  count only): **0.5322 / 0.5301 / 0.0849**.
- **Distance:**
  - lr 0.4: **0.9775** (−0.0038);
  - 40 restarts: 0.9786;
  - lr 0.4 + 40 restarts: 0.9779;
  - 600 steps: 0.9802;
  - lr 0.1: 0.9834 (worse);
  - lr 0.8: 0.9813 (no change).

*Report half (22,116 scenes):*

| Setting | Soft mAP | mAP | Miss rate | minFDE | Δ vs default |
|---|---|---|---|---|---|
| window default, 3k (2 identical runs) | 0.5110 | 0.5079 | 0.0904 | — | — |
| **window `pi=mean`, 3k** | **0.5200** | **0.5176** | 0.0904 | — | **+0.0090 / +0.0097 / 0** |
| window `pi=mean` + scale 0.95 | 0.5174 | 0.5146 | 0.0919 | — | rejected: miss rate worse |
| window default, 6k | 0.5111 | 0.5083 | 0.0887 | — | +0.0001 / +0.0004 / **−0.0017** |
| window `pi=mean`, 6k | 0.5197 | 0.5176 | 0.0887 | — | +0.0087 / +0.0097 / −0.0017 |
| distance default, 3k (runs 1 / 2) | — | — | — | 1.0197 / 1.0191 | — |
| **distance lr 0.4, 3k** | — | — | — | **1.0172** | −0.0022 |
| distance 40 restarts, 3k | — | — | — | 1.0174 | −0.0020 |
| distance lr 0.4, 6k | — | — | — | 1.0153 | −0.0041 |

**vs baseline/previous.** Measured on the report half against default settings with the same samples:
- **window `pi=mean`:** mAP **+0.0097**, Soft mAP +0.0090, about 2.5× the ±0.004 band; miss rate
  unchanged by construction;
- **distance lr 0.4:** minFDE −0.0022, about 3× the ±0.0006 band;
- **6,000 samples:** window miss rate −0.0017 (about 2× its band), and minFDE a further −0.0019 on
  top of lr 0.4.

These are report-half numbers, not full-validation numbers, so they aren't directly comparable to row
0000's full-validation values.

**Conclusion.**
- **Adopt window `pi=mean`.** It's the only sizeable gain here, it's free, and it reproduced on an
  independent 6k draw.
- **Distance lr 0.4 (or 40 restarts)** gives a real but small gain, also free.
- **6k samples** give small, real gains in miss rate and minFDE, at about 2× window-policy cost.
- **Unresolved:**
  - 6k with the default distance learning rate wasn't run on the report half. On the screen quarter
    it beat lr 0.4 at 6k.
  - lr 0.4 + 40 restarts wasn't confirmed.
  - The distance policy's cost is dominated by GPU kernel launches, so batching scenes could cut it
    sharply.

**Artifacts.**
- **Mac:** `bench/0004-policy-tuning/` (43 `*.metrics.txt` score files) and `slurm/tradie_tune.patch`.
- **Vulcan, under `/scratch/nuhgoyin/DONUT-NLL/samples/tune/`** (not backed up):
  - samples: `donut-nll-val.pkl` (3k, 11 GB) and `s6k_{screen,report}/`;
  - `split/`;
  - `full_default/`;
  - `report*/` and `screen/<config>/`.
- **Logs:** `slurm_logs/tune-*`.

---

## Experiment 0003: activation checkpointing for `SocialAttention`

**Date:** 2026-09-25 · **Who/what ran it:** Claude subagent (Killarney) · **Status:** done

**Hypothesis.** Recomputing `SocialAttention`'s per-edge activations in the backward pass, instead
of storing them, removes the crowded-scene memory spike found in 0002 without changing results,
and unblocks larger batch sizes.

**What changed from baseline/previous.** New model flag `--checkpoint_social` (store_true, default
off). When on and training, `SocialAttention.forward` runs its original body under
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. This covers the per-edge
position/Fourier features, the attention, and the feed-forward block. Dropout is reproduced because
RNG state is preserved. Flag off is the original code path; parameters are unchanged, so old
checkpoints load. Files changed: `layers/donut_attentions.py`, `modules/donut_net.py`,
`predictors/donut_nll.py`. `bench_memory.py` and `profile_memory.py` gained a `--checkpoint_social`
option. New scripts: `slurm/test_checkpoint_social.py`, `slurm/ckpt_social_job.sh` and
`slurm/train_1ep_ckpt_social.sh`.

**Setup**
- Cluster: Killarney. H100 80 GB, plus L40S 44 GB for the L40S table. `kn117` excluded.
- Data: the same 990 training scenes as 0001 and 0002, frozen in a copy
  (`ckpt_social/data/...`), because the Killarney download is adding scenes to
  `data/waymo/processed/train`.
- Jobs:
  - Equivalence: default kernels 5684491 (random weights) and 5684524 (released weights);
    deterministic 5684522 and 5684523.
  - Benchmarks: H100 on 5684492, H100 off 5684493 (same node), L40S on 5684494.
  - Scaling profile: 5684495.
  - One epoch: 5684651.

**Results**

*Equivalence.* In default (non-deterministic) mode, two runs with the flag *off* already disagree:
up to about 30% loss difference from random weights, and 0.05–1.2% from the released weights. That's
GPU summation order amplified through 9 decoding steps. With
`torch.use_deterministic_algorithms(True)`, loss, every gradient, and the weights after one AdamW step
are **exactly identical** (difference 0) with the flag on vs off. This held for 3 batches, including
the 228-agent scene alone, from both random and released weights.

*Memory and speed, H100, fp32:*

| batch | flag | worst: most agents | random batches, peak | scenes/s | step time |
|---|---|---|---|---|---|
| 2 | off | OOM | 14.6 GiB | 2.97 | 0.67 s |
| 2 | on | 19.7 GiB | 11.2 GiB | 2.49 | 0.80 s |
| 4 | off | OOM | 24.4 GiB | 5.81 | 0.69 s |
| 4 | on | 34.5 GiB | 17.5 GiB | 4.87 | 0.82 s |
| 8 | off (0001) | OOM | OOM at step 17 | — | — |
| 8 | on | **55.9 GiB** | 29.5 GiB | **9.68** | 0.83 s |

*L40S 44 GB, flag on:* the 228-agent scene fits at batch 1 (12.3 GiB). Batch 4 worst case is
34.0 GiB (3.08 scenes/s); batch 8 runs out of memory on the crowded batch.

*Per-scene peak, batch size 1, flag on vs 0002:*

| agents | 1 | 15 | 36 | 46 | 228 |
|---|---|---|---|---|---|
| on | 0.25 GiB | 2.08 GiB | 3.48 GiB | 3.55 GiB | **12.2 GiB** |
| off (0002) | 0.26 GiB | 2.3 GiB | 5.1 GiB | 5.9 GiB | 73.1 GiB |

*One epoch, batch 8, flag on, 990 scenes, H100:*
- No out-of-memory, finite losses.
- Training took 1 min 56 s, vs 7 min 2 s for the earlier batch-2 run without the flag.
- `train_loss_epoch` 48,474 and `val_loss` 5,647, vs 20,128 and 6,800 for that earlier run.
- These losses aren't comparable: single, non-reproducible runs, the per-epoch average of a
  heavy-tailed loss, and validation on different scene subsets. Given the exact-equivalence
  result, read this only as a sanity check.

**vs baseline/previous.** Not an accuracy experiment. Compared with 0001 and 0002:
- memory on the worst scene falls about 6× (73 → 12 GiB);
- speed at a fixed batch size falls about 16% from the recomputation;
- because step time is roughly flat across batch sizes, batch 8 gives **about 3.3× the throughput**
  of the old batch-2 setting (9.7 vs 2.97 scenes/s per H100).

**Conclusion.** Adopt it. Train with `--checkpoint_social` at **batch 8 per H100**, about 23 GiB of
headroom on the worst batch seen. **L40S can now train, at batch 4 per GPU.**

Open points:
- The full training set may contain scenes with more than 228 agents. The worst case here comes
  from 990 scenes.
- Default training isn't run-to-run reproducible regardless of the flag (GPU summation order).
  Deterministic mode fixes this at an unmeasured speed cost.
- Not yet measured: 4-GPU DDP memory and speed.

**Artifacts.**
- Mac `bench/ckptsoc-*.json`, `bench/ckptsoc-profile-on-5684495.{json,txt}`, `slurm_logs/ckptsoc-*.out`
  and `slurm_logs/donut-nll-1ep-ckptsoc-5684651.out`.
- `csv_logs/ckpt-social-1ep-bs8/`.
- Killarney: the same files, plus `ckpts/ckpt-social-1ep-bs8/` and the frozen data copy
  `ckpt_social/` (1.3 GB; can be deleted).

---

## Experiment 0001: precision (fp32 vs bf16) and batch-size scaling benchmark

**Date:** 2026-09-24 · **Who/what ran it:** Claude (this session) · **Status:** done

**Hypothesis.** Switching training to bf16 mixed precision would cut GPU memory enough to afford a
larger batch size (batch 2 was already OOMing on some batches at training time), and/or speed up
each training step, moving batch size toward the paper's 8-per-GPU setting.

**What changed from baseline/previous.** No model/training code changed. Wrote
`slurm/bench_memory.py`: runs full training steps (forward → `compute_loss` → backward → AdamW
step, matching `DonutNLL.training_step`) under `torch.autocast('cuda', dtype=torch.bfloat16)` vs
plain fp32, at batch sizes 2/4/8, both on random batches (throughput) and on the batch's
worst-case scenes by agent-count/map-point-count (peak memory). `torch.set_float32_matmul_precision('medium')`
was already active in both conditions (that's the existing baseline behavior, not part of this test).

**Setup**
- Cluster: Killarney
- Data: `data/waymo/processed/train`, 990 scenes (the small subset present on Killarney at the time)
- Jobs: `slurm/bench_memory.sh` — H100 80GB job 5660573, L40S 44GB job 5660574 (`--exclude=kn117`,
  the known-bad node). 30 timed steps per config after 3 warm-up steps.
- No accuracy/loss-quality eval — this measured only speed and peak memory, not whether bf16 keeps
  training numerically sound.

**Results** (H100; L40S showed the same pattern at ~35% lower absolute throughput)

| batch size | precision | scenes/s | step time | peak mem (random batches) |
|---|---|---|---|---|
| 2 | fp32 | 2.78 | 0.72 s | 14.6 GiB |
| 2 | bf16 | 2.09 | 0.96 s | 14.0 GiB |
| 4 | fp32 | 5.48 | 0.73 s | 25.2 GiB |
| 4 | bf16 | 4.16 | 0.96 s | 24.0 GiB |
| 8 | fp32 | OOM at step 17 | — | >80 GiB |
| 8 | bf16 | OOM at step 17 | — | >80 GiB |

**vs baseline/previous.** N/A (no accuracy comparison run — this was a systems benchmark, not a
model-quality experiment). Directionally: bf16 was ~25% *slower* than fp32 at every batch size
tested and saved only ~4% peak memory — the opposite of the hypothesis.

**Conclusion.** bf16 is a dead end here — dropped. The useful finding was incidental: step time is
essentially flat from batch 2→4 (0.72s→0.73s), meaning the GPU is mostly idle waiting on per-step
overhead at these batch sizes, so batch size (not precision) is the real lever for throughput —
*if* the batch-8 OOM can be fixed. That OOM happens at the same step (17) regardless of precision,
which motivated Experiment 0002.

**Artifacts.** `bench/bench_memory-5660573.json` (H100), `bench/bench_memory-5660574.json` (L40S),
both on the Mac (synced from `/scratch/nuhgoyin/DONUT-NLL/bench/` on Killarney) and in
`slurm_logs/donut-nll-bench-{5660573,5660574}.out`.

---

## Experiment 0002: memory profiling — root cause of the batch-size-8 OOM

**Date:** 2026-09-25 · **Who/what ran it:** Claude (this session) · **Status:** done (diagnostic)

**Hypothesis.** The batch-8 OOM in Experiment 0001 (and the earlier batch-2 OOM during the
Killarney 1-epoch training run) is caused by a small number of unusually crowded scenes, not a
uniform per-scene memory cost — and one of the four attention types (temporal/road/social/mode) is
likely the culprit given how their edge counts are constructed.

**What changed from baseline/previous.** No model/training code changed. Wrote
`slurm/profile_memory.py`: (1) runs single-scene (batch size 1, fp32) training steps across 12
scenes spanning the full range of agent counts in the training data, recording peak memory and
edge counts per attention type (via a monkey-patch on each `*Attention.create_edges`); (2) runs a
`torch.profiler` memory-profiled step on the single most agent-heavy scene, grouped by op.

**Setup**
- Cluster: Killarney, H100 80GB, job 5676148 (`slurm/profile_memory.sh`)
- Data: `data/waymo/processed/train`, same 990-scene set as Experiment 0001

**Results.** Peak memory scales with agent count, and social-attention edge count is what
explains it — it grows roughly quadratically with agents, unlike the other three attention types:

| agents in scene | SocialAttention edges | peak memory (1 scene, fp32) |
|---|---|---|
| 1 | 96 | 0.26 GiB |
| 15 | 11,070 | 2.3 GiB |
| 46 | 134,860 | 5.9 GiB |
| **228** | **3,663,964** | **73.1 GiB** |

The 228-agent scene *alone*, at batch size 1, uses 73.1 GiB — essentially the entire 80 GiB H100.
The profiler's top-memory ops on that scene (`aten::empty` 83.6 GB, `aten::mul` 71.5 GB, `aten::mm`
41.3 GB, `aten::cat` 24.4 GB, `aten::cos`/`aten::sin` ~12 GB each self-CUDA-mem) are consistent with
per-edge Fourier position features and small per-edge networks in `SocialAttention`, applied over
that ~3.7M-edge tensor.

**vs baseline/previous.** N/A — diagnostic, not a training/accuracy experiment.

**Conclusion.** `SocialAttention` is the root cause of both the batch-2 crash seen in the original
1-epoch Killarney training run and the batch-8 OOM in Experiment 0001: a single crowded scene (not
the batch size per se) can exceed 80 GiB on its own. This also explains why the released
checkpoint's batch size 8 (on the authors' hardware/software versions) may not reproduce here —
either their torch/torch_geometric versions handled this more efficiently, or their training data
sampling avoided/truncated such scenes. **Proposed next step (not yet run):** activation
checkpointing on `SocialAttention`'s forward pass, to recompute the per-edge activations during
backward instead of storing them — should leave results numerically identical while sharply
cutting peak memory, potentially unblocking batch sizes 4-8.

**Artifacts.** `bench/profile_memory-5676148.json` (per-scene scaling data) and
`bench/profile_memory-5676148.txt` (full profiler table), on the Mac and in
`slurm_logs/donut-nll-profile-5676148.out` on Killarney.

## Experiment template

Copy this block for each new row's detail section. Delete fields that don't apply (e.g. no
training args for a policy-only experiment); don't leave placeholders unfilled silently — write
`N/A` or `unknown` rather than deleting a field you just didn't check.

```markdown
## Experiment NNNN: <short name>

**Date:** YYYY-MM-DD · **Who/what ran it:** you / agent name · **Status:** planned | running | done | failed

**Hypothesis.** One or two sentences: what you expected to change and why.

**What changed from baseline/previous.** Exact diff — training args, code change (link the commit
or paste the diff), data subset, or policy parameters. Don't just say "increased num_modes"; say
`--num_modes 6 → 10`, and note the checkpoint/branch it's based on (this repo has local
uncommitted changes — say whether you mean those, or a git ref).

**Setup**
- Cluster: Killarney / Trillium / TamIA / Vulcan
- Data: full val (44,097) / smoke (286, file 00000) / training subset (which shards) / other
- Training: `--name`, batch size, GPUs, epochs (planned vs actually completed), wall time
- Eval: sample.sh batch size, num_samples, which policies applied
- Job IDs: (for finding logs later, e.g. `slurm_logs/donut-nll-1ep-5628452.out`)

**Results**

| | Soft mAP | mAP | Miss rate | minFDE |
|---|---|---|---|---|
| naive | | | | |
| window (mAP_rectangles) | | | | |
| distance (fde_adam) | | | | |

Ignore `min_ade`/`overlap_rate` from eval_waymo.py output — not meaningful with this pipeline's
partial trajectories (see CLUSTER_RUNBOOK.md).

**vs baseline/previous.** State the deltas explicitly (e.g. "window mAP +0.02, miss rate -0.01,
minFDE unchanged") and whether that's within the noise band established by repeat runs (~±0.002 on
naive, ~±0.005-0.01 on policy scores at full validation size — wider on smaller eval subsets).

**Conclusion.** Worth pursuing further? Dead end? Needs a bigger run to tell? One line.

**Artifacts.** Where the checkpoint, samples pickle, and logs live (cluster + path), so someone else
can pull them without rerunning.
```
