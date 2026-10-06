# DONUT-NLL on Alliance clusters: setup and validation runbook

This runbook is for agents that set up this repo on an Alliance GPU cluster (Killarney, Trillium,
TamIA, Vulcan), confirm the setup reproduces the paper, and then train and evaluate models. For
setup, follow §0–§8 in order. For experiments, see §9 (training), §10 (training data) and §11
(what exists where right now). Results and conclusions live in `EXPERIMENTS.md`: read its "Current
conclusions and defaults" section before planning a new experiment. A setup is **confirmed** when the checks in [§7](#7-smoke-test-1-validation-file-30-min)
and [§8](#8-full-validation-44097-scenes-4-h) match the expected numbers within tolerance.

Everything here was run end to end on **Killarney** on 2026-09-23. **Trillium**, **TamIA** and
**Vulcan** passed the smoke test (§7) on 2026-09-24. Each needs its own `sbatch` flags; §2 lists
them. The full validation run (§8) has only been done on Killarney.

---

## 0. Ground rules

- **You run on the user's Mac and reach clusters over SSH.** Every cluster login needs a Duo
  approval that only the user can give. The user opens a shared connection with `ssh -fN <host>`,
  and your `ssh`/`rsync` commands reuse it. Before any remote work, run `ssh -O check <host>`. If it
  says `No such file or directory`, or a command fails with
  `Permission denied (keyboard-interactive)`, stop and ask the user to run `ssh -fN <host>`. Never try
  to get around Duo.
- **Run remote commands in a login shell** (`ssh <host> 'bash -lc "..."'`). A plain
  `ssh <host> '...'` doesn't define `module`, and it may not have Slurm (`sbatch`, `squeue`) on `PATH`.
- **Don't run heavy work on login nodes.** Setting up the environment and small checks are fine.
  Preprocessing, sampling and training go through `sbatch`.
- **Don't touch other agents' outputs.** Shared, read-only once set up: `data/`, `ckpts/donut-nll/`,
  `~/envs/donut`, `../TraDiE-policies`. Give everything you create a unique name: run names,
  `SAVE_DIR`, work directories, job names (see [§8b](#8b-running-experiments-alongside-other-agents)).
- **The Waymo data and the released checkpoint are licensed** to the user for non-commercial use
  (Waymo Dataset License Agreement). Never copy them anywhere except the user's own cluster
  directories and Mac.
- **Never push to `origin`.** It's the authors' repo (`MKnoche/DONUT-NLL`).

---

## 1. What you need

| Item | Where it comes from | Size |
|---|---|---|
| This repo **with local changes** (Mac/CUDA fixes, `slurm/` scripts) | The user's Mac: `~/Desktop/Courses/CSC490/DONUT-NLL`. The GitHub version lacks the changes. | small |
| TraDiE-policies (scoring code) | `https://github.com/MKnoche/TraDiE-policies`, tested at commit `74493c38338efe37491c95d073334169d0521216` | small |
| Released checkpoint `epoch=29-step=228300.ckpt` | `https://omnomnom.vision.rwth-aachen.de/data/donut-nll.tar.gz` (also on the Mac in `ckpts/donut-nll/`) | 33 MB |
| Waymo Open Motion Dataset **v1.3.0** validation, `scenario` format, 150 TFRecord files | Waymo bucket (see [§5](#5-get-the-validation-data)); a verified copy is on the Mac and on Killarney | 41,180,912,383 bytes |

Scratch space used by a full validation run: raw 39 GB, processed 57 GB, samples and policy outputs 25 GB.

---

## 2. Cluster facts

Fill in the "verify" cells before submitting anything, using the discovery commands below.

| | Killarney (full run passed) | Trillium (smoke test passed) | TamIA (smoke test passed) | Vulcan (smoke test passed) |
|---|---|---|---|---|
| Login host | `killarney.alliancecan.ca` | `trillium-gpu.alliancecan.ca` (GPU login node; GPU jobs must be submitted from it) | `tamia.alliancecan.ca` (verified 2026-09-24) | `vulcan.alliancecan.ca` (verified 2026-09-24) |
| Slurm account | `aip-ajbonner` | `def-ajbonner` | `aip-ajbonner` | `aip-ajbonner` |
| GPU request for 1 GPU | `--gpus-per-node=h100:1` (80 GB) or `l40s:1` (48 GB) | `--gpus-per-node=1` (H100 80 GB, 4 per node). Only 1 GPU or a multiple of 4 is allowed. **`--mem` is rejected** (a quarter node gets 24 cores and 187.5 GiB) | **Whole nodes only:** `--gpus-per-node=h100:4` (4× H100 80 GB) or `h200:8`. `h100:1` is refused, and there is no `l40s` type | `--gpus-per-node=l40s:1`: **L40S only** (48 GB, 4 per node), so no training (see below) |
| Where to submit from | `/scratch/$USER` (**submitting from `/home` is refused**) | `$SCRATCH` = `/scratch/$USER`. **`$HOME` is read-only on compute nodes**, so submitting from `~` fails | `/scratch/n/$USER` (note the `/n/`; `/home` also allowed) | `/scratch/$USER` (`/home` also allowed) |
| Software stack | CVMFS `StdEnv/2023`, `python/3.11`, `scipy-stack` | Same as Killarney, same package versions | Same as Killarney, same package versions | Same as Killarney, same package versions |
| CPU-only jobs allowed? | No, all partitions are GPU partitions | Not from the GPU login node. Run CPU work as 1-GPU jobs | Yes (`cpubase_*`): run preprocessing and merge/eval without a GPU | Yes, but only 2 CPU nodes; GPU jobs start sooner |

Discovery commands (run on the login node through `bash -lc`):

```bash
sshare -U -u $USER                          # account name(s)
sinfo -o "%P %G %l %c %m" | sort -u         # partitions, GPU types, time limits, CPUs, memory per node
echo "HOME=$HOME SCRATCH=${SCRATCH:-unset}"; ls -ld /scratch/$USER /project 2>&1
module load StdEnv/2023 python/3.11 scipy-stack && python --version
diskusage_report                            # quotas
```

Things to watch for:
- **Trillium** forbids bulk transfers on its login nodes. Large downloads must run on its data
  transfer node `tri-dm1.scinet.utoronto.ca` (SSH alias `trillium-dm`), which sees the same `$HOME`
  and `/scratch`. Jobs are still submitted from `trillium`. On 2026-09-27 Trillium's system
  cancelled our jobs twice (see §12), so check it's healthy before relying on it.
- **Trillium** has a **24 h** maximum and a **15 min** minimum job length on its main partition,
  25 TiB of scratch, and a read-only `$HOME` on compute nodes. The harmless warning
  `kernel cache directory could not be created` comes from that.
- **TamIA** allocates **whole 4-GPU nodes** only, has a **24 h job limit**, and gives 500 GiB of
  scratch. A 16-task policy array would take 16 whole nodes there, so use fewer chunks. Training
  chains need jobs of 24 h or less.
- Some clusters allocate **whole nodes** or have a minimum GPU count per job. If a 1-GPU request is
  rejected, read the cluster's documentation on `docs.alliancecan.ca` before changing anything.
- **GPU memory:** on 44–48 GB GPUs (such as L40S), **training** runs out of memory even at batch
  size 2. Sampling at batch size 4 works on H100 (full val) and on L40S (smoke test only; the
  largest scenes haven't been tried on L40S). Policies ran on L40S. On other GPUs, lower
  `BATCH_SIZE` for sampling if it runs out of memory.

The scripts in `slurm/` contain **no** account, GPU or memory settings, because each cluster needs
different ones (Trillium rejects `--mem`, TamIA only allocates whole nodes, and so on). Put them on
the command line with these prefixes. All were tested on 2026-09-24:

| Cluster | `SB_GPU` (sample, policies, eval, training) | `SB_CPU` (preprocess, merge_eval) |
|---|---|---|
| Killarney | `sbatch --account=aip-ajbonner --mem=64G --gpus-per-node=h100:1` | `sbatch --account=aip-ajbonner --mem=64G --gpus-per-node=l40s:1` (no CPU-only partitions) |
| Trillium | `sbatch --account=def-ajbonner --gpus-per-node=1` | same as `SB_GPU` |
| TamIA | `sbatch --account=aip-ajbonner --mem=64G --gpus-per-node=h100:4` (whole node) | `sbatch --account=aip-ajbonner --mem=64G` |
| Vulcan | `sbatch --account=aip-ajbonner --mem=64G --gpus-per-node=l40s:1` | same as `SB_GPU` |

On Killarney, policies and eval also run fine on `--gpus-per-node=l40s:1`, and L40S jobs usually
start sooner than H100 jobs.

Two more rules on every cluster:
- **Pass script settings with `--export=ALL,NAME=value,...`**, not as `NAME=value sbatch ...`.
  Trillium's `sbatch` silently drops variables set the second way. The job then falls back to the
  default paths and **overwrites shared outputs**.
- **Create `slurm_logs/` before the first submission** (`mkdir -p slurm_logs` in the repo root).
  The §4.1 copy leaves it out, and Slurm fails without it.

---

## 3. SSH access (the user does this once per cluster)

Add each cluster to `~/.ssh/config` on the Mac. The Killarney entry already exists. `ControlMaster`
lets one Duo approval serve every later command for 4 idle hours.

```
Host trillium
    HostName trillium-gpu.alliancecan.ca
    User nuhgoyin
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 4h
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

The Mac's `~/.ssh/config` has entries like this for `killarney`, `trillium`, `trillium-dm`,
`tamia` and `vulcan`. The keep-alive lines don't prevent drops from a VPN or network change:
those close every connection at once, and the user must rerun `ssh -fN <alias>` for each one
needed.

The user registers the Mac's public key (`~/.ssh/id_ed25519.pub`) in CCDB, enrolls in Duo (both
already done), and runs `ssh -fN trillium`. The connection closes if the Mac sleeps. Background
watchers should check `ssh -O check <host>` on every poll and report when the connection drops.
The jobs themselves keep running on the cluster.

---

## 4. Code, environment and checkpoint

Below, `$W` is your chosen working directory on the cluster (Killarney: `/scratch/$USER`), `<host>`
is the SSH alias, and `<W>` is the same path written out for the Mac-side commands.

**4.1 Copy the code from the Mac** (run on the Mac, from the repo root):

```bash
cd ~/Desktop/Courses/CSC490/DONUT-NLL
ssh <host> "mkdir -p <W>/DONUT-NLL/slurm_logs"
rsync -a --exclude '.venv*' --exclude data --exclude ckpts --exclude csv_logs \
      --exclude slurm_logs --exclude samples --exclude smoke --exclude .git --exclude '__pycache__' \
      ./ <host>:<W>/DONUT-NLL/
```

**4.2 Get TraDiE-policies** next to the repo, on the login node:

```bash
ssh <host> 'cd <W> && git clone https://github.com/MKnoche/TraDiE-policies.git && git -C TraDiE-policies checkout 74493c38338efe37491c95d073334169d0521216'
```

If the login node has no internet, clone on the Mac and `rsync` it to `<W>/TraDiE-policies`.

**4.3 Build the Python environment** (login node, a few minutes). It installs from the Alliance
package store, which gave torch 2.9.1 (CUDA 12.6), torch_geometric 2.7.0, torch_cluster 1.6.3,
pytorch_lightning 2.6.5 and protobuf 7.36.1 on Killarney:

```bash
ssh <host> 'cd <W>/DONUT-NLL && bash -l slurm/setup_env.sh'
```

The last lines must read `torch 2.9.1 | cuda build 12.6` (or similar) and `Environment ready`.
Then check the code imports:

```bash
ssh <host> 'cd <W>/DONUT-NLL && bash -lc "module load StdEnv/2023 python/3.11 scipy-stack && source ~/envs/donut/bin/activate && python -c \"import train_donut_nll; print(\\\"imports ok\\\")\""'
```

**4.4 Install the released checkpoint.** `ckpts/donut-nll/` must contain **only** `.ckpt` files: the
sampler parses every file name there as `epoch=N-...`.

```bash
ssh <host> 'cd <W>/DONUT-NLL && mkdir -p ckpts/donut-nll ckpts/donut-nll-docs && \
  curl -sSfL https://omnomnom.vision.rwth-aachen.de/data/donut-nll.tar.gz | tar -xz -C ckpts/donut-nll && \
  mv ckpts/donut-nll/README.md ckpts/donut-nll/WAYMO_DATASET_LICENSE.md ckpts/donut-nll-docs/ && \
  ls -la ckpts/donut-nll'
```

Expected: exactly `epoch=29-step=228300.ckpt`, 36,102,302 bytes. Without internet, `rsync` the Mac's
`ckpts/donut-nll/` instead.

---

## 5. Get the validation data

The target layout is `<W>/DONUT-NLL/data/waymo/raw/validation/validation.tfrecord-000NN-of-00150`.

Pick one option:

| Option | How | Speed | Needs the user? |
|---|---|---|---|
| **A. Copy from the Mac** | `rsync -a ~/Desktop/Courses/CSC490/DONUT-NLL/data/waymo/raw/validation/ <host>:<W>/DONUT-NLL/data/waymo/raw/validation/` (after `mkdir -p` on the cluster) | about 1.5 h at the 8 MB/s seen for Killarney; only 0.4 MB/s was seen for Trillium while other uploads ran | Only to keep the SSH connection open |
| **B. Copy cluster to cluster** from Killarney (`/scratch/nuhgoyin/DONUT-NLL/data/waymo/raw/validation`) | Globus (Alliance endpoints), set up by the user | fast | Yes |
| **C. Download on the cluster** | Install the Google Cloud CLI in `$HOME` on the cluster, the user runs `gcloud auth login --no-launch-browser`, then `gsutil -m cp "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/validation/*" <dest>/` | fast | Yes (login plus Waymo license on their account) |

Never copy the user's Google credentials to a cluster. Option C means the user logs in there themselves.

**Check the copy is complete.** Compare every file's size with the bucket. The Mac is logged into
Google Cloud, so `gsutil` works there:

```bash
gsutil ls -l "gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/validation/*" \
  | awk '$3 ~ /^gs/ {n=$3; sub(".*/","",n); print n, $1}' | sort > /tmp/bucket_sizes.txt
ssh <host> 'cd <W>/DONUT-NLL/data/waymo/raw/validation && for f in *of-00150; do echo "$f $(stat -c %s "$f")"; done | sort' > /tmp/cluster_sizes.txt
diff /tmp/bucket_sizes.txt /tmp/cluster_sizes.txt && echo "all 150 files match"
```

(Mac-side temporary files can go in your scratchpad directory instead of `/tmp`.)

---

## 6. Job scripts

All scripts are submitted **from the repo root** on the cluster, write logs to `slurm_logs/`, and
activate `~/envs/donut`.

| Script | What it does | Tested runtime (Killarney) |
|---|---|---|
| `slurm/preprocess_val.sh` | Raw TFRecords → processed scenes (`preprocess/preprocess_waymo.py`), then the TraDiE ground-truth pickle (`prepare_waymo_gt.py`). Env: `RAW_DIR`, `PROCESSED_DIR`, `GT_PATH`. | Full val: 23 min + 19 min on 16 CPUs |
| `slurm/sample.sh [model]` | 3,000 samples per agent from the newest `ckpts/<model>/*.ckpt` → `$SAVE_DIR/<model>-val.pkl`. Env: `DATA_ROOT`, `SAVE_DIR`, `BATCH_SIZE`. | Full val: 1 h 22 min on 1 H100 |
| `slurm/eval_tradie.sh <samples> <gt> <pred_dir>` | Policies + scoring in one job. Fine for small sets only (1.3–1.7 s per scene). | 1 file (286 scenes): about 7 min |
| `slurm/policies_array.sh <samples> <gt> <work_dir>` | Policies as a 16-task Slurm array, one chunk each (`slurm/tradie_chunks.py split`). | Full val: about 80 min per task |
| `slurm/merge_eval.sh <work_dir>` | Merges chunks (refuses if any chunk is missing) and scores naive, `fde_adam`, `mAP_rectangles`. | Full val: about 37 min |
| `slurm/train.sh <train_donut_nll.py args>` | Single-node multi-GPU training (see §9). | — |
| `slurm/preprocess_train.sh` | Raw training TFRecords → processed scenes, as a 10-task array; `DELETE_RAW=1` deletes raw files as they're done (see §10). | Full training set: ~16 min per task on 16 CPUs |
| `slurm/stream_train.sh` | Login-node driver: download training shards in batches, preprocess, verify, delete raw (see §10). | — |
| `slurm/count_tfrecords.py <files>` | Exact scenario count in raw TFRecords, from length headers only. | Seconds per file |
| `slurm/make_subset.py manifest\|link` | Build a training subset: scenario-ID manifest from raw shards, then a symlinked data root (see §10). | 48,805 scenes: 13 s + ~1 min |
| `slurm/scan_scenes.py` | Agent and map-point counts per scene after the training transforms, to check memory risk (see §10). | 194,613 scenes: ~15 min on 48 CPUs |
| `slurm/bench_memory.py`, `slurm/profile_memory.py` (+ `.sh`) | Speed and peak-memory benchmark per batch size; per-scene memory profile. `--checkpoint_social` option. | EXPERIMENTS 0001–0003 |
| `slurm/split_val.py`, `slurm/tune_*.sh`, `slurm/sample_n.sh`, `slurm/tradie_tune.patch` | Policy tuning: split validation into tune/report halves, patched TraDiE with settings in environment variables, N-sample prediction. | EXPERIMENTS 0004 |

The policies write three prediction files, and the paper's columns map onto them like this:
- **Naive:** the model's own 6 modes (`naive`).
- **Window policy:** `mAP_rectangles`, which gives the paper's Soft mAP, mAP and Miss rate.
- **Distance policy:** `fde_adam`, which gives the paper's minFDE.

**Ignore `min_ade` (about 5,000–6,000) and `overlap_rate`.** The policy outputs only fill the
3 s, 5 s and 8 s positions and set the rest to zero, so ADE is meaningless. The paper doesn't report either.

---

## 7. Smoke test: 1 validation file, about 30 min

Run this first on every new cluster. It checks the environment, preprocessing, sampling and scoring
on file `00000` (286 scenes). The `smoke/` directory is separate, so the full-run data is untouched.

```bash
cd <W>/DONUT-NLL      # on the cluster, via bash -lc
rm -rf smoke && mkdir -p smoke/raw
ln -s $PWD/data/waymo/raw/validation/validation.tfrecord-00000-of-00150 smoke/raw/
SB_GPU="..."; SB_CPU="..."   # this cluster's prefixes from §2
mkdir -p slurm_logs
p=$($SB_CPU --parsable --cpus-per-task=4 --time=0-00:20 \
    --export=ALL,RAW_DIR=smoke/raw,PROCESSED_DIR=smoke/data/waymo/processed/val,GT_PATH=smoke/gt_val.pkl \
    slurm/preprocess_val.sh)
s=$($SB_GPU --parsable --dependency=afterok:$p --time=0-00:20 \
    --export=ALL,DATA_ROOT=smoke/data,SAVE_DIR=smoke slurm/sample.sh donut-nll)
e=$($SB_GPU --parsable --dependency=afterok:$s --time=0-00:30 \
    slurm/eval_tradie.sh smoke/donut-nll-val.pkl smoke/gt_val.pkl smoke/preds)
echo preprocess=$p sample=$s eval=$e
```

Expected output:
- **Preprocessing:** `processed val scenes: 286`, and the ground-truth step reports `Scenes: 286 Agents: 9842`.
- **Sampling:** `Saved 286 scenes / 1268 agents`.
- **Scores:** in `slurm_logs/donut-nll-eval-$e.out`. Three runs on Killarney gave:

| 286 scenes | Run | soft_map | map | miss_rate | min_fde |
|---|---|---|---|---|---|
| `naive` | A: Mac-preprocessed data, H100 | 0.4884 | 0.4588 | 0.1498 | 1.2326 |
| `naive` | B: these scripts, H100 | 0.4845 | 0.4551 | 0.1502 | 1.2336 |
| `naive` | C: these scripts, L40S | 0.4856 | 0.4562 | 0.1502 | 1.2326 |
| `mAP_rectangles` | A | 0.5717 | 0.5688 | 0.0801 | — |
| `mAP_rectangles` | B | 0.5684 | 0.5656 | 0.0841 | — |
| `mAP_rectangles` | C | 0.5870 | 0.5838 | 0.0661 | — |
| `fde_adam` | A / B / C | — | — | — | 0.9159 / 0.9083 / 0.9024 |

**Pass criteria.** Every value must fall in its range:

| File | soft_map | map | miss_rate | min_fde |
|---|---|---|---|---|
| `naive` | 0.478–0.495 | 0.448–0.465 | 0.145–0.155 | 1.225–1.245 |
| `mAP_rectangles` | 0.55–0.61 | 0.54–0.60 | 0.055–0.095 | — |
| `fde_adam` | — | — | — | 0.88–0.94 |

- **Wide ranges are normal here.** On 286 scenes, tiny numeric differences move the scores. These
  come from preprocessing rounding (about 1e-7 rad) and from GPU operations that aren't bit-for-bit
  repeatable. The naive scores still land within 0.004 of each other.
- **The policies vary more.** They optimize over random samples, so a couple of dozen agents flip
  between hit and miss from run to run.
- **What a failure looks like:** a real setup problem (wrong checkpoint, data version or code) shows
  up as naive scores far outside these ranges, or as a crash.

These 286-scene numbers are **not** comparable with the paper: mAP on one file is about 0.1 higher
than on the full set. Only the full run in §8 compares with the paper.

---

## 8. Full validation: 44,097 scenes, about 4 h

Needs all 150 raw files (§5). Submit as a chain, where each step starts only if the previous one succeeds:

```bash
cd <W>/DONUT-NLL      # on the cluster, via bash -lc
SB_GPU="..."; SB_CPU="..."   # this cluster's prefixes from §2
mkdir -p slurm_logs
p=$($SB_CPU --parsable slurm/preprocess_val.sh)
s=$($SB_GPU --parsable --dependency=afterok:$p slurm/sample.sh donut-nll)
a=$($SB_GPU --parsable --dependency=afterok:$s slurm/policies_array.sh samples/donut-nll-val.pkl samples/gt_val.pkl samples/full_val)
m=$($SB_CPU --parsable --dependency=afterok:$a slurm/merge_eval.sh samples/full_val)
echo preprocess=$p sample=$s policies=$a merge_eval=$m
```

Notes:
- **Sampling and ground truth are independent.** If the processed data already exists, skip
  preprocessing and submit sampling directly. `preprocess_waymo.py` skips scenes that are already
  processed, but it still reads every raw file.
- **Check the outputs as you go:**
  - Preprocessing prints `processed val scenes: 44097`.
  - `policies_array.sh` task 0 prints `samples=44097 gt=44097 shared=44097`.
  - `merge_eval.sh` prints `<policy>: 16 chunks, 44097 scenes` for `fde_adam`, `mAP_rectangles`, `naive` and `gt`.
- **Scores** are in `slurm_logs/donut-nll-merge-eval-$m.out`:

| Waymo val, 44,097 scenes | Soft mAP | mAP | Miss rate | minFDE |
|---|---|---|---|---|
| Paper, Table 2 (Gen. Gaussian, Step-NLL): naive | 0.3759 | 0.3439 | 0.1561 | 1.3127 |
| Killarney: naive | 0.3758 | 0.3439 | 0.1561 | 1.3131 |
| Paper, Table 2: with policies | 0.5101 | 0.5070 | 0.0876 | 1.0103 |
| Killarney: `mAP_rectangles` (Soft mAP, mAP, Miss rate), `fde_adam` (minFDE) | 0.5090 | 0.5060 | 0.0888 | 1.0087 |

**Pass criteria:** naive within **±0.002** of the paper, policies within **±0.005**. With 44k
scenes, run-to-run noise mostly averages out. The Killarney run was within 0.001 of the paper on
every value, though that's a single run.

---

## 8b. Running experiments alongside other agents

- **Code:** each experiment gets its own copy of the repo, for example `<W>/exp-<name>/` (rsync as
  in §4.1). Symlink the shared data instead of copying it:
  `ln -s <W>/DONUT-NLL/data <W>/exp-<name>/data`. Treat `data/` as read-only.
- **Run names:** use a unique `--name` for training (checkpoints go to `ckpts/<name>/`, logs to
  `csv_logs/<name>/`). `train_donut_nll.py` **resumes automatically** from the newest file in
  `ckpts/<name>/`, so a reused name continues someone else's run.
- **Evaluating a new checkpoint:** use `sample.sh <name>` with a unique `SAVE_DIR` (passed with `--export=ALL,SAVE_DIR=...`), then
  `policies_array.sh <SAVE_DIR>/<name>-val.pkl samples/gt_val.pkl <unique work_dir>` and
  `merge_eval.sh <work_dir>`. The ground truth `samples/gt_val.pkl` is shared and read-only.
- **Policy-only experiments** (changing `TraDiE-policies`) don't need re-sampling: reuse
  `samples/donut-nll-val.pkl`. Work on your own copy of `TraDiE-policies` and pass `--export=ALL,TRADIE_DIR=<your copy>`.
  If you tune on validation, report on scenes you didn't tune on.
- **Training:** see §9 below for how to train, what it costs, and the setup to use for comparisons.
- **Quotas:** Killarney scratch is 1000 GiB per user, and a full validation run uses about 121 GB.
  Check `diskusage_report` before creating large outputs. Scratch isn't backed up, so copy results
  worth keeping to the Mac.

---

## 9. Training

**Always train with `--checkpoint_social`** (EXPERIMENTS 0003). It gives bit-identical training,
cuts crowded-scene memory about 6×, and makes batch size 8 per GPU fit on H100/H200.

- **Launch:** `slurm/train.sh` runs `train_donut_nll.py` on one node with one `srun` task per GPU;
  `--ntasks-per-node` must equal the GPU count, and the script checks this. It also:
  - sets data-loader workers per GPU process to CPUs/tasks − 2, capped at 16;
  - sets `PYTORCH_ALLOC_CONF=expandable_segments:True`;
  - sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, so our own checkpoints load when resuming;
  - raises the open-files limit;
  - puts torch's kernel cache on scratch.

  Default time limit 24 h.
- **Batch rule:** `acc_batch_size` (64) must be divisible by batch size × GPUs; Lightning
  accumulates the rest. The effective batch should stay 64, or runs aren't comparable.
- **Resuming:** resubmitting the identical command resumes from the newest checkpoint in
  `ckpts/<name>/`. That's also how to continue past a 24 h limit. `--name` must be unique (see
  `EXPERIMENTS.md` conventions).
- **Checkpoints:** one per epoch is kept (`save_top_k=10`), about 108 MB each.

**Tested launch prefixes and speeds** (batch 8 per GPU, `--checkpoint_social`, fp32):

| Where | Prefix before `slurm/train.sh` | Scenes/s | Tested in |
|---|---|---|---|
| TamIA 8×H200 | `sbatch --account=aip-ajbonner --gpus-per-node=h200:8 --ntasks-per-node=8 --cpus-per-task=8 --mem=0` | ~56 | 0008 |
| TamIA 4×H100 | `sbatch --account=aip-ajbonner --gpus-per-node=h100:4 --mem=0 --cpus-per-task=12` | ~31 | 0006, 0007 |
| Killarney 4 of 8 H100 | `sbatch --account=aip-ajbonner --gpus-per-node=h100:4 --cpus-per-task=6 --mem=256G --exclude=kn117` | ~27–30 | 0007 |
| Trillium 4×H100 | `sbatch --account=def-ajbonner --gpus-per-node=4` | ~31 | 0005 |
| Vulcan L40S | batch 4 per GPU fits; not used for real runs | ~3 per GPU | 0003 (benchmark only) |

The 8×H200 nodes were idle and started immediately on 2026-09-27; Killarney's 8×H100 nodes were
all busy.

**Comparison setup** (EXPERIMENTS 0008). Use this for comparing training changes; the 10% setup of
0005–0007 is too noisy:
```
<TamIA 8×H200 prefix> slurm/train.sh --name <ID>-<slug>-r1 --data_root data --batch_size 8 \
    --checkpoint_social --max_epochs 10 --decay_epochs 10 --limit_val_batches 0.1 [your change]
```
- TamIA's `data` is exactly 40% of the training set (shards 0–399, 194,613 scenes).
- Each run takes ~10 h and ~80 GPU-hours.
- Run at least 2 runs per configuration: two identical runs differ by up to ~0.012 window mAP and
  ~0.04 minFDE at this setup.
- Training isn't bit-reproducible run to run, because GPU summation order varies.

**Full training** (paper setting: all 486,995 scenes, 30 epochs) at 56 scenes/s is about 72 h. On
TamIA that's 3–4 chained 24 h jobs, resubmitting the same command. Not yet run.

**Evaluating a trained checkpoint** (all evaluation so far ran on Vulcan, for consistency):
1. Copy the newest checkpoint (`ls ckpts/<name> | sort -V | tail -1`) through the Mac into Vulcan's
   `ckpts/<name>/`. Check the checksums match.
2. On Vulcan, submit the chain. `$SB` is the Vulcan prefix; `--exclude=rack07-15` avoids the node
   with the filesystem error.
   ```
   s=$($SB --parsable --time=0-04:00 --export=ALL,SAVE_DIR=samples/eval slurm/sample.sh <name>)
   a=$($SB --parsable --dependency=afterok:$s slurm/policies_array.sh samples/eval/<name>-val.pkl samples/gt_val.pkl samples/eval/<name>-default)
   $SB --dependency=afterok:$a slurm/merge_eval.sh samples/eval/<name>-default
   a2=$($SB --parsable --array=0-3 --dependency=afterok:$s \
        --export=ALL,TRADIE_DIR=/scratch/nuhgoyin/TraDiE-policies-tune,TUNE_PI=mean,TUNE_POLICIES=mAP_rectangles \
        slurm/policies_array.sh samples/eval/<name>-val.pkl samples/gt_val.pkl samples/eval/<name>-pimean)
   $SB --dependency=afterok:$a2 slurm/merge_eval.sh samples/eval/<name>-pimean
   ```
3. Total ~4 h per checkpoint and ~16 L40S GPU-hours; eight checkpoints run in parallel fine.
   Each prediction file is ~11 GB.
4. For models with more than 6 modes, `sample_donut_nll.py` keeps the 6 most probable modes for the
   naive score.

---

## 10. Training data

**Sizes** (Waymo Open Motion v1.3.0, `scenario` format):

| Split | Raw | Files | Scenes | Processed |
|---|---|---|---|---|
| Training | 455.4 GB | 1,000 | 486,995 | ~630 GB |
| Validation | 41.2 GB | 150 | 44,097 | 57 GB |
| Test | 31.9 GB | 150 | ~45k | (not used) |

**Downloading on a cluster.** The Google Cloud CLI is installed in `~/google-cloud-sdk` on all four
clusters.
- **Signing in:** only the user can sign in (`~/google-cloud-sdk/bin/gcloud auth login
  --no-launch-browser` in an interactive SSH session). Never copy their credentials.
- **Where to download:**
  - login nodes on Vulcan, TamIA and Killarney (no transfer nodes are documented there);
  - **`trillium-dm`** on Trillium.
- **Speeds seen:** Vulcan ~660 MB/s, Trillium's transfer node ~750 MB/s, TamIA ~12–14 MB/s.
- **Detaching:** run downloads fully detached, or a dropped SSH connection kills them:
  `setsid nohup <cmd> > log 2>&1 < /dev/null & disown`.
- **Checking:** always compare every file's size against `gcloud storage ls -l` on the bucket.

**Preprocessing, with room for raw + processed** (Trillium, Vulcan):
`gcloud storage rsync gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training/
data/waymo/raw/training/`, then `slurm/preprocess_train.sh`. Then check that
`ls data/waymo/processed/train | wc -l` equals the TOTAL from
`python slurm/count_tfrecords.py data/waymo/raw/training/*tfrecord*`.

**Preprocessing, without room for both** (Killarney, TamIA): run `slurm/stream_train.sh` from the
repo root, detached. It downloads 100 shards at a time, checks sizes, counts scenarios, preprocesses
with `DELETE_RAW=1`, checks the count grew by exactly that much, and repeats. It stops at the first
problem, and rerunning skips batches already marked done. Settings: `SB_CPU` (required), `FIRST`,
`LAST`, `BATCH`, `SKIP`, `LIMIT_GIB`; see its header.

**Subsets:**
1. Build a manifest from raw shards: `python slurm/make_subset.py manifest --raw_dir
   data/waymo/raw/training --first 0 --last 99 --out data_manifests/train_shards000-099.txt`.
2. Make a symlinked data root: `python slurm/make_subset.py link --manifest <manifest> --src_root
   data --dst_root data_sub010`.
3. Train with `--data_root data_sub010`.

`data_manifests/train_shards000-099.txt` (48,805 scenes) exists. A 0–399 manifest was never built
(the Trillium job was cancelled), but TamIA's `data` is exactly shards 0–399.
**Symlinks count against file quotas.** TamIA allows 500K files, so a 195K-link subset there would
nearly fill it.

**Before training on new data or a larger batch size,** run `slurm/scan_scenes.py` as a CPU job. It
reports agent counts after the training transforms. Memory grows roughly with agents², and at
batch size 1 with `--checkpoint_social` a 228-agent scene takes ~12 GiB and a 286-agent one about
19 GiB. Results for shards 0–399: max 286 agents, 2 scenes above 228, p99.9 = 106.

---

## 11. Current state (as of 2026-09-28)

**Data on each cluster** (repo at `/scratch/nuhgoyin/DONUT-NLL`, or `/scratch/n/nuhgoyin/DONUT-NLL` on
TamIA):

| Cluster | Training data | Validation | Other | Scratch used / quota |
|---|---|---|---|---|
| Killarney | processed, full (486,995); raw deleted; `data_sub010` | raw + processed + `samples/gt_val.pkl` | released-model samples from the 2026-09-23 run | ~752 GiB / 1000 (2026-09-25) |
| Trillium | raw + processed, full; `data_sub010` | raw + processed + GT | — | ~1.2 TB / 25 TiB |
| TamIA | processed shards 0–399 only (194,613) as `data`; raw deleted; `data_sub010` | raw + processed + GT | `bench/scan-train040.tsv` | ~349 GiB / 500 GiB, and near the 500K-file quota |
| Vulcan | raw + processed, full | raw + processed + GT | `samples/eval/` (481 GB; all evaluation outputs for 0005–0008), `samples/tune/` (207 GB; 0004) | 1802 GiB / 5120 |

**Checkpoints:**
- The released `donut-nll` is on every cluster.
- The final checkpoints of 0005–0008 are on the Mac (`ckpts/`) and on Vulcan.
- All per-epoch checkpoints are on the cluster that trained them.

**Cleanup candidates** (ask the user first):
- Vulcan `samples/eval/*-val.pkl` and `samples/tune/`: ~690 GB, needed only to re-score.
- Killarney `ckpt_social/` (1.3 GB frozen copy from 0003) and the smoke-test directories.
- Per-epoch checkpoints other than the last.
- The Mac's `data/waymo/raw/validation` (40 GB), now copied to all clusters.

**Open items:**
- **Revoke the Google login** on all four clusters until the next download. The user runs
  `~/google-cloud-sdk/bin/gcloud auth revoke --all` on each.
- Killarney node `kn117` has a faulty GPU; exclude it and consider reporting it to support.
- Trillium cancelled jobs twice on 2026-09-27 (§12). Check it's healthy before using it.

---

## 12. Troubleshooting

All of these were hit on Killarney, except the two marked *(prevented)*, which the scripts already guard against.

| Symptom | Cause | Fix |
|---|---|---|
| Jobs die after ~2 min as `CANCELLED by 0` or `NODE_FAIL`, on several nodes at once, with no error in the log (Trillium, 2026-09-27) | System-side problem on the cluster | Resubmit once. If it repeats, run elsewhere and consider emailing the cluster's support address. |
| A driver or watcher reports tasks "not COMPLETED" right after the queue empties, though the output is complete | Slurm accounting lags the queue by a minute or so | Wait until `sacct` shows final states before judging; `stream_train.sh` now does this. |
| `ls`/`find` of 100K+ files, or any SSH command, takes minutes on TamIA | TamIA's login node and filesystem are slow | Be patient. Don't count on quick polling. |
| A Mac-side command like `rsync $EXCLUDES ...` fails with `unrecognized option` | The Mac's zsh doesn't split a variable into separate arguments | Use `--exclude-from=<file>`, or run it under `bash -c`. |
| Several failed logins after a dropped connection | A command fell through to a fresh login | Always `ssh -O check <alias>` before remote work. Every fresh attempt is a failed Duo login. |
| `module: command not found` inside a batch job | The job didn't get a login environment (Vulcan starts jobs in a bare shell unless submitted with `--export=ALL`) | Every `slurm/*.sh` job script now starts with `#!/bin/bash -l`. Resync `slurm/` from the Mac if a cluster copy still has `#!/bin/bash`. |
| `Weights only load failed ... GLOBAL pathlib.PosixPath` when loading **our own** trained checkpoints (resume or sampling) | torch ≥ 2.6 defaults to `weights_only=True`, and our checkpoints store `data_root` as a `Path`. The released checkpoint doesn't have this. | `slurm/train.sh` and `slurm/sample.sh` set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Only use it for checkpoints we made. |
| `OSError: [Errno 5] Input/output error` writing a large output file (seen on Vulcan `rack07-15`) | Transient filesystem or node fault | Delete the possibly truncated output, and rerun the job with `--exclude=<node>`. |
| Every SSH connection drops at once | VPN or network change on the Mac. Reconnecting the network doesn't restore them. | The user reruns `ssh -fN <alias>` for each cluster needed. |
| `SBATCH ERROR: The --mem... options are not allowed on Trillium` | Trillium fixes memory per GPU or node | Use Trillium's prefix from §2, which has no `--mem`. |
| A smoke or experiment job writes into `data/waymo/processed/val` or `samples/` instead of your paths | The variables didn't reach the job (Trillium drops `NAME=value sbatch ...`) | Pass them with `--export=ALL,NAME=value`. Delete anything the job wrote to shared paths. |
| `There is no l40s GPU-type`, or `The h100 GPUs are only allocated by node` | Wrong GPU request for this cluster | Use the cluster's prefix from §2. |
| Job fails at once and writes no log | `slurm_logs/` is missing | `mkdir -p slurm_logs` in the repo root. |
| `Permission denied (keyboard-interactive)` / `Control socket connect(...): No such file` | The shared SSH connection closed (Mac slept or timed out) | Ask the user to run `ssh -fN <host>`. Jobs keep running meanwhile. |
| `module: command not found` | Non-login shell, or `sbatch --wrap` (runs `/bin/sh`) | Use `bash -lc "..."`. For small jobs, write a script with `#!/bin/bash -l` instead of `--wrap`. |
| `squeue: command not found` over SSH | Non-login shell | Same: `bash -lc`. |
| `Submitting jobs from directories residing in /home is not permitted` | Killarney policy | Work in `/scratch/$USER`. The environment in `~/envs` is fine. |
| `ModuleNotFoundError: No module named 'numpy'` in the ground-truth step | `PYTHONPATH` was replaced instead of extended (numpy comes from `scipy-stack` via `PYTHONPATH`) | Append to it: `PYTHONPATH=...:${PYTHONPATH:-}`. `preprocess_val.sh` already does. |
| `FileNotFoundError: ../waymo/raw/validation` | `preprocess_waymo.py --data_root` must be the `data/` folder | `preprocess_val.sh` passes `--raw_dir`/`--processed_dir` explicitly. |
| `Can't instantiate abstract class WaymoTransform with abstract method forward` | torch_geometric ≥ 2.7 | Already fixed in `transforms/waymo_transform.py` (`__call__` → `forward`). Make sure you copied the Mac's version of the code. |
| *(prevented)* `ValueError: invalid literal for int()` when loading a checkpoint | A non-checkpoint file in `ckpts/<model>/` | Move it out (see §4.4). |
| `torch.OutOfMemoryError` in training | 44–48 GB GPU | Use H100 (80 GB). Batch size 2 on L40S still ran out at step 425 of 495. |
| `CUDA error: uncorrectable ECC error encountered` | A faulty GPU node. Killarney's `kn117` failed this way repeatedly on 2026-09-23/24 and was still being scheduled. | Rerun only the failed tasks with `--exclude=<node>` (command below). On Killarney, pass `--exclude=kn117` to every job until support fixes it (report it to support@tech.alliancecan.ca). |
| *(prevented)* `ValueError: Scenarios in gt and samples differ!` | The ground truth and samples cover different scenes | Use `policies_array.sh`, which keeps only the shared scenes and prints the counts. |

Rerunning failed array tasks:

```bash
r=$($SB_GPU --parsable --array=4,9 --exclude=<badnode> --export=ALL,CHUNKS=16 slurm/policies_array.sh <samples> <gt> <work_dir>)
$SB_CPU --dependency=afterany:<original_array_id>,afterok:$r slurm/merge_eval.sh <work_dir>
```

Useful status commands (on the cluster, via `bash -lc`):
- `sacct -j <ids> -X --format=JobID%16,JobName%22,State,Elapsed,NodeList`
- `squeue -u $USER --start` (estimated start times)
- `tr '\r' '\n' < slurm_logs/<file>.out | tail`: progress bars use carriage returns.
