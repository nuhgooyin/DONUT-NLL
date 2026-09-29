# Towards Metric-Agnostic Trajectory Forecasting

**ECCV 2026**

[arXiv](https://arxiv.org/abs/2607.01133) | [Project Page](https://vision.rwth-aachen.de/TraDiE-policies) | [YouTube](https://www.youtube.com/watch?v=VyzY2MS2Iak) | [BibTeX](#Citation)

[Markus Knoche](https://scholar.google.com/citations?user=Kx4v8IMAAAAJ)<sup>1</sup>, [Daan de Geus](https://daandegeus.com/)<sup>2</sup>, [Bastian Leibe](https://scholar.google.com/citations?hl=de&user=ZcULDB0AAAAJ)<sup>1</sup>

<sup>1</sup> RWTH Aachen University
<sup>2</sup> Eindhoven University of Technology

> [!NOTE] 
> This repository contains code for our model **DONUT-NLL**. To apply and evaluate model samples using our **TraDiE policies**, check out [this repository](https://github.com/MKnoche/TraDiE-policies).

## Installation

Clone repository:

```bash
git clone https://github.com/MKnoche/DONUT-NLL.git
cd DONUT-NLL
```

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/).

## Data Preprocessing

Download the raw Waymo data into `{args.data_root}/waymo/raw/<split>`, where `<split>` is `training`, `validation`, or `testing`.

Next, preprocess the data. This will take some time.

```bash
cd preprocessing
uv run preprocess_waymo.py --data_root <data_root>
```

## Training

Adjust the root paths in `train_donut_nll.py` according to your setup.

Distributed training is supported via the `devices` and `nodes` parameters. Gradient accumulation makes sure that the effective batch size is always `acc_batch_size`, as long as `batch_size * devices * nodes <= acc_batch_size`.

Training for 30 epochs on 6 NVIDIA H100 GPUs with a batch size of 8 per GPU takes about 2.7 days.

```bash
uv run train_donut_nll.py
```

## Evaluation

First, generate and store samples from the model. The newest checkpoint is automatically loaded from `{args.ckpt_root}/{args.model_name}/`.

```bash
uv run sample_donut_nll.py <model_name>
```

Next, follow the steps in the [TraDiE-policy repo](https://github.com/MKnoche/TraDiE-policies).

## Model Checkpoint

> [!IMPORTANT]
> This checkpoint is a **Distributed WOD Model** made using the
> [Waymo Open Dataset](https://www.waymo.com/open), provided by Waymo LLC
> under the
> [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms).
> By downloading or using this checkpoint, you agree to the terms of that
> Agreement. Any downstream use or modification is subject to its terms,
> including the **non-commercial restrictions in Section 4** (no use in
> vehicles, production systems, or for any primarily commercial purpose).
> A full copy of the Agreement is included in the archive.

[DONUT-NLL (Generalized Gaussian + Step-NLL)](https://omnomnom.vision.rwth-aachen.de/data/donut-nll.tar.gz)

Download and extract the checkpoint in `{args.ckpt_root}/donut-nll/epoch=29-step=228300.ckpt`.

## Citation

If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{knoche2026tradie,
  title     = {{Towards Metric-Agnostic Trajectory Forecasting}},
  author    = {Knoche, Markus and de Geus, Daan and Leibe, Bastian},
  booktitle = {ECCV},
  year      = {2026}
}
```

## Acknowledgements

This project builds upon code from [DONUT](https://github.com/MKnoche/DONUT) and [QCNet](https://github.com/ZikangZhou/QCNet) (Apache-2.0 License).
