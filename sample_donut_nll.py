# Copyright (c) 2026, Markus Knoche. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import io
import pickle
import zlib
from itertools import compress
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets import ArgoverseV2Dataset, WaymoDataset
from datamodules import DEFAULT_WAYMO_TRANSFORM
from predictors import DonutNLL

torch.set_grad_enabled(False)


def pack_npy(a: np.ndarray, *, compress=True) -> bytes:
    a = np.asarray(a, dtype=np.float32, order='C')
    bio = io.BytesIO()
    np.save(bio, a, allow_pickle=False)
    raw = bio.getvalue()
    return zlib.compress(raw, level=3) if compress else raw


def seed_all(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_str(x) -> str:
    if isinstance(x, bytes):
        try:
            return x.decode('utf-8')
        except Exception:
            return repr(x)
    if hasattr(x, 'item'):
        try:
            return str(x.item())
        except Exception:
            pass
    return str(x)


def _normalize_step(step: int, horizon: int) -> int:
    step = int(step)
    if step == -1:
        return horizon - 1
    if step < 0 or step >= horizon:
        raise IndexError(f'Requested step={step} out of bounds for T={horizon}')
    return step


def _normalize_heading_samples(head: torch.Tensor) -> torch.Tensor:
    if head.ndim == 3 and head.shape[-1] == 1:
        return head[..., 0]
    if head.ndim != 2:
        raise ValueError(
            f'Unexpected heading sample shape {tuple(head.shape)}, expected (N,S) or (N,S,1)'
        )
    return head


def get_newest_checkpoint_path(root, model_name):
    path = root / model_name
    path = list(
        sorted(
            path.glob('*'), key=lambda x: int(x.stem.split('epoch=')[-1].split('-')[0])
        )
    )[-1]
    return path


def load_model(root, model_name, epoch=None):
    print('Loading', model_name)
    if epoch is None:
        ckpt_path = get_newest_checkpoint_path(root, model_name)
    else:
        ckpt_path = root / model_name
        ckpt_path = next(ckpt_path.glob(f'epoch={epoch}*'))
    print('Loading', ckpt_path)
    model = DonutNLL.load_from_checkpoint(ckpt_path)
    model = model.eval()
    return model


def get_dataloader(root, dataset, split='val', batch_size=1, num_workers=5):
    if dataset == 'av2':
        dataset = ArgoverseV2Dataset(
            root=root / dataset,
            split=split,
        )
    elif dataset == 'waymo':
        dataset = WaymoDataset(
            root=root / dataset,
            split=split,
            transform=DEFAULT_WAYMO_TRANSFORM,
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader


def build_sample_pickle(
    model_name: str,
    *,
    ckpt_root: Path,
    data_root: Path,
    save_dir: Path,
    epoch: Optional[int] = None,
    split: str = 'val',
    num_samples: int = 3000,
    batch_size: int = 8,
) -> Path:
    seed_all()
    model = load_model(ckpt_root, model_name, epoch=epoch).cuda()
    dataset = model.map_encoder.dataset

    samples_steps = (29, 49, 79) if dataset == 'waymo' else (59,)

    epoch_str = f'-epoch_{epoch}' if epoch else ''
    output_path = save_dir / f'{model_name}{epoch_str}-{split}.pkl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader(data_root, dataset, batch_size=batch_size, split=split)

    total_scenes = 0
    total_agents = 0

    samples_db = {}

    for _, batch in enumerate(
        tqdm(loader, desc='Sampling scenes', smoothing=50 / len(loader))
    ):
        eval_mask = batch['agent']['category'] >= 3
        if not torch.any(eval_mask):
            print(eval_mask)
            continue

        pred = model(batch.cuda())[-1][0]
        pred = pred[eval_mask]

        pi = pred.pi.detach().cpu()
        if pi.ndim == 3:
            pi = pi[:, :, -1]
        pos_mean = pred.position_mean.detach().cpu()

        flat_ids = [x for xs in batch['agent']['id'] for x in xs]
        agent_ids = [_safe_str(a) for a in compress(flat_ids, eval_mask)]
        batch_ids = batch['agent']['batch'][eval_mask].cpu()

        for batch_id, scenario_id in enumerate(batch.scenario_id):
            scene_mask = batch_ids == batch_id
            if not torch.any(scene_mask):
                continue

            scene_key = _safe_str(scenario_id)
            scene_pred = pred[scene_mask]
            scene_pi = pi[scene_mask]
            scene_pos_mean = pos_mean[scene_mask]
            scene_agent_ids = list(compress(agent_ids, scene_mask))

            scene_samples = {}
            horizon = scene_pos_mean.shape[2]
            for agent_idx, agent_id in enumerate(scene_agent_ids):
                sample_pos_steps = []
                sample_head_steps = []
                naive_pos_steps = []

                for step in samples_steps:
                    step_idx = _normalize_step(step, horizon)
                    step_pos, step_head = scene_pred.sample_t(step_idx, num_samples)
                    step_pos = step_pos.detach().cpu()
                    step_head = _normalize_heading_samples(step_head.detach().cpu())

                    sample_pos_steps.append(step_pos[agent_idx].numpy())
                    sample_head_steps.append(step_head[agent_idx].numpy())
                    naive_pos_steps.append(
                        scene_pos_mean[agent_idx, :, step_idx, :].numpy()
                    )

                scene_samples[agent_id] = {
                    'samples_pos': pack_npy(np.stack(sample_pos_steps, axis=1)),
                    'samples_head': pack_npy(np.stack(sample_head_steps, axis=1)),
                    'naive_pos': pack_npy(np.stack(naive_pos_steps, axis=1)),
                    'naive_pi': pack_npy(scene_pi[agent_idx].numpy()),
                }

            samples_db[scene_key] = scene_samples
            total_scenes += 1
            total_agents += len(scene_agent_ids)

    with open(output_path, 'wb') as f:
        pickle.dump(samples_db, f)

    print(f'Saved {total_scenes} scenes / {total_agents} agents to {output_path}')
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate a sample pickle from a DONUT checkpoint.'
    )
    parser.add_argument('model_name', type=str)
    parser.add_argument('--ckpt_root', type=str, default='ckpts')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--save_dir', type=str, default='samples')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    build_sample_pickle(
        args.model_name,
        ckpt_root=Path(args.ckpt_root),
        data_root=Path(args.data_root),
        save_dir=Path(args.save_dir),
        epoch=args.epoch,
        split=args.split,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
