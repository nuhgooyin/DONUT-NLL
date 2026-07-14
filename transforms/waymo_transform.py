# Copyright (c) 2023, Zikang Zhou. All rights reserved.
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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from random import randrange, randint
from itertools import compress
import numpy as np
from scipy.interpolate import PchipInterpolator

deltas = torch.tensor(np.array([-2 * np.pi, 0, 2 * np.pi])).float()


def continuous_heading(head, valid):
    head, valid = head.clone(), valid.clone()
    prev = None
    for i, (h, v) in enumerate(zip(head, valid)):
        if not v:
            continue
        if prev is None:
            prev = h
        delta_i = (h + deltas - prev).abs().argmin()
        head[i:] += deltas[delta_i]
        prev = head[i]
    return head


def inpaint_agent(agent, valid):
    agent, valid = agent.clone(), valid.clone()
    if valid.sum() < 2:
        agent[:2] = agent[valid]
        valid[:2] = True
    x = torch.arange(len(agent)).numpy()
    y = agent.numpy()

    interpolator = PchipInterpolator(
        x[valid],
        y[valid],
        axis=0,
        extrapolate=True,
    )

    return torch.from_numpy(interpolator(x))


def inpaint_agents(pos, head, valid):
    pos, head, valid = pos.clone(), head.clone(), valid.clone()
    for i, (p, h, v) in enumerate(zip(pos, head, valid)):
        h = continuous_heading(h, v)
        agent = torch.cat([p, h[..., None]], dim=-1)
        res = inpaint_agent(agent, v)
        pos[i, ..., :2] = res[..., :2]
        head[i] = res[..., -1]
        valid[i] = torch.ones_like(valid[i])

    return pos, head, valid


def inpaint_waymo(data):

    valid = data['agent']['valid_mask']
    pos = data['agent']['position']
    head = data['agent']['heading']
    valid = data['agent']['valid_mask']
    pos, head, valid = inpaint_agents(pos, head, valid)
    data['agent']['position'][:, :11] = pos[:, :11]
    data['agent']['heading'][:, :11] = head[:, :11]
    data['agent']['valid_mask'][:, :11] = valid[:, :11]

    return data


def remove_partial(data):
    full_hist = data['agent']['valid_mask'][:, :11].all(dim=1)
    for key in [
        'position',
        'heading',
        'velocity',
        'dimensions',
        'valid_mask',
        'predict_mask',
        'interesting_mask',
        'type',
        'category',
    ]:
        data['agent'][key] = data['agent'][key][full_hist]
    if 'batch' in data['agent']:
        data['agent']['batch'] = data['agent']['batch'][full_hist]
    start = 0
    if 'batch' in data['agent']:
        for i, agent_ids in enumerate(data['agent']['id']):
            end = start + len(agent_ids)
            data['agent']['id'][i] = list(compress(agent_ids, full_hist[start:end]))
            start = end
    else:
        data['agent']['id'] = list(compress(data['agent']['id'], full_hist))

    data['agent']['num_nodes'] = len(data['agent']['position'])

    return data


class WaymoTransform(BaseTransform):
    def __init__(
        self, dataset: str, inpaint_partial: bool = False, remove_partial: bool = False
    ) -> None:
        if inpaint_partial and remove_partial:
            raise ValueError('Either inpaint_partial or remove_partial must be False.')
        self.dataset = dataset
        self.inpaint_partial = inpaint_partial
        self.remove_partial = remove_partial

    def __call__(self, data: HeteroData) -> HeteroData:
        if self.dataset != 'waymo':
            return data

        to_pred = data['agent']['category'] > 0
        full_valid = data['agent']['valid_mask'][:, 11:].all(dim=1)
        data['agent']['category'][to_pred & full_valid] = 3
        data['agent']['category'][to_pred & ~full_valid] = 3  # 1

        if self.inpaint_partial:
            data = inpaint_waymo(data)

        if self.remove_partial:
            data = remove_partial(data)

        return data
