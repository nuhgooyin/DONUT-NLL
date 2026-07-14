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
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform


class SubsampleMapPoints(BaseTransform):
    def __init__(self, every_nth: int = 1) -> None:
        self.every_nth = every_nth

    def forward(self, data: HeteroData) -> HeteroData:
        if self.every_nth <= 1:
            return data

        mask = torch.zeros(data['map_point']['num_nodes'], dtype=bool)
        fr = 0
        fr_id = 0
        side_is = [[], [], []]
        i = 0
        for i, pl_id in enumerate(data['map_point', 'to', 'map_polygon'].edge_index[1]):
            if pl_id != fr_id:
                if len(side_is[2]) > 2 * self.every_nth:
                    for side_i in side_is:
                        mask[side_i[:: self.every_nth]] = 1
                else:
                    mask[fr:i] = 1
                fr = i
                fr_id = pl_id
                side_is = [[], [], []]
            side_is[data['map_point']['side'][i]].append(i)
        i += 1
        if len(side_is[2]) > 2 * self.every_nth:
            for side_i in side_is:
                mask[side_i[:: self.every_nth]] = 1
        else:
            mask[fr:i] = 1

        for key in ['position', 'orientation', 'magnitude', 'height', 'type', 'side']:
            data['map_point'][key] = data['map_point'][key][mask]
        pt_count = len(data['map_point']['position'])
        data['map_point']['num_nodes'] = pt_count
        data['map_point', 'to', 'map_polygon'].edge_index = data[
            'map_point', 'to', 'map_polygon'
        ].edge_index[:, mask]
        data['map_point', 'to', 'map_polygon'].edge_index[0] = torch.arange(pt_count)
        return data


def remove_polygons(data, drop_pls):
    drop_pts = drop_pls[data['map_point', 'to', 'map_polygon'].edge_index[1]]

    for key in ['position', 'orientation', 'magnitude', 'height', 'type', 'side']:
        data['map_point'][key] = data['map_point'][key][~drop_pts]
    data['map_point']['num_nodes'] = len(data['map_point']['position'])
    for key in ['position', 'orientation', 'height', 'type', 'is_interpolating']:
        data['map_polygon'][key] = data['map_polygon'][key][~drop_pls]
    data['map_polygon']['num_nodes'] = len(data['map_polygon']['position'])

    data['map_point', 'to', 'map_polygon'].edge_index = data[
        'map_point', 'to', 'map_polygon'
    ].edge_index[:, ~drop_pts]
    data['map_point', 'to', 'map_polygon'].edge_index[0] = torch.arange(
        len(data['map_point', 'to', 'map_polygon'].edge_index[0])
    )
    pl_ids = data['map_point', 'to', 'map_polygon'].edge_index[1]
    fr = pl_ids.unique()
    pl_ids = torch.cat([torch.zeros(1, dtype=int), (pl_ids[1:] - pl_ids[:-1] > 0)])
    pl_ids = torch.cumsum(pl_ids, dim=0)
    to = pl_ids.unique()
    data['map_point', 'to', 'map_polygon'].edge_index[1] = pl_ids

    translate = {f.item(): t.item() for f, t in zip(fr, to)}
    pl2pl = [[], []]
    pl2pl_type = []
    for fr, to, typ in zip(
        *data['map_polygon', 'to', 'map_polygon'].edge_index,
        data['map_polygon', 'to', 'map_polygon'].type,
    ):
        fr, to = fr.item(), to.item()
        if fr in translate and to in translate:
            pl2pl[0].append(translate[fr])
            pl2pl[1].append(translate[to])
            pl2pl_type.append(typ)
    data['map_polygon', 'to', 'map_polygon'].edge_index = torch.tensor(pl2pl, dtype=int)
    data['map_polygon', 'to', 'map_polygon'].type = torch.tensor(
        pl2pl_type, dtype=torch.uint8
    )

    return data


class RemoveDistantPolygons(BaseTransform):
    def __init__(self, max_dist: int = -1, t_hist: int = 11, t_pred: int = 80) -> None:
        self.max_dist = max_dist
        self.t_hist = t_hist
        self.t_pred = t_pred

    def forward(self, data: HeteroData) -> HeteroData:
        if self.max_dist <= 0:
            return data

        pos = data['agent']['position'][data['agent']['category'] > 0]
        if len(pos) == 0:
            pos = data['agent']['position']
            pos = pos[torch.randperm(len(pos))[: min(len(pos), max(5, len(pos) // 5))]]
        hist = pos[:, : self.t_hist, :2]

        if len(data['map_point']['position']) == 0:
            return data

        max_i = self.t_pred // 10
        points = [hist[:, -1] + (hist[:, -1] - hist[:, -10]) * i for i in range(max_i)]
        points = torch.cat(points, dim=0)

        pt_ag_dists = torch.norm(
            points[:, None] - data['map_point']['position'][None, :, :2], dim=-1
        ).min(0)[0]
        pts_close = pt_ag_dists < self.max_dist
        pls_close = torch.zeros(data['map_polygon']['num_nodes'], dtype=torch.long)
        pls_close.index_add_(
            0, data['map_point', 'to', 'map_polygon'].edge_index[1], pts_close.long()
        )
        drop_pls = ~(pls_close > 0)

        data = remove_polygons(data, drop_pls)

        return data


class RemoveDriveways(BaseTransform):
    def __init__(self, remove=False) -> None:
        self.remove = remove

    def forward(self, data: HeteroData) -> HeteroData:
        if not self.remove:
            return data

        is_driveway = data['map_polygon']['type'] == 5
        is_unknown = data['map_polygon']['type'] == 0
        data = remove_polygons(data, is_driveway | is_unknown)

        return data


class RemoveStationary(BaseTransform):
    def __init__(self, t_hist, max_road_dist=-1) -> None:
        self.t_hist = t_hist
        self.max_road_dist = max_road_dist

    def forward(self, data: HeteroData) -> HeteroData:

        if self.max_road_dist <= 0:
            return data

        if len(data['map_point']['position']) == 0:
            return data

        center_pos = data['map_point']['position'][data['map_point']['side'] == 2, :2]
        if len(center_pos) == 0:
            return data
        important = data['agent']['category'] > 0
        hist = data['agent']['position'][:, : self.t_hist, :2]
        stationary = torch.norm(hist[:, 0] - hist[:, -1], dim=-1) < 0.1
        far = (
            torch.norm(
                center_pos[None] - data['agent']['position'][:, None, 0, :2], dim=-1
            ).min(-1)[0]
            > self.max_road_dist
        )

        keep = important | ~stationary | ~far
        for key, val in data['agent'].items():
            if (
                hasattr(val, 'shape')
                and len(val.shape) > 0
                and val.shape[0] == data['agent']['num_nodes']
            ):
                data['agent'][key] = val[keep]
        data['agent']['id'] = [x for i, x in enumerate(data['agent']['id']) if keep[i]]
        data['agent']['num_nodes'] = len(data['agent']['position'])

        return data
