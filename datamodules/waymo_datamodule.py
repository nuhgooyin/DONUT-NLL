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
from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import WaymoDataset

from transforms import (
    SubsampleMapPoints,
    RemoveDistantPolygons,
    RemoveDriveways,
    WaymoTransform,
    RemoveStationary,
)
from torch_geometric.transforms import Compose


DEFAULT_WAYMO_TRANSFORM = Compose(
    [
        WaymoTransform(
            'waymo',
            inpaint_partial=True,
        ),
        RemoveDriveways(True),
        SubsampleMapPoints(5),
        RemoveDistantPolygons(max_dist=30, t_hist=11, t_pred=80),
        RemoveStationary(t_hist=11, max_road_dist=3),
    ]
)


class WaymoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_processed_dir: Optional[str] = None,
        val_processed_dir: Optional[str] = None,
        test_processed_dir: Optional[str] = None,
        train_transform: Optional[Callable] = DEFAULT_WAYMO_TRANSFORM,
        val_transform: Optional[Callable] = DEFAULT_WAYMO_TRANSFORM,
        test_transform: Optional[Callable] = DEFAULT_WAYMO_TRANSFORM,
        **kwargs,
    ) -> None:
        super(WaymoDataModule, self).__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        WaymoDataset(
            self.data_root,
            'train',
            self.train_processed_dir,
            self.train_transform,
        )
        WaymoDataset(
            self.data_root,
            'val',
            self.val_processed_dir,
            self.val_transform,
        )
        WaymoDataset(
            self.data_root,
            'test',
            self.test_processed_dir,
            self.test_transform,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = WaymoDataset(
            self.data_root,
            'train',
            self.train_processed_dir,
            self.train_transform,
        )
        self.val_dataset = WaymoDataset(
            self.data_root,
            'val',
            self.val_processed_dir,
            self.val_transform,
        )
        self.test_dataset = WaymoDataset(
            self.data_root,
            'test',
            self.test_processed_dir,
            self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
