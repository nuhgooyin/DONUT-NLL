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
import os
import pickle
from typing import Callable, List, Optional, Tuple, Union

from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData


class WaymoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        processed_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        print(f'Initializing WaymoDataset for {split} split...')
        root = os.path.expanduser(os.path.normpath(root))

        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split

        if processed_dir is None:
            processed_dir = os.path.join(root, 'processed', split)
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
        self._processed_dir = processed_dir
        self._processed_file_names = self._scan_processed_file_names()

        self.transform = transform

        super(WaymoDataset, self).__init__(
            root=root, transform=transform, pre_transform=None, pre_filter=None
        )

    def _scan_processed_file_names(self) -> List[str]:
        if not os.path.isdir(self._processed_dir):
            return []
        return sorted(
            entry.name
            for entry in os.scandir(self._processed_dir)
            if entry.is_file() and entry.name.endswith('.pkl')
        )

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def len(self) -> int:
        return len(self._processed_file_names)

    def get(self, idx: int) -> HeteroData:
        with open(self.processed_paths[idx], 'rb') as handle:
            return HeteroData(pickle.load(handle))
