import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from data_utils.ModelNetDataLoader import farthest_point_sample, pc_normalize


class ThreeDMNISTDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.npoints = args.num_point
        self.use_normals = args.use_normals
        self.uniform = args.use_uniform_sample
        self.split = split

        if split not in ('train', 'test'):
            raise ValueError(f'Unsupported split: {split}')

        file_name = 'train_point_clouds.h5' if split == 'train' else 'test_point_clouds.h5'
        self.file_path = os.path.join(root, file_name)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f'3D MNIST file not found: {self.file_path}')

        with h5py.File(self.file_path, 'r') as handle:
            self.sample_keys = sorted(handle.keys(), key=int)

        print('The size of %s data is %d' % (split, len(self.sample_keys)))

    def __len__(self):
        return len(self.sample_keys)

    def _select_points(self, point_set):
        if self.uniform and point_set.shape[0] >= self.npoints:
            return farthest_point_sample(point_set, self.npoints)

        if point_set.shape[0] == self.npoints:
            return point_set

        if point_set.shape[0] > self.npoints:
            if self.split == 'train':
                choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
            else:
                choice = np.linspace(0, point_set.shape[0] - 1, self.npoints, dtype=np.int64)
            return point_set[choice]

        deficit = self.npoints - point_set.shape[0]
        pad_idx = np.random.choice(point_set.shape[0], deficit, replace=True)
        return np.concatenate([point_set, point_set[pad_idx]], axis=0)

    def __getitem__(self, index):
        key = self.sample_keys[index]
        with h5py.File(self.file_path, 'r') as handle:
            group = handle[key]
            points = group['points'][:].astype(np.float32)
            normals = group['normals'][:].astype(np.float32)
            label = int(group.attrs['label'])

        point_set = np.concatenate([points, normals], axis=1) if self.use_normals else points
        point_set = self._select_points(point_set)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        return point_set, label
