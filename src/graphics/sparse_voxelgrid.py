import os.path as osp
import itertools
import math
from collections import namedtuple

import torch
import torch_scatter
from matplotlib import cm
import numpy as np
from tqdm import tqdm
import trimesh


class RegularSparseVoxelGrid(object):

    def __init__(self, grid_origin=None, cell_size=None, n_features=None, compute_device='cuda', store_device='cpu'):

        self.grid_origin = grid_origin
        self.cell_size = cell_size
        self.n_features = n_features
        self.c_dev = compute_device
        self.s_dev = store_device
        self.pin_memory = store_device == 'cpu' and compute_device != 'cpu'

        # For intermediate accumulative storage of results
        self.grid_points = []
        self.values = []

        # For aggregated storage of results and compute
        self.unique_grid_points = torch.empty([0, 3], dtype=torch.long, device=store_device)
        self.unique_values = torch.empty([0, self.n_features], dtype=torch.float, device=store_device)
        self.unique_weights = torch.empty([0], dtype=torch.float, device=store_device)

    def update(self, grid_points, values):
        assert grid_points.shape[0] == values.shape[0]
        assert values.shape[1] == self.n_features  # assertions might slow down the thing drastically.
        update_grid_points = grid_points.to(self.s_dev)
        update_values = values.to(self.s_dev)
        if self.pin_memory:
            update_grid_points = update_grid_points.pin_memory()
            update_values = update_values.pin_memory()

        self.grid_points.append(update_grid_points)
        self.values.append(update_values)

    def average(self):
        sizes = [0] + [self.unique_grid_points.shape[0]]
        sizes += [v.shape[0] for v in self.grid_points]
        sizes = torch.tensor(sizes, dtype=torch.long, device=self.c_dev)

        # Concatenate all grid_points
        grid_points = torch.cat([self.unique_grid_points.to(self.c_dev)] + [v.to(self.c_dev) for v in self.grid_points], dim=0)
        self.grid_points = []

        # Then, coalesce and store new grid points
        grid_points, rulebook = torch.unique(grid_points, return_inverse=True, dim=0)
        self.unique_grid_points = grid_points.to(self.s_dev)
        if self.pin_memory:
            self.unique_grid_points = self.unique_grid_points.pin_memory()
        num_unique_grid_points = grid_points.shape[0]
        del grid_points

        # Assemble weights
        weights = torch.ones([sizes.sum(0)], dtype=torch.float, device=self.c_dev)
        unique_weights = self.unique_weights.to(self.c_dev)
        weights[:self.unique_weights.shape[0]] = unique_weights

        # Then sum and store new weights
        unique_weights = weights.new_zeros([num_unique_grid_points])
        torch_scatter.scatter_add(weights, rulebook, out=unique_weights, dim=0)
        self.unique_weights = unique_weights.to(self.s_dev)
        if self.pin_memory:
            self.unique_weights = self.unique_weights.pin_memory()

        # Concatenate all values and weight them
        values = torch.cat([self.unique_values.to(self.c_dev)] + [v.to(self.c_dev) for v in self.values], dim=0)
        self.values = []

        values = weights.view(-1, 1) * values
        del weights

        # Then average and store new weighted values
        unique_values = values.new_zeros([num_unique_grid_points, self.n_features])
        torch_scatter.scatter_add(values, rulebook, out=unique_values, dim=0)
        del values

        self.unique_values = unique_values / unique_weights.view(-1, 1)

        self.unique_values = self.unique_values.to(self.s_dev)
        if self.pin_memory:
            self.unique_values = self.unique_values.pin_memory()
        del unique_values, unique_weights

    # TODO: this is very nasty, needs to be modified.
    def get_feature_vector_from_index(self, grid_index):
        # assumption: is in list, modify
        tensor_indices = (self.unique_grid_points == grid_index).nonzero()
        tensor_index = tensor_indices[0][0]
        return self.unique_values[tensor_index]
