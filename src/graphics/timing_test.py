import time
import torch
import numpy as np

from voxelgrid import FeatureGrid
from sparse_voxelgrid import RegularSparseVoxelGrid

resolution = 128
cell_size = 1. / resolution
n_features = 9
density_factor = 0.1
grid_origin = np.array([-0.5, -0.5, -0.5])
bbox = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])


def get_random_voxel_indices(density_factor, resolution):
    voxel_indices_list = []
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                sample = np.random.uniform(low=0.0, high=1.0, size=None)
                if sample < density_factor:
                    voxel_indices_list.append([x, y, z])
    return voxel_indices_list


if __name__ == '__main__':
    dense_grid = FeatureGrid(resolution=cell_size,
                             bbox=bbox,
                             origin=grid_origin,
                             n_features=n_features)

    sparse_grid = RegularSparseVoxelGrid(grid_origin=grid_origin,
                                         cell_size=cell_size,
                                         n_features=n_features,
                                         compute_device='cuda',
                                         store_device='cpu')

    # iterate over the 3d grid and take an index with probability = density_factor
    voxel_indices_list = get_random_voxel_indices(density_factor, resolution)
    voxel_indices_np = np.array(voxel_indices_list)
    voxel_indices_torch = torch.tensor(voxel_indices_np)
    n_samples = len(voxel_indices_list)

    random_features_np = np.random.rand(n_samples, n_features)
    random_features_torch = torch.tensor(random_features_np)

    print('populating dense grid ...')
    for idx in range(voxel_indices_np.shape[0]):
        voxel_index = voxel_indices_np[idx]
        dense_grid._data[voxel_index] = random_features_np[idx]

    print('populating sparse grid ...')
    sparse_grid.update(voxel_indices_torch, random_features_torch)
    sparse_grid.average()

    # do the actual queries including timing
    n_queries = 10000
    query_indices = np.random.randint(n_samples, size=n_queries)
    query_voxel_indices_np = voxel_indices_np[query_indices]
    query_voxel_indices_torch = torch.tensor(query_voxel_indices_np)

    # dense grid
    start_time = time.time()
    for idx in range(n_queries):
        query_position = query_voxel_indices_np[idx]
        feature_vector = dense_grid._data[query_position]
    print("--- %s seconds ---" % (time.time() - start_time))

    # sparse grid
    start_time = time.time()
    for idx in range(n_queries):
        query_position = query_voxel_indices_torch[idx]
        feature_vector = sparse_grid.get_feature_vector_from_index(query_position)
    print("--- %s seconds ---" % (time.time() - start_time))

    # sparse grid v2
    start_time = time.time()
    for idx in range(n_queries):
        query_position = query_voxel_indices_torch[idx]
        feature_vector = sparse_grid.get_feature_vector_from_index2(query_position)
    print("--- %s seconds ---" % (time.time() - start_time))
