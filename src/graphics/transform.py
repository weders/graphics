import numpy as np
import dt

from copy import copy


def compute_tsdf(grid):

    grid = copy(grid)

    grid[np.where(grid == 0.)] = 10000000000. #very bugy, need to discuss that
    grid[np.where(grid == 1.)] = 0.

    tsdf, i = dt.compute(grid)

    return tsdf, i







