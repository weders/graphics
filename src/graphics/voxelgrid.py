import numpy as np

from .sampler import mesh_to_pointcloud
from .transform import compute_tsdf
from .utils import extract_mesh_marching_cubes
from .visualization import plot_mesh

class Voxelgrid(object):

    def __init__(self, resolution):

        self.resolution = resolution

        self._volume = None
        self._bbox = None

    def from_pointcloud(self, pointcloud):

        minx = pointcloud.points.x.min()
        miny = pointcloud.points.y.min()
        minz = pointcloud.points.z.min()
        maxx = pointcloud.points.x.max()
        maxy = pointcloud.points.y.max()
        maxz = pointcloud.points.z.max()

        diffx = maxx - minx
        diffy = maxy - miny
        diffz = maxz - minz

        minx -= self.resolution * diffx
        maxx += self.resolution * diffx
        miny -= self.resolution * diffy
        maxy += self.resolution * diffy
        minz -= self.resolution * diffz
        maxz += self.resolution * diffz

        nx = int((maxx - minx) / self.resolution)
        ny = int((maxy - miny) / self.resolution)
        nz = int((maxz - minz) / self.resolution)

        self._bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                              dtype=np.float32)

        volume_shape = np.diff(self._bbox, axis=1).ravel()/self.resolution
        volume_shape = np.ceil(volume_shape).astype(np.int32).tolist() # round up

        self._volume = np.zeros(volume_shape)

        for row, point in pointcloud.points.iterrows():
            x = int((point['x'] - minx) / self.resolution)
            y = int((point['y'] - miny) / self.resolution)
            z = int((point['z'] - minz) / self.resolution)

            self._volume[x, y, z] = 1.

    def from_obj(self, obj):

        # TODO: not the right approach, need to implement correct way of voxelization

        vertices = np.asarray(obj.vertices)
        faces = np.asarray(obj.meshes[None].faces)
        n_points = 100000

        pcl = mesh_to_pointcloud(vertices, faces, n_points)

        minx = pcl[:, 0].min(axis=0)
        maxx = pcl[:, 0].max(axis=0)
        miny = pcl[:, 1].min(axis=0)
        maxy = pcl[:, 1].max(axis=0)
        minz = pcl[:, 2].min(axis=0)
        maxz = pcl[:, 2].max(axis=0)

        diffx = maxx - minx
        diffy = maxy - miny
        diffz = maxz - minz

        minx -= self.resolution * diffx
        maxx += self.resolution * diffx
        miny -= self.resolution * diffy
        maxy += self.resolution * diffy
        minz -= self.resolution * diffz
        maxz += self.resolution * diffz

        nx = int((maxx - minx) / self.resolution)
        ny = int((maxy - miny) / self.resolution)
        nz = int((maxz - minz) / self.resolution)

        self._bbox = np.array(((minx, maxx), (miny, maxy), (minz, maxz)),
                              dtype=np.float32)
        self._volume = np.zeros((nx, ny, nz))

        for point in pcl:

            x = int((point[0] - minx) / self.resolution)
            y = int((point[1] - miny) / self.resolution)
            z = int((point[2] - minz) / self.resolution)

            self._volume[x, y, z] = 1.

    @property
    def bbox(self):
        assert self._bbox is not None
        return self._bbox

    @property
    def volume(self):
        assert self._volume is not None
        return self._volume

    def get_tsdf(self):

        assert self._volume is not None

        tsdf, i = compute_tsdf(self._volume)

        return tsdf

    def plot(self, mode='grid'):

        from mayavi import mlab

        if mode == 'grid':

            xx, yy, zz = np.where(self._volume == 1)
            mlab.points3d(xx, yy, zz,
                          mode='cube',
                          color=(0, 1, 0),
                          scale_factor=1)

        elif mode == 'mesh':

            mesh = extract_mesh_marching_cubes(self._volume)
            plot_mesh(mesh)

        mlab.show()


