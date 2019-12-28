import numpy as np
import math
from functools import lru_cache
import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def viewport2sph_coord(port_w, port_h, fov_x, fov_y):
    """
    Generate the meshgrid of the viewport and transform it to sphere coordinate.
    :param port_w: The width of the viewport in pixel.
    :param port_h: The height of the viewport in pixel.
    :param fov_x: Horizontal FoV in degree.
    :param fov_y: Vertical FoV in degree.
    :return: Sphere 3d coordinates of points of the meshgrid. Numpy ndarray with shape of (N, 3).
    """

    u_mesh, v_mesh = np.meshgrid(range(port_w), range(port_h))
    u_mesh, v_mesh = u_mesh.flatten(), v_mesh.flatten()

    u_mesh = u_mesh.astype(np.float64) + 0.5
    v_mesh = v_mesh.astype(np.float64) + 0.5

    fov_x_rad = math.pi * fov_x / 180
    fov_y_rad = math.pi * fov_y / 180
    fx = port_w / (2 * math.tan(fov_x_rad / 2))
    fy = port_h / (2 * math.tan(fov_y_rad / 2))

    K = np.asmatrix([[fx, 0, port_w / 2], [0, - fy, port_h / 2], [0, 0, 1]])

    e = np.asmatrix([u_mesh, v_mesh, np.ones_like(u_mesh)])
    q = K.I * e
    q_normed = q / np.linalg.norm(q, axis=0, keepdims=True)
    P = np.diag([1, 1, -1]) * q_normed
    return np.asarray(P)


@lru_cache(maxsize=1)
def cal_alignment_grid(viewport_resolution, lat, lon, P):
    """
    Calculate the grid for viewport alignment according to the center of the viewport.
    :param viewport_resolution: Tuple. (height_of_viewport, width_of_viewport)
    :param lat: Latitude of the center of the viewport (i.e., head movement position) in degree. 1-D array
    :param lon: Longitude of the center of the viewport (i.e., head movement position) in degree. 1-D array
    :param P: Viewport meshgrid in sphere cooordinate. Numpy ndarray with shape of (N, 3).
    :return: Grid for interpolatation. Tensor in (viewport_num, *viewport_resolution).
    """
    viewport_num = lat.shape[0]

    phi = lat * math.pi / 180
    tht = -lon * math.pi / 180

    # Rotation matrix
    R = torch.stack(
        (torch.stack((torch.cos(tht), torch.sin(tht) * torch.sin(phi), torch.sin(tht) * torch.cos(phi))),
         torch.stack((torch.zeros_like(phi), torch.cos(phi), - torch.sin(phi))),
         torch.stack((-torch.sin(tht), torch.cos(tht) * torch.sin(phi), torch.cos(tht) * torch.cos(phi)))))

    P = P.to(R)
    E = torch.matmul(R.permute(0, 2, 1), P)

    lat = 90 - torch.acos(E[1, :]) * 180 / math.pi
    lon = torch.atan2(E[0, :], -E[2, :]) * 180 / math.pi
    lat = lat.view((viewport_num, *viewport_resolution))
    lon = lon.view((viewport_num, *viewport_resolution))

    pix_height = -lat / 90
    pix_width = lon / 180
    grid = torch.stack((pix_width, pix_height))
    grid = grid.permute(1, 2, 3, 0).to(torch.float)

    return grid


def viewport_alignment(img, p_lat, t_lon, viewport_resolution=(600, 540)):
    """
    Apply viewport alignment.
    :param img: Tensor of the frame.
    :param p_lat: Latitude of the center of the viewport (i.e., head movement position) in degree. 1-D array.
    :param t_lon: Longitude of the center of the viewport (i.e., head movement position) in degree. 1-D array.
    :param viewport_resolution: Tuple. (height_of_viewport, width_of_viewport).
    :return viewports. (viewport_num, 3, *viewport_resolution).
    """
    viewport_num = p_lat.shape[0]
    port_h, port_w = viewport_resolution

    P = torch.tensor(viewport2sph_coord(port_w, port_h, 71, 74).astype(np.float32))

    grid = cal_alignment_grid(viewport_resolution, p_lat, t_lon, P)
    viewport = F.grid_sample(img.expand(viewport_num, -1, -1, -1), grid)

    return viewport
