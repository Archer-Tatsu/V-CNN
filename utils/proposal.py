import numpy as np
from math import pi
import torch


def generate_anchors(shape):
    """
    Generate anchors on the sphere.
    :param shape: Spatial shape of the feature map: (height, width).
    """

    # Enumerate shifts in feature space
    stride = np.array((180, 360)) / shape
    shifts_lat = np.arange(0, shape[0]) * stride[0] + 0.5 * stride[0] - 90
    shifts_lon = np.arange(0, shape[1]) * stride[1] + 0.5 * stride[1] - 180
    shifts_lon, shifts_lat = np.meshgrid(shifts_lon, np.flip(shifts_lat))

    # Reshape to get a list
    anchors = np.stack([shifts_lat, shifts_lon], axis=2).reshape([-1, 2])

    return anchors.astype(np.float32)


def softer_nms(hm_points, weights, threshold=13.75, top_k=256, proposal_count=20):
    """
    Apply non-maximum suppression at test time to avoid proposing too many overlapping viewports.
    :param hm_points: All predicted HM points on the sphere, Shape: (N, 2).
    :param weights: Predicted weights corresponding to the HM points, Shape: (N, ).
    :param threshold: (float) The threshold for suppressing close HM in degree.
    :param top_k: (int) The maximum number of points to consider.
    :param proposal_count: (int) The maximum number of points to propose.
    :return The proposed HM points with weights.
    """

    proposed_points = []
    proposed_weights = []
    if hm_points.numel() == 0:
        return proposed_points

    lat = hm_points[:, 0] * pi / 180
    lon = hm_points[:, 1] * pi / 180
    _, idx = weights.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest values

    lat1, lat2 = torch.meshgrid((lat[idx], lat[idx]))
    lon1, lon2 = torch.meshgrid((lon[idx], lon[idx]))

    # Great-circle distance
    distances = torch.acos(
        torch.cos(lat1) * torch.cos(lat2) * torch.cos(lon1 - lon2) + torch.sin(lat1) * torch.sin(lat2))

    distances[torch.eye(distances.shape[0], dtype=torch.uint8)] = 0

    while idx.numel() > 0:
        lt_mask = distances[-1].lt(threshold * pi / 180)
        lt_weights = weights[idx[lt_mask]]
        lt_points = hm_points[idx[lt_mask], :]

        new_point = torch.sum(lt_points * lt_weights[:, None], dim=0) / torch.sum(lt_weights)
        proposed_points.append(new_point)
        proposed_weights.append(torch.sum(lt_weights))

        ge_mask = 1 - lt_mask
        idx = idx[ge_mask]

        if torch.any(ge_mask):
            distances = distances[ge_mask]
            distances = distances[:, ge_mask]
        else:
            break

        if idx.size(0) == 1 or len(proposed_points) == proposal_count:
            break

    return proposed_points, proposed_weights


def proposal_layer(weights, offsets, proposal_count, nms_threshold, anchors, mask):
    """
    Receives anchor weights and selects a subset to pass as proposals to the second stage.
    Filtering is done based on anchor weights and non-maximum suppression to remove overlaps.
    It also applies HM refinement offsets to anchors.
    :param weights: Predicted weights corresponding to all anchors.
    :param offsets: Predicted offsets corresponding to all anchors.
    :param proposal_count: (int) The maximum number of points to propose.
    :param nms_threshold: (float) The threshold for suppressing close HM in degree.
    :param anchors: All anchor points.
    :param mask: The mask to down sample anchors near the polars.
    :return The proposed HM points with normalized weights.
    """

    # Currently only supports batch size 1
    weights = weights.squeeze(0)
    offsets = offsets.squeeze(0)

    weights = weights.view(-1)
    offsets = offsets * 180 / pi
    hm_points = anchors + offsets
    if mask is not None:
        weights = weights[mask]
        hm_points = hm_points[mask]

    # Fix boundary at +-180
    ids = hm_points[:, 1] > 180
    hm_points[ids, 1] = hm_points[ids, 1] - 360
    ids = hm_points[:, 1] < -180
    hm_points[ids, 1] = hm_points[ids, 1] + 360

    # Non-max suppression
    if proposal_count is not None:
        hm_points, weights = softer_nms(hm_points, weights, nms_threshold, proposal_count=proposal_count)
        hm_points = torch.stack(hm_points)
        weights = torch.stack(weights)
        if float(weights.sum()) > 0:
            weights /= weights.sum()

    return hm_points, weights
