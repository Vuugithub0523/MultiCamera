"""
Matching functions for ByteTrack.
Pure Python/NumPy implementation - no external dependencies required.
"""
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from . import kalman_filter as kf_module


def bbox_ious(atlbrs, btlbrs):
    """
    Compute IoU between two sets of bounding boxes.
    Pure NumPy implementation replacing cython_bbox.
    
    Args:
        atlbrs: (N, 4) array of boxes in tlbr format
        btlbrs: (M, 4) array of boxes in tlbr format
    
    Returns:
        ious: (N, M) IoU matrix
    """
    atlbrs = np.atleast_2d(atlbrs).astype(np.float64)
    btlbrs = np.atleast_2d(btlbrs).astype(np.float64)
    
    N = atlbrs.shape[0]
    M = btlbrs.shape[0]
    
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float64)
    
    # Expand dimensions for broadcasting
    # atlbrs: (N, 1, 4), btlbrs: (1, M, 4)
    a = atlbrs[:, np.newaxis, :]
    b = btlbrs[np.newaxis, :, :]
    
    # Intersection
    inter_tl = np.maximum(a[:, :, :2], b[:, :, :2])
    inter_br = np.minimum(a[:, :, 2:], b[:, :, 2:])
    inter_wh = np.maximum(0, inter_br - inter_tl)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    
    # Areas
    a_area = (a[:, :, 2] - a[:, :, 0]) * (a[:, :, 3] - a[:, :, 1])
    b_area = (b[:, :, 2] - b[:, :, 0]) * (b[:, :, 3] - b[:, :, 1])
    
    # Union
    union_area = a_area + b_area - inter_area
    
    # IoU
    ious = np.where(union_area > 0, inter_area / union_area, 0)
    
    return ious


def linear_assignment(cost_matrix, thresh):
    """
    Linear assignment using scipy instead of lap.
    
    Args:
        cost_matrix: (N, M) cost matrix
        thresh: threshold for valid matches
    
    Returns:
        matches: (K, 2) array of matched indices
        unmatched_a: indices of unmatched rows
        unmatched_b: indices of unmatched columns
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1]))
        )
    
    # Use scipy's linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_a = set(range(cost_matrix.shape[0]))
    unmatched_b = set(range(cost_matrix.shape[1]))
    
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] <= thresh:
            matches.append([row, col])
            unmatched_a.discard(row)
            unmatched_b.discard(col)
    
    matches = np.array(matches, dtype=int).reshape(-1, 2)
    return matches, np.array(list(unmatched_a)), np.array(list(unmatched_b))


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU.
    
    Args:
        atlbrs: list of tlbr boxes or ndarray
        btlbrs: list of tlbr boxes or ndarray
    
    Returns:
        ious: (N, M) IoU matrix
    """
    atlbrs = np.asarray(atlbrs, dtype=np.float64)
    btlbrs = np.asarray(btlbrs, dtype=np.float64)
    
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    
    return bbox_ious(atlbrs, btlbrs)


def iou_distance(atracks, btracks):
    """
    Compute cost matrix based on IoU distance.
    
    Args:
        atracks: list of STrack or ndarray of tlbr boxes
        btracks: list of STrack or ndarray of tlbr boxes
    
    Returns:
        cost_matrix: (N, M) cost matrix where cost = 1 - IoU
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float64)
    
    if isinstance(atracks[0], np.ndarray) or isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """
    Fuse IoU cost with detection scores.
    
    Args:
        cost_matrix: (N, M) IoU-based cost matrix
        detections: list of detections with score attribute
    
    Returns:
        fused cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost


def embedding_distance(tracks, detections, metric='cosine'):
    """
    Compute embedding distance between tracks and detections.
    
    Args:
        tracks: list of tracks with smooth_feat attribute
        detections: list of detections with curr_feat attribute
        metric: distance metric (default: cosine)
    
    Returns:
        cost_matrix: (N, M) embedding distance matrix
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    
    det_features = np.asarray([det.curr_feat for det in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    Gate cost matrix using Mahalanobis distance.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    gating_dim = 2 if only_position else 4
    gating_threshold = kf_module.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    Fuse appearance cost with motion cost.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    gating_dim = 2 if only_position else 4
    gating_threshold = kf_module.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha'
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    Fuse ReID cost with IoU cost.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost