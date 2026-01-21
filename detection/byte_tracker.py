"""
BYTETracker for multi-object tracking
Simplified version for native backend without config file dependency
"""
import numpy as np
from typing import List

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from . import matching


class STrack(BaseTrack):
    """Single object tracking class using Kalman filter."""
    
    shared_kalman = KalmanFilter()
    
    def __init__(self, tlwh, score):
        """
        Args:
            tlwh: bounding box in (top-left x, top-left y, width, height) format
            score: detection confidence score
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

    def predict(self):
        """Predict next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Predict states for multiple tracks."""
        if len(stracks) == 0:
            return
        
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True  # Always activate new tracks
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """Update a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Get current position in tlbr format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to center-x, center-y, aspect-ratio, height."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh format."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr format."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BYTETracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box.
    
    Simplified version without config file dependency.
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        mot20: bool = False
    ):
        """
        Initialize BYTETracker
        
        Args:
            track_thresh: Threshold for high confidence detections
            match_thresh: IoU threshold for matching
            track_buffer: Frames to keep lost tracks
            frame_rate: Video frame rate
            mot20: Use MOT20 settings (no score fusion)
        """
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh  # Changed from track_thresh + 0.1 to allow lower conf tracks
        self.match_thresh = match_thresh
        self.mot20 = mot20
        
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, img_info=None, img_size=None):
        """
        Update tracker with new detections.
        
        Args:
            output_results: (N, 5) array with [x1, y1, x2, y2, score]
            img_info: (height, width) of original image
            img_size: (height, width) of model input (for scaling)
        
        Returns:
            list of active tracks
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Parse detections
        if len(output_results) == 0:
            scores = np.array([])
            bboxes = np.array([]).reshape(0, 4)
        elif output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            # Handle tensor input
            if hasattr(output_results, 'cpu'):
                output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]

        # Scale bboxes if needed
        if img_info is not None and img_size is not None:
            img_h, img_w = img_info[0], img_info[1]
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
            if scale != 1.0:
                bboxes = bboxes / scale

        # Split into high and low confidence detections
        print(f"[ByteTracker] Total detections: {len(scores)}, track_thresh={self.track_thresh}")
        if len(scores) > 0:
            print(f"[ByteTracker] Score range: {scores.min():.3f} - {scores.max():.3f}")
            print(f"[ByteTracker] First 5 scores: {scores[:5]}")
        
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        
        print(f"[ByteTracker] High conf (>{self.track_thresh}): {remain_inds.sum()}, Low conf: {inds_second.sum()}")
        
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        # Create detections
        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)
                for tlbr, s in zip(dets, scores_keep)
            ]
        else:
            detections = []

        # Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 1: First association with high score detections
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 2: Second association with low score detections
        if len(dets_second) > 0:
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s)
                for tlbr, s in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []
        
        r_tracked_stracks = [
            strack_pool[i] for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Mark lost tracks
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 3: Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: Init new tracks
        for inew in u_detection:
            track = detections[inew]
            print(f"[ByteTracker] New detection score: {track.score:.3f}, det_thresh: {self.det_thresh:.3f}")
            if track.score < self.det_thresh:
                print(f"[ByteTracker] ❌ Rejected: {track.score:.3f} < {self.det_thresh:.3f}")
                continue
            print(f"[ByteTracker] ✅ Creating new track with score {track.score:.3f}")
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update track lists
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return [track for track in self.tracked_stracks if track.is_activated]


def joint_stracks(tlista, tlistb):
    """Join two track lists without duplicates."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if not exists.get(t.track_id, 0):
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract tlistb from tlista."""
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        if t.track_id in stracks:
            del stracks[t.track_id]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks based on IoU."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    
    return resa, resb
