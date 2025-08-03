import numpy as np
import random
import numpy.linalg as la
from modules.one_euro_filter import OneEuroFilter


class Pose:
    num_kpts = 17
    kpt_names = ['nose', 'r_eye', 'l_eye', 'r_ear', 'l_ear',
                 'r_sho', 'l_sho', 'r_elb', 'l_elb', 'r_wri', 'l_wri',
                 'r_hip', 'l_hip', 'r_knee', 'l_knee', 'r_ank', 'l_ank']
    sigmas = np.array([.26, .25, .25, .35, .35,
                       .79, .79, .72, .72, .62, .62,
                       1.07, 1.07, .87, .87, .89, .89],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1


    def __init__(self, keypoints, confidence=1.0):
        super().__init__()
        self.keypoints = Pose.correct_limb_size(keypoints)
        #self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.volume = (self.bbox[3] - self.bbox[0])*(self.bbox[4] - self.bbox[1])*(self.bbox[5] - self.bbox[2])
        self.id = None
        self.color = [random.randint(0, 255) for _ in range(3)]
        self.filters = [[OneEuroFilter(beta=1e-15), OneEuroFilter(beta=1e-15), OneEuroFilter(beta=1e-15)] for _ in range(Pose.num_kpts)]

    @staticmethod
    def get_bbox(keypoints):
        itmax = np.amax(keypoints, axis=0)
        itmin = np.amin(keypoints, axis=0)

        return [itmin[i] for i in range(3)] + [itmax[i] for i in range(3)]

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def update_color(self, color=None):
        if color is not None:
            self.color = color

    @staticmethod
    def correct_limb_size(keypoints):
        ua_range = (50, 500)            # shoulder - elbow
        la_range = (50, 550)            # elbow - wrist
        ul_range = (200, 600)           # hip - knee
        ll_range = (200, 600)           # knee - foot
        ns_range = (50, 400)            # nose - shoulder
        sh_range = (200, 800)           # shoulder - hip
        ss_range = (60, 600)            # shoulder - shoulder

        scale_to_mm = 1.0
        # check head shoulders
        if Pose.test_distance(keypoints, scale_to_mm, 0, 5, *ns_range) and Pose.test_distance(keypoints, scale_to_mm, 0, 6, *ns_range):
            keypoints[0] = None

        if Pose.test_distance(keypoints, scale_to_mm, 5, 6, *ss_range):
            if Pose.test_distance(keypoints, scale_to_mm, 5, 11, *sh_range):
                keypoints[5] = None
            if Pose.test_distance(keypoints, scale_to_mm, 6, 12, *sh_range):
                keypoints[6] = None

        # check left arm
        if Pose.test_distance(keypoints, scale_to_mm, 5, 7, *ua_range):
            keypoints[7] = None
            keypoints[9] = None         # we need to disable wrist too
        elif Pose.test_distance(keypoints, scale_to_mm, 7, 9, *la_range):
            keypoints[9] = None

        # check right arm
        if Pose.test_distance(keypoints, scale_to_mm, 6, 8, *ua_range):
            keypoints[8] = None
            keypoints[10] = None         # we need to disable wrist too
        elif Pose.test_distance(keypoints, scale_to_mm, 8, 10, *la_range):
            keypoints[10] = None

        # check left leg
        if Pose.test_distance(keypoints, scale_to_mm, 11, 13, *ul_range):
            keypoints[13] = None
            keypoints[15] = None  # we need to disable foot too
        elif Pose.test_distance(keypoints, scale_to_mm, 13, 15, *ll_range):
            keypoints[15] = None

        # check right leg
        if Pose.test_distance(keypoints, scale_to_mm, 12, 14, *ul_range):
            keypoints[14] = None
            keypoints[16] = None  # we need to disable foot too
        elif Pose.test_distance(keypoints, scale_to_mm, 14, 16, *ll_range):
            keypoints[16] = None

        return keypoints

    @staticmethod
    def test_distance(keypoints, scale_to_mm, jid1, jid2, lower, higher):
        """
        :param human: [ (x, y, z) ] * J
        :param scale_to_mm:
        :param jid1:
        :param jid2:
        :param lower:
        :param higher:
        :return:
        """
        a = keypoints[jid1]
        b = keypoints[jid2]
        if np.isnan(np.sum(a)) or np.isnan(np.sum(b)):
            return False
        distance = la.norm(a - b) * scale_to_mm
        if lower <= distance <= higher:
            return False
        else:
            return True


def eliminate_duplicated_poses(current_poses):
    if len(current_poses) > 1:
        distance_matrix = np.ones((len(current_poses), len(current_poses))) * 1e8
        for i in range(len(current_poses) - 1):
            for j in range(i + 1, len(current_poses)):
                distance_matrix[i][j] = distance_between_poses(current_poses[i], current_poses[j])

        distance_idxs = []
        for i in range(len(current_poses) - 1):
            if np.min(distance_matrix[i]) <= 500:
                idx_max = np.argmin(distance_matrix[i])
                distance_idxs.append(idx_max)
                distance_matrix[:, idx_max] = np.ones(len(current_poses)) * 1e8
        distance_idxs.sort(reverse=True)
        for i in distance_idxs:
            current_poses.pop(i)

    return current_poses


def distance_between_poses(a, b, z_axis=2):
    """
    :param pose1:
    :param pose2:
    :param z_axis: some datasets are rotated around one axis
    :return:
    """
    pose1 = a.keypoints
    pose2 = b.keypoints
    J = len(pose1)
    assert len(pose2) == J
    distances = []
    for jid in range(J):
        if np.isnan(np.sum(pose2[jid])) or np.isnan(np.sum(pose1[jid])):
            continue
        d = la.norm(pose2[jid] - pose1[jid])
        distances.append(d)

    if len(distances) == 0:
        # TODO check this heuristic
        # take the centre distance in x-y coordinates
        valid1 = []
        valid2 = []
        for jid in range(J):
            if not np.isnan(np.sum(pose1[jid])):
                valid1.append(pose1[jid])
            if np.isnan(np.sum(pose2[jid])):
                valid2.append(pose2[jid])

        assert len(valid1) > 0
        assert len(valid2) > 0
        mean1 = np.mean(valid1, axis=0)
        mean2 = np.mean(valid2, axis=0)
        assert len(mean1) == 3
        assert len(mean2) == 3

        # we only care about xy coordinates
        mean1[z_axis] = 0
        mean2[z_axis] = 0

        return la.norm(mean1 - mean2)
    else:
        return np.mean(distances)  # TODO try different versions


def track_poses_by_distance(previous_poses, current_poses, threshold=200, smooth=True):
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)

    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_color = None
        best_matched_iou = 1e8
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = distance_between_poses(current_pose, previous_pose)
            if iou < best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_color = previous_pose.color
                best_matched_id = id
        if best_matched_iou <= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
            best_matched_color = None
        current_pose.update_id(best_matched_pose_id)
        current_pose.update_color(best_matched_color)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if np.isnan(np.sum(current_pose.keypoints[kpt_id])):
                    continue

                # reuse filter if previous pose has valid filter
                if best_matched_pose_id is not None:
                    nan_values_in_keypoint = np.isnan(np.sum(previous_poses[best_matched_id].keypoints[kpt_id]))
                    if not nan_values_in_keypoint:
                        current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
                current_pose.keypoints[kpt_id, 2] = current_pose.filters[kpt_id][2](current_pose.keypoints[kpt_id, 2])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_color = None
        best_matched_iou = 0
        for id, previous_pose in enumerate(previous_poses):
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose, threshold=0.9)
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_pose.id
                best_matched_color = previous_pose.color
                best_matched_id = id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
            best_matched_color = None
        current_pose.update_id(best_matched_pose_id)
        current_pose.update_color(best_matched_color)

        if smooth:
            for kpt_id in range(Pose.num_kpts):
                if current_pose.keypoints[kpt_id, 0] == -1:
                    continue
                # reuse filter if previous pose has valid filter
                if (best_matched_pose_id is not None
                        and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
                    current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
                current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
                current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
                current_pose.keypoints[kpt_id, 2] = current_pose.filters[kpt_id][2](current_pose.keypoints[kpt_id, 2])
            current_pose.bbox = Pose.get_bbox(current_pose.keypoints)


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.volume, b.volume)
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt