import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment
from scipy.ndimage.filters import gaussian_filter1d
import json


def tracking(poses_per_frame,
             actual_frames=None,
             last_seen_delay=2,
             scale_to_mm=1,
             min_track_length=4,
             max_distance_between_tracks=100,
             merge_distance=100,
             z_axis=2):
    """
    :param min_track_length: drop any track which is shorter than min_track_length frames
    :param last_seen_delay: allow to skip last_seen_delay frames for connecting a lost track
    :param max_distance_between_tracks: maximal distance in [mm] between tracks so that they can be associated
    :return:
    """

    n_frames = len(poses_per_frame)

    all_tracks = []

    for t in range(n_frames):
        if actual_frames is not None:
            real_t = actual_frames[t]
        else:
            real_t = t


        poses = poses_per_frame[t]
        poses = [list(pose) for pose in poses]        # list of poses and each pose is a list of array(1,3)
        if real_t == 112:
            print(f'{real_t}')

        poses = eliminate_duplicated_poses(poses, merge_distance)
        poses = correct_limb_size(poses)

        possible_tracks = []
        for track in all_tracks:
            if track.last_seen() + last_seen_delay < real_t:
                continue  # track is too old..
            possible_tracks.append(track)

        n = len(possible_tracks)
        if n > 0:
            m = len(poses)
            D = np.empty((n, m))
            for tid, track in enumerate(possible_tracks):
                for pid, pose in enumerate(poses):
                    D[tid, pid] = track.distance_to_last(pose)

            rows, cols = linear_sum_assignment(D)
            D = D * scale_to_mm  # ensure that distances in D are in [mm]

            handled_pids = set()
            for tid, pid in zip(rows, cols):
                d = D[tid, pid]
                if d > max_distance_between_tracks:
                    continue

                # merge pose into track
                track = possible_tracks[tid]
                pose = poses[pid]
                track.add_pose(real_t, pose)
                handled_pids.add(pid)

            # add all remaining poses as tracks
            for pid, pose in enumerate(poses):
                if pid in handled_pids:
                    continue
                track = Track(real_t, pose, last_seen_delay=last_seen_delay, z_axis=z_axis)
                all_tracks.append(track)

        else:  # no tracks yet... add them
            for pose in poses:
                track = Track(real_t, pose, last_seen_delay=last_seen_delay, z_axis=z_axis)
                all_tracks.append(track)

    surviving_tracks = []

    for track in all_tracks:
        if len(track) >= min_track_length:
            surviving_tracks.append(track)

    return surviving_tracks


def eliminate_duplicated_poses(poses, merge_distance, scale_to_mm=1, z_axis=2):
    distances = []  # (hid1, hid2, distance)
    n = len(poses)
    for i in range(n):
        for j in range(i + 1, n):
            pose1 = poses[i]
            pose2 = poses[j]
            distance = Track.distance_between_poses(pose1, pose2, z_axis)
            distances.append((i, j, distance * scale_to_mm))

    # the root merge is always the smallest hid
    # go through all merges and point higher hids
    # towards their smallest merge hid

    mergers_root = {}  # hid -> root
    mergers = {}  # root: [ hid, hid, .. ]
    all_merged_hids = set()
    for hid1, hid2, distance in distances:
        if distance > merge_distance:
            continue

        if hid1 in mergers_root and hid2 in mergers_root:
            continue  # both are already handled

        if hid1 in mergers_root:
            hid1 = mergers_root[hid1]

        if hid1 not in mergers:
            mergers[hid1] = [hid1]

        mergers[hid1].append(hid2)
        mergers_root[hid2] = hid1
        all_merged_hids.add(hid1)
        all_merged_hids.add(hid2)

    merged_humans = []
    for hid in range(n):
        if hid in mergers:
            poses_list = [poses[hid2] for hid2 in mergers[hid]]
            merged_humans.append(get_avg_pose(poses_list))
        elif hid not in all_merged_hids:
            merged_humans.append(poses[hid])
    return merged_humans


def get_avg_pose(poses):
    J = len(poses[0])
    result = [None] * J

    for jid in range(J):
        valid_points = []
        for pose in poses:
            if pose[jid] is not None:
                valid_points.append(pose[jid])
        if len(valid_points) > 0:
            result[jid] = np.mean(valid_points, axis=0)
        else:
            result[jid] = None
    return result


def correct_limb_size(poses):
    ua_range = (50, 500)            # shoulder - elbow
    la_range = (50, 550)            # elbow - wrist
    ul_range = (200, 600)           # hip - knee
    ll_range = (200, 600)           # knee - foot
    ns_range = (50, 400)            # nose - shoulder
    sh_range = (200, 800)           # shoulder - hip
    ss_range = (60, 600)            # shoulder - shoulder

    scale_to_mm = 1.0
    corrected_poses = []
    for keypoints in poses:
        # check head shoulders
        if test_distance(keypoints, scale_to_mm, 0, 5, *ns_range) and test_distance(keypoints, scale_to_mm, 0, 6, *ns_range):
            keypoints[0] = None

        if test_distance(keypoints, scale_to_mm, 5, 6, *ss_range):
            if test_distance(keypoints, scale_to_mm, 5, 11, *sh_range):
                keypoints[5] = None
            if test_distance(keypoints, scale_to_mm, 6, 12, *sh_range):
                keypoints[6] = None

        # check left arm
        if test_distance(keypoints, scale_to_mm, 5, 7, *ua_range):
            keypoints[7] = None
            keypoints[9] = None         # we need to disable wrist too
        elif test_distance(keypoints, scale_to_mm, 7, 9, *la_range):
            keypoints[9] = None

        # check right arm
        if test_distance(keypoints, scale_to_mm, 6, 8, *ua_range):
            keypoints[8] = None
            keypoints[10] = None         # we need to disable wrist too
        elif test_distance(keypoints, scale_to_mm, 8, 10, *la_range):
            keypoints[10] = None

        # check left leg
        if test_distance(keypoints, scale_to_mm, 11, 13, *ul_range):
            keypoints[13] = None
            keypoints[15] = None  # we need to disable foot too
        elif test_distance(keypoints, scale_to_mm, 13, 15, *ll_range):
            keypoints[15] = None

        # check right leg
        if test_distance(keypoints, scale_to_mm, 12, 14, *ul_range):
            keypoints[14] = None
            keypoints[16] = None  # we need to disable foot too
        elif test_distance(keypoints, scale_to_mm, 14, 16, *ll_range):
            keypoints[16] = None

        corrected_poses.append(keypoints)

    return corrected_poses


def test_distance(keypoints, scale_to_mm, jid1, jid2, lower, higher):
    a = keypoints[jid1]
    b = keypoints[jid2]
    # if np.isnan(np.sum(a)) or np.isnan(np.sum(b)):
    #     return False
    # distance = la.norm(a - b) * scale_to_mm
    # if lower <= distance <= higher:
    #     return False
    # else:
    #     return True

    if a is None or b is None:
        return False
    distance = la.norm(a - b) * scale_to_mm
    if lower <= distance <= higher:
        return False
    else:
        return True


class Track:

    @staticmethod
    def smoothing(track, sigma,
                  interpolation_range=4,
                  relevant_jids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]):
        """ smoothing of a track
        :param track:
        :param sigma:
        :param interpolation_range:
        :param relevant_jids: is set up for mscoco
        :return:
        """
        first_frame = track.first_frame()
        last_frame = track.last_seen() + 1
        n_frames = last_frame - first_frame
        print("n frames", n_frames)

        relevant_jids_lookup = {}
        relevant_jids = set(relevant_jids)

        delete_jids = []

        # step 0: make sure all relevent jids have entries
        for jid in relevant_jids:
            jid_found = False
            for frame in range(first_frame, last_frame):
                pose = track.get_by_frame(frame)
                if pose is not None and pose[jid] is not None:
                    jid_found = True
                    break

            if not jid_found:
                delete_jids.append(jid)

        for jid in delete_jids:
            relevant_jids.remove(jid)

        # step 1:
        unrecoverable = set()
        for jid in relevant_jids:
            XYZ = np.empty((n_frames, 3))
            for frame in range(first_frame, last_frame):
                pose = track.get_by_frame(frame)

                if pose is None or pose[jid] is None:
                    start_frame = max(first_frame, frame - interpolation_range)
                    end_frame = min(last_frame, frame + interpolation_range)

                    from_left = []
                    for _frame in range(start_frame, frame):
                        _pose = track.get_by_frame(_frame)
                        if _pose is None or _pose[jid] is None:
                            continue
                        from_left.append(_pose[jid])

                    from_right = []
                    for _frame in range(frame, end_frame):
                        _pose = track.get_by_frame(_frame)
                        if _pose is None or _pose[jid] is None:
                            continue
                        from_right.append(_pose[jid])

                    pts = []
                    if len(from_left) > 0:
                        pts.append(from_left[-1])
                    if len(from_right) > 0:
                        pts.append(from_right[0])

                    if len(pts) > 0:
                        pt = np.mean(pts, axis=0)
                    else:
                        # print("JID", jid)
                        # print('n frames', n_frames)
                        # print('current frame', frame)
                        # assert len(pts) > 0, 'jid=' + str(jid)
                        unrecoverable.add((jid, frame))
                        pt = np.array([0., 0., 0.])

                else:
                    pt = pose[jid]
                XYZ[frame - first_frame] = pt

            XYZ_sm = np.empty_like(XYZ)
            for dim in [0, 1, 2]:
                D = XYZ[:, dim]
                D = gaussian_filter1d(D, sigma, mode='reflect')
                XYZ_sm[:, dim] = D
            relevant_jids_lookup[jid] = XYZ_sm

        new_track = None

        for frame in range(first_frame, last_frame):
            person = []
            for jid in range(track.J):
                if jid in relevant_jids_lookup:
                    if (jid, frame) in unrecoverable:
                        person.append(None)
                    else:
                        XYZ_sm = relevant_jids_lookup[jid]
                        pt = XYZ_sm[frame - first_frame]
                        person.append(pt)
                else:
                    pose = track.get_by_frame(frame)
                    if pose is None:
                        person.append(None)
                    else:
                        person.append(pose[jid])
            if new_track is None:
                new_track = Track(frame, person, track.last_seen_delay, track.z_axis)
            else:
                new_track.add_pose(frame, person)

        return new_track

    @staticmethod
    def from_file(fname):
        """ load from file
        :param fname:
        :return:
        """
        track_as_json = json.load(open(fname))
        frames = track_as_json['frames']
        poses = track_as_json['poses']

        last_seen_delay = 99
        z_axis = 0
        frame0 = frames.pop(0)
        pose0 = poses.pop(0)

        track = Track(frame0, pose0, last_seen_delay, z_axis)
        for t, pose in zip(frames, poses):
            track.add_pose(t, pose)

        return track

    def __init__(self, t, pose, last_seen_delay, z_axis):
        """
        :param t: {int} time
        :param pose: 3d * J
        :param last_seen_delay: max delay between times
        :param z_axis: some datasets are rotated around one axis
        """
        self.frames = [int(t)]
        self.J = len(pose)
        self.poses = [pose]
        self.last_seen_delay = last_seen_delay
        self.lookup = None
        self.z_axis = z_axis

    def __len__(self):
        if len(self.frames) == 1:
            return 1
        else:
            first = self.frames[0]
            last = self.frames[-1]
            return last - first + 1

    def to_file(self, fname):

        poses = []
        for p in self.poses:
            if isinstance(p, np.ndarray):
                poses.append(p.tolist())
            elif isinstance(p, list):
                pose = []
                for joint in p:
                    if isinstance(joint, np.ndarray):
                        joint = joint.tolist()
                    pose.append(joint)
                poses.append(pose)
            else:
                raise Exception('Type is not a list:' + str(type(p)))

        data = {
            "J": self.J,
            "frames": self.frames,
            "poses": poses,
            'z_axis': self.z_axis
        }
        with open(fname, 'w') as f:
            json.dump(data, f)

    def last_seen(self):
        return self.frames[-1]

    def first_frame(self):
        return self.frames[0]

    def add_pose(self, t, pose):
        """ add pose
        :param t:
        :param pose:
        :return:
        """
        last_t = self.last_seen()
        assert last_t < t
        diff = t - last_t
        assert diff <= self.last_seen_delay
        self.frames.append(t)
        self.poses.append(pose)
        self.lookup = None  # reset lookup

    def get_by_frame(self, t):
        """ :returns pose by frame
        :param t:
        :return:
        """
        if self.lookup is None:
            self.lookup = {}
            for f, pose in zip(self.frames, self.poses):
                self.lookup[f] = pose

        if t in self.lookup:
            return self.lookup[t]
        else:
            return None

    def distance_to_last(self, pose):
        """ calculates the distance to the
            last pose
        :param pose:
        :return:
        """
        last_pose = self.poses[-1]
        return Track.distance_between_poses(pose, last_pose, self.z_axis)

    @staticmethod
    def distance_between_poses(pose1, pose2, z_axis):
        """
        :param pose1:
        :param pose2:
        :param z_axis: some datasets are rotated around one axis
        :return:
        """
        J = len(pose1)
        assert len(pose2) == J
        distances = []
        for jid in range(J):
            if pose2[jid] is None or pose1[jid] is None:
                continue
            d = la.norm(pose2[jid] - pose1[jid])
            distances.append(d)

        if len(distances) == 0:
            # TODO check this heuristic
            # take the centre distance in x-y coordinates
            valid1 = []
            valid2 = []
            for jid in range(J):
                if pose1[jid] is not None:
                    valid1.append(pose1[jid])
                if pose2[jid] is not None:
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