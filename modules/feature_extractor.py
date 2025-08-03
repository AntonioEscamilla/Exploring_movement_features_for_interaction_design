###########################################################################################
###                        CODE:       WRITTEN BY: ANTONIO ESCAMILLA                    ###
###                        PROJECT:    MULTIPLE CAMERA CALIBRATION                      ###
###                                    BASED ON QT DESIGNER                             ###
###                        LICENCE:    MIT OPENSOURCE LICENCE                           ###
###                                                                                     ###
###                            CODE IS FREE TO USE AND MODIFY                           ###
###########################################################################################

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, OPTICS
import math
from collections import Counter
from scipy.spatial import ConvexHull
from math import sin, cos, atan
import numpy.linalg as linalg


def directional_similarity(trajectories, ids):
    def _get_direction_angle(trajectory):
        if trajectory.shape[0] > 1:
            a = trajectory[-1][:2]                      # actual_pos
            b = trajectory[-2][:2]                      # prev_pos
            c = b + np.array([1.0, 0])
            return math.atan2(a[1] - b[1], a[0] - b[0]) - math.atan2(c[1] - b[1], c[0] - b[0])
        else:
            return 0.0

    num_tracks = len(ids)
    similarity_matrix = np.zeros((num_tracks, num_tracks))
    for i in range(num_tracks - 1):
        for j in range(i+1, num_tracks):
            similarity_matrix[i, j] = np.cos(_get_direction_angle(trajectories[ids[i]]) - _get_direction_angle(trajectories[ids[j]]))
    return similarity_matrix


def displacement_similarity(trajectories, ids):
    def _get_displacement(trajectory):
        if trajectory.shape[0] > 1:
            a = trajectory[-1][:2]  # actual_pos
            b = trajectory[-2][:2]  # prev_pos

            return np.linalg.norm(a-b)
        else:
            return 0.0

    num_tracks = len(ids)
    similarity_matrix = np.zeros((num_tracks, num_tracks))
    for i in range(num_tracks - 1):
        for j in range(i + 1, num_tracks):
            dp = _get_displacement(trajectories[ids[i]])
            dq = _get_displacement(trajectories[ids[j]])
            similarity_matrix[i, j] = 1 - (abs(dp-dq)/(dp+dq))**1.0
    return similarity_matrix


def DynamicInteraction(trajectories, ids, out_format='matrix'):
    num_tracks = len(ids)
    if out_format == 'sum':
        x = np.sum(displacement_similarity(trajectories, ids) * directional_similarity(trajectories, ids)) / 1.0
        return 1 / (1 + math.exp(-x))
    else:
        return displacement_similarity(trajectories, ids) * directional_similarity(trajectories, ids)


def TrajectoryImitation(trajectories, ids, rotation_invariant=False, scale_invariant=True, thresholded=False):
    num_tracks = len(ids)

    similarity_matrix = np.zeros((num_tracks, num_tracks))
    correlations = {}
    all_corr = []
    corr_ids = {i_d: i_d for i_d in ids}
    if num_tracks > 2:
        combinations = 0
        for i in range(num_tracks - 1):
            correlations[ids[i]] = []
            for j in range(i + 1, num_tracks):
                if len(trajectories[ids[i]]) > 1 and len(trajectories[ids[j]]) > 1:
                    if len(trajectories[ids[i]]) < 50:
                        a = np.asarray(resample(trajectories[ids[i]][:, :2].tolist(), n=50))
                    else:
                        a = trajectories[ids[i]][:, :2]
                    if len(trajectories[ids[j]]) < 50:
                        b = np.asarray(resample(trajectories[ids[j]][:, :2].tolist(), n=50))
                    else:
                        b = trajectories[ids[j]][:, :2]
                    similarity_matrix[i, j] = procrustes(a, b, rotation_invariant=rotation_invariant, scale_invariant=True)[2]
                    if ids[i] not in all_corr and similarity_matrix[i, j] > 0.9:
                        if ids[j] not in all_corr:
                            all_corr.append(ids[j])
                            correlations[ids[i]].append(ids[j])
                    #combinations += 1
            if len(correlations[ids[i]]) > 0:
                all_corr.append(ids[i])
        combinations = math.factorial(num_tracks) / (math.factorial(2) * math.factorial(num_tracks - 2))
        # print(ids)
        # print(similarity_matrix)
        # print(correlations)

        for k, v in correlations.items():
            for i in v:
                corr_ids[i] = k
        # print(corr_ids)
        return np.sum(similarity_matrix) / combinations, corr_ids       # similarity_matrix ** 2
    else:
        return np.sum(similarity_matrix), corr_ids                      # similarity_matrix ** 2


def TrajectoryCorrelations(trajectories, ids, rotation_invariant=False, scale_invariant=True, thresholded=False):
    num_tracks = len(ids)

    similarity_matrix = np.zeros((num_tracks, num_tracks))
    correlations = {}
    all_corr = []

    if num_tracks > 2:
        for i in range(num_tracks - 1):
            correlations[i] = []
            for j in range(i + 1, num_tracks):
                if len(trajectories[ids[i]]) > 1 and len(trajectories[ids[j]]) > 1:
                    if len(trajectories[ids[i]]) < 50:
                        a = np.asarray(resample(trajectories[ids[i]][:, :2].tolist(), n=50))
                    else:
                        a = trajectories[ids[i]][:, :2]
                    if len(trajectories[ids[j]]) < 50:
                        b = np.asarray(resample(trajectories[ids[j]][:, :2].tolist(), n=50))
                    else:
                        b = trajectories[ids[j]][:, :2]
                    similarity_matrix[i, j] = procrustes(a, b, rotation_invariant=rotation_invariant, scale_invariant=True)[2]
                    if i not in all_corr and similarity_matrix[i, j] > 0.7:
                        if j not in all_corr:
                            all_corr.append(j)
                            correlations[i].append(j)

            if len(correlations[i]) > 0:
                all_corr.append(i)
        combinations = math.factorial(num_tracks) / (math.factorial(2) * math.factorial(num_tracks - 2))
        # print(similarity_matrix)
        # print(correlations)

        return np.sum(similarity_matrix) / combinations, correlations
    else:
        return np.sum(similarity_matrix), correlations


def resample(points, n=50):
    # Get the length that should be between the returned points
    path_length = pathLength(points) / float(n - 1)
    newPoints = [points[0]]
    D = 0.0
    i = 1
    while i < len(points):
        point = points[i - 1]
        next_point = points[i]
        d = getDistance(point, next_point)
        if D + d >= path_length:
            delta_distance = float((path_length - D) / d)
            q = [0., 0.]
            q[0] = point[0] + delta_distance * (next_point[0] - point[0])
            q[1] = point[1] + delta_distance * (next_point[1] - point[1])
            newPoints.append(q)
            points.insert(i, q)
            D = 0.
        else:
            D += d
        i += 1
    if len(newPoints) == n - 1:  # Fix a possible roundoff error
        newPoints.append(points[0])
    return newPoints


def pathLength(points):
    length = 0
    for (i, j) in zip(points, points[1:]):
        length += getDistance(i, j)
    return length


def getDistance(point1, point2):
    return linalg.norm(np.array(point2) - np.array(point1))


def procrustes(data1, data2, use_x0_translate=False, rotation_invariant=False, scale_invariant=True):
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data using first point(x0) of mean
    if use_x0_translate:
        mtx1 -= mtx1[0]
        mtx2 -= mtx2[0]
    else:
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    if scale_invariant:
        mtx1 /= norm1
        mtx2 /= norm2

    # transform mtx2 to minimize disparity
    if rotation_invariant:
        R, s = orthogonal_procrustes(mtx1, mtx2)
        mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    #similarity = 1 / (1 + np.exp(-4 * (1 - disparity)))
    similarity = 1 - disparity
    if similarity < 0:
        similarity = 0

    return mtx1, mtx2, similarity


def trajectory_transform(data1, use_x0_translate=False):
    mtx1 = np.array(data1, dtype=np.double, copy=True)

    if mtx1.ndim != 2:
        raise ValueError("Input must be two-dimensional")
    if mtx1.size == 0:
        raise ValueError("Input must be >0 rows and >0 cols")

    # translate all the data using first point(x0) of mean
    if use_x0_translate:
        mtx1 -= mtx1[0]
    else:
        mtx1 -= np.mean(mtx1, 0)

    norm1 = np.linalg.norm(mtx1)

    if norm1 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1

    return mtx1


def point_at_length_fraction(lengths, fraction):
    total_length = np.sum(lengths)
    length_to_reach = total_length * fraction
    length = 0.
    index = 0
    while length < length_to_reach:
        length += lengths[index]
        index += 1
    return index


def get_control_points(depth, all_lengths_in_path):
    step_fraction = 1.0/depth
    control_points = []
    for i in range(depth):
        fraction = i * step_fraction
        index = point_at_length_fraction(all_lengths_in_path, fraction)
        control_points.append(index)
    control_points.append(all_lengths_in_path.shape[0])
    return control_points


def distance_geometry(path, depth=4):
    distance_geometry_distances = []
    all_control_points = []
    all_lengths_in_path = np.sqrt(np.sum(np.diff(np.asarray(path), axis=0) ** 2, axis=1))
    travel_distance = np.sum(all_lengths_in_path)
    for d in range(1, depth+1):
        #print(f'd: {d}')
        control_points = get_control_points(d, all_lengths_in_path)
        #print(f'control points: {control_points}')
        all_control_points += control_points
        for i in range(d):
            control_point_distance = getDistance(path[control_points[i]], path[control_points[i+1]])
            distance_geometry_distances.append(control_point_distance / (travel_distance / d))
    return distance_geometry_distances, all_control_points


def convex_hull_feats(path):
    mtx1 = np.array(path, dtype=np.double, copy=True)
    mtx1 -= np.mean(mtx1, 0)
    norm = np.linalg.norm(mtx1)
    mtx1 /= norm

    hull = ConvexHull(mtx1)
    return mtx1,  20*hull.volume


def norm_and_stack_feat(x_data, feat, scaler=None):
    feat = np.asarray(feat).reshape((-1, 1))
    if scaler is None:
        max = np.amax(feat)
        feat = feat / max
        return np.hstack((x_data, feat)), max
    else:
        feat = feat / scaler
        return np.hstack((x_data, feat))


def stack_feat(x_data, feat):
    feat = np.asarray(feat).reshape((-1, 1))

    return np.hstack((x_data, feat))


def DBSCAN_predict(db, x):
    dists = np.sqrt(np.sum((db.components_ - x)**2, axis=1))
    i = np.argmin(dists)
    return db.labels_[db.core_sample_indices_[i]] if dists[i] < db.eps else -1


def InstantPositionCluster(trajectories, ids, canvas_width):
    x_y_positions = []
    for i in range(len(ids)):
        trajectory = trajectories[ids[i]]
        x_y_positions.append(trajectory[-1][:2])
    x_y_positions = np.asarray(x_y_positions) / canvas_width

    bandwidth = estimate_bandwidth(x_y_positions, quantile=0.5)
    clustering = MeanShift(bandwidth=0.5).fit(x_y_positions)
    labels = clustering.labels_

    center_points = clustering.cluster_centers_ * canvas_width
    counted_labels = Counter(clustering.labels_)
    return center_points, counted_labels, labels, x_y_positions * canvas_width


def LongTermPosCluster(trajectories, ids, canvas_width):
    x_y_positions = []
    for i in range(len(ids)):
        trajectory = trajectories[ids[i]]
        x_y_positions += trajectory[:, :2].tolist()
    x_y_positions = np.asarray(x_y_positions) / canvas_width

    #clustering = DBSCAN(eps=0.05, min_samples=15).fit(x_y_positions)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(x_y_positions)      # best working set up

    labels = clustering.labels_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    return n_clusters, labels, x_y_positions*canvas_width


def TrajectoryShapeClassification(cluster_model, trajectories, ids):
    num_tracks = len(ids)
    feat_data = []
    convex_hull_areas = []

    for i in range(num_tracks):
        if len(trajectories[ids[i]]) == 50:
            actual_stroke = trajectories[ids[i]][:, :2]
            feat_vect, _ = distance_geometry(actual_stroke, 2)
            feat_data.append(feat_vect)

            _, f = convex_hull_feats(actual_stroke)
            convex_hull_areas.append(f)

    if len(feat_data) == 0:
        return None

    feat_data = np.asarray(feat_data)
    feat_data = stack_feat(feat_data, convex_hull_areas)

    labels = []
    for i in range(feat_data.shape[0]):
        predict = DBSCAN_predict(cluster_model, feat_data[i])
        print(f'predicted clusters: {predict}')
        labels.append(predict)
    return labels


def TrajectoryShapeKNNClassification(knn_model, trajectories, ids):
    num_tracks = len(ids)
    feat_data = []
    convex_hull_areas = []

    for i in range(num_tracks):
        if len(trajectories[ids[i]]) == 50:
            actual_stroke = trajectories[ids[i]][:, :2]
            feat_vect, _ = distance_geometry(actual_stroke, 2)
            feat_data.append(feat_vect)

            _, f = convex_hull_feats(actual_stroke)
            convex_hull_areas.append(f)

    if len(feat_data) == 0:
        return None

    feat_data = np.asarray(feat_data)
    feat_data = stack_feat(feat_data, convex_hull_areas)

    predict = knn_model.predict(feat_data)
    print(f'predicted classes: {predict}')
    print(knn_model.predict_proba(feat_data))
    return predict


def thresholder(data, value):
    if data < value:
        data = 0
    else:
        data = 1
    return data
