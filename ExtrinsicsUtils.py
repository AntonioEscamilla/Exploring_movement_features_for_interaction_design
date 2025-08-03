import numpy as np


def world_to_camera_frame(x, R, T):
    """
    Args
        x: 3x1 3d point in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: 3X1 3d point in camera coordinates
    """
    x_h = _h = np.ones((4, 1))
    x_h[:3, :] = x
    Rt = np.hstack((R, T))
    return np.dot(Rt, x_h)


def cam_pose(R, T):
    return np.matmul(-R.T, T)


def cam_orientation(R):
    return np.matmul(R.T, np.array([0, 0, 1]).T)


def camera_to_world_frame(x, R, T):
    """
    Args
        x: 3x1 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        xcam: 3x1 points in world coordinates
    """
    return np.dot(R.T, x - T)


def line_intersection(planePoint, planeNormal, linePoint, lineDirection):
    """
    :param planePoint: A point on the plane.
    :param planeNormal: The normal vector of the plane.
    :param linePoint: A point on the line.
    :param lineDirection: The direction vector of the line.
    :return: The point of intersection between the line and the plane
    """
    line_direction = lineDirection / np.linalg.norm(lineDirection)
    if np.abs(planeNormal.dot(line_direction)) < 1e-6:
        return None

    t = (planeNormal.dot(planePoint) - planeNormal.dot(linePoint)) / planeNormal.dot(line_direction)
    return np.add(linePoint, np.multiply(t, line_direction))
