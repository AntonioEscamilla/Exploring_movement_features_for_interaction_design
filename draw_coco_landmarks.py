import math
import cv2
import numpy as np

connections = [[13, 11], [11, 9], [14, 12], [12, 10], [9, 10], [3, 4], [3, 5], [4, 6], [5, 7], [6, 8], [0, 1], [1, 2]]


def draw_landmarks(image, landmarks):
    """Draws the landmarks and the connections on the image.
    Args:
      image: A three channel RGB image represented as numpy ndarray.
      landmarks: landmark numpy array
    """
    idx_to_coordinates = {}
    landmarks = np.vstack((landmarks[0, :], np.mean(landmarks[[5, 6], :], axis=0), np.mean(landmarks[[11, 12], :], axis=0), landmarks[5:, :]))
    num_landmarks = landmarks.shape[0]
    for idx in range(num_landmarks):
        x_px = math.floor(landmarks[idx, 0])
        y_px = math.floor(landmarks[idx, 1])
        idx_to_coordinates[idx] = x_px, y_px
    if connections:
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], (0, 255, 0), 8)

    # Draws landmark points after finishing the connection lines, which is aesthetically better.
    for landmark_px in idx_to_coordinates.values():
        cv2.circle(image, landmark_px, 8, (255, 0, 0), -1)
