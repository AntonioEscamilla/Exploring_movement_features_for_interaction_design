###########################################################################################
###                        CODE:       WRITTEN BY: ANTONIO ESCAMILLA                    ###
###                        PROJECT:    MULTI-VIEW MULTIPLE-PEOPLE POSE ESTIMATION       ###
###                                    BASED ON QT DESIGNER                             ###
###                        LICENCE:    MIT OPENSOURCE LICENCE                           ###
###                                                                                     ###
###                            CODE IS FREE TO USE AND MODIFY                           ###
###########################################################################################
'''

pyqt==5.9.2
pyqtgraph==0.11.0
pyopengl==3.1.5
'''

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import math
from scipy.spatial import ConvexHull


class View3dWidget(QWidget):

    def __init__(self, parent=None):
        super(View3dWidget, self).__init__(parent)

        self.plotter = gl.GLViewWidget()
        #self.plotter.setVisible(True)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.plotter)
        self.setLayout(self.main_layout)

        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  
        self.plotter.setCameraPosition(distance=15000, elevation=45, azimuth=-135)
        #self.plotter.setBackgroundColor('#5B5A5A')

        self.num_cams = 4

        # ground plane
        gz = gl.GLGridItem()
        gz.scale(800, 800, 1)
        self.plotter.addItem(gz)

        # ground plane origin axis
        self.origin = {}
        rgb = [pg.glColor((255, 0, 0)), pg.glColor((0, 255, 0)), pg.glColor((0, 0, 255))]
        axes_size = 500
        axes = np.array([[axes_size, 0, 0], [0, axes_size, 0], [0, 0, axes_size]])
        for i in range(3):
            self.origin[i] = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], axes[i]]), color=rgb[i], width=3, antialias=True)
            self.origin[i].setVisible(True)
            self.plotter.addItem(self.origin[i])

        # cam center
        self.cam_positions = gl.GLScatterPlotItem(pos=np.random.randint(10, size=(self.num_cams, 3)), color=pg.glColor((255, 255, 255)), size=10)
        self.cam_positions.setVisible(False)
        self.plotter.addItem(self.cam_positions)

        # cameras axis
        self.axes = {}
        rgb = [pg.glColor((255, 0, 0)), pg.glColor((0, 255, 0)), pg.glColor((0, 0, 255))]
        for j in range(self.num_cams):
            for i in range(3):
                n = int(3*j + i)
                self.axes[n] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgb[i], width=3, antialias=True)
                self.axes[n].setVisible(False)
                self.plotter.addItem(self.axes[n])

        # cameras frames
        self.far_cam_frame = {}
        for j in range(self.num_cams):
            for i in range(4):
                n = int(4 * j + i)
                self.far_cam_frame[n] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color='w', width=3, antialias=True)
                self.far_cam_frame[n].setVisible(False)
                self.plotter.addItem(self.far_cam_frame[n])

        self.near_cam_frame = {}
        for j in range(self.num_cams):
            for i in range(4):
                n = int(4 * j + i)
                self.near_cam_frame[n] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color='w', width=1, antialias=True)
                self.near_cam_frame[n].setVisible(False)
                self.plotter.addItem(self.near_cam_frame[n])

        self.connectors_cam_frame = {}
        for j in range(self.num_cams):
            for i in range(4):
                n = int(4 * j + i)
                self.connectors_cam_frame[n] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color='w', width=1, antialias=True)
                self.connectors_cam_frame[n].setVisible(False)
                self.plotter.addItem(self.connectors_cam_frame[n])


        keypoints = np.random.randint(10, size=(17, 3))

        # keypoints connectors
        self.body_lines = {}
        self.connections = [[13, 11], [11, 9], [14, 12], [12, 10], [9, 10], [3, 4], [3, 5],
                            [4, 6], [5, 7], [6, 8], [0, 1], [1, 2]]
        self.n_connect = len(self.connections)
        for i in range(6):
            for n, pts in enumerate(self.connections):
                idx = self.n_connect*i + n
                self.body_lines[idx] = gl.GLLinePlotItem(pos=np.array([keypoints[p] for p in pts]), color=pg.glColor((0, 0, 255)), width=3, antialias=True)
                self.body_lines[idx].setVisible(False)
                self.body_lines[idx].setGLOptions('translucent')
                self.plotter.addItem(self.body_lines[idx])

        # keypoints
        kpts_colors = [(0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)]
        kpts_sizes = [60.0, 30.0]
        self.body_points = {}
        for i in range(2):
            self.body_points[i] = gl.GLScatterPlotItem(pos=keypoints, color=kpts_colors[i], size=kpts_sizes[i], pxMode=False)
            self.body_points[i].setVisible(False)
            self.plotter.addItem(self.body_points[i])

        # bounding boxes
        bbox_vertex = np.random.randint(10, size=(8, 3))
        self.bbox_lines = {}
        self.bbox_connections = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        self.n_box_connect = len(self.bbox_connections)
        for i in range(6):
            for n, pts in enumerate(self.bbox_connections):
                idx = self.n_box_connect * i + n
                self.bbox_lines[idx] = gl.GLLinePlotItem(pos=np.array([bbox_vertex[p] for p in pts]), color='g', width=1, antialias=True)
                self.bbox_lines[idx].setVisible(False)
                self.plotter.addItem(self.bbox_lines[idx])

        # trajectories using points
        self.tracks_points = {}
        for i in range(6):
            self.tracks_points[i] = gl.GLScatterPlotItem(pos=keypoints, size=30, pxMode=False)
            self.tracks_points[i].setVisible(False)
            self.plotter.addItem(self.tracks_points[i])

        # shoulders projected on floor
        self.floor_circles = {}
        for i in range(6):
            self.floor_circles[i] = gl.GLLinePlotItem(pos=np.zeros((100, 3)), width=2, antialias=True)
            self.floor_circles[i].setVisible(False)
            self.plotter.addItem(self.floor_circles[i])

        # user heading
        self.heading = {}
        for i in range(6):
            self.heading[i] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=3, antialias=True)
            self.heading[i].setVisible(False)
            self.plotter.addItem(self.heading[i])

        # instant clusters as circles
        self.cluster_circles = {}
        for i in range(6):
            self.cluster_circles[i] = gl.GLLinePlotItem(pos=np.zeros((100, 3)), width=2, antialias=True)
            self.cluster_circles[i].setVisible(False)
            self.plotter.addItem(self.cluster_circles[i])

        # instant clusters as connections
        self.cluster_connections = {}
        for i in range(6):
            self.cluster_connections[i] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=3, antialias=True, color='w')
            self.cluster_connections[i].setVisible(False)
            self.plotter.addItem(self.cluster_connections[i])

        # long-term clusters as convexHulls
        self.convex_hulls = {}
        for i in range(6):
            self.convex_hulls[i] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=2, antialias=True, color='w')
            self.convex_hulls[i].setVisible(False)
            self.plotter.addItem(self.convex_hulls[i])


    def plot_pose(self, kpts, ids=None):
        if ids is None:
            ids = [0 for ix in range(len(kpts))]

        for idx_pose in range(len(kpts)):
            kpts[idx_pose] = np.vstack((kpts[idx_pose][0, :], np.mean(kpts[idx_pose][[5, 6], :], axis=0), np.mean(kpts[idx_pose][[11, 12], :], axis=0), kpts[idx_pose][5:, :]))
            idx_to_coordinates = {}
            num_kpts = 15
            for idx_joint in range(num_kpts):
                x_px = kpts[idx_pose][idx_joint, 0]
                y_px = kpts[idx_pose][idx_joint, 1]
                z_px = kpts[idx_pose][idx_joint, 2]
                idx_to_coordinates[idx_joint] = x_px, y_px, z_px

            for n, connection in enumerate(self.connections):
                start_idx = connection[0]
                end_idx = connection[1]
                idx_line = self.n_connect * idx_pose + n
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    self.body_lines[idx_line].setData(pos=np.array([idx_to_coordinates[start_idx], idx_to_coordinates[end_idx]]), color=pg.intColor(ids[idx_pose]))
                    self.body_lines[idx_line].setVisible(True)
                else:
                    self.body_lines[idx_line].setVisible(False)

        for idx_pose in range(len(kpts), 5):
            for n, connection in enumerate(self.connections):
                idx_line = self.n_connect * idx_pose + n
                self.body_lines[idx_line].setVisible(False)

        for i in range(2):
            self.body_points[i].setData(pos=np.asarray(kpts))
            self.body_points[i].setVisible(True)

    def plot_bbox(self, kpts, ids=None):
        if ids is None:
            ids = [0 for ix in range(len(kpts))]

        def get_bbox(keypoints):
            itmax = np.amax(keypoints, axis=0)
            itmin = np.amin(keypoints, axis=0)
            return [itmin[i] for i in range(3)] + [itmax[i] for i in range(3)]

        for idx_pose in range(len(kpts)):
            xmin, ymin, zmin, xmax, ymax, zmax = get_bbox(kpts[idx_pose])
            bbox_vertex = np.array([[xmin, ymin, zmin], [xmin, ymax, zmin], [xmax, ymax, zmin], [xmax, ymin, zmin],
                                    [xmin, ymin, zmax], [xmin, ymax, zmax], [xmax, ymax, zmax], [xmax, ymin, zmax]])

            for n, connection in enumerate(self.bbox_connections):
                start_idx = connection[0]
                end_idx = connection[1]
                idx = self.n_box_connect * idx_pose + n
                self.bbox_lines[idx].setData(pos=np.array([bbox_vertex[start_idx], bbox_vertex[end_idx]]), color=pg.glColor(ids[idx_pose]))
                self.bbox_lines[idx].setVisible(True)

        for idx_pose in range(len(kpts), 6):
            for n, connection in enumerate(self.bbox_connections):
                idx = self.n_box_connect * idx_pose + n
                self.bbox_lines[idx].setVisible(False)

    def plot_tracks(self, trajectories, ids):
        for id_track in range(len(ids)):
            #plot using x,y data and zeroing z
            data = np.zeros(trajectories[ids[id_track]].shape)
            data[:, :2] = trajectories[ids[id_track]][:, :2]
            #plot 3d trajectories
            data = trajectories[ids[id_track]][:]
            self.tracks_points[id_track].setData(pos=data, color=pg.intColor(ids[id_track]))
            self.tracks_points[id_track].setVisible(True)

        for id_track in range(len(ids), 6):
            self.tracks_points[id_track].setVisible(False)

    def plot_floor_projection(self, trajectories, ids):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 200.0 * np.cos(theta)
        y = 200.0 * np.sin(theta)
        z = 0 * theta

        for id_track in range(len(ids)):
            mean_point = trajectories[ids[id_track]][-1][:2]
            pts = np.column_stack([mean_point[0] + x, mean_point[1] + y, z])

            self.floor_circles[id_track].setData(pos=pts, color=pg.glColor(ids[id_track]))
            self.floor_circles[id_track].setVisible(True)

        for id_track in range(len(ids), 6):
            self.floor_circles[id_track].setVisible(False)

    def plot_heading(self, trajectories, ids):
        def _get_direction_angle(trajectory):
            if trajectory.shape[0] > 1:
                a = trajectory[-1][:2]                      # actual_pos
                b = trajectory[-2][:2]                      # prev_pos
                c = b + np.array([1.0, 0])
                return math.atan2(a[1] - b[1], a[0] - b[0]) - math.atan2(c[1] - b[1], c[0] - b[0])
            else:
                return 0.0

        for id_track in range(len(ids)):
            angle = _get_direction_angle(trajectories[ids[id_track]])
            center_point = trajectories[ids[id_track]][-1][:2]
            end_point = center_point + 500.0*np.array([np.cos(angle), np.sin(angle)])

            self.heading[id_track].setData(pos=np.vstack((center_point, end_point)), color=pg.glColor(ids[id_track]))
            self.heading[id_track].setVisible(True)

        for id_track in range(len(ids), 6):
            self.heading[id_track].setVisible(False)

    def plot_instant_clusters(self, center_points, counted_labels):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 100.0 * np.cos(theta)
        y = 100.0 * np.sin(theta)
        z = 0 * theta

        lines_drawn = 0
        for element in counted_labels:
            center = center_points[element]
            radio = counted_labels[element]
            pts = np.column_stack([center[0] + x*(radio**2), center[1] + y*(radio**2), z])

            self.cluster_circles[lines_drawn].setData(pos=pts)
            self.cluster_circles[lines_drawn].setVisible(True)
            lines_drawn += 1

        for j in range(lines_drawn, 6):
            self.cluster_circles[j].setVisible(False)

    def plot_clusters_connections(self, center_points, counted_labels, labels, input_data):
        lines_drawn = 0
        for element in counted_labels:
            if counted_labels[element] > 1:
                center_point = center_points[element].reshape((1, 2))
                indexes = np.where(labels == element)
                corresponding_data = input_data[indexes]
                for i in range(corresponding_data.shape[0]):
                    self.cluster_connections[i+lines_drawn].setData(pos=np.vstack((center_point, corresponding_data[i])))
                    self.cluster_connections[i+lines_drawn].setVisible(True)
                lines_drawn += counted_labels[element]

        for i in range(lines_drawn, 6):
            self.cluster_connections[i].setVisible(False)

    def plot_hotspot_clusters(self, n_clusters, labels, input_data):
        for k in range(n_clusters):
            corresponding_idxs = labels == k
            corresponding_data = input_data[corresponding_idxs, :]
            if corresponding_data.size == 0:
                break
            hull = ConvexHull(corresponding_data)
            vertices = corresponding_data[hull.vertices]

            vertices_z0 = np.column_stack([vertices, np.zeros((vertices.shape[0], 1))])
            self.convex_hulls[k].setData(pos=np.vstack((vertices_z0, vertices_z0[0, :])))
            self.convex_hulls[k].setVisible(True)

        for i in range(n_clusters, 6):
            self.convex_hulls[i].setVisible(False)
            self.cluster_circles[i].setVisible(False)

    def plot_traj_correlations(self, trajectories, ids, correlations):
        theta = np.linspace(0, 2 * np.pi, 100)
        x = 200.0 * np.cos(theta)
        y = 200.0 * np.sin(theta)
        z = 0 * theta

        color = 0
        drawn_circles = 0

        for k, v in correlations.items():
            if len(v) > 0:
                v.append(k)
                print(f'circles in indexes {v}')
                for id_track in v:
                    mean_point = trajectories[ids[id_track]][-1][:2]
                    pts = np.column_stack([mean_point[0] + x, mean_point[1] + y, z])

                    self.floor_circles[drawn_circles].setData(pos=pts, color=pg.glColor(ids[id_track]))
                    self.floor_circles[drawn_circles].setVisible(True)
                    drawn_circles += 1
                color += 1

        for j in range(drawn_circles, 6):
            self.floor_circles[j].setVisible(False)

    def clear_assets(self):
        for i in range(6):
            self.heading[i].setVisible(False)
            self.floor_circles[i].setVisible(False)
            self.tracks_points[i].setVisible(False)
            self.cluster_circles[i].setVisible(False)
            self.cluster_connections[i].setVisible(False)
            self.convex_hulls[i].setVisible(False)
            for n, pts in enumerate(self.connections):
                idx = self.n_connect*i + n
                self.body_lines[idx].setVisible(False)
            for n, pts in enumerate(self.bbox_connections):
                idx = self.n_box_connect * i + n
                self.bbox_lines[idx].setVisible(False)
        for i in range(2):
            self.body_points[i].setVisible(False)

    def plot_system(self, system, **kwargs):
        return self.plot_cameras(system, **kwargs)

    def plot_cameras(self, system, scale=800, axes_size=800):
        points = []
        for name, cam in system.get_camera_dict().items():
            # cam center
            cam_center = cam.get_camcenter().reshape(1, 3)
            points.append(cam_center)

            #cam frame
            cam_idx = int(name.split('cam')[1])
            world_coords = cam.project_camera_frame_to_3d([[axes_size, 0, 0], [0, axes_size, 0], [0, 0, axes_size]])
            for i in range(3):
                n = 3 * cam_idx + i
                self.axes[n].setData(pos=np.array([cam_center, world_coords[i].reshape(1, 3)]))
                self.axes[n].setVisible(True)

            #cam body
            if cam.width is None or cam.height is None:
                raise ValueError('Camera width/height must be defined to plot.')
            uv_raw = np.array([[0, 0], [0, cam.height], [cam.width, cam.height], [cam.width, 0], [0, 0]])
            pts3d_near = cam.project_pixel_to_3d_ray(uv_raw, distorted=False, distance=0.2 * scale)
            pts3d_far = cam.project_pixel_to_3d_ray(uv_raw, distorted=False, distance=scale)
            # ring at far depth
            for i in range(4):
                n = 4 * cam_idx + i
                self.far_cam_frame[n].setData(pos=np.array([pts3d_far[i], pts3d_far[i + 1]]))
                self.far_cam_frame[n].setVisible(True)

            # ring at near depth
            for i in range(4):
                n = 4 * cam_idx + i
                self.near_cam_frame[n].setData(pos=np.array([pts3d_near[i], pts3d_near[i + 1]]))
                self.near_cam_frame[n].setVisible(True)

            # connectors
            for i in range(4):
                n = 4 * cam_idx + i
                self.connectors_cam_frame[n].setData(pos=np.array([pts3d_near[i], pts3d_far[i]]))
                self.connectors_cam_frame[n].setVisible(True)

        points = np.asarray(points)
        self.cam_positions.setData(pos=points)
        self.cam_positions.setVisible(True)
        return points
