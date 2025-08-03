###########################################################################################
###                        CODE:       WRITTEN BY: ANTONIO ESCAMILLA                    ###
###                        PROJECT:    MULTI-VIEW MULTIPLE-PEOPLE POSE ESTIMATION       ###
###                                    BASED ON QT DESIGNER                             ###
###                        LICENCE:    MIT OPENSOURCE LICENCE                           ###
###                                                                                     ###
###                            CODE IS FREE TO USE AND MODIFY                           ###
###########################################################################################


from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QVector3D
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QObject
import pyqtgraph as pg
from MultiVideoWidget import MultiVideoWidget
from MultiCamSystem import build_multi_camera_system
from View3dWidget import View3dWidget
from modules.pose import Pose, eliminate_duplicated_poses, track_poses_by_distance
from modules.tracker import tracking, Track
from modules.one_euro_filter import OneEuroFilter
from modules.feature_extractor import TrajectoryCorrelations, DynamicInteraction, TrajectoryImitation, InstantPositionCluster, LongTermPosCluster, TrajectoryShapeClassification, TrajectoryShapeKNNClassification, thresholder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import sys
import numpy as np
import math
from dvg_ringbuffer import RingBuffer
from modules.qdollar.recognizer import Gesture, Recognizer, Point
import json
import os
import time
import glob

from pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage as NonUniImage
from scipy.ndimage.filters import gaussian_filter
from pyqtgraph.Qt import QtGui


class App(QWidget):
    update_tracks_signal = pyqtSignal(list)

    @staticmethod
    def prepare_calibration_data(calib_data):
        multi_cam_calibration_data = []
        for cam_id, camera in calib_data.items():
            calibration_data = {}
            name = camera['id']
            calibration_data['name'] = 'cam' + name
            K = np.zeros((3, 3))
            K[0, 0] = camera['fx']
            K[1, 1] = camera['fy']
            K[2, 2] = 1.0
            K[0, 2] = camera['cx']
            K[1, 2] = camera['cy']
            P = np.zeros((3, 4))
            P[:3, :3] = K
            calibration_data['height'] = camera['height']
            calibration_data['width'] = camera['width']
            calibration_data['P'] = P
            calibration_data['K'] = K
            calibration_data['D'] = np.vstack((camera['k'], camera['p'])).reshape((5,))
            calibration_data['R'] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

            calibration_data['translation'] = camera['R'] @ -camera['T']
            calibration_data['Q'] = camera['R']
            multi_cam_calibration_data.append(calibration_data)

        return multi_cam_calibration_data

    @staticmethod
    def plot_heatmap(cam_centers, input_data, curves, image):
        '''generate image of size according to camera positions'''
        # img_size = int(self.max_floor_distance / 100)
        # heat_img = np.zeros((2 * img_size, 2 * img_size))
        size_x_min = abs(np.amin(cam_centers[:, 0]))
        size_x_max = np.amax(cam_centers[:, 0])
        size_y_min = abs(np.amin(cam_centers[:, 1]))
        size_y_max = np.amax(cam_centers[:, 1])
        size_x = int((size_x_max + size_x_min) / 100)
        size_y = int((size_y_max + size_y_min) / 100)
        heat_img = np.zeros((size_y, size_x))

        input_data = (input_data + np.array([[size_x_min, size_y_min]])) / 100
        # input_data = (input_data / 100) + img_size
        data = input_data.astype(int).tolist()
        for x, y in data:
            if 0 <= x < size_x and 0 <= y < size_y:
                heat_img[y, x] += 5

        heat_img = np.rot90(heat_img, k=-1)
        heat_img = np.fliplr(heat_img)
        heat_img = gaussian_filter(heat_img, sigma=8)
        heat_img = heat_img / np.amax(heat_img)

        image.setImage(heat_img)
        for c in curves:
            c.setData(heat_img)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-view Multiple-people Pose Estimation")
        self.resize(1300, 1000)
        self.setFixedSize(1300, 1000)

        self.btn1 = QPushButton('Run Multi-view Video')
        self.btn1.clicked.connect(self.run_video)
        self.btn2 = QPushButton('Stop Video Playback')
        self.btn2.clicked.connect(self.run_pose_detection)
        self.cb = QComboBox()
        self.cb.addItems(["00-Pose", "01-Bounding Box", "02-Trajectory", "03-Heading", "04-Instant Clustering", "05- Long-term Clustering (Hotspots)", "06-Trajectory Similarity", "07-Correlations across Movement Patterns"])
        self.cb.currentIndexChanged.connect(self.plot_mode_change)

        # multi-view video widget
        self.multi_video = MultiVideoWidget()
        self.multi_video.frame_counter_signal.connect(self.newMultiVideoFrame)
        self.multi_video.thread.on_video_end.connect(self.video_ended)
        # 3d plot widget
        self.plotter = View3dWidget()

        # heatmap plot widget and isocurves
        self.heatmap_widget = pg.PlotWidget()
        self.heatmap_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.heatmap_img = pg.ImageItem()
        self.heatmap_widget.addItem(self.heatmap_img)
        cm = pg.colormap.get('viridis')
        bar = pg.ColorBarItem(values=(0, 1), colorMap=cm)
        bar.setImageItem(self.heatmap_img)

        self.curves = []
        levels = np.linspace(0, 1.0, 3)
        for i in range(len(levels)):
            v = levels[i]
            c = pg.IsocurveItem(level=v, pen=(i, len(levels) * 1.5))    # generate isocurve with automatic color selection
            c.setParentItem(self.heatmap_img)                           # make sure isocurve is always correctly displayed over image
            c.setZValue(10)
            self.curves.append(c)

        # event signal plot widget
        self.signal_plotter = pg.PlotWidget(name='Plot1')
        self.signal_plotter.setVisible(False)
        self.signal_plotter.setYRange(0, 1.0)
        self.plot = self.signal_plotter.plot()
        self.plot.setPen((200, 200, 100))
        self.plot2 = self.signal_plotter.plot()
        self.plot2.setPen((200, 0, 0))

        ## Create a grid layout to manage the widgets size and position
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        ## Add widgets to the layout in their proper positions
        self.layout.addWidget(self.multi_video, 0, 0, 3, 1)
        self.layout.addWidget(self.plotter, 0, 1, 3, 3)
        self.layout.addWidget(self.heatmap_widget, 0, 4, 1, 1)
        self.layout.addWidget(self.signal_plotter, 3, 0, 1, 4)
        self.layout.addWidget(self.btn1, 4, 0, 1, 1)
        self.layout.addWidget(self.btn2, 4, 1, 1, 1)
        self.layout.addWidget(self.cb, 4, 3, 1, 1)

        self.heatmap_widget.resize(300, 300)
        self.heatmap_widget.setVisible(False)
        self.plot_mode = 0

        # init variables and objects
        self.correlation_plots = {}
        self.correlation_data = {}
        self.correlation_filters = {}

        for i in range(6):
            self.correlation_plots[i] = self.signal_plotter.plot()

        # self.correlation_data.append(RingBuffer(500, dtype=np.float))
        # self.correlation_filters.append(OneEuroFilter(beta=1e-5))

        self.previous_poses = []

        self.trajectories = {}
        self.track_filters = {}

        self.feature_data = RingBuffer(500, dtype=np.float)
        self.filter = OneEuroFilter(beta=1e-5)
        self.filtered_data = RingBuffer(500, dtype=np.float)
        self.filter2 = OneEuroFilter(beta=1e-5)
        self.filtered_data2 = RingBuffer(500, dtype=np.float)



        # DBSCAN clustering
        with open('./modules/feat_vect_data.npy', 'rb') as f:
            x_data = np.load(f)

        self.clustering = DBSCAN(eps=0.2, min_samples=2).fit(x_data)
        y_data = self.clustering.labels_

        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(x_data, y_data)

    def run_pose_detection(self):
        self.multi_video.thread.force_loop_abortion = True

    def plot_mode_change(self, i):
        self.plot_mode = i
        self.plotter.clear_assets()
        if self.plot_mode == 6:
            self.signal_plotter.setVisible(True)
        else:
            self.signal_plotter.setVisible(False)
        if self.plot_mode == 5:
            self.heatmap_widget.setVisible(True)
            self.heatmap_widget.resize(300, 300)
            self.plotter.plotter.setCameraPosition(pos=QVector3D(1000, 1000, 0), distance=7000, elevation=90, azimuth=-90)
        elif self.plot_mode == 7:
            self.heatmap_widget.setVisible(False)
            self.plotter.plotter.setCameraPosition(distance=9000, elevation=90, azimuth=-90)
        else:
            self.heatmap_widget.setVisible(False)
            self.plotter.plotter.setCameraPosition(distance=15000, elevation=45, azimuth=-135)

    def run_video(self):
        cb_index = self.cb.currentIndex()
        self.cb.setCurrentIndex(5)
        if cb_index != 5:
            self.heatmap_widget.setVisible(False)
        self.multi_video.thread.force_loop_abortion = True
        scene_dir = QFileDialog.getExistingDirectory()
        self.multi_video.thread.set_video_path(scene_dir)
        self.multi_video.load_metadata(scene_dir)

        multi_cam_calibration_data = self.prepare_calibration_data(self.multi_video.cameras)
        self.camera_system = build_multi_camera_system(multi_cam_calibration_data)
        cam_centers = self.plotter.plot_system(self.camera_system, scale=300, axes_size=300)
        self.cam_centers = cam_centers[:, :, :-1].reshape(-1, 2)
        self.max_floor_distance = np.amax(np.abs(cam_centers[:, :, :-1]))
        self.dist_offset = np.amax(cam_centers)

        self.multi_video.thread.start()
        self.plotter.clear_assets()
        self.cb.setCurrentIndex(cb_index)

    def video_ended(self):
        if self.multi_video.thread.reached_custom_frame:
            print('video reached custom frame')
            self.plotter.clear_assets()
        elif self.multi_video.thread.force_loop_abortion:
            print('video playback stopped by user')
        else:
            print('End of Video')
            self.plotter.clear_assets()

        self.trajectories = {}
        self.track_filters = {}
        self.previous_poses = []

        self.feature_data = RingBuffer(500, dtype=np.float)
        self.filter = OneEuroFilter(beta=1e-5)
        self.filtered_data = RingBuffer(500, dtype=np.float)
        self.filter2 = OneEuroFilter(beta=1e-5)
        self.filtered_data2 = RingBuffer(500, dtype=np.float)


    def newMultiVideoFrame(self):
        # ------> use pose data to plot user in 3d space <------
        # frame_as_idx = str(self.multi_video.frame_counter - 1)
        # print(f'frame number as index for inference: ' + frame_as_idx)
        # if frame_as_idx in self.multi_video.predictions:
        #     pose_entries = self.multi_video.predictions[frame_as_idx]
        #     #id_entries = self.multi_video.predicted_ids[frame_as_idx]
        #     if len(pose_entries) > 0:
        #         #self.plotter.plot_bbox([pose.bbox for pose in current_poses], [pose.color for pose in current_poses])
        #         self.plotter.plot_pose(pose_entries, None)


            # current_poses = [Pose(pose) for pose in pose_entries]
            # eliminate_duplicated_poses(current_poses)
            # track_poses_by_distance(self.previous_poses, current_poses, threshold=300)
            # self.previous_poses = current_poses

            # if len(current_poses) > 0:
            #     #self.plotter.plot_bbox([pose.bbox for pose in current_poses], [pose.color for pose in current_poses])
            #     self.plotter.plot_pose([pose.keypoints for pose in current_poses])

        # ------> use track data to plot user in 3d space <------
        frame_as_idx = self.multi_video.frame_counter - 1
        current_poses = []

        if len(self.multi_video.tracks_by_frame[frame_as_idx]) > 0:
            tracks_on_frame = self.multi_video.tracks_by_frame[frame_as_idx]
            for tid in tracks_on_frame:
                p = self.multi_video.pose_by_track_and_frame[tid, frame_as_idx]
                current_poses.append(p)
                if tid not in self.trajectories:
                    self.trajectories[tid] = RingBuffer(50, dtype=(np.float, 3))
                    self.track_filters[tid] = [OneEuroFilter(beta=1e-5), OneEuroFilter(beta=1e-5), OneEuroFilter(beta=1e-5)]
                mean_point = np.mean(p[[5, 6], :], axis=0)
                x_f = self.track_filters[tid][0](mean_point[0])
                y_f = self.track_filters[tid][1](mean_point[1])
                z_f = self.track_filters[tid][2](mean_point[2])
                self.trajectories[tid].append(np.asarray([x_f, y_f, z_f]))

            if self.plot_mode == 0:
                self.plotter.plot_pose(current_poses, tracks_on_frame)
            elif self.plot_mode == 1:
                self.plotter.plot_bbox(current_poses, tracks_on_frame)
            elif self.plot_mode == 2:
                self.plotter.plot_tracks(self.trajectories, tracks_on_frame)
            elif self.plot_mode == 3:
                self.plotter.plot_heading(self.trajectories, tracks_on_frame)
                self.plotter.plot_floor_projection(self.trajectories, tracks_on_frame)
            elif self.plot_mode == 4:
                center_points, counted_labels, labels, input_data = InstantPositionCluster(self.trajectories, tracks_on_frame, self.max_floor_distance)
                self.plotter.plot_clusters_connections(center_points, counted_labels, labels, input_data)
                self.plotter.plot_instant_clusters(center_points, counted_labels)
                self.plotter.plot_floor_projection(self.trajectories, tracks_on_frame)
            elif self.plot_mode == 5:
                n_clusters, labels, input_data = LongTermPosCluster(self.trajectories, tracks_on_frame, self.max_floor_distance)
                self.plotter.plot_tracks(self.trajectories, tracks_on_frame)
                self.plot_heatmap(self.cam_centers, input_data, self.curves, self.heatmap_img)
            elif self.plot_mode == 6:
                self.plotter.plot_tracks(self.trajectories, tracks_on_frame)
                similarity_sum, user_correlations = TrajectoryImitation(self.trajectories, tracks_on_frame, rotation_invariant=True)
                self.filtered_data2.append(thresholder(self.filter2(similarity_sum), 0.95))
                self.plot2.setData(self.filtered_data2[:])
            elif self.plot_mode == 7:
                self.plotter.plot_tracks(self.trajectories, tracks_on_frame)
                similarity_sum, user_correlations = TrajectoryCorrelations(self.trajectories, tracks_on_frame)
                self.plotter.plot_traj_correlations(self.trajectories, tracks_on_frame, user_correlations)



            # self.feature_data.append(DynamicInteraction(self.trajectories, tracks_on_frame, out_format='sum'))
            # self.plot.setData(self.feature_data[:])
            # self.filtered_data.append(self.filter(DynamicInteraction(self.trajectories, tracks_on_frame, out_format='sum')))
            # self.plot.setData(self.filtered_data[:])



            # similarity_sum, user_correlations = TrajectoryImitation(self.trajectories, tracks_on_frame)
            # self.filtered_data.append(thresholder(self.filter(similarity_sum), 0.7))
            # self.plot.setData(self.filtered_data[:])

            #TrajectoryShapeKNNClassification(self.knn_classifier, self.trajectories, tracks_on_frame)

            # for tid in tracks_on_frame:
            #     if tid not in self.correlation_data:
            #         self.correlation_data[tid] = RingBuffer(500, dtype=np.float)
            #         self.correlation_data[tid].extend([0] * 500)
            #         self.correlation_filters[tid] = OneEuroFilter(beta=1e-5)
            #     u_c_filtered = self.correlation_filters[tid](tid/3.0)
            #     self.correlation_data[tid].append(u_c_filtered)
            # for i in range(len(tracks_on_frame)):
            #     self.correlation_plots[i].setPen(pg.intColor(tracks_on_frame[i]))
            #     self.correlation_plots[i].setData(self.correlation_data[tracks_on_frame[i]][:])

            # for i in range(len(user_correlations)):
            #     u_c = user_correlations[i] / 3.0
            #     u_c_filtered = self.correlation_filters[i](u_c)
            #     self.correlation_data[i].append(u_c_filtered)
            #     self.correlation_plots[i].setData(self.correlation_data[i][:])



if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
