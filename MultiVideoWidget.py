###########################################################################################
###                        CODE:       WRITTEN BY: ANTONIO ESCAMILLA                    ###
###                        PROJECT:    MULTI-VIEW MULTIPLE-PEOPLE POSE ESTIMATION       ###
###                                    BASED ON QT DESIGNER                             ###
###                        LICENCE:    MIT OPENSOURCE LICENCE                           ###
###                                                                                     ###
###                            CODE IS FREE TO USE AND MODIFY                           ###
###########################################################################################

from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QGridLayout, QFrame
from PyQt5.QtGui import QImage, QPixmap
from draw_coco_landmarks import draw_landmarks
from ExtrinsicsTransformation import transform_rotation, transform_translation
from ExtrinsicsUtils import cam_pose
from datetime import datetime
import numpy as np
import time
import glob
import os
import cv2
import ntpath
import pickle
import json


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    on_video_end = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.paths = []
        self.names = []
        self.caps = []
        self.n_cams = 0
        self.reached_custom_frame = False
        self.force_loop_abortion = False

    def set_video_path(self, scene_dir):
        # paths_list = glob.glob(os.path.join(videos_dir, '*.mov'))
        # self.paths = [p for idx, p in enumerate(paths_list) if idx < self.n_cams]
        types = ('*.jpg', '*.mp4', '*.mov')
        videos_list = []
        for files in types:
            videos_list.extend(glob.glob(os.path.join(scene_dir + '/videos', files)))
        self.paths = videos_list  # [p for idx, p in enumerate(videos_list) if idx < self.n_cams]
        self.n_cams = len(videos_list)
        self.names = [ntpath.basename(p).split('.')[0] for idx, p in enumerate(videos_list)]
        print(f'Names of cameras in video files: {self.names}')

    def set_first_frame_to_read(self, frame):
        self.caps = []
        for i in range(len(self.names)):
            cap = cv2.VideoCapture(self.paths[i])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            self.caps.append(cap)

    def run(self):
        if len(self.caps) == 0:
            for i in range(len(self.names)):
                cap = cv2.VideoCapture(self.paths[i])
                self.caps.append(cap)

        while self.caps[0].isOpened() and self.caps[1].isOpened() and self.caps[2].isOpened():
            start_t = time.perf_counter()
            result = [cap.read() for cap in self.caps]

            ret_val = np.asarray([r[0] for r in result])
            abort_loop = not (np.prod(ret_val))
            if abort_loop or self.reached_custom_frame or self.force_loop_abortion:
                break

            cv_images = np.asarray([r[1] for r in result])
            self.change_pixmap_signal.emit(cv_images)
            elapsed_t = time.perf_counter() - start_t
            self.spin(0.04 - elapsed_t)
        for cap in self.caps:
            cap.release()
        self.on_video_end.emit()
        self.wait()

    def stop(self):
        """Waits for thread to finish"""
        self.wait()

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""
        time_end = time.time() + seconds
        while time.time() < time_end:
            QApplication.processEvents()


class MultiVideoWidget(QWidget):
    frame_counter_signal = pyqtSignal(bool)

    def __init__(self, parent=None, aspect_ratio=False):
        super(MultiVideoWidget, self).__init__(parent)
        self.frame_counter = 0
        self.screen_width = 0
        self.screen_height = 0
        self.maintain_aspect_ratio = aspect_ratio
        self.n_videos = 4
        self.cameras = {}
        self.calib_data = []
        self.detections = {}
        self.predictions = {}
        self.predicted_ids = {}

        self.gridlayout = QGridLayout(self)
        self.imageLabels = []
        for i in range(self.n_videos):
            self.imageLabels.append(QLabel(self))  # create the labels that holds the images

        for i in range(self.n_videos):
            self.gridlayout.addWidget(self.imageLabels[i], i, 0)

        self.setLayout(self.gridlayout)

        self.thread = VideoThread()  # create the video capture thread
        self.thread.change_pixmap_signal.connect(self.UpdateImage)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        #self.screen_width = self.parent().size().width() // 3
        self.screen_width = 360
        self.screen_height = self.parent().size().height() // self.n_videos
        if self.screen_width % 2 == 1:
            self.screen_width -= 1
        if self.screen_height % 2 == 1:
            self.screen_height -= 1

    @pyqtSlot(np.ndarray)
    def UpdateImage(self, cv_images):
        self.frame_counter += 1

        for i in range(self.thread.n_cams):
            qt_img = self.toQPix(cv_images[i], idx=i)
            self.imageLabels[i].setPixmap(qt_img)

        self.frame_counter_signal.emit(True)

        if self.frame_counter == self.frame_range[1]:
            self.thread.reached_custom_frame = True
            print(f'reach frame number {self.frame_counter}')

    def toQPix(self, frame, idx):
        """Convert from an opencv image to QPixmap"""

        # Draw the pose annotation on the image.
        frame.flags.writeable = True
        cam_frame_key = f'{idx}_{self.frame_counter}'
        if cam_frame_key in self.detections:
            kpts_list = self.detections[cam_frame_key]
            # print(f'cam, frame: {cam_frame_key}')
            for kpts in kpts_list:
                draw_landmarks(frame, kpts['pred'])

        # Keep frame aspect ratio or force resize
        frame = cv2.resize(frame, (self.screen_width, self.screen_height))

        # Add timestamp to camera
        cv2.rectangle(frame, (self.screen_width - 115, 25), (self.screen_width, 50), color=(0, 0, 0), thickness=-1)
        cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (self.screen_width - 110, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), lineType=cv2.LINE_AA)

        # Convert to pixmap and set to video frame
        img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(img)

    def get_image_label(self, idx):
        return self.imageLabels[idx]

    def load_metadata(self, data_dir):
        if self.thread.n_cams == 3:
            self.cameras = self.get_campus_calib_data(data_dir)
        elif self.thread.n_cams == 4:
            self.calib_data = self.get_calib_data(data_dir, self.thread.n_cams)
            self.calib_data = self.transform_calibration_data(self.calib_data, 'Upb')
            self.cameras = self.cameras_dict_for_inference(self.calib_data)

        self.detections = self.get_pred_pose2d(data_dir + '/detections')

        self.tracks_by_frame, self.pose_by_track_and_frame, self.frame_range = self.get_pred_pose3d(
            data_dir + '/detections')
        # self.predictions = self.get_pred_pose3d(data_dir + '/detections')

        self.thread.set_first_frame_to_read(self.frame_range[0])
        self.frame_counter = self.frame_range[0]
        self.thread.reached_custom_frame = False
        self.thread.force_loop_abortion = False
        print(f'Reading detections for {self.n_videos} videos. Detection dict of length = {len(self.detections)}')

    def get_pred_pose2d(self, dataset_root):
        fp = [s for s in glob.glob(os.path.join(dataset_root, '*.pkl')) if 'coco' in s]
        with open(fp[0], "rb") as f:
            detections_2d = pickle.load(f)

        return detections_2d

    def get_pred_pose3d(self, dataset_root):
        # -----> read data using fixed frame range and pose data only <-------
        # fp = [s for s in glob.glob(os.path.join(dataset_root, '*.pkl')) if 'Campus_222' in s]
        # with open(fp[0], "rb") as f:
        #     pose_data = pickle.load(f)
        #
        # pred_3d = {}
        # test_range = [i for i in range(350, 471)] + [i for i in range(650, 751)]
        # for idx, frame in enumerate(test_range):
        #     pred_3d[str(frame)] = [1000*pose.T for pose in pose_data[idx]]
        #
        # return pred_3d

        # -----> read data using frame range in pkl file and pose data only <-------
        # fp = [s for s in glob.glob(os.path.join(dataset_root, '*.pkl')) if 'from_colab' in s]
        # with open(fp[0], "rb") as f:
        #     pose_data = pickle.load(f)
        #
        # pred_3d = {}
        # file = ntpath.basename(fp[0])                   # keep only the file name without path
        # file = os.path.splitext(file)[0]                # keep only the name without the extension
        # start_frame = int(file.split('_')[-2])
        # end_frame = int(file.split('_')[-1])
        # for idx, frame in enumerate(range(start_frame, end_frame)):
        #     pred_3d[str(frame)] = [1000*pose.T for pose in pose_data[idx]]
        #
        # return pred_3d

        # -----> read multiple pkl files and return pose and ID data <-------
        # fp = [s for s in glob.glob(os.path.join(dataset_root, '*.pkl')) if 'tracked_100_500' in s]
        # pred_3d = {}
        # pred_ID = {}
        # for i in range(len(fp)):
        #     file = ntpath.basename(fp[i])               # keep only the file name without path
        #     file = os.path.splitext(file)[0]            # keep only the name without the extension
        #     start_frame = int(file.split('_')[-2])
        #     end_frame = int(file.split('_')[-1])
        #
        #     with open(fp[i], "rb") as f:
        #         data = pickle.load(f)
        #     for idx, frame in enumerate(range(start_frame, end_frame, 1)):
        #         # pred_3d[str(frame)] = [1000 * pose.T for pose in data[idx]]
        #         pred_3d[str(frame)] = [1000*id_pose_tuple[1].T for id_pose_tuple in data[idx]]
        #         pred_ID[str(frame)] = [id_pose_tuple[0] for id_pose_tuple in data[idx]]
        #
        # return pred_3d, pred_ID

        # -----> read tracks by frame and pose by track data <-------
        fp = [s for s in glob.glob(os.path.join(dataset_root, '*.pkl')) if 'tracks_by_frame' in s]
        with open(fp[0], "rb") as f:
            tracks_by_frame, pose_by_track_and_frame = pickle.load(f)
            file = ntpath.basename(fp[0])  # keep only the file name without path
            file = os.path.splitext(file)[0]  # keep only the name without the extension
            start_frame = int(file.split('_')[-2])
            end_frame = int(file.split('_')[-1])
            frame_range = (start_frame, end_frame)

        return tracks_by_frame, pose_by_track_and_frame, frame_range

    @staticmethod
    def get_campus_calib_data(dataset_root):
        json_file = glob.glob(os.path.join(dataset_root, '*.json'))[0]

        with open(json_file) as f:
            cameras = json.load(f)

        for cam_id, cam in cameras.items():
            for k, v in cam.items():
                cameras[cam_id][k] = np.array(v)
            cameras[cam_id]['id'] = cam_id
            cameras[cam_id]['width'] = 360
            cameras[cam_id]['height'] = 288

        return cameras

    @staticmethod
    def get_calib_data(dataset_root, num_cams):
        json_file = glob.glob(os.path.join(dataset_root, '*.json'))[0]

        with open(json_file) as f:
            data = json.load(f)

        multi_cam_calibration_data = []
        for i in range(num_cams):
            calibration_data = {}
            name = f'cam{i + 1}'
            calibration_data['name'] = name
            P = np.zeros((3, 4))
            P[:3, :3] = np.asarray(data['cameras'][name]['K'])
            calibration_data['width'] = data['cameras'][name]['image_size'][0]
            calibration_data['height'] = data['cameras'][name]['image_size'][1]
            calibration_data['P'] = P
            calibration_data['K'] = np.asarray(data['cameras'][name]['K'])
            calibration_data['D'] = np.asarray(data['cameras'][name]['dist']).reshape((5,))
            calibration_data['R'] = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

            keys_list = sorted(data['camera_poses'])
            calibration_data['translation'] = np.asarray(
                data['camera_poses'][keys_list[int(name.split('cam')[1]) - 1]]['T'])
            calibration_data['Q'] = np.asarray(data['camera_poses'][keys_list[int(name.split('cam')[1]) - 1]]['R'])
            multi_cam_calibration_data.append(calibration_data)

        return multi_cam_calibration_data

    @staticmethod
    def transform_calibration_data(multi_cam_calibration_data, dataset='Hexagonos'):
        # ----------- new R,T from chArUco board in ground plane -----------#
        if dataset == 'Upb':
            R1w = np.asarray([[0.99836602, 0.01636879, -0.05474802],
                              [-0.03822196, -0.52092004, -0.85274931],
                              [-0.04247781, 0.85344852, -0.51944322]])
            T1w = np.asarray([[-0.50525486],
                              [0.88358993],
                              [2.63720055]])
        elif dataset == 'Hexagonos':
            R1w = np.asarray([[0.906657, -0.41233829, 0.08916402],
                              [-0.12746963, -0.46923809, -0.87382327],
                              [0.40214994, 0.78089228, -0.47799861]])
            T1w = np.asarray([[-0.31621033],
                              [0.99795041],
                              [2.26775274]])
        else:
            return multi_cam_calibration_data

        multi_cam_calibration_data[0]['translation'] = T1w
        multi_cam_calibration_data[0]['Q'] = R1w

        # ----------- new R,T based on chArUco board in ground plane -----------#
        # read calibration data: cam1 as seen from cam2
        R21 = multi_cam_calibration_data[1]['Q']
        T21 = multi_cam_calibration_data[1]['translation']
        # obtain inverse transformation: cam2 as seen from cam 1
        T12 = cam_pose(R21, T21)
        R12 = R21.T
        # transform operation: ground plane as seen from cam2
        multi_cam_calibration_data[1]['translation'] = transform_translation(T1w, R12, T12).reshape(-1, 1)
        multi_cam_calibration_data[1]['Q'] = transform_rotation(R1w, R12)

        # ----------- new R,T based on chArUco board in ground plane -----------#
        # read calibration data: cam1 as seen from cam3
        R31 = multi_cam_calibration_data[2]['Q']
        T31 = multi_cam_calibration_data[2]['translation']
        # obtain inverse transformation: cam2 as seen from cam 1
        T13 = cam_pose(R31, T31)
        R13 = R31.T
        # transform operation: ground plane as seen from cam3
        multi_cam_calibration_data[2]['translation'] = transform_translation(T1w, R13, T13).reshape(-1, 1)
        multi_cam_calibration_data[2]['Q'] = transform_rotation(R1w, R13)

        # ----------- new R,T based on chArUco board in ground plane -----------#
        R41 = multi_cam_calibration_data[3]['Q']
        T41 = multi_cam_calibration_data[3]['translation']
        # obtain inverse transformation: cam2 as seen from cam 1
        T14 = cam_pose(R41, T41)
        R14 = R41.T
        # transform operation: ground plane as seen from cam2
        multi_cam_calibration_data[3]['translation'] = transform_translation(T1w, R14, T14).reshape(-1, 1)
        multi_cam_calibration_data[3]['Q'] = transform_rotation(R1w, R14)

        return multi_cam_calibration_data

    @staticmethod
    def cameras_dict_for_inference(multi_cam_calibration_data):
        cameras = {}
        for calib_data in multi_cam_calibration_data:
            key = str(int(calib_data['name'].strip('cam')) - 1)
            cam = {'id': key,
                   'width': calib_data['width'],
                   'height': calib_data['height'],
                   'R': calib_data['Q'],
                   'T': calib_data['Q'].T @ (-calib_data['translation'] * 1000.0),
                   'fx': calib_data['K'][0, 0],
                   'fy': calib_data['K'][1, 1],
                   'cx': calib_data['K'][0, 2],
                   'cy': calib_data['K'][1, 2],
                   # 'k': calib_data['D'][:3].reshape(3, 1),
                   # 'p': calib_data['D'][3:].reshape(2, 1)}
                   'k': np.zeros((3, 1)),
                   'p': np.zeros((2, 1))}
            cameras[key] = cam
        return cameras
