from tkinter import *
from modules.feature_extractor import procrustes, pathLength, getDistance, resample, convex_hull_feats, distance_geometry, norm_and_stack_feat
from modules.feature_extractor import DBSCAN_predict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

import matplotlib.pyplot as plt
import math
import cv2

actual_stroke = []
all_strokes = []
x_data = []
convex_hull_areas = []
volume_norm_ratios = []
scalers = []


def draw_stroke(stroke):
    for point in stroke:
        x1, y1 = (point[0] - 1), (point[1] - 1)
        x2, y2 = (point[0] + 1), (point[1] + 1)
        w.create_oval(x1, y1, x2, y2, fill="#FF0000")


def draw_control_points(points, control):
    for idx in control:
        x1, y1 = (points[idx][0] - 2), (points[idx][1] - 2)
        x2, y2 = (points[idx][0] + 2), (points[idx][1] + 2)
        w.create_oval(x1, y1, x2, y2, fill="#0000FF")


def draw_hull(points, control):
    for idx in range(len(control)):
        x1, y1 = points[control[idx]][0], points[control[idx]][1]
        if idx == len(control) - 1:
            x2, y2 = points[control[0]][0], points[control[0]][1]
        else:
            x2, y2 = points[control[idx+1]][0], points[control[idx+1]][1]
        w.create_line(x1, y1, x2, y2, fill="#FF00FF")


def paint(event):
    actual_stroke.append([event.x, event.y])
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill="#00FF00")


def right_click(event):
    global actual_stroke
    global all_strokes
    global x_data
    global convex_hull_areas
    global volume_norm_ratios

    # num_strokes = len(all_strokes)
    # similarity_matrix = np.zeros((num_strokes, num_strokes))
    # for i in range(num_strokes - 1):
    #     for j in range(i + 1, num_strokes):
    #         similarity_matrix[i, j] = 1.0 - procrustes(all_strokes[i], all_strokes[j])[2]
    # ssq_similarity = np.sum(similarity_matrix ** 2) / (math.factorial(num_strokes)/(math.factorial(2) * math.factorial(num_strokes-2)))
    #
    # #mtx1, mtx2, disparity = procrustes(all_strokes[0], all_strokes[1])
    # print(f'similarity: {ssq_similarity}')
    # print(f'sim_matrix: {similarity_matrix}')

    # mtx1, mtx2, similarity = procrustes(all_strokes[0], all_strokes[1], run_ortho_procrustes=True)
    #
    # w.delete('all')
    # actual_stroke = []
    # all_strokes = []
    # pts1 = np.array(mtx1 * 500 + 250, np.int32)
    # pts2 = np.array(mtx2 * 500 + 250, np.int32)
    # draw_stroke(pts1)
    # draw_stroke(pts2)
    # print(f'similarity: {similarity}')

    new_data = []
    for stroke in all_strokes:
        feat_vect, control_points = distance_geometry(stroke, 2)
        new_data.append(feat_vect)
        draw_control_points(stroke, control_points)
    new_data = np.asarray(new_data)
    new_data = norm_and_stack_feat(new_data, convex_hull_areas, scaler=scalers[0])
    new_data = norm_and_stack_feat(new_data, volume_norm_ratios, scaler=scalers[1])
    print(f'{new_data.shape}')

    #x_data = StandardScaler().fit_transform(x_data)
    # bandwidth = estimate_bandwidth(x_data, quantile=0.5)
    # print(f'bandwidth: {bandwidth}')
    # clustering = MeanShift(bandwidth=bandwidth).fit(x_data)
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(x_data)

    print(f'shape of input features: {x_data.shape}')
    print(f'labels: {clustering.labels_}')
    for i in range(new_data.shape[0]):
        predict = DBSCAN_predict(clustering, new_data[i])
        print(f'predicted clusters: {predict}')

    plt.figure()
    plt.clf()
    plt.plot(x_data[:, 0], x_data[:, 1], "o")
    plt.show()


def fit_model():
    global all_strokes
    global x_data
    global convex_hull_areas
    global volume_norm_ratios
    global scalers

    x_data = []
    scalers = []
    for stroke in all_strokes:
        feat_vect, control_points = distance_geometry(stroke, 2)
        x_data.append(feat_vect)
        draw_control_points(stroke, control_points)
    x_data = np.asarray(x_data)
    x_data, scale = norm_and_stack_feat(x_data, convex_hull_areas)
    scalers.append(scale)
    x_data, scale = norm_and_stack_feat(x_data, volume_norm_ratios)
    scalers.append(scale)
    print(f'{x_data.shape}')

    all_strokes = []
    convex_hull_areas = []
    volume_norm_ratios = []


def left_up(e):
    global all_strokes
    global actual_stroke
    global convex_hull_areas
    global volume_norm_ratios

    #all_strokes.append(resample(actual_stroke))
    all_strokes.append(actual_stroke)
    draw_stroke(all_strokes[-1])
    f1, f2 = convex_hull_feats(actual_stroke)
    convex_hull_areas.append(f1)
    volume_norm_ratios.append(f2)


def left_down(e):
    global actual_stroke
    actual_stroke = []


def delete_strokes():
    global actual_stroke
    global all_strokes
    global convex_hull_areas
    global volume_norm_ratios

    w.delete('all')
    t1.delete(0, END)
    actual_stroke = []
    all_strokes = []
    convex_hull_areas = []
    volume_norm_ratios = []


# Tkinter
canvas_width = 500
canvas_height = 500

root = Tk()
root.title("Procrustes Analysis")
w = Canvas(root, width=canvas_width, height=canvas_height)
w.grid(padx=5, pady=5, row=0, column=0, columnspan=3)
w.bind("<B1-Motion>", paint)
w.bind("<Button-3>", right_click)
w.bind('<ButtonPress-1>', left_down)
w.bind('<ButtonRelease-1>', left_up)

t1 = Entry(root)
t1.grid(padx=5, pady=5, row=1, column=0, sticky=E)
b1 = Button(root, text="Delete All Strokes", command=delete_strokes)
b1.grid(padx=5, pady=5, row=1, column=1, sticky=W)
b2 = Button(root, text="Fit Model", command=fit_model)
b2.grid(padx=5, pady=5, row=1, column=2, sticky=W)

message = Label(root, text="Right Click for Procrustes Analysis")
message.grid(pady=5, row=3, column=0, columnspan=2)

mainloop()

# canvas = np.full([400, 400], 255).astype('uint8')
# canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

# color = (255, 0, 0)
# pts = mtx1.reshape((-1, 1, 2))
# pts = np.array(pts * 200 + 200, np.int32)
# cv2.polylines(canvas, [pts], True, color, 2)
#
# color2 = (0, 255, 0)
# pts2 = mtx2.reshape((-1, 1, 2))
# pts2 = np.array(pts2 * 200 + 200, np.int32)
# cv2.polylines(canvas, [pts2], True, color2, 2)
# cv2.imshow('', canvas)
# cv2.waitKey()
