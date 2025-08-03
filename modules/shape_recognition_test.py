from tkinter import *
from modules.feature_extractor import procrustes, pathLength, getDistance, resample, convex_hull_feats, distance_geometry, norm_and_stack_feat
from modules.feature_extractor import DBSCAN_predict, stack_feat
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, KMeans
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


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

    new_data = []
    for stroke in all_strokes:
        feat_vect, control_points = distance_geometry(stroke, 2)
        new_data.append(feat_vect)
        draw_control_points(stroke, control_points)
    new_data = np.asarray(new_data)
    # new_data = norm_and_stack_feat(new_data, convex_hull_areas, scaler=scalers[0])
    # new_data = norm_and_stack_feat(new_data, volume_norm_ratios, scaler=scalers[1])
    new_data = stack_feat(new_data, convex_hull_areas)
    print(f'{new_data.shape}')

    #x_data = StandardScaler().fit_transform(x_data)
    # bandwidth = estimate_bandwidth(x_data, quantile=0.5)
    # print(f'bandwidth: {bandwidth}')
    # clustering = MeanShift(bandwidth=bandwidth).fit(x_data)
    clustering = DBSCAN(eps=0.2, min_samples=2).fit(x_data)

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

    x_data = []
    for stroke in all_strokes:
        feat_vect, control_points = distance_geometry(stroke, 2)
        x_data.append(feat_vect)
        draw_control_points(stroke, control_points)
    x_data = np.asarray(x_data)
    # x_data, scale = norm_and_stack_feat(x_data, convex_hull_areas)
    # scalers.append(scale)
    # x_data, scale = norm_and_stack_feat(x_data, volume_norm_ratios)
    # scalers.append(scale)

    x_data = stack_feat(x_data, convex_hull_areas)
    print(f'{x_data.shape}')

    with open('feat_vect_data.npy', 'wb') as f:
        np.save(f, x_data)

    all_strokes = []
    convex_hull_areas = []


def left_up(e):
    global all_strokes
    global actual_stroke
    global convex_hull_areas

    all_strokes.append(actual_stroke)
    mtx, f = convex_hull_feats(actual_stroke)
    convex_hull_areas.append(f)
    # a = np.array(mtx * 500 + 250, np.int32)
    # draw_stroke(a.tolist())


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
