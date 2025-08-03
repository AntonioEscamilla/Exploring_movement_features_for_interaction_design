from tkinter import *
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.utils import shuffle
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from pyqtgraph.graphicsItems.NonUniformImage import NonUniformImage as NonUniImage
from scipy.ndimage.filters import gaussian_filter
from itertools import cycle
from collections import Counter
from modules.feature_extractor import resample
import scipy.ndimage

from PyQt5 import QtWidgets
import pyqtgraph as pg


actual_stroke = []
all_strokes = []


def draw_clusters(center_points, counted_labels):
    for element in counted_labels:
        point = center_points[element]
        radio = counted_labels[element] * 10
        x1, y1 = (point[0] - radio), (point[1] - radio)
        x2, y2 = (point[0] + radio), (point[1] + radio)
        w.create_oval(x1, y1, x2, y2, fill="#FF0000")


def draw_hull(data, n_clusters, labels):
    for k in range(n_clusters):
        corresponding_idxs = labels == k
        corresponding_data = data[corresponding_idxs, :]
        if corresponding_data.size == 0:
            break
        hull = ConvexHull(corresponding_data)
        points = corresponding_data[hull.vertices]
        for idx in range(points.shape[0]):
            x1, y1 = points[idx][0], points[idx][1]
            if idx == points.shape[0] - 1:
                x2, y2 = points[0][0], points[0][1]
            else:
                x2, y2 = points[idx+1][0], points[idx+1][1]
            w.create_line(x1, y1, x2, y2, fill="#FF00FF")


def draw_heatmap(data):
    ## matplotlib version generating a histogram from scatter points (trajectories)
    H, xedges, yedges = np.histogram2d(data[:, 0], 500-data[:, 1], bins=50)
    H = gaussian_filter(H, sigma=8)
    H = H.T
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111, title='NonUniformImage: interpolated', aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
    im = NonUniformImage(ax, interpolation='bilinear')
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    im.set_data(xcenters, ycenters, H)
    ax.images.append(im)

    ## pyqtgraph version
    p = pg.plot()
    p.setWindowTitle("heatmap")
    image = pg.ImageItem()
    p.addItem(image)
    cm = pg.colormap.get('viridis')
    bar = pg.ColorBarItem(values=(0, 1), colorMap=cm)
    bar.setImageItem(image)
    image.setImage(H.T/np.amax(H[:]))

    ## complete heatmap from trajectories
    heat_img = np.zeros((500, 500))
    data = data.astype(int).tolist()
    for x, y in data:
        heat_img[y, x] += 5

    heat_img = np.rot90(heat_img, k=-1)
    heat_img = gaussian_filter(heat_img, sigma=16)
    heat_img = scipy.ndimage.zoom(heat_img, 0.2, order=3)
    heat_img = heat_img / np.amax(heat_img)

    p = pg.plot()
    p.setWindowTitle("heatmap")
    p.setAspectLocked(True)
    image = pg.ImageItem()
    p.addItem(image)
    cm = pg.colormap.get('viridis')
    bar = pg.ColorBarItem(values=(0, 1), colorMap=cm)
    bar.setImageItem(image)
    image.setImage(heat_img)



def draw_stroke(stroke):
    for point in stroke:
        x1, y1 = (point[0] - 1), (point[1] - 1)
        x2, y2 = (point[0] + 1), (point[1] + 1)
        w.create_oval(x1, y1, x2, y2, fill="#FF0000")


def draw_points_by_label(data, n_clusters, labels):
    colors = ["red", "green", "blue", "cyan", "yellow", "magenta", "white"]
    for k, col in zip(range(n_clusters), colors):
        corresponding_idxs = labels == k
        for x, y in zip(data[corresponding_idxs, 0], data[corresponding_idxs, 1]):
            x1, y1 = (x - 3), (y - 3)
            x2, y2 = (x + 3), (y + 3)
            w.create_oval(x1, y1, x2, y2, fill=col)


def paint(event):
    actual_stroke.append([event.x, event.y])
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill="#00FF00")


def right_click(event):
    points_array = []
    for stroke in all_strokes:
        points_array += stroke
    points_array = shuffle(np.asarray(points_array))
    points_array = points_array / canvas_width

    # bandwidth = estimate_bandwidth(points_array, quantile=0.2, n_samples=50)
    # clustering = MeanShift(bandwidth=bandwidth).fit(points_array)
    # cluster_centers = clustering.cluster_centers_

    clustering = DBSCAN(eps=0.05, min_samples=15).fit(points_array)

    labels = clustering.labels_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print(f'number of clusters: {n_clusters}, clusters: {labels_unique}')
    draw_points_by_label(points_array*canvas_width, n_clusters, labels)
    draw_hull(points_array*canvas_width, n_clusters, labels)
    draw_heatmap(points_array * canvas_width)


def left_up(e):
    global all_strokes
    global actual_stroke

    all_strokes.append(resample(actual_stroke))
    draw_stroke(all_strokes[-1])


def left_down(event):
    global actual_stroke
    actual_stroke = []


def delete_strokes():
    global actual_stroke
    global all_strokes

    w.delete('all')
    t1.delete(0, END)
    actual_stroke = []
    all_strokes = []



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

message = Label(root, text="Right Click for Procrustes Analysis")
message.grid(pady=5, row=3, column=0, columnspan=2)

mainloop()
