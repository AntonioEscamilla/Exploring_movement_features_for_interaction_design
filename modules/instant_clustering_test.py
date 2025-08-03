from tkinter import *
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from collections import Counter


points = []
x_data = []


def draw_clusters(center_points, counted_labels):
    for element in counted_labels:
        point = center_points[element]
        radio = counted_labels[element] * 10
        x1, y1 = (point[0] - radio), (point[1] - radio)
        x2, y2 = (point[0] + radio), (point[1] + radio)
        w.create_oval(x1, y1, x2, y2, fill="#FF0000")


def draw_hull(points):
    for idx in range(points.shape[0]):
        x1, y1 = points[idx][0], points[idx][1]
        if idx == points.shape[0] - 1:
            x2, y2 = points[0][0], points[0][1]
        else:
            x2, y2 = points[idx+1][0], points[idx+1][1]
        w.create_line(x1, y1, x2, y2, fill="#FF00FF")


def draw_points_connection(center_point, corresponding_data):
    x1, y1 = center_point[0, 0], center_point[0, 1]
    for i in range(corresponding_data.shape[0]):
        x2, y2 = corresponding_data[i][0], corresponding_data[i][1]
        w.create_line(x1, y1, x2, y2, fill="#00FF00")


def right_click(event):
    points_array = np.asarray(points) / canvas_width

    bandwidth = estimate_bandwidth(points_array, quantile=0.5)
    clustering = MeanShift(bandwidth=0.5).fit(points_array)
    print(clustering.labels_)

    center_points = clustering.cluster_centers_ * canvas_width
    counted_labels = Counter(clustering.labels_)
    draw_clusters(center_points, counted_labels)

    for element in counted_labels:
        if counted_labels[element] > 1:
            print(f'label repetido {element}, {counted_labels[element]} veces')
            center_point = center_points[element].reshape((1, 2))
            indexes = np.where(clustering.labels_ == element)
            corresponding_data = points_array[indexes] * canvas_width
            print(center_point.shape)
            print(corresponding_data.shape)
            # all_points = np.vstack((center_point, corresponding_data))
            # print(all_points)
            # hull = ConvexHull(all_points)
            # draw_hull(hull.points)
            draw_points_connection(center_point, corresponding_data)

    # plt.figure()
    # plt.clf()
    # plt.plot(x_data[:, 0], x_data[:, 1], "o")
    # plt.show()


def left_down(event):
    global points

    points.append([event.x, event.y])
    x1, y1 = (event.x - 3), (event.y - 3)
    x2, y2 = (event.x + 3), (event.y + 3)
    w.create_oval(x1, y1, x2, y2, fill="#00FF00")


def delete_strokes():
    global points

    w.delete('all')
    t1.delete(0, END)
    points = []


# Tkinter
canvas_width = 500
canvas_height = 500

root = Tk()
root.title("Procrustes Analysis")
w = Canvas(root, width=canvas_width, height=canvas_height)
w.grid(padx=5, pady=5, row=0, column=0, columnspan=3)
w.bind("<Button-3>", right_click)
w.bind('<ButtonPress-1>', left_down)


t1 = Entry(root)
t1.grid(padx=5, pady=5, row=1, column=0, sticky=E)
b1 = Button(root, text="Delete All Strokes", command=delete_strokes)
b1.grid(padx=5, pady=5, row=1, column=1, sticky=W)

message = Label(root, text="Right Click for Procrustes Analysis")
message.grid(pady=5, row=3, column=0, columnspan=2)

mainloop()
