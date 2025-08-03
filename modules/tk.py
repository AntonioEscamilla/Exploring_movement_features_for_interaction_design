from tkinter import *
from modules.qdollar.recognizer import Gesture, Recognizer, Point
import os
import json

# f = open("template.txt", "w+")
points1 = []
points_as_list = []
templates1 = []
templates_as_dicts = []

def paint(event):
    python_green = "#476042"
    # f.write("Point(%d,%d,%d)," %(event.x,event.y, strokeId))
    points1.append(Point(int(event.x), int(event.y), int(strokeId)))
    points_as_list.append([event.x, event.y, strokeId])
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill=python_green)


def right_click(event):
    global points1
    global strokeId
    global points_as_list
    w.delete('all')
    gesture1 = Gesture("", points1)
    res = Recognizer().classify(gesture1, templates1)
    print("$Q = ", res[0].name, res[1], flush=True)
    points1 = []
    points_as_list = []
    strokeId = 1


def increase_strokeId(e):
    global strokeId
    strokeId += 1


def addtemplates():
    global points1
    global points_as_list
    global strokeId
    templates_as_dicts.append({'name': t1.get(), 'points': points_as_list})
    template1 = Gesture(t1.get(), points1)
    templates1.append(template1)
    w.delete('all')
    t1.delete(0, END)
    points1 = []
    points_as_list = []
    strokeId = 1


def save_templates():
    with open('trajectory_data.json', 'w') as outfile:
        json.dump(templates_as_dicts, outfile, indent=4)


def load_templates():
    global templates_as_dicts
    if os.path.exists('trajectory_data.json'):
        with open('trajectory_data.json', 'r') as fp:
            templates_as_dicts = json.load(fp)
        for entry in templates_as_dicts:
            points = [Point(int(x), int(y), int(strokeId)) for x, y, strokeId in entry['points']]
            templates1.append(Gesture(entry['name'], points))
    else:
        print('trajectory_data.json doesnt exist in folder')

# for filename in os.listdir("templates/"):
#     path = os.path.join('templates/', filename)
#     f1 = open(path, 'r')
#     f1 = f1.readlines()
#     points1 = []
#     for line in f1:
#         x, y, strokeId = line.split(" ")
#         points1.append(Point(int(x), int(y), int(strokeId)))
#     template1 = Gesture(filename, points1)
#     templates1.append(template1)

# Tkinter
canvas_width = 500
canvas_height = 300
strokeId = 1

root = Tk()
root.title("Qdollar")
w = Canvas(root, width=canvas_width, height=canvas_height)
w.grid(padx=5, pady=5, row=0, column=0, columnspan=2)
w.bind("<B1-Motion>", paint)
w.bind("<Button-3>", right_click)
w.bind('<ButtonRelease-1>', increase_strokeId)

t1 = Entry(root)
t1.grid(padx=5, pady=5, row=1, column=0, sticky=E)
b1 = Button(root, text="Add to Templates", command=addtemplates)
b1.grid(padx=5, pady=5, row=1, column=1, sticky=W)

b2 = Button(root, text="Save Templates", command=save_templates)
b2.grid(pady=5, row=2, column=0, sticky=E)
b3 = Button(root, text="Load Templates", command=load_templates)
b3.grid(pady=5, row=2, column=1, sticky=W)

message = Label(root, text="Right Click to identify gesture")
message.grid(pady=5, row=3, column=0, columnspan=2)

mainloop()
