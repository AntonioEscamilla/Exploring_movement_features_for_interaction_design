from tkinter import *
from modules.dollarone.recognizer import Recognizer
from modules.dollarone.template import Template

import os
import json


actual_stroke = []
all_strokes = []
recognizer = Recognizer()


def paint(event):
    python_green = "#476042"
    actual_stroke.append([event.x, event.y])
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    w.create_oval(x1, y1, x2, y2, fill=python_green)


def right_click(event):
    global actual_stroke
    global all_strokes

    for stroke in all_strokes:
        matched_template, score = recognizer.recognize(stroke)
        print(f'output: {matched_template.name}, score: {score}')
    w.delete('all')
    actual_stroke = []
    all_strokes = []


def left_up(e):
    global all_strokes
    global actual_stroke

    all_strokes.append(actual_stroke)
    for stroke in all_strokes:
        print(len(stroke))


def left_down(e):
    global actual_stroke
    actual_stroke = []


def add_template():
    global actual_stroke
    global all_strokes

    name = t1.get()
    print(f'adding template: {name} with stroke size: {len(actual_stroke)}')
    recognizer.addTemplate(Template(name, actual_stroke))

    w.delete('all')
    t1.delete(0, END)
    actual_stroke = []
    all_strokes = []



# def save_templates():
#     with open('trajectory_data.json', 'w') as outfile:
#         json.dump(templates_as_dicts, outfile, indent=4)
#
#
# def load_templates():
#     global templates_as_dicts
#     if os.path.exists('trajectory_data.json'):
#         with open('trajectory_data.json', 'r') as fp:
#             templates_as_dicts = json.load(fp)
#         for entry in templates_as_dicts:
#             points = [Point(int(x), int(y), int(strokeId)) for x, y, strokeId in entry['points']]
#             templates1.append(Gesture(entry['name'], points))
#     else:
#         print('trajectory_data.json doesnt exist in folder')


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
w.bind('<ButtonPress-1>', left_down)
w.bind('<ButtonRelease-1>', left_up)

t1 = Entry(root)
t1.grid(padx=5, pady=5, row=1, column=0, sticky=E)
b1 = Button(root, text="Add to Templates", command=add_template)
b1.grid(padx=5, pady=5, row=1, column=1, sticky=W)

# b2 = Button(root, text="Save Templates", command=save_templates)
# b2.grid(pady=5, row=2, column=0, sticky=E)
# b3 = Button(root, text="Load Templates", command=load_templates)
# b3.grid(pady=5, row=2, column=1, sticky=W)

message = Label(root, text="Right Click to identify gesture")
message.grid(pady=5, row=3, column=0, columnspan=2)

mainloop()