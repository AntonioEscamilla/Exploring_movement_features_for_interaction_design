import os
import numpy as np
import json
import modules.dollarN as dN
import tkinter as tk

# _______________________________________________________________________________
# Globals
dNr = dN.recognizer()
m_window = tk.Tk()
m_canvas = None
result_txt = None
drawn_strokes = []
cb_var_1 = tk.IntVar()
cb_var_2 = tk.IntVar()


# _______________________________________________________________________________
def where_json(file_name):
    return os.path.exists(file_name)


if where_json('trajectory_data.json'):
    with open('trajectory_data.json', 'r') as fp:
        data = json.load(fp)
    for gesture in data:
        dNr.add_gesture(gesture['label'], gesture['strokes'])
else:
    data = []
    with open('trajectory_data.json', 'w') as outfile:
        json.dump(data, outfile)


# _______________________________________________________________________________
# Tkinter management
def c_boxes():
    dNr.set_rotation_invariance(cb_var_1.get())
    dNr.set_same_nb_strokes(cb_var_2.get())


def recognize():
    if len(drawn_strokes):
        res = dNr.recognize(drawn_strokes)
        txt = res['name'] + ' (' + str(res['value']) + ')'
        result_txt.configure(text=txt)
    clean()


def drawing(event):
    h = m_canvas.winfo_height()
    drawn_strokes[-1].append([float(event.x), float(h - event.y)])
    m_canvas.create_line(drawn_strokes[-1][-2][0],
                         h - drawn_strokes[-1][-2][1],
                         drawn_strokes[-1][-1][0],
                         h - drawn_strokes[-1][-1][1])


def start_drawing(event):
    drawn_strokes.append([[float(event.x), float(m_canvas.winfo_height() - event.y)]])


def stop_drawing(event):
    # print(drawn_strokes)
    pass


def clean():
    global drawn_strokes
    m_canvas.delete("all")
    drawn_strokes = []


def addGesture():
    label = input_txt.get("1.0", "end-1c")

    if len(drawn_strokes):
        entry = {'label': label, 'strokes': drawn_strokes}
        with open('trajectory_data.json', "r+") as file:
            data = json.load(file)
            data.append(entry)
            file.seek(0)
            json.dump(data, file, indent=4)
        dNr.add_gesture(label, drawn_strokes)
    clean()


def close(event):
    os._exit(os.EX_OK)


# Interactive window
m_window.title('$N example')
m_window.bind('<Escape>', close)
frame_cb = tk.Frame(m_window, borderwidth=2, relief=tk.FLAT)
frame_cb.pack()
cb1 = tk.Checkbutton(frame_cb, text="rotation invariance",
                     command=c_boxes,
                     variable=cb_var_1,
                     onvalue=1, offvalue=0)
cb2 = tk.Checkbutton(frame_cb, text="same number of strokes",
                     command=c_boxes,
                     variable=cb_var_2,
                     onvalue=1, offvalue=0)

m_canvas = tk.Canvas(m_window, width=400, height=400, background='lightgrey')
m_canvas.pack()
m_canvas.bind("<ButtonPress-1>", start_drawing)
m_canvas.bind("<ButtonRelease-1>", stop_drawing)
m_canvas.bind("<B1-Motion>", drawing)

cb1.pack(side=tk.LEFT)
cb2.pack(side=tk.RIGHT)
if dNr.get_rotation_invariance():    cb1.select()
if dNr.get_same_nb_strokes():        cb2.select()

frame_bt = tk.Frame(m_window, borderwidth=2, relief=tk.FLAT)
frame_bt.pack()
input_txt = tk.Text(frame_bt, height=2, width=10)
input_txt.pack(side=tk.RIGHT)
tk.Button(frame_bt, text="Add", command=addGesture).pack(side=tk.RIGHT)
tk.Button(frame_bt, text="Recognize", command=recognize).pack(side=tk.RIGHT)
tk.Button(frame_bt, text="Clean", command=clean).pack(side=tk.LEFT)

result_txt = tk.Label(m_window, text="")
result_txt.pack()

m_window.mainloop()
