import matplotlib
matplotlib.use('TkAgg')

from functools import partial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import Tkinter as Tk

import data


def show_rand_print(ax, canvas):
    img, fname = data.FVC2002_1A.read_random_image()
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title("Image: " + fname)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    canvas.show()

f = Figure(figsize=(5,4), dpi=100)
a = f.add_subplot(111)

root = Tk.Tk()
root.wm_title("Fingerprint Seperation Demo")
canvas = FigureCanvasTkAgg(f, master=root)
show_rand_print(a, canvas)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

quit_button = Tk.Button(master=root, text='Quit', command=sys.exit)
quit_button.pack(padx=5, pady=20, side=Tk.LEFT)

np_callback = partial(show_rand_print, ax=a, canvas=canvas)
new_img_button = Tk.Button(master=root, text='New Image', command=np_callback)
new_img_button.pack(padx=5, pady=20, side=Tk.LEFT)

Tk.mainloop()
