"""
Demo GUI for senior design show. Started at 11:55 PM, 12/7/2014.
"""

import cmath
import numpy as np
import os.path
import PIL.Image
import PIL.ImageDraw
import PIL.ImageMath
import PIL.ImageTk
import random
import Tkinter as TK

from functools import partial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import data
import model
import utils
import visualization

CANVAS_WIDTH = 720
CANVAS_HEIGHT = 480

IMG_WIDTH = 480
IMG_HEIGHT = 640

# global cur_rhs_img_name
# global cur_lhs_img_name
global canvas
global olpped_img
global olpped_img_id
global olpped_pht
global show_step
global demo_dat
global orf

show_step = "SHOW_OVERLAPPED"
demo_dat = {}

def bname_to_rgbt(bn):
    return 'data/design_show_transp/' + bn + '.png'

def bname_to_gray(bn):
    return 'data/design_show/' + bn + '.tif'

demo_image_data_flist = data.DESIGN_SHOW_TIF.file_list
img_names = map(lambda fn: os.path.basename(fn).split('.')[0], \
                 demo_image_data_flist)


# Instantiate Root Window and Grid Layout
# =======================================
root = TK.Tk()
root.configure(background="white")
root_width = root.winfo_screenwidth()
root_height = root.winfo_screenheight()-54
root.geometry("%dx%d+0+0" % (root_width, root_height))

root.grid_columnconfigure(0, weight=0)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=0)

for row in range(5):
    root.grid_rowconfigure(row, weight=1)


# Load Image Data
# ===============
src_rgbt_imgs = {'CURLHS': 'NONE', 'CURRHS': 'NONE'}
src_gray_imgs = {'CURLHS': 'NONE', 'CURRHS': 'NONE'}
src_rgbt_phot = {'CURLHS': 'NONE', 'CURRHS': 'NONE'}

fnames_to_show = random.sample(img_names, 10)


# Instantiate Canvas for Photo Manipulation
# =========================================
canv_frame = TK.Frame(root, bd=-2, background="white", padx=0, pady=0)
canv_frame.grid(column=1, row=0, rowspan=5, sticky=TK.N+TK.S+TK.E+TK.W)
canvas = TK.Canvas(canv_frame, background="white", bd=-2, height=CANVAS_HEIGHT,
                   width=CANVAS_WIDTH, selectborderwidth=-2)
canvas.place(relx=0.5, rely=0.5, anchor=TK.CENTER)
# canv_frame.config(bd=0, padx=1, pady=0, )

lhs_img_coord_x = 50
lhs_img_coord_y = 0
rhs_img_coord_x = CANVAS_WIDTH-IMG_WIDTH-50
rhs_img_coord_y = lhs_img_coord_y

lhs_img_id = canvas.create_image(0, 0, anchor=TK.N+TK.W)
rhs_img_id = canvas.create_image(rhs_img_coord_x, rhs_img_coord_y,
                                 anchor=TK.N+TK.W)

olpped_img_id = canvas.create_image(0, 0, anchor=TK.N+TK.W)

def update_img(_, photo_name, img_id):
    global olpped_img_id
    global show_step
    show_step = "SHOW_OVERLAPPED"
    canvas.itemconfig(olpped_img_id, state=TK.HIDDEN)
    if img_id == lhs_img_id:
        src_rgbt_imgs['CURLHS'] = src_rgbt_imgs[photo_name]
        src_gray_imgs['CURLHS'] = src_gray_imgs[photo_name]
        src_rgbt_phot['CURLHS'] = src_rgbt_phot[photo_name]
    else:
        src_rgbt_imgs['CURRHS'] = src_rgbt_imgs[photo_name]
        src_gray_imgs['CURRHS'] = src_gray_imgs[photo_name]
        src_rgbt_phot['CURRHS'] = src_rgbt_phot[photo_name]

    new_photo = src_rgbt_phot[photo_name]
    canvas.itemconfig(img_id, image=new_photo, state=TK.NORMAL)
    print "Click on pn: " + photo_name + "img id: " + str(img_id)
    print "\n"


olpped_img_data = {}
def show_generated_img(_):
    global canvas
    global olpped_img
    global olpped_pht
    global show_step
    global demo_dat
    global orf

    if show_step == "SHOW_OVERLAPPED":
        print "SHOW OVERLAPPED GEN"
        lhs_img_src = src_rgbt_imgs['CURLHS']
        rhs_img_src = src_rgbt_imgs['CURRHS']

        if (lhs_img_src == 'NONE') or (rhs_img_src == 'NONE'):
            print "LEAVING EARLY FROM SHOW GEN IMG"
            return None

        canvas.itemconfig(lhs_img_id, state=TK.HIDDEN)
        canvas.itemconfig(rhs_img_id, state=TK.HIDDEN)
        gen_img_lhs = PIL.Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), 255)
        # olpped_img_data["gen_img_lhs"] = gen_img_lhs
        gen_img_rhs = PIL.Image.new('L', (CANVAS_WIDTH, CANVAS_HEIGHT), 255)
        # olpped_img_data["gen_img_rhs"] = gen_img_rhs
        lhs_img = src_gray_imgs['CURLHS']
        # olpped_img_data["lhs_img"] = lhs_img
        rhs_img = src_gray_imgs['CURRHS']
        # olpped_img_data["rhs_img"] = rhs_img
        gen_img_lhs.paste(lhs_img, (50, 0))
        gen_img_rhs.paste(rhs_img, (CANVAS_WIDTH-IMG_WIDTH-50, 0))

        eval_exp = "convert(min(a,b), 'L')"
        olpped_img = PIL.ImageMath.eval(eval_exp, a=gen_img_lhs, b=gen_img_rhs)
        demo_dat['olpped_img'] = olpped_img
                    # a=gen_img_lhs.convert('1', dither=0),\
                    # b=gen_img_rhs.convert('1', dither=0))
        olpped_pht = PIL.ImageTk.PhotoImage(olpped_img)
        canvas.itemconfig(olpped_img_id, image=olpped_pht, state=TK.NORMAL)


        prnt_img_lhs_ary = np.array(gen_img_lhs, copy=True)
        demo_dat['prnt_img_lhs_ary'] = prnt_img_lhs_ary
        prnt_img_rhs_ary = np.array(gen_img_rhs, copy=True)
        demo_dat['prnt_img_rhs_ary'] = prnt_img_rhs_ary
        demo_dat['prnt_layers'] =  \
                     np.dstack((prnt_img_lhs_ary, prnt_img_rhs_ary))
        reg_lhs_prnt = utils.extract_region_chull(prnt_img_lhs_ary)
        demo_dat['reg_lhs_prnt'] = reg_lhs_prnt
        reg_rhs_prnt = utils.extract_region_chull(prnt_img_rhs_ary)
        demo_dat['reg_rhs_prnt'] = reg_rhs_prnt

        plhs = PIL.Image.fromarray(reg_lhs_prnt.astype(np.uint8) * 255, \
                                    mode='L')
        prhs = PIL.Image.fromarray(reg_rhs_prnt.astype(np.uint8) * 255, \
                                    mode='L')

        prnt_reg_img_rgb = PIL.Image.merge('RGB', [plhs,prhs,olpped_img])
        demo_dat['prnt_reg_img_rgb'] = prnt_reg_img_rgb.copy()
        prnt_reg_pht_rgb = PIL.ImageTk.PhotoImage(prnt_reg_img_rgb.copy())
        # canvas.itemconfig(olpped_img_id, image=prnt_reg_pht_rgb)
        show_step = "PRINT_REGIONS"


    ## ==== SHOW PRINT REGIONS
    elif show_step == "PRINT_REGIONS":
        print "SHOW PRINT_REGIONS"
        prnt_reg_pht_rgb = PIL.ImageTk.PhotoImage(demo_dat['prnt_reg_img_rgb'])
        demo_dat['prnt_reg_pht_rgb'] = prnt_reg_pht_rgb
        canvas.itemconfig(olpped_img_id, image=prnt_reg_pht_rgb)

        olpped_ary = np.array(demo_dat['olpped_img'], dtype=np.uint8, \
                              copy=True)

        show_step = "OR_FIELD"


    elif show_step == "OR_FIELD":
        either_reg = demo_dat["reg_rhs_prnt"] | demo_dat["reg_lhs_prnt"]
        tka_frame = TK.Frame(root, bd=-2, background="white", padx=0, pady=0)
        tka_frame.grid(column=1, row=0, rowspan=5, sticky=TK.N+TK.S+TK.E+TK.W)
        demo_dat['tka_frame'] = tka_frame

        fig = Figure(figsize=(7, 5.5), facecolor="w", frameon=False, \
                     edgecolor='w')
        demo_dat['fig'] = fig
        ax = fig.add_axes((0,0,1,1), frameon=False, axis_bgcolor='w',\
                            axisbg='w')
        demo_dat['ax'] = ax
        tkagg = FigureCanvasTkAgg(fig, master=tka_frame)
        demo_dat['tkagg'] = tkagg
        tkagg.get_tk_widget().place(relx=0.5, rely=0.5, anchor=TK.CENTER)
        tkagg.get_tk_widget().config(height=(CANVAS_HEIGHT+100), \
                                     width=(CANVAS_WIDTH+150), bd=-2, \
                                     bg='white', selectborderwidth=-2,
                                     highlightthickness=0)
        ax.axis('off')
        fig.patch.set_visible(False)

        canvas.delete(TK.ALL)
        # canvas.place(relx=0.5, rely=0.5, anchor=TK.CENTER)

        olpped_ary = np.array(demo_dat['olpped_img'], dtype=np.uint8, \
                      copy=True)
        ax.imshow(olpped_ary, cmap="gray", interpolation="nearest")
        either_reg = model._blockify_print_region(either_reg)
        if 'orf' not in globals().keys():
            orf = model.estimate_orientation(olpped_ary)

        # demo_dat['orf'] = orf
        for ii in range(orf.shape[0]):
            for jj in range(orf.shape[1]):

                angle = orf[ii, jj, 0]
                if angle == np.nan or not (either_reg[ii,jj]):
                    continue

                i_base = 16*ii + 8
                j_base = 16*jj + 8
                ioffst = 8*np.sin(angle+cmath.pi/2)
                joffst = 8*np.cos(angle+cmath.pi/2)
                ax.plot([j_base-joffst, j_base+joffst],
                         [i_base-ioffst, i_base+ioffst],
                         color='b', lw=1.5)

        for ii in range(orf.shape[0]):
            for jj in range(orf.shape[1]):


                angle = orf[ii, jj, 1]
                if angle == np.nan or not (either_reg[ii,jj]):
                    continue

                i_base = 16*ii + 8
                j_base = 16*jj + 8
                ioffst = 8*np.sin(angle+cmath.pi/2)
                joffst = 8*np.cos(angle+cmath.pi/2)
                ax.plot([j_base-joffst, j_base+joffst],
                         [i_base-ioffst, i_base+ioffst],
                         color='r', lw=1.5)

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        tkagg.show()

        show_step = "RELAX_LABEL_PROC"

    if show_step == "RELAX_LABEL_PROC":
        # orf = demo_dat['orf']
        # oc = np.copy(orf)

        pl_lhs = model._blockify_print_region(\
                 np.array(demo_dat["reg_lhs_prnt"], dtype=np.bool, copy=True))
        pl_rhs = model._blockify_print_region( \
                 np.array(demo_dat["reg_rhs_prnt"], dtype=np.bool, copy=True))

        p_lyrs = np.dstack((pl_lhs, pl_rhs))
        tkagg = demo_dat['tkagg']
        ax = demo_dat['ax']

        print "orf.shape:" + str(orf.shape)
        print "orf.dtype:" + str(orf.dtype)
        print "p_lyrs.shape:" + str(p_lyrs.shape)
        print "p_lyrs.dtype:" + str(p_lyrs.dtype)
        _ = model.relax_label_two(orf, p_lyrs, tkagg, view_axis=ax)


    ## ==== SHOW ORF RESULT
    ## ==== SHOW RELAXATION LABELING RESULT
    ## ==== SHOW GABOR FILTER RESULT


# space_cb = partial(show_generated_img, img_id=olpped_img_id)
root.bind("<space>", show_generated_img)


# Instantiate Print Selection Side Panels
# =======================================
thumb_height = int(root_height/5 - 10)
thumb_width = int(thumb_height*1.33)

prnt_lbls = []
prnt_lbl_callbacks = []
prnt_thumbs = []
for ii, fname in enumerate(fnames_to_show):
    row = ii % 5
    if ii < 5:
        col = 0
    else:
        col = 2

    rgbt_fname = bname_to_rgbt(fname)
    gray_fname = bname_to_gray(fname)

    img_rgbt = PIL.Image.open(rgbt_fname)
    src_rgbt_imgs[fname] = img_rgbt.copy()
    src_rgbt_phot[fname] = PIL.ImageTk.PhotoImage(img_rgbt.copy())

    img_gray = PIL.Image.open(gray_fname)
    src_gray_imgs[fname] = img_gray.copy()


    # Generate thumb image from src image and display on side panel
    thumb_image = img_rgbt.resize((thumb_width, thumb_height))
    photo = PIL.ImageTk.PhotoImage(thumb_image)
    label = TK.Label(root, image=photo, bd=0, background="white")
    label.grid(column=col, row=row)
    prnt_lbls.append(label)

    if ii < 5:
        img_id = lhs_img_id

    else:
        img_id = rhs_img_id

    # Bind callback to update print1 image in canvas
    cb = partial(update_img, photo_name=fname, img_id=img_id)
    prnt_lbl_callbacks.append(cb)
    label.bind("<ButtonPress-1>", cb)

    # Save a reference to the thumbnail image so that it is not garbage
    # collected
    prnt_thumbs.append(photo)

TK.mainloop()
