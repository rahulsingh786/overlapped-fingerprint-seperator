import cv2
import data
import os.path
import numpy as np

dataset = data.FVC2002_1A
img_names = dataset.get_file_list()

print "Generating transparent background pngs..."

for img_name in img_names:
    file_name = os.path.basename(img_name)
    print "Processing " + file_name + "..."
    gs_img = dataset.read_image(file_name)
    alpha = 255 - gs_img
    blk = np.zeros(gs_img.shape)
    rgba = np.dstack((blk, blk, blk, alpha))
    new_name = img_name[:-4] + ".png"
    print "Saving to path: " + new_name + "\n"
    cv2.imwrite(new_name, rgba)
