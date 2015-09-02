import cv2
import scipy.spatial
import matplotlib.pyplot as plt
import skimage.filter.rank

import numpy as np
import skimage.morphology as skmorph

import data
import model

def extract_region_opening(img, is_demo=False):
    """
    Extracts fingerprint region of image via mophological opening
    """

    after_median = skimage.filter.rank.median(img, skmorph.disk(9))
    after_erode = skmorph.erosion(after_median, skmorph.disk(11))
    after_dil = skmorph.dilation(after_erode, skmorph.disk(5))
    _, t_dil_img = cv2.threshold(after_dil, 240, 40, cv2.THRESH_BINARY)

    if is_demo:
        _, t_med_img = cv2.threshold(after_median, 240, 255, cv2.THRESH_BINARY)
        _, t_erd_img = cv2.threshold(after_erode, 240, 40, cv2.THRESH_BINARY)
        erd_gry = t_erd_img.astype(np.uint8) * 255
        rgb_erd = np.dstack((erd_gry, img, img))
        dil_gry = t_dil_img.astype(np.uint8) * 255
        rgb_dil = np.dstack((dil_gry, img, img))

        plt.subplot(2,2,1)
        plt.imshow(after_erode, cmap="gray", interpolation="nearest")

        plt.subplot(2,2,2)
        plt.imshow(rgb_erd, interpolation="nearest")

        plt.subplot(2,2,3)
        plt.imshow(after_dil, cmap="gray", interpolation="nearest")

        plt.subplot(2,2,4)
        plt.imshow(rgb_dil, interpolation="nearest")
        plt.show()

    return t_dil_img


def extract_region_chull(img, is_demo=False):
    """
    Extracts fingerprint region of image as convex hull
    """

    after_median = skimage.filter.rank.median(img, skmorph.disk(5))
    _, t_img = cv2.threshold(after_median, 245, 255, cv2.THRESH_BINARY)
    t_img = t_img.astype(np.int32)
    is_zero = t_img == 0
    pixel_verts = np.transpose(np.nonzero(is_zero))
    chull = scipy.spatial.ConvexHull(np.array(pixel_verts))
    ch_verts = chull.points[chull.vertices]
    ch_verts = np.array([ch_verts[:,1], ch_verts[:,0]], dtype=np.int32)
    ch_verts = np.transpose(ch_verts)
    fprnt_area = np.zeros_like(img)
    cv2.fillConvexPoly(fprnt_area, ch_verts, 1)
    fingprint_region =  fprnt_area.astype(np.bool)

    if is_demo:
        cr_img = (fingprint_region != 1).astype(np.uint8) * 255
        rgb_img = np.dstack((cr_img, img, img))
        plt.imshow(rgb_img, interpolation="nearest")
        plt.show()

    return fingprint_region







