"""
Demo basic image manipulation and visualization actions via Scipy/Matplotlib
and OpenCV.
"""
import cv2, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cmath
import model
import data

DEMO_IMAGE_PATH = "data/fvc2002/db1_a/101_1.tif"


def load_img():
    """Utility function loads and returns demo fingerprint image."""
    img = cv2.imread(DEMO_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise Exception("Demo image not found at: " + DEMO_IMAGE_PATH)

    return img


def demo_imshow_cv():
    """Demo OpenCV version of imshow(). """

    window_name = "Demo Fingerprint"

    img = load_img()
    cv2.namedWindow(window_name)
    cv2.startWindowThread()
    cv2.imshow(window_name, img)

    # keep window open until user presses esc key
    while not cv2.waitKey(0) == 27:
        continue

    # esc was pressed, close the demo window
    cv2.waitKey(1)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)


def show_fprint(img):
    """
    Wrapper that uses matplotlib.pyplot to display grayscale fingerprint
    image.
    """

    plt.imshow(img, cmap = "gray", interpolation = "bicubic")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def demo_imshow_matplotlib():
    """Demo matplotlib version of imshow()."""

    img = load_img()
    show_fprint(img)


def show_rescaled_image():
    """Show rescaled grayscale image so min is 0 and max is 255."""

    img = load_img()
    img = img.astype(np.float)
    img_unscaled = img / np.max(img)
    img_max = np.max(img_unscaled)
    img_min = np.min(img_unscaled)
    img_scaled = (img_unscaled - img_min) / (img_max - img_min)
    img_under = img_scaled < img_min
    img_under = np.multiply(img_under, img_scaled)

    cdict = {
      'red'  :  ( (0.0, 1.0, 1.0), (img_min-0.001, 0.0, 0.0), (1., 1., 1.)),
      'green':  ( (0.0, 0.0, 0.0), (img_min, 0.0, 0.0), (1.0, 1.0, 1.0)),
      'blue' :  ( (0.0, 0.0, 0.0), (img_min, 0.0, 0.0), (1.0, 1.0, 1.0))
    }

    color_map = matplotlib.colors.LinearSegmentedColormap(
        'show_clipped', cdict, 1024)

    plt.subplot(1, 2, 1)
    plt.imshow(
        img_unscaled,
        cmap = color_map,
        interpolation = "nearest",
        vmin = 0.,
        vmax = 1.,
        )
    plt.xticks([])
    plt.yticks([])
    plt.title("Unscaled image.")

    plt.subplot(1, 2, 2)
    plt.imshow(
        img_scaled,
        cmap = color_map,
        interpolation = "nearest",
        vmin = 0.,
        vmax = 1.,
        )
    plt.xticks([])
    plt.yticks([])
    plt.title("Scaled image.")
    plt.show()


def show_thresholding():
    """
    Demonstrate the effects of different types of thresholding on a
    grayscale image.
    """

    img = load_img()
    _, img_bin_global = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    img_bin_mean_c = cv2.adaptiveThreshold(img, 255, \
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    img_bin_gauss_c = cv2.adaptiveThreshold(img, 255, \
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap = "gray", interpolation = "nearest")
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(img_bin_global, cmap = "gray", interpolation = "nearest")
    plt.title("Global (v = 64/255)")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(img_bin_mean_c, cmap = "gray", interpolation = "nearest")
    plt.title("Adaptive Mean Thresholding")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.imshow(img_bin_gauss_c, cmap = "gray", interpolation = "nearest")
    plt.title("Adaptive Gaussian Thresholding")
    plt.xticks([])
    plt.yticks([])

    plt.show()


def show_random_from_dataset(dataset):
    """Show four random fingerprint images from dataset."""
    for i in range(1, 5):
        plt.subplot(2, 2, i)
        img, img_name = dataset.read_random_image()
        plt.imshow(img, cmap = "gray", interpolation = "nearest")
        plt.title(img_name)
        plt.xticks([])
        plt.yticks([])

    plt.show()


def plot_orient_field(orf, img, sep_subplots=False):

    # if sep_subplots:
    #     plt.subplot(2,1,1)
    #     plt.imshow(img, cmap="gray", interpolation="nearest")

    # for ii in range(orf.shape[0]):
    #     for jj in range(orf.shape[1]):

    #         angle = orf[ii, jj, 0]
    #         if angle == np.nan:
    #             continue

    #         i_base = 16*ii + 8
    #         j_base = 16*jj + 8
    #         ioffst = 8*np.sin(angle+cmath.pi/2)
    #         joffst = 8*np.cos(angle+cmath.pi/2)
    #         plt.plot([j_base-joffst, j_base+joffst],
    #             [i_base-ioffst, i_base+ioffst],
    #             color='r', lw=1.5)


    # if sep_subplots:
    #     plt.subplot(2,1,2)

    for ii in range(orf.shape[0]):
        for jj in range(orf.shape[1]):

            angle = orf[ii, jj, 0]
            if angle == np.nan:
                continue

            i_base = 16*ii + 8
            j_base = 16*jj + 8
            ioffst = 8*np.sin(angle+cmath.pi/2)
            joffst = 8*np.cos(angle+cmath.pi/2)
            plt.plot([j_base-joffst, j_base+joffst],
                     [i_base-ioffst, i_base+ioffst],
                     color='b', lw=1.5)

    plt.imshow(img, cmap="gray", interpolation="nearest")
    plt.show()
    return


def full_demo_orient_field():
    """
    Estimates orientation field of a random print from FVC2002-1A, and
    displays the result as a vector field overlayed on the original print
    image a la Feng 2012.
    """
    img = data.FVC2002_1A.read_random_image()[0]
    orf = model.estimate_orientation(img, True)[:,:,0]

    # This will plot a red line in the direction of the detected orientation
    # for each block, overlayed on the original print, just like the figures
    # in Chen 2011. This should probably be a utility function.
    for ii in range(orf.shape[0]):
        for jj in range(orf.shape[1]):
            i_base = 16*ii + 8
            j_base = 16*jj + 8
            angle = orf[ii][jj]

            if angle == np.nan:
                continue

            else:
                offset = cmath.rect(8.0, angle)
                plt.plot([j_base-offset.imag, j_base+offset.imag],
                    [i_base-offset.real, i_base+offset.real],
                    color='r')

    plt.imshow(img, cmap = "gray", interpolation = "nearest")
    plt.show()

if __name__ == "__main__":
    full_demo_orient_field()
