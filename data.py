"""
This module defines the utilities and interface for loading
fingerprint datasets.
"""

import glob, random, cv2, os.path

class FingerprintDataset:
    """Provides ulility to read and load images from a specific dataset."""

    def __init__(self, name, dir_path, image_extension):
        if dir_path[-1] != "/":
            dir_path += "/"

        if image_extension[0] != ".":
            image_extension = "." + image_extension

        self.name = name
        self.dir_path = dir_path
        self.image_extension = image_extension
        self.file_list = glob.glob(dir_path + "*" + image_extension)
        self.file_list.sort()

    def get_num_images(self):
        """Return number of images in dataset."""
        return len(self.file_list)

    def get_file_list(self):
        """Print list of names of files in the dataset."""
        return self.file_list

    def read_random_image(self):
        """
        Read from random image file in dataset and return grayscale image
        matrix.
        """

        img_path = random.choice(self.file_list)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        fname = os.path.basename(img_path)
        return (img, fname)

    def read_image_at_ind(self, index):
        """
        Read from image file at specified index in alphabetically sorted list
        of file names.
        """

        if not (index >= -len(self.file_list)) and \
               (index < len(self.file_list)):
            raise IndexError("file index out of range")

        img_path = self.file_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img

    def read_image(self, name):
        """Load image with given name."""
        img = cv2.imread(self.dir_path + name, cv2.IMREAD_GRAYSCALE)
        return img


FVC2002_1A = FingerprintDataset(
    name = "Fingerprint Verification Competition 2002, Database 1.A",
    dir_path = "data/fvc2002/db1_a/", image_extension = ".tif")

DESIGN_SHOW_TIF = FingerprintDataset(
    name = "Subset of FVC2002_1A to use in design show application",
    dir_path = "data/design_show/", image_extension = ".tif")

DESIGN_SHOW_TRANSP = FingerprintDataset(
    name = "Subset of FVC2002_1A to use in design show application",
    dir_path = "data/design_show_transp/", image_extension = ".png")
