"""
Provides algorithm for fingerprint seperation as described in Feng 2012.

Method includes three steps:
    1. Orientation Field Estimation (implemented)
    2. Relaxation Labeling (not completed)
    3. Individual Print Extraction via Gabor Filter
"""

import cmath
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm
# import pylab
import numpy as np
import scipy.spatial
import scipy.stats

from functools import partial
from math import atan2, cos

BLOCK_SIZE = 16
MIN_FREQ = 5
MAX_FREQ = 32
MIN_ORIENT_DIFF = cmath.pi/12


# generate static frequency domain indexing
# -----------------------------------------
def gen_locals(locii, locjj):
    irange = np.arange(locii-1, locii+2)
    jrange = np.arange(locjj-1, locjj+2)
    ilocs, jlocs = np.mod(np.meshgrid(irange, jrange), 64)
    not_cntr = lambda coords: ((coords[0] != locii) or (coords[1] != locjj))
    coords = zip(ilocs.flatten(), jlocs.flatten())
    coords = np.array(filter(not_cntr, coords))
    return (locii, locjj, coords)

locs = gen_locals(2, 2)

ii, jj = np.meshgrid(np.arange(0, 64, dtype=np.float),
                     np.arange(0, 64, dtype=np.float),
                     indexing="ij")

dist_llc = np.sqrt(ii*ii+jj*jj)
LLC_MASK = (dist_llc >= MIN_FREQ) & (dist_llc < MAX_FREQ)
LLC_COORDS = np.transpose(np.nonzero(LLC_MASK))
LLC_POINTS = [gen_locals(*coords) for coords in LLC_COORDS]

dist_ulc = np.sqrt((63-ii)*(63-ii)+jj*jj)
ULC_MASK = (dist_ulc >= MIN_FREQ) & (dist_ulc < MAX_FREQ)
ULC_COORDS = np.transpose(np.nonzero(ULC_MASK))
ULC_POINTS = [gen_locals(*coords) for coords in ULC_COORDS]


def _find_two_largest_extrema(img): #, pimg, pimgnorm, plot_max = False):
    """
    This private utility function finds the two local extrema in the given
    frequency-domain image, after application of a selective mask which negates
    out-of-band modes.
    """

    lmaxes = []

    if np.any(np.isnan(img.flat)):
        # NaNs should not be here and they certainly shouldn't propagate
        raise RuntimeError

    # Lower left quadrant of FFT image
    for pti, ptj, locs in LLC_POINTS:
        pix_val = img[pti,ptj]

        # Select the 9x9 square surrounding the pixel in question
        local_region = img[locs[:,0], locs[:,1]]
        if pix_val >= np.max(local_region):
            orientation = atan2(float(pti), float(ptj))
            assert (not np.isnan(orientation)), \
                   "NAN orientation for LLC at (pti, ptj) = (%d, %d)" % \
                   (pti, ptj)

            lmaxes.append((pix_val, orientation, (pti, ptj)))

    # Lower left quadrant of FFT image
    for pti, ptj, locs in ULC_POINTS:

        # Select the 9x9 square surrounding the pixel in question
        local_region = img[locs[:,0], locs[:,1]]

        pix_val = img[pti, ptj]
        assert not np.isnan(pix_val)

        if pix_val >= np.max(local_region):
            orientation = atan2(float(pti-63), float(ptj))
            assert (not np.isnan(orientation)), \
                   "NAN orientation for LLC at (pti, ptj) = (%d, %d)" % \
                   (pti, ptj)

            lmaxes.append((pix_val, orientation, (pti, ptj)))


    largest_max = (-np.inf, None, None)
    for lmax in lmaxes:
        if lmax > largest_max:
            largest_max = lmax
    next_largest_max = ()

    next_largest_max_close = (-np.inf, None, None)
    for lmax in lmaxes:
        if lmax > next_largest_max and lmax != largest_max:
            if cos(lmax[1] - largest_max[1]) > cos(MIN_ORIENT_DIFF):
                next_largest_max = lmax
            elif lmax > next_largest_max_close:
                next_largest_max_close = lmax


    if largest_max[0] == -np.inf:
        largest_max = (0, 0, (0, 1))

    if next_largest_max == ():
        next_largest_max = next_largest_max_close

    if next_largest_max[0] == -np.inf:
        next_largest_max = largest_max

    lmax1 = {'value': largest_max[0], 'orientation': largest_max[1],\
             'coords': largest_max[2]}

    lmax2 = {'value': next_largest_max[0], 'orientation': next_largest_max[1],\
             'coords': next_largest_max[2]}

    if np.isnan(lmax1['orientation']) or np.isnan(lmax2['orientation']):
        raise RuntimeError

    return (lmax1, lmax2)


def estimate_orientation(img):
    """
    Find the estimated orientation field for a print image. Utilizes method
    given in Feng et. al., 'Robust and Efficient Algorithms for Seperating
    Latent Overlapped Fingerprints', IEEE Trans. Inf. Forensics Security, June
    2011.

    Parameters
    ----------
    img : numpy.ndarray
        Input two-dimentional numpy array, representing a grayscale fingerprint
        image
    show_steps : boolean
        If true, display plots of intermediate states, data etc.  Helpful for
        visualizing the algorithm's behavior. False by default.

    Returns
    -------
    orient_mat : A n x m x 2 block matrix of the primary and secondary
        orientation field estimations for the input image. The orientation in
        the (i, j, 1) entry corresponds to a primary field, (i, j, 2) a
        secondary.

    Raises
    ------
    TypeError
        if 'img' is not of type numpy.ndarray.
    """

    if type(img) is not np.ndarray:
        raise TypeError("Argument img must be a numpy array")

    # Get image matrix dimensions and calculate size of corresponding
    # orientation matrix
    img_shape = np.array(img.shape, dtype=float)
    orient_dim = np.ceil(img_shape / float(BLOCK_SIZE)).astype(int)

    # Pad the fingerprint image with zeros. We pad to compesate for two
    # effects:
    # 1) expand image so that each dimension is a multiple of 16
    # 2) expand further to accomodate taking the FFT on the size 64 square
    #    centered on a block.
    block_size_pad = orient_dim * BLOCK_SIZE - img_shape
    fft_pad = (64 - BLOCK_SIZE) / 2
    pad_widths = [(int(fft_pad), int(fft_pad + pad_width))
         for pad_width in block_size_pad]

    pad_value = np.max(np.max(img))
    img = np.pad(img, pad_widths, mode='constant',
        constant_values=((pad_value, pad_value), (pad_value, pad_value)))

    # Add third dimension of size two for primary/secondary orientations
    orient_dim = list(orient_dim) + [2]

    # Instatiate the orientation field array
    orf = np.zeros(shape=orient_dim, dtype=np.float_)
    # corns = np.zeros_like(orf[:,:,0], dtype=np.bool)

    # Instantiate isometric bivariate normal pdf
    biv_normal = scipy.stats.multivariate_normal(mean=[31.5, 31.5],
        cov=[[BLOCK_SIZE, 0], [0, BLOCK_SIZE]])
    biv_normal_pdf = np.zeros((64, 64))
    for ii in np.arange(64):
        for jj in np.arange(64):
            biv_normal_pdf[ii, jj] = biv_normal.pdf((ii, jj))

    # Iterate through the image and find the orientation field block-by-block.
    # Numpy provides functionality to do iterations faster than native 'for'
    # loops, but this is easier to read for now.
    for ii in range(orient_dim[0]):
        i_start = ii * BLOCK_SIZE
        i_end = i_start + 64

        for jj in range(orient_dim[1]):
            j_start = jj * BLOCK_SIZE
            j_end = j_start + 64

            # Extract the 64 x 64 subregion centered on the orientation block
            img_block = img[i_start:i_end, j_start:j_end]

            # Multiply element-wise by isometric bivariate normal distribution
            filtered_img = np.multiply(img_block, biv_normal_pdf)

            # Take 2-D FFT of filtered image block
            fft = np.fft.fft2(filtered_img)

            # Take magnitude of FFT and find the two largest local max/min
            fft_mag = np.abs(fft).astype(np.float)
            peak1, peak2 = _find_two_largest_extrema(fft_mag)

            # Add to orientation field
            orf[ii, jj, 0] = peak1.get('orientation')
            orf[ii, jj, 1] = peak2.get('orientation')

    return orf


def _sort_overlapped_coords(overlapped_bmap):
    """
    Given overlapped region bitmap 'olp_reg', returns coordinates
    of the overlapped entries in increasing order of distance from
    non-overlapped region
    """
    unsorted_bmp = overlapped_bmap.copy().astype(np.bool)
    sorted_coords = []
    max_inds = np.array(overlapped_bmap.shape)
    # print "max_inds:" + str(max_inds)

    unsorted_coords = np.transpose(np.nonzero(unsorted_bmp))
    unsorted_coords = map(tuple, unsorted_coords)
    maxiter = len(unsorted_coords)
    i = 0

    print "np.sum(unsorted_bmp):" + str(np.sum(unsorted_bmp))


    def is_on_border(crds, unsorted_bmp):
        crd_arry = np.array((crds, crds, crds, crds))
        nbr_crds = crd_arry + np.array([[-1, 0] , [1, 0], [0, -1], [0, 1]])
        is_in_bnd = lambda crd: np.all((crd >= 0) | (crd < max_inds))
        nbr_crds_in_bnd = filter(is_in_bnd, nbr_crds)
        try:
            is_unsorted = map(lambda crd: unsorted_bmp[crd[0], crd[1]], nbr_crds_in_bnd)
        except IndexError:
            print "unsorted_bmp.shape:" + str(unsorted_bmp.shape)
            print "nbr_crds:" + str(nbr_crds)
            print "nbr_crds_in_bnd:" + str(nbr_crds_in_bnd)
            raise IndexError

        iob = not np.all(is_unsorted)
        return iob


    while unsorted_coords:
        i = i + 1
        if i > maxiter:
            raise RuntimeError("_sort_overlapped_coords went over max iter")

        unsorted_next = []
        new_sorted = []
        for coord in unsorted_coords:
            if is_on_border(coord, unsorted_bmp.copy()):
                new_sorted.append(coord)

            else:
                unsorted_next.append(coord)

        for coord in new_sorted:
            unsorted_bmp[coord[0], coord[1]] = False

        sorted_coords = sorted_coords + new_sorted
        unsorted_coords = unsorted_next

    return sorted_coords



def _blockify_print_region(pregion):
    """
    Given a full-resolution print region bitmap 'pregion', return an
    approximate bitmap with the exact dimensions of the corresponding
    orientation field 'block_region'. The returned block_region[i,j] is equal
    to one if the print region corresponding to the ith, jth entry in the
    orientation field contains information from the print represented in
    'pregion'.
    """

    min_print_pixels = 0.1 * BLOCK_SIZE
    img_shape = np.array(pregion.shape, dtype=float)
    blocked_dims = np.ceil(img_shape / float(BLOCK_SIZE)).astype(int)
    block_region = np.zeros(blocked_dims, dtype=np.bool)

    for ii in range(blocked_dims[0]):
        i_start = ii * BLOCK_SIZE
        i_end = i_start + BLOCK_SIZE

        for jj in range(blocked_dims[1]):
            j_start = jj * BLOCK_SIZE
            j_end = j_start + BLOCK_SIZE

            n_pxls_with_print = np.sum(pregion[i_start:i_end, j_start:j_end])
            block_region[ii][jj] = n_pxls_with_print >= min_print_pixels

    return block_region


def relax_label_two(orf, print_layers, tkagg=False, view_axis=False):
    """
    Perform relaxation labeling of a two-class orientation field 'orf'
    """
    assert orf.shape == print_layers.shape
    do_update_view = tkagg and view_axis

    # Used to determine convergence of labels
    EPSILON = 0.1

    olpd_blks = print_layers[:,:,0] & print_layers[:,:,1]
    any_prnt_blks = print_layers[:,:,0] | print_layers[:,:,1]
    no_print_i, no_print_j = np.nonzero(~any_prnt_blks)
    sorted_coords = _sort_overlapped_coords(olpd_blks)
    n_overlapped = len(sorted_coords)


    p0 = print_layers[:,:,0] & (~olpd_blks)
    p0 = p0.astype(np.float)
    p0[olpd_blks] = 0.5

    p1 = print_layers[:,:,1] & (~ olpd_blks)
    p1 = p1.astype(np.float)
    p1[olpd_blks] = 0.5

    pk_next = np.dstack((p0, p1))

    def unpack_and_zip_data(varg):
        """
        Helper function which zips all data from the sorted overlapped pixels
        into one convenient iterable
        """
        cntr, idx = varg
        imin = max(0, cntr[0] - 2)
        jmin = max(0, cntr[1] - 2)
        imax = min(orf.shape[0], cntr[0] + 3)
        jmax = min(orf.shape[1], cntr[1] + 3)
        nbri, nbrj = np.meshgrid(np.arange(imin,imax), \
                                   np.arange(jmin, jmax), \
                                  indexing="ij")
        not_cntr = (nbri.flat != cntr[0]) | (nbrj.flat != cntr[1])
        has_print_inf = any_prnt_blks[nbri.flat, nbrj.flat]
        nbri = nbri.flat[not_cntr & has_print_inf]
        nbrj = nbrj.flat[not_cntr & has_print_inf]
        offsti = cntr[0] - nbri
        offstj = cntr[1] - nbrj
        epnt = -0.5 * (offsti ** 2 + offstj ** 2)
        wt = np.exp(epnt)
        wt = wt / np.sum(wt)
        return (cntr, (nbri, nbrj), wt, idx)

    ovlp_data = map(unpack_and_zip_data, \
                    zip(sorted_coords, range(n_overlapped)))

    def compat_func(lab_cnt, lab_nbr, cnt_coords, nbr_coords):
        """ Returns compatability between a center pixel and a neighbor. Used
        as a helper function. """

        thta_cnt = orf[cnt_coords]
        thta_nbr = orf[nbr_coords]
        is_nbr_olpd = olpd_blks[nbr_coords]

        if not is_nbr_olpd:
            thta_diff = thta_cnt[lab_cnt] - thta_nbr[lab_cnt]
            r = 2 * abs(cos(thta_diff)) - 1.0
            if not ( -1 <= r <= 1 ):
                print "BAD R!"
                raise RuntimeError

        if lab_cnt == lab_nbr:
            thta_diff = thta_cnt - thta_nbr[::1]
        else:
            thta_diff = thta_cnt - thta_nbr[::-1]

        abs_cos_thta = np.abs(np.cos(thta_diff))
        r = np.sum(abs_cos_thta) - 1
        if not (-1 <= r <= 1):
            print "BAD R!"
            raise RuntimeError

        return r

    def get_sop_rp(nbr_crds, cnt_crds, lab_cnt, pk):
        """Helper function which returns the sum of products term of
         the support update function """

        r0 = compat_func(lab_cnt=lab_cnt, lab_nbr=0, \
                         cnt_coords=cnt_crds, nbr_coords=nbr_crds)
        sop0 = r0 * pk[nbr_crds][lab_cnt]
        r1 = compat_func(lab_cnt=lab_cnt, lab_nbr=1, \
                         cnt_coords=cnt_crds, nbr_coords=nbr_crds)
        sop1 = r1 * pk[nbr_crds][lab_cnt]
        return (r0 * sop0) + (r1 + sop1)

    k = 0
    did_converge = False

    while (not did_converge) and k < 50:
        k = k + 1
        pk = pk_next.copy()
        q = np.zeros((n_overlapped, 2))

        if do_update_view:
            try:
                pkshow = np.copy(pk_next[:,:,0])
                minpk = np.min(pkshow)
                maxpk = np.max(pkshow)
                pkshow = (pkshow-minpk)/(maxpk-minpk)
                cm = matplotlib.cm.get_cmap('rainbow')
                pkrgb = cm(pkshow)
                pkrgb = pkrgb[:,:,0:3]
                pkrgb[no_print_i, no_print_j] = 1.0
                view_axis.imshow(pkrgb, interpolation="nearest")
                tkagg.show()

            except IndexError as err:
                print "pkshow.shape:" + str(pkshow.shape)
                print "cm:" + str(cm)
                print "pkrgb.shape:" + str(pkrgb.shape)
                print "any_prnt_blks:" + str(any_prnt_blks)
                raise err


        for crds, nbrs, wts, idx in ovlp_data:
            assert olpd_blks[crds]

            for lbl in [0, 1]:
                get_rp = partial(get_sop_rp, cnt_crds=crds, lab_cnt=lbl, pk=pk)
                sums_prod_rp = map(get_rp, zip(*nbrs))
                wtd_sop = wts * sums_prod_rp
                q_i = np.sum(wtd_sop)
                q[idx, lbl] = q_i

            pk_i_nxt_unnorm = pk[crds] * (1 + q[idx])
            pk_i_nxt = pk_i_nxt_unnorm / np.sum(pk_i_nxt_unnorm)
            pk_next[crds] = pk_i_nxt

        p_diff = np.sum(np.abs(pk - pk_next))
        did_converge = p_diff < EPSILON
        print "p_diff:" + str(p_diff)

        if did_converge:
            print "CONVERGED!!"

    # extract labeled orientations from overlapped area
    p0_gt_p1 = pk[:,:,0] >= pk[:,:,1]
    print_0_orf = np.where(p0_gt_p1, orf[:,:,0], orf[:,:,1])
    print_1_orf = np.where(~p0_gt_p1, orf[:,:,0], orf[:,:,1])

    # extract primary orientation into known print blocks
    p0_only = print_layers[:,:,0] & (~olpd_blks)
    p1_only = print_layers[:,:,1] & (~olpd_blks)
    print_0_orf[p0_only] = orf[:,:,0][p0_only]
    print_1_orf[p1_only] = orf[:,:,1][p1_only]

    # set all other blocks (those without a part of a print) to NaN
    no_info = (~print_layers[:,:,0]) & (~print_layers[:,:,1])
    print_0_orf[no_info] = np.nan
    print_1_orf[no_info] = np.nan

    return np.dstack((print_0_orf, print_1_orf))


def prep():
    pr0_gr = cv2.imread("print_1_region.tiff", cv2.IMREAD_GRAYSCALE)
    pr0_bmask = pr0_gr.astype(np.bool)
    pr0 = _blockify_print_region(pr0_bmask)
    pr1_gr = cv2.imread("print_2_region.tiff", cv2.IMREAD_GRAYSCALE)
    pr1_bmask = pr1_gr.astype(np.bool)
    pr1 = _blockify_print_region(pr1_bmask)
    test_img = cv2.imread("overlapped_prints_test.tiff", cv2.IMREAD_GRAYSCALE)

    overlapped_region = np.logical_and(pr0, pr1)
    orf = estimate_orientation(test_img)
    assert np.all(orf[:,:,0].shape == overlapped_region.shape)
    assert np.all(orf[:,:,0].shape == pr0.shape)
    assert np.all(orf[:,:,0].shape == pr1.shape)
    return (orf, np.dstack((pr0, pr1)), test_img)


def test_sep(orf, prnt_bmps):
    print "orf.shape:" + str(orf.shape)
    print "prnt_bmps.shape:" + str(prnt_bmps.shape)
    seperated_orientations = relax_label_two(orf, prnt_bmps)
    seperated_orientations0 = seperated_orientations[:,:,0]

    for iii in range(seperated_orientations0.shape[0]):
        for jjj in range(seperated_orientations0.shape[1]):
            if not prnt_bmps[iii,jjj,0]:
                continue

            i_base = 16*iii + 8
            j_base = 16*jjj + 8
            angle = seperated_orientations0[iii][jjj]

            ioffst = 8*np.sin(angle+cmath.pi/2)
            joffst = 8*np.cos(angle+cmath.pi/2)
            plt.plot([j_base-joffst, j_base+joffst],
                [i_base-ioffst, i_base+ioffst],
                color='r', lw=1.5)

    test_img = cv2.imread("overlapped_prints_test.tiff", cv2.IMREAD_GRAYSCALE)
    plt.imshow(test_img, cmap = "gray", interpolation = "nearest")
    plt.show()
