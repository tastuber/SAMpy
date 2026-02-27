"""Image reduction routines: reading, cleaning, windowing, and subframing."""

import copy

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy import ndimage
from jwst import datamodels
from jwst.datamodels import dqflags
from ipywidgets import IntProgress
from IPython.display import display as idisplay


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def supergauss_fractional_width(pixel_radius, fraction, order, size):
    """Generate a super-Gaussian window from a fractional-width specification.

    Parameters
    ----------
    pixel_radius : float
        Radius in pixels at which the window reaches ``fraction``.
    fraction : float
        Value of the window at ``pixel_radius`` (e.g. 0.95).
    order : float
        Super-Gaussian order (higher = steeper roll-off).
    size : int
        Side length of the output square array.

    Returns
    -------
    numpy.ndarray
        2D super-Gaussian window of shape ``(size, size)``.
    """
    sigma = (-float(pixel_radius) ** float(order)
             / np.log(fraction)) ** (1.0 / float(order))
    distances = np.array([
        [np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
         for x in range(size)]
        for y in range(size)
    ])
    return np.exp(-distances ** order / sigma ** order)


def supergauss_half_width(fwhm, order, size):
    """Generate a super-Gaussian window from a FWHM specification.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum in pixels.
    order : float
        Super-Gaussian order.
    size : int
        Side length of the output square array.

    Returns
    -------
    numpy.ndarray
        2D super-Gaussian window of shape ``(size, size)``.
    """
    k = -1.0 / (2 * (fwhm / 2.35482) ** order)
    distances = np.array([
        [np.sqrt((x - size // 2) ** 2 + (y - size // 2) ** 2)
         for x in range(size)]
        for y in range(size)
    ])
    return np.exp(k * distances ** order)


def apply_window(images, fwhm, display=False):
    """Apply a super-Gaussian window to a stack of images.

    The window is a 4th-order super-Gaussian whose full width at half
    maximum is ``fwhm`` pixels.  This apodizes the image edges to
    reduce spectral leakage in the FFT.

    A typical choice for JWST/NIRISS AMI is the diffraction-limited
    PSF width of a single subaperture::

        fwhm = wavelength_um * 1e-6 * 206265.0 / (pixel_scale * 0.65)

    where ``0.65`` accounts for the subaperture fill factor.

    Parameters
    ----------
    images : array_like
        3D array of shape ``(n_frames, ny, nx)``.
    fwhm : float
        Full width at half maximum of the window, in pixels.
    display : bool, optional
        If True, plot the window and a contour overlay.

    Returns
    -------
    numpy.ndarray
        Windowed images, same shape as input.
    """
    windowed = []
    size = len(images[0])
    window = supergauss_half_width(fwhm, 4.0, size)
    for image in images:
        windowed.append(image * window)
    if display:
        plt.imshow(window)
        plt.colorbar()
        plt.contour(window, levels=[0.5], colors='w')
        plt.contour(images[-1] / np.max(images[-1]), levels=[0.1], colors='k')
        plt.axhline(size // 2)
        plt.axvline(size // 2)
        plt.show()
    return np.array(windowed)


def apply_window_groups(cubes, wavelength, pixel_scale, display=False):
    """Apply a super-Gaussian window to grouped image cubes.

    Parameters
    ----------
    cubes : numpy.ndarray
        4D array of shape ``(n_groups, n_frames, ny, nx)``.
    wavelength : float
        Wavelength in microns.
    pixel_scale : float
        Pixel scale in arcseconds per pixel.
    display : bool, optional
        If True, plot the window.

    Returns
    -------
    numpy.ndarray
        Windowed image cubes.
    """
    windowed = np.zeros(cubes.shape)
    window_width = wavelength * 1.0e-6 * 206265.0 / pixel_scale * 0.65
    size = len(cubes[0, 0])
    window = supergauss_fractional_width(window_width, 0.95, 4.0, size)
    for group_idx, group in enumerate(cubes):
        for frame_idx, frame in enumerate(group):
            windowed[group_idx, frame_idx] = frame * window
    if display:
        plt.imshow(window)
        plt.colorbar()
        plt.show()
    return np.array(windowed)


# ---------------------------------------------------------------------------
# FITS readers for JWST pipeline products
# ---------------------------------------------------------------------------

def _extract_header_info(fits_file):
    """Extract common header information from an open FITS file.

    Returns
    -------
    dict
        Dictionary with keys: images, bad_pixels, position_angle,
        mid_time, filter_name.
    """
    images = fits_file[1].data
    header_primary = fits_file[0].header
    header_sci = fits_file[1].header
    roll_ref = header_sci['ROLL_REF']
    v_parity = header_sci['VPARITY']
    v3i_yang = header_sci['V3I_YANG']
    position_angle = roll_ref - v3i_yang * v_parity
    filter_name = header_primary['FILTER']
    mid_time = header_primary['EXPMID']
    return {
        'images': images,
        'position_angle': position_angle,
        'mid_time': mid_time,
        'filter_name': filter_name,
    }


def read_calints(filename):
    """Read a JWST calints (CubeModel) FITS file.

    Parameters
    ----------
    filename : str
        Path to the calints FITS file.

    Returns
    -------
    images, dq_arrays, bad_pixel_maps, position_angle, mid_time, filter_name
    """
    fits_file = pyfits.open(filename)
    info = _extract_header_info(fits_file)
    input_model = datamodels.CubeModel(filename)
    dq_arrays = input_model.dq
    bad_pixel_maps = np.zeros(info['images'].shape)
    for flag in ['DO_NOT_USE']:
        bad_pixel_maps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
    return (info['images'], dq_arrays, bad_pixel_maps,
            info['position_angle'], info['mid_time'], info['filter_name'])


def read_jumpstep_images(filelist):
    """Read a list of JWST jump-step (QuadModel) FITS files.

    Parameters
    ----------
    filelist : list of str
        Paths to the FITS files.

    Returns
    -------
    images, dq_arrays, bad_pixel_maps, roll_angles, mid_times, filter_names
    """
    all_images = []
    all_dq = []
    all_bpmaps = []
    all_rolls = []
    all_times = []
    all_filters = []
    for filename in filelist:
        fits_file = pyfits.open(filename)
        info = _extract_header_info(fits_file)
        input_model = datamodels.QuadModel(filename)
        dq_arrays = input_model.groupdq
        bad_pixel_maps = np.zeros(info['images'].shape)
        for flag in ['DO_NOT_USE']:
            bad_pixel_maps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
        for idx in range(len(info['images'])):
            all_images.append(info['images'][idx])
            all_dq.append(dq_arrays[idx])
            all_bpmaps.append(bad_pixel_maps[idx])
            all_filters.append(info['filter_name'])
            all_rolls.append(info['position_angle'])
            all_times.append(info['mid_time'])
    return (np.array(all_images), np.array(all_dq), np.array(all_bpmaps),
            np.array(all_rolls), np.array(all_times), all_filters)


def read_calint_images(filelist):
    """Read a list of JWST calint (CubeModel) FITS files with extended DQ flags.

    Parameters
    ----------
    filelist : list of str
        Paths to the FITS files.

    Returns
    -------
    images, dq_arrays, bad_pixel_maps, roll_angles, mid_times, filter_names
    """
    all_images = []
    all_dq = []
    all_bpmaps = []
    all_rolls = []
    all_times = []
    all_filters = []
    flaglist = ['DO_NOT_USE', 'SATURATED', 'JUMP_DET', 'DROPOUT', 'OUTLIER',
                'AD_FLOOR', 'DEAD', 'HOT', 'WARM', 'NONLINEAR']
    for filename in filelist:
        fits_file = pyfits.open(filename)
        info = _extract_header_info(fits_file)
        input_model = datamodels.CubeModel(filename)
        dq_arrays = input_model.dq
        bad_pixel_maps = np.zeros(info['images'].shape)
        for flag in flaglist:
            bad_pixel_maps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
        for idx in range(len(info['images'])):
            all_images.append(info['images'][idx])
            all_dq.append(dq_arrays[idx])
            all_bpmaps.append(bad_pixel_maps[idx])
            all_filters.append(info['filter_name'])
            all_rolls.append(info['position_angle'])
            all_times.append(info['mid_time'])
    return (np.array(all_images), np.array(all_dq), np.array(all_bpmaps),
            np.array(all_rolls), np.array(all_times), all_filters)


def read_cal(filename):
    """Read a single JWST calibrated (ImageModel) FITS file.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    images, dq_arrays, bad_pixel_maps, position_angle, mid_time, filter_name
    """
    fits_file = pyfits.open(filename)
    info = _extract_header_info(fits_file)
    input_model = datamodels.ImageModel(filename)
    dq_arrays = input_model.dq
    bad_pixel_maps = np.zeros(info['images'].shape)
    for flag in ['DO_NOT_USE']:
        bad_pixel_maps[np.where(input_model.dq & dqflags.pixel[flag] > 0)] = 1.0
    return (info['images'], dq_arrays, bad_pixel_maps,
            info['position_angle'], info['mid_time'], info['filter_name'])


def parse_dqmap(filename):
    """Parse and print all data quality flags for an ImageModel.

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    list
        List of [y, x, flag1, flag2, ...] entries for flagged pixels.
    """
    input_model = datamodels.ImageModel(filename)
    dq_array = input_model.dq
    flag_values = list(dqflags.pixel.values())
    flag_names = list(dqflags.pixel.keys())
    flagged_pixels = []
    for y_idx in range(len(dq_array)):
        for x_idx in range(len(dq_array[y_idx])):
            entry = [y_idx, x_idx]
            for flag_idx, flag_val in enumerate(flag_values):
                if dq_array[y_idx, x_idx] & flag_val > 0:
                    entry.append(flag_names[flag_idx])
            if len(entry) > 2:
                print(entry)
                flagged_pixels.append(entry)
    return flagged_pixels


def parse_dqmap_ints(filename):
    """Parse and print all data quality flags for a CubeModel (first integration).

    Parameters
    ----------
    filename : str
        Path to the FITS file.

    Returns
    -------
    list
        List of [y, x, flag1, flag2, ...] entries for flagged pixels.
    """
    input_model = datamodels.CubeModel(filename)
    dq_array = input_model.dq
    flag_values = list(dqflags.pixel.values())
    flag_names = list(dqflags.pixel.keys())
    frame_idx = 0
    flagged_pixels = []
    for y_idx in range(len(dq_array[frame_idx])):
        for x_idx in range(len(dq_array[frame_idx, y_idx])):
            entry = [y_idx, x_idx]
            for flag_idx, flag_val in enumerate(flag_values):
                if dq_array[frame_idx, y_idx, x_idx] & flag_val > 0:
                    entry.append(flag_names[flag_idx])
            if len(entry) > 2:
                print(entry)
                flagged_pixels.append(entry)
    return flagged_pixels


def read_calims(filelist, input_dir=''):
    """Read a list of calibrated FITS files (ImageModel).

    Parameters
    ----------
    filelist : list of str
        File names to read.
    input_dir : str, optional
        Directory prefix to prepend to each file name.

    Returns
    -------
    images, dq_arrays, bad_pixel_maps, roll_angles, mid_times, filter_names
    """
    all_images = []
    all_dq = []
    all_bpmaps = []
    all_rolls = []
    all_times = []
    all_filters = []
    for filename in filelist:
        print(filename)
        if filename.endswith('.fits'):
            image, dq_arr, bpmap, roll, mid_time, filt = read_cal(input_dir + filename)
            all_images.append(image)
            all_dq.append(dq_arr)
            all_bpmaps.append(bpmap)
            all_rolls.append(roll)
            all_times.append(mid_time)
            all_filters.append(filt)
    return (np.array(all_images), np.array(all_dq), np.array(all_bpmaps),
            np.array(all_rolls), np.array(all_times), all_filters)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def background_subtract(images, r_inner=40, r_outer=50):
    """Subtract the mean background estimated from an annular region.

    Parameters
    ----------
    images : numpy.ndarray
        3D array of shape ``(n_frames, ny, nx)``.
    r_inner : float
        Inner radius of the background annulus in pixels.
    r_outer : float
        Outer radius of the background annulus in pixels.

    Returns
    -------
    numpy.ndarray
        Background-subtracted images.
    """
    n_frames, ny, nx = images.shape
    print(images.shape)
    distances = np.array([
        [np.sqrt((x - nx / 2.0) ** 2 + (y - ny / 2.0) ** 2)
         for x in range(nx)]
        for y in range(ny)
    ])
    annulus_mask = np.zeros(images[0].shape)
    annulus_mask[np.where((distances < r_outer) & (distances > r_inner))] = 1.0
    subtracted = []
    for image in images:
        image_copy = copy.deepcopy(image)
        background_pixels = image_copy[np.where(
            (distances < r_outer) & (distances > r_inner)
        )]
        mean_background = np.mean(background_pixels)
        image_copy -= mean_background
        subtracted.append(image_copy)
    plt.imshow(images[-1] ** 0.1)
    plt.contour(annulus_mask, levels=[1], colors='w')
    plt.show()
    return np.array(subtracted)


def fix_bad_pixels(images, bad_pixel_maps=None, box_size=1, display=False):
    """Replace bad pixels with the median of surrounding good pixels.

    Parameters
    ----------
    images : numpy.ndarray
        3D array of shape ``(n_frames, ny, nx)``.
    bad_pixel_maps : numpy.ndarray or None
        3D array of same shape, nonzero where pixels are bad.
        If None, images are returned unchanged.
    box_size : int, optional
        Half-size of the replacement box.
    display : bool, optional
        If True, show before/after comparison.

    Returns
    -------
    numpy.ndarray
        Corrected images.
    """
    if bad_pixel_maps is None:
        return images
    # Handle legacy call pattern with empty list
    if isinstance(bad_pixel_maps, list) and len(bad_pixel_maps) == 0:
        return images
    corrected = copy.deepcopy(images)
    print(images.shape)
    print(bad_pixel_maps.shape)
    print(corrected.shape)
    progress_bar = IntProgress(min=0, max=len(corrected))
    idisplay(progress_bar)
    progress_bar.value = 0
    for frame_idx in range(len(corrected)):
        bp_copy = copy.deepcopy(bad_pixel_maps[frame_idx])
        bad_positions = np.where(bp_copy != 0)
        pixel_idx = 0
        while False in np.unique(bp_copy == 0):
            y_pos, x_pos = bad_positions[0][pixel_idx], bad_positions[1][pixel_idx]
            y_lo = max(y_pos - box_size, 0)
            y_hi = min(y_pos + box_size + 1, len(bad_pixel_maps[frame_idx]))
            x_lo = max(x_pos - box_size, 0)
            x_hi = min(x_pos + box_size + 1, len(bad_pixel_maps[frame_idx][0]))
            good_neighbors = images[frame_idx, y_lo:y_hi, x_lo:x_hi][
                np.where(bp_copy[y_lo:y_hi, x_lo:x_hi] == 0.0)
            ]
            if len(good_neighbors) > 0:
                corrected[frame_idx, y_pos, x_pos] = np.median(good_neighbors)
                bp_copy[y_pos, x_pos] = 0
                bad_positions = np.where(bp_copy != 0)
                pixel_idx = 0
            else:
                pixel_idx += 1
        progress_bar.value += 1
    if display:
        plot_idx = 0
        fig = plt.figure(figsize=(12, 6))
        fig.add_subplot(131)
        plt.title('Input')
        plt.imshow(images[plot_idx] ** 0.1)
        fig.add_subplot(132)
        plt.title('Bad Pixel Map')
        plt.imshow(bad_pixel_maps[plot_idx] ** 0.1)
        fig.add_subplot(133)
        plt.title('Output')
        plt.imshow(corrected[plot_idx] ** 0.1)
        plt.show()
    return corrected


def center_interferogram(image, smooth_size, display=False):
    """Find the center of an interferogram by Gaussian smoothing.

    Parameters
    ----------
    image : numpy.ndarray
        2D image.
    smooth_size : float
        Gaussian smoothing sigma in pixels.
    display : bool, optional
        If True, show the smoothed image and center.

    Returns
    -------
    y, x : int
        Center coordinates.
    """
    image_copy = copy.deepcopy(image)
    image_copy[np.where(np.isnan(image_copy))] = 0.0
    smoothed = ndimage.gaussian_filter(image_copy, smooth_size)
    y_center, x_center = np.unravel_index(np.argmax(smoothed.flatten()), smoothed.shape)
    if display:
        plt.figure()
        plt.imshow(smoothed, origin='lower')
        plt.scatter(x_center, y_center)
        plt.show()
    return y_center, x_center


def subframe(images, subframe_size=64, smooth_sigma=5):
    """Extract a centered subframe from each image.

    Parameters
    ----------
    images : numpy.ndarray
        3D ``(n_frames, ny, nx)`` or 4D ``(n_groups, n_frames, ny, nx)`` array.
    subframe_size : int, optional
        Side length of the extracted subframe.
    smooth_sigma : float, optional
        Gaussian smoothing sigma for centering.

    Returns
    -------
    numpy.ndarray
        Subframed images.
    """
    half = subframe_size // 2
    dims = images.shape
    if len(dims) == 4:
        y_cen, x_cen = center_interferogram(
            np.nanmedian(np.nanmedian(images, axis=0), axis=0), smooth_sigma
        )
        subframed = []
        for cube in images:
            group = []
            for frame in cube:
                group.append(frame[y_cen - half:y_cen + half,
                                   x_cen - half:x_cen + half])
            subframed.append(group)
    else:
        y_cen, x_cen = center_interferogram(
            np.nanmedian(images, axis=0), smooth_sigma
        )
        print(y_cen, x_cen)
        subframed = []
        for frame in images:
            subframed.append(frame[y_cen - half:y_cen + half,
                                   x_cen - half:x_cen + half])
    return np.array(subframed)


def subframe_circular(images, subframe_size=64, smooth_sigma=5):
    """Extract a centered circular subframe from each image.

    Like :func:`subframe`, but zeros pixels outside a circular aperture.

    Parameters
    ----------
    images : numpy.ndarray
        3D or 4D image array.
    subframe_size : int, optional
        Side length and diameter of the circular region.
    smooth_sigma : float, optional
        Gaussian smoothing sigma for centering.

    Returns
    -------
    numpy.ndarray
        Subframed images with circular masking.
    """
    half = subframe_size // 2
    dims = images.shape
    if len(dims) == 4:
        median_image = np.nanmedian(np.nanmedian(images, axis=0), axis=0)
        y_cen, x_cen = center_interferogram(median_image, smooth_sigma)
        distances = np.array([
            [np.sqrt((col - x_cen) ** 2 + (row - y_cen) ** 2)
             for col in range(len(median_image))]
            for row in range(len(median_image))
        ])
        subframed = []
        for cube in images:
            group = []
            for frame in cube:
                frame_copy = copy.deepcopy(frame)
                frame_copy[np.where(distances > half)] = 0.0
                group.append(frame_copy[y_cen - half:y_cen + half,
                                        x_cen - half:x_cen + half])
            subframed.append(group)
    else:
        median_image = np.nanmedian(images, axis=0)
        y_cen, x_cen = center_interferogram(median_image, smooth_sigma)
        distances = np.array([
            [np.sqrt((col - x_cen) ** 2 + (row - y_cen) ** 2)
             for col in range(len(median_image))]
            for row in range(len(median_image))
        ])
        subframed = []
        for frame in images:
            frame_copy = copy.deepcopy(frame)
            frame_copy[np.where(distances > half)] = 0.0
            subframed.append(frame_copy[y_cen - half:y_cen + half,
                                        x_cen - half:x_cen + half])
    return np.array(subframed)
