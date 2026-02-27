"""Utility functions for image centering and Fourier operations."""

import numpy as np
from scipy import ndimage
from scipy.signal import medfilt2d


def find_psf_center(image, verbose=True, n_iterations=10):
    """Locate the center of a PSF using iterative windowed centroiding.

    Uses an iterative method with a window of shrinking size to
    minimize possible biases (non-uniform background, hot pixels, etc).

    Parameters
    ----------
    image : numpy.ndarray
        2D image containing the PSF.
    verbose : bool, optional
        Print convergence info at each iteration.
    n_iterations : int, optional
        Number of iterations (default 10 is good for 512x512 images).

    Returns
    -------
    xc, yc : float
        Estimated center coordinates.
    """
    temp = image.copy()
    background = np.median(temp)
    temp -= background
    median_filtered = medfilt2d(temp, 3)
    size_y, size_x = median_filtered.shape
    xc, yc = size_x / 2, size_y / 2

    signal = np.zeros_like(image)
    signal[median_filtered > 0.1 * median_filtered.max()] = 1.0

    for iteration in range(n_iterations):
        window_size = size_x / 2 / (1.0 + (0.1 * size_x / 2 * iteration / (4 * n_iterations)))
        x0 = max(int(0.5 + xc - window_size), 0)
        y0 = max(int(0.5 + yc - window_size), 0)
        x1 = min(int(0.5 + xc + window_size), size_x)
        y1 = min(int(0.5 + yc + window_size), size_y)

        mask = np.zeros_like(image)
        mask[y0:y1, x0:x1] = 1.0

        profile_x = (median_filtered * mask * signal).sum(axis=0)
        profile_y = (median_filtered * mask * signal).sum(axis=1)

        xc = (profile_x * np.arange(size_x)).sum() / profile_x.sum()
        yc = (profile_y * np.arange(size_y)).sum() / profile_y.sum()

        if verbose:
            print(f"it #{iteration + 1:2d} center = ({xc:.2f}, {yc:.2f})")

    return xc, yc


def gauss_smooth_image(image, wavelength, diameter, pixel_scale):
    """Smooth an image with a Gaussian kernel matched to the diffraction limit.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    wavelength : float
        Wavelength in microns.
    diameter : float
        Aperture diameter in meters.
    pixel_scale : float
        Pixel scale in arcseconds per pixel.

    Returns
    -------
    numpy.ndarray
        Smoothed image.
    """
    sigma = (wavelength * 1.0e-6) / diameter * 206265.0 / pixel_scale / 2.35482
    return ndimage.gaussian_filter(image, sigma=sigma)


def get_center(image, wavelength, diameter, pixel_scale):
    """Find the center of an image by Gaussian smoothing and peak finding.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    wavelength : float
        Wavelength in microns.
    diameter : float
        Aperture diameter in meters.
    pixel_scale : float
        Pixel scale in arcseconds per pixel.

    Returns
    -------
    y, x : int
        Peak pixel coordinates.
    """
    smoothed = gauss_smooth_image(image, wavelength, diameter, pixel_scale)
    pos = np.where(smoothed == np.max(smoothed))
    return pos[0][0], pos[1][0]


def fourier_center(image, dy_shift, dx_shift):
    """Shift an image in Fourier space by a sub-pixel offset.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    dy_shift : float
        Shift in the y direction (pixels).
    dx_shift : float
        Shift in the x direction (pixels).

    Returns
    -------
    ft_image : numpy.ndarray
        Fourier transform of the input image.
    ft_shifted : numpy.ndarray
        Fourier transform with phase shift applied.
    """
    ft_image = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(ft_image)))
    phase_shift = np.outer(
        np.exp(-2.0 * np.pi * freqs * dy_shift * 1.0j),
        np.exp(-2.0 * np.pi * freqs * dx_shift * 1.0j),
    )
    return ft_image, ft_image * phase_shift
