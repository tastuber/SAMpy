"""Fourier-plane analysis: closure phases, squared visibilities, and covariances."""

import cmath
import copy
import math
import os

import h5py
import numpy as np
import astropy.io.fits as pyfits
from matplotlib import pyplot as plt
from scipy import ndimage, interpolate
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from sampy.reduction import supergauss_fractional_width
from sampy.utils import find_psf_center, gauss_smooth_image






def gen_window(window_type, box_size, wavelength_um, pixel_scale):
    """Generate a 2-D window function for apodizing images before FFT.

    Parameters
    ----------
    window_type : {'sg', 'h', 'nw'}
        Window type: ``'sg'`` for super-Gaussian, ``'h'`` for Hanning,
        ``'nw'`` for no window (uniform).
    box_size : int
        Side length of the (square) image in pixels.
    wavelength_um : float
        Observing wavelength in microns.  Used only for ``'sg'``.
    pixel_scale : float
        Plate scale in arcsec/pixel.  Used only for ``'sg'``.

    Returns
    -------
    window : ndarray, shape (box_size, box_size)
        The 2-D window function.
    """
    if window_type == 'sg':
        # Diffraction-limited PSF width in pixels, scaled by 0.65
        window_width = wavelength_um * 1.0e-6 * 206265.0 / pixel_scale * 0.65
        window = supergauss_fractional_width(window_width, 0.95, 8.0, box_size)
    elif window_type == 'h':
        window = np.outer(np.hanning(box_size), np.hanning(box_size))
    elif window_type == 'nw':
        window = np.ones((box_size, box_size))
    else:
        raise ValueError(
            f"Unknown window_type '{window_type}'; expected 'sg', 'h', or 'nw'."
        )
    return window

def fft_image(image, nx, ny):
    """Zero-pad an image to (ny, nx) and compute its centered 2-D FFT.

    The image is zero-padded symmetrically, then a centered FFT is computed
    via ``fftshift(fft2(fftshift(...)))``.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input 2-D image.
    nx : int
        Desired padded width (columns).
    ny : int
        Desired padded height (rows).

    Returns
    -------
    ft : complex ndarray, shape (ny, nx)
        Centered Fourier transform of the padded image.
    """
    pad_y = (ny - image.shape[0]) // 2
    pad_x = (nx - image.shape[1]) // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)))
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(padded)))
    return ft

def make_pspec(image, nx, ny):
    """Compute the power spectrum of a single image.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input 2-D image.
    nx : int
        Padded width passed to :func:`fft_image`.
    ny : int
        Padded height passed to :func:`fft_image`.

    Returns
    -------
    power_spectrum : ndarray, shape (ny, nx)
        Squared modulus of the centered FFT.
    """
    ft = fft_image(image, nx, ny)
    power_spectrum = np.abs(ft) ** 2
    return power_spectrum

def make_cpcoords(mask_dir):
    """Load closure-phase splodge pixel coordinates from the mask directory.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory (must end with ``/``).

    Returns
    -------
    cp_coords : ndarray, shape (n_triangles, 3, 2)
        Pixel coordinates of the three splodge centres for each closing
        triangle.
    """
    try:
        cp_coords = np.array(pyfits.getdata(mask_dir + 'cp_pix.fits'))
    except IOError:
        cp_coords = np.array(pyfits.getdata(mask_dir + 'cp_coords.fits'))
    return cp_coords

def make_v2coords(mask_dir):
    """Load squared-visibility splodge pixel coordinates from the mask directory.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory.

    Returns
    -------
    v2_coords : ndarray, shape (n_baselines, 2)
        Pixel coordinates of each baseline splodge centre.
    """
    try:
        v2_coords = np.array(pyfits.getdata(mask_dir + 'bl_pix.fits'))
    except IOError:
        v2_coords = np.array(pyfits.getdata(mask_dir + 'bl_coords.fits'))
    return v2_coords



def make_cviscoords(mask_dir):
    """Load complex-visibility splodge pixel coordinates from the mask directory.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory.

    Returns
    -------
    cvis_coords : ndarray, shape (n_baselines, 2)
        Pixel coordinates of each complex-visibility splodge centre.
    """
    try:
        cvis_coords = np.array(pyfits.getdata(mask_dir + 'cvis_pix.fits'))
    except IOError:
        cvis_coords = np.array(pyfits.getdata(mask_dir + 'cvis_coords.fits'))
    return cvis_coords


def find_v2_pix(mask_dir, v2type='multi'):
    """Load per-baseline pixel sampling coordinates for squared visibilities.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory.
    v2type : {'multi', 'single'}
        ``'multi'``  — load multi-pixel splodge regions from FITS files
        (Monnier method).
        ``'single'`` — use a single pixel at the rounded splodge centre.

    Returns
    -------
    baseline_pixels : list of ndarray
        One array per baseline.  Each array has shape ``(n_pixels, 2)``
        giving the (x, y) pixel coordinates that sample that baseline.
    """
    v2_coords = make_v2coords(mask_dir)
    baseline_pixels = []
    if v2type == 'single':
        for i in range(len(v2_coords)):
            baseline_pixels.append(
                np.array([np.round(v2_coords[i])], dtype='int')
            )
    elif v2type == 'multi':
        for i in range(len(v2_coords)):
            pixel_region = pyfits.getdata(mask_dir + 'v2_ind' + str(i) + '.fits')
            baseline_pixels.append(pixel_region)
    return baseline_pixels


def find_cvis_pix(mask_dir, cvistype='multi'):
    """Load per-baseline pixel sampling coordinates for complex visibilities.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory.
    cvistype : {'multi', 'single'}
        ``'multi'``  — load multi-pixel splodge regions from FITS files.
        ``'single'`` — use a single pixel at the rounded splodge centre.

    Returns
    -------
    baseline_pixels : list of ndarray
        One array per baseline.  Each array has shape ``(n_pixels, 2)``
        giving the (x, y) pixel coordinates that sample that baseline.
    """
    cvis_coords = make_cviscoords(mask_dir)
    baseline_pixels = []
    if cvistype == 'single':
        for i in range(len(cvis_coords)):
            baseline_pixels.append(
                np.array([np.round(cvis_coords[i])], dtype='int')
            )
    elif cvistype == 'multi':
        for i in range(len(cvis_coords)):
            pixel_region = pyfits.getdata(mask_dir + 'cvis_ind' + str(i) + '.fits')
            baseline_pixels.append(pixel_region)
    return baseline_pixels



def find_tris_multi_pavg(cp_coords, mask_dir, ny=256, nx=256, meters=False,
                         uv=False, pscam=0.065, lamc=3.8, redo_calc=False):
    """Load per-vertex pixel regions for the phasor-averaged closure phase method.

    Each closing triangle has three vertices; this function loads the
    multi-pixel splodge region for each vertex from pre-computed FITS files.

    Parameters
    ----------
    cp_coords : ndarray, shape (n_triangles, 3, 2)
        Closure-phase splodge centre coordinates (from :func:`make_cpcoords`).
    mask_dir : str
        Path to the mask directory containing ``ind*_vert*.fits`` files.
    ny, nx : int
        FFT grid dimensions (unused here, kept for API compatibility).
    meters, uv, pscam, lamc, redo_calc : various
        Unused; kept for API compatibility with :func:`find_tris_multi`.

    Returns
    -------
    triangle_vertices : list of list of ndarray
        ``triangle_vertices[t]`` is a 3-element list, where each element is
        an ndarray of shape ``(n_pixels, 2)`` giving the pixel coordinates
        for that vertex of closing triangle ``t``.
    """
    triangle_vertices = []
    for tri_idx in range(len(cp_coords)):
        file_base = mask_dir + 'ind' + str(tri_idx) + '_vert'
        vert0 = pyfits.getdata(file_base + '0.fits')
        vert1 = pyfits.getdata(file_base + '1.fits')
        vert2 = pyfits.getdata(file_base + '2.fits')
        triangle_vertices.append([vert0, vert1, vert2])
    return triangle_vertices







def find_tris_multi(cp_coords, mask_dir, ny=256, nx=256, meters=False,
                    uv=False, pscam=0.065, lamc=3.8, redo_calc=False):
    """Load or compute multi-pixel triangle sampling for closure phases.

    For each closing triangle of baselines, finds all pixel-triplets
    (one pixel from each splodge) satisfying the closure relation
    u1+u2+u3 = 0, v1+v2+v3 = 0.  Results are cached to FITS files
    in ``mask_dir`` on first run.

    Parameters
    ----------
    cp_coords : ndarray, shape (n_triangles, 3, 2)
        Closure-phase splodge centre coordinates (from :func:`make_cpcoords`).
    mask_dir : str
        Path to the mask directory.
    ny, nx : int
        FFT grid dimensions.
    meters : bool
        If True, convert pixel coordinates to metres in the (u, v) plane.
    uv : bool
        If True (and ``meters`` is True), flip u to match the true (u, v)
        convention.
    pscam : float
        Camera pixel scale in arcsec/pixel (used when ``meters=True``).
    lamc : float
        Central wavelength in microns (used when ``meters=True``).
    redo_calc : bool
        Force recomputation even if cached FITS files exist.

    Returns
    -------
    pixel_triangles : list of ndarray
        ``pixel_triangles[t]`` has shape ``(n_valid_triangles, 3, 2)`` giving
        the pixel coordinates for each valid pixel-triplet of closing
        triangle ``t``.  The list is ragged (different triangles may have
        different numbers of valid pixel-triplets).
    """
    center = np.array([ny // 2, nx // 2])
    pixel_triangles = []

    for tri_idx in range(len(cp_coords)):
        cache_file = mask_dir + 'cpsamp_ind_' + str(tri_idx) + '.fits'

        if not os.path.isfile(cache_file) or redo_calc:
            print('\n Calculating triangle: '
                  + str(tri_idx + 1) + ' of ' + str(len(cp_coords)))
            file_base = mask_dir + 'ind' + str(tri_idx) + '_vert'
            vert0_pixels = pyfits.getdata(file_base + '0.fits')
            vert1_pixels = pyfits.getdata(file_base + '1.fits')
            vert2_pixels = pyfits.getdata(file_base + '2.fits')

            valid_triplets = []
            for pix0 in tqdm(vert0_pixels):
                for pix1 in vert1_pixels:
                    # Closure relation: displacement vectors must sum to zero
                    disp0 = pix0 - center
                    disp1 = pix1 - center
                    disp2 = -disp0 - disp1
                    pix2 = disp2 + center

                    # Check if the predicted third pixel is in vertex 2's region
                    match_row = np.where(vert2_pixels[:, 0] == pix2[0])[0]
                    match_col = np.where(vert2_pixels[:, 1] == pix2[1])[0]
                    if np.any(np.isin(match_row, match_col)):
                        valid_triplets.append(
                            np.array([pix0, pix1, pix2], dtype='int')
                        )

            valid_triplets = np.array(valid_triplets)
            pyfits.writeto(cache_file, valid_triplets, overwrite=True)
        else:
            valid_triplets = pyfits.getdata(cache_file)

        pixel_triangles.append(valid_triplets)

    if meters:
        # Convert pixel offsets to physical (u, v) coordinates in metres
        pixel_triangles_m = copy.deepcopy(pixel_triangles)
        pixel_to_meters = (1.0 / (float(nx) * pscam)) * 206265.0 * lamc * 1e-06
        for tri_idx in range(len(pixel_triangles_m)):
            pixel_triangles_m[tri_idx] = (
                (pixel_triangles_m[tri_idx] - int(ny / 2)) * pixel_to_meters
            )
            if uv:
                # Flip u coordinate to match true (u, v) convention
                pixel_triangles_m[tri_idx][:, :, 0] *= -1
        return pixel_triangles_m

    return pixel_triangles

def make_blens(mask_dir):
    """Compute baseline lengths from (u, v) sampling coordinates.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory containing ``bl_uvs.fits``.

    Returns
    -------
    baseline_lengths : ndarray, shape (n_baselines,)
        Euclidean length of each baseline.
    """
    baseline_uvs = np.array(pyfits.getdata(mask_dir + 'bl_uvs.fits'))
    baseline_lengths = np.sqrt(baseline_uvs[:, 0] ** 2 + baseline_uvs[:, 1] ** 2)
    return baseline_lengths


def make_cplens(mask_dir):
    """Compute baseline lengths for each leg of every closing triangle.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory containing ``cp_uvs.fits``.

    Returns
    -------
    cp_baseline_lengths : ndarray, shape (n_triangles, 3)
        Euclidean length of each baseline in every closing triangle.
    """
    cp_uvs = np.array(pyfits.getdata(mask_dir + 'cp_uvs.fits'))
    cp_baseline_lengths = np.sqrt(cp_uvs[:, :, 0] ** 2 + cp_uvs[:, :, 1] ** 2)
    return cp_baseline_lengths

def calc_cps_single_DFT(images, mask_dir, nx=256, ny=256, display=False,
                        use_weights=True):
    """Compute closure phases via DFT using a single pixel per splodge.

    Uses a pre-computed DFT matrix to transform images directly into
    complex triple products, then extracts closure phases.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory containing ``cpDFTmat_sing_*.fits``.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        Unused; kept for API consistency.
    use_weights : bool
        If True, weight the covariance by triple amplitudes.

    Returns
    -------
    result : dict
        ``'raw'``            — per-image bispectra, shape (n_images, n_triangles).
        ``'closure_phases'`` — mean closure phases in degrees.
        ``'covariance'``     — covariance matrix.
        ``'variance'``       — diagonal of covariance.
        ``'std_error'``      — standard error of the mean.
    """
    dft_re, dft_im = pyfits.getdata(
        mask_dir + 'cpDFTmat_sing_' + str(nx) + '.fits'
    )
    dft_matrix = dft_re + dft_im * 1.0j

    bispectra_all = []
    for image in tqdm(images):
        ct_phasors = np.dot(dft_matrix, image.flatten()) / np.sum(image)
        n_triangles = len(ct_phasors) // 3
        bispectra = np.empty(n_triangles, dtype=complex)
        for tri in range(n_triangles):
            bispectra[tri] = np.prod(ct_phasors[tri * 3 : tri * 3 + 3])
        bispectra_all.append(bispectra)

    bispectra_all = np.array(bispectra_all)
    closure_phase_per_image = np.angle(bispectra_all, deg=True)
    triple_amp_per_image = np.abs(bispectra_all)
    closure_phases = np.angle(np.nanmean(bispectra_all, axis=0), deg=True)
    covariance, variance, std_error = gen_cov(
        closure_phases, closure_phase_per_image,
        weights=triple_amp_per_image, use_weights=use_weights,
    )
    return {
        'raw': bispectra_all,
        'closure_phases': closure_phases,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
    }

def calc_cps_single(images, mask_dir, nx=256, ny=256, display=False,
                    use_weights=True):
    """Compute closure phases using a single pixel at each splodge centre.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry for the first image.
    use_weights : bool
        If True, weight the covariance by triple amplitudes.

    Returns
    -------
    result : dict
        ``'raw'``            — per-image bispectra, shape (n_images, n_triangles).
        ``'closure_phases'`` — mean closure phases in degrees.
        ``'covariance'``     — covariance matrix.
        ``'variance'``       — diagonal of covariance.
        ``'std_error'``      — standard error of the mean.
    """
    cp_coords = np.array(np.round(make_cpcoords(mask_dir), 1), dtype='int')
    bispectra_all = []

    for img_idx, image in enumerate(tqdm(images)):
        ft = fft_image(image, nx, ny)

        # For each triangle, multiply the FT values at the 3 splodge centres
        n_triangles = len(cp_coords)
        bispectra = np.empty(n_triangles, dtype=complex)
        for tri_idx, triangle in enumerate(cp_coords):
            product = 1.0 + 0.0j
            for vertex in range(3):
                product *= ft[triangle[vertex, 1], triangle[vertex, 0]]
            bispectra[tri_idx] = product
        bispectra_all.append(bispectra)

        if display and img_idx == 0:
            plt.imshow(np.abs(ft) ** 0.1)
            plt.scatter(cp_coords[:, :, 0], cp_coords[:, :, 1],
                        edgecolors='k', facecolors='None')
            sample_indices = np.random.choice(len(cp_coords), 5)
            for si in sample_indices:
                plt.plot([cp_coords[si, jj, 0] for jj in [0, 1, 2, 0]],
                         [cp_coords[si, jj, 1] for jj in [0, 1, 2, 0]], c='w')
            plt.show()

    bispectra_all = np.array(bispectra_all)
    closure_phase_per_image = np.angle(bispectra_all, deg=True)
    triple_amp_per_image = np.abs(bispectra_all)
    closure_phases = np.angle(np.nanmean(bispectra_all, axis=0), deg=True)
    covariance, variance, std_error = gen_cov(
        closure_phases, closure_phase_per_image,
        weights=triple_amp_per_image, use_weights=use_weights,
    )
    return {
        'raw': bispectra_all,
        'closure_phases': closure_phases,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
    }



def calc_cps_pavg_image(image, triangle_vertices, image_index, nx=256, ny=256,
                        display=False, save=False, fout=None):
    """Compute bispectra for a single image using phasor-averaged method.

    For each closing triangle, the complex visibility at each vertex is
    averaged over its multi-pixel splodge region first, then the three
    mean phasors are multiplied to form the bispectrum.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image.
    triangle_vertices : list of list of ndarray
        Per-triangle vertex pixel regions (from :func:`find_tris_multi_pavg`).
    image_index : int
        Index of this image in the stack (used for HDF5 keys).
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry.
    save : bool
        If True, write per-vertex visibilities to ``fout``.
    fout : h5py.File or None
        Open HDF5 file for writing.

    Returns
    -------
    bispectra : ndarray, shape (n_triangles,)
        Mean bispectrum for each closing triangle.
    """
    ft = fft_image(image, nx, ny)
    zero_freq = ft[ny // 2, nx // 2]
    bispectra = []

    for tri_idx, vertices in enumerate(triangle_vertices):
        # Average complex visibility over each vertex's pixel region
        mean_vis_per_vertex = []
        per_vertex_vis = []
        for vert in range(3):
            pixel_vis = ft[vertices[vert][:, 1], vertices[vert][:, 0]] / zero_freq
            per_vertex_vis.append(pixel_vis)
            mean_vis_per_vertex.append(np.mean(pixel_vis))

        bispectrum = np.prod(mean_vis_per_vertex)

        if save:
            for vert in range(3):
                fout['int' + str(image_index) + '/tri' + str(tri_idx)
                     + '/ind' + str(vert)] = np.array(per_vertex_vis[vert],
                                                       dtype='complex')
        bispectra.append(bispectrum)

        if display:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            plt.imshow(np.angle(ft, deg=True), origin='lower')
            for vert in range(3):
                plt.scatter(vertices[vert][:, 0], vertices[vert][:, 1],
                            edgecolors='k', facecolors='None')
            plt.show()

    return np.array(bispectra)

def calc_cps_multi_image(image, pixel_triangles, image_index, nx=256, ny=256,
                         display=False, save=False, fout=None):
    """Compute bispectra for a single image using multi-pixel triangle sampling.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image.
    pixel_triangles : list of ndarray
        Per-closing-triangle pixel triplets (from :func:`find_tris_multi`).
    image_index : int
        Index of this image in the stack (used for HDF5 keys).
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry for the first triangle.
    save : bool
        If True, write per-pixel bispectra to ``fout``.
    fout : h5py.File or None
        Open HDF5 file for writing (required if ``save`` is True).

    Returns
    -------
    bispectra : ndarray, shape (n_triangles,)
        Mean bispectrum for each closing triangle.
    """
    ft = fft_image(image, nx, ny)
    zero_freq = ft[ny // 2, nx // 2]
    bispectra = []

    for tri_idx, triplets in enumerate(pixel_triangles):
        if display:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            plt.imshow(np.abs(ft) ** 0.1, origin='lower')
            plt.scatter(triplets[:, :, 0], triplets[:, :, 1],
                        edgecolors='k', facecolors='None')
            sample_indices = np.random.choice(len(triplets), 5)
            for si in sample_indices:
                plt.plot([triplets[si, jj, 0] for jj in [0, 1, 2, 0]],
                         [triplets[si, jj, 1] for jj in [0, 1, 2, 0]], c='w')
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()

        # For each pixel-triplet, compute the product of 3 normalized
        # complex visibilities, then average over all triplets
        n_triplets = len(triplets)
        per_triplet_bispectra = np.empty(n_triplets, dtype=complex)
        for p in range(n_triplets):
            vis_product = 1.0 + 0.0j
            for vertex in range(3):
                vis_product *= ft[triplets[p, vertex, 1],
                                  triplets[p, vertex, 0]] / zero_freq
            per_triplet_bispectra[p] = vis_product

        mean_bispectrum = np.mean(per_triplet_bispectra)

        if save:
            fout['int' + str(image_index) + '/tri' + str(tri_idx)] = \
                np.array(per_triplet_bispectra, dtype='complex')

        bispectra.append(mean_bispectrum)

    return np.array(bispectra)

def calc_cps_multi_groupimage(image, pixel_triangles, image_index, group_index,
                              nx=256, ny=256, display=False, save=False,
                              fout=None):
    """Compute bispectra for a single image within a group (integration).

    Same algorithm as :func:`calc_cps_multi_image` but writes HDF5 keys
    that include a group index for grouped/cubed data.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image.
    pixel_triangles : list of ndarray
        Per-closing-triangle pixel triplets (from :func:`find_tris_multi`).
    image_index : int
        Integration index (used for HDF5 keys).
    group_index : int
        Group index within the integration (used for HDF5 keys).
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry.
    save : bool
        If True, write per-pixel bispectra to ``fout``.
    fout : h5py.File or None
        Open HDF5 file for writing.

    Returns
    -------
    bispectra : ndarray, shape (n_triangles,)
        Mean bispectrum for each closing triangle.
    """
    ft = fft_image(image, nx, ny)
    zero_freq = ft[ny // 2, nx // 2]
    bispectra = []

    for tri_idx, triplets in enumerate(pixel_triangles):
        if display:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            plt.imshow(np.abs(ft) ** 0.1, origin='lower')
            plt.scatter(triplets[:, :, 0], triplets[:, :, 1],
                        edgecolors='k', facecolors='None')
            sample_indices = np.random.choice(len(triplets), 5)
            for si in sample_indices:
                plt.plot([triplets[si, jj, 0] for jj in [0, 1, 2, 0]],
                         [triplets[si, jj, 1] for jj in [0, 1, 2, 0]], c='w')
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()

        n_triplets = len(triplets)
        per_triplet_bispectra = np.empty(n_triplets, dtype=complex)
        for p in range(n_triplets):
            vis_product = 1.0 + 0.0j
            for vertex in range(3):
                vis_product *= ft[triplets[p, vertex, 1],
                                  triplets[p, vertex, 0]] / zero_freq
            per_triplet_bispectra[p] = vis_product

        mean_bispectrum = np.mean(per_triplet_bispectra)
        if save:
            fout['int' + str(image_index) + '/group' + str(group_index)
                 + '/tri' + str(tri_idx)] = per_triplet_bispectra
        bispectra.append(mean_bispectrum)

    return np.array(bispectra)

def calc_cps_multi(images, mask_dir, display=True, nx=256, ny=256,
                   use_weights=True, save_allpix=False, filebase='',
                   redo_calc=False):
    """Compute closure phases using multi-pixel triangle sampling.

    For each closing triangle, averages the bispectrum over all valid
    pixel-triplets (Monnier method), then derives closure phases,
    triple amplitudes, and covariance from the image stack.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory.
    display : bool
        If True, plot the sampling geometry for the first image.
    nx, ny : int
        FFT grid dimensions.
    use_weights : bool
        If True, weight the covariance by triple amplitudes.
    save_allpix : bool
        If True, save per-pixel bispectra to an HDF5 file.
    filebase : str
        Base path for output files (FITS and HDF5).
    redo_calc : bool
        Force recomputation of triangle sampling coordinates.

    Returns
    -------
    result : dict
        ``'raw'``            — list of per-image bispectrum arrays (complex).
        ``'closure_phases'`` — mean closure phases in degrees, shape (n_triangles,).
        ``'triple_amps'``    — mean triple amplitudes, shape (n_triangles,).
        ``'covariance'``     — covariance matrix, or None if single image.
        ``'variance'``       — variance (diagonal of covariance), or None.
        ``'std_error'``      — standard error of the mean, or None.
    """
    cp_coords = make_cpcoords(mask_dir)
    pixel_triangles = find_tris_multi(cp_coords, mask_dir,
                                      redo_calc=redo_calc, nx=nx, ny=ny)

    if save_allpix and not filebase:
        raise ValueError("filebase is required when save_allpix=True")
    if save_allpix:
        fout = h5py.File(filebase + '_pixtri.hdf5', 'w')
    else:
        fout = None

    bispectra_all = []
    for img_idx, image in tqdm(enumerate(images)):
        bispectra = calc_cps_multi_image(
            image, pixel_triangles, img_idx,
            nx=nx, ny=ny, display=display,
            save=save_allpix, fout=fout,
        )
        display = False  # only display for the first image
        bispectra_all.append(bispectra)

    bispectra_all = np.array(bispectra_all)  # (n_images, n_triangles), complex
    closure_phase_per_image = np.angle(bispectra_all, deg=True)
    triple_amp_per_image = np.abs(bispectra_all)
    mean_triple_amps = np.abs(np.mean(bispectra_all, axis=0))
    closure_phases = np.angle(np.mean(bispectra_all, axis=0), deg=True)

    if save_allpix:
        fout.close()

    if len(images) > 1:
        covariance, variance, std_error = gen_cov(
            closure_phases, closure_phase_per_image,
            weights=triple_amp_per_image, use_weights=use_weights,
        )
        pyfits.writeto(
            filebase + '_bspecs.fits',
            np.array([np.real(bispectra_all), np.imag(bispectra_all)]),
            overwrite=True,
        )
        pyfits.writeto(filebase + '_cps.fits', closure_phases, overwrite=True)
        pyfits.writeto(filebase + '_cpcov.fits', covariance, overwrite=True)
    else:
        covariance, variance, std_error = None, None, None

    return {
        'raw': bispectra_all,
        'closure_phases': closure_phases,
        'triple_amps': mean_triple_amps,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
    }

def calc_cps_pavg(images, mask_dir, display=True, nx=256, ny=256,
                  use_weights=True, save_allpix=False, filebase='',
                  redo_calc=False):
    """Compute closure phases using the phasor-averaged method.

    Like :func:`calc_cps_multi`, but averages the complex visibility over
    each vertex's pixel region *before* forming the triple product
    (phasor averaging).

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory.
    display : bool
        If True, plot the sampling geometry for the first image.
    nx, ny : int
        FFT grid dimensions.
    use_weights : bool
        If True, weight the covariance by triple amplitudes.
    save_allpix : bool
        If True, save per-vertex visibilities to an HDF5 file.
    filebase : str
        Base path for output files (FITS and HDF5).
    redo_calc : bool
        Force recomputation of sampling coordinates.

    Returns
    -------
    result : dict
        ``'raw'``            — per-image bispectra (complex).
        ``'closure_phases'`` — mean closure phases in degrees.
        ``'triple_amps'``    — mean triple amplitudes.
        ``'covariance'``     — covariance matrix, or None if single image.
        ``'variance'``       — variance, or None.
        ``'std_error'``      — standard error, or None.
    """
    cp_coords = make_cpcoords(mask_dir)
    triangle_vertices = find_tris_multi_pavg(cp_coords, mask_dir,
                                              redo_calc=redo_calc, nx=nx, ny=ny)

    if save_allpix and not filebase:
        raise ValueError("filebase is required when save_allpix=True")
    if save_allpix:
        fout = h5py.File(filebase + '_pixtri.hdf5', 'w')
    else:
        fout = None

    bispectra_all = []
    for img_idx, image in tqdm(enumerate(images)):
        bispectra = calc_cps_pavg_image(
            image, triangle_vertices, img_idx,
            nx=nx, ny=ny, display=display,
            save=save_allpix, fout=fout,
        )
        display = False
        bispectra_all.append(bispectra)

    bispectra_all = np.array(bispectra_all)
    closure_phase_per_image = np.angle(bispectra_all, deg=True)
    triple_amp_per_image = np.abs(bispectra_all)
    mean_triple_amps = np.abs(np.mean(bispectra_all, axis=0))
    closure_phases = np.angle(np.mean(bispectra_all, axis=0), deg=True)

    if save_allpix:
        fout.close()

    if len(images) > 1:
        covariance, variance, std_error = gen_cov(
            closure_phases, closure_phase_per_image,
            weights=triple_amp_per_image, use_weights=use_weights,
        )
        pyfits.writeto(
            filebase + '_bspecs.fits',
            np.array([np.real(bispectra_all), np.imag(bispectra_all)]),
            overwrite=True,
        )
        pyfits.writeto(filebase + '_cps.fits', closure_phases, overwrite=True)
        pyfits.writeto(filebase + '_cpcov.fits', covariance, overwrite=True)
    else:
        covariance, variance, std_error = None, None, None

    return {
        'raw': bispectra_all,
        'closure_phases': closure_phases,
        'triple_amps': mean_triple_amps,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
    }


def calc_cps_multi_groups(image_cubes, mask_dir, display=True, nx=256, ny=256,
                          use_weights=True, save_allpix=False, filename='',
                          redo_calc=False):
    """Compute closure-phase bispectra for grouped (cubed) image data.

    Each element of ``image_cubes`` is a list/array of images belonging
    to one integration; bispectra are computed per-group per-integration.

    Parameters
    ----------
    image_cubes : list of list of ndarray
        Outer list is integrations, inner list is groups within each.
    mask_dir : str
        Path to the mask directory.
    display : bool
        If True, plot the sampling geometry for the first image.
    nx, ny : int
        FFT grid dimensions.
    use_weights : bool
        Unused; kept for API consistency.
    save_allpix : bool
        If True, save per-pixel bispectra to an HDF5 file.
    filename : str
        Base filename for HDF5 output.
    redo_calc : bool
        Force recomputation of triangle sampling coordinates.

    Returns
    -------
    bispectra_all : ndarray, complex
        Shape ``(n_integrations, n_groups, n_triangles)``.
    """
    cp_coords = make_cpcoords(mask_dir)
    pixel_triangles = find_tris_multi(cp_coords, mask_dir,
                                      redo_calc=redo_calc, nx=nx, ny=ny)

    if save_allpix and not filename:
        raise ValueError("filename is required when save_allpix=True")
    if save_allpix:
        fout = h5py.File(filename + '.hdf5', 'w')
    else:
        fout = None

    bispectra_all = []
    for int_idx, cube in enumerate(tqdm(image_cubes)):
        per_group = []
        for grp_idx, image in enumerate(cube):
            bispectra = calc_cps_multi_groupimage(
                image, pixel_triangles, int_idx, grp_idx,
                nx=nx, ny=ny, display=display,
                save=save_allpix, fout=fout,
            )
            display = False
            per_group.append(bispectra)
        bispectra_all.append(per_group)

    if save_allpix:
        fout.close()

    return np.array(bispectra_all, dtype='complex')

def calc_cps_multi_DFT(images, mask_dir, display=True, nx=256, ny=256,
                       use_weights=True, save_allpix=False, filename='',
                       redo_calc=False):
    """Compute closure phases via DFT using multi-pixel triangle sampling.

    Uses pre-computed DFT matrices (one per closing triangle) to transform
    images into complex triple products, then extracts closure phases.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory containing ``cpDFTmat_multi_*.fits``.
    display : bool
        Unused; kept for API consistency.
    nx, ny : int
        FFT grid dimensions.
    use_weights : bool
        If True, weight the covariance by triple amplitudes.
    save_allpix : bool
        If True, save per-pixel bispectra to an HDF5 file.
    filename : str
        Base filename for HDF5 output.
    redo_calc : bool
        Unused; kept for API consistency.

    Returns
    -------
    result : dict
        ``'raw'``            — per-image bispectra (complex).
        ``'closure_phases'`` — mean closure phases in degrees.
        ``'triple_amps'``    — mean triple amplitudes.
        ``'covariance'``     — covariance matrix, or None if single image.
        ``'variance'``       — variance, or None.
        ``'std_error'``      — standard error, or None.

    Notes
    -----
    The number of closing triangles (currently 35) is hard-coded and should
    be derived from the mask geometry in a future release.
    """
    # TODO: remove hard-coded 35; derive from mask geometry
    n_triangles_hardcoded = 35
    dft_matrices = []
    for tri_idx in range(n_triangles_hardcoded):
        dft_re, dft_im = pyfits.getdata(
            mask_dir + 'cpDFTmat_multi_' + str(nx) + '_' + str(tri_idx) + '.fits'
        )
        dft_matrices.append(dft_re + dft_im * 1.0j)

    if save_allpix and not filename:
        raise ValueError("filename is required when save_allpix=True")
    if save_allpix:
        fout = h5py.File(filename + '.hdf5', 'w')

    bispectra_all = []
    for img_idx, image in enumerate(tqdm(images)):
        per_triangle = []
        for tri_idx, dft_matrix in enumerate(dft_matrices):
            ct_phasors = np.dot(dft_matrix, image.flatten())
            n_pixel_triplets = len(ct_phasors) // 3
            triplet_bispectra = np.empty(n_pixel_triplets, dtype=complex)
            for p in range(n_pixel_triplets):
                triplet_bispectra[p] = np.prod(
                    ct_phasors[p * 3 : p * 3 + 3]
                )
            mean_bispectrum = np.mean(triplet_bispectra)
            if save_allpix:
                fout['int' + str(img_idx) + '/tri' + str(tri_idx)] = \
                    triplet_bispectra
            per_triangle.append(mean_bispectrum)
        bispectra_all.append(per_triangle)

    bispectra_all = np.array(bispectra_all)
    closure_phase_per_image = np.angle(bispectra_all, deg=True)
    triple_amp_per_image = np.abs(bispectra_all)
    mean_triple_amps = np.abs(np.mean(bispectra_all, axis=0))
    closure_phases = np.angle(np.mean(bispectra_all, axis=0), deg=True)

    if save_allpix:
        fout.close()

    if len(images) > 1:
        covariance, variance, std_error = gen_cov(
            closure_phases, closure_phase_per_image,
            weights=triple_amp_per_image, use_weights=use_weights,
        )
    else:
        covariance, variance, std_error = None, None, None

    return {
        'raw': bispectra_all,
        'closure_phases': closure_phases,
        'triple_amps': mean_triple_amps,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
    }

def gen_cov(mean_values, per_image_values, weights=None, use_weights=True):
    """Compute covariance matrix, variance, and standard error of the mean.

    Computes the sample covariance across images, optionally weighted by
    per-image amplitude weights (e.g., triple amplitudes for closure phases).

    Parameters
    ----------
    mean_values : ndarray, shape (n_observables,)
        Mean observable values (closure phases or squared visibilities).
    per_image_values : ndarray, shape (n_images, n_observables)
        Per-image observable values.
    weights : ndarray or None, shape (n_images, n_observables)
        Per-image weights. Ignored if ``use_weights`` is False.
    use_weights : bool
        If True and ``weights`` is not None, compute a weighted covariance.

    Returns
    -------
    covariance : ndarray, shape (n_observables, n_observables)
        Sample covariance matrix (weighted or unweighted).
    variance : ndarray, shape (n_observables,)
        Diagonal of the covariance matrix.
    std_error : ndarray, shape (n_observables,)
        Standard error of the mean (sqrt(variance / n_images)).
    """
    residuals = per_image_values - mean_values  # (n_images, n_obs)
    n_images = residuals.shape[0]

    if use_weights and weights is not None:
        # weighted_residuals[:, i] * weighted_residuals[:, j] summed over images
        # numerator: sum_im W[im,i]*W[im,j]*res[im,i]*res[im,j]
        weighted_resid = weights * residuals  # (n_images, n_obs)
        cov_numerator = np.einsum('ki,kj->ij', weighted_resid, weighted_resid)
        # denominator: sum_im W[im,i]*W[im,j]
        cov_denominator = np.einsum('ki,kj->ij', weights, weights)
        # Note: original code multiplied numerator by 1/(N-1) before dividing
        # by denominator. The factor belongs on the ratio for proper weighted
        # sample covariance:  (1/(N-1)) * (sum W_i W_j res_i res_j) / (sum W_i W_j)
        covariance = (1.0 / (n_images - 1)) * cov_numerator / cov_denominator
    else:
        # Unweighted: cov[i,j] = (1/(N-1)) * sum_im res[im,i]*res[im,j]
        covariance = (1.0 / (n_images - 1)) * np.einsum('ki,kj->ij', residuals, residuals)

    variance = np.diag(covariance)
    std_error = np.sqrt(variance) / np.sqrt(n_images)
    return covariance, variance, std_error

def mask_sig_pspec(mask_dir, nx, ny):
    """Create a mask selecting the signal region of the power spectrum.

    Identifies the bounding box of all baseline splodge pixels (expanded
    by 10%), and returns a binary mask that is 1 inside this region and
    0 outside.  The region *outside* the mask (where mask == 0) represents
    background used for bias estimation.

    Parameters
    ----------
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        Power spectrum dimensions.

    Returns
    -------
    signal_mask : ndarray, shape (ny, nx)
        Binary mask: 1.0 in the signal region, 0.0 outside.
    """
    baseline_pixels = find_v2_pix(mask_dir)
    # Flatten all baseline pixel coordinates into a single array
    all_pixels = np.concatenate(baseline_pixels, axis=0)

    ymin = int(ny / 2 + np.min(all_pixels[:, 1] - ny / 2) * 1.1)
    ymax = int(ny / 2 + np.max(all_pixels[:, 1] - ny / 2) * 1.1)
    xmin = int(nx / 2 + np.min(all_pixels[:, 0] - nx / 2) * 1.1)
    xmax = int(nx / 2 + np.max(all_pixels[:, 0] - nx / 2) * 1.1)

    ymin = max(ymin, 0)
    xmin = max(xmin, 0)
    ymax = min(ymax, ny)
    xmax = min(xmax, nx)

    signal_mask = np.zeros((ny, nx))
    signal_mask[ymin:ymax, xmin:xmax] = 1.0
    return signal_mask


def calc_vis_bias(power_spectrum, signal_mask, baseline_pixel_coords):
    """Estimate the visibility bias from background pixels in the power spectrum.

    Computes the mean power in the non-splodge region (where
    ``signal_mask == 0``) and multiplies by the number of pixels in the
    baseline splodge, giving the total bias to subtract from the summed
    squared visibility.

    Parameters
    ----------
    power_spectrum : ndarray, shape (ny, nx)
        Power spectrum of a single image.
    signal_mask : ndarray, shape (ny, nx)
        Binary mask (1 in signal region, 0 in background).
    baseline_pixel_coords : ndarray, shape (n_pixels, 2)
        Pixel coordinates for one baseline's splodge.

    Returns
    -------
    bias : float
        Total bias estimate (mean background × number of pixels).
    """
    background_mean = np.mean(power_spectrum[signal_mask == 0])
    bias = background_mean * len(baseline_pixel_coords)
    return bias

def calc_v2s(images, mask_dir, nx=256, ny=256, display=False,
             save_allpix=False, filename=None):
    """Compute squared visibilities for a stack of images.

    Sums the power spectrum over each baseline's multi-pixel splodge region,
    normalizes by the zero-frequency power, subtracts a background bias,
    and computes the covariance across images.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry for the first image.
    save_allpix : bool
        If True, save per-pixel squared visibilities to an HDF5 file.
    filename : str or None
        Base filename for HDF5 output (required if ``save_allpix`` is True).

    Returns
    -------
    result : dict
        ``'v2'``           — mean bias-corrected V², shape (n_baselines,).
        ``'covariance'``   — covariance matrix, or None if single image.
        ``'variance'``     — variance (diagonal of covariance), or None.
        ``'std_error'``    — standard error of the mean, or None.
        ``'v2_scatter'``   — bias-corrected V² per image, shape (n_images, n_baselines).
        ``'amplitudes'``   — zero-frequency power per image, shape (n_images,).
        ``'unnormalized'`` — unnormalized summed power per image, shape (n_images, n_baselines).
        ``'bias'``         — estimated bias per image, shape (n_images, n_baselines).
    """
    if save_allpix and not filename:
        raise ValueError("filename is required when save_allpix=True")
    baseline_pixels = find_v2_pix(mask_dir)
    n_baselines = len(baseline_pixels)
    signal_mask = mask_sig_pspec(mask_dir, nx, ny)

    if save_allpix:
        fout = h5py.File(filename + '.hdf5', 'w')

    v2_per_image = []
    amplitudes = []
    unnormalized_per_image = []
    bias_per_image = []
    summed_pspec = np.zeros((ny, nx))

    for img_idx, image in enumerate(tqdm(images)):
        pspec = make_pspec(image, nx, ny)
        summed_pspec += pspec
        zero_freq = pspec[ny // 2, nx // 2]

        # Sum power over each baseline's pixel region
        summed_power = np.empty(n_baselines)
        for bl in range(n_baselines):
            pixels = baseline_pixels[bl]
            summed_power[bl] = np.sum(pspec[pixels[:, 1], pixels[:, 0]])

        v2_per_image.append(summed_power / zero_freq)
        unnormalized_per_image.append(summed_power)
        amplitudes.append(zero_freq)

        # Bias estimate per baseline
        bias = np.array([
            calc_vis_bias(pspec, signal_mask, baseline_pixels[bl]) / zero_freq
            for bl in range(n_baselines)
        ])
        bias_per_image.append(bias)

        if save_allpix:
            for bl in range(n_baselines):
                pixels = baseline_pixels[bl]
                per_pixel_v2 = pspec[pixels[:, 1], pixels[:, 0]]
                fout['int' + str(img_idx) + '/v2s' + str(bl)] = per_pixel_v2
            fout['int' + str(img_idx) + '/bias'] = bias
            fout['int' + str(img_idx) + '/zsp'] = zero_freq

        if display and img_idx == 0:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            plt.imshow(summed_pspec ** 0.1, origin='lower')
            plt.scatter(baseline_pixels[0][:, 0], baseline_pixels[0][:, 1],
                        edgecolors='k', facecolors='none')
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()

    v2_bias_corrected = np.array(v2_per_image) - np.array(bias_per_image)
    v2_mean = np.mean(v2_bias_corrected, axis=0)

    if len(images) > 1:
        covariance, variance, std_error = gen_cov(
            v2_mean, v2_bias_corrected, use_weights=False,
        )
    else:
        covariance, variance, std_error = None, None, None

    if save_allpix:
        fout.close()

    return {
        'v2': v2_mean,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
        'v2_scatter': v2_bias_corrected,
        'amplitudes': np.array(amplitudes),
        'unnormalized': np.array(unnormalized_per_image),
        'bias': np.array(bias_per_image),
    }

def calc_v2s_groups(image_cubes, mask_dir, nx=256, ny=256, display=False,
                    save_allpix=False, filename=None):
    """Compute squared visibilities for grouped (cubed) image data.

    Each element of ``image_cubes`` is a list/array of images belonging
    to one integration; V² values are computed per-group per-integration.

    Parameters
    ----------
    image_cubes : list of list of ndarray
        Outer list is integrations, inner list is groups within each.
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry for the first image.
    save_allpix : bool
        If True, save per-pixel V² to an HDF5 file.
    filename : str or None
        Base filename for HDF5 output (required if ``save_allpix`` is True).

    Returns
    -------
    result : dict
        ``'v2'``           — mean bias-corrected V².
        ``'covariance'``   — covariance matrix, or None.
        ``'variance'``     — variance, or None.
        ``'std_error'``    — standard error, or None.
        ``'v2_scatter'``   — bias-corrected V² per integration/group.
        ``'amplitudes'``   — zero-frequency power per integration/group.
        ``'unnormalized'`` — unnormalized summed power per integration/group.
        ``'bias'``         — bias estimates per integration/group.
    """
    if save_allpix and not filename:
        raise ValueError("filename is required when save_allpix=True")
    baseline_pixels = find_v2_pix(mask_dir)
    n_baselines = len(baseline_pixels)
    signal_mask = mask_sig_pspec(mask_dir, nx, ny)

    if save_allpix:
        fout = h5py.File(filename + '.hdf5', 'w')

    v2_all = []
    amplitudes_all = []
    unnormalized_all = []
    bias_all = []
    summed_pspec = np.zeros((ny, nx))

    for int_idx, cube in enumerate(tqdm(image_cubes)):
        v2_groups = []
        amp_groups = []
        unnorm_groups = []
        bias_groups = []

        for grp_idx, image in enumerate(cube):
            pspec = make_pspec(image, nx, ny)
            summed_pspec += pspec
            zero_freq = pspec[ny // 2, nx // 2]

            summed_power = np.empty(n_baselines)
            for bl in range(n_baselines):
                pixels = baseline_pixels[bl]
                summed_power[bl] = np.sum(pspec[pixels[:, 1], pixels[:, 0]])

            v2_groups.append(summed_power / zero_freq)
            unnorm_groups.append(summed_power)
            amp_groups.append(zero_freq)

            bias = np.array([
                calc_vis_bias(pspec, signal_mask, baseline_pixels[bl]) / zero_freq
                for bl in range(n_baselines)
            ])
            bias_groups.append(bias)

            if save_allpix:
                prefix = 'int' + str(int_idx) + '/group' + str(grp_idx)
                for bl in range(n_baselines):
                    pixels = baseline_pixels[bl]
                    fout[prefix + '/v2s' + str(bl)] = \
                        pspec[pixels[:, 1], pixels[:, 0]]
                fout[prefix + '/bias'] = bias
                fout[prefix + '/zsp'] = zero_freq

        v2_all.append(v2_groups)
        amplitudes_all.append(amp_groups)
        unnormalized_all.append(unnorm_groups)
        bias_all.append(bias_groups)

        if display and int_idx == 0:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            plt.imshow(summed_pspec ** 0.1, origin='lower')
            plt.scatter(baseline_pixels[0][:, 0], baseline_pixels[0][:, 1],
                        edgecolors='k', facecolors='none')
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()

    v2_bias_corrected = np.array(v2_all) - np.array(bias_all)
    v2_mean = np.mean(v2_bias_corrected, axis=0)

    if len(image_cubes) > 1:
        covariance, variance, std_error = gen_cov(
            v2_mean, v2_bias_corrected, use_weights=False,
        )
    else:
        covariance, variance, std_error = None, None, None

    if save_allpix:
        fout.close()

    return {
        'v2': v2_mean,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
        'v2_scatter': v2_bias_corrected,
        'amplitudes': np.array(amplitudes_all),
        'unnormalized': np.array(unnormalized_all),
        'bias': np.array(bias_all),
    }

def calc_v2s_single(images, mask_dir, nx=256, ny=256, display=False):
    """Compute squared visibilities using a single pixel per baseline.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry after processing all images.

    Returns
    -------
    result : dict
        ``'v2'``           — mean V² (no bias correction applied).
        ``'covariance'``   — covariance matrix, or None if single image.
        ``'variance'``     — variance, or None.
        ``'std_error'``    — standard error, or None.
        ``'v2_scatter'``   — per-image V² (no bias correction).
        ``'amplitudes'``   — zero-frequency power per image.
        ``'unnormalized'`` — unnormalized summed power per image.
        ``'bias'``         — bias estimates per image.
    """
    baseline_pixels = find_v2_pix(mask_dir, v2type='single')
    n_baselines = len(baseline_pixels)
    signal_mask = mask_sig_pspec(mask_dir, nx, ny)

    v2_per_image = []
    amplitudes = []
    unnormalized_per_image = []
    bias_per_image = []
    summed_pspec = np.zeros((ny, nx))

    for image in tqdm(images):
        pspec = make_pspec(image, nx, ny)
        summed_pspec += pspec
        zero_freq = pspec[ny // 2, nx // 2]

        summed_power = np.empty(n_baselines)
        for bl in range(n_baselines):
            pixels = baseline_pixels[bl]
            summed_power[bl] = np.sum(pspec[pixels[:, 1], pixels[:, 0]])

        v2_per_image.append(summed_power / zero_freq)
        unnormalized_per_image.append(summed_power)
        amplitudes.append(zero_freq)

        bias = np.array([
            calc_vis_bias(pspec, signal_mask, baseline_pixels[bl]) / zero_freq
            for bl in range(n_baselines)
        ])
        bias_per_image.append(bias)

    if display:
        fig = plt.figure(figsize=(18, 9))
        plt.subplots_adjust(right=0.99, left=0.02, bottom=0.04, top=0.95)
        plt.title(mask_dir)
        plt.imshow(summed_pspec ** 0.1, origin='lower')
        for bl in range(n_baselines):
            plt.scatter(baseline_pixels[bl][:, 0], baseline_pixels[bl][:, 1],
                        edgecolors='k', facecolors='none')
        plt.show()

    # Note: original code does NOT subtract bias for single-pixel mode
    v2_all = np.array(v2_per_image)
    v2_mean = np.mean(v2_all, axis=0)

    if len(images) > 1:
        covariance, variance, std_error = gen_cov(
            v2_mean, v2_all, use_weights=False,
        )
    else:
        covariance, variance, std_error = None, None, None

    return {
        'v2': v2_mean,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
        'v2_scatter': v2_all,
        'amplitudes': np.array(amplitudes),
        'unnormalized': np.array(unnormalized_per_image),
        'bias': np.array(bias_per_image),
    }

def calc_cvis(images, mask_dir, nx=256, ny=256, display=False,
              save_allpix=False, filebase='', subpixel=False,
              write_FTs=False):
    """Compute complex visibility amplitudes and phases for an image stack.

    Parameters
    ----------
    images : array-like of ndarray
        Stack of 2-D images.
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the last image's FT with splodge overlays.
    save_allpix : bool
        If True, save per-pixel complex visibilities to an HDF5 file.
    filebase : str
        Base path for HDF5 output.
    subpixel : bool
        If True, apply sub-pixel centering via Fourier shift before FFT.
    write_FTs : bool
        If True (and ``save_allpix``), also write the full FT to HDF5.

    Returns
    -------
    result : dict
        ``'amplitudes'``     — mean visibility amplitudes, shape (n_baselines,).
        ``'phases'``         — mean visibility phases in degrees.
        ``'covariance'``     — covariance matrix (currently None).
        ``'variance'``       — variance (currently None).
        ``'std_error'``      — standard error (currently None).
        ``'phases_per_image'`` — per-image visibility phases in degrees.
    """
    baseline_pixels = find_cvis_pix(mask_dir)
    n_baselines = len(baseline_pixels)
    signal_mask = mask_sig_pspec(mask_dir, nx, ny)

    if save_allpix and not filebase:
        raise ValueError("filebase is required when save_allpix=True")
    if save_allpix:
        fout = h5py.File(filebase + '_cvis.hdf5', 'w')

    complex_vis_all = []
    bias_all = []
    summed_pspec = np.zeros((ny, nx))

    for img_idx, image in enumerate(tqdm(images)):
        if not subpixel:
            ft = fft_image(image, nx, ny)
        else:
            yint, xint = get_center(image, 4.3, 6.5, 0.065)
            x_cen, y_cen = find_psf_center(image, verbose=False)
            dy, dx = y_cen - yint, x_cen - xint
            _ft_orig, ft_shifted = fourier_center(image, dy, dx)
            centered_image = np.real(
                np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ft_shifted)))
            )
            ft = fft_image(centered_image, nx, ny)

        zero_freq = ft[ny // 2, nx // 2]
        summed_pspec += np.abs(ft) ** 2

        # Mean complex visibility per baseline
        mean_vis = np.empty(n_baselines, dtype=complex)
        for bl in range(n_baselines):
            pixels = baseline_pixels[bl]
            mean_vis[bl] = np.mean(ft[pixels[:, 1], pixels[:, 0]])

        normalized_vis = mean_vis / zero_freq
        complex_vis_all.append(normalized_vis)

        bias = np.array([
            calc_vis_bias(np.abs(ft), signal_mask,
                          baseline_pixels[bl]) / np.abs(zero_freq)
            for bl in range(n_baselines)
        ])
        bias_all.append(bias)

        if save_allpix:
            prefix = 'int' + str(img_idx)
            for bl in range(n_baselines):
                pixels = baseline_pixels[bl]
                fout[prefix + '/cvis' + str(bl)] = ft[pixels[:, 1], pixels[:, 0]]
            fout[prefix + '/bias'] = bias
            fout[prefix + '/zsp'] = zero_freq
            if write_FTs:
                fout[prefix + '/FT'] = ft

    if display:
        fig = plt.figure(figsize=(18, 9))
        plt.subplots_adjust(right=0.99, left=0.02, bottom=0.04, top=0.95)
        plt.title(mask_dir)
        cpl = plt.imshow(np.abs(ft), origin='lower')
        plt.colorbar(cpl)
        for bl in range(n_baselines):
            plt.scatter(baseline_pixels[bl][:, 0], baseline_pixels[bl][:, 1],
                        edgecolors='k', facecolors='none')
        plt.show()

    complex_vis_all = np.array(complex_vis_all)
    mean_amplitudes = np.mean(np.abs(complex_vis_all), axis=0)
    mean_phases = np.angle(np.mean(complex_vis_all, axis=0), deg=True)
    phases_per_image = np.angle(complex_vis_all, deg=True)

    # Covariance not yet implemented for complex visibilities
    covariance, variance, std_error = None, None, None

    if save_allpix:
        fout.close()

    return {
        'amplitudes': mean_amplitudes,
        'phases': mean_phases,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
        'phases_per_image': phases_per_image,
    }

def calc_cvis_groups(image_cubes, mask_dir, nx=256, ny=256, display=False,
                     save_allpix=False, filename='', subpixel=False):
    """Compute complex visibility amplitudes and phases for grouped image data.

    Parameters
    ----------
    image_cubes : list of list of ndarray
        Outer list is integrations, inner list is groups within each.
    mask_dir : str
        Path to the mask directory.
    nx, ny : int
        FFT grid dimensions.
    display : bool
        If True, plot the sampling geometry after processing.
    save_allpix : bool
        If True, save per-pixel complex visibilities to an HDF5 file.
    filename : str
        Base filename for HDF5 output.
    subpixel : bool
        Unused; kept for API consistency.

    Returns
    -------
    result : dict
        ``'amplitudes'``       — mean visibility amplitudes.
        ``'phases'``           — mean visibility phases in degrees.
        ``'covariance'``       — covariance matrix (currently None).
        ``'variance'``         — variance (currently None).
        ``'std_error'``        — standard error (currently None).
        ``'phases_per_image'`` — per-integration/group phases in degrees.
        ``'complex_vis'``      — raw complex visibilities per integration/group.
        ``'bias'``             — bias estimates per integration/group.
    """
    baseline_pixels = find_cvis_pix(mask_dir)
    n_baselines = len(baseline_pixels)
    signal_mask = mask_sig_pspec(mask_dir, nx, ny)

    if save_allpix and not filename:
        raise ValueError("filename is required when save_allpix=True")
    if save_allpix:
        fout = h5py.File(filename + '.hdf5', 'w')

    complex_vis_all = []
    bias_all = []
    summed_pspec = np.zeros((ny, nx))

    for int_idx, cube in enumerate(tqdm(image_cubes)):
        vis_groups = []
        bias_groups = []

        for grp_idx, image in enumerate(cube):
            ft = fft_image(image, nx, ny)
            zero_freq = ft[ny // 2, nx // 2]
            summed_pspec += np.abs(ft) ** 2

            mean_vis = np.empty(n_baselines, dtype=complex)
            for bl in range(n_baselines):
                pixels = baseline_pixels[bl]
                mean_vis[bl] = np.mean(ft[pixels[:, 1], pixels[:, 0]])

            normalized_vis = mean_vis / zero_freq
            vis_groups.append(normalized_vis)

            bias = np.array([
                calc_vis_bias(np.abs(ft), signal_mask,
                              baseline_pixels[bl]) / np.abs(zero_freq)
                for bl in range(n_baselines)
            ])
            bias_groups.append(bias)

            if save_allpix:
                prefix = ('int' + str(int_idx)
                          + '/group' + str(grp_idx))
                for bl in range(n_baselines):
                    pixels = baseline_pixels[bl]
                    fout[prefix + '/cvis' + str(bl)] = \
                        ft[pixels[:, 1], pixels[:, 0]]
                fout[prefix + '/bias'] = bias
                fout[prefix + '/zsp'] = zero_freq

        complex_vis_all.append(vis_groups)
        bias_all.append(bias_groups)

    if display:
        fig = plt.figure(figsize=(18, 9))
        plt.subplots_adjust(right=0.99, left=0.02, bottom=0.04, top=0.95)
        plt.title(mask_dir)
        plt.imshow(summed_pspec ** 0.1, origin='lower')
        for bl in range(n_baselines):
            plt.scatter(baseline_pixels[bl][:, 0], baseline_pixels[bl][:, 1],
                        edgecolors='k', facecolors='none')
        plt.show()

    complex_vis_arr = np.array(complex_vis_all)
    mean_amplitudes = np.mean(np.abs(complex_vis_arr), axis=0)
    mean_phases = np.angle(np.mean(complex_vis_arr, axis=0), deg=True)
    phases_per_image = np.angle(complex_vis_arr, deg=True)

    # Covariance not yet implemented for complex visibilities
    covariance, variance, std_error = None, None, None

    if save_allpix:
        fout.close()

    return {
        'amplitudes': mean_amplitudes,
        'phases': mean_phases,
        'covariance': covariance,
        'variance': variance,
        'std_error': std_error,
        'phases_per_image': phases_per_image,
        'complex_vis': complex_vis_all,
        'bias': bias_all,
    }
