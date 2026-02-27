"""Mask geometry, coordinate generation, and Fourier-plane sampling setup."""

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy import ndimage
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from sampy import get_data_path


def filter_significant_eigenvectors(matrix):
    """Extract eigenvectors with eigenvalues above a threshold.

    Computes the eigendecomposition of a Hermitian matrix and returns
    only the eigenvectors whose eigenvalues exceed 1e-6.

    Parameters
    ----------
    matrix : numpy.ndarray
        2D Hermitian matrix.

    Returns
    -------
    numpy.ndarray
        Real part of the matrix whose rows are the significant eigenvectors.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    row_eigenvectors = eigenvectors.T
    significant = row_eigenvectors[eigenvalues >= 1e-6]
    return significant.real


def make_coords(output_dir, jwst_filt='f380m', inst='niriss', pixel_scale=0.0656,
                n_pixels=256, rotation=0, x_offset=0, y_offset=0,
                zero_spacing_radius=125, spectral_sampling=20, recompute=False,
                fourier_cutoff=0.5):
    """Generate Fourier-plane sampling coordinates for an aperture mask.

    Performs the following steps:

    1. Loads the Non-Redundant Mask hole coordinates.
    2. Applies rotation and offset to the mask geometry.
    3. Computes baseline vectors and closure phase triangles.
    4. Builds the baseline-to-closure-phase mapping matrix.
    5. Generates synthetic power spectra for each baseline using the
       filter transmission profile.
    6. Identifies Fourier-plane pixels to sample for each baseline
       and closure phase triangle.

    All outputs are written as FITS files to ``output_dir``.

    Parameters
    ----------
    output_dir : str
        Directory for output coordinate files (created if needed).
    jwst_filt : str
        JWST/NIRISS filter name: ``'f227w'``, ``'f380m'``, ``'f430m'``,
        or ``'f480m'``.
    inst : str
        Instrument name (currently only ``'niriss'`` is supported).
    pixel_scale : float
        Pixel scale in arcseconds per pixel.
    n_pixels : int
        Image size in pixels.
    rotation : float
        Mask rotation angle in degrees.
    x_offset, y_offset : float
        Mask center offsets.
    zero_spacing_radius : float
        Radius (in pixels) for masking the zero-spacing (DC) peak in the
        power spectrum.
    spectral_sampling : int
        Sample every Nth wavelength from the filter transmission curve.
    recompute : bool
        If True, recompute even if output files already exist.
    fourier_cutoff : float
        Threshold for selecting Fourier-plane sampling pixels
        (fraction of peak power).
    """
    # Map filter name to transmission file
    filter_map = {
        'f227w': 'NIRISS_F277W.txt',
        'f380m': 'NIRISS_F380M.txt',
        'f430m': 'NIRISS_F430M.txt',
        'f480m': 'NIRISS_F480M.txt',
    }
    if jwst_filt not in filter_map:
        raise ValueError(
            f"jwst_filt must be one of {list(filter_map.keys())}, "
            f"got '{jwst_filt}'"
        )
    transmission_file = str(get_data_path(filter_map[jwst_filt]))

    if inst == 'niriss':
        n_holes = 7
        subaperture_diameter = 0.75  # flat-to-flat distance in meters
        mask_file = str(get_data_path('NIRISS_7holeMask.txt'))
    else:
        raise ValueError(f"inst must be 'niriss', got '{inst}'")

    # Create output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load and transform mask coordinates
    hole_positions_raw = np.loadtxt(mask_file)
    hole_positions_raw[np.where(hole_positions_raw[:, 0] > 0)] += np.array(
        [x_offset, y_offset]
    )
    cos_rot, sin_rot = np.cos(np.radians(-rotation)), np.sin(np.radians(-rotation))
    hole_positions = np.array([
        [p[0] * cos_rot - p[1] * sin_rot, p[0] * sin_rot + p[1] * cos_rot]
        for p in hole_positions_raw
    ])

    # Compute closure phase triangle vectors
    cp_vectors = np.array([
        [hole_positions[idx_a] - hole_positions[idx_b],
         hole_positions[idx_b] - hole_positions[idx_c],
         hole_positions[idx_c] - hole_positions[idx_a]]
        for idx_a in range(len(hole_positions))
        for idx_b in range(len(hole_positions))
        for idx_c in range(len(hole_positions))
        if idx_c > idx_b > idx_a
    ])

    # Compute baseline vectors
    baseline_vectors = np.array([
        hole_positions[idx_a] - hole_positions[idx_b]
        for idx_a in range(len(hole_positions))
        for idx_b in range(len(hole_positions))
        if idx_b > idx_a
    ])

    # Build baseline-to-closure-phase mapping matrix
    cp_to_bl_matrix = []
    for triangle in cp_vectors:
        row = np.zeros(len(baseline_vectors))
        for vec in triangle:
            for bl_idx in range(len(baseline_vectors)):
                if (vec[0] == baseline_vectors[bl_idx, 0]
                        and vec[1] == baseline_vectors[bl_idx, 1]):
                    row[bl_idx] = 1.0
                if (vec[0] == -baseline_vectors[bl_idx, 0]
                        and vec[1] == -baseline_vectors[bl_idx, 1]):
                    row[bl_idx] = -1.0
        cp_to_bl_matrix.append(row)
        np.savetxt(output_dir + 'k_mat.txt', np.array(cp_to_bl_matrix))

    # Hole pairs for each baseline
    hole_pairs = np.array([
        [hole_positions[idx_a], hole_positions[idx_b]]
        for idx_a in range(len(hole_positions))
        for idx_b in range(len(hole_positions))
        if idx_b > idx_a
    ])

    # Baseline UV coordinates (u is +ve going left in image)
    baseline_uvs = np.array([row[:] for row in baseline_vectors])
    baseline_uvs[:, 0] *= -1
    pyfits.writeto(output_dir + 'bl_uvs.fits', baseline_uvs, overwrite=True)
    baseline_power = np.zeros([len(baseline_uvs), n_pixels, n_pixels])

    # Load filter transmission profile
    transmission_data = np.loadtxt(transmission_file, skiprows=1)
    wavelengths = transmission_data[:, 0]
    transmissions = transmission_data[:, 1]
    # Select high-transmission wavelengths and subsample
    high_trans = np.where(transmissions > 0.5)
    wavelengths = wavelengths[high_trans][::spectral_sampling]
    transmissions = transmissions[high_trans][::spectral_sampling]

    central_wavelength = np.sum(wavelengths * transmissions) / np.sum(transmissions)
    wavelength_range = [np.min(wavelengths), np.max(wavelengths)]

    plate_scale = (1.0 / (float(n_pixels) * pixel_scale) * 206265.0
                   * central_wavelength * 1e-06)

    baseline_pixels = np.round(np.array([n_pixels // 2, n_pixels // 2])) \
                      + baseline_vectors / plate_scale
    pyfits.writeto(output_dir + 'bl_pix.fits', baseline_pixels, overwrite=True)
    max_smoothing = np.max([
        subaperture_diameter / (1.0 / (float(n_pixels) * pixel_scale)
                                * 206265.0 * wl * 1e-06)
        for wl in wavelength_range
    ])

    # Generate Fourier-plane sampling for each baseline
    bl_pixel_arrays = []
    total_power_spectrum = np.zeros([n_pixels, n_pixels])
    total_pupil = None  # initialized on first wavelength computation
    pixel_grid = np.array([
        [[y, x] for x in range(n_pixels)]
        for y in range(n_pixels)
    ])

    for bl_idx in range(len(baseline_uvs)):
        if not os.path.isfile(output_dir + f'v2_ind{bl_idx}.fits') or recompute:
            fourier_map = np.zeros([n_pixels, n_pixels])

            print(f'Doing baseline {bl_idx + 1} of {len(baseline_uvs)} ->')

            for wl_idx in tqdm(range(len(wavelengths))):
                wl = wavelengths[wl_idx]
                trans = transmissions[wl_idx]

                # Compute mask in Fourier plane at desired plate scale
                desired_plate_scale = 0.025
                n_pix_ft = int(np.round(
                    1.0 / (desired_plate_scale / (206265.0 * wl * 1e-06) * pixel_scale)
                ))
                if n_pix_ft % 2 == 0:
                    n_pix_ft += 1
                if total_pupil is None:
                    total_pupil = np.zeros([n_pix_ft, n_pix_ft])

                ft_pixel_grid = np.array([
                    [[y, x] for x in range(n_pix_ft)]
                    for y in range(n_pix_ft)
                ])

                pupil = np.zeros([n_pix_ft, n_pix_ft])
                hole_pair_pixels = np.array(
                    np.round(np.array([n_pix_ft // 2, n_pix_ft // 2])
                             + hole_pairs[bl_idx] / desired_plate_scale),
                    dtype=int
                )

                # Hexagonal holes (NIRISS)
                hex_side = subaperture_diameter / desired_plate_scale / math.sqrt(3)
                rotation_rad = np.radians(-rotation)
                for hole_center in hole_pair_pixels:
                    vertices = []
                    for vertex_idx in range(6):
                        angle = np.radians(vertex_idx * 60) + rotation_rad
                        vx = hole_center[1] + hex_side * np.sin(angle)
                        vy = hole_center[0] + hex_side * np.cos(angle)
                        vertices.append(np.array([vx, vy]))
                    hexagon = Polygon(np.array(vertices))

                    hole_mask = []
                    for pixel_row in ft_pixel_grid:
                        row_mask = []
                        for pixel in pixel_row:
                            if hexagon.contains(Point(pixel)):
                                row_mask.append(1)
                            else:
                                row_mask.append(0)
                        hole_mask.append(np.array(row_mask))
                    pupil = pupil + hole_mask

                if wl_idx == 0:
                    plt.imshow(pupil, origin='lower')
                    plt.show()

                # Compute power spectrum
                ft_image = abs(np.fft.fftshift(
                    np.fft.fft2(np.fft.fftshift(pupil))
                )) ** 2
                if n_pixels % 2 != 0:
                    half = n_pixels // 2
                    fourier_map += ft_image[
                        n_pix_ft // 2 - half:n_pix_ft // 2 + half + 1,
                        n_pix_ft // 2 - half:n_pix_ft // 2 + half + 1
                    ] * trans
                else:
                    half = n_pixels // 2
                    fourier_map += ft_image[
                        n_pix_ft // 2 - half:n_pix_ft // 2 + half,
                        n_pix_ft // 2 - half:n_pix_ft // 2 + half
                    ] * trans
                if wl_idx == 0:
                    total_pupil += np.transpose(pupil)

            power_spectrum = np.fft.fftshift(
                abs(np.fft.fft2(np.fft.fftshift(fourier_map)))
            )
            power_spectrum = power_spectrum / np.amax(power_spectrum)

            # Mask central peak (hexagonal for NIRISS)
            mask_layer = np.zeros(power_spectrum.shape)
            mask_center = (n_pixels // 2, n_pixels // 2)
            mask_side = (int(125.0 / 4.0) / 4.8 * 3.8
                         * n_pixels / 257 / math.sqrt(3))
            vertices = []
            rotation_rad = np.radians(-rotation)
            for mask_vtx_idx in range(6):
                angle = np.radians(mask_vtx_idx * 60) + rotation_rad
                vx = mask_center[0] + mask_side * np.sin(angle)
                vy = mask_center[1] + mask_side * np.cos(angle)
                vertices.append(np.array([vx, vy]))
            mask_hexagon = Polygon(np.array(vertices))
            hex_mask = []
            for pixel_row in pixel_grid:
                row_mask = []
                for pixel in pixel_row:
                    if mask_hexagon.contains(Point(pixel)):
                        row_mask.append(1)
                    else:
                        row_mask.append(0)
                hex_mask.append(np.array(row_mask))
            mask_layer = mask_layer + hex_mask
            mask_layer = 1 - mask_layer
            mask_layer = np.transpose(mask_layer)
            power_spectrum = np.multiply(mask_layer, power_spectrum)
            if wl_idx == 0:
                plt.imshow(power_spectrum ** 0.1)
                plt.show()

            baseline_power[bl_idx] = power_spectrum
            total_power_spectrum += power_spectrum

            # Extract sampling pixel coordinates
            above_cutoff = np.where(baseline_power[bl_idx] > fourier_cutoff)
            sampling_coords = np.array([
                [above_cutoff[1][idx], above_cutoff[0][idx]]
                for idx in range(len(above_cutoff[0]))
            ])
            if len(sampling_coords) == 0:
                raise RuntimeError(
                    f'No Fourier-plane pixels above cutoff={fourier_cutoff} '
                    f'for baseline {bl_idx}. Try lowering fourier_cutoff.'
                )
            pyfits.writeto(output_dir + f'v2_ind{bl_idx}.fits',
                           np.array(sampling_coords), overwrite=True)
        else:
            sampling_coords = pyfits.getdata(output_dir + f'v2_ind{bl_idx}.fits')
        bl_pixel_arrays.append(sampling_coords)

    if not recompute and os.path.isfile(output_dir + 'syn_pspec.fits'):
        total_power_spectrum = pyfits.getdata(output_dir + 'syn_pspec.fits')
    else:
        pyfits.writeto(output_dir + 'syn_pspec.fits',
                       np.array(total_power_spectrum), overwrite=True)

    plt.imshow(total_power_spectrum ** 0.1, origin='lower')
    for bl_pix in bl_pixel_arrays:
        plt.scatter(np.array(bl_pix)[:, 0], np.array(bl_pix)[:, 1])
    plt.show()

    if recompute and total_pupil is not None:
        plt.imshow(total_pupil, origin='lower')
        plt.show()

    baseline_uvs = np.array(baseline_uvs)
    baseline_pixels = np.array(baseline_pixels)

    # Generate CP sampling coordinates from baseline coordinates
    cp_pixel_arrays = []
    cp_uv_coords = []
    cp_pixel_coords = []
    for tri_idx in range(len(cp_to_bl_matrix)):
        matrix_row = np.array(cp_to_bl_matrix)[tri_idx]
        nonzero_indices = np.where(matrix_row != 0)[0]
        vertex_count = 0
        uv_temp = []
        uv_pix_temp = []
        for bl_index in nonzero_indices:
            diff_pos = np.sum(
                (bl_pixel_arrays[bl_index] - baseline_pixels[bl_index]) ** 2,
                axis=-1
            )
            diff_neg = np.sum(
                (bl_pixel_arrays[bl_index]
                 - (np.array([n_pixels, n_pixels]) - baseline_pixels[bl_index])) ** 2,
                axis=-1
            )
            if matrix_row[bl_index] == 1:
                selected = bl_pixel_arrays[bl_index][np.where(diff_pos < diff_neg)]
                uv_temp.append(np.squeeze(np.asarray(baseline_uvs[bl_index])))
                uv_pix_temp.append(baseline_pixels[bl_index])
                plt.scatter(bl_pixel_arrays[bl_index][:, 0],
                            bl_pixel_arrays[bl_index][:, 1], c='grey')
                plt.scatter(selected[:, 0], selected[:, 1], c='b')
                plt.scatter(baseline_pixels[bl_index][0],
                            baseline_pixels[bl_index][1], c='k')
                plt.axhline(n_pixels / 2)
                plt.axvline(n_pixels / 2)
            if matrix_row[bl_index] == -1:
                selected = bl_pixel_arrays[bl_index][np.where(diff_pos > diff_neg)]
                uv_temp.append(-(np.squeeze(np.asarray(baseline_uvs[bl_index]))))
                uv_pix_temp.append(
                    np.array([n_pixels, n_pixels]) - baseline_pixels[bl_index]
                )
                plt.scatter(bl_pixel_arrays[bl_index][:, 0],
                            bl_pixel_arrays[bl_index][:, 1], c='grey')
                plt.scatter(selected[:, 0], selected[:, 1], c='r')
                plt.scatter(baseline_pixels[bl_index][0],
                            baseline_pixels[bl_index][1], c='k')
                plt.axhline(n_pixels / 2)
                plt.axvline(n_pixels / 2)
            selected = np.array(selected)
            pyfits.writeto(
                output_dir + f'ind{tri_idx}_vert{vertex_count}.fits',
                selected, overwrite=True
            )
            vertex_count += 1
        plt.show()
        cp_uv_coords.append(uv_temp)
        cp_pixel_coords.append(uv_pix_temp)

    pyfits.writeto(output_dir + 'cp_uvs.fits',
                   np.array(cp_uv_coords), overwrite=True)
    pyfits.writeto(output_dir + 'cp_pix.fits',
                   np.array(cp_pixel_coords), overwrite=True)

    # Generate complex visibility sampling coordinates
    cvis_uv_coords = []
    cvis_pixel_coords = []
    for bl_index in range(len(baseline_uvs)):
        diff_pos = np.sum(
            (bl_pixel_arrays[bl_index] - baseline_pixels[bl_index]) ** 2,
            axis=-1
        )
        diff_neg = np.sum(
            (bl_pixel_arrays[bl_index]
             - (np.array([n_pixels, n_pixels]) - baseline_pixels[bl_index])) ** 2,
            axis=-1
        )
        selected = bl_pixel_arrays[bl_index][np.where(diff_pos < diff_neg)]
        cvis_uv_coords.append(np.squeeze(np.asarray(baseline_uvs[bl_index])))
        cvis_pixel_coords.append(baseline_pixels[bl_index])
        pyfits.writeto(output_dir + f'cvis_ind{bl_index}.fits',
                       np.array(selected), overwrite=True)

    pyfits.writeto(output_dir + 'cvis_uvs.fits',
                   np.array(cvis_uv_coords), overwrite=True)
    pyfits.writeto(output_dir + 'cvis_pix.fits',
                   np.array(cvis_pixel_coords), overwrite=True)
