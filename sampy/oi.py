"""OIFITS file generation from SAMpy observables."""

import datetime
import math

import numpy as np
import astropy.io.fits as pyfits

from oifits import (
    oifits, OI_WAVELENGTH, OI_TARGET, OI_VIS2, OI_T3,
)


def rotate_point(point, angle_deg):
    """Rotate a 2D point by a given angle.

    Parameters
    ----------
    point : array_like
        (x, y) coordinates.
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    tuple
        Rotated (x, y) coordinates.
    """
    sin_a, cos_a = math.sin(math.radians(angle_deg)), math.cos(math.radians(angle_deg))
    dx = cos_a * point[0] - sin_a * point[1]
    dy = sin_a * point[0] + cos_a * point[1]
    return (dx, dy)


def rotate_cp_uvs(position_angles, coord_dir):
    """Rotate closure phase UV coordinates by a set of position angles.

    Parameters
    ----------
    position_angles : array_like
        Position angles in degrees.
    coord_dir : str
        Directory containing ``cp_uvs.fits``.

    Returns
    -------
    numpy.ndarray
        Rotated UV coordinates, shape ``(n_angles, n_triangles, 3, 2)``.
    """
    cp_uvs = pyfits.getdata(coord_dir + 'cp_uvs.fits')
    rotated = []
    for angle in position_angles:
        rotated.append([
            [rotate_point(cp_uvs[tri][vert], -angle)
             for vert in range(len(cp_uvs[0]))]
            for tri in range(len(cp_uvs))
        ])
    return np.array(rotated)


def rotate_baseline_uvs(position_angles, coord_dir):
    """Rotate baseline UV coordinates by a set of position angles.

    Parameters
    ----------
    position_angles : array_like
        Position angles in degrees.
    coord_dir : str
        Directory containing ``bl_uvs.fits``.

    Returns
    -------
    numpy.ndarray
        Rotated UV coordinates, shape ``(n_angles, n_baselines, 2)``.
    """
    bl_uvs = pyfits.getdata(coord_dir + 'bl_uvs.fits')
    rotated = []
    for angle in position_angles:
        rotated.append([
            rotate_point(bl_uvs[bl_idx], -angle)
            for bl_idx in range(len(bl_uvs))
        ])
    return np.array(rotated)


def build_oifits(target_name, wavelength, coord_dir, output_filename,
                 closure_phases=None, cp_errors=None,
                 squared_visibilities=None, v2_errors=None,
                 position_angles=None):
    """Build and save an OIFITS file from SAMpy observables.

    Parameters
    ----------
    target_name : str
        Name of the science target.
    wavelength : float
        Central wavelength in microns.
    coord_dir : str
        Directory containing coordinate FITS files.
    output_filename : str
        Path for the output OIFITS file.
    closure_phases : numpy.ndarray or None
        Shape ``(n_pointings, n_triangles)``.
    cp_errors : numpy.ndarray or None
        Same shape as ``closure_phases``.
    squared_visibilities : numpy.ndarray or None
        Shape ``(n_pointings, n_baselines)``.
    v2_errors : numpy.ndarray or None
        Same shape as ``squared_visibilities``.
    position_angles : numpy.ndarray or None
        Position angle for each pointing (degrees).
    """
    if closure_phases is None:
        closure_phases = []
    if cp_errors is None:
        cp_errors = []
    if squared_visibilities is None:
        squared_visibilities = []
    if v2_errors is None:
        v2_errors = []
    if position_angles is None:
        position_angles = []

    oi_data = oifits()
    oi_data.wavelength['ARRAY'] = OI_WAVELENGTH(wavelength * 1.0e-06)
    oi_data.target = np.append(oi_data.target,
                               OI_TARGET(target_name, 0, 0))

    v2_uvs = rotate_baseline_uvs(position_angles, coord_dir)
    cp_uvs = rotate_cp_uvs(position_angles, coord_dir)

    for pointing_idx in range(len(squared_visibilities)):
        for bl_idx in range(len(squared_visibilities[pointing_idx])):
            oi_data.vis2 = np.append(oi_data.vis2, OI_VIS2(
                timeobs=datetime.datetime(2000, 1, 1, 0, 0, 0),
                int_time=0.0,
                vis2data=squared_visibilities[pointing_idx, bl_idx],
                vis2err=v2_errors[pointing_idx, bl_idx],
                flag=np.array([False], dtype=bool),
                ucoord=v2_uvs[pointing_idx, bl_idx, 0],
                vcoord=v2_uvs[pointing_idx, bl_idx, 1],
                wavelength=oi_data.wavelength['ARRAY'],
                target=oi_data.target[0],
            ))
        for tri_idx in range(len(closure_phases[pointing_idx])):
            oi_data.t3 = np.append(oi_data.t3, OI_T3(
                timeobs=datetime.datetime(2000, 1, 1, 0, 0, 0),
                int_time=0.0,
                t3amp=0,
                t3amperr=0,
                t3phi=closure_phases[pointing_idx, tri_idx],
                t3phierr=cp_errors[pointing_idx, tri_idx],
                flag=np.array([False], dtype=bool),
                u1coord=cp_uvs[pointing_idx, tri_idx, 0, 0],
                v1coord=cp_uvs[pointing_idx, tri_idx, 0, 1],
                u2coord=cp_uvs[pointing_idx, tri_idx, 1, 0],
                v2coord=cp_uvs[pointing_idx, tri_idx, 1, 1],
                wavelength=oi_data.wavelength['ARRAY'],
                target=oi_data.target[0],
            ))
    oi_data.save(output_filename)
