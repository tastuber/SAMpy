"""Grid-search model fitting for binary companion detection."""

import os

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from scipy.interpolate import griddata
from ipywidgets import IntProgress
from IPython.display import display as idisplay


def binary_model(wavelength, cp_uvs, v2_uvs, model_params, angle,
                 compute_kernel_phase=False, include_v2s=True):
    """Compute closure phases and squared visibilities for a binary model.

    Parameters
    ----------
    wavelength : float
        Central wavelength in microns.
    cp_uvs : numpy.ndarray
        Closure phase UV coordinates.
    v2_uvs : numpy.ndarray
        Baseline UV coordinates.
    model_params : array_like
        Binary model parameters [PA, sep, delta_mag, ...].
    angle : float
        Position angle offset in degrees.
    compute_kernel_phase : bool, optional
        Reserved for future kernel phase support.
    include_v2s : bool, optional
        If True, also return model squared visibilities.

    Returns
    -------
    closure_phases : numpy.ndarray
        Model closure phases in degrees.
    squared_visibilities : numpy.ndarray
        Only returned if ``include_v2s=True``.
    """
    cp_uvs_rad = cp_uvs / (wavelength * 1.0e-06) / 206265 * 2.0 * np.pi
    cp_uvs_rad[:, :, 1] *= -1.0

    v2_uvs_rad = v2_uvs / (wavelength * 1.0e-06) / 206265 * 2.0 * np.pi
    v2_uvs_rad[:, 1] *= -1.0

    n_params = len(model_params)
    n_companions = int(n_params / 3)
    pa_values = np.array([model_params[int(idx * 3)] for idx in range(n_companions)])
    sep_values = np.array([model_params[int(idx * 3 + 1)] for idx in range(n_companions)])
    dm_values = np.array([model_params[int(idx * 3 + 2)] for idx in range(n_companions)])

    pa_sky = np.round(90 - angle + pa_values, 3)
    flux_ratios = 10 ** (dm_values / -2.5)
    for idx in range(len(pa_sky)):
        while pa_sky[idx] < -180.0:
            pa_sky[idx] += 360
        while pa_sky[idx] > 180.0:
            pa_sky[idx] -= 360
    pa_rad = np.radians(pa_sky)
    x_offsets = sep_values * np.cos(pa_rad)
    y_offsets = sep_values * np.sin(pa_rad)

    cp_list = []
    for triangle in cp_uvs_rad:
        phases = []
        for pixel in triangle:
            u_coord, v_coord = pixel[0], pixel[1]
            phase_terms = u_coord * x_offsets + v_coord * y_offsets
            ft_value = 1.0 + np.sum(
                flux_ratios * (np.cos(phase_terms) + 1.0j * np.sin(phase_terms))
            )
            phases.append(np.angle(ft_value, deg=True))
        cp_list.append(np.sum(phases))

    if include_v2s:
        v2_list = []
        for baseline in v2_uvs_rad:
            u_coord, v_coord = baseline[0], baseline[1]
            phase_terms = u_coord * x_offsets + v_coord * y_offsets
            ft_value = 1.0 + np.sum(
                flux_ratios * (np.cos(phase_terms) + 1.0j * np.sin(phase_terms))
            )
            amplitude = np.abs(ft_value) / (1.0 + np.sum(flux_ratios))
            v2_list.append(amplitude ** 2)
        return np.array(cp_list), np.array(v2_list)
    else:
        return np.array(cp_list)


def _binary_chi2(cp_data, cp_errors, cp_uvs, v2_data, v2_errors, v2_uvs,
                 rotation_list, params, wavelength, include_v2s=True):
    """Compute chi-squared for a binary model at a set of rotation angles."""
    all_cp = []
    all_v2 = []
    for angle in rotation_list:
        if include_v2s:
            cp_model, v2_model = binary_model(
                wavelength, cp_uvs, v2_uvs, params,
                angle, include_v2s=True
            )
            all_cp.append(cp_model)
            all_v2.append(v2_model)
        else:
            cp_model = binary_model(
                wavelength, cp_uvs, v2_uvs, params,
                angle, include_v2s=False
            )
            all_cp.append(cp_model)

    chi2 = np.sum(
        (np.array(cp_data) - np.array(all_cp)) ** 2
        / np.array(cp_errors) ** 2
    )
    if include_v2s:
        chi2 += np.sum(
            (np.array(v2_data) - np.array(all_v2)) ** 2
            / np.array(v2_errors) ** 2
        )
    return chi2


def generate_chi2_grid(cp_data, cp_errors, cp_uvs, v2_data, v2_errors, v2_uvs,
                       rotation_list, wavelength, verbose=False,
                       include_v2s=True, n_pa=11, n_sep=51, n_dm=51,
                       max_sep=0.5, max_contrast=10.0):
    """Generate a chi-squared grid over PA, separation, and contrast.

    Parameters
    ----------
    cp_data : numpy.ndarray
        Closure phase data.
    cp_errors : numpy.ndarray
        Closure phase errors.
    cp_uvs : numpy.ndarray
        Closure phase UV coordinates.
    v2_data : numpy.ndarray
        Squared visibility data.
    v2_errors : numpy.ndarray
        Squared visibility errors.
    v2_uvs : numpy.ndarray
        Baseline UV coordinates.
    rotation_list : array_like
        Position angles for each pointing.
    wavelength : float
        Central wavelength in microns.
    verbose : bool, optional
        Print progress.
    include_v2s : bool, optional
        Include squared visibilities in the chi-squared.
    n_pa, n_sep, n_dm : int
        Number of grid points in PA, separation, and contrast.
    max_sep : float
        Maximum separation in arcseconds.
    max_contrast : float
        Maximum contrast in magnitudes.

    Returns
    -------
    grid : numpy.ndarray
        Chi-squared values, shape ``(n_pa, n_dm, n_sep)``.
    coordinates : numpy.ndarray
        Parameter coordinates at each grid point.
    """
    pa_grid = np.linspace(0.0, 360.0, n_pa)
    sep_grid = np.linspace(0.0, max_sep, n_sep)
    dm_grid = np.linspace(0.0, max_contrast, n_dm)

    chi2_grid = [[[[] for _ in sep_grid] for _ in dm_grid] for _ in pa_grid]
    coord_grid = [[[[] for _ in sep_grid] for _ in dm_grid] for _ in pa_grid]

    progress_bar = IntProgress(min=0, max=len(dm_grid) * len(sep_grid) * len(pa_grid))
    idisplay(progress_bar)
    progress_bar.value = 0

    for dm_idx, dm in enumerate(dm_grid):
        if verbose:
            print(dm)
        for sep_idx, sep in enumerate(sep_grid):
            for pa_idx, pa in enumerate(pa_grid):
                chi2 = _binary_chi2(
                    cp_data, cp_errors, cp_uvs,
                    v2_data, v2_errors, v2_uvs,
                    rotation_list, [pa, sep, dm], wavelength,
                    include_v2s=include_v2s
                )
                chi2_grid[pa_idx][dm_idx][sep_idx] = chi2
                coord_grid[pa_idx][dm_idx][sep_idx] = [
                    pa_grid[pa_idx], dm_grid[dm_idx], sep_grid[sep_idx]
                ]
                progress_bar.value += 1

    return np.array(chi2_grid), np.array(coord_grid)


def make_contrast_curve(chi2_grid, coordinates, cp_data, v2_data,
                        include_v2s=True, filename=None):
    """Generate a contrast curve from a chi-squared grid.

    Parameters
    ----------
    chi2_grid : numpy.ndarray
        Output from :func:`generate_chi2_grid`.
    coordinates : numpy.ndarray
        Coordinate array from :func:`generate_chi2_grid`.
    cp_data : numpy.ndarray
        Closure phase data (for DOF calculation).
    v2_data : numpy.ndarray
        Squared visibility data (for DOF calculation).
    include_v2s : bool, optional
        Whether V2s were included in the grid.
    filename : str or None, optional
        If given, save the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The contrast curve figure.
    contour_segments : list
        Contour segment data.
    """
    if include_v2s:
        dof = len(cp_data.flatten()) + len(v2_data.flatten()) - 3.0
    else:
        dof = len(cp_data.flatten()) - 3.0

    averaged = np.mean(chi2_grid, axis=0)
    averaged = averaged / np.min(averaged) * dof
    averaged = averaged - averaged[0, 0]

    seps = np.unique(coordinates[:, :, :, -1])
    dms = np.unique(coordinates[:, :, :, -2])
    sep_mesh, dm_mesh = np.meshgrid(seps, dms)

    fig = plt.figure(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.15)
    contours = plt.contour(sep_mesh, dm_mesh, averaged,
                           levels=[np.min(averaged), 1.0, 4.0, 9.0, 16.0, 25.0],
                           colors='k')
    filled = plt.contourf(sep_mesh, dm_mesh, averaged,
                          levels=[np.min(averaged), 1.0, 4.0, 9.0, 16.0, 25.0],
                          alpha=0.8)
    fig.patch.set_facecolor('white')
    contour_segments = contours.allsegs
    plt.ylim(11, 2)
    plt.grid(which='major')
    plt.xlabel('Separation (arcsec)', fontsize=14)
    plt.ylabel('Contrast (mag)', fontsize=14)
    colorbar = plt.colorbar(filled)
    colorbar.ax.yaxis.set_ticklabels(
        ['', r'$1 \sigma$', r'$2 \sigma$', r'$3 \sigma$',
         r'$4 \sigma$', r'$5 \sigma$'],
        fontsize=12
    )
    if filename is not None:
        plt.savefig(filename)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    return fig, contour_segments


def process_binary_grid(chi2_grid, coordinates, cp_data, v2_data,
                        filename=None, rescale=True, include_v2s=True):
    """Find the best-fit binary and plot the chi-squared surface.

    Parameters
    ----------
    chi2_grid : numpy.ndarray
        Output from :func:`generate_chi2_grid`.
    coordinates : numpy.ndarray
        Coordinate array from :func:`generate_chi2_grid`.
    cp_data : numpy.ndarray
        Closure phase data.
    v2_data : numpy.ndarray
        Squared visibility data.
    filename : str or None, optional
        If given, save the plot.
    rescale : bool, optional
        If True, rescale chi-squared to DOF.
    include_v2s : bool, optional
        Whether V2s were included in the fit.

    Returns
    -------
    best_model : list
        Best-fit [PA, separation, contrast] parameters.
    min_chi2 : float
        Minimum (rescaled) chi-squared value.
    """
    best_fit = coordinates[np.where(chi2_grid == np.min(chi2_grid))][0]
    best_model = [best_fit[0], best_fit[2], best_fit[1]]

    dof = len(cp_data.flatten()) - 3.0
    if include_v2s:
        dof += len(v2_data.flatten())

    best_dm_idx = np.where(chi2_grid == np.min(chi2_grid))[1][0]
    if rescale:
        grid_rescaled = chi2_grid * dof / np.min(chi2_grid)
    else:
        grid_rescaled = chi2_grid

    grid_slice = grid_rescaled[:, best_dm_idx, :]
    coord_slice = coordinates[:, best_dm_idx, :]
    pa_values = coord_slice[:, :, 0]
    sep_values = coord_slice[:, :, 2]

    x_coords = -np.sin(np.radians(pa_values)) * sep_values
    y_coords = np.cos(np.radians(pa_values)) * sep_values

    points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    values = grid_slice.ravel()

    interp_x = np.linspace(-np.max(sep_values), np.max(sep_values), 100)
    x_mesh, y_mesh = np.meshgrid(interp_x, interp_x)
    interpolated = griddata(points, values, (x_mesh, y_mesh))

    fig = plt.figure(figsize=(6, 5))
    fig.add_subplot(111)
    pcm = plt.pcolormesh(x_mesh, y_mesh, interpolated / np.nanmin(interpolated) * 32,
                         rasterized=True, cmap=plt.get_cmap('cubehelix'))
    plt.colorbar(pcm)
    plt.scatter(-np.sin(np.radians(best_model[0])) * best_model[1],
                np.cos(np.radians(best_model[0])) * best_model[1],
                marker='x', color='w', s=100)
    plt.xticks([-0.4, -0.2, 0.0, 0.2, 0.4],
               ['0.4', '0.2', '0.0', '-0.2', '-0.4'])
    plt.title(r'$\chi^2$ Surface: Contrast = ' + str(best_model[-1]) + ' mag')
    plt.xlabel(r'$\Delta$RA (arcsec)')
    plt.ylabel(r'$\Delta$DEC (arcsec)')
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    return best_model, np.min(grid_rescaled)
