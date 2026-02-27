"""MCMC model fitting using emcee's parallel-tempered sampler."""

import sys
import os
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import corner as corner_plot
import emcee
from emcee import PTSampler
from ipywidgets import IntProgress
from IPython.display import display as idisplay

# emcee 2.x PTSampler references np.float, which was removed in numpy 1.24.
# This shim restores it.  When the project migrates to emcee 3.x (which
# replaced PTSampler with a different API), this line can be deleted.
if not hasattr(np, 'float'):
    np.float = float

def binary_phase(wavelength, cp_uvs, v2_uvs, model_params, angle,
                 include_v2s=True):
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
        Model parameters as [PA, separation, delta_mag, ...] triplets.
    angle : float
        Position angle offset in degrees.
    include_v2s : bool, optional
        If True, also return model squared visibilities.

    Returns
    -------
    closure_phases : numpy.ndarray
        Model closure phases in degrees.
    squared_visibilities : numpy.ndarray
        Model squared visibilities (only if ``include_v2s=True``).
    """
    cp_uvs_rad = cp_uvs / (wavelength * 1.0e-06) / 206265 * 2.0 * np.pi
    cp_uvs_rad[:, :, 1] *= -1.0

    v2_uvs_rad = v2_uvs / (wavelength * 1.0e-06) / 206265 * 2.0 * np.pi
    v2_uvs_rad[:, 1] *= -1.0

    n_params = len(model_params)
    n_companions = int(n_params / 3)
    pa_values = np.array([model_params[idx * 3] for idx in range(n_companions)])
    sep_values = np.array([model_params[idx * 3 + 1] for idx in range(n_companions)])
    dm_values = np.array([model_params[idx * 3 + 2] for idx in range(n_companions)])

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
        closure_phase = 0.0
        for pixel in triangle:
            u_coord, v_coord = pixel[0], pixel[1]
            phase_terms = u_coord * x_offsets + v_coord * y_offsets
            ft_value = 1.0 + np.sum(
                flux_ratios * (np.cos(phase_terms) + 1.0j * np.sin(phase_terms))
            )
            closure_phase += np.angle(ft_value, deg=True)
        cp_list.append(closure_phase)

    if include_v2s:
        v2_list = []
        for baseline in v2_uvs_rad:
            u_coord, v_coord = baseline[0], baseline[1]
            phase_terms = u_coord * x_offsets + v_coord * y_offsets
            ft_value = 1.0 + np.sum(
                flux_ratios * (np.cos(phase_terms) + 1.0j * np.sin(phase_terms))
            )
            v2 = (np.abs(ft_value) / (1.0 + np.sum(flux_ratios))) ** 2
            v2_list.append(v2)
        return np.array(cp_list), np.array(v2_list)
    else:
        return np.array(cp_list)


def binary_cps_pixel_average(wavelength, cp_uvs_scaled, model_params, angle,
                             compute_kernel_phase=False, average=True):
    """Compute closure phases from a binary model using pixel averaging.

    Parameters
    ----------
    wavelength : float
        Central wavelength in microns.
    cp_uvs_scaled : numpy.ndarray
        Pre-scaled closure phase UV coordinates.
    model_params : array_like
        Binary model parameters.
    angle : float
        Position angle offset in degrees.
    compute_kernel_phase : bool, optional
        Reserved for future kernel phase support.
    average : bool, optional
        If True, return the average bispectrum phase per triangle.

    Returns
    -------
    numpy.ndarray
        Model closure phases.
    """
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
    for triangle in cp_uvs_scaled:
        u_vals = triangle[:, :, 0]
        v_vals = triangle[:, :, 1]
        phase_terms = np.array([
            u_vals * x_offsets[comp] + v_vals * y_offsets[comp]
            for comp in range(len(x_offsets))
        ])
        ft_values = np.sum(1.0 + np.array([
            flux_ratios[comp] * (np.cos(phase_terms[comp])
                                 + 1.0j * np.sin(phase_terms[comp]))
            for comp in range(len(flux_ratios))
        ]), axis=0)
        if average:
            cp = np.angle(np.mean(np.prod(ft_values, axis=-1)), deg=True)
        else:
            cp = np.angle(np.prod(ft_values, axis=-1), deg=True)
        cp_list.append(cp)
    return np.array(cp_list)


def _log_likelihood(params, v2_data, v2_errors, bl_uvs_scaled,
                    cp_data, cp_errors, cp_uvs_scaled, angles,
                    include_v2s, wavelength, avg_pixels, all_pixels):
    """Log-likelihood function for the binary model."""
    all_cp = []
    all_v2 = []
    for angle in angles:
        if include_v2s:
            cp_model, v2_model = binary_phase(
                wavelength, cp_uvs_scaled, bl_uvs_scaled,
                params, angle, include_v2s=True
            )
            all_v2.append(v2_model)
        else:
            if avg_pixels:
                cp_model = binary_cps_pixel_average(
                    wavelength, cp_uvs_scaled, params, angle
                )
            elif all_pixels:
                cp_model = binary_cps_pixel_average(
                    wavelength, cp_uvs_scaled, params, angle, average=False
                )
            else:
                cp_model = binary_phase(
                    wavelength, cp_uvs_scaled, bl_uvs_scaled,
                    params, angle, include_v2s=False
                )
        all_cp.append(cp_model)

    chi2_cp = np.sum((np.array(all_cp) - np.array(cp_data)) ** 2
                     / np.array(cp_errors) ** 2)
    chi2_v2 = 0
    if include_v2s:
        chi2_v2 = np.sum((np.array(all_v2) - np.array(v2_data)) ** 2
                         / np.array(v2_errors) ** 2)
    return -0.5 * (chi2_cp + chi2_v2)


def _log_prior(model_params):
    """Flat prior on binary model parameters."""
    n_companions = int(len(model_params) / 3)
    pa_values = np.array([model_params[idx * 3] for idx in range(n_companions)])
    sep_values = np.array([model_params[idx * 3 + 1] for idx in range(n_companions)])
    dm_values = np.array([model_params[idx * 3 + 2] for idx in range(n_companions)])

    for pa in pa_values:
        if pa < -180.0 or pa >= 180.0:
            return -np.inf
    for sep in sep_values:
        if sep < 0.0 or sep > 0.5:
            return -np.inf
    for dm in dm_values:
        if dm < 0.0 or dm > 10.0:
            return -np.inf
    for idx in range(len(pa_values)):
        for jdx in range(idx):
            if jdx > idx and pa_values[idx] > pa_values[jdx]:
                return -np.inf
    return 0.0


def run_pt_emcee(v2_data, v2_errors, v2_uvs, cp_data, cp_errors, cp_uvs,
                 angles, wavelength, include_v2s=False, n_dim=3,
                 n_walkers=100, n_temps=10, n_threads=1, write_interval=100,
                 output_dir='./', n_iterations=1000, verbose=False,
                 overwrite=True, suffix='', avg_pixels=False, all_pixels=False):
    """Run a parallel-tempered MCMC fit for a binary companion model.

    Parameters
    ----------
    v2_data : numpy.ndarray
        Squared visibility data.
    v2_errors : numpy.ndarray
        Squared visibility errors.
    v2_uvs : numpy.ndarray
        Baseline UV coordinates.
    cp_data : numpy.ndarray
        Closure phase data.
    cp_errors : numpy.ndarray
        Closure phase errors.
    cp_uvs : numpy.ndarray
        Closure phase UV coordinates.
    angles : array_like
        Position angles for each pointing.
    wavelength : float
        Central wavelength in microns.
    include_v2s : bool, optional
        Include squared visibilities in the fit.
    n_dim : int, optional
        Number of model parameters.
    n_walkers : int, optional
        Number of MCMC walkers.
    n_temps : int, optional
        Number of temperatures for parallel tempering.
    n_threads : int, optional
        Number of threads.
    write_interval : int, optional
        Write chain to disk every this many steps.
    output_dir : str, optional
        Directory for chain output files.
    n_iterations : int, optional
        Total number of MCMC iterations.
    verbose : bool, optional
        Print progress every 10 steps.
    overwrite : bool, optional
        If False, resume from existing chain files.
    suffix : str, optional
        Suffix for output filenames.
    avg_pixels : bool, optional
        Use pixel-averaged closure phases.
    all_pixels : bool, optional
        Use all-pixel closure phases.

    Returns
    -------
    chain, log_priors, log_likelihoods : list
        MCMC chain and log-probability arrays.
    """
    param_starts = [-180.0, 0, 0]
    param_scales = [350.0, 0.5, 10.0]
    initial_positions = np.array([
        [np.array(param_starts) + np.array(param_scales)
         * np.random.uniform(low=0.0, high=1.0, size=len(param_scales))
         for _ in range(n_walkers)]
        for _ in range(n_temps)
    ])

    # Sort PAs for ordering constraint
    for temp_idx in range(n_temps):
        for walker_idx in range(n_walkers):
            params = initial_positions[temp_idx, walker_idx]
            n_companions = int(len(params) / 3)
            pa_values = np.array([params[int(comp * 3)] for comp in range(n_companions)])
            pa_sorted = np.sort(pa_values)
            pa_indices = [int(comp * 3) for comp in range(n_companions)]
            for comp in range(len(pa_indices)):
                params[pa_indices[comp]] = pa_sorted[comp]
            initial_positions[temp_idx, walker_idx] = params

    sampler = PTSampler(
        n_temps, n_walkers, n_dim, _log_likelihood, _log_prior,
        loglargs=[v2_data, v2_errors, v2_uvs, cp_data, cp_errors,
                  cp_uvs, angles, include_v2s, wavelength,
                  avg_pixels, all_pixels],
        threads=n_threads,
    )

    # Resume or start fresh
    if not overwrite:
        chain_file = output_dir + f'lnls{suffix}.fits'
        if os.path.isfile(chain_file):
            chain_list = list(pyfits.getdata(output_dir + f'chain{suffix}.fits'))
            lnp_list = list(pyfits.getdata(output_dir + f'lnps{suffix}.fits'))
            lnl_list = list(pyfits.getdata(output_dir + f'lnls{suffix}.fits'))
            step_count = len(chain_list)
            initial_positions = chain_list[-1]
        else:
            chain_list, lnp_list, lnl_list = [], [], []
            step_count = 0
    else:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        chain_list, lnp_list, lnl_list = [], [], []
        step_count = 0

    write_count = 0
    print('about to sample')
    print(f'running steps {step_count} to {n_iterations}')
    progress_bar = IntProgress(min=step_count, max=n_iterations)
    idisplay(progress_bar)
    progress_bar.value = step_count

    for result in sampler.sample(initial_positions,
                                 iterations=n_iterations - step_count,
                                 storechain=False):
        if verbose and step_count % 10 == 0:
            print(step_count)
        positions, log_prior, log_like = result
        chain_list.append(positions)
        lnp_list.append(log_prior)
        lnl_list.append(log_like)
        progress_bar.value += 1
        step_count += 1
        write_count += 1
        if write_count == write_interval:
            pyfits.writeto(output_dir + f'chain{suffix}.fits',
                           np.array(chain_list), overwrite=True)
            pyfits.writeto(output_dir + f'lnps{suffix}.fits',
                           np.array(lnp_list), overwrite=True)
            pyfits.writeto(output_dir + f'lnls{suffix}.fits',
                           np.array(lnl_list), overwrite=True)
            write_count = 0

    pyfits.writeto(output_dir + f'chain{suffix}.fits',
                   np.array(chain_list), overwrite=True)
    pyfits.writeto(output_dir + f'lnps{suffix}.fits',
                   np.array(lnp_list), overwrite=True)
    pyfits.writeto(output_dir + f'lnls{suffix}.fits',
                   np.array(lnl_list), overwrite=True)
    return chain_list, lnp_list, lnl_list


def plot_corner(chain, log_likelihoods, n_temps=10, n_walkers=100,
                burnin=100, title='', truths=None, filename='',
                param_names=None, smooth=False, figsize=(7, 6),
                degrees_of_freedom=None):
    """Make a corner plot from an MCMC chain and report best-fit parameters.

    Parameters
    ----------
    chain : numpy.ndarray
        MCMC chain array.
    log_likelihoods : numpy.ndarray
        Log-likelihood values.
    n_temps : int
        Number of temperatures used.
    n_walkers : int
        Number of walkers used.
    burnin : int
        Number of burn-in steps to discard.
    title : str
        Plot title.
    truths : list or None
        True parameter values to overplot.
    filename : str
        If non-empty, save the figure to this path.
    param_names : list of str or None
        Labels for each parameter.
    smooth : bool
        Apply smoothing to the corner plot.
    figsize : tuple
        Figure size.
    degrees_of_freedom : int
        Degrees of freedom for reduced chi-squared calculation
        (typically ``n_data_points - n_model_params``).  Required.

    Returns
    -------
    best_params : numpy.ndarray
        Maximum-likelihood parameters.
    median_params : list
        Median and 1-sigma intervals from the posterior.
    """
    if degrees_of_freedom is None:
        raise ValueError(
            "degrees_of_freedom is required. Pass the number of data points "
            "minus the number of model parameters (e.g. n_cp + n_v2 - n_dim)."
        )
    if truths is None:
        truths = []
    if param_names is None:
        param_names = [r'PA ($^\circ$)', r'$\rho$ (arcsec)', r'$\Delta$ (mag)']

    lnl_reshaped = log_likelihoods.reshape([n_temps, n_walkers,
                                             len(log_likelihoods)])
    n_dim = len(chain[0, 0, 0])

    logl_dummy = 0
    logp_dummy = 0
    temp_sampler = emcee.PTSampler(n_temps, n_walkers, n_dim,
                                    logl_dummy, logp_dummy)
    log_evidence, log_evidence_err = \
        temp_sampler.thermodynamic_integration_log_evidence(
            lnl_reshaped, fburnin=0.0
        )
    print(log_evidence, log_evidence_err)
    print(chain.shape)

    flat_chain = chain[burnin:, 0, :].reshape(
        [(len(chain) - burnin) * n_walkers, n_dim]
    )
    print(flat_chain.shape)

    fig = plt.figure(figsize=figsize)
    if len(truths) > 0:
        corner_plot.corner(flat_chain, labels=param_names, fig=fig,
                           max_n_ticks=4, truths=truths, fontsize=10,
                           smooth=smooth)
    else:
        corner_plot.corner(flat_chain, labels=param_names, fig=fig,
                           max_n_ticks=4, fontsize=10, smooth=smooth)
    plt.subplots_adjust(top=0.925)
    fig.suptitle(title, fontsize=12)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

    best_params = chain[np.where(log_likelihoods == np.max(log_likelihoods))][0]
    print(np.max(log_likelihoods) * -2)
    print('reduced chi^2 =', np.max(log_likelihoods) * -2 / degrees_of_freedom)
    print('inflate errors by:',
          np.sqrt(np.max(log_likelihoods) * -2 / degrees_of_freedom),
          'to get red chi^2=1')
    print(best_params)

    percentiles = list(map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(flat_chain, [16, 50, 84], axis=0))
    ))
    median_params = []
    for interval in percentiles:
        print(interval)
        median_params.append(interval[0])

    return best_params, median_params
