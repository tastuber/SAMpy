"""Polynomial calibration of closure phases and squared visibilities."""

import numpy as np
import matplotlib.pyplot as plt


def evaluate_polynomial(coefficients, coeff_variance, time_value):
    """Evaluate a polynomial and its variance at a given time.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Polynomial coefficients (column vector from least-squares fit).
    coeff_variance : numpy.ndarray
        Diagonal of the coefficient covariance matrix.
    time_value : float
        Time at which to evaluate the polynomial.

    Returns
    -------
    result : float
        Polynomial value.
    variance : float
        Propagated variance.
    """
    result = 0.0
    variance = 0.0
    for order in range(len(coefficients)):
        coeff = coefficients[order, 0]
        result += coeff * time_value ** order
        variance += time_value ** order * coeff_variance[order]
    return result, variance


def polynomial_calibrate(target_observables, calibrator_observables,
                         target_times, calibrator_times, poly_order,
                         data_type, target_variance=None,
                         calibrator_variance=None, display=False):
    """Calibrate observables using polynomial interpolation of calibrator data.

    Fits a polynomial of the specified order to the calibrator observables
    as a function of time, then uses it to calibrate the target observables.
    For closure phases (``data_type='cps'``), calibration is subtractive.
    For squared visibilities (``data_type='v2s'``), calibration is divisive.

    Parameters
    ----------
    target_observables : numpy.ndarray
        Shape ``(n_pointings, n_observables)``.
    calibrator_observables : numpy.ndarray
        Shape ``(n_cal_pointings, n_observables)``.
    target_times : numpy.ndarray
        UT times for each target pointing.
    calibrator_times : numpy.ndarray
        UT times for each calibrator pointing.
    poly_order : int
        Order of the polynomial to fit.
    data_type : str
        ``'cps'`` for closure phases (subtractive) or ``'v2s'``
        for squared visibilities (divisive).
    target_variance : numpy.ndarray or None, optional
        Variance of target observables, same shape as ``target_observables``.
    calibrator_variance : numpy.ndarray or None, optional
        Variance of calibrator observables.
    display : bool, optional
        If True, plot the calibration for each observable.

    Returns
    -------
    calibrated : numpy.ndarray
        Calibrated observables.
    cal_variance : numpy.ndarray
        Propagated variance of calibrated observables.
    cal_scatter : numpy.ndarray
        Scatter-based variance estimate from calibrator data.
    cal_parameters : numpy.ndarray
        Fitted polynomial parameters for each observable.
    """
    cal_parameters = []
    calibrated = []
    cal_variance = []
    cal_scatter = []

    print(f'poly_order = {poly_order}')

    n_observables = len(target_observables[0])

    for obs_idx in range(n_observables):
        # Build the data vector
        data = np.array([calibrator_observables[pointing, obs_idx]
                         for pointing in range(len(calibrator_observables))])

        # Build the weight matrix
        if calibrator_variance is not None and len(calibrator_variance) > 0:
            var = np.array([calibrator_variance[pointing, obs_idx]
                           for pointing in range(len(calibrator_observables))])
            weight_matrix = np.diag(1.0 / np.array(var))
        else:
            weight_matrix = np.diag(np.ones(len(calibrator_observables)))

        # Design matrix
        cal_t = np.array(calibrator_times)
        design_matrix = np.array([
            [t_val ** order for order in range(poly_order + 1)]
            for t_val in cal_t
        ])
        data_vector = np.array(data).reshape(len(data), 1)

        # Weighted least-squares fit: coeffs = (X^T W X)^-1 X^T W d
        xtwx = design_matrix.T @ weight_matrix @ design_matrix
        xtwx_inv = np.linalg.inv(xtwx)
        coeffs = xtwx_inv @ design_matrix.T @ weight_matrix @ data_vector
        coeff_var = np.diag(xtwx_inv)
        cal_parameters.append([np.array(coeffs)[:, 0], coeff_var])

        # Evaluate model at calibrator and target times
        all_times = list(cal_t) + list(target_times)
        model_times = np.linspace(np.min(all_times), np.max(all_times), 100)
        model_values = [evaluate_polynomial(coeffs, coeff_var, t)[0]
                        for t in model_times]
        cal_predictions = [evaluate_polynomial(coeffs, coeff_var, t)[0]
                           for t in cal_t]

        target_predictions = [evaluate_polynomial(coeffs, coeff_var, t)[0]
                              for t in target_times]
        target_pred_var = [evaluate_polynomial(coeffs, coeff_var, t)[1]
                           for t in target_times]

        if display:
            t_min = np.min([np.min(cal_t), np.min(target_times)])
            print((cal_t - t_min) * 24)
            plt.figure(figsize=(4, 3.5))
            plt.title(f' Triangle {obs_idx}; Polycal Order {poly_order}')
            plt.plot((model_times - t_min) * 24.0, model_values,
                     color='k', label='Model', zorder=-1)
            plt.plot([(target_times - t_min) * 24.0,
                       (target_times - t_min) * 24],
                     [target_predictions, target_observables[:, obs_idx]],
                     'k--', lw=0.5)
            plt.scatter((cal_t - t_min) * 24.0, data,
                        facecolors='grey', edgecolors='k', label='Ref. PSF')
            plt.scatter((target_times - t_min) * 24,
                        target_observables[:, obs_idx],
                        facecolors='purple', edgecolors='k', label='Science')
            plt.legend()
            plt.ylabel(r'CP ($^\circ$)')
            plt.xlabel('Time (hours)')
            plt.subplots_adjust(left=0.2)
            plt.show()

        if data_type == 'cps':
            calibrated.append(target_observables[:, obs_idx] - target_predictions)
            if target_variance is not None and len(target_variance) > 0:
                cal_variance.append(target_variance[:, obs_idx] + target_pred_var)
            else:
                cal_variance.append(target_pred_var)
        elif data_type == 'v2s':
            calibrated.append(target_observables[:, obs_idx] / target_predictions)
            if target_variance is not None and len(target_variance) > 0:
                cal_variance.append(target_variance[:, obs_idx] + target_pred_var)
            else:
                cal_variance.append(target_pred_var)

        cal_scatter.append([np.std(calibrator_observables[:, obs_idx])
                            for _ in target_observables[:, obs_idx]])

    calibrated = np.array(calibrated)
    cal_variance = np.array(cal_variance)
    cal_scatter = np.array(cal_scatter)
    cal_parameters = np.array(cal_parameters, dtype=object)
    return calibrated, cal_variance, cal_scatter, cal_parameters
