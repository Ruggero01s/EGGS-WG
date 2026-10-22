"""Opt-in temperature residual conditioning and forward hourly VAR fitting."""

import numpy as np


def positive_conditioned_scale(hourly_scale, target_mean):
    """Retain additive conditioning unless it would produce invalid hourly SDs.

    Proportional rescaling is the positive fallback. Both constructions retain
    the fitted mean hourly SD; observations are not clipped or changed.
    """
    hourly_scale = np.asarray(hourly_scale, dtype=float)
    if (hourly_scale.ndim != 1 or not np.isfinite(hourly_scale).all()
            or np.any(hourly_scale <= 0) or not np.isfinite(target_mean) or target_mean <= 1e-8):
        raise ValueError('Positive scale conditioning requires positive hourly scales and target mean.')
    mean = hourly_scale.mean()
    additive = hourly_scale + (target_mean - mean)
    if np.all(additive > 1e-8):
        return additive, False
    return hourly_scale * (target_mean / mean), True


def conditioned_moments(wet, day_numbers, *, mean_wet, mean_dry, std_wet, std_dry):
    """Select the same calendar/wet-state templates as Full_Simulate_Day.

    The existing model maps ordinal day 366 to day 365. This helper retains
    that convention so residual fitting and simulation use identical inputs.
    Nonpositive scales fail instead of silently changing the fitted model.
    """
    wet = np.asarray(wet, dtype=bool)
    days = np.asarray(day_numbers)
    if wet.ndim != 2 or days.shape != (len(wet),):
        raise ValueError('Expected wet states (days, stations) and one ordinal per day.')
    if not np.isfinite(days).all() or np.any(days != days.astype(int)) or np.any((days < 1) | (days > 366)):
        raise ValueError('Temperature calendar ordinals must be integers from 1 to 366.')
    templates = {name: np.asarray(value, dtype=float) for name, value in
                 [('mean_wet', mean_wet), ('mean_dry', mean_dry),
                  ('std_wet', std_wet), ('std_dry', std_dry)]}
    shape = (365 * 24, wet.shape[1])
    for name, values in templates.items():
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f'Temperature {name} must be finite with shape {shape}.')
        if name.startswith('std') and np.any(values <= 1e-8):
            raise ValueError(f'Matched temperature fit requires {name} > 1e-8 everywhere; '
                             f'minimum={values.min():.6g}. Review seasonal scale fitting; no clipping applied.')
    hours = np.minimum(days.astype(int) - 1, 364)[:, None] * 24 + np.arange(24)[None, :]
    occurrence = wet[:, None, :]
    mean = np.where(occurrence, templates['mean_wet'][hours], templates['mean_dry'][hours])
    std = np.where(occurrence, templates['std_wet'][hours], templates['std_dry'][hours])
    return mean, std


def matched_residuals(temperature, wet, day_numbers, **templates):
    """Standardize observations with their simulation reconstruction templates."""
    temperature = np.asarray(temperature, dtype=float)
    mean, std = conditioned_moments(wet, day_numbers, **templates)
    if temperature.shape != mean.shape or not np.isfinite(temperature).all():
        raise ValueError(f'Expected finite temperatures with shape {mean.shape}.')
    return (temperature - mean) / std


def fit_forward_var(current, previous, *, maximum_radius=0.999):
    """Fit z[t] = X z[t-1] + Y epsilon[t], using genuine adjacent pairs.

    Data have shape (variables, paired observations). Least squares uses the
    forward cross-covariance, Cov(current, previous), rather than its transpose.
    Monthly residual means are removed for fitting; the generated residual
    process remains zero-mean. Unstable fitted transitions are scaled to the
    stated radius, with the change recorded. Innovations are recomputed after
    that scaling, then factored by a symmetric PSD eigendecomposition.
    """
    current, previous = np.asarray(current, dtype=float), np.asarray(previous, dtype=float)
    if (current.ndim != 2 or previous.shape != current.shape
            or current.shape[1] <= current.shape[0] + 1):
        raise ValueError('Forward VAR needs equally shaped (variables, pairs) arrays and more pairs than variables + 1.')
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError('Forward VAR received non-finite standardized residuals.')
    if not 0 < maximum_radius < 1:
        raise ValueError('Forward VAR maximum_radius must lie between 0 and 1.')
    current_mean, previous_mean = current.mean(axis=1), previous.mean(axis=1)
    response = current - current_mean[:, None]
    predictor = previous - previous_mean[:, None]
    transition = np.linalg.lstsq(predictor.T, response.T, rcond=None)[0].T
    radius_before = float(np.max(np.abs(np.linalg.eigvals(transition))))
    scale = min(1., maximum_radius / radius_before) if radius_before else 1.
    transition *= scale
    innovations = response - transition @ predictor
    covariance = innovations @ innovations.T / innovations.shape[1]
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = np.finfo(float).eps * max(1., float(np.linalg.norm(covariance, 2))) * len(covariance) * 100
    if eigenvalues.min() < -tolerance:
        raise ValueError('Forward VAR innovation covariance is not positive semidefinite.')
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))[None, :]
    diagnostics = {'pairs': current.shape[1], 'variables': current.shape[0],
                   'spectral_radius_before': radius_before,
                   'spectral_radius': float(np.max(np.abs(np.linalg.eigvals(transition)))),
                   'stability_scale': float(scale),
                   'innovation_minimum_eigenvalue': float(eigenvalues.min()),
                   'maximum_absolute_monthly_residual_mean': float(max(np.abs(current_mean).max(), np.abs(previous_mean).max()))}
    return transition, factor, diagnostics
