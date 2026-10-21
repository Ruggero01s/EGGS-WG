"""Batched calibration of EGGS-WG's Gaussian spatial rainfall copulas.

The Markov transitions, rainfall transform, sample lengths, and 15 Monte Carlo
replicates follow upstream. Draws are batched and Cholesky factors reused.
Common random numbers make candidate comparisons reproducible.
"""

import numpy as np
from scipy.special import ndtr


def correlation_matrix(matrix, floor=1e-8):
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError('Spatial calibration requires finite observed correlations.')
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    positive = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    scale = np.sqrt(np.diag(positive))
    positive /= scale[:, None] * scale[None, :]
    np.fill_diagonal(positive, 1)
    return positive


def occurrence_samples(normals, p0, p1):
    batches, days, stations = normals.shape
    state = np.zeros((batches, stations), dtype=bool)
    wet = np.empty(normals.shape, dtype=bool)
    thresholds = np.empty(normals.shape)
    for day in range(days):
        threshold = np.where(state, p1, p0)
        state = normals[:, day] <= threshold
        wet[:, day] = state
        thresholds[:, day] = threshold
    return wet, thresholds


def sample_correlation(wet, amounts=None):
    correlations = []
    for index, mask in enumerate(wet):
        weights = mask.astype(float)
        if amounts is None:
            centred = weights - weights.mean(axis=0)
            covariance = centred.T @ centred
            variance = np.diag(covariance)
            denominator = np.sqrt(variance[:, None] * variance[None, :])
        else:
            values = amounts[index]
            count = weights.T @ weights
            sums = values.T @ weights
            squares = (values ** 2).T @ weights
            safe = np.maximum(count, 1)
            covariance = values.T @ values - sums * sums.T / safe
            variance = np.maximum(squares - sums ** 2 / safe, 0)
            denominator = np.sqrt(variance * variance.T)
        result = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
        np.fill_diagonal(result, 1)
        correlations.append(np.clip(result, -1, 1))
    return np.mean(correlations, axis=0)


def calibrate(target, initial, evaluate, month, kind, rates, max_steps=40):
    if len(target) == 1:
        return np.eye(1), {'month': month + 1, 'kind': kind, 'mean_absolute_error': 0.0, 'iterations': 0}
    mask = ~np.eye(len(target), dtype=bool)
    current = correlation_matrix(initial)
    estimate = evaluate(current)
    error = float(np.abs(target - estimate)[mask].mean())
    initial_error = error
    phase = 0
    steps = 0
    while error > 0.005 and phase < len(rates) and steps < max_steps:
        candidate = correlation_matrix(current + rates[phase] * (target - estimate))
        candidate_estimate = evaluate(candidate)
        candidate_error = float(np.abs(target - candidate_estimate)[mask].mean())
        steps += 1
        if candidate_error < error:
            current, estimate, error = candidate, candidate_estimate, candidate_error
        else:
            phase += 1
        if steps % 5 == 0:
            print(f'{kind} month {month + 1}: correlation MAE {error:.4f}, step {steps}', flush=True)
    report = {'month': month + 1, 'kind': kind, 'initial_mean_absolute_error': initial_error,
              'mean_absolute_error': error, 'iterations': steps, 'target_tolerance': 0.005,
              'converged': error <= 0.005, 'maximum_absolute_error': float(np.max(np.abs(target - estimate)[mask]))}
    print(f'{kind} month {month + 1} complete: MAE {error:.4f} after {steps} steps', flush=True)
    return current, report


def calibrate_occurrence(p0, p1, target, initial, month, seed):
    if len(p0) == 1:
        return np.eye(1), {'month': month + 1, 'kind': 'occurrence', 'mean_absolute_error': 0.0, 'iterations': 0}
    rng = np.random.default_rng(np.random.SeedSequence([seed, month, 0]))
    normals = rng.standard_normal((15, 6000, len(p0)))

    def evaluate(matrix):
        correlated = normals @ np.linalg.cholesky(matrix).T
        wet, _ = occurrence_samples(correlated, p0, p1)
        return sample_correlation(wet)

    return calibrate(target, initial, evaluate, month, 'occurrence', [0.4, 0.2, 0.3, 0.05])


def rainfall_amounts(alpha, beta1, beta2, occurrence_normals, amount_normals, thresholds, wet):
    uniform = ndtr(occurrence_normals)
    scale = alpha * ndtr(thresholds)
    ratio = np.divide(uniform, scale, out=np.zeros_like(uniform), where=scale > 0)
    beta = np.where(uniform < scale, beta2 + 2 * (beta1 - beta2) * (1 - ratio), beta2)
    probability = np.minimum(ndtr(amount_normals), 1 - np.finfo(float).eps)
    return (1 - beta * np.log1p(-probability)) * wet


def calibrate_amounts(p0, p1, target, initial, occurrence_omega, coefficients, month, seed):
    if len(p0) == 1:
        return np.eye(1), {'month': month + 1, 'kind': 'amounts', 'mean_absolute_error': 0.0, 'iterations': 0}
    rng = np.random.default_rng(np.random.SeedSequence([seed, month, 1]))
    normals = rng.standard_normal((15, 12000, len(p0))) @ np.linalg.cholesky(occurrence_omega).T
    wet, thresholds = occurrence_samples(normals, p0, p1)
    amount_normals = rng.standard_normal(normals.shape)
    alpha, beta1, beta2 = coefficients
    uniform = ndtr(normals)
    scale = alpha * ndtr(thresholds)
    ratio = np.divide(uniform, scale, out=np.zeros_like(uniform), where=scale > 0)
    beta = np.where(uniform < scale, beta2 + 2 * (beta1 - beta2) * (1 - ratio), beta2)

    def evaluate(matrix):
        probability = np.minimum(ndtr(amount_normals @ np.linalg.cholesky(matrix).T), 1 - np.finfo(float).eps)
        amounts = (1 - beta * np.log1p(-probability)) * wet
        return sample_correlation(wet, amounts)

    return calibrate(target, initial, evaluate, month, 'amounts', [0.4, 0.25, 0.1])
