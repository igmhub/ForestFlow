"""Notebook-only GP prototype; LaCE optimizer and ForestFlow transforms."""
from pathlib import Path
import hashlib
import pickle
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from lace.emulator.gp_emulator_multi import optimizer


def columns(records, labels, fit_key=None):
    values = {name: np.asarray([r[name] if fit_key is None else r[fit_key][name]
                               for r in records], dtype=float) for name in labels}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite archive inputs/targets; inspect the selected records.')
    return values


def matrix(transform, values, labels, kind):
    transformed = transform.transf_stand(values, type_stand=kind, direct=True)
    result = np.column_stack([transformed[name] for name in labels])
    if not np.isfinite(result).all():
        raise ValueError('Invalid transform domain in selected records')
    return result


def short_optimizer(objective, initial, bounds):
    """Small, serializable optimization budget for smoke execution only."""
    from scipy.optimize import minimize
    result = minimize(objective, initial, jac=True, bounds=bounds,
                      method='L-BFGS-B', options={'maxiter': 15})
    return result.x, result.fun


def fit_gp(x, y, nu, alpha, smoke=False):
    # Smoke limits optimization; full experiment reuses LaCE's optimizer.
    return GaussianProcessRegressor(
        kernel=Matern(length_scale=np.ones(x.shape[1]), nu=nu),
        alpha=alpha, normalize_y=False,
        optimizer=short_optimizer if smoke else optimizer,
        random_state=32, n_restarts_optimizer=0,
    ).fit(x, y)


def transformed_prediction(bundle, inputs, return_std=False):
    single = isinstance(inputs, dict)
    rows = [inputs] if single else inputs
    x = matrix(bundle['transform'], columns(rows, bundle['inputs']), bundle['inputs'], 'input')
    return bundle['gp'].predict(x, return_std=return_std)


def physical_prediction(bundle, inputs):
    pred = transformed_prediction(bundle, inputs)
    values = {name: pred[:, j] for j, name in enumerate(bundle['outputs'])}
    physical = bundle['transform'].transf_stand(values, type_stand='output', direct=False)
    return {k: v[0] for k, v in physical.items()} if isinstance(inputs, dict) else physical


def save_bundle(path, bundle):
    with Path(path).open('wb') as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path):
    # Local experiment artifacts only; pickle is executable serialization.
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def fingerprint(x, y):
    return hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()


def metrics(pred, ref):
    residual = np.asarray(pred) - np.asarray(ref)
    if residual.shape != np.asarray(ref).shape or not np.isfinite(residual).all():
        raise ValueError('Nonfinite or mismatched predictions')
    return dict(bias=residual.mean(0), rmse=np.sqrt((residual**2).mean(0)),
                p99=np.percentile(np.abs(residual), 99, axis=0),
                maximum=np.abs(residual).max(0))


def latency(call, repeats=20):
    call()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return {'median_ms': 1000*np.median(times), 'p95_ms': 1000*np.percentile(times, 95)}
