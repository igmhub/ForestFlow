import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt


def P1D_Mpc(
    z,
    k_par,
    p3d_fun,
    p3d_params={},
    new_cosmo_params=None,
    k_perp_min=0.001,
    k_perp_max=100,
    n_k_perp=99,
    **kwargs,
):
    """
    Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

    Parameters:
        z (float): Redshift. It modifies the linear power spectrum but not the value of the Arinyo parameters.
        k_par (array-like): Array or list of values for which P1D is to be computed.
        p3d_fun (function): Function that returns P3D. It takes as input z, k/kpar, mu/kperp, with the difference
            depending on the value of p3d_fun.coordinates. It also takes as input p3d_params and optionally new_cosmo_params.
        p3d_params (dict, optional): Additional parameters for the model. Defaults to {}.
        new_cosmo_params (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.
        k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
        k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
        n_k_perp (int, optional): Number of points in integral. Defaults to 99.

    Returns:
        array-like: Computed values of P1D.
    """

    ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)

    p1d = _P1D_lnkperp_fast(
        z,
        ln_k_perp,
        k_par,
        p3d_fun,
        p3d_params,
        new_cosmo_params=new_cosmo_params,
        **kwargs,
    )

    return p1d


def _P1D_lnkperp_fast(
    z, ln_k_perp, kpars, p3d_fun, p3d_params={}, new_cosmo_params=None, **kwargs
):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp) using a fast method.

    Parameters:
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        p3d_fun (function): Function that returns P3D.
        p3d_params (dict, optional): Additional parameters for the model. Defaults to {}.
        new_cosmo_params (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

    Returns:
        array-like: Computed values of P1D.
    """

    # get interval for integration
    dlnk = ln_k_perp[1] - ln_k_perp[0]

    # get function to be integrated
    # it is equivalent of the inner loop of _P1D_lnkperp
    k_perp = np.exp(ln_k_perp)
    fact = (1 / (2 * np.pi)) * k_perp[:, np.newaxis] ** 2
    fact = fact.swapaxes(0, 1)

    k = np.sqrt(kpars[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
    mu = kpars[np.newaxis, :] / k
    k = k.swapaxes(0, 1)
    mu = mu.swapaxes(0, 1)

    if p3d_fun.coordinates == "k_mu":
        p3d_fix_k_par = (
            p3d_fun(z, k, mu, p3d_params, new_cosmo_params=new_cosmo_params, **kwargs)
            * fact
        )
    elif p3d_fun.coordinates == "kpar_kperp":
        kpar = k * mu
        kperp = k * np.sqrt(1 - mu**2)
        p3d_fix_k_par = (
            p3d_fun(
                z, kpar, kperp, p3d_params, new_cosmo_params=new_cosmo_params, **kwargs
            )
            * fact
        )
    else:
        raise ValueError(
            "p3d_fun must have coordinates attribute set to 'k_mu' or 'kpar_kperp'"
        )

    # perform numerical integration
    p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

    return p1d


def _P1D_lnkperp_fast_smooth(
    z, ln_k_perp, kpars, k3d_smooth, p3d_fun, p3d_params={}, new_cosmo_params=None
):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp) with smoothing.

    Parameters:
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        k3d_smooth (float): Smoothing scale in units of k_perp.
        p3d_fun (function): Function that returns P3D.
        p3d_params (dict, optional): Additional parameters for the model. Defaults to {}.
        new_cosmo_params (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

    Returns:
        array-like: Computed values of P1D.
    """

    # get interval for integration
    dlnk = ln_k_perp[1] - ln_k_perp[0]

    # get function to be integrated
    # it is equivalent of the inner loop of _P1D_lnkperp
    k_perp = np.exp(ln_k_perp)
    k = np.sqrt(kpars[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
    mu = kpars[np.newaxis, :] / k
    k = k.swapaxes(0, 1)
    mu = mu.swapaxes(0, 1)

    fact = (1 / (2 * np.pi)) * k_perp[:, np.newaxis] ** 2
    fact = fact.swapaxes(0, 1)
    p3d_fix_k_par = (
        p3d_fun(z, k, mu, p3d_params, new_cosmo_params=new_cosmo_params) * fact
    )

    # perform numerical integration
    kernel = np.sinc(k3d_smooth * np.exp(ln_k_perp))
    # print(p3d_fix_k_par.shape, kernel.shape, kernel.shape)
    p1d = simpson(
        p3d_fix_k_par * kernel[np.newaxis, :] ** 2,
        ln_k_perp,
        dx=dlnk,
        axis=1,
    )

    return p1d


def p1d_from_p3d(kpar_3d, fun_p3d, z, params, vol=0, niter=1000, seed=0):
    """
    Compute P1D(kpar) from P3D(kpar,kper).

    Parameters
    ----------
    kpar : (Nkpar,) array
        Parallel wavenumbers.
    kper : (Nkper,) array
        Perpendicular wavenumbers.
    p3d : (Nkpar, Nkper) array
        3D power spectrum evaluated on the grid.

    Returns
    -------
    p1d : (Nkpar,) array
        One-dimensional power spectrum.
    """

    # P3D
    kper_3d = np.logspace(-3, 2, 100)
    kpar2d_3D, kperp2d_3D = np.meshgrid(kpar_3d, kper_3d, indexing="ij")
    p3d = fun_p3d(z, kpar2d_3D, kperp2d_3D, params)

    # P1D
    fact = (1 / (2 * np.pi)) * kperp2d_3D**2
    integrand = p3d * fact
    log_kperp2d = np.log(kperp2d_3D)
    p1d = simpson(integrand, log_kperp2d, axis=1)

    if vol != 0:
        rng = np.random.default_rng(seed)
        rea_p3d = np.zeros((niter, p3d.shape[0], p3d.shape[1]))
        rea_p1d = np.zeros((niter, p1d.shape[0]))

        sigma_3D = get_sigma(kpar2d_3D, kperp2d_3D, p3d, vol)
        for ii in range(niter):

            err3D = rng.normal(scale=sigma_3D)
            rea_p3d[ii] = p3d + err3D
            rea_p1d[ii] = simpson(rea_p3d[ii] * fact, log_kperp2d, axis=1)

    res = {}
    res["kpar"] = kpar2d_3D
    res["kper"] = kperp2d_3D
    res["p3d"] = p3d
    res["p1d"] = p1d
    res["rea_p3d"] = rea_p3d
    res["rea_p1d"] = rea_p1d

    return res


def get_sigma(kpar2d, kperp2d, p3d, vol):
    dkpar2d = np.gradient(kpar2d, axis=0)
    dkperp2d = np.gradient(kperp2d, axis=1)

    # get Gaussian covariance
    Nmodes = (vol / (2 * np.pi) ** 2) * kperp2d * dkperp2d * dkpar2d
    sigma = np.sqrt(2.0 / Nmodes) * p3d

    return sigma
