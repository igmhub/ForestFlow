import numpy as np
from scipy.integrate import simpson


def P1D_Mpc(
    z,
    k_par,
    p3d_fun,
    p3d_params={},
    cosmo_new=None,
    k_perp_min=0.001,
    k_perp_max=100,
    n_k_perp=99,
):
    """
    Returns P1D for specified values of k_par, with the option to specify values of k_perp to be integrated over.

    Parameters:
        z (float): Redshift. It modifies the linear power spectrum but not the value of the Arinyo parameters.
        k_par (array-like): Array or list of values for which P1D is to be computed.
        p3d_fun (function): Function that returns P3D. It takes as input z, k/kpar, mu/kperp, with the difference
            depending on the value of p3d_fun.coordinates. It also takes as input p3d_params and optionally cosmo_new.
        p3d_params (dict, optional): Additional parameters for the model. Defaults to {}.
        cosmo_new (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.
        k_perp_min (float, optional): Lower bound of integral. Defaults to 0.001.
        k_perp_max (float, optional): Upper bound of integral. Defaults to 100.
        n_k_perp (int, optional): Number of points in integral. Defaults to 99.

    Returns:
        array-like: Computed values of P1D.
    """

    ln_k_perp = np.linspace(np.log(k_perp_min), np.log(k_perp_max), n_k_perp)

    p1d = _P1D_lnkperp_fast(
        z, ln_k_perp, k_par, p3d_fun, p3d_params, cosmo_new=cosmo_new
    )

    return p1d


def _P1D_lnkperp_fast(z, ln_k_perp, kpars, p3d_fun, p3d_params={}, cosmo_new=None):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp) using a fast method.

    Parameters:
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        p3d_fun (function): Function that returns P3D.
        p3d_params (dict, optional): Additional parameters for the model. Defaults to {}.
        cosmo_new (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

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
        p3d_fix_k_par = p3d_fun(z, k, mu, p3d_params, cosmo_new=cosmo_new) * fact
    elif p3d_fun.coordinates == "kpar_kperp":
        kpar = k * mu
        kperp = k * np.sqrt(1 - mu**2)
        p3d_fix_k_par = p3d_fun(z, kpar, kperp, p3d_params, cosmo_new=cosmo_new) * fact
    else:
        raise ValueError(
            "p3d_fun must have coordinates attribute set to 'k_mu' or 'kpar_kperp'"
        )

    # perform numerical integration
    p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

    return p1d


def _P1D_lnkperp_fast_smooth(
    z, ln_k_perp, kpars, k3d_smooth, p3d_fun, p3d_params={}, cosmo_new=None
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
        cosmo_new (dict, optional): Optional cosmology override passed through to `P3D_Mpc`.

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
    p3d_fix_k_par = p3d_fun(z, k, mu, p3d_params, cosmo_new=cosmo_new) * fact

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
