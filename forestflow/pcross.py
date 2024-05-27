import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline


def P1D_Mpc(P3D_Mpc, z, ln_k_perp, kpars, P3D_mode="pol", **P3D_kwargs):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp) using a fast method.
    Replicates the function in model_p3d_arinyo.py, but with flexibility for the style of P3D input.
    Parameters:
        P3D (function): Function that takes arguments
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        parameters (dict, optional): Additional parameters for the model. Defaults to {}.
    Optional parameters:
        P3D_mode: 'pol' or 'cart' for polar or cartesian. 'pol' assumes that the function takes parameters z and an array of k and mu. 'cart' assumes that the parameters are z, kpar and kperp, both arrays.

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
    if P3D_mode == "pol":
        p3d_fix_k_par = P3D_Mpc(z, k, mu, **P3D_kwargs) * fact
    elif P3D_mode == "cart":
        # tile
        kperp2d = np.tile(
            k_perp[:, np.newaxis], len(kpars)
        ).T  # mu grid for P3D
        kpar2d = np.tile(kpars[:, np.newaxis], len(k_perp))
        p3d_fix_k_par = P3D_Mpc(z, kpar2d, kperp2d, **P3D_kwargs) * fact

    # perform numerical integration
    p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

    return p1d


def Px_Mpc(kpars, P3D_Mpc, z, rperp_choice=None, P3D_mode="pol", **P3D_kwargs):
    """Calls Px_Mpc_detailed to calculate P_cross, the cross-correlation of k_parallel modes from pairs of lines-of-sight separated by perpendicular distance rperp, given a 3D power spectrum
    Calculation is done with the Hankel transform.
    Required Parameters:
        kpars (array-like): array of k parallel (usually log-spaced)
        P3D (function): Function that takes arguments
        z (float): single redshift to evaluate
    Optional Parameters:
        rperp_choice: a list of rperp values [Mpc] at which to evaluate Px. The function will return rperp and Px values close, but not exactly equal to, the requested values. If not set, the function will return a finely-log-spaced grid of values from rperp=0.01 to 100.
        P3D_mode: 'pol' or 'cart' for polar or cartesian. 'pol' assumes that the function takes parameters z and an array of k and mu. 'cart' assumes that the parameters are z, kpar and kperp, both arrays.
        **P3D_kwargs: optional named arguments to be passed to the P3D function.
    Returns:
        tuple of (rperp, Px_per_kpar)
        rperp (array-like): array of r-perpendicular (float) (separation in Mpc)
        Px_per_kpar (array-like): P-cross in Mpc at each rperp (float), has shape (len(kpars), len(rperp)).
    """
    return Px_Mpc_detailed(
        kpars, P3D_Mpc, z, rperp_choice, P3D_mode, **P3D_kwargs
    )


def Px_Mpc_detailed(
    kpars,
    P3D_Mpc,
    z,
    rperp_choice=None,
    P3D_mode="pol",
    min_rperp=10**-2,
    max_rperp=100,
    min_kperp=10.0**-7,
    max_kperp=10.0**3,
    nkperp=2**11,
    trans_to_p1d=True,
    interpmin=0.005,
    interpmax=0.2,
    fast_transition=False,
    **P3D_kwargs
):
    """Calculates P_cross, the power for a given k_parallel mode from pairs of lines-of-sight separated by perpendicular distance rperp, given a 3D power spectrum
    Calculation is done with the Hankel transform.
    See Px_Mpc for a more user-friendly version.
    Required Parameters:
        kpars (array): array of k parallel (usually log-spaced)
        P3D_Mpc (function): Function that takes arguments
        z (float): single redshift to evaluate
    Optional Parameters:
        rperp_choice: a list of rperp values [Mpc] at which to evaluate Px. The function will return rperp and Px values close, but not exactly equal to, the requested values. If not set, the function will return a finely-log-spaced grid of values from rperp=0.01 to 100.
        P3D_mode: 'pol' or 'cart' for polar or cartesian. 'pol' assumes that the function takes parameters z and an array of k and mu. 'cart' assumes that the parameters are z, kpar and kperp, both arrays.
        **P3D_kwargs: optional named arguments to be passed to the P3D_Mpc function.
        min_rperp, max_rperp (float): desired range of rperp values to return
        min_kperp, max_kperp (float): range of kperp values to use in the calculation. Decreasing this range can cause unwanted artifacts
        nkperp (int): number of kperps for the hankl transform (and number of output rperp). Decreasing this speeds up calculation but decreases accuracy
        trans_to_p1d (bool): determines whether to transition to the P1D result at low rperp
        interpmin (float): value of r-perp to start the interpolation to P1D at
        interpmax (float): value of r-perp to end the interpolation to P1D at
        fast_transition (bool): if true, the transition to P1D is done faster, without interpolation, but there will be a discontinuity
    Returns:
        rperp: array of log-space r-perpendicular (separation in Mpc)
        Px_per_kpar: P-cross as an array with shape (len(kpars), len(rperp)).
    """

    import hankl
    if len(z)>1:
        raise ValueError("Only one z value can be passed.")
    if 0 in kpars:
        raise ValueError("kpar list must not contain zero.")
    nkpar = len(kpars)
    kperps = np.logspace(
        np.log10(min_kperp), np.log10(max_kperp), nkperp
    )  # set up an array of kperp
    # tile
    kperp2d = np.tile(kperps[:, np.newaxis], nkpar)  # mu grid for P3D
    kpar2d = np.tile(kpars[:, np.newaxis], nkperp).T
    k2d = np.sqrt(kperp2d**2 + kpar2d**2)
    mu2d = kpar2d / k2d
    if P3D_mode == "cart":
        # assume P3D_Mpc is a function of (z, kpar, kperp)
        P3D_eval = P3D_Mpc(z=z, kpar=kpar2d, kperp=kperp2d, **P3D_kwargs)
    elif P3D_mode == "pol":
        # assume P3D_Mpc is a function of (z, k, mu)
        P3D_eval = P3D_Mpc(z=z, k=k2d, mu=mu2d, **P3D_kwargs)
    P1D = P1D_Mpc(
        P3D_Mpc,
        z,
        np.linspace(np.log(0.001), np.log(100), 99),
        kpars,
        P3D_mode=P3D_mode,
        **P3D_kwargs
    )  # get P1D
    Px_per_kpar = []

    for ik, kpar in enumerate(
        kpars
    ):  # for each value of k parallel to evaluate Px at
        P3D_kpar = P3D_eval[:, ik]  # get the P3D
        func = P3D_kpar * kperps
        rperp, LHS = hankl.FFTLog(
            kperps, func, q=0, mu=0
        )  # returns an array of log-spaced rperps, and the Hankel Transform
        Px = (
            LHS / rperp / (2 * np.pi)
        )  # Divide out by remaining factor to get Px
        # transition
        if fast_transition:
            replace = rperp < 0.04
            Px[replace] = P1D[ik]
        else:
            # replace the values left of the minimum
            replace = rperp < (0.02)
            # return the P1D result for that kpar
            Px[replace] = P1D[ik]
            # between rperp = 0.02 and 0.08, interpolate from P1D to Px values
            idxmin = (np.abs(rperp - 0.02)).argmin()
            idxmax = (np.abs(rperp - 0.08)).argmin()
            rperps_interp = rperp[idxmin:idxmax]
            Px_tointerp = np.delete(Px, np.arange(idxmin, idxmax))
            rperp_tointerp = np.delete(rperp, np.arange(idxmin, idxmax))
            interpmin_id = np.abs(rperp_tointerp - interpmin).argmin()
            interpmax_id = np.abs(rperp_tointerp - interpmax).argmin()
            cs = CubicSpline(
                rperp_tointerp[interpmin_id:interpmax_id],
                Px_tointerp[interpmin_id:interpmax_id],
            )
            Px_interpd = cs(rperps_interp)
            Px = np.insert(Px_tointerp, idxmin, Px_interpd)
        if rperp_choice is None:
            if ik == 0:
                # enforce rperp limits, only need to do once (same for every kpar)
                if min_rperp > min(rperp):
                    rperp_minidx = np.argmin(abs(rperp - min_rperp))
                else:
                    rperp_minidx = None
                if max_rperp < max(rperp):
                    rperp_maxidx = np.argmin(abs(rperp - max_rperp))
                else:
                    rperp_maxidx = None
                rperp_save = rperp[rperp_minidx:rperp_maxidx]
            # save Px for that range
            Px_per_kpar.append(Px[rperp_minidx:rperp_maxidx])
        else:
            # get an interpolator function that returns Px at the user-requested values for rperp
            Px_func = CubicSpline(rperp, Px)
            Px_per_kpar.append(Px_func(rperp_choice))
            if ik == 0: # only need to rename once
                rperp_save = rperp_choice
    Px_per_kpar = np.asarray(Px_per_kpar)
    return rperp_save, Px_per_kpar
