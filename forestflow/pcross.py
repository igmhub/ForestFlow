import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline


def P1D_Mpc(P3D_Mpc, z, ln_k_perp, kpars, P3D_mode="pol", P3D_kwargs={}):
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
        p3d_fix_k_par = P3D_Mpc(z, k, mu, P3D_kwargs) * fact
    elif P3D_mode == "cart":
        # tile
        kperp2d = np.tile(k_perp[:, np.newaxis], len(kpars)).T  # mu grid for P3D
        kpar2d = np.tile(kpars[:, np.newaxis], len(k_perp))
        p3d_fix_k_par = P3D_Mpc(z, kpar2d, kperp2d, P3D_kwargs) * fact

    # perform numerical integration
    p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

    return p1d


def Px_Mpc(z, kpars, rperp_Mpc, P3D_Mpc, P3D_mode="pol", P3D_params=[]):
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
        z, kpars, rperp_Mpc, P3D_Mpc, P3D_mode, P3D_params=P3D_params
    )


def Px_Mpc_detailed(
    z,
    kpar_iMpc,
    rperp_Mpc,
    P3D_Mpc,
    P3D_mode="pol",
    min_kperp=10.0**-7,
    max_kperp=10.0**3,
    nkperp=2**11,
    interpmin=0.005,
    interpmax=0.2,
    fast_transition=False,
    P3D_params={},  # list of dictionaries with keyword parameters to be passed to the P3D function
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
        min_kperp, max_kperp (float): range of kperp values to use in the calculation. Decreasing this range can cause unwanted artifacts
        nkperp (int): number of kperps for the hankl transform (and number of output rperp). Decreasing this speeds up calculation but decreases accuracy
        interpmin (float): value of r-perp to start the interpolation to P1D at
        interpmax (float): value of r-perp to end the interpolation to P1D at
        fast_transition (bool): if true, the transition to P1D is done faster, without interpolation, but there will be a discontinuity
    Returns:
        rperp: array of log-space r-perpendicular (separation in Mpc)
        Px_per_kpar: P-cross as an array with shape (len(kpar_iMpc), len(rperp)).
    """

    import hankl

    # make everything numpy arrays
    kpar_iMpc = np.atleast_1d(kpar_iMpc)
    z = np.atleast_1d(z)
    rperp_Mpc = np.atleast_1d(rperp_Mpc)
    if 0 in kpar_iMpc:
        raise ValueError("kpar list must not contain zero.")
    Nz = len(z)
    if Nz > 1 and kpar_iMpc.ndim == 1:
        kpar_iMpc = np.tile(
            kpar_iMpc, (Nz, 1)
        )  # assume kpar_iMpc is the same for all z
    if Nz > 1 and rperp_Mpc.ndim == 1:
        rperp_Mpc = np.tile(
            rperp_Mpc, (Nz, 1)
        )  # assume rperp_Mpc is the same for all z
    elif Nz == 1 and kpar_iMpc.ndim > 1:
        raise ValueError("kpar_iMpc must be a 1D array or list if z is a single value.")
    if Nz == 1:
        # now convert to 2d (first dimension is z, second is kpar)
        kpar_iMpc = np.array([kpar_iMpc])
        rperp_Mpc = np.array([rperp_Mpc])
    nkpar = kpar_iMpc.shape[1]

    # understand what is passed to P3D_params. Turn P3D_params into a list of dictionaries if it is not already
    if P3D_params:
        if isinstance(P3D_params, dict):
            if Nz == 1:
                P3D_params_byz = [P3D_params]
            elif Nz > 1:
                P3D_params_byz = []
                for iz in range(Nz):
                    P3Dsubdictz = {}
                    for key in P3D_params.keys():
                        if isinstance(P3D_params[key], list) or isinstance(
                            P3D_params[key], np.ndarray
                        ):
                            assert (
                                len(P3D_params[key]) == Nz
                            ), f"Parameter {key} must be a list of length {Nz} if z is an array."
                            P3Dsubdictz[key] = P3D_params[key]
                        else:
                            P3Dsubdictz[key] = P3D_params[key]
                            if iz == 0:
                                print(
                                    f"Warning: Number of input z values ({Nz}) does not match the number of values input for parameter {key}. Calculating model with {key} = {P3D_params[key]} for all z."
                                )
                            # assume the parameter is the same for all z, with a warning

                    P3D_params_byz.append(P3Dsubdictz)
        elif isinstance(P3D_params, list):
            # make sure each element is a dictionary
            for P3D_par in P3D_params:
                if not isinstance(P3D_par, dict):
                    raise ValueError("P3D_params must be a list of dictionaries.")
            # make sure the length of the list matches the number of z values
            assert (
                len(P3D_params) == Nz
            ), f"Number of z values ({Nz}) does not match the number of P3D_params dictionaries ({len(P3D_params)})."
            P3D_params_byz = P3D_params
    else:
        raise Warning(
            "P3D_params is empty. Assuming no parameters are needed for P3D_Mpc."
        )

    kperps = np.logspace(np.log10(min_kperp), np.log10(max_kperp), nkperp)
    Px_pertheta_perz = []
    for iz in range(Nz):
        # tile

        kperp2d = np.tile(kperps[:, np.newaxis], nkpar)  # mu grid for P3D
        kpar2d = np.tile(kpar_iMpc[iz][:, np.newaxis], nkperp).T
        k2d = np.sqrt(kperp2d**2 + kpar2d**2)
        mu2d = kpar2d / k2d

        if P3D_mode == "cart":
            # assume P3D_Mpc is a function of (z, kpar, kperp)
            P3D_eval = P3D_Mpc(z[iz], kpar2d, kperp2d, P3D_params_byz[iz])
        elif P3D_mode == "pol":
            # assume P3D_Mpc is a function of (z, k, mu)
            P3D_eval = P3D_Mpc(z[iz], k2d, mu2d, P3D_params_byz[iz])
        P1D = P1D_Mpc(
            P3D_Mpc,
            z[iz],
            np.linspace(np.log(0.001), np.log(100), 99),
            kpar_iMpc[iz],
            P3D_mode,
            P3D_params_byz[iz],
        )  # get P1D
        Px_per_kpar = []

        for ik, kpar in enumerate(
            kpar_iMpc[iz]
        ):  # for each value of k parallel to evaluate Px at
            P3D_kpar = P3D_eval[:, ik]  # get the P3D
            func = P3D_kpar * kperps
            rperp, LHS = hankl.FFTLog(
                kperps, func, q=0, mu=0
            )  # returns an array of log-spaced rperps, and the Hankel Transform
            Px = LHS / rperp / (2 * np.pi)  # Divide out by remaining factor to get Px
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
            # get an interpolator function that returns Px at the user-requested values for rperp
            Px_func = CubicSpline(rperp, Px)
            Px_per_kpar.append(Px_func(rperp_Mpc[iz]))
        Px_pertheta_perz.append(np.asarray(Px_per_kpar).T)
    if Nz > 1:
        Px_pertheta_perz = np.asarray(Px_pertheta_perz)
    else:
        Px_pertheta_perz = np.asarray(Px_pertheta_perz[0])

    return Px_pertheta_perz


def kaiser(bias1, bias2, beta_rsd1, beta_rsd2, mu):
    """Returns the Kaiser factor for multiplying linear power for a given bias, beta and mu.
    Parameters:
        bias (float): Bias factor.
        beta_rsd (float): Redshift space distortion parameter.
        mu (float): 2D array, Cosine of the angle between the line of sight and the wavevector.
    Returns:
        float: Kaiser factor.
    """
    return (
        bias1
        * bias2
        * (1 + beta_rsd1 * mu**2 + beta_rsd2 * mu**2 + beta_rsd1 * beta_rsd2 * mu**4)
    )

