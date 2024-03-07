import numpy as np
from scipy.integrate import simpson

def P1D_Mpc(P3D_Mpc, z, ln_k_perp, kpars, P3D_mode='pol',  **P3D_kwargs):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp) using a fast method.

    Parameters:
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        parameters (dict, optional): Additional parameters for the model. Defaults to {}.

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
    if P3D_mode == 'pol':
        p3d_fix_k_par = P3D_Mpc(z, k, mu,  **P3D_kwargs) * fact
        print(P3D_Mpc(z, k, mu, **P3D_kwargs).shape)
        print(fact.shape)
    elif P3D_mode == 'cart':
        # tile
        kperp2d  = np.tile(k_perp[:, np.newaxis], len(kpars)).T # mu grid for P3D
        kpar2d   = np.tile(kpars[:, np.newaxis], len(k_perp))
        print(P3D_Mpc(z, kpar2d, kperp2d, **P3D_kwargs).shape)
        print(fact.shape)
        p3d_fix_k_par = P3D_Mpc(z, kpar2d, kperp2d, **P3D_kwargs) * fact

    # perform numerical integration
    p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

    return p1d

def get_Px(
    kpars,
    P3D,
    z,
    P3D_mode='pol',
    **P3D_kwargs
):
    """ Calculates P_cross, the power for a given k_parallel mode from pairs of lines-of-sight separated by perpendicular distance rperp, given a 3D power spectrum
    Calculation is done with the hankl transform.
    This code is a simpler version with fewer changeable parameters. See get_Px_detailed for a version with all possible tweaks.
    Required Parameters:
        kpars (array): array of k parallel (usually log-spaced)    
        P3D (function): Function that takes arguments
        z (float): single redshift to evaluate
    Optional Parameters:
        P3D_mode: 'pol' or 'cart' for polar or cartesian. 'pol' assumes that the function takes parameters z and an array of k and mu. 'cart' assumes that the parameters are z, kpar and kperp, both arrays.
        fast (bool): if true, the transition to P1D is done faster, without interpolation, but there will be a discontinuity
        **P3D_kwargs: optional named arguments to be passed to the P3D function.
    Returns:
        rperp, Px_per_rperp, Px_per_kpar
        rperp: array of log-space r-perpendicular (separation in Mpc)
        Px_per_kpar: P-cross as an array with shape (len(kpars), len(rperp)).
    """
    import hankl
    from scipy.interpolate import CubicSpline
    nkperp = 2**16
    nkpar  = len(kpars)
    kperps = np.logspace(
        np.log10(10.0**-20), np.log10(10.0**3), nkperp
    )  # set up an array of kperp
    # tile
    kperp2d  = np.tile(kperps[:, np.newaxis], nkpar) # mu grid for P3D
    kpar2d   = np.tile(kpars[:, np.newaxis], nkperp).T
    k2d = np.sqrt(kperp2d**2 + kpar2d**2)
    mu2d = kpar2d / k2d
    if P3D_mode == 'cart':
        # assume P3D is a function of (z, kpar, kperp)
        P3D_eval = P3D(z=z, kpar=kpar2d, kperp=kperp2d, **P3D_kwargs)
    elif P3D_mode == 'pol':
        # assume P3D is a function of (z, k, mu)
        P3D_eval = P3D(z=z, k=k2d, mu=mu2d, **P3D_kwargs)
    P1D = P1D_Mpc(P3D, z, np.linspace(np.log(0.001), np.log(100), 99), kpars, P3D_mode = P3D_mode, **P3D_kwargs) # get P1D
    Px_per_kpar = []
    print(P3D_eval.shape)
    for ik, kpar in enumerate(kpars):  # for each value of k parallel to evaluate Px at
        P3D_kpar = P3D_eval[:,ik] # get the P3D
        func     = P3D_kpar * kperps
        rperp, LHS = hankl.FFTLog(
            kperps, func, q=0, mu=0
        )  # returns an array of log-spaced rperps, and the Hankel Transform
        Px = LHS / rperp / (2*np.pi)  # Divide out by remaining factor to get Px
        # transition    
        # replace the values left of the minimum
        # replace = rperp < 0.02
        # # return the P1D result for that kpar
        # Px[replace] = P1D[ik]
        # # between rperp = 0.02 and 0.08, interpolate from P1D to Px values
        # idxmin = (np.abs(rperp - 0.02)).argmin()
        # idxmax = (np.abs(rperp - 0.08)).argmin()
        # rperps_interp = rperp[idxmin:idxmax]
        # Px_tointerp = np.delete(Px, np.arange(idxmin, idxmax))
        # rperp_tointerp = np.delete(rperp, np.arange(idxmin, idxmax))
        # interpmin = np.abs(rperp_tointerp - 0.005).argmin()
        # interpmax = np.abs(rperp_tointerp - 0.2).argmin()
        # cs = CubicSpline(
        #     rperp_tointerp[interpmin:interpmax],
        #     Px_tointerp[interpmin:interpmax],
        # )
        # Px_interpd = cs(rperps_interp)
        # Px = np.insert(Px_tointerp, idxmin, Px_interpd)
        # min_rperp, max_rperp = 10**-2, 100
        # if min_rperp > min(rperp):
        #     rperp_minidx = np.argmin(abs(rperp - min_rperp))
        # if max_rperp < max(rperp):
        #     rperp_maxidx = np.argmin(abs(rperp - max_rperp))
        # else:
        #     rperp_minidx, rperp_maxidx = None, None
        # Px_per_kpar.append(Px[rperp_minidx:rperp_maxidx])
        # rperp = rperp[rperp_minidx:rperp_maxidx
        Px_per_kpar.append(Px)
        
    Px_per_kpar = np.asarray(Px_per_kpar)
    return rperp, Px_per_kpar


def get_Px_detailed(
    kpars,
    P3D,
    z,
    P3D_mode='pol',
    min_rperp=10**-2,
    max_rperp=100,
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    nkperp=2**16,
    trans_to_p1d=True,
    interpmin = 0.005,
    interpmax = 0.2,
    fast_transition=False,
    **P3D_kwargs
):
    """ Calculates P_cross, the power for a given k_parallel mode from pairs of lines-of-sight separated by perpendicular distance rperp, given a 3D power spectrum
    Calculation is done with the hankl transform.

    Required Parameters:
        kpars (array): array of k parallel (usually log-spaced)    
        arinyo (ArinyoModel instance): Arinyo model with a pre-chosen cosmology
        z (float): single redshift to evaluate
    Optional Parameters:
        min_rperp, max_rperp (float): desired range of rperp values to return
        min_kperp, max_kperp (float): range of kperp values to use in the calculation. Decreasing this range can cause unwanted artifacts
        Nsteps_kperp (int): number of kperps for the hankl transform (and number of output rperp). Decreasing this speeds up calculation but decreases accuracy
        trans_to_p1d (bool): determines whether to transition to the P1D result at low rperp
        fast_transition (bool): if true, the transition to P1D is done faster, without interpolation, but there will be a discontinuity

    Returns:
        rperp, Px_per_rperp, Px_per_kpar
        rperp: array of log-space r-perpendicular (separation in Mpc)
        Px_per_kpar: P-cross as an array with shape (len(kpars), len(rperp)).
    """
    import hankl
    from scipy.interpolate import CubicSpline


    nkpar  = len(kpars)
    kperps = np.logspace(
        np.log10(min_kperp), np.log10(max_kperp), nkperp
    )  # set up an array of kperp
    # tile
    kperp2d  = np.tile(kperps[:, np.newaxis], nkpar) # mu grid for P3D
    kpar2d   = np.tile(kpars[:, np.newaxis], nkperp).T
    k2d = np.sqrt(kperp2d**2 + kpar2d**2)
    mu2d = kpar2d / k2d
    if P3D_mode == 'cart':
        # assume P3D is a function of (z, kpar, kperp)
        P3D_eval = P3D(z=z, kpar=kpar2d, kperp=kperp2d, **P3D_kwargs)
    elif P3D_mode == 'pol':
        # assume P3D is a function of (z, k, mu)
        P3D_eval = P3D(z=z, k=k2d, mu=mu2d, **P3D_kwargs)
    P1D = P1D_Mpc(P3D, z, np.linspace(np.log(0.001), np.log(100), 99), kpars, P3D_mode = P3D_mode, **P3D_kwargs) # get P1D
    Px_per_kpar = []

    for ik, kpar in enumerate(kpars):  # for each value of k parallel to evaluate Px at
        k = k2d[:,ik]
        mu = mu2d[:,ik]
        P3D_kpar = P3D_eval[:,ik] # get the P3D
        func     = P3D_kpar * kperps
        rperp, LHS = hankl.FFTLog(
            kperps, func, q=0, mu=0
        )  # returns an array of log-spaced rperps, and the Hankel Transform
        Px = LHS / rperp / (2*np.pi)  # Divide out by remaining factor to get Px
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
        if min_rperp > min(rperp):
            rperp_minidx = np.argmin(abs(rperp - min_rperp))
        if max_rperp < max(rperp):
            rperp_maxidx = np.argmin(abs(rperp - max_rperp))
        else:
            rperp_minidx, rperp_maxidx = None, None
        Px_per_kpar.append(Px[rperp_minidx:rperp_maxidx])
        rperp = rperp[rperp_minidx:rperp_maxidx]
    Px_per_kpar = np.asarray(Px_per_kpar)
    return rperp, Px_per_kpar
