import numpy as np


def get_Px(
    kpars,
    arinyo,
    z,
    fast=False,
    min_rperp=0.01,
    max_rperp=30,
    min_kperp=10.0**-20,
    max_kperp=10.0**3,
    Nsteps_kperp=5000,
    trans_to_p1d=True,
    fast_transition=False,
    params=None,
):
    """ Calculates P_cross, the power for a given k_parallel mode from pairs of lines-of-sight separated by perpendicular distance rperp, given a 3D power spectrum
    Calculation is done with the hankl transform.

    Required Parameters:
        kpars (array): array of k parallel (usually log-spaced)    
        arinyo (ArinyoModel instance): Arinyo model with a pre-chosen cosmology
        params (dictionary): parameters for the Arinyo parameters. If not given, they will be set to default values
        z (float): single redshift to evaluate
    Optional Parameters:
        fast (bool): if true, accuracy is <0.1% and the runtime is ~0.8s for 100 kpar values and all other default settings. If false, accuracy is <1% and the runtime is ~0.25s
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

    if params is None:
        params = arinyo.default_params
    if fast:
        Nsteps_kperp = 1000
    if min_rperp > 0.08:
        trans_to_p1d = False  # not necessary to transition to the P1D result if minimum requested rperp is larger than 0.08

    if min_rperp > 0.2 and fast:
        Nsteps_kperp = 500  # speed it up because we will cut out the range of low-rperp oscillations

    Px_per_kpar = []
    for kpar in kpars:  # for each value of k parallel to evaluate Px at
        kperps = np.logspace(
            np.log10(min_kperp), np.log10(max_kperp), Nsteps_kperp
        )  # set up an array of kperp
        kpars_prime = np.full(
            len(kperps), kpar
        )  # each kperp gets the same kpar for this iteration -- make a full array of the kpar value
        k = np.sqrt(kpars_prime**2 + kperps**2)  # get the corresponding k array
        mu = kpars_prime / k  # array of corresponding mu
        P3D = arinyo.P3D_Mpc(z, k, mu, params)  # get the P3D
        func = P3D * 0.5 * np.sqrt(kperps / (2 * np.pi))
        rperp, LHS = hankl.FFTLog(
            kperps, func, q=0, mu=0.5
        )  # returns an array of log-spaced rperps, and the Hankel Transform
        Px = LHS / (rperp ** (3 / 2))  # Divide out by remaining factor to get Px
        # transition
        if trans_to_p1d:
            p1d = arinyo.P1D_Mpc(z, np.array([kpar]), parameters=params)
            if fast_transition:
                replace = rperp < 0.04
                Px[replace] = p1d
            else:
                # replace the values left of the minimum
                replace = rperp < 0.02
                # return the P1D result for that kpar
                Px[replace] = p1d
                # between rperp = 0.02 and 0.08, interpolate from P1D to Px values
                idxmin = (np.abs(rperp - 0.02)).argmin()
                idxmax = (np.abs(rperp - 0.08)).argmin()
                rperps_interp = rperp[idxmin:idxmax]
                Px_tointerp = np.delete(Px, np.arange(idxmin, idxmax))
                rperp_tointerp = np.delete(rperp, np.arange(idxmin, idxmax))
                interpmin = np.abs(rperp_tointerp - 0.005).argmin()
                interpmax = np.abs(rperp_tointerp - 0.2).argmin()
                cs = CubicSpline(
                    rperp_tointerp[interpmin:interpmax],
                    Px_tointerp[interpmin:interpmax],
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
