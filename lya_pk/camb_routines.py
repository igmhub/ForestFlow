import numpy as np
from scipy import interpolate


def P_camb(pk_intp, z, kh, grid=None):
    if grid is None:
        grid = not np.isscalar(z) and not np.isscalar(kh)
    if pk_intp.islog:
        return pk_intp.logsign * np.exp(pk_intp(z, np.log(kh), grid=grid))
    else:
        return pk_intp(z, np.log(kh), grid=grid)


def get_matter_power_interpolator(
    camb_results,
    nonlinear=True,
    var1=None,
    var2=None,
    hubble_units=True,
    k_hunit=True,
    return_z_k=False,
    log_interp=True,
    extrap_kmax=None,
    silent=False,
):
    r"""
    Same routine as in CAMB, but for using with pre-computed camb_results

    Assuming transfers have been calculated, return a 2D spline interpolation object to evaluate matter
    power spectrum as function of z and k/h (or k). Uses self.Params.Transfer.PK_redshifts as the spline node
    points in z. If fewer than four redshift points are used the interpolator uses a reduced order spline in z
    (so results at intermediate z may be innaccurate), otherwise it uses bicubic.
    Usage example:

    .. code-block:: python

       PK = results.get_matter_power_interpolator();
       print('Power spectrum at z=0.5, k/h=0.1 is %s (Mpc/h)^3 '%(PK.P(0.5, 0.1)))

    For a description of outputs for different var1, var2 see :ref:`transfer-variables`.

    :param nonlinear: include non-linear correction from halo model
    :param var1: variable i (index, or name of variable; default delta_tot)
    :param var2: variable j (index, or name of variable; default delta_tot)
    :param hubble_units: if true, output power spectrum in :math:`({\rm Mpc}/h)^{3}` units,
                         otherwise :math:`{\rm Mpc}^{3}`
    :param k_hunit: if true, matter power is a function of k/h, if false, just k (both :math:`{\rm Mpc}^{-1}` units)
    :param return_z_k: if true, return interpolator, z, k where z, k are the grid used
    :param log_interp: if true, interpolate log of power spectrum
                       (unless any values cross zero in which case ignored)
    :param extrap_kmax: if set, use power law extrapolation beyond kmax to extrap_kmax
                        (useful for tails of integrals)
    :param silent: Set True to silence warnings
    :return: An object PK based on :class:`~scipy:scipy.interpolate.RectBivariateSpline`,
             that can be called with PK.P(z,kh) or PK(z,log(kh)) to get log matter power values.
             If return_z_k=True, instead return interpolator, z, k where z, k are the grid used.
    """

    assert camb_results.Params.WantTransfer
    khs, zs, pk = camb_results.get_linear_matter_power_spectrum(
        var1, var2, hubble_units, nonlinear=nonlinear
    )
    kh_max = khs[-1]
    if not k_hunit:
        khs *= camb_results.Params.H0 / 100
    sign = 1
    if log_interp and np.any(pk <= 0):
        if np.all(pk < 0):
            sign = -1
        else:
            log_interp = False
    p_or_log_p = np.log(sign * pk) if log_interp else pk
    logkh = np.log(khs)
    deg_z = min(len(zs) - 1, 3)
    kmax = khs[-1]
    if extrap_kmax and extrap_kmax > kmax:
        # extrapolate to ultimate power law
        # TODO: use more physical extrapolation function for linear case
        if not silent and (
            kh_max < 3 and extrap_kmax > 2 and nonlinear or kh_max < 0.4
        ):
            logging.warning(
                "Extrapolating to higher k with matter transfer functions "
                "only to k=%.3g Mpc^{-1} may be inaccurate.\n "
                % (kh_max * self.Params.H0 / 100)
            )
        if not log_interp:
            raise CAMBValueError(
                "Cannot use extrap_kmax with log_inter=False (e.g. PK crossing zero for %s, %s.)"
                % (var1, var2)
            )

        logextrap = np.log(extrap_kmax)
        log_p_new = np.empty((pk.shape[0], pk.shape[1] + 2))
        log_p_new[:, :-2] = p_or_log_p
        delta = logextrap - logkh[-1]

        dlog = (log_p_new[:, -3] - log_p_new[:, -4]) / (logkh[-1] - logkh[-2])
        log_p_new[:, -1] = log_p_new[:, -3] + dlog * delta
        log_p_new[:, -2] = log_p_new[:, -3] + dlog * delta * 0.9
        logkh = np.hstack((logkh, logextrap - delta * 0.1, logextrap))
        p_or_log_p = log_p_new

    deg_k = min(len(logkh) - 1, 3)
    res = interpolate.RectBivariateSpline(zs, logkh, p_or_log_p, kx=deg_z, ky=deg_k)
    res.kmin = np.min(khs)
    res.kmax = kmax
    res.islog = log_interp
    res.logsign = sign
    res.zmin = np.min(zs)
    res.zmax = np.max(zs)
    if return_z_k:
        return res, zs, khs
    else:
        return res
