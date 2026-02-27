import numpy as np
from scipy.interpolate import CubicSpline
from forestflow.p1d import P1D_Mpc 

def Px_Mpc(z, kpars, rperp_Mpc, p3d_fun_Mpc, p3d_params={}, max_k_for_p3d=200, **kwargs):
    """
    Compute the cross-power spectrum P_cross(k_parallel, r_perp) for a given 3D
    power spectrum model, as a function of parallel wavenumber and transverse
    separation.

    This is the user-friendly interface to `Px_Mpc_detailed`. It computes the
    cross-correlation of line-of-sight modes separated by r_perp using a Hankel
    transform of the 3D power spectrum.

    Parameters
    ----------
    z : float or array-like of shape (Nz,)
        Redshift(s) at which to evaluate the cross-power spectrum.
    kpars : array-like
        Parallel wavenumbers k_parallel in units of Mpc⁻¹.
        - Shape can be (Nk,) for a single redshift or (Nz, Nk) for multiple redshifts.
    rperp_Mpc : array-like
        Transverse separations r_perp (in Mpc) at which to evaluate the cross-power spectrum.
        - Shape can be (Nr,) for a single redshift or (Nz, Nr) for multiple redshifts.
    p3d_fun_mpc : callable
        Function returning the 3D power spectrum in Mpc units.
        Must have an attribute `coordinates` that is either "kpar_kperp" or "k_mu", which determines how the function is called.
    p3d_params : dict or list of dicts, optional
        Extra keyword parameters for `p3d_Mpc`.
        - Dict is broadcast to all z values (with a warning if multi-z).
        - List of dicts must have length Nz.

    Returns
    -------
    Px_per_kpar : ndarray
        Cross-power spectrum P_cross in Mpc units evaluated at each input r_perp and k_parallel.
        - Shape is (Nz, Nr, Nk) for multi-z input, or (Nr, Nk) for single z.

    Notes
    -----
    - This function is a thin wrapper around `Px_Mpc_detailed`, which provides
      additional control over the k_perp grid, interpolation, and P1D transition.
    - The Hankel transform is performed internally using `hankl.FFTLog`.
    """

    return Px_Mpc_detailed(
        z, kpars, rperp_Mpc, p3d_fun_Mpc=p3d_fun_Mpc, p3d_params=p3d_params, max_k_for_p3d=max_k_for_p3d, **kwargs
    )


def Px_Mpc_detailed(
    z,
    kpar_iMpc,
    rperp_Mpc,
    p3d_fun_Mpc,
    min_kperp=10.0**-7,
    max_kperp=10.0**3,
    nkperp=2**11,
    interpmin=0.005,
    interpmax=0.2,
    p3d_params={},  # list of dictionaries with keyword parameters to be passed to the P3D function
    max_k_for_p3d=200, # maximum kperp that will be calculated for the P3D, which will otherwise be zero
    **kwargs # extra keyword arguments for the P3D function that aren't parameter values
):
    """
    Compute the cross-power spectrum P_cross(k_parallel, r_perp) using a Hankel transform
    for pairs of lines-of-sight separated by transverse distance r_perp, given a 3D power
    spectrum model p3d_Mpc.

    This is the low-level implementation called by `Px_Mpc`. It allows explicit control
    over the k_perp grid, interpolation, and the transition to the 1D power spectrum at
    very small r_perp.

    Parameters
    ----------
    z : float or array-like of shape (Nz,)
        Redshift(s) at which to evaluate the cross-power spectrum.
    kpar_iMpc : array-like
        Parallel wavenumbers k_parallel in units of Mpc⁻¹.
        - Shape can be (Nk,) for a single redshift or (Nz, Nk) for multiple redshifts.
    rperp_Mpc : array-like
        Perpendicular separations r_perp (in Mpc) at which to evaluate the cross-power spectrum.
        - Shape can be (Nr,) for a single redshift or (Nz, Nr) for multiple redshifts.
    p3d_fun_Mpc : callable
        Function returning the 3D power spectrum in Mpc units.
        p3d_fun.coordinates should equal "kpar_kperp" or "k_mu", which determines how the function is called.
    min_kperp, max_kperp : float, optional
        Minimum and maximum k_perp (Mpc⁻¹) used for the Hankel transform. Default: 1e-7, 1e3.
    nkperp : int, optional
        Number of k_perp points for the Hankel transform. Controls the output r_perp sampling.
        Default is 2**11 (~2048).
    interpmin, interpmax : float, optional
        r_perp range (in Mpc) over which to smoothly interpolate between the 3D cross-power
        and the 1D power spectrum to avoid divergences. Default: 0.005–0.2 Mpc.
    p3d_params : dict or list of dicts, optional
        Extra keyword parameters for p3d_Mpc.
        - Dict is broadcast to all z values (with a warning if multi-z).
        - List of dicts must have length Nz.

    Returns
    -------
    Px_pertheta_perz : ndarray
        Cross-power spectrum P_cross in Mpc units evaluated at each input r_perp and k_parallel.
        - Shape is (Nz, Nr, Nk) for multi-z input, [1, Nr, Nk] for the case where a single z is 
        input within a list or array (e.g., [my_z]), or (Nr, Nk) for single z input as float.
    """
    import hankl

    # make everything numpy arrays
    kpar_iMpc = np.atleast_1d(kpar_iMpc)
    z_input_type = type(z)
    z = np.atleast_1d(z)
    rperp_Mpc = np.atleast_1d(rperp_Mpc)
    Nz = len(z)
    if Nz > 1 and kpar_iMpc.ndim == 1:
        kpar_iMpc = np.tile(
            kpar_iMpc, (Nz, 1)
        )  # assume kpar_iMpc is the same for all z
    if Nz > 1 and rperp_Mpc.ndim == 1:
        rperp_Mpc = np.tile(
            rperp_Mpc, (Nz, 1)
        )  # assume rperp_Mpc is the same for all z

    if Nz == 1 and kpar_iMpc.ndim == 1:
        # convert to 2d (first dimension is z, second is kpar)
        kpar_iMpc = np.array([kpar_iMpc])
        rperp_Mpc = np.array([rperp_Mpc])
    # ensure that all arrays now have the same first axis
    assert len(z) == kpar_iMpc.shape[0], (
        f"Number of redshifts ({len(z)}) does not match number of kpar values ({kpar_iMpc.shape[0]})."
    )
    assert len(z) == rperp_Mpc.shape[0], (
        f"Number of redshifts ({len(z)}) does not match number of rperp values ({rperp_Mpc.shape[0]})."
    )
    assert np.all(kpar_iMpc < max_k_for_p3d), (
        f"Maximum k_parallel ({kpar_iMpc.max()}) exceeds maximum k for P3D ({max_k_for_p3d}). Please reduce the k_parallel range or increase max_k_for_p3d."
    )

    nkpar = kpar_iMpc.shape[1]
    
    # understand form of p3d_fun_Mpc
    if not hasattr(p3d_fun_Mpc, "coordinates"):
        raise ValueError("p3d_fun_Mpc must have a 'coordinates' attribute.")
    elif p3d_fun_Mpc.coordinates not in ["kpar_kperp", "k_mu"]:
        raise ValueError(
            "p3d_fun_Mpc.coordinates must be 'kpar_kperp' or 'k_mu'."
        )
    elif p3d_fun_Mpc.coordinates == "kpar_kperp":
        p3d_mode = "cart"
    elif p3d_fun_Mpc.coordinates == "k_mu":
        p3d_mode = "pol"

    # understand what is passed to p3d_params. Turn P3D_params into a list of dictionaries if it is not already
    if p3d_params:
        if isinstance(p3d_params, dict):
            if Nz == 1:
                p3d_params_byz = [p3d_params]
            elif Nz > 1:
                p3d_params_byz = []
                for iz in range(Nz):
                    p3dsubdictz = {}
                    for key in p3d_params.keys():
                        if isinstance(p3d_params[key], list) or isinstance(
                            p3d_params[key], np.ndarray
                        ):
                            assert len(p3d_params[key]) == Nz, (
                                f"Parameter {key} must be a list of length {Nz} if z is an array."
                            )
                            p3dsubdictz[key] = p3d_params[key][iz]
                        else:
                            p3dsubdictz[key] = p3d_params[key]
                            if iz == 0:
                                print(
                                    f"Warning: Number of input z values ({Nz}) does not match the number of values input for parameter {key}. Calculating model with {key} = {p3d_params[key]} for all z."
                                )
                            # assume the parameter is the same for all z, with a warning
                    p3d_params_byz.append(p3dsubdictz)
        elif isinstance(p3d_params, list):
            # make sure each element is a dictionary
            for p3d_par in p3d_params:
                if not isinstance(p3d_par, dict):
                    raise ValueError("p3d_params must be a list of dictionaries.")
            # make sure the length of the list matches the number of z values
            assert len(p3d_params) == Nz, (
                f"Number of z values ({Nz}) does not match the number of p3d_params dictionaries ({len(p3d_params)})."
            )
            p3d_params_byz = p3d_params
    else:
        raise Warning(
            "p3d_params is empty. Assuming no parameters are needed for p3d_Mpc."
        )
    kperps = np.logspace(np.log10(min_kperp), np.log10(max_kperp), nkperp)
    Px_pertheta_perz = []

    for iz in range(Nz):
        # tile

        kperp2d = np.tile(kperps[:, np.newaxis], nkpar)  # mu grid for p3d
        kpar2d = np.tile(kpar_iMpc[iz][:, np.newaxis], nkperp).T
        k2d = np.sqrt(kperp2d**2 + kpar2d**2)
        mu2d = kpar2d / k2d
        # limit the k values that will be calculated for the p3d, which will otherwise be zero.
        # this is because going to very high k require high k calls for CAMB, which is slow
        within_range = k2d < max_k_for_p3d
        p3d_eval = np.zeros_like(k2d)
        if p3d_mode == "cart":
            # assume p3d_Mpc is a function of (z, kpar, kperp)
            p3d_eval[within_range] = p3d_fun_Mpc(z[iz], kpar2d[within_range], kperp2d[within_range], p3d_params_byz[iz], **kwargs)
        elif p3d_mode == "pol":
            # assume p3d_Mpc is a function of (z, k, mu)
            p3d_eval[within_range] = p3d_fun_Mpc(z[iz], k2d[within_range], mu2d[within_range], p3d_params_byz[iz], **kwargs)

        P1D = P1D_Mpc(
            z=z[iz],
            k_par=kpar_iMpc[iz],
            p3d_fun=p3d_fun_Mpc,
            p3d_params=p3d_params_byz[iz],
        )  # get P1D

        Px_matrix = [] 
        
        for ik, kpar in enumerate(
            kpar_iMpc[iz]
        ):  # for each value of k parallel to evaluate Px at
            p3d_kpar = p3d_eval[:, ik]  # get the p3d
            func = p3d_kpar * kperps
            rperp, LHS = hankl.FFTLog(
                kperps, func, q=0, mu=0
            )  # returns an array of log-spaced rperps, and the Hankel Transform
            Px = LHS / rperp / (2 * np.pi)  # Divide out by remaining factor to get Px
            Px_matrix.append(Px)
        Px_matrix = np.asarray(Px_matrix)  # shape (Nkpar, len(rperp))
        
        # transition to P1D
        # first, simply replace the values left of the minimum with P1D
        replace = rperp < 0.02
        Px_matrix[:, replace] = P1D[:, np.newaxis]
        # between rperp = 0.02 and 0.08, interpolate from P1D to Px values
        # tests show that we should definitely cut out the results from fftlog in this range, as they oscillate. 
        # To spline interpolate, we need to use the surrounding region down to interpmin and up to interpmax, so we fill in the gap with some knowledge from the outskirts
        idxmin = (np.abs(rperp - 0.02)).argmin()
        idxmax = (np.abs(rperp - 0.08)).argmin()
        rperps_interp = rperp[idxmin:idxmax]
        Px_tointerp = np.delete(Px_matrix, np.arange(idxmin, idxmax), axis=1)
        rperp_tointerp = np.delete(rperp, np.arange(idxmin, idxmax))
        interpmin_id = np.abs(rperp_tointerp - interpmin).argmin()
        interpmax_id = np.abs(rperp_tointerp - interpmax).argmin()
        cs = CubicSpline(
            rperp_tointerp[interpmin_id:interpmax_id],
            Px_tointerp[:, interpmin_id:interpmax_id],
            axis=1
        )

        Px_matrix_interpd = cs(rperps_interp)
        Px_matrix = np.concatenate([
            Px_tointerp[:, :idxmin],
            Px_matrix_interpd,
            Px_tointerp[:, idxmin:]
        ], axis=1)

        # get an interpolator function that returns Px at the user-requested values for rperp
        Px_func = CubicSpline(rperp, Px_matrix, axis=1)
        Px_pertheta_perz.append(Px_func(rperp_Mpc[iz]).T)  # shape (Nr, Nkpar) for this z
    Px_pertheta_perz = np.asarray(Px_pertheta_perz)
    # return the cross-power spectrum in the same shape as z was input.
    # if 1 z was input as a float, return a 2D array (Nr, Nk)
    # if 1 or more z was input as an array, return a 3D array (Nz, Nr, Nk)

    if Nz == 1:
        # check if input is a single number
        if z_input_type in [float, int, np.float64, np.int64, np.float32, np.int32]:
            # return a 2D array (Nr, Nk)
            Px_pertheta_perz = Px_pertheta_perz.squeeze()
    return Px_pertheta_perz