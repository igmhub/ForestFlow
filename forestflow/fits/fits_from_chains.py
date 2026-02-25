import numpy as np


def get_params_4_emu(data, n_std=1):
    means = np.mean(data, axis=0)
    cov = np.cov(data.T)
    stds = np.sqrt(np.diag(cov))
    corrs = []
    for ii in range(data.shape[1]):
        for jj in range(ii):
            corrs.append(cov[ii, jj] / np.sqrt(cov[ii, ii] * cov[jj, jj]))
    corrs = np.array(corrs)
    return means, stds, corrs


def get_input_emulator(folder_input, ntot, file_out):
    # best nmax points to compute best-fitting params
    nmax = 200

    for ii in range(ntot):
        file = (
            folder_input
            + "fit_indsim_"
            + str(ii)
            + "_kmax3d_5_noise3d_0.075_kmax1d_5_noise1d_0.01.npz"
        )
        fil = np.load(file)
        par_chain = fil["chain"].copy()
        if ii == 0:
            bests = np.zeros((ntot, par_chain.shape[1]))
            means = np.zeros((ntot, par_chain.shape[1]))
            stds = np.zeros((ntot, par_chain.shape[1]))
            ncorrs = int(
                par_chain.shape[1] * (par_chain.shape[1] + 1) / 2
                - par_chain.shape[1]
            )
            corrs = np.zeros((ntot, ncorrs))

        # we fit b**2
        par_chain[:, 0] = np.abs(par_chain[:, 0])

        # best params
        _, ind = np.unique(fil["lnprob"], return_index=True)
        ind_sort = np.argsort(fil["lnprob"][ind])[::-1]
        ind_keep = ind[ind_sort[:nmax]]
        bests[ii, :] = np.mean(par_chain[ind_keep], axis=0)

        # means, stds, corrs: everything needed for the emulator
        means[ii, :], stds[ii, :], corrs[ii, :] = get_params_4_emu(par_chain)
    np.savez(file_out, bests=bests, means=means, stds=stds, corrs=corrs)
