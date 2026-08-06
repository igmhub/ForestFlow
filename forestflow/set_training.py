"""
Prepare and normalize ForestFlow training data.
"""

import numpy as np

from forestflow.model_p3d_arinyo import ArinyoModel
from lace.cosmo import cosmology

from forestflow.play_with_power import get_fisher


def get_training_data(list_sims, zmin=0, zmax=10, drop_sim=None, type_fit="Arinyo_min"):

    """
    Return training data.

    Parameters
    ----------
    list_sims : object
        List sims used by the calculation.
    zmin : int, optional
        Zmin used by the calculation.
    zmax : int, optional
        Zmax used by the calculation.
    drop_sim : object, optional
        Drop sim used by the calculation.
    type_fit : str, optional
        Type fit used by the calculation.

    Returns
    -------
    object
        Result produced when the function is used to return training data.
    """
    input_params = ["Delta2_p", "n_p", "mF", "sigT_Mpc", "gamma", "kF_Mpc"]
    other_params = ["z", "As", "ns"]
    output_params = ["bias", "bias_eta", "q1", "kvav", "av", "bv", "kp", "q2"]

    nn_train = 0
    for ii in range(len(list_sims)):
        if (drop_sim is not None) and (list_sims[ii]["sim_label"] == drop_sim):
            continue
        if (list_sims[ii]["z"] >= zmin) and (list_sims[ii]["z"] <= zmax):
            nn_train += 1

    # cosmo + IGM
    all_input_par = {}
    for par in input_params:
        all_input_par[par] = np.zeros((nn_train))

    # z, As, and ns
    all_other_par = {}
    for par in other_params:
        all_other_par[par] = np.zeros((nn_train))

    # Arinyo
    all_output_par = {}
    for par in output_params:
        all_output_par[par] = np.zeros((nn_train))

    ii = 0
    for sim in list_sims:
        if (drop_sim is not None) and (sim["sim_label"] == drop_sim):
            continue
        if (sim["z"] < zmin) or (sim["z"] > zmax):
            continue
        for par in input_params:
            all_input_par[par][ii] = sim[par]
        for par in other_params:
            if par != "z":
                all_other_par[par][ii] = sim["cosmo_params"][par]
            else:
                all_other_par[par][ii] = sim[par]
        for par in output_params:
            all_output_par[par][ii] = sim[type_fit][par]
        ii += 1

    data = {
        "input_par": all_input_par,
        "other_par": all_other_par,
        "output_par": all_output_par,
    }

    return data


class Transf_data(object):
    """
    Class transf data
    """

    def __init__(
        self,
        dict_all_params=None,
        sim_model=None,
        preload_file=None,
        save_file=None,
        compute_fisher=False,
        type_fit="Arinyo_min",
    ):
        """
        1. Set standarize for the input and output parameters (self.stand_input and self.stand_output)
        2. Get Fisher matrix for the transf + stand output parameters
        3. Set whitening for the output parameters (self.white_output)
        4. Set global norm for the output parameters (self.alpha_output)

        Parameters
        ----------
        dict_all_params : object, optional
            Dict all params used by the calculation.
        sim_model : object, optional
            Sim model used by the calculation.
        preload_file : object, optional
            Preload file used by the calculation.
        save_file : object, optional
            Save file used by the calculation.
        compute_fisher : bool, optional
            Compute fisher used by the calculation.
        type_fit : str, optional
            Type fit used by the calculation.
        """

        if preload_file is None:
            if dict_all_params is None:
                raise ValueError(
                    "If preload_file is not None, dict_all_params must be provided."
                )
            if save_file is None:
                print("Not saving data, save_file is None")

            self.set_standarize(dict_all_params["input_par"], type_stand="input")
            self.set_standarize(dict_all_params["output_par"], type_stand="output")

            if compute_fisher:
                if sim_model is None:
                    raise ValueError(
                        "If compute_fisher is True, sim_model must be provided."
                    )
                # for output data, whitening and global norm
                pars_model = {}
                pars_model["z"] = sim_model["z"]
                pars_model["Arinyo"] = {}
                for par in dict_all_params["output_par"]:
                    pars_model["Arinyo"][par] = sim_model[type_fit][par]

                # set Arinyo model
                cosmo_params_dict = {}
                for par in sim_model["cosmo_params"]:
                    if par != "omk":
                        cosmo_params_dict[par] = sim_model["cosmo_params"][par]
                    else:
                        cosmo_params_dict[par] = 0.0

                fid_cosmo = cosmology.Cosmology(cosmo_params_dict=cosmo_params_dict)
                model_Arinyo = ArinyoModel(fid_cosmo)

                fisher_output = get_fisher(self, pars_model, model_Arinyo)

                self.set_whitening(fisher_output, type_stand="output")
                tfw_params = self.transf_stand_white(
                    dict_all_params["output_par"], direct=True, type_stand="output"
                )
                self.set_global_norm(tfw_params, type_stand="output")
            else:
                self.white_output = None
                self.alpha_output = None

            if save_file is not None:
                np.save(
                    save_file,
                    {
                        "stand_input": self.stand_input,
                        "stand_output": self.stand_output,
                        "white_output": self.white_output,
                        "alpha_output": self.alpha_output,
                    },
                )
        else:
            if (
                (dict_all_params is not None)
                or (sim_model is not None)
                or (save_file is not None)
            ):
                print(
                    "If preload_file is None, dict_all_params, sim_model, and save_file are ignored."
                )

            preload_data = np.load(preload_file, allow_pickle=True).item()

            self.stand_input = preload_data["stand_input"]
            self.stand_output = preload_data["stand_output"]
            self.white_output = preload_data["white_output"]
            self.alpha_output = preload_data["alpha_output"]

        return

    def set_standarize(self, dict_params, type_stand="input"):

        """
        Set standarize.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.
        """
        t_params = self.transform(dict_params, direct=True)

        standarize = {}
        standarize["mean"] = {}
        standarize["std"] = {}
        for par in t_params:
            standarize["mean"][par] = np.mean(t_params[par])
            standarize["std"][par] = np.std(t_params[par])

        if type_stand == "input":
            self.stand_input = standarize
        elif type_stand == "output":
            self.stand_output = standarize

        return

    def set_whitening(self, fisher, type_stand="output"):

        """
        Set whitening.

        Parameters
        ----------
        fisher : object
            Fisher used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.
        """
        npars = len(fisher)
        # the fisher matrix is computed in the transf_stand space
        arr_fisher = np.zeros((npars, npars))
        for ii, key1 in enumerate(fisher):
            for jj, key2 in enumerate(fisher):
                arr_fisher[ii, jj] = fisher[key1][key2]

        weights = np.linalg.cholesky(arr_fisher).T

        if type_stand == "output":
            self.white_output = weights
        else:
            self.white_input = weights

        return

    def set_global_norm(self, dict_params, type_stand="output"):

        """
        Set global norm.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.
        """
        npars = len(dict_params)
        key = list(dict_params.keys())[0]
        nelem = dict_params[key].shape[0]
        arr_dict_params = np.zeros((nelem, npars))
        for ii, par in enumerate(dict_params):
            arr_dict_params[:, ii] = dict_params[par]

        alpha = np.percentile(np.abs(arr_dict_params), 95)

        if type_stand == "output":
            self.alpha_output = alpha
        else:
            self.alpha_input = alpha

    def transform(self, dict_params, direct=True):
        """
        The input params are expected to be the original ones

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to the input params are expected to be the original ones.
        """

        t_params = {}
        for par in dict_params:
            if par in ["bias"]:
                if direct:
                    t_params[par] = np.log(-dict_params[par])
                else:
                    t_params[par] = -np.exp(dict_params[par])
            elif par in ["kvav", "Delta2_p"]:
                if direct:
                    t_params[par] = np.log(dict_params[par])
                else:
                    t_params[par] = np.exp(dict_params[par])
            # elif par in ["q1"]:
            #     if direct:
            #         t_params[par] = np.log(dict_params["q1"] + dict_params["q2"])
            #     else:
            #         q1pq2 = np.exp(dict_params["q1"])
            #         q1mq2 = dict_params["q2"]
            #         t_params[par] = 0.5 * (q1pq2 + q1mq2)
            # elif par in ["q2"]:
            #     if direct:
            #         t_params[par] = dict_params["q1"] - dict_params["q2"]
            #     else:
            #         q1pq2 = np.exp(dict_params["q1"])
            #         q1mq2 = dict_params["q2"]
            #         t_params[par] = 0.5 * (q1pq2 - q1mq2)
            else:
                t_params[par] = dict_params[par]

        return t_params

    def standarize(self, dict_params, direct=True, type_stand="input"):

        """
        Standardize the requested values.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to standardize the requested values.
        """
        if type_stand == "input":
            stand = self.stand_input
        elif type_stand == "output":
            stand = self.stand_output

        s_params = {}
        for par in dict_params:
            if direct:
                s_params[par] = (dict_params[par] - stand["mean"][par]) / stand["std"][
                    par
                ]
            else:
                s_params[par] = (
                    dict_params[par] * stand["std"][par] + stand["mean"][par]
                )

        return s_params

    def whitening(self, dict_params, direct=True, type_stand="output"):

        """
        Whiten the requested values.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to whiten the requested values.
        """
        if type_stand == "output":
            weight = self.white_output
        else:
            weight = self.white_input

        npars = len(dict_params)
        key = list(dict_params.keys())[0]
        nelem = dict_params[key].shape[0]
        arr_dict_params = np.zeros((nelem, npars))
        for ii, par in enumerate(dict_params):
            arr_dict_params[:, ii] = dict_params[par]

        if direct:
            arr_w_params = arr_dict_params @ weight
        else:
            # both methods work, second suggested by chati
            # arr_w_params = arr_dict_params @ np.linalg.inv(weight)
            arr_w_params = np.linalg.solve(weight.T, arr_dict_params.T).T

        w_params = {}
        for ii, par in enumerate(dict_params):
            w_params[par] = arr_w_params[:, ii]

        return w_params

    def global_norm(self, dict_params, direct=True, type_stand="output"):

        """
        Normalize norm.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to normalize norm.
        """
        if type_stand == "output":
            alpha = self.alpha_output
        else:
            alpha = self.alpha_input

        n_params = {}
        for par in dict_params:
            if direct:
                n_params[par] = dict_params[par] / alpha
            else:
                n_params[par] = dict_params[par] * alpha

        return n_params

    def transf_stand(self, dict_params, direct=True, type_stand="input"):

        """
        Transform stand.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to transform stand.
        """
        if direct:
            dir_t_params = self.transform(dict_params, direct=True)
            dir_ts_params = self.standarize(
                dir_t_params, direct=True, type_stand=type_stand
            )
            out_params = dir_ts_params
        else:
            inv_st_params = dict_params  # expected
            inv_t_params = self.standarize(
                inv_st_params, direct=False, type_stand=type_stand
            )
            inv_params = self.transform(inv_t_params, direct=False)
            out_params = inv_params  # original

        return out_params

    def transf_stand_white(self, dict_params, direct=True, type_stand="output"):

        """
        Transform stand white.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to transform stand white.
        """
        if direct:
            dir_ts_params = self.transf_stand(
                dict_params, direct=True, type_stand=type_stand
            )
            dir_tsw_params = self.whitening(
                dir_ts_params, direct=True, type_stand=type_stand
            )
            out_params = dir_tsw_params
        else:
            inv_wst_params = dict_params  # expected
            inv_st_params = self.whitening(
                inv_wst_params, direct=False, type_stand=type_stand
            )
            inv_params = self.transf_stand(
                inv_st_params, direct=False, type_stand=type_stand
            )
            out_params = inv_params  # original

        return out_params

    def transf_stand_white_norm(self, dict_params, direct=True, type_stand="output"):

        """
        Transform stand white norm.

        Parameters
        ----------
        dict_params : dict
            Dict params used by the calculation.
        direct : bool, optional
            Direct used by the calculation.
        type_stand : str, optional
            Type stand used by the calculation.

        Returns
        -------
        object
            Result produced when the function is used to transform stand white norm.
        """
        if direct:
            dir_tsw_params = self.transf_stand_white(
                dict_params, direct=True, type_stand=type_stand
            )
            dir_tswn_params = self.global_norm(
                dir_tsw_params, direct=True, type_stand=type_stand
            )
            out_params = dir_tswn_params
        else:
            inv_wstn_params = dict_params  # expected
            inv_wst_params = self.global_norm(
                inv_wstn_params, direct=False, type_stand=type_stand
            )
            inv_params = self.transf_stand_white(
                inv_wst_params, direct=False, type_stand=type_stand
            )
            out_params = inv_params  # original

        return out_params
