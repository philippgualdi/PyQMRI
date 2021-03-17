#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module holds the classes for IPiano Optimization without streaming.

Attribues:
  DTYPE (complex64):
    Complex working precission. Currently single precission only.
  DTYPE_real (float32):
    Real working precission. Currently single precission only.
"""
from __future__ import division
import time
import numpy as np

from pkg_resources import resource_filename
import pyopencl.array as clarray
import h5py

import pyqmri.operator as operator
import pyqmri.solver as optimizer
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _utils as utils


class IPianoOptimizer:
    def __init__(
        self,
        par,
        model,
        trafo=1,
        imagespace=False,
        SMS=0,
        reg_type="LOG",
        config="",
        streamed=False,
        DTYPE=np.complex64,
        DTYPE_real=np.float32,
    ):
        self.par = par
        self.gn_res = []
        self.ipiano_par = utils.read_config(config, reg_type)
        utils.save_config(self.ipiano_par, par["outdir"], reg_type)
        num_dev = len(par["num_dev"])
        self._fval_old = 0
        self._fval = 0
        self._fval_init = 0
        self._ctx = par["ctx"]
        self._queue = par["queue"]
        self._model = model
        self._reg_type = reg_type
        self._prg = []

        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real

        self._streamed = streamed
        self._imagespace = imagespace
        self._SMS = SMS
        if streamed and par["NSlice"] / (num_dev * par["par_slices"]) < 2:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices and devices needs to be larger two.\n"
                "Current values are %i total Slices, %i parallel slices and "
                "%i compute devices." % (par["NSlice"], par["par_slices"], num_dev)
            )
        if streamed and par["NSlice"] % par["par_slices"]:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices needs to be an integer.\n"
                "Current values are %i total Slices with %i parallel slices."
                % (par["NSlice"], par["par_slices"])
            )
        if DTYPE == np.complex128:
            if streamed:
                kernname = "kernels/OpenCL_Kernels_double_streamed.c"
            else:
                kernname = "kernels/OpenCL_Kernels_double.c"
            for j in range(num_dev):
                self._prg.append(
                    Program(
                        self._ctx[j], open(resource_filename("pyqmri", kernname)).read()
                    )
                )
        else:
            if streamed:
                kernname = "kernels/OpenCL_Kernels_streamed.c"
            else:
                kernname = "kernels/OpenCL_Kernels.c"
            for j in range(num_dev):
                self._prg.append(
                    Program(
                        self._ctx[j], open(resource_filename("pyqmri", kernname)).read()
                    )
                )

        if imagespace:
            self._coils = []
            self.sliceaxis = 1
        else:
            self._data_shape = (
                par["NScan"],
                par["NC"],
                par["NSlice"],
                par["Nproj"],
                par["N"],
            )
            if self._streamed:
                self._data_trans_axes = (2, 0, 1, 3, 4)
                self._coils = np.require(
                    np.swapaxes(par["C"], 0, 1), requirements="C", dtype=DTYPE
                )

                if SMS:
                    self._data_shape = (
                        par["packs"] * par["numofpacks"],
                        par["NScan"],
                        par["NC"],
                        par["dimY"],
                        par["dimX"],
                    )
                    self._data_shape_T = (
                        par["NScan"],
                        par["NC"],
                        par["packs"] * par["numofpacks"],
                        par["dimY"],
                        par["dimX"],
                    )
                    self._expdim_dat = 1
                    self._expdim_C = 0
                else:
                    self._data_shape = (
                        par["NSlice"],
                        par["NScan"],
                        par["NC"],
                        par["Nproj"],
                        par["N"],
                    )
                    self._data_shape_T = self._data_shape
                    self._expdim_dat = 2
                    self._expdim_C = 1
            else:
                self._coils = clarray.to_device(self._queue[0], self.par["C"])

        self._MRI_operator, self._FT = operator.Operator.MRIOperatorFactory(
            par, self._prg, DTYPE, DTYPE_real, trafo, imagespace, SMS, streamed
        )

        self._grad_op = self._setupLinearOps(DTYPE, DTYPE_real)

        self._pdop = optimizer.IPianoBaseSolver.factory(
            self._prg,
            self._queue,
            self.par,
            self.ipiano_par,
            self._fval_init,
            self._coils,
            linops=(self._MRI_operator, self._grad_op),
            model=model,
            reg_type=self._reg_type,
            SMS=self._SMS,
            streamed=self._streamed,
            imagespace=self._imagespace,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real,
        )

        self._gamma = None
        self._delta = None
        self._omega = None
        self._step_val = None
        self._modelgrad = None

    def _setupLinearOps(self, DTYPE, DTYPE_real):
        grad_op = operator.Operator.GradientOperatorFactory(
            self.par, self._prg, DTYPE, DTYPE_real, self._streamed
        )
        return grad_op

    def execute(self, data):
        """Start the iPiano optimization.

        This method performs iterative regularized Piano optimization
        and calls the inner loop after precomputing the current linearization
        point. Results of the fitting process are saved after each
        linearization step to the output folder.

        Parameters
        ----------
              data : numpy.array
                the data to perform optimization/fitting on.
        """
        # self.ipiano_par["lambd"] *= (
        #                             (self.par["SNR_est"]))
        self._gamma = self.ipiano_par["gamma"]
        self._delta = self.ipiano_par["delta"]
        self._omega = self.ipiano_par["omega"]

        # iters = self.ipiano_par["start_iters"]
        x = np.copy(self._model.guess)
        x_new = np.copy(self._model.guess)

        # TODO: delete line
        self._step_val = np.nan_to_num(self._model.execute_forward(x))

        self._pdop.model = self._model

        for it in range(self.ipiano_par["max_gn_it"]):
            start = time.time()

            self._modelgrad = np.nan_to_num(self._model.execute_gradient(x))

            self._balanceModelGradients(x)
            self._step_val = np.nan_to_num(self._model.execute_forward(x))

            x_old = x
            x = x_new

            # _jacobi = np.sum(np.abs(self._modelgrad) ** 2, 1).astype(self._DTYPE_real)
            # _jacobi[_jacobi == 0] = 1e-8

            self._modelgrad = clarray.to_device(self._queue[0], self._modelgrad)

            self._pdop.modelgrad = self._modelgrad
            # self._pdop.jacobi = clarray.to_device(self._queue[0], _jacobi)

            self._updateIPIANORegPar(it)
            self._pdop.updateRegPar(self.ipiano_par)

            ### ipiano solver execute
            x_new = self._ipianoSolve3D(x, x_old, data, it)

            # iters = np.fmin(iters * 2, self.ipiano_par["max_iters"])

            end = time.time() - start
            self.gn_res.append(self._fval)
            print("-" * 75)
            print("iPIANO-Iter: %d  Elapsed time: %f seconds" % (it, end))
            print("-" * 75)
            self._fval_old = self._fval
            self._saveToFile(it, self._model.rescale(x_new)["data"])
        self._calcResidual(x_new, data, it + 1)

    def _updateIPIANORegPar(self, it):
        """Update the iPiano parameter.

        This method performs iterative regularized Piano optimization
        and calls the inner loop after precomputing the current linearization
        point. Results of the fitting process are saved after each
        linearization step to the output folder.

        Parameters
        ----------
              it : int
                the actual iteration of the optimization.
        """
        self.ipiano_par["delta"] = np.minimum(
            self._delta * self.ipiano_par["delta_inc"] ** it,
            self.ipiano_par["delta_max"],
        )
        self.ipiano_par["gamma"] = np.maximum(
            self._gamma * self.ipiano_par["gamma_dec"] ** it,
            self.ipiano_par["gamma_min"],
        )
        self.ipiano_par["omega"] = np.maximum(
            self._omega * self.ipiano_par["omega_dec"] ** it,
            self.ipiano_par["omega_min"],
        )

        # TODO: alpha calc
        # self.ipiano_par["beta"] = 0.04
        self.ipiano_par["alpha"] = 2 * (1 - self.ipiano_par["beta"]) / 1.0

    def _balanceModelGradients(self, result):
        """
        TODO: doc string
        """
        scale = np.reshape(
            self._modelgrad,
            (
                self.par["unknowns"],
                self.par["NScan"]
                * self.par["NSlice"]
                * self.par["dimY"]
                * self.par["dimX"],
            ),
        )
        scale = np.linalg.norm(scale, axis=-1)
        print("Initial Norm: ", np.linalg.norm(scale))
        print("Initial Ratio: ", scale)
        scale /= np.linalg.norm(scale) / np.sqrt(self.par["unknowns"])
        scale = 1 / scale
        scale[~np.isfinite(scale)] = 1
        for uk in range(self.par["unknowns"]):
            self._model.constraints[uk].update(scale[uk])
            result[uk, ...] *= self._model.uk_scale[uk]
            self._modelgrad[uk] /= self._model.uk_scale[uk]
            self._model.uk_scale[uk] *= scale[uk]
            result[uk, ...] /= self._model.uk_scale[uk]
            self._modelgrad[uk] *= self._model.uk_scale[uk]
        scale = np.reshape(
            self._modelgrad,
            (
                self.par["unknowns"],
                self.par["NScan"]
                * self.par["NSlice"]
                * self.par["dimY"]
                * self.par["dimX"],
            ),
        )
        scale = np.linalg.norm(scale, axis=-1)
        print("Norm after rescale: ", np.linalg.norm(scale))
        print("Ratio after rescale: ", scale)

    ###############################################################################
    # New .hdf5 save files ########################################################
    ###############################################################################
    def _saveToFile(self, myit, result):
        """TODO: doc string
        """
        f = h5py.File(self.par["outdir"] + "output_" + self.par["fname"], "a")
        f.create_dataset(
            "ipiano_result_" + str(myit), result.shape, dtype=self._DTYPE, data=result
        )
        f.attrs["res_ipiano_iter_" + str(myit)] = self._fval
        f.close()


    def _ipianoSolve3D(self, x, xold, data, GN_it):
        b = self._calcResidual(x, data, GN_it)

        tmpx = clarray.to_device(self._queue[0], x)
        # b = FCAx|xk
        # tt = FC dA/dx|xk
        tt = self._MRI_operator.fwdoop([tmpx, self._coils, self._modelgrad]).get()
        res = data - b + tt
        del tmpx

        tmpres = self._pdop.run(x, xold, res)
        for key in tmpres:
            if key == "x":
                if isinstance(tmpres[key], np.ndarray):
                    x = tmpres["x"]
                else:
                    x = tmpres["x"].get()

        return x

    def _calcResidual(self, x, data, GN_it):
        """TODO: doc string
        """
        b, grad = self._calcFwdGNPartLinear(x)
        grad_tv = grad[: self.par["unknowns_TGV"]]
        grad_H1 = grad[self.par["unknowns_TGV"] :]
        del grad

        datacost = self.ipiano_par["lambd"] / 2 * np.linalg.norm(data - b) ** 2
        L2Cost = np.linalg.norm(x) / (2.0 * self.ipiano_par["delta"])
        regcost = np.sum(
            np.abs(np.log(1 + self.ipiano_par["gamma"] * np.vdot(grad_tv, grad_tv)))
        )

        self._fval = (
            datacost
            + regcost
            + L2Cost
            # + self.ipiano_par["omega"] / 2 * np.linalg.norm(grad_H1) ** 2
        )
        del grad_tv, grad_H1

        if GN_it == 0:
            self._fval_init = self._fval
            # self._pdop.setFvalInit(self._fval)

        print("-" * 75)
        print("Initial Cost: %f" % (self._fval_init))
        print("Costs of Data: %f" % (datacost))
        print("Costs of LOG: {:.3E}".format(regcost))
        print("Costs of L2 Term: %f" % (L2Cost))
        print("-" * 75)
        print("Function value at iPiano-Step %i: %f" % (GN_it, self._fval))
        print("-" * 75)
        return b

    def _calcFwdGNPartLinear(self, x):
        """TODO: doc string
        """
        if self._imagespace is False:
            b = clarray.empty(self._queue[0], self._data_shape, dtype=self._DTYPE)
            self._FT.FFT(
                b,
                clarray.to_device(
                    self._queue[0], (self._step_val[:, None, ...] * self.par["C"])
                ),
            ).wait()
            b = b.get()
        else:
            b = self._step_val

        x = clarray.to_device(self._queue[0], np.require(x, requirements="C"))
        grad = clarray.to_device(
            self._queue[0], np.zeros(x.shape + (4,), dtype=self._DTYPE)
        )
        grad.add_event(self._grad_op.fwd(grad, x, wait_for=grad.events + x.events))
        x = x.get()
        grad = grad.get()
        return b, grad