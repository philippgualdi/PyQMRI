#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Graz University of Technology.
#
# PyQMRI is a free software; you can redistribute it and/or modify it
# under the terms of the Apache-2.0 License.


""" This module holds the classes for IPiano Optimization without streaming.
Attribues:
  DTYPE (complex64):
    Complex working precission. Currently single precission only.
  DTYPE_real (float32):
    Real working precission. Currently single precission only.
"""
from __future__ import division

import time

import h5py
import numpy as np
import pyopencl.array as clarray
from pkg_resources import resource_filename

import pyqmri.operator as operator
from pyqmri._helper_fun import CLProgram as Program
from pyqmri._helper_fun import _utils as utils

from .solvers import IPianoBaseSolver


class IPianoOptimizer:
    """Main iPiano Optimization class.
    This Class performs iPiano Optimization either with LOG(barrier)  # or TV regularization.
    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      model : pyqmri.model
        Which model should be used for fitting.
        Expects a pyqmri.model instance.
      trafo : int, 1
        Select radial (1, default) or cartesian (0) sampling for the fft.
      imagespace : bool, false
        Perform the fitting from k-space data (false, default) or from the
        image series (true)
      SMS : int, 0
        Select if simultaneous multi-slice acquisition was used (1) or
        standard slice-by-slice acquisition was done (0, default).
      reg_type : str, "LOG"
        Select between "LOG" (default) or !NOT IMPLEMENTED! regularization.
      config : str, ''
        Name of config file. If empty, default config file will be generated.
      streamed : bool, false
        Select between standard reconstruction (false)
        or streamed reconstruction (true) for large volumetric data which
        does not fit on the GPU memory at once.
      DTYPE : numpy.dtype, numpy.complex64
        Complex working precission.
      DTYPE_real : numpy.dtype, numpy.float32
        Real working precission.

    Attributes
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      gn_res : list of floats
        The residual values for of each Gauss-Newton step. Each iteration
        appends its value to the list.
      irgn_par : dict
        The parameters read from the config file to guide the IRGN
        optimization process
    """

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

        if streamed or SMS == 1:
            raise NotImplementedError("Not implemented")

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

        self._pdop = IPianoBaseSolver.factory(
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

        self._delta = None
        self._step_val = None
        self._modelgrad = None

        # Possible delete
        self._gamma = None
        self._omega = None

    def _setupLinearOps(self, DTYPE, DTYPE_real):
        """Setup the Gradient Operator."""

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

        self._delta = self.ipiano_par["delta"]

        # Todo: remove
        self._gamma = self.ipiano_par["gamma"]
        self._omega = self.ipiano_par["omega"]

        result = np.copy(self._model.guess)

        self._step_val = np.nan_to_num(self._model.execute_forward(result))

        self._pdop.model = self._model

        # Start optimizations
        for ign in range(self.ipiano_par["max_gn_it"]):
            start = time.time()

            self._modelgrad = np.nan_to_num(self._model.execute_gradient(result))

            self._balanceModelGradients(result)
            self._step_val = np.nan_to_num(self._model.execute_forward(result))

            self._modelgrad = clarray.to_device(self._queue[0], self._modelgrad)

            self._pdop.modelgrad = self._modelgrad

            self._updateIPIANORegPar(ign)
            self._pdop.updateRegPar(self.ipiano_par)

            ### ipiano solver execute
            result = self._ipianoSolve3D(result, data, ign)

            # iters = np.fmin(iters * 2, self.ipiano_par["max_iters"])

            end = time.time() - start
            self.gn_res.append(self._fval)
            print("-" * 75)
            print("iPIANO-Iter: %d  Elapsed time: %f seconds" % (ign, end))
            print("-" * 75)
            self._fval_old = self._fval
            self._saveToFile(ign, self._model.rescale(result)["data"])
            # self._calcResidual(result, data, ign + 1)

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

    def _balanceModelGradients(self, result):
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
        """TODO: doc string"""
        f = h5py.File(self.par["outdir"] + "output_" + self.par["fname"]+ ".h5", "a")
        f.create_dataset(
            "ipiano_result_" + str(myit), result.shape, dtype=self._DTYPE, data=result
        )
        f.attrs["res_ipiano_iter_" + str(myit)] = self._fval
        f.close()

    def _ipianoSolve3D(self, result, data, it):
        """TODO: doc string"""
        b = self._calcResidual(result, data, it)

        tmpx = clarray.to_device(self._queue[0], result)
        # b = FCAx|xk
        # tt = FC dA/dx|xk
        tt = self._MRI_operator.fwdoop([tmpx, self._coils, self._modelgrad]).get()
        res = data - b + tt
        del tmpx

        tmpres = self._pdop.run(result, res)
        for key in tmpres:
            if key == "x":
                if isinstance(tmpres[key], np.ndarray):
                    x = tmpres["x"]
                else:
                    x = tmpres["x"].get()

        return x

    def _calcResidual(self, x, data, it):
        """TODO: doc string"""
        b, grad = self._calcFwdGNPartLinear(x)
        grad_reg = grad[: self.par["unknowns_TGV"]]
        grad_H1 = grad[self.par["unknowns_TGV"] :]
        del grad

        datacost = self.ipiano_par["delta"] / 2  * np.linalg.norm(data - b) ** 2
        regcost = self.ipiano_par["lambd"] * np.sum(
            np.abs(np.log(1 + np.vdot(grad_reg, grad_reg)))
        )

        self._fval = (
            datacost
            + regcost
        )
        del grad_reg, grad_H1

        if it == 0:
            self._fval_init = self._fval
            self._pdop.setFvalInit(self._fval)

        print("-" * 75)
        print("Initial Cost: {:.5E}".format(self._fval_init))
        print("Costs of Data: {:.5E}".format((1e3*datacost / self._fval_init)))
        print("Costs of REG: {:.5E}".format((1e3*regcost / self._fval_init)))
        print("-" * 75)
        print("Function value at iPiano-Step {}: {:.5f}".format(it,  1e3*self._fval / self._fval_init))
        print("-" * 75)
        return b

    def _calcFwdGNPartLinear(self, x):
        """TODO: doc string"""
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
