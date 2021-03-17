""" This module holds the classes for iPALM Optimization without streaming.

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


class IPALMOptimizer:
    """Main iPALM Optimization class.

    """

    def __init__(self, par, model, trafo=1, imagespace=False, SMS=0,
                 reg_type='LOG', config='', streamed=False,
                 DTYPE=np.complex64, DTYPE_real=np.float32):

        self.par = par
        self.gn_res = []
        self.irgn_par = utils.read_config(config, reg_type)
        utils.save_config(self.irgn_par, par["outdir"], reg_type)
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
                "%i compute devices."
                % (par["NSlice"], par["par_slices"], num_dev))
        if streamed and par["NSlice"] % par["par_slices"]:
            raise ValueError(
                "Number of Slices devided by parallel "
                "computed slices needs to be an integer.\n"
                "Current values are %i total Slices with %i parallel slices."
                % (par["NSlice"], par["par_slices"]))
        if DTYPE == np.complex128:
            if streamed:
                kernname = 'kernels/OpenCL_Kernels_double_streamed.c'
            else:
                kernname = 'kernels/OpenCL_Kernels_double.c'
            for j in range(num_dev):
                self._prg.append(Program(
                    self._ctx[j],
                    open(
                        resource_filename(
                            'pyqmri', kernname)
                    ).read()))
        else:
            if streamed:
                kernname = 'kernels/OpenCL_Kernels_streamed.c'
            else:
                kernname = 'kernels/OpenCL_Kernels.c'
            for j in range(num_dev):
                self._prg.append(Program(
                    self._ctx[j],
                    open(
                        resource_filename(
                            'pyqmri', kernname)).read()))

        if imagespace:
            self._coils = []
            self.sliceaxis = 1
        else:
            self._data_shape = (par["NScan"], par["NC"],
                                par["NSlice"], par["Nproj"], par["N"])
            if self._streamed:
                self._data_trans_axes = (2, 0, 1, 3, 4)
                self._coils = np.require(
                    np.swapaxes(par["C"], 0, 1), requirements='C',
                    dtype=DTYPE)

                if SMS:
                    self._data_shape = (par["packs"] * par["numofpacks"],
                                        par["NScan"],
                                        par["NC"], par["dimY"], par["dimX"])
                    self._data_shape_T = (par["NScan"], par["NC"],
                                          par["packs"] * par["numofpacks"],
                                          par["dimY"], par["dimX"])
                    self._expdim_dat = 1
                    self._expdim_C = 0
                else:
                    self._data_shape = (par["NSlice"], par["NScan"], par["NC"],
                                        par["Nproj"], par["N"])
                    self._data_shape_T = self._data_shape
                    self._expdim_dat = 2
                    self._expdim_C = 1
            else:
                self._coils = clarray.to_device(self._queue[0],
                                                self.par["C"])

        self._MRI_operator, self._FT = operator.Operator.MRIOperatorFactory(
            par,
            self._prg,
            DTYPE,
            DTYPE_real,
            trafo,
            imagespace,
            SMS,
            streamed
        )

        # TODO: implement gradient for iPALM
        self._grad_op = self._setupOps(
            DTYPE,
            DTYPE_real)

        self._pdop = optimizer.PDBaseSolver.factory(
            self._prg,
            self._queue,
            self.par,
            self.irgn_par,
            self._fval_init,
            self._coils,
            linops=(self._MRI_operator, self._grad_op),
            model=model,
            reg_type=self._reg_type,
            SMS=self._SMS,
            streamed=self._streamed,
            imagespace=self._imagespace,
            DTYPE=DTYPE,
            DTYPE_real=DTYPE_real
        )

        self._gamma = None
        self._delta = None
        self._omega = None
        self._step_val = None
        self._modelgrad = None
        pass

    def _setupOps(self, DTYPE, DTYPE_real):
        grad_op = operator.Operator.GradientOperatorFactory(
            self.par,
            self._prg,
            DTYPE,
            DTYPE_real,
            self._streamed)

        return grad_op

    def execute(self, data):
        """Start the iPALM optimization.

        This method performs iterative regularized Gauss-Newton optimization
        and calls the inner loop after precomputing the current linearization
        point. Results of the fitting process are saved after each
        linearization step to the output folder.

        Parameters
        ----------
          data : numpy.array
            the data to perform optimization/fitting on.
        """
        self._gamma = self.irgn_par["gamma"]
        self._delta = self.irgn_par["delta"]
        self._omega = self.irgn_par["omega"]

        iters = self.irgn_par["start_iters"]
        result = np.copy(self._model.guess)

        self._step_val = np.nan_to_num(self._model.execute_forward(result))

        for ign in range(self.irgn_par["max_gn_it"]):
            start = time.time()
            self._modelgrad = np.nan_to_num(
                self._model.execute_gradient(result))

            self._balanceModelGradients(result)
            self._step_val = np.nan_to_num(self._model.execute_forward(result))

            _jacobi = np.sum(
                    np.abs(
                        self._modelgrad)**2, 1).astype(self._DTYPE_real)
            _jacobi[_jacobi == 0] = 1e-8
            self._modelgrad = clarray.to_device(
                    self._queue[0],
                    self._modelgrad)
            self._pdop.model = self._model
            self._pdop.modelgrad = self._modelgrad
            self._pdop.jacobi = clarray.to_device(
                    self._queue[0],
                    _jacobi)
            self._updateRegressionPar(ign)
            self._pdop.updateRegPar(self.irgn_par)

            result = self._irgnSolve3D(result, iters, data, ign)

            iters = np.fmin(iters * 2, self.irgn_par["max_iters"])

            end = time.time() - start
            self.gn_res.append(self._fval)
            print("-" * 75)
            print("iPALM-Iter: %d  Elapsed time: %f seconds" % (ign, end))
            print("-" * 75)
            self._fval_old = self._fval
            self._saveToFile(ign, self._model.rescale(result)["data"])
            self._calcResidual(result, data, ign+1)

    ###############################################################################
    # New .hdf5 save files ########################################################
    ###############################################################################
    def _saveToFile(self, myit, result):
        f = h5py.File(self.par["outdir"] + "output_" + self.par["fname"], "a")
        f.create_dataset("ipalm_result_" + str(myit), result.shape,
                                 dtype=self._DTYPE, data=result)
        f.attrs['res_ipalm_iter_' + str(myit)] = self._fval
        f.close()


    def _updateRegressionPar(self, ign):
        self.irgn_par["alpha"] = np.maximum(0, (ign - 1) / (ign + 2))
        self.irgn_par["beta"] = np.maximum(0, (ign - 1) / (ign + 2))
