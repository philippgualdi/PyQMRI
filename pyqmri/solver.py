#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module holding the classes for different numerical Optimizer."""

from __future__ import division

import sys

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.clmath as clmath
from pkg_resources import resource_filename

import pyqmri.operator as operator
import pyqmri.streaming as streaming
from pyqmri._helper_fun import CLProgram as Program


class CGSolver:
    """
    Conjugate Gradient Optimization Algorithm.

    This Class performs a CG reconstruction on single precission complex input
    data.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      NScan : int
        Number of Scan which should be used internally. Do not
        need to be the same number as in par["NScan"]
      trafo : bool
        Switch between radial (1) and Cartesian (0) fft.
      SMS : bool
        Simultaneouos Multi Slice. Switch between noraml (0)
        and slice accelerated (1) reconstruction.
    """

    def __init__(self, par, NScan=1, trafo=1, SMS=0):
        self._NSlice = par["NSlice"]
        NScan_save = par["NScan"]
        par["NScan"] = NScan
        self._NScan = NScan
        self._dimX = par["dimX"]
        self._dimY = par["dimY"]
        self._NC = par["NC"]
        self._queue = par["queue"][0]
        file = open(resource_filename("pyqmri", "kernels/OpenCL_Kernels.c"))
        self._prg = Program(par["ctx"][0], file.read())
        file.close()
        self._coils = cl.Buffer(
            par["ctx"][0],
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=par["C"].data,
        )
        self._DTYPE = par["DTYPE"]
        self._DTYPE_real = par["DTYPE_real"]

        self.__op, FT = operator.Operator.MRIOperatorFactory(
            par, [self._prg], self._DTYPE, self._DTYPE_real, trafo=trafo, SMS=SMS
        )

        if SMS:
            self._tmp_sino = clarray.empty(
                self._queue,
                (
                    self._NScan,
                    self._NC,
                    int(self._NSlice / par["MB"]),
                    par["Nproj"],
                    par["N"],
                ),
                self._DTYPE,
                "C",
            )
        else:
            self._tmp_sino = clarray.empty(
                self._queue,
                (self._NScan, self._NC, self._NSlice, par["Nproj"], par["N"]),
                self._DTYPE,
                "C",
            )
        self._FT = FT.FFT
        self._FTH = FT.FFTH
        self._tmp_result = clarray.empty(
            self._queue,
            (self._NScan, self._NC, self._NSlice, self._dimY, self._dimX),
            self._DTYPE,
            "C",
        )
        par["NScan"] = NScan_save
        self._scan_offset = 0

    def __del__(self):
        """Destructor.

        Releases GPU memory arrays.
        """
        del self._coils
        del self._tmp_result
        del self._queue
        del self._tmp_sino
        del self._FT
        del self._FTH

    def run(self, data, iters=30, lambd=1e-5, tol=1e-8, guess=None, scan_offset=0):
        """
        Start the CG reconstruction.

        All attributes after data are considered keyword only.

        Parameters
        ----------
          data : numpy.array
            The complex k-space data which serves as the basis for the images.
          iters : int
            Maximum number of CG iterations
          lambd : float
            Weighting parameter for the Tikhonov regularization
          tol : float
            Termination criterion. If the energy decreases below this
            threshold the algorithm is terminated.
          guess : numpy.array
            An optional initial guess for the images. If None, zeros is used.

        Returns
        -------
          numpy.Array:
              The result of the image reconstruction.
        """
        self._scan_offset = scan_offset
        if guess is not None:
            x = clarray.to_device(self._queue, guess)
        else:
            x = clarray.zeros(
                self._queue,
                (self._NScan, 1, self._NSlice, self._dimY, self._dimX),
                self._DTYPE,
                "C",
            )
        b = clarray.empty(
            self._queue,
            (self._NScan, 1, self._NSlice, self._dimY, self._dimX),
            self._DTYPE,
            "C",
        )
        Ax = clarray.empty(
            self._queue,
            (self._NScan, 1, self._NSlice, self._dimY, self._dimX),
            self._DTYPE,
            "C",
        )

        data = clarray.to_device(self._queue, data)
        self._operator_rhs(b, data)
        res = b
        p = res
        delta = np.linalg.norm(res.get()) ** 2 / np.linalg.norm(b.get()) ** 2

        for i in range(iters):
            self._operator_lhs(Ax, p)
            Ax = Ax + lambd * p
            alpha = (clarray.vdot(res, res) / (clarray.vdot(p, Ax))).real.get()
            x = x + alpha * p
            res_new = res - alpha * Ax
            delta = np.linalg.norm(res_new.get()) ** 2 / np.linalg.norm(b.get()) ** 2
            if delta < tol:
                print("Converged after %i iterations to %1.3e." % (i, delta))
                del Ax, b, res, p, data, res_new
                return np.squeeze(x.get())
            beta = (clarray.vdot(res_new, res_new) / clarray.vdot(res, res)).real.get()
            p = res_new + beta * p
            (res, res_new) = (res_new, res)
        del Ax, b, res, p, data, res_new
        return np.squeeze(x.get())

    def eval_fwd_kspace_cg(self, y, x, wait_for=None):
        """Apply forward operator for image reconstruction.

        Parameters
        ----------
          y : PyOpenCL.Array
            The result of the computation
          x : PyOpenCL.Array
            The input array
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg.operator_fwd_cg(
            self._queue,
            (self._NSlice, self._dimY, self._dimX),
            None,
            y.data,
            x.data,
            self._coils,
            np.int32(self._NC),
            np.int32(self._NScan),
            wait_for=wait_for,
        )

    def _operator_lhs(self, out, x, wait_for=None):
        """Compute the left hand side of the CG equation.

        Parameters
        ----------
          out : PyOpenCL.Array
            The result of the computation
          x : PyOpenCL.Array
            The input array
          wait_for : list of PyopenCL.Event, None
            A List of PyOpenCL events to wait for.

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        self._tmp_result.add_event(
            self.eval_fwd_kspace_cg(
                self._tmp_result, x, wait_for=self._tmp_result.events + x.events
            )
        )
        self._tmp_sino.add_event(
            self._FT(self._tmp_sino, self._tmp_result, scan_offset=self._scan_offset)
        )
        return self._operator_rhs(out, self._tmp_sino)

    def _operator_rhs(self, out, x, wait_for=None):
        """Compute the right hand side of the CG equation.

        Parameters
        ----------
          out : PyOpenCL.Array
            The result of the computation
          x : PyOpenCL.Array
            The input array
          wait_for : list of PyopenCL.Event
            A List of PyOpenCL events to wait for.

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        self._tmp_result.add_event(
            self._FTH(
                self._tmp_result,
                x,
                wait_for=wait_for + x.events,
                scan_offset=self._scan_offset,
            )
        )
        return self._prg.operator_ad_cg(
            self._queue,
            (self._NSlice, self._dimY, self._dimX),
            None,
            out.data,
            self._tmp_result.data,
            self._coils,
            np.int32(self._NC),
            np.int32(self._NScan),
            wait_for=(self._tmp_result.events + out.events + wait_for),
        )


class IPianoBaseSolver:
    """iPiano optimization.

    This Class performs a linearized ipiano based reconstruction
    on single precission complex input data.

    Parameters
    ----------
      par : dict
        A python dict containing the necessary information to
        setup the object. Needs to contain the number of slices (NSlice),
        number of scans (NScan), image dimensions (dimX, dimY), number of
        coils (NC), sampling points (N) and read outs (NProj)
        a PyOpenCL queue (queue) and the complex coil
        sensitivities (C).
      irgn_par : dict
        A python dict containing the regularization
        parameters for a given gauss newton step.
      queue : list of PyOpenCL.Queues
        A list of PyOpenCL queues to perform the optimization.
      tau : float
        Estimate of the initial step size based on the
        operator norm of the linear operator.
      fval : float
        Estimate of the initial cost function value to
        scale the displayed values.
      prg : PyOpenCL Program A PyOpenCL Program containing the
        kernels for optimization.
      reg_type : string String to choose between "TV" and "TGV"
        optimization.
      data_operator : PyQMRI Operator The operator to traverse from
        parameter to data space.
      coil : PyOpenCL Buffer or empty list
        coil buffer, empty list if image based fitting is used.
      model : PyQMRI.Model
        Instance of a PyQMRI.Model to perform plotting

    Attributes
    ----------
      delta : float
        Regularization parameter for L2 penalty on linearization point.
      omega : float
        Not used. Should be set to 0
      lambd : float
        Regularization parameter in front of data fidelity term.
      tol : float
        Relative toleraze to stop iterating
      stag : float
        Stagnation detection parameter
      display_iterations : bool
        Switch between plotting (true) of intermediate results
      mu : float
        Strong convecity parameter (inverse of delta).
      tau : float
        Estimated step size based on operator norm of regularization.
      beta_line : float
        Ratio between dual and primal step size
      theta_line : float
        Line search parameter
      unknwons_TGV : int
        Number of T(G)V unknowns
      unknowns_H1 : int
        Number of H1 unknowns (should be 0 for now)
      unknowns : int
        Total number of unknowns (T(G)V+H1)
      num_dev : int
        Total number of compute devices
      dz : float
        Ratio between 3rd dimension and isotropic 1st and 2nd image dimension.
      model : PyQMRI.Model
        The model which should be fitted
      modelgrad : PyOpenCL.Array or numpy.Array
        The partial derivatives evaluated at the linearization point.
        This variable is set in the PyQMRI.irgn Class.
      min_const : list of float
        list of minimal values, one for each unknown
      max_const : list of float
        list of maximal values, one for each unknown
      real_const : list of int
        list if a unknown is constrained to real values only. (1 True, 0 False)
    """

    def __init__(
        self,
        par,
        ipiano_par,
        queue,
        # tau,
        fval,
        prg,
        coil,
        model,
        DTYPE=np.complex64,
        DTYPE_real=np.float32,
    ):
        self._DTYPE = DTYPE
        self._DTYPE_real = DTYPE_real
        self.delta = ipiano_par["delta"]
        self.omega = ipiano_par["omega"]
        self.lambd = ipiano_par["lambd"]
        self.tol = ipiano_par["tol"]
        self.stag = ipiano_par["stag"]
        self.display_iterations = ipiano_par["display_iterations"]
        # self.mu = 1 / self.delta
        # self.tau = tau
        self.beta_line = 1e3  # 1e10#1e12
        self.theta_line = DTYPE_real(1.0)
        self.unknowns_TGV = par["unknowns_TGV"]
        self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self.num_dev = len(par["num_dev"])
        self.dz = par["dz"]
        # Delete
        self._fval_init = fval
        self._prg = prg
        self._queue = queue
        self.model = model
        self._coils = coil
        self.modelgrad = None
        self.min_const = None
        self.max_const = None
        self.real_const = None
        self._kernelsize = (
            par["par_slices"] + par["overlap"],
            par["dimY"],
            par["dimX"],
        )

    @staticmethod
    def factory(
        prg,
        queue,
        par,
        ipiano_par,
        init_fval,
        coils,
        linops,
        model,
        reg_type="LOG",
        SMS=False,
        streamed=False,
        imagespace=False,
        DTYPE=np.complex64,
        DTYPE_real=np.float32,
    ):
        """
        Generate a PDSolver object.

        Parameters
        ----------
          prg : PyOpenCL.Program
            A PyOpenCL Program containing the
            kernels for optimization.
          queue : list of PyOpenCL.Queues
            A list of PyOpenCL queues to perform the optimization.
          par : dict
            A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par : dict
            A python dict containing the regularization
            parameters for a given gauss newton step.
          init_fval : float
            Estimate of the initial cost function value to
            scale the displayed values.
          coils : PyOpenCL Buffer or empty list
            The coils used for reconstruction.
          linops : list of PyQMRI Operator
            The linear operators used for fitting.
          model : PyQMRI.Model
            The model which should be fitted
          reg_type : string, "TGV"
            String to choose between "TV" and "TGV"
            optimization.
          SMS : bool, false
            Switch between standard (false) and SMS (True) fitting.
          streamed : bool, false
            Switch between streamed (1) and normal (0) reconstruction.
          imagespace : bool, false
            Switch between k-space (false) and imagespace based fitting (true).
          DTYPE : numpy.dtype, numpy.complex64
             Complex working precission.
          DTYPE_real : numpy.dtype, numpy.float32
            Real working precission.
        """
        if reg_type == "LOG":
            pdop = IPianoSolverLog(
                par,
                ipiano_par,
                queue,
                init_fval,
                prg,
                linops,
                coils,
                model,
                DTYPE=DTYPE,
                DTYPE_real=DTYPE_real,
            )
        else:
            raise NotImplementedError
        return pdop

    def __del__(self):
        """Destructor.

        Releases GPU memory arrays.
        """

    def run(self, x, xold, data):
        """
        Optimization with 3D T(G)V regularization.

        Parameters
        ----------
          x (numpy.array):
            Initial guess for the unknown parameters
          x (numpy.array):
            The complex valued data to fit.
          iters : int
            Number of primal-dual iterations to run

        Returns
        -------
          tupel:
            A tupel of all primal variables (x,v in the Paper). If no
            streaming is used, the two entries are opf class PyOpenCL.Array,
            otherwise Numpy.Array.
        """
        self._updateConstraints()

        (step_vars, step_vars_new, tmp_results, data) = self._setupVariables(
            x, xold, data
        )

        self._updateInitial(out_fwd=step_vars, tmp_res=tmp_results, in_step=step_vars)

        self._update(out_step=step_vars, out_tmp=tmp_results, in_step=step_vars)

        # TODO: compute fwd step
        self._update_fwd(
            out_step_new=step_vars_new, out_tmp=tmp_results, in_step=step_vars
        )

        if self.display_iterations:
            if isinstance(step_vars_new["x"], np.ndarray):
                self.model.plot_unknowns(step_vars_new["x"])
            else:
                self.model.plot_unknowns(step_vars_new["x"].get())

        self._calcResidual(step_vars, tmp_results, step_vars_new, data)
        return step_vars_new

    def _update(self, out_step, out_tmp, in_step):
        pass

    def _update_inertial(self, out_step, out_fwd, in_step):
        pass

    def update_func(
        self,
    ):
        pass

    def _update_fwd(self, out_step, out_tmp, in_step):
        pass

    def _updateInitial(self, out_fwd, out_adj, in_primal, in_dual):
        pass

    def _calcResidual(self, step_vars, tmp_results, step_vars_new, data):
        return ({}, {})

    def _setupVariables(self, x, xold, data):
        return ({}, {}, {})

    def _updateConstraints(self):
        num_const = len(self.model.constraints)
        min_const = np.zeros((num_const), dtype=self._DTYPE_real)
        max_const = np.zeros((num_const), dtype=self._DTYPE_real)
        real_const = np.zeros((num_const), dtype=np.int32)
        for j in range(num_const):
            min_const[j] = self._DTYPE_real(self.model.constraints[j].min)
            max_const[j] = self._DTYPE_real(self.model.constraints[j].max)
            real_const[j] = np.int32(self.model.constraints[j].real)

        self.min_const = []
        self.max_const = []
        self.real_const = []
        for j in range(self.num_dev):
            self.min_const.append(clarray.to_device(self._queue[4 * j], min_const))
            self.max_const.append(clarray.to_device(self._queue[4 * j], max_const))
            self.real_const.append(clarray.to_device(self._queue[4 * j], real_const))

    def updateRegPar(self, ipiano_par):
        """Update the regularization parameters.

          Performs an update of the regularization parameters as these usually
          vary from one to another Gauss-Newton step.

        Parameters
        ----------
          irgn_par (dic): A dictionary containing the new parameters.
        """
        # self.alpha = ipiano_par["alpha"]
        # self.beta = ipiano_par["beta"]
        # self.delta = ipiano_par["delta"]
        # self.omega = ipiano_par["omega"]
        # self.lambd = ipiano_par["lambd"]
        # self.mu = 1 / self.delta

    def update_fwd(self, outp, inp, par, idx=0, idxq=0, bound_cond=0, wait_for=None):
        """Forward update of the x variable in the iPiano Algorithm.

        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update step
          inp : PyOpenCL.Array
            The previous values of x
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          bound_cond : int
            Apply boundary condition (1) or not (0).
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for

        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_ipiano_fwd(
            self._queue[4 * idx + idxq],
            self._kernelsize,
            None,
            outp.data,
            inp[0].data,
            inp[1].data,
            inp[2].data,
            inp[3].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self._DTYPE_real(par[0] / par[2]),
            self._DTYPE_real(1 / (1 + par[0] / par[2])),
            self.min_const[idx].data,
            self.max_const[idx].data,
            self.real_const[idx].data,
            np.int32(self.unknowns),
            wait_for=(
                inp[0].events + inp[1].events + inp[2].events + inp[3].events + wait_for
            ),
        )

    def setFvalInit(self, fval):
        """Set the initial value of the cost function.

        Parameters
        ----------
          fval : float
            The initial cost of the optimization problem
        """
        self._fval_init = fval


class IPianoSolverLog(IPianoBaseSolver):
    """Primal Dual splitting optimization for Log.

    This Class performs a primal-dual variable splitting based reconstruction
    on single precission complex input data.

    Parameters
    ----------
          par : dict
            A python dict containing the necessary information to
            setup the object. Needs to contain the number of slices (NSlice),
            number of scans (NScan), image dimensions (dimX, dimY), number of
            coils (NC), sampling points (N) and read outs (NProj)
            a PyOpenCL queue (queue) and the complex coil
            sensitivities (C).
          irgn_par : dict
            A python dict containing the regularization
            parameters for a given gauss newton step.
          queue : list of PyOpenCL.Queues
            A list of PyOpenCL queues to perform the optimization.
          tau : float
            Estimated step size based on operator norm of regularization.
          fval : float
            Estimate of the initial cost function value to
            scale the displayed values.
          prg : PyOpenCL.Program
            A PyOpenCL Program containing the
            kernels for optimization.
          linops : list of PyQMRI Operator
            The linear operators used for fitting.
          coils : PyOpenCL Buffer or empty list
            The coils used for reconstruction.
          model : PyQMRI.Model
            The model which should be fitted

      Attributes
      ----------
        alpha : float
          TV regularization weight
    """

    def __init__(
        self,
        par,
        irgn_par,
        queue,
        # tau,
        fval,
        prg,
        linop,
        coils,
        model,
        **kwargs
    ):
        super().__init__(
            par,
            irgn_par,
            queue,
            # tau,
            fval,
            prg,
            coils,
            model,
            **kwargs
        )
        self.gamma = 1  # irgn_par["gamma"]
        self.alpha = 0.01  # irgn_par["alpha"]
        self.beta = 0.95  # irgn_par["beta"]
        self._op = linop[0]
        self._grad_op = linop[1]

    def _setupVariables(self, x, xold, data):
        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        step_vars = {}
        step_vars_new = {}
        tmp_results = {}

        step_vars["x"] = clarray.to_device(self._queue[0], x)
        step_vars["xold"] = clarray.to_device(self._queue[0], xold)
        step_vars["xk"] = step_vars["x"].copy()

        step_vars_new["x"] = clarray.empty_like(step_vars["x"])

        tmp_results["gradFx"] = step_vars["x"].copy()
        tmp_results["gradGx"] = step_vars["x"].copy()
        tmp_results["DADA"] = clarray.empty_like(step_vars["x"])
        tmp_results["DAd"] = clarray.empty_like(step_vars["x"])
        tmp_results["d"] = data.copy()
        tmp_results["Ax"] = clarray.empty_like(data)

        tmp_results["gradx"] = clarray.zeros(
            self._queue[0], step_vars["x"].shape + (4,), dtype=self._DTYPE
        )

        return (step_vars, step_vars_new, tmp_results, data)

    def _updateInitial(self, out_fwd, tmp_res, in_step):

        tmp_res["Ax"].add_event(
            self._op.fwd(tmp_res["Ax"], [in_step["x"], self._coils, self.modelgrad])
        )
        tmp_res["DADA"].add_event(
            self._op.adj(
                tmp_res["DADA"],
                [tmp_res["Ax"], self._coils, self.modelgrad, self._grad_op.ratio],
                wait_for=tmp_res["Ax"].events,
            )
        )
        tmp_res["DAd"].add_event(
            self._op.adj(
                tmp_res["DAd"],
                [tmp_res["d"], self._coils, self.modelgrad, self._grad_op.ratio],
            )
        )

        tmp_res["gradx"].add_event(self._grad_op.fwd(tmp_res["gradx"], out_fwd["x"]))

    def _update(self, out_step, out_tmp, in_step):

        out_tmp["gradFx"] = (
            out_tmp["DADA"]
            - out_tmp["DAd"]
            + self.lambd
            * abs(
                (
                    clarray.vdot(out_tmp["gradx"], out_tmp["gradx"])
                    / (
                        1
                        + self.lambd * clarray.vdot(out_tmp["gradx"], out_tmp["gradx"])
                    )
                ).get()
            )
        )


    def _update_fwd(self, out_step_new, out_tmp, in_step):
        out_step_new["x"].add_event(
            self.update_fwd(
                outp=out_step_new["x"],
                inp=(in_step["x"], out_tmp["gradFx"], in_step["xk"], in_step["xold"]),
                par=(self.alpha, self.beta, self.gamma),
            )
        )

    def _calcResidual(self, step_vars, tmp_results, step_vars_new, data):

        # TODO: step_new equal to f(x)
        f_new = clarray.vdot(
            tmp_results["DADA"], tmp_results["DAd"]
        ) + clarray.sum(  # self.lambd *
            abs(
                clmath.log(
                    1
                    + self.gamma
                    * clarray.vdot(tmp_results["gradx"], tmp_results["gradx"])
                )
            )
        )  # .real

        # TODO: g_new equal to g(x)
        g_new = (
            1
            / (2 * self.delta)
            * clarray.vdot(  # x_new -xk
                step_vars_new["x"] - step_vars["xk"],
                step_vars_new["x"] - step_vars["xk"],
            )
        )  # .real
        return f_new.get(), g_new.get()
