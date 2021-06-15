# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Graz University of Technology.
#
# PyQMRI is a free software; you can redistribute it and/or modify it
# under the terms of the Apache-2.0 License.

import sys
from abc import ABC

import numpy as np
import pyopencl.array as clarray
import pyopencl.clmath as clmath
import pyopencl.elementwise as elwis


class IPianoBaseSolver(ABC):
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
        self.alpha = ipiano_par.get("alpha", 0.0)
        self.alpha_max = ipiano_par.get("alpha_max", 0.0)
        self.beta = ipiano_par.get("beta", 1.0)
        # Iterations used by the solver
        self.iters = ipiano_par.get("max_iters", 10)
        self.tol = ipiano_par["tol"]
        self.stag = ipiano_par["stag"]
        self.display_iterations = ipiano_par["display_iterations"]
        # self.mu = 1 / self.delta
        self.beta_line = 1e3  # 1e10#1e12
        self.theta_line = DTYPE_real(1.0)
        self.unknowns_TGV = par["unknowns_TGV"]
        # self.unknowns_H1 = par["unknowns_H1"]
        self.unknowns = par["unknowns"]
        self.num_dev = len(par["num_dev"])
        self.dz = par["dz"]
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

        var_size = "float"
        if self._DTYPE_real == np.float64:
            var_size = "double"

        self.normkernl = elwis.ElementwiseKernel(
            context=par["ctx"][0],
            arguments="float *out, {}2 *x".format(var_size, var_size),
            operation="out[i]=x[i].s0*x[i].s0+x[i].s1*x[i].s1",
            name="norm_kernel",
        )
        self.abskernl = elwis.ElementwiseKernel(
            context=par["ctx"][0],
            arguments="float *out, {}2 *x".format(var_size, var_size),
            operation="out[i]=x[i].s0+x[i].s1",
            name="abs_kernel",
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

    def run(self, x, data):
        """
        Optimization with 3D iPiano regularization.
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

        (step_out, tmp_results, step_in, data) = self._setupVariables(x, data)

        self._calcStepsize(x_shape=x.shape, data_shape=data.shape)

        for i in range(self.iters):
            self._preUpdate(tmp_results, step_in, i)

            self._update_f(tmp_results, step_in)

            self._update(step_out, tmp_results, step_in)

            self._postUpdate(step_out, tmp_results, step_in)

            if self.display_iterations and np.mod(i, 100):
              if isinstance(step_out["x"], np.ndarray):
                  self.model.plot_unknowns(step_out["x"])
              else:
                  self.model.plot_unknowns(step_out["x"].get())

            if not np.mod(i, 10):
                # TODO: f_new use or remove
                cost, f_new = self._calcResidual(step_out, tmp_results, step_in, data)
                sys.stdout.write(
                    "Iteration: %04d ---- \u03B1  %2.2e  \u2207 f: %2.8e, \r"
                    % (i, self.alpha, cost)
                )
            sys.stdout.flush()

        return step_out

    def _preUpdate(self, tmp_results, step_in):
        ...

    def _update_f(self, tmp_results, step_in):
        ...

    def _calcStepsize(self, x_shape, data_shape, iterations=50):
        ...

    def _update(self, step_out, tmp_results, step_in):
        ...

    def _postUpdate(self, step_out, tmp_results, step_in):
        ...

    def _calcResidual(self, step_out, tmp_results, step_in, data):
        ...

    def _setupVariables(self, x, data):
        ...

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
          Performs an update of the regularization parameter.
        Parameters
        ----------
          ipiano_par (dic): A dictionary containing the new parameters.
        """
        self.alpha = ipiano_par["alpha"]
        self.beta = ipiano_par["beta"]
        self.delta = ipiano_par["delta"]
        self.omega = ipiano_par["omega"]
        self.lambd = ipiano_par["lambd"]
        print("delta: %s, lambd %s" % (self.delta, self.lambd))

    def setFvalInit(self, fval):
        """Set the initial value of the cost function.
        Parameters
        ----------
          fval : float
            The initial cost of the optimization problem
        """
        self._fval_init = fval


class IPianoSolverLog(IPianoBaseSolver):
    """Logbarrier optimization with iPiano.
    This Class performs inertial proximal optimization reconstruction
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
          Log regularization weight
    """

    def __init__(
        self,
        par,
        ipiano_par,
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
            ipiano_par,
            queue,
            # tau,
            fval,
            prg,
            coils,
            model,
            **kwargs
        )

        self._op = linop[0]
        self._grad_op = linop[1]

    def _setupVariables(self, x, data):
        data = clarray.to_device(self._queue[0], data.astype(self._DTYPE))

        step_in = {}
        step_out = {}
        tmp_results = {}

        step_in["x"] = clarray.to_device(self._queue[0], x)
        step_in["xold"] = clarray.to_device(self._queue[0], x)
        step_in["xk"] = step_in["x"].copy()

        step_out["x"] = clarray.zeros_like(step_in["x"])

        tmp_results["gradFx"] = step_in["x"].copy()
        tmp_results["DADA"] = clarray.zeros_like(step_in["x"])
        tmp_results["DAd"] = clarray.zeros_like(step_in["x"])
        tmp_results["d"] = data.copy()
        tmp_results["Ax"] = clarray.zeros_like(data)

        tmp_results["temp_reg"] = clarray.zeros_like(step_in["x"])
        tmp_results["gradx"] = clarray.zeros(
            self._queue[0], step_in["x"].shape + (4,), dtype=self._DTYPE
        )

        tmp_results["reg_norm"] = clarray.zeros(
            self._queue[0],
            step_in["x"].shape + (2,),
            dtype=self._DTYPE_real,
        )
        tmp_results["reg"] = clarray.zeros(
            self._queue[0], step_in["x"].shape, dtype=self._DTYPE_real
        )
        return (step_out, tmp_results, step_in, data)

    def update_f(self, outp, inp, par, idx=0, idxq=0, wait_for=None):
        """Forward update of the gradient f(x) function in the iPiano Algorithm.
        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update gradient function f
          inp : list(PyOpenCL.Array)
            The values for calculate f
                DADA    -> the adjointness of calcuated gradient
                Ad      -> the adjointness of data
                x       -> the actual parameters
                x_k     -> linearisation point
                reg     -> regularization parameter
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for
        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_ipiano_log_grad_f(
            self._queue[4 * idx + idxq],
            self._kernelsize,
            None,
            outp.data,
            inp[0].data,
            inp[1].data,
            inp[2].data,
            inp[3].data,
            inp[4].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self.min_const[idx].data,
            self.max_const[idx].data,
            self.real_const[idx].data,
            np.int32(self.unknowns),
            wait_for=(
                inp[0].events + inp[1].events + inp[2].events + inp[3].events + wait_for
            ),
        )

    def _update_f(self, tmp_results, step_in):
        tmp_results["gradFx"].add_event(
            self.update_f(
                outp=tmp_results["gradFx"],
                inp=(
                    tmp_results["DADA"],
                    tmp_results["DAd"],
                    step_in["x"],
                    step_in["xk"],
                    tmp_results["reg"],
                ),
                par=(self.delta, self.lambd),
            )
        )

    def _preUpdate(self, tmp_results, step_in, i):
        tmp_results["Ax"].add_event(
            self._op.fwd(tmp_results["Ax"], [step_in["x"], self._coils, self.modelgrad])
        )
        tmp_results["DADA"].add_event(
            self._op.adj(
                tmp_results["DADA"],
                [tmp_results["Ax"], self._coils, self.modelgrad, self._grad_op.ratio],
                wait_for=tmp_results["Ax"].events,
            )
        )
        # Ändert sich net jedesmal vor for
        tmp_results["DAd"].add_event(
            self._op.adj(
                tmp_results["DAd"],
                [tmp_results["d"], self._coils, self.modelgrad, self._grad_op.ratio],
            )
        )

        tmp_results["gradx"].add_event(
            self._grad_op.fwd(tmp_results["gradx"], step_in["x"])
        )

        tmp_results["temp_reg"].add_event(
            self._grad_op.adj(tmp_results["temp_reg"], tmp_results["gradx"])
        )

        # Calc GPU
        self.normkernl(tmp_results["reg_norm"], tmp_results["gradx"])
        self.abskernl(tmp_results["reg"],  tmp_results["reg_norm"])
        tmp_results["reg"] = -self.lambd * (tmp_results["temp_reg"] / (1 + tmp_results["reg"]))
        
        # TODO: remove test on cpu!!!
        # if not np.mod(i, 100):
            
        #     # Test CPU
        #     grad_x = tmp_results["gradx"].get()
        #     grad_x = np.linalg.norm(grad_x, axis=-1) ** 2

        #     grad_x = clarray.to_device(self._queue[0], grad_x)

        #     # Calc
        #     test_result = -self.lambd * (tmp_results["temp_reg"] / (1 + grad_x))

        #     reg_test = np.sum(np.abs(tmp_results["reg"].get()))
        #     test_test = np.sum(np.abs(test_result.get()))
        #     diff_test = np.sum(np.abs(tmp_results["reg"].get() - test_result.get()))
        #     print(
        #         "Reg: {:.9E} , Test {:.9E}, Diff: {:.9E}".format(
        #             reg_test, test_test, diff_test
        #         )
        #     )

    def _calcStepsize(self, x_shape, data_shape, iterations=50):
        """Rescale the step size"""

        x_temp = np.random.randn(*(x_shape)).astype(
            self._DTYPE_real
        ) + 1j * np.random.randn(*(x_shape)).astype(self._DTYPE_real)
        x = clarray.to_device(self._queue[0], x_temp)
        x_old = clarray.to_device(self._queue[0], x_temp)
        data_temp = np.random.randn(*(data_shape)).astype(
            self._DTYPE_real
        ) + 1j * np.random.randn(*(data_shape)).astype(self._DTYPE_real)
        x1 = clarray.to_device(self._queue[0], data_temp)
        L = 0
        for _ in range(iterations):

            x_norm = self._DTYPE_real(np.linalg.norm(x.get()))
            x = x / x_norm

            x_old = x
            self._op.fwd(
                out=x1,
                inp=[x_old, self._coils, self.modelgrad],
                wait_for=x.events,
            ).wait()
            self._op.adj(
                x,
                [x1, self._coils, self.modelgrad],
                wait_for=x1.events,
            ).wait()

        # Norm forward operator, Norm Gradient,
        L = np.maximum(
            L, np.abs(clarray.vdot(x, x_old).get()) + 8 * self.lambd + 1 / self.delta
        )

        L = self._DTYPE_real(L)
        self.alpha = 2 * (1 - self.beta) / L
        print("Step Size alpha: %s, beta: %s, L: %s " % (self.alpha, self.beta, L))

    def update(self, outp, inp, par, idx=0, idxq=0, wait_for=None):
        """Forward update of the x values in the iPiano Algorithm.
        Parameters
        ----------
          outp : PyOpenCL.Array
            The result of the update gradient function f
          inp : list(PyOpenCL.Array)
            The values for calculate f
                x       -> the actual parameters
                x_(n-1) -> old x
                df(x)   -> regularization parameter
          par : list
            List of necessary parameters for the update
          idx : int
            Index of the device to use
          idxq : int
            Index of the queue to use
          wait_for : list of PyOpenCL.Events, None
            A optional list for PyOpenCL.Events to wait for
        Returns
        -------
            PyOpenCL.Event:
                A PyOpenCL.Event to wait for.
        """
        if wait_for is None:
            wait_for = []
        return self._prg[idx].update_ipiano_log_fwd(
            self._queue[4 * idx + idxq],
            self._kernelsize,
            None,
            outp.data,
            inp[0].data,
            inp[1].data,
            inp[2].data,
            self._DTYPE_real(par[0]),
            self._DTYPE_real(par[1]),
            self.min_const[idx].data,
            self.max_const[idx].data,
            self.real_const[idx].data,
            np.int32(self.unknowns),
            wait_for=(inp[0].events + inp[1].events + inp[2].events + wait_for),
        )

    def _update(self, step_out, tmp_results, step_in):
        return self.update(
            outp=step_out["x"],
            inp=(step_in["x"], step_in["xold"], tmp_results["gradFx"]),
            par=(self.alpha, self.beta),
        )

    def _postUpdate(self, step_out, tmp_results, step_in):
        del step_in["xold"]
        step_in["xold"] = step_in["x"].copy()
        step_in["x"] = step_out["x"].copy()

    def _calcResidual(self, step_out, tmp_results, step_in, data):

        f_new = clarray.vdot(tmp_results["DADA"], tmp_results["DAd"]) + clarray.sum(
            self.lambd
            * clmath.log(1 + clarray.vdot(tmp_results["gradx"], tmp_results["gradx"]))
        )
        
        # TODO: datacosts calculate or get from outside
        datacost = 0 # self._fval_init
        # self._FT.FFT(b, clarray.to_device(
        #       self._queue[0], (self._step_val[:, None, ...] *
        #          self.par["C"]))).wait()
        # b = b.get()
        #datacost = 2 * np.linalg.norm(data - b) ** 2
        L2Cost = np.linalg.norm(step_out["x"].get()) / (2.0 * self.delta)
        regcost = self.lambd * np.sum(
            np.abs(clmath.log(1 + clarray.vdot(tmp_results["gradx"], tmp_results["gradx"])).get())
        )
        costs = datacost + L2Cost + regcost
        return costs, f_new.get()
