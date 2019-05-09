# MBPQ - Model-Based Parameter Quantification

* Requires [OpenCL](https://www.khronos.org/opencl/) >= 1.2
* Requires [clfft](https://github.com/clMathLibraries/clFFT)
* Requires Cython, pip >= 19, python >= 3.6

The Software is tested on Linux using the latest Nvidia driver (418.56 CUDA Version: 10.1) but should be compatible with older drivers as well as different hardware (AMD). The following Installation Guide is targeted at Ubuntu but should work on any distribution provided the required packages are present (could be differently named).

* It is highly recommended to use an Anaconda environment

Quick Installing Guide:
---------------
First make sure that you have a working OpenCL installation
  - OpenCL is usually shipped with GPU driver (Nvidia/AMD)
  - Install the ocl_icd and the OpenCL-Headers
    ```
    apt-get install ocl_icd* opencl-headers
    ```
Possible restart of system after installing new drivers
  - Build [clinfo](https://github.com/Oblomov/clinfo)
  - Run clinfo in terminal and check for errors

Install clFFT library:
  - Either use the package repository,e.g.:
    ```
    apt-get install libclfft*
    ```
  - Or download a prebuild binary of [clfft](https://github.com/clMathLibraries/clFFT)
    - Please refer to the [clFFT](https://github.com/clMathLibraries/clFFT) docs regarding building
    - If build from source symlink clfft libraries from lib64 to the lib folder and run ``` ldconfig ```

  - Navigate to the root directory of MBPQ and typing
    ```
    pip install .
    ```
    should take care of the other dependencies using PyPI and install the package.

In case OCL > 1.2 is present, e.g. by some CPU driver, and NVidia GPUs needs to be used the flag
PRETENED_OCL 1.2 has to be passed to PyOpenCL during the build process. This
can be done by:
```
./configure.py --cl-pretend-version=1.2
rm -Rf build
python setup.py install
```

## Sample Data

In-vivo datasets used in the original publication (doi: [10.1002/mrm.27502](http://onlinelibrary.wiley.com/doi/10.1002/mrm.27502/full)) can be found at
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1410918.svg)](https://doi.org/10.5281/zenodo.1410918)

Prerequests on the .h5 file:
-------------------------
The toolbox expects a .h5 file with a certain structure.
  - kspace data (assumed to be 5D for VFA) and passed as:
    - real_dat (Scans, Coils, Slices, Projections, Samples)
    - imag_dat (Scans, Coils, Slices, Projections, Samples)

    If radial sampling is used the trajectory is expected to be:
    - real_traj (Scans, Projections, Samples)
    - imag_traj (Scans, Projections, Samples)

    Density compensation is performed internally assuming a simple ramp

    For Cartesian data Projections and Samples are replaced by ky and kx encodings points and no trajectory is needed.

    Data is assumed to be 2D stack-of-stars, i.e. already Fourier transformed along the fully sampled z-direction.

  - flip angle correction (optional) can be passed as:
    - fa_corr (Scans, Coils, Slices, dimY, dimX)
  - The image dimension for the full dataset is passed as attribute consiting of:
    - image_dimensions = (dimX, dimY, NSlice)
  - Parameters specific to the used model (e.g. TR or flip angle) need to be set as attributes e.g.:
    - TR = 5.38
    - flip_angle(s) = (1,3,5,7,9,11,13,15,17,19)

    The specific structure is determined according to the Model file.

  If predetermined coil sensitivity maps are available they can be passed as complex dataset, which can saved bedirectly using Python. Matlab users would need to write/use low level hdf5 functions to save a complex array to .h5 file. Coil sensitivities are assumed to have the same number of slices as the original volume and are intesity normalized. The corresponding .h5 entry is named "Coils". If no "Coils" parameter is found or the number of "Coil" slices is less than the number of reconstructed slices, the coil sensitivities are determined using the [NLINV](https://doi.org/10.1002/mrm.21691) algorithm and saved into the file.

Running the reconstruction:
-------------------------
First, start an ipcluster for speeding up the coil sensitivity estimation:
```
ipcluster start -n N
```
where N amounts to the number of processe to be used. If -n N is ommited,
as many processes as number of CPU cores available are started.

Reconstruction of the parameter maps can be started either using the terminal by typing:
```
mbpq
```
or from python by:
```
import mbpq
mbpq.mbpq()
```
A list of accepted flags can be printed using
```
mbpq -h
```

or by fewing the documentation of mbpq.mbpq in python.

If reconstructing fewer slices from the volume than acquired, slices will be picked symmetrically from the center of the volume. E.g. reconstructing only a single slice will reconstruct the center slice of the volume.

The config file (\*.ini):
-------------------------
A default config file will be generated if no path to a config file is passed as an argument or if no default.ini file is present in the current working directory. After the initial generation the values can be altered to influence regularization or the number of iterations. Seperate values for TV and TGV regularization can be used.

  - max_iters: Maximum primal-dual (PD) iterations
  - start_iters: PD iterations in the first Gauss-Newton step
  - max_gn_it: Maximum number of Gauss Newton iterations
  - lambd: Data weighting
  - gamma: TGV weighting
  - delta: L2-step-penalty weighting (inversely weighted)
  - omega: optional H1 regularization (should be set to 0 if no H1 is used)
  - display_iterations: Flag for displaying grafical output
  - gamma_min: Minimum TGV weighting
  - delta_max: Maximum L2-step-penalty weighting
  - omega_min: Minimum H1 weighting (should be set to 0 if no H1 is used)
  - tol: relative convergence toleranze for PD and Gauss-Newton iterations
  - stag: optional stagnation detection between successive PD steps
  - delta_inc: Increase factor for delta after each GN step
  - gamma_dec: Decrease factor for gamma after each GN step
  - omega_dec: Decrease factor for omega after each GN step

Limitations and known Issues:
-------------------------
Currently runs only on GPUs due to having only basic CPU support for the clfft.

Citation:
----------
Please cite "Oliver Maier, Matthias Schloegl, Kristian Bredies, and Rudolf Stollberger; 3D Model-Based Parameter Quantification on Resource Constrained Hardware using Double-Buffering. Proceedings of the 27th meeting of the ISMRM, 2019, Montreal, Canada" if using the software or parts of it, specifically the PyOpenCL based NUFFT, in your work.

Older Releases:
---------
You can find the code for

Maier O, Schoormans J,Schloegl M, Strijkers GJ, Lesch A, Benkert T, Block T, Coolen BF, Bredies K, Stollberger R <br>
  __Rapid T1 quantification from high
resolution 3D data with model‐based reconstruction.__<br>
  _Magn Reson Med._, 2018; 00:1–16<br>
  doi: [10.1002/mrm.27502](http://onlinelibrary.wiley.com/doi/10.1002/mrm.27502/full)

at [v1.0](https://github.com/IMTtugraz/MBPQ/tree/v1.0)
