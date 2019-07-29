from setuptools import setup, find_packages

setup(name='pyqmri',
      version='0.1',
      description='Model-based parameter quantification using OpenCL and Python',
      url='https://github.com/IMTtugraz/PyQMRI',
      author='Oliver Maier',
      author_email='oliver.maier@tugraz.at',
      license='Apache-2.0',
      package_data={'pyqmri': ['kernels/*.c']},
      include_package_data=True,
      exclude_package_data = {'': ['data*','output*']},
      packages=find_packages(exclude=("output*","data*")),
      setup_requires=["cython"],
      python_requires ='>=3.6',
      install_requires=[
        'cython',
        'pyopencl',
        'numpy',
        'h5py',
        'mako',
        'matplotlib',
        'gpyfft @ git+https://github.com/geggo/gpyfft.git#egg=gpyfft',
        'ipyparallel',
        'pyfftw'],
      entry_points={
        'console_scripts': ['pyqmri = pyqmri.pyqmri:main'],
        },
      zip_safe=False) 
