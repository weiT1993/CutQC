# CutQC
A Python package for [CutQC](https://dl.acm.org/doi/10.1145/3445814.3446758)

## Installation
1. Make a Python virtual environment and install required packages:
```
conda create -n cutqc-env python=3
conda deactivate && conda activate cutqc-env
pip install numpy qiskit matplotlib pydot
```
2. CutQC uses the [Gurobi](https://www.gurobi.com) solver. Install Gurobi and obtain a license.
To install Gurobi for Python, follow the [instructions](https://www.gurobi.com/documentation/9.1/quickstart_linux/cs_python_installation_opt.html). Here we copy paste the up-to-date command as of 05/10/2021 for convenience.
```
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi
```
3. Install the latest [Qiskit helper functions](https://github.com/weiT1993/qiskit_helper_functions).
```
pip install .
```
4. Install [Intel oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html). - Add the following to `.bashrc` or `.bash_profile`:
(file location may vary depending on installation)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64
```