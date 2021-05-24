# CutQC
CutQC is the backend codes for the paper [CutQC: using small quantum computers for large quantum circuit evaluations](https://dl.acm.org/doi/10.1145/3445814.3446758).
CutQC cuts a large quantum circuits into smaller subcircuits and run on small quantum computers.
By combining classical and quantum computation, CutQC significantly expands the computational reach beyond either platform alone.

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
4. Install [Intel oneAPI](https://software.intel.com/content/www/us/en/develop/tools/oneapi/base-toolkit/download.html).
Add MKL to path (file location may vary depending on installation):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64
```
Note that the installations have only been tested on Linux.
Windows/MacOS may require different setups and support is currently not provided.

## Example Code
For example, use CutQC to cut a 3*5 Supremacy circuit and run on a 10-qubit quantum computer
```
python example.py
```

## Citing CutQC
If you use CutQC in your work, we would appreciate it if you cite our paper:

Tang, Wei, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi. "CutQC: using small quantum computers for large quantum circuit evaluations." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 473-486. 2021.

## Questions
Please reach out to Wei Tang (weit@princeton.edu) for any questions and clarifications.

## Coming soon
- [ ] Multi-node classical post-processing tools for HPC clusters