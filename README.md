# CutQC
CutQC is the backend codes for the paper [CutQC: using small quantum computers for large quantum circuit evaluations](https://dl.acm.org/doi/10.1145/3445814.3446758).
CutQC cuts a large quantum circuits into smaller subcircuits and run on small quantum computers.
By combining classical and quantum computation, CutQC significantly expands the computational reach beyond either platform alone.

## Installation
1. Make a Python virtual environment:
```
conda create -n cutqc python=3.8
conda deactivate && conda activate cutqc
```
2. CutQC uses the [Gurobi](https://www.gurobi.com) solver. Install Gurobi and obtain a license.
To install Gurobi for Python, follow the [instructions](https://www.gurobi.com/documentation/9.1/quickstart_linux/cs_python_installation_opt.html). Here we copy paste the up-to-date command as of 05/10/2021 for convenience.
```
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi
```
3. Install required packages:
```
pip install numpy qiskit matplotlib pydot scipy tqdm pylatexenc scikit-learn tensorflow
```
Download and install the latest [Qiskit helper functions](https://github.com/weiT1993/qiskit_helper_functions).
```
pip install . --use-feature=in-tree-build
```

## Example Code
For an example, run:
```
python example.py
```

## Citing CutQC
If you use CutQC in your work, we would appreciate it if you cite our paper:

Tang, Wei, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi. "CutQC: using small quantum computers for large quantum circuit evaluations." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 473-486. 2021.

## Questions
Please reach out to Wei Tang (weit@princeton.edu) for any questions and clarifications.

## Latest
- Added GPU support