# CutQC
CutQC is the backend codes for the paper [CutQC: using small quantum computers for large quantum circuit evaluations](https://dl.acm.org/doi/10.1145/3445814.3446758).
CutQC cuts a large quantum circuits into smaller subcircuits and run on small quantum computers.
By combining classical and quantum computation, CutQC significantly expands the computational reach beyond either platform alone.

## Important note:
There are currently no fault tolerant quantum computers available.
As a result, the perfect fidelity toolchain of CutQC has to rely on classical simulators.
Therefore, using CutQC nowadays will NOT provide better performance than purely classical simulations.
However, with the rapid development of the various hardware vendors,
CutQC is expected to achieve the advantage discussed in the paper over either quantum or classical platforms.

This code repo hence provides two CutQC backends:
1. Using classical simulators as the ``QPU'' backend.
2. Using random number generator as the ``QPU'' backend.
Use this mode if you are just interested in the runtime performance of CutQC.

## Latest Developments
- Added GPU support

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
pip install numpy qiskit matplotlib pydot scipy tqdm pylatexenc scikit-learn tensorflow networkx
```

## Example Code
For an example, run:
```
python example.py
```
This runs an example 16-qubit supremacy circuit.
The output qubits are in a scrambled order based on the subcircuit post-processing sequence.
A function that converts an arbitrary state of interest to the original order will be added.

## Citing CutQC
If you use CutQC in your work, we would appreciate it if you cite our paper:

Tang, Wei, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi. "CutQC: using small quantum computers for large quantum circuit evaluations." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 473-486. 2021.

## Questions
Please reach out to Wei Tang (weit@princeton.edu) for any questions and clarifications.

## TODO
- [ ] Qubit reorder function