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

## Installation
1. Make a Python 3.10 virtual environment:
2. CutQC uses the [Gurobi](https://www.gurobi.com) solver. Obtain and install a Gurobi license.
Follow the [instructions](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).
3. Install required packages:
```
pip install -r requirements.txt
```

## Example Code
For an example, run:
```
python example.py
```
This runs an example 16-qubit supremacy circuit.
The output qubits are in a scrambled order based on the subcircuit post-processing sequence.

## Citing CutQC
If you use CutQC in your work, we would appreciate it if you cite our paper:

Tang, Wei, Teague Tomesh, Martin Suchara, Jeffrey Larson, and Margaret Martonosi. "CutQC: using small quantum computers for large quantum circuit evaluations." In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, pp. 473-486. 2021.

## Contact Us
Please open an issue here. Please reach out to [Wei Tang](https://www.linkedin.com/in/weitang39/).