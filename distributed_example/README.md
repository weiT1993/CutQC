## How to Use CutQC to Efficiently Perform Subcircuit Reconstruction

This directory contains a simple example of subcircuit reconstruction on an adder circuit, along with a notebook that provides more details on how the environment should be set up.


## Cutting and Evaluation 

Before reconstruction can be performed, any given circuit must first be cut and evaluated, and the results saved to a `.pkl` file. This can be done using `cutqc.save_cutqc_obj(filename)` as seen in the `cut_and_eval.py` file.

Run `cut_and_eval.py` using the corresponding Slurm script. In other words:

    sbatch cut_and_eval.slurm

The output should be simular to the provided slurm output file above.

## Reconstruction

Once the circuit is cut and the results are computed, you can run parallel reconstruction by calling the slurm script:

    sbatch dist_driver.slurm