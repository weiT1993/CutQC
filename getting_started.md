## How to Use CutQC to Efficiently Perform Subcircuit Reconstruction

This directory contains a simple example of subcircuit reconstruction on an adder circuit, along with a notebook that provides more details on how the environment should be set up.

## Setting up the environment

First we need to setup a conda environement with the following

    conda create --name cutqc python=3.12
    conda activate cutqc 
    pip install -r requirements.txt
conda config --add channels https://conda.anaconda.org/gurobi
conda install gurobi

pip install numpy qiskit matplotlib pydot scipy tqdm pylatexenc scikit-learn tensorflow networkx torch qiskit-aer psutil
## Running with slurm

When initializing distributed, the worker nodes must have a way to initialize communication with the host node. The following must be set in the slurm file for running the distributed reconstruction to ensure this can happen. 

    # Setup for Multi-node Workload
    export MASTER_PORT=$(get_free_port)  # Get a free Port
    export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) 
    export MASTER_ADDR=$master_addr

    # Sanity Print
    echo "MASTER_ADDR="$MASTER_ADDR
    echo "MASTER_PORT="$MASTER_PORT
    echo "WORLD_SIZE="$WORLD_SIZE


## Cutting and Evaluation 

Before reconstruction can be performed, any given circuit must first be cut and evaluated, and the results saved to a `.pkl` file. This can be done using `cutqc.save_cutqc_obj(filename)` as seen in the `cut_and_eval.py` file.

Run `cut_and_eval.py` using the corresponding Slurm script. In other words:

    sbatch cut_and_eval.slurm

Running the slurm script should produce an output file containing:

    --- Cut --- 
    Set parameter GURO_PAR_SPECIAL
    Set parameter TokenServer to value "license.rc.princeton.edu"
    --- Evaluate ---
    --- Dumping CutQC Object into adder_example.pkl ---
    Completed

## Reconstruction

Once the circuit is cut and the results are computed, you can run parallel reconstruction by calling the slurm script:

    sbatch dist_driver.slurm

Running the slurm script should produce an output file containing:

      MASTER_ADDR=adroit-h11g3
      MASTER_PORT=31179
      WORLD_SIZE=2
      --- Running adder_example.pkl ---
      self.parallel_reconstruction: True
      Worker 1, compute_device: cuda:1
      --- Running adder_example.pkl ---
      self.parallel_reconstruction: True
      Worker 0, compute_device: cuda:0
      subcircuit_entry_length: [32, 32]
      LEN(DATASET): 16
      NUMBER BATCHES: 1
      Compute Time: 1.5637431228533387
      Approximate Error: 1.2621774483536279e-29
      verify took 0.011
      --- Reconstruction Complete ---
      Total Reconstruction Time:	1.5637431228533387
      Approxamate Error:	 1.2621774483536279e-29
      DESTROYING NOW! 1.5637431228533387
      WORKER 1 DYING