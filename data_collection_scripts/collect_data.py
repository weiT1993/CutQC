import os
import subprocess

# Define the sets of variables
variable_sets = [
    {'circuit_size': 22, 'max_subcircuit_width': 20, 'circuit_type': 'adder'},
    {'circuit_size': 24, 'max_subcircuit_width': 20, 'circuit_type': 'adder'},
    {'circuit_size': 26, 'max_subcircuit_width': 20, 'circuit_type': 'adder'},
    {'circuit_size': 28, 'max_subcircuit_width': 20, 'circuit_type': 'adder'},
    {'circuit_size': 30, 'max_subcircuit_width': 20, 'circuit_type': 'adder'}
]

# Read the SLURM script template
with open('run.slurm', 'r') as file:
    slurm_template = file.read()

# Directory to store generated SLURM scripts
slurm_scripts_dir = 'generated_slurm_scripts'
os.makedirs(slurm_scripts_dir, exist_ok=True)

previous_job_id = None

# Generate and submit SLURM scripts for each set of variables
for i, variables in enumerate(variable_sets):
    slurm_script_content = slurm_template.format(**variables)
    slurm_script_path = os.path.join(slurm_scripts_dir, f'slurm_script_{i}.slurm')

    # Write the generated SLURM script to a file
    with open(slurm_script_path, 'w') as slurm_script_file:
        slurm_script_file.write(slurm_script_content)

    # Construct the sbatch command
    sbatch_command = ['sbatch']
    if previous_job_id:
        sbatch_command.extend(['--dependency=afterok:' + previous_job_id])
    sbatch_command.append(slurm_script_path)

    # Submit the SLURM script using sbatch and capture the job ID
    result = subprocess.run(sbatch_command, capture_output=True, text=True)
    output = result.stdout.strip()

    # Extract job ID from sbatch output
    job_id = output.split()[-1]
    previous_job_id = job_id

print('All SLURM scripts have been submitted.')