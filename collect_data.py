# Python script to submit SLURM scripts with varying circuit parameters. Tests will execute in series.
# Call collect_data.py to submit SLURM jobs

import os
import subprocess

# Set tests here
variable_sets = [
    {'circuit_size': 12, 'max_subcircuit_width': 10, 'circuit_type': 'supremacy'}
]

with open('slurm/collect_data.slurm', 'r') as file:
    slurm_template = file.read()
slurm_scripts_dir = 'generated_slurm_scripts'
os.makedirs(slurm_scripts_dir, exist_ok=True)

previous_job_id = None

# Generate and submit SLURM scripts for each set of variables
for i, variables in enumerate(variable_sets):
    # Construct the output file name
    output_file = f"{variables['circuit_type']}_{variables['circuit_size']}_{variables['max_subcircuit_width']}.%j.out"
    
    # Add the output directive to the SLURM script content
    slurm_script_content = slurm_template.format(**variables)
    slurm_script_content = slurm_script_content.replace(
        '#SBATCH --output=_output/%x.%j.out',
        f'#SBATCH --output=_output/{output_file}'
    )

    slurm_script_path = os.path.join(slurm_scripts_dir, f'slurm_script_{i}.slurm')

    # Write the generated SLURM script to a file
    with open(slurm_script_path, 'w') as slurm_script_file:
        slurm_script_file.write(slurm_script_content)

    # Make sure tests are executed in series
    sbatch_command = ['sbatch']
    if previous_job_id:
        sbatch_command.extend(['--dependency=afterok:' + previous_job_id])
    sbatch_command.append(slurm_script_path)

    # Submit the SLURM script
    result = subprocess.run(sbatch_command, capture_output=True, text=True)
    output = result.stdout.strip()

    # Extract job ID from sbatch output
    job_id = output.split()[-1]
    previous_job_id = job_id

print('All SLURM scripts have been submitted.')
