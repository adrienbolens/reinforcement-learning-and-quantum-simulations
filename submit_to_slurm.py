import sys
from subprocess import run
from pathlib import Path
import json
import re
from parameters import parameters
import datetime
from database import add_to_database

profile = False
#  profile = False

if len(sys.argv) < 2:
    print("Please specifiy the name of the .py file to execute.")
    sys.exit()


file_name = sys.argv[1]
if file_name[-3:] == '.py':
    file_name = file_name[:-3]


if len(sys.argv) < 3:
    n_arrays = 1
else:
    n_arrays = int(sys.argv[2])


#  output = Path('.') / 'output'
output = Path('/data3/bolensadrien/output')
run(['mkdir', '-p', output])


def largest_existing_job_index():
    dirs_starting_with_a_number = [
        d.name for d in output.iterdir() if d.is_dir() and
        d.name[:1].isdigit()
    ]
    if not dirs_starting_with_a_number:
        return None
    return max(int(re.search(r"\d+", dir_str).group()) for dir_str in
               dirs_starting_with_a_number)


def submit_to_slurm(params):
    job_index = (largest_existing_job_index() or 0) + 1
    job_name = f'{job_index}_{file_name}'

    job_path = output / job_name
    job_path = job_path.absolute()
    run(['mkdir', '-p', job_path])

    info_dic = {
        'n_submitted_tasks': n_arrays,
        'parameters': params
    }

    with (job_path / "info.json").open('w') as f:
        json.dump(info_dic, f, indent=2)
    print("info.json written.")

    options = {
        "array": f"1-{n_arrays}",
        #  "partition": "long",
        "partition": "medium",
        #  "partition": "short",
        "job-name": f'{job_index}_{file_name}',
        #  "time": "4-00:00:00",
        "time": "12:00:00",
        #  "time": "2:00:00",
        #  "mail-type": "END",
        #  "mem-per-cpu": "50000",
        "mem-per-cpu": "10000",
        "o": job_path / "slurm-%a.out"
    }

    options_str = ""
    for name, value in options.items():
        if len(name) == 1:
            options_str += f"#SBATCH -{name} {value}\n"
        else:
            options_str += f"#SBATCH --{name}={value}\n"

    python_files = ['q_learning', 'environments', 'systems', 'discrete']
    python_str = ""
    for name in python_files:
        python_str += f"cp $(pwd)/{name}.py $WORK_DIR\n"

    if profile:
        run_command = (f'python -m cProfile -o file.prof {file_name}.py '
                       f'$SLURM_ARRAY_TASK_ID')
    else:
        run_command = f"python {file_name}.py $SLURM_ARRAY_TASK_ID"

    job_script_slurm = (
        "#!/bin/bash\n\n"
        f"{options_str.strip()}\n\n"
        "WORK_DIR=/scratch/$USER/$SLURM_ARRAY_JOB_ID'_'$SLURM_ARRAY_TASK_ID\n"
        "mkdir -p $WORK_DIR\n"
        #  "cp $(pwd)/*.py $WORK_DIR\n"
        f"{python_str.strip()}\n\n"
        f"cp {job_path / 'info.json'} $WORK_DIR\n"
        "module purge\n"
        "module load intelpython3.2019.0.047\n"
        "cd $WORK_DIR\n"
        #  f"python {file_name}.py $SLURM_ARRAY_TASK_ID\n\n"
        f"{run_command}\n\n"
        f"OUTPUT_DIR={job_path}/array-$SLURM_ARRAY_TASK_ID\n"
        "mkdir -p $OUTPUT_DIR\n"
        f"cp ./results_info.json $OUTPUT_DIR\n"
        "cp ./*.npy $OUTPUT_DIR\n"
        "cp ./*.txt $OUTPUT_DIR\n"
        "cp ./*.prof $OUTPUT_DIR\n"
        "cd\n"
        "rm -rf $WORK_DIR\n"
        "unset WORK_DIR\n"
        "unset OUTPUT_DIR\n"
        "exit 0"
        )

    with (job_path / "job_script_slurm.sh").open('w') as f:
            f.write(job_script_slurm)

    run([
        "sbatch",
        f"{job_path}/job_script_slurm.sh"
    ])

    add_to_database({
        'name': job_name,
        'status': 'running',
        'n_completed_tasks': 0,
        'n_submitted_tasks': n_arrays,
        'submission_date': str(datetime.datetime.now())
    })


if __name__ == '__main__':
    #  for n_sites in range(10, 11):
        #  parameters['n_sites'] = n_sites
        #  submit_to_slurm(parameters)
    #  CAREFULE, TIME SET TO 4-DAYS
    #  parameters['n_sites'] = 11
    #  parameters['time_segment'] = 1.0
    #  submit_to_slurm(parameters)
    #  parameters['time_segment'] = 100.0
    submit_to_slurm(parameters)
