from pathlib import Path
import json
import numpy as np
from subprocess import run
import sys
database_path = '/home/bolensadrien/Documents/RL'
sys.path.insert(0, database_path)
from database import clean_database, read_database, add_to_database
from datetime import datetime

output = Path(__file__).parent.resolve()

clean_database()

database_entries = read_database()
for entry in database_entries:
    status = entry['status']
    raw_files_exist = entry.get('raw_files_exist', False)
    entry['raw_files_exist'] = raw_files_exist
    name = entry['name']
    result_dir = output / name

    if status == 'processed' or status == 'overtime':
        if raw_files_exist:
            answer = None
            while answer not in ("yes", "no"):
                print(f"{name} still contains the raw results files.")
                answer = input(f"Delete the arrays folder and slurm files? ")
                if answer == "yes":
                    run('rm -r ' + str(result_dir / 'array-*'), shell=True)
                    run('rm ' + str(result_dir / 'slurm-*.out'), shell=True)
                    run(['rm', result_dir / 'job_script_slurm.sh'])
                    entry['raw_files_exist'] = False
                    add_to_database(entry)
                elif answer == "no":
                    pass
                else:
                    print("Please enter yes or no.")
        continue

    array_dirs = [
        d.name for d in result_dir.iterdir() if
        d.is_dir() and
        d.name[:6] == 'array-' and
        d.name[6:].isdigit()
    ]
    n_arrays = len(array_dirs)
    entry['n_completed_tasks'] = n_arrays

    if status == 'running':
        if n_arrays == entry['n_submitted_tasks']:
            status = 'to_be_processed'
        else:
            submission_date = datetime.strptime(
                entry['submission_date'], "%Y-%m-%d %H:%M:%S.%f"
            )
            hours_elapsed = int(
                (datetime.now() - submission_date).total_seconds()
            ) // 3600
            if hours_elapsed > 48:
                print(f'\n{name} has been running for more than two days '
                      f'({hours_elapsed} hours) -> overtime.')
                status = 'overtime'

    if status == 'to_be_processed' or status == 'overtime':
        print('\n', '{:=^75}'.format(f'Processing results in {name}.'), '\n')
        if n_arrays == 0:
            print(f"No results in {name} with status {status}.")
            answer = None
            while answer not in ("yes", "no"):
                answer = input(f"Delete the {name} folder?  ")
                if answer == "yes":
                    print(f"Deleting {name}")
                    run(['rm', '-r', result_dir])
                elif answer == "no":
                    pass
                else:
                    print("Please enter yes or no.")
            continue
        print(f"{name} contains {n_arrays} arrays (status = {status}).")

        with open(result_dir / 'info.json') as f:
            info = json.load(f)
        params = info['parameters']
        if params['system_class'] == 'LongRangeIsing':
            with open(result_dir / array_dirs[0] / 'results_info.json') as f:
                results_info = json.load(f)
            info['initial_reward'] = results_info['initial_reward']
        n_episodes = params['n_episodes']

        reward_array = np.empty((n_arrays, n_episodes), dtype=np.float32)
        total_hours_average = 0
        for i, a_dir in enumerate(array_dirs):
            reward_array[i, :] = np.load(result_dir / f'{a_dir}/rewards.npy')
            with open(result_dir / a_dir / 'results_info.json') as f:
                results_info = json.load(f)
                if 'total_time' not in results_info.keys():
                    print("No 'total_time' key in results_info of {a_dir}.")
                total_hours_average += results_info.get('total_time', -7*86400)
        total_hours_average /= n_arrays * 3600

        with open(result_dir / 'rewards.npy', 'wb') as f:
            np.save(f, reward_array)

        max_final_reward = np.max(reward_array[:, -1])
        max_final_array = np.argmax(reward_array[:, -1])
        max_final_array_dir = array_dirs[max_final_array]
        print(
            f'{max_final_array_dir} '
            f'had the best final reward = {max_final_reward}.'
        )

        #  info['max_final_reward'] = max_final_reward
        q_matrix_file = result_dir / f'{max_final_array_dir}/q_matrix.npy'
        max_reward_file = result_dir / \
            f'{max_final_array_dir}/post_episode_rewards__final.npy'
        run(['cp', q_matrix_file, result_dir])
        run(['cp', max_reward_file, result_dir])

        max_reward = np.max(reward_array)
        max_array, max_episode = np.unravel_index(np.argmax(reward_array),
                                                  reward_array.shape)
        max_array_dir = array_dirs[max_array]
        print(
            f'{max_array_dir} '
            f'had the best overall reward = {max_reward} '
            f"at episode {max_episode}/{params['n_episodes']}."
        )

        #  info['max_reward'] = max_reward
        best_sequence_file = \
            result_dir / f'{max_array_dir}/best_gate_sequence.txt'
        best_reward_file = result_dir / \
            f'{max_array_dir}/post_episode_rewards__best.npy'
        trotter_reward_file = result_dir / \
            f'{max_array_dir}/post_episode_rewards__trotter.npy'
        run(['cp', best_sequence_file, result_dir])
        run(['cp', best_reward_file, result_dir])
        run(['cp', trotter_reward_file, result_dir])

        status = 'processed'

        answer = None
        print('\n')
        while answer not in ("yes", "no"):
            answer = input(f"Delete the array folders and the slurm files? ")
            if answer == "yes":
                run('rm -r ' + str(result_dir / 'array-*'), shell=True)
                run('rm ' + str(result_dir / 'slurm-*.out'), shell=True)
                run(['rm', result_dir / 'job_script_slurm.sh'])
                raw_files_exist = True
            elif answer == "no":
                raw_files_exist = True
                pass
            else:
                print("Please enter yes or no.")
        print('\n')

        info['n_completed_tasks'] = n_arrays
        info['total_hours_average'] = total_hours_average

        with (result_dir / "info.json").open('w') as f:
            json.dump(info, f, indent=2)
        print("info.json written.")
        status = 'processed'

    entry['raw_files_exist'] = raw_files_exist
    entry['status'] = status
    add_to_database(entry)

clean_database()
