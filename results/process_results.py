from pathlib import Path
import json
import numpy as np
from subprocess import run
from database import clean_database, read_database, add_to_database
from info_database import update_info_database
from datetime import datetime

output = Path('/data3/bolensadrien/output')

clean_database()

database_entries = read_database()
yes_all = False
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
                    #  prof_file = result_dir / 'array-1' / 'file.prof'
                    #  if prof_file.is_file():
                    #      run(['cp', prof_file, result_dir])
                    run('rm -r ' + str(result_dir / 'array-*'), shell=True)
                    run('rm ' + str(result_dir / 'slurm-*.out'), shell=True)
                    run(['rm', result_dir / 'job_script_slurm.sh'])
                    entry['raw_files_exist'] = False
                    add_to_database(entry)
                elif answer == "no":
                    pass
                else:
                    print("Please enter yes or no.")
        if status == 'processed':
            continue

    array_dirs = [
        d.name for d in result_dir.iterdir() if
        d.is_dir() and
        d.name[:6] == 'array-' and
        d.name[6:].isdigit()
    ]

    for d in array_dirs:
        if not (result_dir / d / 'rewards.npy').is_file():
            print(f'Deleting {d}.')
            run(['rm', '-r', result_dir / d])
            array_dirs.remove(d)

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
            print(f"{name} -- hours elapsed: {hours_elapsed}")
            if hours_elapsed > 4 * 24:
                print(f'\n{name} has been running for more than four days '
                      f'({hours_elapsed} hours) -> overtime.')
                status = 'overtime'

    print(f"{name} -- status: {status}")
    if status == 'to_be_processed' or status == 'overtime':
        print('\n', '{:=^75}'.format(f'Processing results in {name}.'), '\n')
        if status == 'overtime':
            answer = input(f"status is {status}, should the results be "
                           "processed? Any answer other than `yes` will cancel"
                           " the process (no change will be made).")
            if answer != "yes":
                print(f'Canceling processing of results in {name}.')
                continue
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
        with_history = False
        verify_argmax_q = params.get('verify_argmax_q', False)
        if params.get('subclass', None) == "WithReplayMemory":
            import pandas as pd
            with_history = True
            history_file_path = result_dir / array_dirs[0] / 'NN_history.csv'
            if history_file_path.is_file():
                df = pd.read_csv(result_dir / array_dirs[0] / 'NN_history.csv')
                metrics = df.columns
                n_metrics, len_history = len(metrics), len(df)
            else:
                with_history = False
            #  n_metrics, len_history = (np.load(result_dir / array_dirs[0] /
            #                                    'NN_history.npy').shape)
        if params['system_class'] == 'LongRangeIsing':
            with open(result_dir / array_dirs[0] / 'results_info.json') as f:
                results_info = json.load(f)
            info['initial_reward'] = results_info['initial_reward']
            if 'rerun_path' in results_info:
                info['rerun_path'] = results_info['rerun_path']
        n_episodes = params['n_episodes']

        n_episodes_total = np.load(result_dir / array_dirs[0] /
                                   'rewards.npy').shape[0]
        reward_array = np.empty((n_arrays, n_episodes_total), dtype=np.float32)
        if with_history:
            history_array = np.empty((n_metrics,  n_arrays, len_history))
        q_chosen_array = np.array([])
        q_disc_max_array = np.array([])
        q_disc_min_array = np.array([])
        total_hours_average = 0

        for i, a_dir in enumerate(array_dirs):
            a_path = result_dir / a_dir
            reward_array[i, :] = np.load(a_path / 'rewards.npy')
            with open(result_dir / a_dir / 'results_info.json') as f:
                results_info = json.load(f)
                if 'total_time' not in results_info.keys():
                    print("No 'total_time' key in results_info of {a_dir}.")
                total_hours_average += results_info.get('total_time', -7*86400)
            if with_history:
                history_df = pd.read_csv(a_path / 'NN_history.csv')
                history_array[:, i, :] = history_df.values.T
                #  history_array[:, i, :] = np.load(a_path / 'NN_history.npy')
            if verify_argmax_q:
                q_chosen_array = (
                    np.append(q_chosen_array,
                              np.load(a_path / 'list_q_max_chosen.npy'))
                )
                q_disc_max, q_disc_min = np.load(a_path /
                                                 'list_q_max_discretized.npy')
                q_disc_max_array = np.append(q_disc_max_array, q_disc_max)
                q_disc_min_array = np.append(q_disc_min_array, q_disc_min)
        total_hours_average /= n_arrays * 3600

        with open(result_dir / 'rewards.npy', 'wb') as f:
            np.save(f, reward_array)

        if with_history:
            with open(result_dir / 'NN_histories.npy', 'wb') as f:
                np.save(f, history_array)

        if verify_argmax_q:
            with open(result_dir / 'q_arrays_comparison.npy', 'wb') as f:
                np.save(f, np.stack([q_chosen_array, q_disc_max_array,
                                    q_disc_min_array]))

        max_final_reward = np.max(reward_array[:, -1])
        max_final_array = np.argmax(reward_array[:, -1])
        max_final_array_dir = array_dirs[max_final_array]
        print(
            f'{max_final_array_dir} '
            f'had the best final reward = {max_final_reward}.'
        )

        #  info['max_final_reward'] = max_final_reward
        q_matrix_file = result_dir / f'{max_final_array_dir}/q_matrix.npy'
        #  max_reward_file = result_dir / \
        #      f'{max_final_array_dir}/post_episode_rewards__final.npy'
        final_weights_file = (result_dir / f'{max_final_array_dir}' /
                              'final_weights.npy')

        if q_matrix_file.is_file():
            run(['cp', q_matrix_file, result_dir])
        #  run(['cp', max_reward_file, result_dir])
        if final_weights_file.is_file():
            run(['cp', final_weights_file, result_dir])

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
        prof_file = result_dir / array_dirs[0] / 'file.prof'
        if prof_file.is_file():
            run(['cp', prof_file, result_dir])

        status = 'processed'

        info['n_completed_tasks'] = n_arrays
        info['total_hours_average'] = total_hours_average
        if with_history:
            info['NN_metrics'] = list(metrics)

        with (result_dir / "info.json").open('w') as f:
            json.dump(info, f, indent=2)
        print("info.json written.")

        answer = None
        while answer not in ("yes", "no", "yes all"):
            if not yes_all:
                answer = input("Delete the array folders, the .py files  "
                               "and the slurm .out files? ")
            if answer == "yes all":
                yes_all = True
            if answer == "yes" or yes_all:
                if yes_all:
                    print('---> yes all')
                    answer = "yes all"
                run('rm -r ' + str(result_dir / 'array-*'), shell=True)
                run('rm ' + str(result_dir / 'slurm-*.out'), shell=True)
                run(['rm', result_dir / 'job_script_slurm.sh'])
                run('rm ' + str(result_dir / '*.py'), shell=True)
                raw_files_exist = False
            elif answer == "no":
                raw_files_exist = True
            else:
                print("Please enter 'yes', 'no', or 'yes all'.")

    entry['raw_files_exist'] = raw_files_exist
    entry['status'] = status
    add_to_database(entry)

clean_database()
update_info_database()
