from pathlib import Path
from create_plot import create_plot
import json
import numpy as np
import subprocess
import sys
database_path = '/home/bolensadrien/Documents/RL'
sys.path.insert(0, database_path)
from database import read_database

database = read_database()
dir_names, algorithm = zip(*[(ent['name'], ent['algorithm']) for ent in
                             database if ent['status'] == 'processed'])

#  output = Path(__file__).parent
output = Path('/data3/bolensadrien/output')

#  replot = True
replot = False
plot_name = 'plot'

system_class_list = []
n_sites_list = []
n_steps_list = []
n_directions_list = []
n_all_list = []
n_one_list = []
algo_list = []

for name in dir_names:
    result_dir = output / name
    with open(result_dir / 'info.json') as f:
        info = json.load(f)
    params = info['parameters']

    if (replot
            #  or params.get('subclass', None) == 'WithReplayMemory'
            or not (result_dir / (plot_name + '.pdf')).is_file()):
        print('\n', f"{f'Producing plot for results in {result_dir}.':-^70}")
        create_plot(output, name)

    system_class_list.append(params['system_class'])
    n_sites_list.append(int(params['n_sites']))
    n_steps_list.append(int(params['n_steps']))
    n_directions_list.append(int(params['n_directions']))
    n_all_list.append(int(params.get('n_allqubit_actions', 0)))
    n_one_list.append(int(params.get('n_oqbgate_parameters', 0)))


algo_sorted, *_, system_class_sorted, dir_names_sorted = zip(*sorted(zip(
    algorithm,
    n_sites_list,
    n_steps_list,
    n_directions_list,
    n_all_list,
    n_one_list,
    system_class_list,
    dir_names
)))


for system_class in np.unique(system_class_list):
    for algo in np.unique(algorithm):
        plot_to_gather = [output / dir_name / (plot_name + '.pdf') for
                          dir_name, sys_class, algorithm in
                          zip(dir_names_sorted, system_class_sorted,
                              algo_sorted) if sys_class == system_class and
                          algorithm == algo]

        subprocess.run(['pdfunite'] + plot_to_gather +
                       [f'{plot_name}_{system_class}_{algo}.pdf'])
