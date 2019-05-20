from subprocess import run
from pathlib import Path
import sys
database_path = '/home/bolensadrien/Documents/RL'
sys.path.insert(0, database_path)
from database import clean_database, read_database
import json

#  output = Path(__file__).parent.resolve()
output = Path('/data3/bolensadrien/output')
destination_folder = 'test_runs'
dest_path = output / destination_folder
run(['mkdir', '-p', dest_path])

clean_database()

database_entries = read_database()
for entry in database_entries:
    name = entry['name']
    result_dir = output / name
    with open(result_dir / 'info.json') as f:
        info = json.load(f)
    #  n_s = info['n_submitted_tasks']
    #  n_c = info.get('n_completed_tasks', 0)
    #  print(f'{name} had {n_s} subbmited tasks and completed {n_c}.')
    #  if n_s != 100:
    #      if n_s != n_c:
    #          raise RuntimeError('Some folders to be move are still '
    #                             'potentially running.')
    #      answer = None
    #      while answer not in ('yes', 'no'):
    #          answer = input(f"Move {name} to the folder {destination_folder}?
    #          ")
    #          if answer == 'yes':
    #              print(f"Moving {name} to {destination_folder}...")
    #              run(['mv', result_dir, dest_path])
    #          if answer == 'no':
    #              pass
    #          else:
    #              print("Enter 'yes' or 'no'.")

    if 'total_time_average' in info.keys():
        info['total_hours_average'] = \
            info['total_time_average'] / info['n_completed_tasks'] / 3600
        del info['total_time_average']

    with (result_dir / "info.json").open('w') as f:
        json.dump(info, f, indent=2)
    print("info.json written.")

clean_database()
