from pathlib import Path
import json
from database import read_database

output = Path('/data3/bolensadrien/output')
results = Path('/home/bolensadrien/Documents/RL/results')
database_path = results / 'info_database.json'

if not database_path.exists():
    print(f'No database.json found in {results}.')
    print(f'Creating {database_path}')
    with database_path.open('w') as f:
        json.dump({}, f, indent=2)


def update_info_database(complete_rewrite=True):
    print('Updating `info_database.json`, '
          f'`complete_rewrite={complete_rewrite}`')
    status_database = read_database(only_processed=True)
    if complete_rewrite:
        info_database = {}
    else:
        with database_path.open() as f:
            info_database = json.load(f)
    dir_names = [ent['name'] for ent in status_database]
    for name in dir_names:
        if name in info_database:
            continue
        result_dir = output / name
        with open(result_dir / 'info.json') as f:
            info = json.load(f)
        info_database[name] = info

    with database_path.open() as f:
        with database_path.open('w') as f:
            json.dump(info_database, f, indent=2)


if __name__ == '__main__':
    update_info_database()
