from pathlib import Path
import json

output = Path('/data3/bolensadrien/output')
results = Path('/home/bolensadrien/Documents/RL/results')
database_path = results / 'status_database.json'

if not database_path.exists():
    print(f'No database.json found in {results}.')
    answer = None
    while answer not in ("yes", "no"):
        answer = input(f"Create one? ")
        if answer == "yes":
            with database_path.open('w') as f:
                json.dump([], f, indent=2)
        elif answer == "no":
            pass
        else:
            print("Please enter yes or no.")


#  def change_dir(dir_path):
#      global output, database_path
#      output = Path(dir_path)
#      database_path = output / 'database.json'


def read_database(only_processed=False):
    with database_path.open() as f:
        entries = json.load(f)
    if only_processed:
        entries = [entry for entry in entries if
                   entry['status'] == 'processed']
    return entries


def add_to_database(entry):
    print(f"{'Adding ' + entry['name'] + ' to the database.':-^60}")
    with database_path.open('r+') as f:
        entries = json.load(f)
        entries = [ent for ent in entries if ent['name'] != entry['name']]
        entries.append(entry)
        f.seek(0)
        json.dump(entries, f, indent=2)
        f.truncate()


def clean_database(add_folders=False):
    """Remove elements which are not in the output directory from the
    database."""
    result_names = [
        d.name for d in output.iterdir() if
        d.is_dir() and d.name[:1].isdigit()
    ]
    print(f"\nCleaning {database_path}")
    with database_path.open('r+') as f:
        entries = json.load(f)
        entries = [entry for entry in entries if entry['name'] in result_names]
        results_not_in_database \
            = set(result_names) - {e['name'] for e in entries}
        if len(results_not_in_database):
            print('The following result folders are not in the database:')
            print('\n'.join(results_not_in_database))
            answer = None
            while answer not in ('yes', 'no'):
                answer = input("Add them all with default values? ")
                if answer not in ('yes', 'no'):
                    print("Please enter yes or no.")
                    continue
                add_folders = (answer == 'yes')

        if add_folders:
            for name in results_not_in_database:
                print(f"{'Adding ' + name + ' to the database.':-^60}")
                entry = {
                    "name": name,
                    "status": "to_be_processed",
                    "n_completed_tasks": 0,
                    "n_submitted_tasks": 100,
                    "submission_date": "2019-04-01 00:00:00.000000"
                }
                entries.append(entry)
        f.seek(0)
        json.dump(entries, f, indent=2)
        f.truncate()


if __name__ == '__main__':
    clean_database()
