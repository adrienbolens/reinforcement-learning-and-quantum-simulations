#  from pathlib import Path
#  import json
#  import numpy as np
import sys
database_path = '/home/bolensadrien/Documents/RL'
sys.path.insert(0, database_path)
from database import read_database, add_to_database
import re

database = read_database()
for entry in database:
    name = entry['name']
    alg = None
    if re.match(r'\d*_q_learning', name):
        alg = 'q_learning'
    if re.match(r'\d*_deep_q_learning', name):
        alg = 'deep_q_learning'
    entry['algorithm'] = alg
    add_to_database(entry)
