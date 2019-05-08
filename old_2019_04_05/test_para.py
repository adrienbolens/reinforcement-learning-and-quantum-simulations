import ParallelAverage as pa
import time
import numpy as np
import os

@pa.parallel_average(
    N_runs = 40,
    N_tasks = 40,
    ignore_cache=True,
    slurm=True
)
def test_pa():
    run_id = int(os.environ["RUN_ID"])
    time.sleep(50)
    print(f'Using run_id {run_id} as seed')
    np.random.seed(run_id)
    return np.random.randint(10, size=10)


re = test_pa()
