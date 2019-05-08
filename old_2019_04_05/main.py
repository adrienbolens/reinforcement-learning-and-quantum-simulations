import numpy as np
import ParallelAverage as pa
import os
#  import sys
#  sys.path.append('/home/bolensadrien/Documents/RL')
import q_learning as ql
#  from q_learning import q_learning, parameters, env


options = {
    'mem-per-cpu': '5000',
    'partition': 'medium',
    'time': '24:00:00'
}


@pa.parallel_average(
    N_runs=60,
    N_tasks=30,
    ignore_cache=False,
    force_caching=False,
    slurm=True,
    **options
)
def q_learning_parallel(n_istates, n_episodes, learning_rate, epsilon_max,
                        epsilon_min, epsilon_decay, **other_params):
    """return an array of final reward for n_istates random initial states"""

    #  for p in params:
    #          print(f'{p}: {eval(p)}')

    run_id = int(os.environ["RUN_ID"])
    print(f'Using run_id**3 {run_id}**3 as seed for exploration')
    list_rewards = []
    for i in range(n_istates):
        if i == 0:
                ql.env.set_initial_state(initial_state='antiferro')
        else:
            seed = i**3
            ql.env.set_initial_state(seed=seed,
                                     initial_state='random_product_state')
        list_rewards.append(
            ql.q_learning(n_episodes, learning_rate, epsilon_max, epsilon_min,
                          epsilon_decay, seed=run_id**2)
        )
    return np.array(list_rewards)


result3 = q_learning_parallel(n_istates=5, **ql.parameters)
