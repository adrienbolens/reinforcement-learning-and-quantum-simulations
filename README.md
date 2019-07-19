
# Q-learning applied to quantum digitial simulations.

This repository contains the files that I used for my current research project.

In quantum computing, a [quantum gate](https://en.wikipedia.org/wiki/Quantum_logic_gate) is a basic operation (mathematically, a unitary matrix) applied to a small number of quantum bits (qubits).

One promising short-term application of quantum computers is to simulate quantum many-body dynamics [3, 4]. However, it is a challenge to optimize the algorithms according to the resources available today.

A quantum computer has at its disposal a _universal set of gates_.
Namely, any unitary can be approximated to any desired accuracy with the right sequence of quantum gates.
But how can we find this sequence of gates? 
In some cases, there exist some systematic methods (e.g. the Trotter-Suzuki decomposition), but they usually fail on large time scales and require a unrealistically large amount of gates.

## A quantum compiler:
The goal of the project is to __use reinforcement learning to find an optimal sequence of quantum gates__, to reproduce the dynamics of a given quantum many-body system.

![digital simulation](digital_simulation.png)

[3] S. Lloyd, Science 273, 1073 (1996).<br>
[4] E. A. Martinez, et al., Nature 534, 516 (2016).



# Description of the files:

- main_DQL.py
- main_qlearning.py
- deep_q_learning.py
- q_learning.py
- models.py
- environements.py
- systems.py
- discrete.py

- results
  - view_results.ipynb
  - process_results.py
  - database.py
  - status_databse.json
  - info_database.py
  - info_database.json
  - plots/plots.py
