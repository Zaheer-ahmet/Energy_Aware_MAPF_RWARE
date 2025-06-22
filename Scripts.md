*** Source Codes ***
rware.warehouse :
  -Goals change posisitons
  -individual rewards for the agents
  -Used for mapf scripts

rware.hrpf_team_warehouse :
  -Episodic environment, episodes reinitiate when all agents reach their goal
  -agents start at random positions at the start of each episodes
  -individual and team rewards accumulate
  -Used for HRPF scripts and RL training

rware.comb_warehouse :
  -Episodic environment, episodes reinitiate when all agents reach their goal
  -agents start at random positions at the start of each episodes
  -individual and team rewards accumulate
  -Also charging and charging station logic integrated.
  -Used for Desired_Algorithms and testing with BFS

rware.rendering.py
  -The same as default rware rendering.py
  -Episode, global step, step and reward counters
  -Shelves changed to obstacles with gray color
  -Charging stations as green tiles denoted by "C" added to the environment

*** mapf scripts= simple mobing goals visualization using mapf and accumulating rewards. ***
python mapf_human_play_MovingGoals.py = Human play with moving goals
  Warehouse= rware.warehouse
  Renderer= rware.rendering.py

python mapf_individual_rew.py = Large grid automatic pathfinding moving goals, only individual rewards
  Warehouse= rware.warehouse
  Renderer= rware.rendering.py

python mapf_team_rew.py = Large grid automatic pathfinding stationary goals, individual and team rewards
  Warehouse= rware.warehouse
  Renderer= rware.rendering.py

*** hrpf scripts= environment setup for training and evaluating the agents as episodic and accumulating rewards both individual and team ***

python trainer.py
  rware.hrpf_team_warehouse
  rware.rendering.py

python agent_networks.py

python evaluate.py
  rware.hrpf_team_warehouse
  rware.rendering.py

python human_play_hrpf.py
  rware.hrpf_team_warehouse
  rware.rendering.py

*** Desired_algorithm scripts = Environment setup for testing and evaluating the final Reward+charging logic as episodic ***

python Desired_algorithm_HP.py
  rware.Comb_warehouse
  rware.rendering.py

python Desired_algorithm_BFS_Charge.py
  rware.Comb_warehouse
  rware.rendering.py

python Desired_algorithm_BFS_No_Charge.py
  rware.Comb_warehouse
  rware.rendering.py
