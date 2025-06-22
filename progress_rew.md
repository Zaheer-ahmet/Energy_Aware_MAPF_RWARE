1. Created a new script `mapf_individual_rew.py` for a custom MAPF environment using RWARE, with a 10x10 grid, 4 agents, and BFS pathfinding.
2. Implemented logic for agents to treat shelves as obstacles, using their positions for pathfinding and collision avoidance.
3. Updated the reward system: +10 for reaching goal, -10 for collision, -0.1 for moving/staying off goal.
4. Added slow rendering (0.5s per step) and ESC key handling for live observation and early termination.
5. Modified the rendering logic in `rware/rendering.py` so that all shelves (requested or not) are drawn as gray obstacles, removing the distinction between requested (teal) and normal (purple) shelves.
6. Ensured that all shelf logic in the environment and visualization is consistent with the new obstacle treatment.
7. Created a separate layout file 'custom_layout.txt' to define the warehouse environment, making it easy to test different layouts without changing the script.
8. Refactored the script to load the environment layout from 'custom_layout.txt' using a dedicated loader function, improving modularity and flexibility.
9. Corrected layout file format issues ('.' for empty, 'x' for obstacles, 'g' for goals) and ensured it was strictly 10x10 to fix the 'Layout must be rectangular' error.
10. Added `pygame.init()` to the script to fix the 'video system not initialized' error.
11. Installed `pyglet` dependency required for rendering.
12. Downgraded `pyglet` to version 1.5.27 to resolve compatibility issues and fix the `TypeError: unsupported operand type(s) for |`.
13. Confirmed the script now runs, loads the layout, and initializes the environment correctly.
14. Updated the script to capture the final state (positions, rewards, step, termination reason) before termination.
15. Added functionality to save the final state details to a text file in `final_states/` with a timestamp.
16. Added functionality to capture the final rendered frame as a screenshot (PNG) and save it to `final_states/mapf_individual_rew/` with a timestamp.
17. Corrected the save path for the final state text file to ensure it is saved in the script-specific subdirectory (`final_states/mapf_individual_rew/`) alongside the screenshot.
18. Added a visual reward counter below each agent in the rendering (`rware/rendering.py`).
19. Updated the main script (`mapf_individual_rew.py`) to track accumulated rewards per agent and pass them to the renderer.
20. Changed the reward counter background color to solid black (removed transparency).
21. Refactored the reward calculation logic (`compute_individual_reward`) and the main loop in `mapf_individual_rew.py` to correctly implement the desired reward structure (+10 first goal, -10 collision, -0.1 move, 0 after goal reached).
22. Improved collision detection to handle agent-agent and agent-obstacle scenarios based on final positions.
23. Ensured reward accumulation stops for an agent once its goal is reached.
24. Modified `select_next_move` to account for the agent's current direction, enabling it to choose TURN_LEFT, TURN_RIGHT, or FORWARD actions appropriately based on the BFS path.
25. Updated the main loop to pass the agent's current direction to `select_next_move`.
26. Updated `custom_layout.txt` to include 4 goal locations ('g'), ensuring each agent has a specified goal in the layout.
27. Increased `RENDER_DELAY` to 1.0 second per step to slow down the simulation for easier observation.
28. Fixed `AttributeError: UP` by switching action selection to return integers (0-4 for NOOP, UP, DOWN, LEFT, RIGHT) instead of `Action.UP` etc., as the base `Warehouse` environment seemed to accept these direct actions.
29. Fixed `AttributeError: 'Viewer' object has no attribute 'render_step'` by removing the attempt to call `render_step` before saving the final state.
30. Commented out final screenshot saving because the human-mode renderer (`Viewer`) lacks a standard `get_frame` method. Final state is still saved to text.
31. Diagnosed agent rotation issue: realized agents need Turn/Forward actions, not direct directional moves.
32. Modified `select_next_move` to use the agent's current orientation (`agent.dir` of type `Direction`) to calculate `ACTION_FORWARD`, `ACTION_TURN_LEFT`, or `ACTION_TURN_RIGHT` based on the BFS path.
33. Simplified collision detection and reward calculation to rely on the environment's `step()` return values (`RewardType.INDIVIDUAL`) instead of predictive checks and custom calculations.
34. Observed agent turning loop near goal. Suspect incorrect assumption about `Direction` enum values (0-3). Planning to print enum values for verification.
## Progress Log (2025-04-28)
    -Implemented dynamic goal relocation: when an agent reaches its goal, the goal is moved to a new random free cell and the agent pursues the new goal.
    -Fixed bug with goal assignment (ensured new goals are tuples, not numpy arrays).
    -Agents now continuously pursue new goals instead of stopping at the first goal.
    -The main script now always uses the environment's current goals, so agents never stop moving and always chase the latest dynamically assigned goal.
    -Simulation only terminates when the maximum number of steps is reached or ESC is pressed.
    -Confirmed correct per-step penalty (-0.1) and goal reward (+10) in simulation output and visual display.

## Progress Log (2025-04-29)
    -Reviewed reward tracking system:
        * Verified existing reward counter functionality beneath each agent
        * Confirmed rewards are correctly displayed with one decimal place precision
        * Validated that the reward system shows -0.1 per step and +10 for reaching goals
        * Reward counters use black background with white text for optimal visibility
    -Added reward list display at top of screen:
        * Shows same reward values that appear beneath agents
        * Located in top-right corner for easy reference
        * Displays "Agent X: Y.Y" format with one decimal precision
        * Uses black text on white background for readability
        * Maintained consistency between individual agent reward displays and top list

## Environment Description: Dynamic MAPF in a Robotic Warehouse

This environment is a **Multi-Agent Path Finding (MAPF) simulation** in a 2D grid-based robotic warehouse, implemented using a custom extension of the RWARE (Robotic Warehouse) environment.

### Key Features

- **Grid World:** The warehouse is represented as a 2D grid (e.g., 20x40), with obstacles (`x`), free spaces (`.`), and dynamic goals (`g`).
- **Obstacles/Shelves:** Obstacles are static and block agent movement. The number and placement of obstacles can be customized via a layout file.
- **Agents:** Multiple agents (e.g., 4) are initialized at random positions and orientations. Each agent is assigned a goal location.
- **Dynamic Goals:** When an agent reaches its goal, the goal is immediately relocated to a new random free cell (not an obstacle, not occupied by another agent or goal). The agent then pursues the new goal.
- **Continuous Operation:** Agents never stop; they continuously pursue new goals for the entire episode.
- **Termination:** The simulation only ends when the maximum number of steps is reached or the user presses ESC.
- **Reward Structure:**
  - **Step Penalty:** Each agent receives a reward of -0.1 per step.
  - **Goal Reward:** Each agent receives +10.0 when reaching its current goal.
- **Rendering:** The environment is visualized using Pyglet, with clear margins, agent reward counters, and real-time updates.
- **Layout File:** The warehouse layout is loaded from a text file (e.g., `custom_layout_2`), which specifies the grid, obstacles, and initial goal locations.

### Main Attributes

- **Grid Size:** Customizable (e.g., 20x40).
- **Obstacles:** Specified by `x` in the layout file.
- **Goals:** Specified by `g` in the layout file; dynamically relocated during simulation.
- **Agents:** Number of agents is configurable; each agent has a position, orientation, and battery level.
- **Rewards:** Per-step penalty and goal reward as described above.
- **Termination:** By max steps or ESC key.

### Example Use Case

This environment is ideal for benchmarking MAPF algorithms in dynamic, obstacle-rich settings, and for studying agent coordination, path planning, and continuous goal assignment in warehouse-like domains.

## Team Reward Implementation (mapf_team_rew.py)
- Created a new script `mapf_team_rew.py` based on `mapf_individual_rew.py` to implement team rewards
- Team reward mechanism tracks overall team performance:
  - Time-based penalty: -0.1 × total steps taken until all agents complete their goals
  - Large completion bonus: +50.0 when all agents reach their goals
- Team rewards are tracked separately from individual rewards
- Results are saved in a separate directory `final_states/mapf_team_rew/`

## Latest Updates (2025-04-29)
- Modified team reward script to use stationary goals:
  - Goals remain fixed at their initial positions from the layout
  - Agents stop moving (NOOP) once they reach their assigned goals
  - Simulation ends when all agents complete their initial goals
- Implemented team reward visualization:
  - Added team reward display in the top-right corner
  - Shows cumulative team reward (step penalties + completion bonus)
  - Updates in real-time during simulation
- Verified functionality with test run:
  - All agents successfully reached their goals
  - Team reward properly accumulated (-0.1 per step)
  - +50.0 bonus awarded upon full team completion
  - Final states saved with both individual and team rewards

## Progress Log (2025-05-01)
- Added charging stations to the environment:
    - Charging stations are denoted by 'C' in the layout file.
    - Rendered as green tiles with a white 'C' label, visually similar to goals.
    - When an agent passes through a charging station, its battery is restored to maximum (10).
    - Updated the layout parser and environment step logic to support charging stations and battery restoration.
    - Added a charging station to the test layout (custom_layout_2.txt) for validation.

## Progress Log (2025-06-02)
- Created a new HRPFTeamWarehouse environment in rware/hrpf_team_warehouse.py:
    - Implements HRPF team reward logic: static goals, per-agent and team rewards, proper episode termination.
    - Per-agent: +10 for first goal, -0.1 per step, -10 for collision, 0 after goal.
    - Team: -0.1 per step, +50 once when all agents complete, only once.
    - Tracks completed agents, team completion, and exposes info fields for RL.
- Created a new script mapf_hrpf_team.py based on mapf_team_rew.py to use HRPFTeamWarehouse and demonstrate the new logic.
- Verified that the new environment and script run as expected.

## Progress Log (2025-06-02, Update 2)
- Refactored HRPFTeamWarehouse:
    - Fixed redundant reward overwrites in step logic (collision, step, goal rewards).
    - Made reward logic conditional on reward_type (INDIVIDUAL or GLOBAL).
    - Exposed completed_agents in the info dict for RL agent access.
    - Fixed goal indexing bug in _make_img_obs (layer[goal_y, goal_x]).
    - Added support for render_mode='none' to allow headless RL training.

## Progress Log (2025-06-03)

- **Created `agent_networks.py`:**
    - Implements the core neural network architecture for RL agents.
    - Defines `QNetwork`, a feedforward (MLP) Q-network used for both individual (Q_in) and team (Q_te) Q-learning.
    - Provides `combine_q_values`, which implements the HRPF Q-value mixing equation for action selection.
    - Includes `create_agent_networks`, a utility to instantiate both Q_in and Q_te networks for each agent.

- **Created `trainer.py`:**
    - Implements the full RL training loop for multi-agent Q-learning in the HRPFTeamWarehouse environment.
    - Initializes per-agent Q_in and Q_te networks, their target networks, and optimizers.
    - Uses a replay buffer for experience replay.
    - Handles epsilon-greedy action selection, target network synchronization, and periodic model checkpointing.
    - Computes both individual and team Q-losses as per the HRPF equations.
    - Flattens observations for use with fully-connected Q-networks.

- **Latest Improvements:**
    - Model checkpoints are now saved in a dedicated `models/` directory for better organization.
    - Added logging: every 10 episodes, prints average episode reward and team reward for easier monitoring and debugging.
    - Confirmed that observation flattening is appropriate for the current MLP-based Q-network architecture.
    - Ensured that the environment's info dict provides `team_reward` for compatibility with the training script.

## Progress Log (2025-06-03)

- Fixed the `TypeError` caused by passing `layout_path` to the base `Warehouse` class by removing it from `kwargs` in `HRPFTeamWarehouse.__init__`.
- Updated `trainer.py` to provide all required positional arguments to `HRPFTeamWarehouse` (including `shelf_columns`, `column_height`, etc.).
- Fixed assertion error by ensuring `shelf_columns` is an odd number.
- Fixed crash in `trainer.py` by inferring and setting `self.observation_space` in `HRPFTeamWarehouse` using a sample observation from `self.reset()`.

## Progress Log (2025-06-04)
- **Major debugging and compatibility fixes for RL training:**
    - Fixed observation space in `HRPFTeamWarehouse` to be a `Tuple` of `Box` spaces (one per agent), matching the base environment and RL code expectations.
    - Updated `trainer.py` to always unpack both `obs, info = env.reset()` and use the correct observation structure.
    - Fixed action space handling: set `n_actions = len(Action)` to match the per-agent discrete action space, avoiding errors with `Tuple` action spaces.
    - Ensured `obs_dim` is always an `int` (not `numpy.float64`) when passed to PyTorch layers, preventing type errors in `QNetwork`.
    - Fixed all downstream errors related to observation and action space mismatches, so RL training can now proceed without manual intervention.
    - Fixed `NoneType` error in `trainer.py` by extracting `obs_shape` from the first Box in the Tuple observation space (`obs_shape = env.observation_space[0].shape`).
    - Added a check in `rware/hrpf_team_warehouse.py` to ensure the number of goals in the layout matches `n_agents`, raising a clear error if not. This prevents IndexError and helps users debug layout/agent mismatches.
    - Fixed a bug where the check for goal/agent mismatch in the layout was not triggered because `self.n_agents` was set after parsing the layout. Moved `self.n_agents = n_agents` before layout parsing to ensure the check works and prevents IndexError in step.
    - Fixed persistent IndexError in step by ensuring the number of goals in `custom_layout_2.txt` matches `n_agents` (4). Removed an extra goal from the layout so there are exactly 4 goals for 4 agents.
    - Added a goal to line 17 of `custom_layout_2.txt` to ensure there are exactly 4 goals for 4 agents, resolving the goal/agent mismatch.
    - Added a debug print to the step function in `rware/hrpf_team_warehouse.py` to trace the length of `self.goals`, the value of `i`, and the number of agents when the IndexError occurs, to help diagnose persistent goal/agent mismatch issues.
    - Added a defensive check in the step function to pad the goals list if it is shorter than the number of agents, preventing IndexError and aiding debugging of goal list corruption.
    - Added debug prints in reset to show the length and contents of self.goals and self._initial_goals, and a defensive check for self._initial_goals in step to prevent IndexError and aid debugging.
    - Removed dynamic goal relocation logic from the environment; agents now have stationary goals throughout the episode, matching the intended stationary-goal MAPF setting.
## Progress Log (2025-06-04, Evaluation and Clean Output)
- Created `evaluate.py` to:
    - Load trained .pt models from `/models`.
    - Run the custom `HRPFTeamWarehouse` environment.
    - Let agents act greedily using the trained Q-networks.
    - Visualize the run using `render_mode="human"`.
- Updated `evaluate.py` to match the environment instantiation in `trainer.py` (all required arguments).
- Removed all debug print statements from `rware/hrpf_team_warehouse.py` to ensure clean output during evaluation and training runs.
- Improved the kill switch in `evaluate.py` to work reliably on all platforms by listening for ESC in the Pyglet window and terminating evaluation when pressed.
## Progress Log (2025-06-04, Default Layout and 2 Agents)
- Switched all HRPF RL scripts (`trainer.py`, `evaluate.py`) to use the default RWARE layout (11x11, high obstacle density) and set number of agents to 2.
- Removed all code related to custom layout files for consistency and better benchmarking.
## Progress Log (2025-06-05)
- Enforced classic RWARE collision logic in `HRPFTeamWarehouse`:
    - Agents cannot move into cells occupied by other agents or obstacles (gray cells).
    - If an agent attempts such a move, the move is blocked (agent stays in place) and receives a -10 penalty.
    - No two agents can occupy the same cell at the same time.
- Reconstructed the penalty logic:
    - +10 for reaching the goal (first time only)
    - -10 for a failed move (blocked by agent or obstacle)
    - 0 for already completed
    - -0.1 for a normal step
- Added a human play script (`human_play_hrpf.py`):
    - Lets the user control agent 0 with the keyboard (arrows or WASD).
    - Other agents do NOOP.
    - Renders the environment and prints per-step and cumulative rewards.
    - ESC or window close to exit.
- The log will continue to be updated as further changes are made.

## Progress Log (2025-06-11)
- Fixed reward assignment order in `HRPFTeamWarehouse` so agents always receive +10 when reaching the goal, and this is reflected in cumulative returns.
- Suppressed pyglet window close error in `human_play_hrpf.py` by wrapping `env.close()` in a try/except block.
- Debugged and improved human play: ensured correct movement, turning, failed move penalties, and goal rewards.
- Confirmed that agents cannot move through obstacles or each other, and that all reward/penalty logic matches classic RWARE.
- Human play script now prints step rewards and cumulative returns, and episode returns include the goal reward.
- All changes tested interactively and confirmed to work as intended.

## Progress Log (2025-06-11, Reward Bug Fix)
- **Resolved lingering reward bug in HRPFTeamWarehouse:**
    - **Problem:** The +10 reward for reaching the goal was not being added to the cumulative return. This was because the reward assignment logic checked if the agent was in `completed_agents` before checking if it had just reached the goal, causing the +10 to be overwritten by 0.0 in the same step.
    - **Fix:** Refactored the `step` method to handle all movement, collision, and reward assignment in a single pass, and to check for goal completion before checking `completed_agents`. Now, the +10 reward is always given on the correct step and cannot be overwritten.
    - **Verification:** Human play output now shows the +10 reward added to the cumulative return on the step the agent reaches the goal, and the episode returns are correct.
    - **Cleanup:** Removed debug prints from the reward assignment loop as the logic is now confirmed to work.

## Progress Log (2025-06-11, Loss Function and Training Updates)
- **Corrected Individual Loss Function Implementation:**
    - Fixed the individual loss (Eq. 3) to strictly follow: L_in = ((R_in[i] + gamma*max_a'[i](Q_in[i](o',a'))-Q_in[i](o,a))^2
    - Removed incorrect hybrid reward assumption that mixed individual and team rewards
    - Retained proper terminal state handling with (1-d) term for both individual and team losses
    - Verified mathematical correctness of both loss implementations

- **Training Configuration Updates:**
    - Reduced training episodes from 5000 to 1000 for faster iteration
    - Maintained other hyperparameters:
        * Buffer capacity: 100,000
        * Batch size: 64
        * Gamma: 0.99
        * Epsilon: 0.5
        * Beta: 1.0
        * Max steps per episode: 200
    - Target networks still sync every 20 episodes
    - Models save checkpoints every 100 episodes

## Progress Log (2025-06-12, Training and Evaluation Updates)
- **Training Improvements:**
    - Increased batch size to 128 for more stable learning
    - Lowered epsilon_end to 0.05 for better exploitation
    - Reduced learning rate to 3e-4 for more stable updates
    - Added observation normalization to handle different scales

- **Observation Normalization:**
    - Added RunningMeanStd class to track observation statistics
    - Normalizes observations before feeding to networks
    - Updates statistics during both training and evaluation
    - Should help with the Q-value scale issues

- **Evaluation Updates:**
    - Added same normalization to evaluation script
    - Maintains running statistics during evaluation
    - Better debugging output for Q-values

- **Expected Outcomes:**
    - More stable Q-value scales
    - Better exploration-exploitation balance
    - More consistent learning across agents
    - Proper state observation processing

- **Next Steps:**
    - Delete existing model files in the `models/` directory
    - Run training again with new changes
    - Evaluate to see if agents move properly

## Progress Log (2025-06-13, Prioritized Experience Replay)
- **Switched to Prioritized Experience Replay (PER) in RL training:**
    - Replaced the uniform ReplayBuffer in `trainer.py` with `PrioritizedReplayBuffer` from torchrl, using `ListStorage` for efficient experience management.
    - PER samples transitions with higher TD error more frequently, focusing learning on the most informative experiences.
    - Buffer now supports priority-based sampling and is ready for future updates to priorities after each learning step.
    - Motivation: PER is expected to accelerate learning and improve sample efficiency by focusing updates on transitions that matter most, as described in [Schaul et al., 2015](https://arxiv.org/abs/1511.05952) and implemented in [torchrl.data.PrioritizedReplayBuffer](https://docs.pytorch.org/rl/0.6/reference/generated/torchrl.data.PrioritizedReplayBuffer.html).

## Progress Log (2025-06-13, PER Training and Evaluation Success)
- **Successfully completed PER-enabled training and evaluation:**
    - Fixed all batch extraction and TD error calculation issues for Prioritized Experience Replay (PER) with torchrl's ListStorage.
    - Training now runs to completion, saving models and logging progress as expected.
    - Evaluation script loads the trained models and runs the environment, showing agents taking actions and receiving step rewards.
    - Output confirms that PER is fully integrated and stable in the RL pipeline.
    - See below for interpretation of the latest evaluation results.

### Evaluation Output Interpretation (2025-06-13)
- **Agents' Q-values:**
    - Both agents show negative Q-values for all actions, indicating the environment is still challenging and the agents are mostly experiencing penalties (step cost, failed moves, etc.).
    - The Q-values are not all identical, showing the agents have learned some action preferences.
- **Actions:**
    - Agents are selecting a mix of LEFT, RIGHT, and NOOP actions, indicating they are not stuck and are exploring the action space.
- **Rewards:**
    - Both agents receive -0.1 per step, which is the step penalty, and no positive rewards in the shown steps, indicating they have not reached their goals in this evaluation run.
    - Total rewards for both agents are negative (e.g., -4.60 after 46 steps), which is expected if no goals are reached.
- **Team reward:**
    - The team reward matches the sum of individual rewards, confirming correct reward aggregation.
- **Conclusion:**
    - The PER buffer is working, and the RL pipeline is stable.
    - Agents are learning to move, but further reward shaping, hyperparameter tuning, or longer training may be needed for more consistent goal-reaching behavior.

## Progress Log (2025-06-13, Reward Shaping and Exploration)
- **Reward shaping for forward movement:**
    - Modified the reward structure in `HRPFTeamWarehouse.step` to add a small positive reward (+0.05) for reducing the Manhattan distance to the goal compared to the previous step.
    - This encourages agents to move toward their goals, not just turn or NOOP.
- **Slowed epsilon decay:**
    - Changed `epsilon_decay` in `trainer.py` from 0.995 to 0.999 to ensure agents explore for longer during training.
    - This should help agents discover the value of moving forward and reaching goals.

## Progress Log (2025-06-13, Longer Episodes and Training)
- **Increased max_steps per episode to 400:**
    - Agents now have more time per episode to reach their goals, which should increase the number of successful goal completions and improve learning signal.
- **Increased num_episodes to 5000:**
    - Training will run for more episodes, giving agents more experience and a better chance to learn effective policies.

## Progress Log (2025-06-13, PER Acceleration)
- **Increased batch size to 256:**
    - Larger batches should stabilize and accelerate learning.
- **PER tuning:**
    - Set alpha=0.8 for stronger prioritization of high-TD-error samples.
    - Anneal beta from 0.4 to 1.0 linearly over training episodes for improved bias correction.

## Progress Log (2025-06-13, Collision Penalty and HER Plan)
- **Reduced collision/failed move penalty to -1.0:**
    - This makes the environment less punishing and should encourage more exploration and forward movement.
    - Agents are now less likely to be dominated by large negative rewards from collisions.
- **Plan to implement Hindsight Experience Replay (HER):**
    - HER is effective for sparse reward problems and will be integrated next to further improve learning efficiency.

## Progress Log (2025-06-13, HER + PER Integration)
- **Integrated Hindsight Experience Replay (HER) with PER:**
    - After each episode, for each agent, relabels the episode using the final position as the HER goal.
    - Recomputes rewards for HER transitions and adds them to the prioritized replay buffer (PER).
    - Both original and HER transitions are sampled and prioritized together, combining the benefits of HER (for sparse rewards) and PER (for efficient learning).

## Progress Log (2025-06-13, Evaluation Diagnostics)
- **Added epsilon-greedy (epsilon=0.05) to evaluation:**
    - During evaluation, each agent now selects a random action with 5% probability, otherwise acts greedily.
    - This helps diagnose if agents can reach goals with a bit of exploration.
- **Logging initial agent positions:**
    - For episodes where both agents reach their goals, the initial positions are logged for further analysis.

## Progress Log (2025-06-14, Goal Completion Fixes)
- **Fixed agent behavior after goal completion:**
    - In `HRPFTeamWarehouse.step`, agents that have reached their goal are now forced to perform NOOP, cannot move or turn, and always receive 0.0 reward for subsequent steps.
    - In `evaluate.py`, the evaluation loop now tracks which agents have completed and always sets their action to NOOP after goal completion.
    - This ensures agents stop moving after reaching their goal and the reward calculation is correct, matching the intended environment logic.

## Progress Log (2025-06-14, Evaluation Reward Accumulation Fix & Next Steps)
- **Problem presented by user:**
    - After recent improvements, agent 0 reliably reached its goal (albeit inefficiently), but agent 1 continued to wander aimlessly, even when not blocked by obstacles or other agents.
    - The user also observed that the per-agent total rewards at the end of evaluation were always 0, even though human play showed correct reward accumulation.
- **Fixes and updates:**
    - Identified that the reward accumulation bug was due to using a Python list for `total_rewards` and adding lists of `np.float64` rewards, which does not perform elementwise addition.
    - Fixed by switching `total_rewards` to a numpy array and accumulating rewards as numpy arrays in `evaluate.py`.
    - Now, per-agent total rewards are correctly reported and match human play results.
- **Policy update:**
    - From this point forward, each progress log entry will explicitly mention the user problem or observation that triggered the change, for better traceability.
- **Next phase:**
    - Proceeding to change the environment layout to a smaller size with fewer obstacles, as previously discussed, to make the task easier and facilitate learning for both agents.

## Progress Log (2025-06-14, Initial Layout Size Reduction)
- **Problem/motivation presented by user:**
    - To help agents learn basic goal-reaching and avoid getting stuck, the user requested reducing the environment size to make it simpler and less cluttered.
- **Change:**
    - Updated both `trainer.py` and `evaluate.py` to use a much smaller layout: `shelf_columns=1`, `column_height=3`, `shelf_rows=1`.
    - The goal was to create a minimal warehouse for easier learning and debugging.
- **Outcome:**
    - This change made the grid too small to fit all agents, shelves, and goals, leading to a ValueError during environment reset (see next log entry for the fix).

## Progress Log (2025-06-14, Grid Size Increase for Placement and Movement)
- **Problem presented by user:**
    - When attempting to run evaluation before training (to view the layout), a ValueError was encountered: `a cannot be empty unless no samples are taken`. This was due to the grid being too small to place all agents, shelves, and goals without overlap.
- **Fixes and updates:**
    - Increased the grid size in both `trainer.py` and `evaluate.py` to `shelf_columns=3`, `column_height=3`, `shelf_rows=2`.
    - This creates a small but not cramped warehouse, ensuring there is enough space for agents, shelves, goals, and some movement.
    - The environment is now suitable for both training and evaluation, and the ValueError is resolved.
- **Policy reminder:**
    - As per the new policy, this log entry explicitly mentions the user problem that triggered the change for better traceability.

## Progress Log (2025-06-14, Training Data Interpretation & Initial Position Logging)
- **Problem/observation presented by user:**
    - The user observed that the number of episodes where both agents reach their goals increased in the middle of training but decreased again in later episodes.
    - The user also noted that the initial positions of agents for successful (2-goal) episodes were not being saved in the training data, making it hard to analyze which starts led to success.
- **Interpretation:**
    - Training data shows early and late episodes are dominated by failures (0 or 1 goal reached), while the middle of training sees more frequent success (2 goals reached).
    - This may be due to learning instability, exploration decay, or environment randomness.
- **Change:**
    - The training script now saves the initial (x, y) position of each agent for every episode in the training data CSV.
    - Added columns: 'Agent 0 X', 'Agent 0 Y', 'Agent 1 X', 'Agent 1 Y'.
    - This enables post-hoc analysis of which initial positions are associated with successful episodes.
- **Motivation:**
    - To allow deeper diagnostics and understanding of agent performance and environment difficulty, especially for episodes where both agents succeed.

## Progress Log (2025-06-14, Agent 0 Stuck Loop Diagnostic & Goal Logging)
- **Problem/observation presented by user:**
    - After the last training, agent 1 occasionally reaches the goal, but agent 0 often gets stuck in a loop.
- **Analysis:**
    - Possible causes include replay buffer imbalance, initial position or goal bias, or learning instability.
    - No agent-specific logic or network asymmetry was found in the code.
- **Change:**
    - The training script now logs each agent's goal (x, y) position for every episode in the training data CSV.
    - Added columns: 'Agent 0 Goal X', 'Agent 0 Goal Y', 'Agent 1 Goal X', 'Agent 1 Goal Y'.
    - This enables analysis of whether agent 0's failures are correlated with its initial or goal positions.
- **Recommendation:**
    - Retrain with this new logging enabled, then analyze the data to determine if the stuck behavior is due to environment setup or learning dynamics.

## Progress Log (2025-06-14, Evaluation Reward Counter Fix & Custom Init Request)
- **Problem/observation presented by user:**
    - The per-agent reward counters below each agent in the evaluation script were not updating during evaluation runs.
- **Fix:**
    - The evaluation script now tracks per-agent accumulated rewards during each episode and sets `env._agent_accumulated_rewards` before each render call, ensuring the renderer displays live, up-to-date reward counters for each agent.
- **New request:**
    - The user wants to be able to evaluate from specific initial agent/goal positions (as found in the training data for successful episodes), as an option, without removing the default random initialization code.
- **Next steps:**
    - Add an option to the evaluation script to accept a set of initial positions/goals and use them for environment reset when desired, while keeping the default random initialization as the fallback.

## Progress Log (2025-06-14, RL Improvements & Bug Fix)
- **Problem/motivation presented by user:**
    - Agents were not learning efficiently; rare goal-reaching, erratic movement, and poor reward signals.
    - User requested: increase reward for progress, more training episodes, and balanced sampling for both agents.
- **Changes:**
    - Increased reward for progress to +0.2.
    - Increased training episodes to 10,000.
    - Implemented balanced sampling in the replay buffer.
    - Fixed bug in agent indices concatenation for sampling.

## Progress Log (2025-06-14, Human Play Script Creation)
- **Problem/motivation presented by user:**
    - Needed a way to manually test reward and charging station logic, and visualize agent/team rewards in a human-playable environment.
- **Changes:**
    - Created Desired_algorithm.py for human play and evaluation using Comb_warehouse.py.
    - Episodic individual rewards, cumulative team reward, no max steps, ESC to exit.
    - Live display of agent and team rewards, with controls for agent switching and environment reset.

## Progress Log (2025-06-14, Human Play Logging & Charging Station Logic)
- **Problem/motivation presented by user:**
    - Wanted to suppress per-step terminal output in Desired_algorithm.py, save step-by-step reports to a folder, and add charging station logic (battery depletion and recharge on green tiles) as in hrpf_team_warehouse.py.
- **Changes:**
    - Suppressed per-step terminal output; only final rewards are printed.
    - Step-by-step reports are now saved under Desired_algorithm/Human_Play with a timestamp.
    - Added charging station logic: agents lose 1 battery per move, and recharge to max when passing over a charging station (green tile), matching the logic in hrpf_team_warehouse.py.

## Progress Log (2025-06-14, Reward Structure Simplification & Team Reward/Battery Fixes)
- **Problem/motivation presented by user:**
    - The user requested removal of the additional reward for reducing Manhattan distance to the goal, to simplify the reward structure and avoid unintended incentives.
    - The user observed that the cumulative team reward penalty was increasing quadratically (0.1, 0.3, 0.6, 1.0, ...) instead of linearly (0.1, 0.2, 0.3, 0.4, ...), indicating a bug in the accumulation logic.
    - The user also requested that the small box on each agent (representing battery life) should display the live battery count, not just a static indicator.
- **Changes:**
    - Removed the distance-based progress reward from Comb_warehouse.py; now only +10 for goal, -1 for failed move, -0.1 for step, 0 after goal.
    - Identified and clarified the bug in team reward accumulation; the penalty should increase linearly per step, not quadratically. (Pending fix if not already applied.)
    - Updated the agent battery display so the box on each agent shows the current battery count live during the episode.

## Progress Log (2025-06-14, Team Reward Persistence & Battery Display Bug)
- **Problem/observation presented by user:**
    - The team reward is incorrectly resetting at the start of each episode; it should be cumulative across all episodes, not per-episode.
    - The battery life counter on each agent is not updating live in the simulation display (renderer).
- **Next steps:**
    - Fix the team reward logic so it persists across episodes and only individual rewards reset.
    - Update the renderer to display the live battery count for each agent during the simulation.

## Progress Log (2025-06-14, Team Reward & Battery Display Fixes Applied)
- **Fixes implemented:**
    - Team reward is now cumulative across all episodes and does not reset at episode start; only individual rewards reset.
    - The renderer now displays the live battery count for each agent in the small box during the simulation, using env._agent_batteries.
- **Outcome:**
    - Team reward persistence and live battery display now match user expectations and simulation requirements.

## Progress Log (2025-06-15, Agent Visual Feedback & Index Badge Tweaks)
- **Changes implemented:**
    - Added logic to flash agents red for a split second after a failed move, using env._agent_failed_moves. User reported the effect is not visible and requests further improvement.
    - Added a small badge to the bottom right of each agent showing its index. User requested to remove the ring/border and to display the index starting from 0 (to match logs and system).
- **Next steps:**
    - Improve the red flash effect for failed moves to ensure it is clearly visible.
    - Update the badge to remove the border and display the correct index.

## Progress Log (2025-06-15, Episode Counter & Margin Improvement)
- **Changes implemented:**
    - Added an episode counter above the step counter in the simulation display.
    - Increased the top margin of the environment window to prevent counter displays from overlapping the environment.
- **Motivation:**
    - User requested clearer display of episode and step counters, and to avoid overlap with the environment grid.

## Progress Log (2025-06-15, Episode Counter Logic Finalized)
- **Change implemented:**
    - The episode counter now starts from 1 and increments by one at the start of each new episode, matching the displayed episode number to the actual episode count.
- **Motivation:**
    - To ensure the episode display is intuitive and matches user expectations and naming conventions.

## Progress Log (2025-06-15, Team Reward Bonus Per Episode)
- **Change implemented:**
    - The team reward logic now adds the +50 completion bonus at the end of every episode (when both agents complete their tasks), not just once for the entire run.
    - The _team_bonus_given flag is reset at the start of each episode to ensure the bonus is awarded per episode.
- **Motivation:**
    - To match the intended semantics of team reward in episodic evaluation, where each episode's completion is rewarded.

## Progress Log (2025-06-15, Charging Stations Added to Layout)
- **Change implemented:**
    - Two charging stations have been added to the environment layout at positions (4,4) and (5,4). These are now rendered as green tiles with 'C' labels, and agents will recharge their batteries when passing over them.
- **Motivation:**
    - To complete the environment setup for multi-agent path finding with charging logic, enabling realistic battery management for agents.

## Progress Log (2025-06-15, Battery Recharge & Persistence Fixes)
- **Change implemented:**
    - Agent battery is now restored to max whenever the agent's position matches any charging station's current position, not just hardcoded locations. This works for any layout.
    - Agent battery life is no longer reset at the start of each episode; battery persists across episodes unless recharged by a charging station.
- **Motivation:**
    - To ensure correct and flexible battery management for multi-agent path finding with charging stations, supporting dynamic layouts and persistent agent state.

## Progress Log (2025-06-15, Battery Persistence Bug Fixed)
- **Change implemented:**
    - Fixed battery persistence: agent battery is now preserved across episodes by saving battery values before reset and restoring them after agent recreation.
    - Battery is only recharged at charging stations, never reset at episode start.
- **Motivation:**
    - To ensure correct battery management for all layouts and episode transitions, matching user expectations for persistent agent state.

## Progress Log (2025-06-15, Battery Logic Final Fix)
- **Change implemented:**
    - In `Comb_warehouse.py`, battery is now only recharged at charging stations and is never reset at the start of an episode.
    - Removed all code in `reset` that restored or reset battery values.
    - Recharge logic is now only in `step`, after agent moves, matching the working logic in `warehouse.py`.
- **Motivation:**
    - To ensure correct, persistent battery management for all layouts and episode transitions, and to match the intended RWARE behavior.

## Progress Log (2025-06-15, Battery Logic Debugging and Renderer Sync)
- **Debugging and Fixes:**
    - Added debug prints to show battery depletion and recharge events for each agent at every step.
    - Fixed battery logic so that battery is depleted by 1 for every FORWARD move and only recharged at charging stations, never at episode start.
    - Ensured battery is not reset at the start of each episode; battery persists across episodes unless recharged.
    - Synced the battery life indicator in the renderer with the true agent.battery value by setting `env._agent_batteries = [agent.battery for agent in self.agents]` before each render call.
    - Now, the battery indicator above each agent in the simulation always matches the debug output and the true battery value.
- **Outcome:**
    - Battery logic and display are now fully consistent and correct, matching both the debug output and the intended simulation behavior.

## Progress Log (2025-06-15, Battery Display Sync Finalized)
- **Final Resolution:**
    - The battery indicator above each agent in the simulation is now always in sync with the true battery value and the environment logic.
    - The sync is performed automatically at the start of the render method, so the display is always correct regardless of where render() is called.
    - All debug print statements related to battery depletion, recharge, and after reset/step have been removed from the code to clean up terminal output.
    - The battery logic and renderer integration are now robust, clean, and production-ready.

## Progress Log (2025-06-16, Automated BFS Script and Initial Issue)
- **Created Desired_algorithm_BFS.py:**
    - New script automates agent movement using BFS pathfinding to their goals, based on Desired_algorithm_HP.py (human play).
    - All reward, battery, team reward, display, and logging logic is identical to the human play script.
    - Each agent plans a path to its goal using BFS and attempts to follow it at each step.
    - Simulation continues until ESC is pressed.
- **Current Issue:**
    - The simulation starts, but the agents are not moving toward their respective goals as expected.
    - Further debugging of BFS path planning or action selection logic is required to resolve this.

## Progress Log (2025-06-16, BFS Action Selection Fix)
- **Action Selection Fix:**
    - Updated the agent action selection logic in Desired_algorithm_BFS.py so that agents now turn toward the next cell in their BFS path if not already facing it, and move forward when facing the correct direction.
    - This resolves the issue where agents were not moving toward their goals despite valid BFS paths being found.

## Progress Log (2025-06-16, Agent Turning Logic Fix for BFS)
- **Problem:** Agents were observed to move for some frames but then loop randomly, indicating a flaw in the action selection's turning logic, specifically for Agent 1.
- **Fix:** Rewrote the `_get_agent_action` function in `Desired_algorithm_BFS.py` to use a more robust turning mechanism. The function now explicitly determines the `desired_dir` as a `Direction` enum and then calculates the optimal `LEFT` or `RIGHT` turn using the `wraplist` (the circular sequence of directions defined in the `Agent` class). This ensures agents accurately align themselves with the next step in their BFS path.
- **Outcome:** Agents should now correctly navigate towards their goals without random looping, taking appropriate turns based on their current orientation and the planned path.

## Progress Log (2025-06-16, Agent Turning Logic Fix for BFS - Iteration 2)
- **Problem:** Despite previous attempts, agents (especially Agent 1) continued to exhibit random looping behavior, indicating an issue with how `_get_agent_action` determined turns based on the BFS path.
- **Fix:** Rewrote the `_get_agent_action` function in `Desired_algorithm_BFS.py` to use an explicit, case-by-case turning logic. Instead of relying on modular arithmetic for `Direction` enum values, the function now directly compares the `current_dir` and `desired_dir` (both as `Direction` enum objects) and explicitly selects `LEFT`, `RIGHT`, or `FORWARD` actions. This includes specific handling for 180-degree turns (e.g., UP to DOWN) by defaulting to two `RIGHT` turns. This approach directly aligns the action selection with the environment's turning mechanics, eliminating ambiguity.
- **Outcome:** Agents are expected to now move correctly along their BFS-determined paths without random looping, due to the precise and unambiguous turning logic.

## Progress Log (2025-06-16, Removed Charging Station and Battery Logic from Desired_algorithm_BFS.py)
- **Problem/Motivation presented by user:** The user requested to remove all charging station and battery-related logic from `Desired_algorithm_BFS.py` and freeze its rendering for this script, as a new script `Desired_algorithm_BFS_Charge.py` will be created later to handle charging constraints.
- **Change:** All code related to battery initialization (`_init_batteries`), charging station detection (`_is_on_charging_station`), battery depletion during movement, battery recharge at charging stations, and battery display in the simulation report (`_display_info` and `env._agent_batteries` assignment for rendering) has been removed from `Desired_algorithm_BFS.py`.
- **Outcome:** `Desired_algorithm_BFS.py` now focuses solely on BFS pathfinding without any battery or charging station constraints, paving the way for a separate script for charging-aware BFS.

## Progress Log (2025-06-16, Conflict Resolution for Agent Deadlocks in Desired_algorithm_BFS.py)
- **Problem/Observation presented by user:** When agent paths cross, both agents freeze due to the environment's collision logic, leading to deadlocks. The user requested a mechanism to resolve these conflicts, allowing one agent to wait for the other to pass.
- **Fix:** Implemented a conflict resolution strategy within the `_cycle` method of `Desired_algorithm_BFS.py`. Agents are now prioritized by their index (lower index = higher priority). Before executing actions, each agent's predicted next position is determined. If a lower-priority agent's predicted move conflicts with an already approved move of a higher-priority agent (e.g., trying to occupy the same target cell or attempting a head-on swap), the lower-priority agent's action is overridden to `Action.NOOP`, forcing it to wait.
- **Outcome:** This strategy prevents deadlocks by ensuring that only higher-priority agents proceed in case of predicted conflicts, allowing the lower-priority agents to yield and wait, thus enabling continuous movement and avoiding complete freezes.

## Progress Log (2025-06-16, Debug Print Removal from Desired_algorithm_BFS.py)
- **Problem/Motivation presented by user:** All known issues in `Desired_algorithm_BFS.py` have been resolved, and the user requested the removal of all remaining debug print statements for a cleaner output.
- **Change:** All debug print statements related to BFS paths, action selection, and conflict resolution have been removed from `Desired_algorithm_BFS.py`.
- **Outcome:** The script now runs silently, providing a clean terminal output while maintaining all previously implemented functionalities and fixes.

## Progress Log (2025-06-17, Battery-Aware BFS Logic in Desired_algorithm_BFS_Charge.py)
- **Problem/Motivation presented by user:** Implement battery and charging station logic into a new script, `Desired_algorithm_BFS_Charge.py`. Agents should initially target their goals, but switch to the nearest charging station when their battery depletes to 0. Upon full recharge, they should revert to their original goals. A `battery_max` variable was requested for experimentation.
- **Change:**
    - Introduced `self.battery_max` as a configurable variable for maximum battery life.
    - Modified `_cycle` to initialize agent batteries to `self.battery_max` at the start of each episode.
    - Implemented battery depletion: `self.agent_batteries[i] -= 1` for every `FORWARD` movement.
    - Added dynamic goal retargeting logic: When `self.agent_batteries[i]` reaches 0, the agent's goal is updated to the `_get_nearest_charging_station`. The `self.agents_going_to_charge[i]` flag is set to `True`.
    - Implemented battery recharge logic: If an agent is on a charging station (`_is_on_charging_station`) and its battery is not full, it recharges by 1 unit per step until `self.battery_max`.
    - Added logic to revert to the `original_goals` when an agent's battery is full after a charging mission (`self.agents_going_to_charge[i]` is reset to `False`).
    - Activated battery display in simulation and step reports.
- **Outcome:** The `Desired_algorithm_BFS_Charge.py` script now correctly integrates battery constraints, allowing agents to seek charging stations when needed and return to their primary goals once recharged. This includes explicitly updating `self.env.goals[i]` when an agent's goal dynamically changes (to a charging station or back to its original goal), ensuring consistency with the environment's internal state and preventing 'disappearing' goals.

## Progress Log (2025-06-17, Agent Goal Achievement and Continuous Movement Fix in Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:** When both agents reach their goals simultaneously, they continue to attempt forward movement (performing NOOP) and print "WARNING" messages, instead of the episode simply ending. This indicates redundant path planning after goal achievement.
- **Fix:** Modified the `_cycle` method in `Desired_algorithm_BFS_Charge.py`. A new condition was added before path planning for each agent: if an agent is currently at its `self.original_goals[i]` and is *not* in the process of going to a charging station (`not self.agents_going_to_charge[i]`), its `agent_paths[i]` will be explicitly set to an empty list `[]`. This ensures that agents that have already completed their primary goal gracefully perform `NOOP` actions without continuously recalculating paths or generating unnecessary warning messages.
- **Outcome:** Agents that have reached their final destination will now correctly remain at their goal, performing `NOOP` actions, and will no longer attempt redundant path planning, leading to cleaner simulation behavior and output.

## Progress Log (2025-06-17, Pygame Initialization Fix in Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:** The script produced a `pygame.error: video system not initialized` traceback, indicating that Pygame functions were called before the video system was set up. This happened after restructuring the `_cycle` method.
- **Fix:** Added `pygame.init()` to the `if __name__ == "__main__":` block to ensure Pygame's video system is initialized once at the start of the script. Additionally, `self.clock = pygame.time.Clock()` was moved to the `__init__` method of `DesiredAlgorithmBFSEnv` to ensure it is initialized only once as a class member.
- **Outcome:** The `pygame.error` is resolved, and the simulation should now run correctly across multiple episodes, with proper video system initialization and consistent clock management.

## Progress Log (2025-06-17, Agent Direction Attribute Fix in Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:** The script encountered an `AttributeError: 'Agent' object has no attribute 'direction'`, indicating an incorrect attribute access when trying to retrieve an agent's current direction.
- **Fix:** Corrected the typo in the `_cycle` method of `Desired_algorithm_BFS_Charge.py`, replacing `agent.direction` with `agent.dir` where the predicted next position was calculated. The `agent.dir` attribute correctly stores the agent's current direction as a `Direction` enum.
- **Outcome:** The `AttributeError` is resolved, and agents should now correctly access their direction for path planning and movement, preventing the script from crashing.

## Progress Log (2025-06-17, Negative Battery & Pyglet Close Error Fixes in Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:**
    1. Agent battery levels were dropping below zero.
    2. A `AttributeError: 'CocoaAlternateEventLoop' object has no attribute 'platform_event_loop'` occurred during `env.close()` on macOS.
- **Fix:**
    1. Modified the battery depletion logic in `_cycle` to use `max(0, self.agent_batteries[i] - 1)` for `FORWARD` movements, ensuring battery levels never go below 0.
    2. Wrapped the `self.env.close()` call in a `try-except` block to gracefully handle and suppress the `pyglet`-related `AttributeError` during shutdown.
- **Outcome:** Battery levels will now correctly clamp at 0, and the script will exit without crashing due to the `pyglet` error, allowing for continued debugging of the agent behavior.

## Progress Log (2025-06-17, Removed Redundant Battery Logic from Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:** Agent battery levels were dropping below zero and the agent appeared stuck at the charging station, even after reaching it. This indicated an issue with battery depletion and recharge logic.
- **Fix:** Identified that `Desired_algorithm_BFS_Charge.py` contained redundant battery depletion and recharge logic that was duplicating the functionality already present and correctly handled within the `rware/Comb_warehouse.py` environment's `step` method. Removed all such redundant logic from the `_cycle` method in `Desired_algorithm_BFS_Charge.py`.
- **Outcome:** Battery management is now solely handled by the environment, preventing over-depletion and ensuring correct recharge behavior. This should resolve the negative battery levels and clarify the agent's behavior at charging stations.

## Progress Log (2025-06-17, Agent Battery Synchronization in Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:** Agents were ignoring battery constraints and exhibiting "uncontrollable" behavior (getting stuck at charging stations or not returning to original goals). This was due to the script's internal `self.agent_batteries` list becoming stale and not reflecting the actual battery levels managed by the environment.
- **Fix:** Added a synchronization step within the `_cycle` method of `Desired_algorithm_BFS_Charge.py` after each `env.step()` call: `self.agent_batteries = [agent.battery for agent in self.env.unwrapped.agents]`. This ensures that the script's internal battery tracking is always up-to-date with the environment's true battery state.
- **Outcome:** Agents should now correctly respond to battery depletion by seeking charging stations, and revert to their original goals once fully recharged, as their decision-making will be based on accurate, real-time battery information.

## Progress Log (2025-06-17, Episode Counter Fix in Desired_algorithm_BFS_Charge.py)
- **Problem/Observation presented by user:** The episode counter displayed in the simulation was frozen and not updating correctly at the start of each new episode.
- **Fix:** Re-introduced a local `self.episode` counter in `DesiredAlgorithmBFSEnv`, initialized to 0. This counter is now explicitly incremented at the beginning of the `_cycle` loop (before `env.reset()`). All logging and display statements (`print` and `_display_info`) have been updated to use this local counter, ensuring the displayed episode count accurately reflects the current episode number.
- **Outcome:** The episode counter in the simulation display and terminal output will now accurately reflect the current episode number, incrementing correctly after each episode reset. This resolves the problem of the frozen counter and provides a clear visual indication of episode progression.

## Progress Log (2025-06-17, BFS Charge Algorithm Final Fixes)
- Team reward now only increases by +50 when both agents reach their own goals in the same episode. Per-step, the team reward only accumulates the penalty (i.e., -0.1 * n_agents), and does not sum the individual +10 rewards. Individual rewards remain unchanged and are still given +10 for reaching their own goal.
- The in-simulation overlay now correctly displays the team reward and episode number by setting `self.env._team_reward` and `self.env._episode` before each render call.
- At the start of each episode, all agent goals are forcibly restored to their original goals, and any agent's goal that was set to a charging station is reset to its original goal. This prevents the bug where an agent would treat the charging station as its default goal in the episode after battery depletion at the goal.
- Fixed a KeyError bug by always re-initializing `original_goals` at the start of each episode to match the current environment goals, ensuring no missing indices and robust goal restoration.

## Progress Log (2025-06-17, Final Goal Restoration Bug Fix)
- **Problem:** When an agent's battery was depleted to 0 at the same step it reached its goal, the goal was incorrectly set to the charging station for the next episode, causing the agent to treat the station as its default goal.
- **Fix:** The code was updated so that the true original goals are stored only once after the very first `env.reset()`, and at the start of every episode, `env.goals` are always restored from these original goals. This ensures agents always pursue the correct goals, regardless of battery depletion or temporary goal changes.
- **Additional:** Static obstacles are now correctly initialized at the start of every episode to prevent `AttributeError`.

## Progress Log (2025-06-20, Swapped-Goal Deadlock Back-Off Fix)
- **Problem:** When both agents reached each other's goals simultaneously (agent 0 at agent 1's goal and agent 1 at agent 0's goal), the simulation entered a deadlock loop, with both agents unable to complete the episode.
- **Fix:** Implemented a back-off policy: in the swapped-goal deadlock, agent 1 (higher index) automatically moves away to a free adjacent cell (if possible), while agent 0 waits (NOOP). This allows agent 0 to reach its own goal, resolving the deadlock and ensuring the episode can end successfully.

## Progress Log (2025-06-20, Custom Layout File Support)
- **Feature:** Added support for custom warehouse layouts via text file. Users can now specify a layout file (e.g., BFS_Layout.txt) with obstacles (x), goals (g), and charging stations (C).
- **Implementation:** The script parses the file to set up the environment, goals, and charging stations, enabling flexible experimentation with different layouts and numbers of agents/goals/charging stations.
- **Robustness:** The script remains robust to layout changes and falls back to the default setup if no file is provided.

## Progress Log (2025-06-20, Custom Layout Enforcement and Parsing Fix)
- **Feature:** The environment now always uses the exact layout provided in BFS_Layout.txt (or any custom layout file), with no fallback to defaults.
- **Implementation:** All agent, goal, obstacle, and charging station logic is 100% driven by the layout file. The environment is initialized with the layout string, not the path, ensuring the correct warehouse is rendered.
- **Impact:** This enables true reproducibility and layout-driven experimentation, and ensures the simulation always matches the user-provided layout.

## Progress Log (2025-06-20, Charging Station Contention Deadlock Fix)
- **Problem:** When an agent's battery depletes and its nearest charging station is occupied by another agent, a deadlock could occur as both agents contend for the same station.
- **Solution:** The agent now performs NOOP for one step if the station is occupied, then proceeds to the station once it is free. This prevents deadlock and ensures smooth multi-agent charging behavior.

## Progress Log (2025-06-20, Persistent Goal/Charging Station Bug Fix)
- **Problem:** If an agent's battery depleted at its goal, the goal could be set to a charging station and persist into the next episode, causing the agent to treat the station as its default goal.
- **Solution:** At the start of every episode, the true original goals from the layout file are always restored for each agent, never allowing a charging station to persist as a goal. This ensures agents never confuse a charging station for their default goal, even after battery depletion at the goal in the previous episode.

## Progress Log (2025-06-20, Multi-Agent Goal Restoration Robustness)
- **Problem:** Previous fixes only restored the original goal for agent 0. Agent 1 could still have its goal set to a charging station, causing deadlocks or incorrect episode termination.
- **Solution:** At the start of every episode, all agents' goals are now forcibly restored to their true original goals from the layout file, never to a charging station. This guarantees correct behavior for all agents, even if they ended the previous episode at a charging station.

## Progress Log (2025-06-20, Persistent Battery Grace Period Fix)
- **Problem:** Agents with battery=0 at the end of an episode would immediately switch to charging mode at the start of the next episode, even if they were at their goal, causing incorrect behavior and deadlocks.
- **Solution:** Updated the per-step logic so that agents only switch to charging mode if their battery is 0, they are not already going to charge, and they are NOT at their original goal. This implements a 'grace period' at episode start, ensuring agents always pursue their true goals at the start of each episode, regardless of battery state. Agents only seek charging stations if they need to move and have no battery, matching the intended persistent battery logic.

## Progress Log (2025-06-20, Multi-Agent Generalization & Battery Fixes)
- **Battery max is now correctly set for all agents after every env.reset(), and is used everywhere in the simulation.**
- **After charging, agents always restore their goal to the original goal, regardless of position.**
- **The deadlock/backoff logic is now generalized for any number of agents:** for every pair of agents swapped at each other's goals, the lower-index agent waits and the higher-index agent moves away if possible.
- **All per-agent logic is robust for n_agents > 2.**
- These changes ensure correct multi-agent, multi-battery, and deadlock handling for arbitrary team sizes.

## Progress Log (2025-06-20, Persistent Battery & Robust Goal Restoration)
- **Battery is no longer reset at the start of each episode; agent battery persists across episodes unless recharged at a charging station.**
- **After charging, agents always restore their goal to the original goal, and this is now robust for all agents.**
- **battery_max is used everywhere battery is restored, but not at episode start.**
- These changes ensure correct persistent battery logic and robust goal restoration after charging for all agents.

## Progress Log (2025-06-20, Final Fixes & Global Step Counter)
- **Fixed battery_max propagation:** battery_max is now passed from the script to the environment and to each agent, so battery initialization and recharge always use the correct value.
- **Robust goal restoration after charging:** After charging, agents always restore their goal to the original goal, both in the script and in the environment, for all agents.
- **Global step counter:** Added a global step counter that increments every step across all episodes, shown in both the report and the rendered simulation overlay (below the episode and step counters).
- These changes complete the robust, configurable, and fully instrumented multi-agent warehouse simulation, ready for systematic testing and benchmarking.

## Progress Log (2025-06-20, Max Episodes Trigger Fix & Finalization)
- **Fixed `--max_episodes` termination logic:** The check for `max_episodes` now runs *after* an episode finishes, not before it starts. This ensures the simulation runs for the exact number of specified episodes (e.g., `--max_episodes 20` runs up to and including episode 20).
- **Ready for testing:** The simulation is now fully instrumented and finalized for the testing and benchmarking phase.

## Progress Log (2025-06-20, Creation of No-Charge BFS Script for Testing)
- **Created `Desired_algorithm_BFS_No_Charge.py`:** A new script was created for testing and benchmarking without battery or charging station constraints.
- **Functionality:** This script is a direct copy of `Desired_algorithm_BFS_Charge.py` but with all battery and charging logic removed. All other features, including episodic goal-reaching, deadlock resolution, and termination triggers (`--max_episodes`), are identical.
- **Purpose:** This provides a simplified environment for comparing agent performance with and without battery constraints, facilitating the final testing phase.

## Progress Log (2025-06-12, Structured Test Reporting)
- **New Test Reporting System:**
    - Implemented a structured reporting system for both `Desired_algorithm_BFS_Charge.py` and `Desired_algorithm_BFS_No_Charge.py`.
    - Each test run now generates a detailed output file in a new `Test_Results_Logs/` directory, organized into subdirectories for each script version (`BFS_Charge_Play` and `BFS_No_Charge_Play`).
- **Run Identification:**
    - Added a `--run_id` command-line argument to uniquely name each output file (e.g., `run_output_<run_id>.txt`), making it easier to track and compare test results.
- **Detailed Metrics Tracking:**
    - The new reports capture key performance indicators for each episode:
        - `Steps taken`: The number of steps each agent took to reach its goal.
        - `Recharges`: The number of times each agent recharged its battery (specific to the charge script).
        - `Episode Duration`: The total number of steps in the episode.
        - `Individual Rewards`: The final episodic reward for each agent.
        - `Team Reward`: The final team reward for the episode.
- **Formatted Output:**
    - The output file is formatted for clarity and easy parsing, with a summary section that includes total accumulated team reward, total steps, and the total number of episodes for the entire run.
- **Code Cleanup:**
    - Removed the old, less detailed, per-step logging in favor of the comprehensive end-of-run report.

## Progress Log (2025-06-21, Reporting Logic Fixes)
- **Problem/observation presented by user:**
    - The `Steps taken` metric in the output reports was inaccurate. It sometimes showed `-1` for an agent even when the episode completed successfully, indicating the agent's goal completion was not recorded.
    - The `Recharges` metric only counted deliberate charging missions and missed instances where an agent passed over a charging station incidentally.
- **Fixes and updates:**
    - **`Steps taken`:** Corrected a timing issue in the reporting logic. The check for an agent reaching its goal is now performed *after* the environment takes a step, ensuring that the final move to a goal is always captured correctly. This fix was applied to both the charge and no-charge scripts.
    - **`Recharges`:** The recharge counting mechanism in `Desired_algorithm_BFS_Charge.py` was completely reworked. It now detects *any* instance of an agent's battery increasing by comparing battery levels before and after each step, providing a comprehensive count of all recharge events.
- **Outcome:** The output reports are now accurate and reliable for testing and analysis.

## Progress Log (2025-06-21, Output File Corruption Fix)
- **Problem/observation presented by user:**
    - The structured report files were being corrupted with extraneous text, including the Pygame community support message and the per-episode `print` statements that were intended for terminal debugging. This broke the intended format.
- **Fixes and updates:**
    - **Suppressed `print` statements:** The per-episode `print` calls in both scripts have been commented out to prevent them from being redirected into the report file.
    - **Silenced Pygame Prompt:** Added `os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"` before `pygame.init()` to stop the "Hello from the pygame community" message from appearing in the output.
    - **Standardized Filenames:** Ensured the report filenames are generated directly from the `--run_id` argument (e.g., `MyRun.txt`) as requested.
- **Outcome:** The report files are no longer corrupted and now adhere strictly to the specified structured format, making them clean and ready for automated parsing and analysis.

*** Progress Log (2025-06-21, Automated Testing Framework for Battery-Aware BFS) ***
Testing Plan Implementation:
  -Defined a comprehensive testing matrix to evaluate the performance of BFS agents under different layout configurations and battery constraints.
  Configurations include:
    -Obstacle densities: 10%, 30%, and 50%
    -Agent counts: 2, 4, and 6 (with charging station count = agents / 2)
    -Battery levels: 15, 25, 50 (only for battery-aware tests)
    -Script versions: Desired_algorithm_BFS_Charge.py and Desired_algorithm_BFS_No_Charge.py
  -Total test cases: 27 for charge script and 9 for no-charge script.
Custom Layouts:
  -Manually created and validated 18 distinct layout .txt files reflecting the above configurations.
  -Each layout encodes agent starting positions, obstacles, goal positions, and charging stations (if any).
  -Layouts were hand-edited to ensure no unreachable goals or completely blocked paths, especially in dense configurations (6 agents with 50% obstacles).
-Logging Format Finalized:
  -Standardized the per-run log structure to include:
    -Run ID, episode-wise Steps taken, Recharges, Episode Duration, Individual Rewards, and Team Reward.
    -Final summary: Total Accumulated Team Reward, Total Steps Taken, and Total Number of Episodes.
  -Structured logging is designed to support easy parsing and batch analysis.
Automation Scripts:
  -Implemented an automated Python runner to execute all configurations in sequence with correct CLI arguments and run IDs.
  -Scripts store result logs in:
    -Test_Results_Logs/BFS_Charge_Play/
    -Test_Results_Logs/BFS_No_Charge_Play/
  -Identifiers include key info: agent count, obstacle percentage, charging station count, and battery level.
Execution Notes:
  -Due to compatibility issues and layout limitations (particularly for A6-O50 configurations), some layouts were revised and re-tested manually.
  -All test cases were executed manually using the prepared script arguments due to instabilities in automated looping.
  -Final results are stored in individual .txt files corresponding to each configuration, ready for parsing.