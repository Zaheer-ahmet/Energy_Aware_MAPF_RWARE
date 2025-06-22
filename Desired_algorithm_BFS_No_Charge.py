"""
Automated BFS play and evaluation script for Comb_warehouse.py (No Charging)

Features:
- Configurable number of agents (via --n_agents)
- 2+ agents, 2+ goals
- Each agent automatically finds a path to its goal using BFS and follows it
- Individual rewards reset each episode, team reward is cumulative
- No max steps; episode ends only on ESC or max_episodes
- Live display of individual and team rewards
- Per-step reports saved to Desired_algorithm/BFS_Play, only final rewards printed to terminal

Note: The layout file must have at least as many goals ('g') as n_agents.
"""
import numpy as np
import pygame
import os
from datetime import datetime
from collections import deque
from rware.Comb_warehouse import HRPFTeamWarehouse, Action, RewardType, Direction
import argparse

# ==== Custom Environment to Hide Battery ====
class HRPFTeamWarehouseNoBatteryDisplay(HRPFTeamWarehouse):
    def render(self):
        # Override the _agent_batteries attribute with empty strings
        # to prevent the renderer from displaying battery counts.
        self._agent_batteries = [''] * self.n_agents
        
        if self.render_mode != 'human':
            return # Support headless mode
        if not self.renderer:
            from rware.rendering import Viewer
            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=False)

# ==== BFS Pathfinding (refactored) ====
def bfs(env, start, goal, obstacles, grid=None, grid_size=None, using_custom_layout=False):
    """Return a list of (x, y) positions from start to goal using BFS, or [] if unreachable."""
    queue = deque()
    queue.append((start, []))
    visited = set()
    visited.add(start)
    while queue:
        (x, y), path = queue.popleft()
        if (x, y) == goal:
            return path
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if using_custom_layout:
                if 0 <= nx < grid_size[1] and 0 <= ny < grid_size[0] and grid[ny][nx] == '.' and (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path+[(nx, ny)]))
            else:
                if 0 <= nx < env.grid_size[1] and 0 <= ny < env.grid_size[0] and env._is_highway(nx, ny) and (nx, ny) not in obstacles and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path+[(nx, ny)]))
    return []

def get_static_obstacles(env, grid=None, grid_size=None, using_custom_layout=False):
    obstacles = set()
    if using_custom_layout:
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                if grid[y][x] == 'x':
                    obstacles.add((x, y))
    else:
        for y in range(env.grid_size[0]):
            for x in range(env.grid_size[1]):
                if not env._is_highway(x, y):
                    obstacles.add((x, y))
    return obstacles

def get_goal(env, agent_idx, goals=None, using_custom_layout=False):
    if using_custom_layout:
        return goals[agent_idx]
    else:
        return env.goals[agent_idx]

def get_station_xy(station):
    # Helper to get (x, y) from either a tuple or an object with .x/.y
    if isinstance(station, tuple):
        return station[0], station[1]
    else:
        return station.x, station.y

# ==== Layout Parsing ====
def parse_layout_file(layout_path):
    grid = []
    goals = []
    charging_stations = []
    with open(layout_path, 'r') as f:
        for y, line in enumerate(f):
            row = []
            for x, char in enumerate(line.strip()):
                if char == 'x':
                    row.append('x')
                elif char == 'g':
                    row.append('.')
                    goals.append((x, y))
                elif char == 'C':
                    row.append('.')
                    charging_stations.append((x, y))
                else:
                    row.append('.')
            grid.append(row)
    grid_size = (len(grid), len(grid[0]) if grid else 0)
    return grid, goals, charging_stations, grid_size

# ==== Environment Setup ====
class DesiredAlgorithmBFSEnv:
    def __init__(self, layout_path=None, n_agents=None, **kwargs):
        if not layout_path:
            raise ValueError("A custom layout file (e.g., BFS_Layout.txt) must be provided via --layout. Default layout logic is disabled.")
        # Read the layout file as a string
        with open(layout_path, 'r') as f:
            layout_str = f.read()
        grid, goals, _, grid_size = parse_layout_file(layout_path)
        self.using_custom_layout = True
        self.grid = grid
        self.goals = goals
        self.grid_size = grid_size
        if n_agents is None:
            self.n_agents = len(goals)
        else:
            if len(goals) < n_agents:
                raise ValueError(f"Layout file has only {len(goals)} goals but n_agents={n_agents}. Please add more goals to the layout.")
            self.n_agents = n_agents
        self.env = HRPFTeamWarehouseNoBatteryDisplay(
            shelf_columns=grid_size[1]//2, # rough estimate
            column_height=grid_size[0]//2, # rough estimate
            shelf_rows=1, # not used with custom layout
            n_agents=self.n_agents,
            msg_bits=0,
            sensor_range=1,
            request_queue_size=1,
            max_inactivity_steps=None,
            max_steps=10000,
            reward_type=RewardType.INDIVIDUAL,
            render_mode="human",
            layout=layout_str,
        )
        self.running = True
        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.team_cumulative_reward = 0.0
        self._agent_accumulated_rewards = [0.0] * self.n_agents
        self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
        self.env._team_reward = 0.0
        self.original_goals = None  # To store original goals for each agent (set only once)
        self.episode = 0 # Initialize local episode counter
        self.global_step = 0 # Global step counter across all episodes
        self.max_episodes = kwargs.get('max_episodes', None) # Max episodes before termination
        self.run_id = kwargs.get('run_id', 'unspecified_run') # Run ID for output file
        self.clock = pygame.time.Clock() # Initialize clock once
        self._setup_report_file()
        self.episode_data = [] # To store structured data for the final report
        self._cycle()

    def _setup_report_file(self):
        report_dir = os.path.join("Test_Results_Logs", "BFS_No_Charge_Play")
        os.makedirs(report_dir, exist_ok=True)
        self.report_path = os.path.join(report_dir, f"{self.run_id}.txt")

    def _display_info(self, obss, rews, done):
        # This function is now for live debugging only; structured report is saved at the end.
        pass

    def _get_agent_action(self, agent_idx, path, agent_dir):
        if not path:
            return Action.NOOP
        
        agent = self.env.unwrapped.agents[agent_idx]
        x, y = agent.x, agent.y
        current_dir = agent_dir # This is a Direction enum object
        next_x, next_y = path[0]

        # Determine desired direction (as a Direction enum object)
        if next_x > x:
            desired_dir = Direction.RIGHT
        elif next_x < x:
            desired_dir = Direction.LEFT
        elif next_y > y:
            desired_dir = Direction.DOWN # Grid Y increases downwards
        elif next_y < y:
            desired_dir = Direction.UP # Grid Y decreases upwards
        else:
            return Action.NOOP
        
        if current_dir == desired_dir:
            return Action.FORWARD
        
        # Calculate turns using wraplist indices
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_idx = wraplist.index(current_dir)
        desired_idx = wraplist.index(desired_dir)
        
        next_right_idx = (current_idx + 1) % len(wraplist)
        next_left_idx = (current_idx - 1 + len(wraplist)) % len(wraplist)

        # Check for RIGHT turn (clockwise one step)
        if next_right_idx == desired_idx:
            return Action.RIGHT
        # Check for LEFT turn (counter-clockwise one step)
        elif next_left_idx == desired_idx:
            return Action.LEFT
        # 180-degree turn (two steps, prefer RIGHT)
        elif (current_idx + 2) % len(wraplist) == desired_idx:
            return Action.RIGHT
        else:
            return Action.NOOP

    def _cycle(self):
        running = True
        while running:
            self.t = 0
            self.episode += 1
            obss, info = self.env.reset()
            # On the very first episode only, set original_goals from the layout file, never from env.goals
            if self.original_goals is None:
                self.original_goals = {i: self.goals[i] for i in range(self.n_agents)}
            # Always restore env.goals to original_goals at episode start for all agents
            for i in range(self.n_agents):
                self.env.goals[i] = self.original_goals[i]
            self.static_obstacles = get_static_obstacles(self.env, self.grid if self.using_custom_layout else None, self.grid_size if self.using_custom_layout else None, self.using_custom_layout)
            self._team_bonus_given = False
            self.ep_returns = np.zeros(self.n_agents)
            self._agent_accumulated_rewards = [0.0] * self.n_agents
            self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
            self.env._team_reward = 0.0
            
            # Metrics for structured report
            steps_taken = [-1] * self.n_agents

            episode_done = False
            while not episode_done and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                if not running:
                    break
                self.t += 1
                self.global_step += 1
                agent_actions = [Action.NOOP] * self.n_agents
                agent_paths = [[]] * self.n_agents
                predicted_next_positions = [None] * self.n_agents
                approved_positions = set()
                for i in range(self.n_agents):
                    agent = self.env.unwrapped.agents[i]
                    current_agent_pos = (agent.x, agent.y)

                    # This is now checked after the step to be accurate
                    # if current_agent_pos == self.original_goals[i] and steps_taken[i] == -1:
                    #     steps_taken[i] = self.t

                    goal_for_path_planning = self.original_goals[i]
                    self.env.goals[i] = goal_for_path_planning
                    if current_agent_pos == self.original_goals[i]:
                        agent_paths[i] = []
                        continue
                    dynamic_obstacles = set(p for j, p in enumerate([(a.x, a.y) for a in self.env.unwrapped.agents]) if j != i)
                    combined_obstacles = self.static_obstacles.union(dynamic_obstacles)
                    path = bfs(self.env, current_agent_pos, goal_for_path_planning, combined_obstacles, self.grid if self.using_custom_layout else None, self.grid_size if self.using_custom_layout else None, self.using_custom_layout)
                    agent_paths[i] = path
                for i in range(self.n_agents):
                    agent = self.env.unwrapped.agents[i]
                    chosen_action = Action.NOOP # Default action
                    if agent_paths[i]: # If a path exists
                        action = self._get_agent_action(i, agent_paths[i], agent.dir)
                        predicted_pos = (agent.x, agent.y)
                        if action == Action.FORWARD:
                            if agent.dir == Direction.UP:
                                predicted_pos = (agent.x, agent.y - 1)
                            elif agent.dir == Direction.DOWN:
                                predicted_pos = (agent.x, agent.y + 1)
                            elif agent.dir == Direction.LEFT:
                                predicted_pos = (agent.x - 1, agent.y)
                            elif agent.dir == Direction.RIGHT:
                                predicted_pos = (agent.x + 1, agent.y)
                        # Conflict Resolution: Higher index waits for lower index
                        # Check if predicted position conflicts with already approved positions
                        if action == Action.FORWARD and predicted_pos in approved_positions:
                            chosen_action = Action.NOOP # Wait if there's a conflict
                        else:
                            chosen_action = action
                            if chosen_action == Action.FORWARD:
                                approved_positions.add(predicted_pos) # Approve this position for higher priority agents
                    agent_actions[i] = chosen_action
                # Generalized deadlock/backoff logic for n_agents >= 2
                for i in range(self.n_agents):
                    for j in range(i+1, self.n_agents):
                        ai, aj = self.env.unwrapped.agents[i], self.env.unwrapped.agents[j]
                        gi, gj = self.original_goals[i], self.original_goals[j]
                        posi, posj = (ai.x, ai.y), (aj.x, aj.y)
                        # If agents are swapped at each other's goals and neither is at their own goal
                        if posi == gj and posj == gi and posi != gi and posj != gj:
                            # Lower index agent waits, higher index agent moves away if possible
                            agent_actions[i] = Action.NOOP
                            found_move = False
                            for dx, dy, desired_dir in [(-1,0,Direction.LEFT),(1,0,Direction.RIGHT),(0,-1,Direction.UP),(0,1,Direction.DOWN)]:
                                nx, ny = aj.x+dx, aj.y+dy
                                if 0 <= nx < self.env.grid_size[1] and 0 <= ny < self.env.grid_size[0] and self.env._is_highway(nx, ny):
                                    # Not occupied by agent i or agent j's own goal
                                    if (nx, ny) != posi and (nx, ny) != gj:
                                        # Set agent j's action to turn/move toward (nx, ny)
                                        if aj.dir == desired_dir:
                                            agent_actions[j] = Action.FORWARD
                                        else:
                                            wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
                                            current_idx = wraplist.index(aj.dir)
                                            desired_idx = wraplist.index(desired_dir)
                                            next_right_idx = (current_idx + 1) % len(wraplist)
                                            next_left_idx = (current_idx - 1 + len(wraplist)) % len(wraplist)
                                            if next_right_idx == desired_idx:
                                                agent_actions[j] = Action.RIGHT
                                            elif next_left_idx == desired_idx:
                                                agent_actions[j] = Action.LEFT
                                            elif (current_idx + 2) % len(wraplist) == desired_idx:
                                                agent_actions[j] = Action.RIGHT
                                            else:
                                                agent_actions[j] = Action.NOOP
                                        found_move = True
                                        break
                            if not found_move:
                                agent_actions[j] = Action.NOOP
                # Execute actions
                obss, rews, done, info = self.env.step(agent_actions)
                # Update per-agent rewards
                self._agent_accumulated_rewards = [a + r for a, r in zip(self._agent_accumulated_rewards, rews)]
                self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
                self.ep_returns += np.array(rews)
                # Update team cumulative reward: only per-step penalty, not individual +10 rewards
                if not hasattr(self, 'team_cumulative_reward'):
                    self.team_cumulative_reward = 0.0
                self.team_cumulative_reward += -0.1 * self.n_agents

                # New steps_taken logic: check after the step is taken
                for i, agent in enumerate(self.env.unwrapped.agents):
                    if (agent.x, agent.y) == self.original_goals[i] and steps_taken[i] == -1:
                        steps_taken[i] = self.t
                        
                # Only end the episode when both agents are at their own original goals
                agents_at_own_goal = [
                    (agent.x, agent.y) == self.original_goals[i]
                    for i, agent in enumerate(self.env.unwrapped.agents)
                ]
                if all(agents_at_own_goal) and not self._team_bonus_given:
                    self.team_cumulative_reward += 50
                    self._team_bonus_given = True
                    # End the episode
                    done = True
                # Set team reward and episode for renderer overlay
                self.env._team_reward = self.team_cumulative_reward
                self.env._episode = self.episode
                self.env._global_step = self.global_step  # For renderer overlay if supported
                self._display_info(obss, rews, done)
                self.env.render()
                self.clock.tick(1000)
                if done:
                    print(f"Episode {self.episode} finished. Episodic returns: {self.ep_returns}")
                    print(f"Team Reward (cumulative): {self.team_cumulative_reward}")
                    
                    # Store data for this episode
                    self.episode_data.append({
                        "episode": self.episode,
                        "steps_taken": steps_taken,
                        "recharges": [0] * self.n_agents, # Placeholder for no-charge version
                        "episode_duration": self.t,
                        "individual_rewards": self.ep_returns,
                        "team_reward": self.team_cumulative_reward
                    })
                    episode_done = True
            
            # Check for max episodes termination *after* the episode is fully finished
            if self.max_episodes is not None and self.episode >= self.max_episodes:
                running = False
                
            if not running: # If the inner loop broke due to QUIT/ESCAPE, break the outer loop too
                break
        self._save_report()
        try:
            self.env.close()
        except Exception as e:
            print(f"Warning: Exception on env.close(): {e}")

    def _save_report(self):
        with open(self.report_path, "w") as f:
            f.write(f"Run ID: {self.run_id}\n\n")

            total_team_reward = 0
            for data in self.episode_data:
                f.write(f"Episode {data['episode']}:\n")
                f.write(f"Steps taken: {data['steps_taken']}\n")
                f.write(f"Recharges: {data['recharges']}\n")
                f.write(f"Episode Duration: {data['episode_duration']}\n")
                f.write(f"Individual Rewards: {data['individual_rewards'].tolist()}\n")
                f.write(f"Team Reward: {data['team_reward']}\n\n")
                total_team_reward += data['team_reward']

            f.write("\n")
            f.write(f"Total Accumulated Team Reward: {total_team_reward}\n")
            f.write(f"Total Steps Taken: {self.global_step}\n")
            f.write(f"Total Number of Episodes: {self.episode}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=str, required=True, help='Path to custom layout file (e.g., BFS_Layout.txt). Must have at least as many goals (g) as agents.')
    parser.add_argument('--n_agents', type=int, default=None, help='Number of agents to use (default: number of goals in layout)')
    parser.add_argument('--max_episodes', type=int, default=None, help='Number of episodes to run before terminating (default: run indefinitely)')
    parser.add_argument('--run_id', type=str, default='test_run', help='A unique identifier for this run')
    args = parser.parse_args()
    pygame.init() # Initialize pygame once at the start
    game = DesiredAlgorithmBFSEnv(layout_path=args.layout, n_agents=args.n_agents, max_episodes=args.max_episodes, run_id=args.run_id) 