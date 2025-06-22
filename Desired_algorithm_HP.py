"""
Human play and evaluation script for Comb_warehouse.py

Controls:
- Up Arrow: move current agent forward
- Left/Right Arrow: rotate current agent left/right
- SPACE: do nothing
- TAB: change the current agent
- R: reset the environment
- H: show help
- D: display agent info (per step)
- ESC: exit

Features:
- 2 agents, 2 goals, 1 charging station (green tile)
- Individual rewards reset each episode, team reward is cumulative
- No max steps; episode ends only on ESC
- Live display of individual and team rewards
- Battery logic: agents lose 1 battery per move, recharge to max (10) on charging station
- Per-step reports saved to Desired_algorithm/Human_Play, only final rewards printed to terminal
"""
import numpy as np
import pygame
import os
from datetime import datetime
from rware.Comb_warehouse import HRPFTeamWarehouse, Action, RewardType

# ==== Environment Setup ====
class DesiredAlgorithmEnv:
    def __init__(self):
        # Layout with 1 charging station (green tile at (2,2))
        self.env = HRPFTeamWarehouse(
            shelf_columns=3,
            column_height=3,
            shelf_rows=2,
            n_agents=2,
            msg_bits=0,
            sensor_range=1,
            request_queue_size=1,
            max_inactivity_steps=None,
            max_steps=10000,  # Large number, but we terminate on ESC
            reward_type=RewardType.INDIVIDUAL,
            render_mode="human",
        )
        self.n_agents = self.env.unwrapped.n_agents
        self.running = True
        self.current_agent_index = 0
        self.current_action = None
        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.team_cumulative_reward = 0.0
        self.display_info = False  # Suppress per-step terminal output
        self._agent_accumulated_rewards = [0.0] * self.n_agents
        self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
        self.env._team_reward = 0.0
        self.battery_max = 10
        self.episode = 1  # Start episode counter at 1
        self._init_batteries()
        self._setup_report_file()
        self._help()
        self._cycle()

    def _help(self):
        print("Use the up arrow key to move the current agent forward")
        print("Use the left/ right arrow keys to rotate the current agent left/ right")
        print("Use the SPACE key to do nothing")
        print("Press TAB to change the current agent")
        print("Press R to reset the environment")
        print("Press H to show help")
        print("Press D to display agent info")
        print("Press ESC to exit")
        print()

    def _init_batteries(self):
        self.agent_batteries = [self.battery_max for _ in range(self.n_agents)]

    def _setup_report_file(self):
        report_dir = os.path.join("Desired_algorithm", "Human_Play")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = os.path.join(report_dir, f"human_play_{timestamp}.txt")
        self.report_lines = []

    def _get_current_agent_info(self):
        agent = self.env.unwrapped.agents[self.current_agent_index]
        agent_x = agent.x
        agent_y = agent.y
        agent_str = f"Agent {self.current_agent_index + 1} (at row {agent_y + 1}, col {agent_x + 1})"
        return agent_str

    def _display_info(self, obss, rews, done):
        info_str = (
            f"Step {self.t}:\n"
            f"\tSelected: {self._get_current_agent_info()}\n"
            f"\tObs: {obss[self.current_agent_index]}\n"
            f"\tRew: {round(rews[self.current_agent_index], 3)}\n"
            f"\tDone: {done}\n"
            f"\tAccumulated Rew: {self._agent_accumulated_rewards[self.current_agent_index]:.1f}\n"
            f"\tTeam Reward (cumulative): {self.team_cumulative_reward:.1f}\n"
            f"\tBattery: {self.agent_batteries[self.current_agent_index]}\n"
        )
        self.report_lines.append(info_str)

    def _increment_current_agent_index(self, index):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _is_on_charging_station(self, agent):
        # Charging station at (2,2) (row,col)
        return (agent.x, agent.y) == (2, 2)

    def _cycle(self):
        pygame.init()
        clock = pygame.time.Clock()
        obss, info = self.env.reset()
        rews = [0] * self.n_agents
        done = False
        self.team_cumulative_reward = 0.0
        self._init_batteries()
        self.env._episode = self.episode
        self._team_bonus_given = False  # Reset bonus flag at start of each episode
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.current_action = Action.LEFT
                    elif event.key == pygame.K_RIGHT:
                        self.current_action = Action.RIGHT
                    elif event.key == pygame.K_UP:
                        self.current_action = Action.FORWARD
                    elif event.key == pygame.K_SPACE:
                        self.current_action = Action.NOOP
                    elif event.key == pygame.K_TAB:
                        self.current_action = None
                        self.current_agent_index = self._increment_current_agent_index(self.current_agent_index)
                    elif event.key == pygame.K_r:
                        self.current_action = None
                        obss, info = self.env.reset()
                        self.ep_returns = np.zeros(self.n_agents)
                        self.t = 0
                        self._agent_accumulated_rewards = [0.0] * self.n_agents
                        self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
                        self._init_batteries()
                    elif event.key == pygame.K_h:
                        self.current_action = None
                        self._help()
                    elif event.key == pygame.K_d:
                        self.current_action = None
                        # Print last step info to terminal
                        if self.report_lines:
                            print(self.report_lines[-1])
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        self.current_action = None
            if self.current_action is not None:
                actions = [Action.NOOP] * self.n_agents
                actions[self.current_agent_index] = self.current_action
                actions = [a.value for a in actions]  # Convert to int
                obss, rews, done, info = self.env.step(actions)
                self.ep_returns += np.array(rews)
                self._agent_accumulated_rewards[self.current_agent_index] += rews[self.current_agent_index]
                # Battery logic: deplete 1 per move, recharge if on charging station
                if self.current_action == Action.FORWARD:
                    self.agent_batteries[self.current_agent_index] = max(0, self.agent_batteries[self.current_agent_index] - 1)
                agent = self.env.unwrapped.agents[self.current_agent_index]
                if self._is_on_charging_station(agent):
                    self.agent_batteries[self.current_agent_index] = self.battery_max
                self.t += 1
                # Team reward: accumulate linearly per step
                if not info.get('team_completed', False):
                    self.team_cumulative_reward += -0.1
                elif info.get('team_completed', False) and not self._team_bonus_given:
                    self.team_cumulative_reward += 50.0
                    self._team_bonus_given = True
                self.env._team_reward = self.team_cumulative_reward
                self._display_info(obss, rews, done)
                if done:
                    # Print only final rewards to terminal
                    print(f"Episode finished. Episodic returns: {[round(ret, 3) for ret in self.ep_returns]}")
                    print(f"Team Reward (cumulative): {self.team_cumulative_reward:.1f}")
                    obss, info = self.env.reset()
                    self.ep_returns = np.zeros(self.n_agents)
                    self.t = 0
                    self._agent_accumulated_rewards = [0.0] * self.n_agents
                    self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
                    # Do NOT reset team_cumulative_reward; keep cumulative across episodes
                    self._init_batteries()
                    self.episode += 1
                    self.env._episode = self.episode
                    self._team_bonus_given = False  # Reset for next episode
                self.current_action = None
            self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
            self.env._team_reward = self.team_cumulative_reward
            self.env._agent_batteries = self.agent_batteries
            self.env.render()
            clock.tick(10)
        # Save report at end of session
        with open(self.report_path, "w") as f:
            f.writelines(self.report_lines)
        try:
            self.env.close()
        except Exception as e:
            print(f"Warning: Exception on env.close(): {e}")
        pygame.quit()

if __name__ == "__main__":
    DesiredAlgorithmEnv() 