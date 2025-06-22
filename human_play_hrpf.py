"""
Use this script to interactively play HRPFTeamWarehouse

Controls:
- Up Arrow: move current agent forward
- Left/Right Arrow: rotate current agent left/right
- SPACE: do nothing
- TAB: change the current agent
- R: reset the environment
- H: show help
- D: display agent info (per step)
- ESC: exit
"""
from argparse import ArgumentParser
import warnings
import numpy as np
import pygame
from rware.hrpf_team_warehouse import HRPFTeamWarehouse, Action, RewardType

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--n_agents",
        type=int,
        default=2,
        help="Number of agents.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--display_info",
        action="store_true",
        help="Display agent info per step",
    )
    return parser.parse_args()

class InteractiveHRPFEnv:
    def __init__(self, n_agents, max_steps, display_info=True):
        self.env = HRPFTeamWarehouse(
            shelf_columns=3,
            column_height=5,
            shelf_rows=2,
            n_agents=n_agents,
            msg_bits=0,
            sensor_range=1,
            request_queue_size=1,
            max_inactivity_steps=None,
            max_steps=max_steps,
            reward_type=RewardType.INDIVIDUAL,
            render_mode="human",
        )
        self.n_agents = self.env.unwrapped.n_agents
        self.running = True
        self.current_agent_index = 0
        self.current_action = None
        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.reset = False
        self.display_info = display_info
        self._agent_accumulated_rewards = [0.0] * self.n_agents
        self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
        obss, _ = self.env.reset()
        self.env.render()
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

    def _get_current_agent_info(self):
        agent = self.env.unwrapped.agents[self.current_agent_index]
        agent_x = agent.x
        agent_y = agent.y
        agent_str = f"Agent {self.current_agent_index + 1} (at row {agent_y + 1}, col {agent_x + 1})"
        return agent_str

    def _display_info(self, obss, rews, done):
        print(f"Step {self.t}:")
        print(f"\tSelected: {self._get_current_agent_info()}")
        print(f"\tObs: {obss[self.current_agent_index]}")
        print(f"\tRew: {round(rews[self.current_agent_index], 3)}")
        print(f"\tDone: {done}")
        print(f"\tAccumulated Rew: {self._agent_accumulated_rewards[self.current_agent_index]:.1f}")
        print()

    def _increment_current_agent_index(self, index):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _cycle(self):
        pygame.init()
        clock = pygame.time.Clock()
        obss, _ = self.env.reset()
        rews = [0] * self.n_agents
        done = False
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
                        if self.display_info:
                            print(f"Now selected: {self._get_current_agent_info()}")
                    elif event.key == pygame.K_r:
                        self.current_action = None
                        obss, _ = self.env.reset()
                        self.ep_returns = np.zeros(self.n_agents)
                        self.t = 0
                        self._agent_accumulated_rewards = [0.0] * self.n_agents
                        self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
                        if self.display_info:
                            self._display_info(obss, [0] * self.n_agents, False)
                    elif event.key == pygame.K_h:
                        self.current_action = None
                        self._help()
                    elif event.key == pygame.K_d:
                        self.current_action = None
                        self.display_info = not self.display_info
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        self.current_action = None
                        warnings.warn(f"Key {event.key} not recognized")
            if self.current_action is not None:
                actions = [Action.NOOP] * self.n_agents
                actions[self.current_agent_index] = self.current_action
                actions = [a.value for a in actions]  # Convert to int
                obss, rews, done, info = self.env.step(actions)
                self.ep_returns += np.array(rews)
                self._agent_accumulated_rewards[self.current_agent_index] += rews[self.current_agent_index]
                self.t += 1
                if self.display_info:
                    self._display_info(obss, rews, done)
                if done:
                    print(f"Episode finished. Episodic returns: {[round(ret, 3) for ret in self.ep_returns]}")
                    obss, _ = self.env.reset()
                    self.ep_returns = np.zeros(self.n_agents)
                    self.t = 0
                    self._agent_accumulated_rewards = [0.0] * self.n_agents
                    self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
                self.current_action = None
            self.env._agent_accumulated_rewards = self._agent_accumulated_rewards
            self.env.render()
            clock.tick(10)
        try:
            self.env.close()
        except Exception as e:
            print(f"Warning: Exception on env.close(): {e}")
        pygame.quit()

if __name__ == "__main__":
    args = parse_args()
    InteractiveHRPFEnv(
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        display_info=args.display_info,
    ) 