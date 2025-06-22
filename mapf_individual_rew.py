import time
import numpy as np
import gymnasium as gym
from rware.warehouse import Warehouse, Action, RewardType, Direction
import pygame
import os
from datetime import datetime
from collections import deque

# --- Print Direction Enum Info ---
try:
    print("--- RWARE Direction Enum Info ---")
    for direction in Direction:
        print(f"Name: {direction.name}, Value: {direction.value}")
    print("---------------------------------")
except NameError:
    print("Error: Could not access Direction enum after import.")
# --------------------------------

RENDER_DELAY = 0.20
MAX_STEPS = 200
N_AGENTS = 4
LAYOUT_FILE = "Layouts_for_MAPF/custom_layout_rew.txt"
FINAL_STATE_DIR = "final_states/mapf_individual_rew/"

# Define action constants based on probable RWARE mapping
# (Assuming 0:NOOP, 1:FORWARD, 2:LEFT, 3:RIGHT, 4:TOGGLE_LOAD)
# We only need movement actions here.
ACTION_NOOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3

# Define orientation constants (assuming standard RWARE/Gymnasium directions)
# 0: Up, 1: Right, 2: Down, 3: Left
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# --- Utility Functions ---
def load_layout(filename):
    """Loads the layout string from a file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Layout file '{filename}' not found.")
        return None

def find_goals(layout):
    """Finds all goal locations 'g' in the layout."""
    goals = []
    for y, row in enumerate(layout.splitlines()):
        for x, cell in enumerate(row):
            if cell == 'g':
                goals.append((x, y))
    return goals

def find_obstacles(layout):
    """Finds all obstacle locations 'x' in the layout."""
    obstacles = set()
    for y, row in enumerate(layout.splitlines()):
        for x, cell in enumerate(row):
            if cell == 'x':
                obstacles.add((x, y))
    return obstacles

def bfs(start, goal, obstacles, agent_positions, grid_size):
    """
    Performs Breadth-First Search to find the shortest path.
    Avoids obstacles and other agent positions.
    """
    queue = deque([(start, [])])  # ((x, y), path_list_of_tuples)
    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == goal:
            return path # Return the list of coordinates in the path

        # Explore neighbors (Up, Down, Left, Right)
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy

            # Check grid boundaries
            if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                next_pos = (nx, ny)
                # Check if visited, obstacle, or occupied by another agent
                if next_pos not in visited and next_pos not in obstacles and next_pos not in agent_positions:
                    visited.add(next_pos)
                    new_path = path + [next_pos]
                    queue.append((next_pos, new_path))

    return [] # No path found

def select_next_move(agent_idx, agent_pos, agent_orientation, goal, obstacles, all_agent_positions, grid_size):
    """
    Selects the next action (TURN_LEFT, TURN_RIGHT, FORWARD, NOOP) based on BFS path and agent orientation.
    Returns integer actions: 0:NOOP, 1:FORWARD, 2:TURN_LEFT, 3:TURN_RIGHT
    Agent orientation is expected to be of type rware.warehouse.Direction.
    Considers other agents as temporary obstacles for pathfinding.
    """
    # Treat positions of *other* agents as temporary obstacles for this agent's planning
    other_agent_positions = set(all_agent_positions[:agent_idx] + all_agent_positions[agent_idx+1:])

    path = bfs(agent_pos, goal, obstacles, other_agent_positions, grid_size)

    if not path:
        # No path found (or already at goal), stay put
        return ACTION_NOOP

    # Get the next coordinate in the path
    next_step_pos = path[0]

    # Determine target direction (as Direction enum member) to reach next_step_pos
    dx = next_step_pos[0] - agent_pos[0]
    dy = next_step_pos[1] - agent_pos[1]

    target_direction_enum = None # Invalid default
    if dx == 1:
        target_direction_enum = Direction.RIGHT
    elif dx == -1:
        target_direction_enum = Direction.LEFT
    elif dy == 1: # Grid y increases downwards
        target_direction_enum = Direction.DOWN
    elif dy == -1: # Grid y decreases upwards
        target_direction_enum = Direction.UP
    else:
        # This should not happen if path exists and is not empty
        return ACTION_NOOP

    # Compare target direction enum with current orientation enum
    if agent_orientation == target_direction_enum:
        # Facing the right way, move forward
        return ACTION_FORWARD
    else:
        # Need to turn. Determine shortest turn based on current orientation and target.
        current_dir = agent_orientation
        target_dir = target_direction_enum

        # UP=0, DOWN=1, LEFT=2, RIGHT=3
        if current_dir == Direction.UP:
            if target_dir == Direction.RIGHT: return ACTION_TURN_RIGHT
            else: return ACTION_TURN_LEFT # Turn left for LEFT or DOWN
        elif current_dir == Direction.DOWN:
            if target_dir == Direction.LEFT: return ACTION_TURN_RIGHT
            else: return ACTION_TURN_LEFT # Turn left for RIGHT or UP
        elif current_dir == Direction.LEFT:
            if target_dir == Direction.UP: return ACTION_TURN_RIGHT
            else: return ACTION_TURN_LEFT # Turn left for DOWN or RIGHT
        elif current_dir == Direction.RIGHT:
            if target_dir == Direction.DOWN: return ACTION_TURN_RIGHT
            else: return ACTION_TURN_LEFT # Turn left for UP or LEFT

        # Fallback - should not be reached if logic is correct
        return ACTION_NOOP

def compute_individual_reward(agent_idx, final_pos, goal, collided, reached_goal, already_done):
    """Calculates the reward for an agent based on the rules."""
    if already_done:
        return 0.0 # No more rewards or penalties after reaching goal
    if collided:
        return -10.0
    if reached_goal:
        return 10.0 # Only awarded on the step the goal is reached
    # Penalty for any move or stay action when not at the goal
    return -0.1

def save_final_state(env, accumulated_rewards, step, reason):
    """Saves the final agent positions, rewards, step count, and termination reason."""
    os.makedirs(FINAL_STATE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_file = os.path.join(FINAL_STATE_DIR, f"final_state_{timestamp}.txt")

    with open(state_file, 'w') as f:
        f.write(f"Termination Step: {step}\n")
        f.write(f"Termination Reason: {reason}\n")
        f.write("-" * 20 + "\n")
        for i, rew in enumerate(accumulated_rewards):
            agent = env.unwrapped.agents[i]
            f.write(f"Agent {i + 1}: Final Pos=({agent.x}, {agent.y}), Accumulated Reward={rew:.2f}\n")

    # Attempt to save screenshot
    try:
        # Ensure rendering is up to date for the final frame
        if hasattr(env.unwrapped, 'renderer') and env.unwrapped.renderer:
            # The env.render() call in the main loop should suffice for updating
            # Just get the frame directly
            # img = env.unwrapped.renderer.get_frame()
            # from PIL import Image
            # im = Image.fromarray(img)
            # screenshot_file = os.path.join(FINAL_STATE_DIR, f"screenshot_{timestamp}.png")
            # im.save(screenshot_file)
            print(f"Saved final state to {state_file}")
            # print(f"Saved screenshot to {screenshot_file}")
            print("Skipping screenshot saving in human render mode.") # Indicate skipping
        else:
            print("Renderer not available, skipping screenshot.")

    except Exception as e:
        print(f"Error during final state saving (screenshot attempt): {e}") # Adjusted error message


def main():
    """Main simulation loop."""
    pygame.init() # Required for event handling (like ESC key)
    layout = load_layout(LAYOUT_FILE)
    if layout is None:
        pygame.quit()
        return

    goals = find_goals(layout)
    obstacles = find_obstacles(layout)
    grid_width = len(layout.splitlines()[0])
    grid_height = len(layout.splitlines())
    grid_size = (grid_width, grid_height)

    if len(goals) < N_AGENTS:
        print(f"Error: Layout file '{LAYOUT_FILE}' has {len(goals)} goals, but {N_AGENTS} agents were requested.")
        pygame.quit()
        return
    # Assign the first N_AGENTS goals found
    agent_goals = goals[:N_AGENTS]

    env = None # Initialize env to None for finally block
    try:
        env = Warehouse(
            shelf_columns=1, # Irrelevant for this MAPF setup
            column_height=1, # Irrelevant
            shelf_rows=1,    # Irrelevant
            n_agents=N_AGENTS,
            msg_bits=0,      # No communication needed
            sensor_range=1,  # Basic sensor range
            request_queue_size=1, # Irrelevant
            max_inactivity_steps=None, # Agents are always active
            max_steps=MAX_STEPS,
            reward_type=RewardType.INDIVIDUAL,
            layout=layout,
            observation_type=None, # Obs not used by BFS directly
            render_mode="human",
        )

        obs, info = env.reset()
        accumulated_rewards = [0.0] * N_AGENTS
        step = 0
        running = True

        # Ensure the reward attribute exists for the renderer
        if not hasattr(env.unwrapped, '_agent_accumulated_rewards'):
             env.unwrapped._agent_accumulated_rewards = [0.0] * N_AGENTS
        env.unwrapped._agent_accumulated_rewards = list(accumulated_rewards) # Initial rewards

        # --- Main Loop ---
        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("ESC pressed, terminating.")
                    running = False
            if not running:
                break # Exit loop if ESC was pressed

            # --- Agent Action Selection ---
            current_agent_positions = [(env.unwrapped.agents[i].x, env.unwrapped.agents[i].y) for i in range(N_AGENTS)]
            current_orientations = []
            try:
                for i in range(N_AGENTS):
                    current_orientations.append(env.unwrapped.agents[i].dir)
            except AttributeError:
                 print("\nError: Could not access agent orientation (tried 'agent.orientation' and 'agent.dir').\n"
                       "Agent movement logic requires orientation. Cannot proceed.\n"
                       "Check RWARE documentation or Agent class definition.\n")
                 running = False
                 continue

            actions = []
            for i in range(N_AGENTS):
                # Always plan move for every agent
                action = select_next_move(
                    i,
                    current_agent_positions[i],
                    current_orientations[i],
                    env.unwrapped.goals[i],  # Use dynamic goal from environment
                    obstacles,
                    current_agent_positions,
                    grid_size
                )
                actions.append(action)

            final_actions = list(actions)

            # --- Environment Step ---
            obs_next, rewards, terminated, truncated, info = env.step([a for a in final_actions])
            #print(f"DEBUG: rewards from env.step: {rewards}")

            # --- Reward Calculation & State Update ---
            current_rewards = [0.0] * N_AGENTS
            for i in range(N_AGENTS):
                current_rewards[i] = rewards[i]
                accumulated_rewards[i] += rewards[i]

            # --- Rendering & Delay ---
            #print(f"DEBUG: accumulated_rewards before render: {accumulated_rewards}")
            env.unwrapped._agent_accumulated_rewards = list(accumulated_rewards)
            env.render()
            time.sleep(RENDER_DELAY)
            step += 1

            # --- Check Termination Conditions ---
            if step >= MAX_STEPS:
                print(f"Max steps ({MAX_STEPS}) reached.")
                running = False
            elif truncated:
                print("Environment truncated the episode.")
                running = False

        # --- End of Simulation ---
        reason = "Max steps reached" if step >= MAX_STEPS else ("ESC pressed" if not running else "Environment truncated")
        save_final_state(env, accumulated_rewards, step, reason)

    except FileNotFoundError as e:
         print(f"Error: {e}. Could not find necessary files.")
    except ImportError as e:
         print(f"Error: Missing dependency - {e}. Please install required packages (e.g., gymnasium, pygame, numpy, Pillow).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        if env:
            env.close() # Ensure environment resources are released
        pygame.quit() # Clean up pygame

if __name__ == "__main__":
    main()