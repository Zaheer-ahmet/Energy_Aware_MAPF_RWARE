import subprocess
import os
import itertools
from datetime import datetime

# === CONSTANTS ===
EPISODES = 20
BATTERY_LEVELS = [15, 25, 50]
AGENT_COUNTS = [2, 4, 6]
OBSTACLE_PCTS = [10, 30, 50]
CHARGE_SCRIPT = "Desired_algorithm_BFS_Charge.py"
NOCHARGE_SCRIPT = "Desired_algorithm_BFS_No_Charge.py"

# === UTILITY FUNCTIONS ===

def run_experiment(command, run_id, output_dir):
    # This function now only launches the experiment.
    # The simulation script itself handles all file and directory creation.
    # stdout and stderr are sent to DEVNULL to keep the main console clean.
    print(f"Running: {run_id}")
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === CHARGE RUNS ===
for agents, obstacles, battery in itertools.product(AGENT_COUNTS, OBSTACLE_PCTS, BATTERY_LEVELS):
    chargers = agents // 2
    layout = f"Layouts/layout_A{agents}_O{obstacles}_C{chargers}.txt"
    run_id = f"Charge_A{agents}_O{obstacles}_C{chargers}_B{battery}"
    cmd = f"python {CHARGE_SCRIPT} --layout {layout} --max_episodes {EPISODES} --battery_max {battery} --run_id {run_id}"
    run_experiment(cmd, run_id, "Test_Results_Logs/BFS_Charge_Play")

# === NO-CHARGE RUNS ===
for agents, obstacles in itertools.product(AGENT_COUNTS, OBSTACLE_PCTS):
    layout = f"Layouts/layout_A{agents}_O{obstacles}_C0.txt"
    run_id = f"NoCharge_A{agents}_O{obstacles}_C0"
    cmd = f"python {NOCHARGE_SCRIPT} --layout {layout} --max_episodes {EPISODES} --run_id {run_id}"
    run_experiment(cmd, run_id, "Test_Results_Logs/BFS_No_Charge_Play")
