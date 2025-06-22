# Energy-Aware Multi-Agent Path Finding (MAPF) in RWARE

This repository contains all source code, layouts, scripts, and results for a research project on Multi-Agent Path Finding (MAPF) in a robotic warehouse environment. The project investigates the impact of energy constraints (battery life and charging stations) on agent path-finding and team performance, using both classical algorithms and reinforcement learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder and File Structure](#folder-and-file-structure)
- [Script Groups](#script-groups)
- [Setup and Installation](#setup-and-installation)
- [Running Experiments](#running-experiments)
- [Automated Batch Experiments](#automated-batch-experiments)
- [Results and Reports](#results-and-reports)
- [Credits](#credits)

---

## Project Overview

The project is built on the RWARE (Robotic Warehouse) environment, extended to support:
- Custom layouts and obstacles
- Energy management (battery depletion and charging stations)
- Team and individual reward structures
- Both classical (BFS) and RL-based agent controllers

---

## Folder and File Structure

- **`rware/`**  
  Source code for the custom warehouse environments and renderer.  
  - `rware.warehouse`: 
    -Goals change posisitons
    -individual rewards for the agents
    -Used for mapf scripts 
  - `rware.hrpf_team_warehouse`:
    -Episodic environment, episodes reinitiate when all agents reach their goal
    -agents start at random positions at the start of each episodes
    -individual and team rewards accumulate
    -Used for HRPF scripts and RL training
  - `rware.comb_warehouse`:
    -Episodic environment, episodes reinitiate when all agents reach their goal
    -agents start at random positions at the start of each episodes
    -individual and team rewards accumulate
    -Also charging and charging station logic integrated.
    -Used for Desired_Algorithms and testing with BFS
    
  - `rendering.py`: Unified renderer for all experiments
    -The same as default rware rendering.py
    -Episode, global step, step and reward counters
    -Shelves changed to obstacles with gray color
    -Charging stations as green tiles denoted by "C" added to the environment

- **`Models/`**  
  Contains all Q-value models and outputs from `trainer.py` (RL). Used by `evaluate.py` for HRPF evaluation.

- **`Layouts/`**  
  Layouts for the final BFS/Desired_Algorithm scripts. Each file specifies grid size, agent/obstacle/charger positions, etc.

- **`Layouts_for_mapf/`**  
  Layouts for the earlier `mapf_*` scripts.

- **`Evaluation_reports/`**  
  Step-by-step logs from `evaluate.py` (HRPF RL testing).

- **`Test_Results_Logs/`**  
  Output directory for all structured report files from the BFS experiments, organized by experiment type.

- **`Scripts.md`**  
  Quick reference for all scripts and their dependencies.

---

## Script Groups

### 1. MAPF Scripts (`mapf_*`)
**Purpose:** Early exploration of pathfinding, reward structures, and dynamic goals.

**Examples:**  
- `mapf_human_play_MovingGoals.py` (human play, moving goals)  
- `mapf_individual_rew.py` (automated, moving goals, individual rewards)  
- `mapf_team_rew.py` (automated, stationary goals, team rewards)

### 2. HRPF Reinforcement Learning Scripts
**Purpose:** Training and evaluating agents using Hierarchical RL.

**Key Files:**  
- `trainer.py` (training)  
- `evaluate.py` (evaluation)  
- `agent_networks.py` (Q-network definitions)  
- `human_play_hrpf.py` (manual play/testing)

### 3. Final BFS Algorithm Testing Scripts (`Desired_algorithm_*`)
**Purpose:** Systematic evaluation of BFS-based pathfinding with/without energy constraints.

**Key Files:**  
- `Desired_algorithm_HP.py` (human play, final environment)  
- `Desired_algorithm_BFS_No_Charge.py` (BFS, no battery logic)  
- `Desired_algorithm_BFS_Charge.py` (BFS, with battery/charging logic)

---

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Zaheer-ahmet/Energy_Aware_MAPF_RWARE.git
    cd Energy_Aware_MAPF_RWARE
    ```

2. **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Running Experiments

### A. BFS Experiments (Final Evaluation)

#### No-Charge Version
Run the BFS algorithm without battery constraints:
```bash
python Desired_algorithm_BFS_No_Charge.py --layout Layouts/layout_A2_O10_C0.txt --max_episodes 20 --run_id NoCharge_A2_O10
```

#### Charge Version
Run the BFS algorithm with battery and charging station logic:
```bash
python Desired_algorithm_BFS_Charge.py --layout Layouts/layout_A2_O10_C1.txt --max_episodes 20 --battery_max 15 --run_id Charge_A2_O10_B15
```

**Arguments:**
- `--layout`: Path to the layout file (required)
- `--n_agents`: Number of agents (optional; defaults to number of goals in layout)
- `--max_episodes`: Number of episodes to run
- `--battery_max`: (Charge version only) Max battery per agent
- `--run_id`: Name for the output report file

### B. RL Training and Evaluation

#### Train a model:
```bash
python trainer.py
```
Models will be saved in the `Models/` directory.

#### Evaluate a trained model:
```bash
python evaluate.py
```
Step-by-step reports will be saved in `Evaluation_reports/`.

---

## Automated Batch Experiments

To run a full suite of BFS experiments across different agent counts, obstacle densities, and battery levels, use:
```bash
python run_all_experiments.py
```
This will populate `Test_Results_Logs/` with all result files.

---

## Results and Reports

- **BFS experiment reports:**  
  Located in `Test_Results_Logs/BFS_Charge_Play/` and `Test_Results_Logs/BFS_No_Charge_Play/`, named by `run_id`.
- **RL evaluation reports:**  
  Located in `Evaluation_reports/`.
- **Trained models:**  
  Located in `Models/`.

---

## Credits

- **Project Author:** Ahmad Zahir RAHIMI - Uğur EDNİRLİK
- **Core Environment:** [RWARE: Multi-Agent Robotic Warehouse Environment](https://github.com/semitable/robotic-warehouse)
- **Repository:** [Energy_Aware_MAPF_RWARE](https://github.com/Zaheer-ahmet/Energy_Aware_MAPF_RWARE)

---

**For a quick reference to all scripts and their dependencies, see `Scripts.md`.**