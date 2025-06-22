import torch
import numpy as np
import time
import os
from datetime import datetime
from agent_networks import QNetwork, combine_q_values
from rware.hrpf_team_warehouse import HRPFTeamWarehouse, RewardType, Action

# ==== Configuration ====
model_dir = "models"
n_agents = 2
max_steps = 400
beta = 1.0
epsilon = 0.2  # Epsilon for evaluation (increased for more exploration)
num_eval_episodes = 1  # Number of evaluation episodes

# ==== Custom Initial Positions for Evaluation (Optional) ====
# To use, set custom_init = True and fill in the coordinates below.
custom_init = False  # Set to True to use custom positions
custom_agent_positions = [(0, 7), (4, 8)]  # Example: [(x0, y0), (x1, y1)]
custom_agent_dirs = [0, 0]  # Example: [Direction.UP.value, Direction.UP.value]
custom_goal_positions = [(4, 9), (5, 9)]  # Example: [(gx0, gy0), (gx1, gy1)]

# ==== Running Statistics for Observation Normalization ====
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.std = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        self.var = M2 / (self.count + batch_count)
        self.std = np.sqrt(self.var + 1e-8)
        self.count += batch_count

    def normalize(self, x):
        return (x - self.mean) / self.std

# ==== Device Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Environment Setup ====
env = HRPFTeamWarehouse(
    shelf_columns=3,
    column_height=3,
    shelf_rows=2,
    n_agents=2,
    msg_bits=0,
    sensor_range=1,
    request_queue_size=1,
    max_inactivity_steps=None,
    max_steps=400,
    reward_type=RewardType.INDIVIDUAL,
    render_mode="human"
)

obs_shape = env.observation_space[0].shape
obs_dim = int(np.prod(obs_shape))
n_actions = len(Action)

obs_normalizer = RunningMeanStd(obs_dim)

# ==== Load Models ====
q_ind_models = []
q_te_models = []
for i in range(n_agents):
    q_ind = QNetwork(obs_dim, n_actions).to(device)
    q_te = QNetwork(obs_dim, n_actions).to(device)

    # Debug: Verify model loading
    print(f"Loading model for agent {i} from {model_dir}/q_ind_agent_{i}.pt")
    q_ind.load_state_dict(torch.load(f"{model_dir}/q_ind_agent_{i}.pt", map_location=device))
    print(f"Loading model for agent {i} from {model_dir}/q_te_agent_{i}.pt")
    q_te.load_state_dict(torch.load(f"{model_dir}/q_te_agent_{i}.pt", map_location=device))

    q_ind.eval()
    q_te.eval()

    q_ind_models.append(q_ind)
    q_te_models.append(q_te)

# ==== Evaluation Loop ====
report_dir = "Evaluation_reports"
os.makedirs(report_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join(report_dir, f"eval_report_{timestamp}.txt")
report_lines = []

for episode in range(num_eval_episodes):
    obs, info = env.reset()
    # If custom_init is enabled, override agent and goal positions
    if custom_init:
        for i, (x, y) in enumerate(custom_agent_positions):
            env.agents[i].x = x
            env.agents[i].y = y
        env.goals = list(custom_goal_positions)
        env._initial_goals = list(custom_goal_positions)
        # Recalculate grid for rendering
        env._recalc_grid()
    initial_positions = [(agent.x, agent.y) for agent in env.agents]
    done = False
    step = 0
    total_rewards = np.zeros(n_agents, dtype=np.float32)
    goals_reached = [False for _ in range(n_agents)]
    # Track per-agent accumulated rewards for rendering
    accumulated_rewards = np.zeros(n_agents, dtype=np.float32)
    while not done and step < max_steps:
        # Normalize and convert observations to tensors
        obs_flat = [obs_normalizer.normalize(o.flatten()) for o in obs]
        obs_tensor = [torch.FloatTensor(o).unsqueeze(0).to(device) for o in obs_flat]
        actions = []
        for i in range(n_agents):
            if goals_reached[i]:
                action = 0
            else:
                with torch.no_grad():
                    q_i = q_ind_models[i](obs_tensor[i])
                    q_t = q_te_models[i](obs_tensor[i])
                    q_c = combine_q_values(q_i, q_t, beta)
                    report_lines.append(f"\nAgent {i} Q-values:")
                    report_lines.append(f"Individual Q: {q_i.squeeze().cpu().numpy()}")
                    report_lines.append(f"Team Q: {q_t.squeeze().cpu().numpy()}")
                    report_lines.append(f"Combined Q: {q_c.squeeze().cpu().numpy()}")
                    if np.random.rand() < epsilon:
                        action = np.random.choice(n_actions)
                    else:
                        action = int(np.argmax(q_c))
                    report_lines.append(f"Selected action: {Action(action).name}")
            actions.append(action)
        obs, rewards, done, info = env.step(actions)
        obs_flat = [o.flatten() for o in obs]
        obs_normalizer.update(np.stack(obs_flat))
        total_rewards += np.array(rewards, dtype=np.float32)
        accumulated_rewards += np.array(rewards, dtype=np.float32)
        report_lines.append(f"\nStep {step}: Actions={[Action(a).name for a in actions]}, Rewards={rewards}")
        # Set per-agent accumulated rewards for rendering
        env._agent_accumulated_rewards = accumulated_rewards.tolist()
        isopen = env.render()
        time.sleep(0.2)
        if isopen is False:
            report_lines.append("[INFO] Window closed or ESC pressed. Terminating evaluation loop.")
            break
        for i in range(n_agents):
            if rewards[i] == 10.0:
                goals_reached[i] = True
        step += 1
    if all(goals_reached):
        report_lines.append(f"[LOG] Both agents reached goals. Initial positions: {initial_positions}")

# ==== Results ====
report_lines.append("Evaluation complete.")
for i in range(n_agents):
    report_lines.append(f"Agent {i} total reward: {float(total_rewards[i]):.2f}")
report_lines.append(f"Team reward: {info.get('team_reward', 0.0):.2f}")

# Save report to file
with open(report_path, "w") as f:
    for line in report_lines:
        f.write(str(line) + "\n")

# Print summary to terminal
print(f"Evaluation complete. Report saved to {report_path}")
for i in range(n_agents):
    print(f"Agent {i} total reward: {float(total_rewards[i]):.2f}")
print(f"Team reward: {info.get('team_reward', 0.0):.2f}")
