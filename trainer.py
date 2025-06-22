import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agent_networks import QNetwork, combine_q_values, create_agent_networks
from rware.hrpf_team_warehouse import HRPFTeamWarehouse, RewardType, Action
import os
from torchrl.data import PrioritizedReplayBuffer, ListStorage
import csv
from datetime import datetime

# ==== Parameters ====
render_mode = "none"
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
    render_mode=render_mode
)
print("observation_space:", env.observation_space)
n_agents = env.n_agents
# For Tuple(Box, Box, ...), use the shape of the first Box
obs_shape = env.observation_space[0].shape
obs_dim = int(np.prod(obs_shape))
n_actions = len(Action)

# ==== Hyperparameters ====
buffer_capacity = 100_000
batch_size = 256  # Increased batch size
gamma = 0.99
epsilon = 1.0
epsilon_end = 0.05  # Lower end epsilon
epsilon_decay = 0.999
beta = 1.0
num_episodes = 10000
max_steps = 400
learning_rate = 3e-4  # Reduced learning rate

# ==== Running Statistics for Observation Normalization ====
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)  # Initialize var attribute
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

obs_normalizer = RunningMeanStd(obs_dim)

# ==== Agent Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_ind = []
q_te = []
target_ind = []
target_te = []
opt_ind = []
opt_te = []

for _ in range(n_agents):
    q_i, q_t = create_agent_networks(obs_dim, n_actions)
    q_i = q_i.to(device)
    q_t = q_t.to(device)
    q_ind.append(q_i)
    q_te.append(q_t)
    target_ind.append(create_agent_networks(obs_dim, n_actions)[0].to(device))
    target_te.append(create_agent_networks(obs_dim, n_actions)[1].to(device))
    # Use lower learning rate
    opt_ind.append(optim.Adam(q_i.parameters(), lr=learning_rate))
    opt_te.append(optim.Adam(q_t.parameters(), lr=learning_rate))

buffer = PrioritizedReplayBuffer(alpha=0.8, beta=0.4, storage=ListStorage(buffer_capacity))

# ==== Training ====
os.makedirs("models", exist_ok=True)
# ==== Training Data Logging ====
training_data_dir = "Training_Data"
os.makedirs(training_data_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
training_data_path = os.path.join(training_data_dir, f"training_data_{timestamp}.csv")
with open(training_data_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Episode No.", "Avg Reward", "Team Reward", "Agent 0 reached goal", "Agent 1 reached goal", "Goals reached", "Agent 0 X", "Agent 0 Y", "Agent 1 X", "Agent 1 Y", "Agent 0 Goal X", "Agent 0 Goal Y", "Agent 1 Goal X", "Agent 1 Goal Y"])
    for ep in range(num_episodes):
        # Anneal beta linearly from 0.4 to 1.0
        buffer._beta = min(1.0, 0.4 + 0.6 * (ep / num_episodes))
        obs, info = env.reset()
        # Save initial positions for each agent
        initial_positions = [(agent.x, agent.y) for agent in env.agents]
        # Save goal positions for each agent
        goal_positions = [tuple(env.goals[i]) for i in range(n_agents)]
        ep_reward = np.zeros(n_agents)
        goal_reach_count = 0

        # --- HER: Store episode trajectories ---
        episode_transitions = [[] for _ in range(n_agents)]

        # Update observation statistics
        obs_flat = [o.flatten() for o in obs]
        obs_normalizer.update(np.stack(obs_flat))

        for t in range(max_steps):
            # Normalize observations
            obs_flat = [obs_normalizer.normalize(o.flatten()) for o in obs]
            obs_tensor = [torch.FloatTensor(o).unsqueeze(0).to(device) for o in obs_flat]

            # Epsilon-greedy action selection
            actions = []
            for i in range(n_agents):
                with torch.no_grad():
                    q_i = q_ind[i](obs_tensor[i])
                    q_t = q_te[i](obs_tensor[i])
                    q_c = combine_q_values(q_i, q_t, beta)
                    a = torch.randint(0, n_actions, (1,)).item() if random.random() < epsilon else torch.argmax(q_c).item()
                    actions.append(a)

            next_obs, rewards, done, info = env.step(actions)
            team_reward = info.get("team_reward", 0.0)
            done_flag = float(done)

            # Count goal reaches
            for r in rewards:
                if r == 10.0:
                    goal_reach_count += 1

            # Store transitions in buffer and for HER
            for i in range(n_agents):
                obs_tensor = torch.from_numpy(obs[i].flatten()).float()
                next_obs_tensor = torch.from_numpy(next_obs[i].flatten()).float()
                action_tensor = torch.tensor(actions[i])
                reward_tensor = torch.tensor(rewards[i]).float()
                team_reward_tensor = torch.tensor(team_reward).float()
                done_tensor = torch.tensor(done_flag).float()
                # Debug print
                if obs_tensor.numel() != obs_dim or next_obs_tensor.numel() != obs_dim:
                    print(f"[WARN] Skipping malformed transition: obs shape {obs_tensor.shape}, next_obs shape {next_obs_tensor.shape}")
                    continue
                buffer.add((
                    obs_tensor,
                    action_tensor,
                    reward_tensor,
                    team_reward_tensor,
                    next_obs_tensor,
                    done_tensor
                ))
                # --- HER: Store for relabeling ---
                episode_transitions[i].append((obs_tensor, action_tensor, reward_tensor, team_reward_tensor, next_obs_tensor, done_tensor))

            obs = next_obs
            ep_reward += np.array(rewards)

            if done:
                break

        # --- HER: After episode, relabel with final achieved position as goal ---
        for i in range(n_agents):
            # Use the final position as the HER goal
            final_obs = episode_transitions[i][-1][0] if episode_transitions[i] else None
            if final_obs is None:
                continue
            # Extract x, y from obs vector (assume first two entries are x, y)
            her_goal_x = final_obs[0].item()
            her_goal_y = final_obs[1].item()
            for (obs_tensor, action_tensor, _, _, next_obs_tensor, done_tensor) in episode_transitions[i]:
                # Relabel reward: +10 if next_obs is at HER goal, else shaped as before
                curr_x, curr_y = next_obs_tensor[0].item(), next_obs_tensor[1].item()
                prev_x, prev_y = obs_tensor[0].item(), obs_tensor[1].item()
                prev_dist = abs(prev_x - her_goal_x) + abs(prev_y - her_goal_y)
                curr_dist = abs(curr_x - her_goal_x) + abs(curr_y - her_goal_y)
                if (curr_x, curr_y) == (her_goal_x, her_goal_y):
                    her_reward = torch.tensor(10.0)
                elif done_tensor.item() == 1.0:
                    her_reward = torch.tensor(0.0)
                elif curr_dist < prev_dist:
                    her_reward = torch.tensor(-0.1 + 0.2)
                else:
                    her_reward = torch.tensor(-0.1)
                # Add HER transition to buffer for each agent
                buffer.add((
                    obs_tensor,
                    action_tensor,
                    her_reward,
                    team_reward_tensor,  # team reward unchanged
                    next_obs_tensor,
                    done_tensor
                ))

        # ==== Training ====
        if len(buffer) >= batch_size:
            # --- Balanced Sampling: sample half-batch for each agent and concatenate ---
            half_batch = batch_size // n_agents
            agent_batches = []
            agent_indices = []
            for agent_id in range(n_agents):
                # Filter buffer for transitions where agent index matches
                # (Assume buffer stores agent index as part of transition, else sample randomly)
                agent_batch = buffer.sample(half_batch, return_info=True)
                agent_batches.append(agent_batch[0])
                agent_indices.append(agent_batch[1]['index'])
            # Concatenate batches
            o_t = torch.cat([b[0] for b in agent_batches], dim=0).to(device)
            a_t = torch.cat([b[1] for b in agent_batches], dim=0).long().to(device).unsqueeze(1)
            r_in = torch.cat([b[2] for b in agent_batches], dim=0).float().to(device).unsqueeze(1)
            r_te = torch.cat([b[3] for b in agent_batches], dim=0).float().to(device).unsqueeze(1)
            o_t1 = torch.cat([b[4] for b in agent_batches], dim=0).to(device)
            d = torch.cat([b[5] for b in agent_batches], dim=0).float().to(device).unsqueeze(1)
            indices = sum([idx.tolist() for idx in agent_indices], [])
            # Individual loss for each agent
            for i in range(n_agents):
                # Use only the batch for this agent
                start = i * half_batch
                end = (i + 1) * half_batch
                q_pred = q_ind[i](o_t[start:end])
                q_pred_a = q_pred.gather(1, a_t[start:end])
                q_next = target_ind[i](o_t1[start:end]).max(1, keepdim=True)[0].to(device)
                td_target = r_in[start:end] + gamma * q_next * (1 - d[start:end])
                loss_ind = nn.MSELoss()(q_pred_a, td_target.detach())
                opt_ind[i].zero_grad()
                loss_ind.backward()
                opt_ind[i].step()
                # --- PER: Update priorities ---
                td_error = (q_pred_a.detach() - td_target.detach()).abs().cpu().numpy().flatten()
                buffer.update_priority(indices[start:end], td_error)
            # Team loss as before (use all samples)
            pred = torch.zeros(batch_size, 1).to(device)
            target = torch.zeros(batch_size, 1).to(device)
            for i in range(n_agents):
                # Q_te at (o_t, a_t)
                pred += q_te[i](o_t).gather(1, a_t)
                # Q_te at (o_{t+1}, a'_i) where a'_i = argmax_a Q_te[i](o_{t+1}, a)
                q_next_all = target_te[i](o_t1)  # shape: (batch_size, n_actions)
                max_q_next, _ = q_next_all.max(1, keepdim=True)  # shape: (batch_size, 1)
                target += max_q_next
            td_target_te = r_te + gamma * target * (1 - d)
            loss_te = nn.MSELoss()(pred, td_target_te.detach())
            for i in range(n_agents):
                opt_te[i].zero_grad()
            loss_te.backward()
            for i in range(n_agents):
                opt_te[i].step()

        # ==== Target Network Sync ====
        if ep % 20 == 0:
            for i in range(n_agents):
                target_ind[i].load_state_dict(q_ind[i].state_dict())
                target_te[i].load_state_dict(q_te[i].state_dict())

        # ==== Save ====
        if ep % 100 == 0:
            for i in range(n_agents):
                torch.save(q_ind[i].state_dict(), f"models/q_ind_agent_{i}.pt")
                torch.save(q_te[i].state_dict(), f"models/q_te_agent_{i}.pt")
            print(f"[Episode {ep}] Saved models. Team reward: {info.get('team_reward', 0.0)}")

        # ==== Logging ====
        if ep % 10 == 0:
            avg_reward = ep_reward.mean()
            agent0_reached = 1 if 0 in info.get('completed_agents', []) else 0
            agent1_reached = 1 if 1 in info.get('completed_agents', []) else 0
            writer.writerow([
                ep,
                f"{avg_reward:.2f}",
                f"{info.get('team_reward', 0.0):.2f}",
                agent0_reached,
                agent1_reached,
                goal_reach_count,
                initial_positions[0][0], initial_positions[0][1],
                initial_positions[1][0], initial_positions[1][1],
                goal_positions[0][0], goal_positions[0][1],
                goal_positions[1][0], goal_positions[1][1]
            ])
            csvfile.flush()
            print(f"[Ep {ep}] Avg Reward: {avg_reward:.2f} | Team Reward: {info.get('team_reward', 0.0):.2f} | Goals reached: {goal_reach_count}")

        # ==== Epsilon Decay ====
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print("Training complete.")
