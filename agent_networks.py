import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Feedforward Q-network used for both Q_in and Q_te.
    Accepts a flattened observation vector and outputs Q-values for each discrete action.
    """
    def __init__(self, obs_dim, n_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, n_actions)

    def forward(self, obs):
        """
        obs: Tensor of shape (batch_size, obs_dim)
        returns: Q-values of shape (batch_size, n_actions)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.q_out(x)


def combine_q_values(q_ind, q_te, beta=1.0):
    """
    Combine Q_ind and Q_te using HRPF Equation (5):
    Q_comb_i(o_{t,i}, a_{t,i}) = (1 / (1 + β)) * Q_in_i + (β / (1 + β)) * Q_te_i

    q_ind, q_te: Tensors of shape (batch_size, n_actions)
    beta: scalar weighting coefficient (float)

    returns: Tensor of shape (batch_size, n_actions)
    """
    return (1.0 / (1.0 + beta)) * q_ind + (beta / (1.0 + beta)) * q_te


def create_agent_networks(obs_dim, n_actions):
    """
    Utility function to create a pair of Q-networks for an agent:
    Q_in (individual) and Q_te (team)
    """
    q_ind = QNetwork(obs_dim, n_actions)
    q_te = QNetwork(obs_dim, n_actions)
    return q_ind, q_te
