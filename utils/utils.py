import numpy as np
import torch
from collections import namedtuple

Step = namedtuple('Step', 'cur_state action next_state next_context reward done mask')

def get_gae(rewards, learner_len, values, gamma=0.99, lamda=0.95, device=None):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards (list | np.ndarray | torch.Tensor): 1D sequence of rewards (concatenated across episodes).
        learner_len (list | np.ndarray | torch.Tensor): Sequence of episode lengths.
        values (list | np.ndarray | torch.Tensor): 1D sequence of value estimates aligned with `rewards`.
        gamma (float): Discount factor.
        lamda (float): GAE lambda (0–1).
        device (torch.device | None): Optional torch device. Defaults to CPU if None.

    Returns:
        (torch.Tensor, torch.Tensor):
            - returns: Same shape as `rewards`, the discounted returns.
            - advants: Same shape as `rewards`, the (z-scored) GAE advantages.
    """
    if not isinstance(rewards, (list, np.ndarray, torch.Tensor)) or \
       not isinstance(learner_len, (list, np.ndarray, torch.Tensor)) or \
       not isinstance(values, (list, np.ndarray, torch.Tensor)):
        return torch.zeros(0), torch.zeros(0)

    if len(rewards) == 0 or len(learner_len) == 0:
        return torch.zeros(0), torch.zeros(0)

    total_len = sum(learner_len)
    if total_len == 0:
        return torch.zeros(0), torch.zeros(0)

    device = device if device is not None else torch.device("cpu")
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    values = torch.as_tensor(values, dtype=torch.float32, device=device)
    learner_len = np.array(learner_len)

    if rewards.shape[0] != values.shape[0] or rewards.shape[0] != total_len:
        return torch.zeros(0), torch.zeros(0)

    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    masks = torch.ones_like(rewards)

    start_idx = 0
    for length in learner_len:
        if length <= 0:
            continue
        end_idx = start_idx + length
        if end_idx > 0 and (end_idx - 1) < masks.shape[0]:
            masks[end_idx - 1] = 0.0
        start_idx = end_idx

    running_returns = torch.tensor(0.0, device=device)
    previous_value = torch.tensor(0.0, device=device)
    running_advants = torch.tensor(0.0, device=device)

    for t in reversed(range(len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        td_error = rewards[t] + gamma * previous_value * masks[t] - values[t]
        running_advants = td_error + gamma * lamda * running_advants * masks[t]
        returns[t] = running_returns
        advants[t] = running_advants
        previous_value = values[t]

    adv_mean = advants.mean()
    adv_std = advants.std() + 1e-8
    advants = (advants - adv_mean) / adv_std

    return returns, advants
