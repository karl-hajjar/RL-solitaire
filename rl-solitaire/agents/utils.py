import numpy as np


def mask_infeasible_actions(policy: np.array, feasible_action_indices: np.array, normalize=True) -> np.array:
    policy[~feasible_action_indices] = 0.0
    if normalize:
        policy = policy / np.sum(policy)

    return policy


