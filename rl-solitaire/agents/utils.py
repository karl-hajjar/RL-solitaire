import numpy as np
from .base_agent import BaseAgent
import importlib

SUPPORTED_AGENTS = {'actor_critic'}


def mask_infeasible_actions(policy: np.array, feasible_action_indices: np.array, normalize=True) -> np.array:
    policy[~feasible_action_indices] = 0.0
    if normalize:
        policy = policy / np.sum(policy)

    return policy


def get_agent_class_from_name(name: str):
    if name not in SUPPORTED_AGENTS:
        raise ValueError(f"Agent {name} not in supported agents: {SUPPORTED_AGENTS}")
    else:
        agent_module = importlib.import_module(name=f"agents.{name}.{name}_agent")
        agent_class_name = get_agent_class_name_from_name(name)
        agent_class = getattr(agent_module, agent_class_name)
        return agent_class


def get_agent_class_name_from_name(name: str) -> str:
    try:
        names = name.split('_')
        class_name = ''.join([name_[0].upper() + name_[1:] for name_ in names]) + "Agent"
    except:
        try:
            names = name.split('_')
            class_name = ''.join([names[0].upper()] + [name_[0].upper() + name_[1:] for name_ in names[1:]]) + "Agent"
        except Exception as e:
            raise ValueError(f"Exception while trying to import name {name} with class name {class_name} :\n{e}")

    return class_name

