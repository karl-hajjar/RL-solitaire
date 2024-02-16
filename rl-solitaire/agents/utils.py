import numpy as np
import importlib

SUPPORTED_AGENTS = {'actor_critic'}


def mask_infeasible_actions(policy: np.array, feasible_action_indices: np.array, normalize=True) -> np.array:
    policy[~feasible_action_indices] = 0.0
    if normalize:
        policy = policy / np.sum(policy)

    return policy


def get_class_from_name(name: str, class_type="agent"):
    """
    Returns the class from a corresponding agent name. The class is either that of an agent or an agent trainer.
    :param name: string representing the name of the object to load the corresponding class
    :param class_type: string in {'agent', 'trainer'}
    :return: an agent or trainer class
    """
    if class_type not in {'agent', 'trainer'}:
        raise ValueError(f"`class_type` argument must be 'agent' or 'trainer' but was {class_type}")
    upper_class_type = class_type[0].upper() + class_type[1:]
    if name not in SUPPORTED_AGENTS:
        raise ValueError(f"Agent {name} not in supported agents: {SUPPORTED_AGENTS}")
    else:
        module_ = importlib.import_module(name=f"agents.{name}.{name}_{class_type}")
        class_name = get_class_name_from_name(name, upper_class_type)
        class_ = getattr(module_, class_name)
        return class_


def get_class_name_from_name(name: str, class_type) -> str:
    """
    Returns the class name from a corresponding agent name. The class is either that of an agent or an agent trainer.
    :param name: string representing the name of the object to load the corresponding class
    :param class_type: string in {'Agent', 'Trainer'}
    :return: an agent or trainer class name
    """
    if class_type not in {'Agent', 'Trainer'}:
        raise ValueError(f"`class_type` argument must be 'Agent' or 'Trainer' but was {class_type}")
    try:
        names = name.split('_')
        class_name = ''.join([name_[0].upper() + name_[1:] for name_ in names]) + class_type
    except:
        try:
            names = name.split('_')
            class_name = ''.join([names[0].upper()] + [name_[0].upper() + name_[1:] for name_ in names[1:]]) + \
                         class_type
        except Exception as e:
            raise ValueError(f"Exception while trying to import name {name} with class name {class_name} :\n{e}")

    return class_name

