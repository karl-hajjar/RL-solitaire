import os
from datetime import datetime
from pytorch_lightning import seed_everything
import click
import yaml

from env.env import Env
from utils.tools import strp_datetime, set_up_logger, read_yaml
from nn.utils import get_network_dir_from_name, get_network_class_from_name
from nn.network_config import NetConfig
from agents.utils import get_class_from_name


ROOT = "./"

RUNS_DIRNAME = "runs"
RUN_CONFIG_FILENAME = "run_config.yaml"
LOG_FILENAME = "log.txt"
CHECKPOINTS_DIRNAME = "checkpoints"
RESULTS_FILENAME = "agent_results.pickle"
SEED = 42
DEFAULT_DISCOUNT_FACTOR = 1.0


# @click.command()
# @click.option('-an', '--agent_name', required=True, type=click.STRING, help='The name of the agent to train')
# @click.option('-nn', '--network_name', required=False, type=click.STRING, default=None,
#               help='The name of the agent to train')
def run(agent_name: str, network_name: str = None):
    # file paths and dirs
    agent_dir = os.path.join(ROOT, "agents", agent_name)
    run_dirname = strp_datetime(datetime.now())
    run_dir = os.path.join(agent_dir, RUNS_DIRNAME, run_dirname)
    log_filepath, checkpoints_dir, results_filepath = set_up_files_dirs_and_paths(run_dir)
    logger = set_up_logger(path=log_filepath)

    # trainer config
    trainer_config_filename = f"{agent_name}_trainer_config.yaml"
    trainer_config_filepath = os.path.join(agent_dir, trainer_config_filename)
    trainer_config = read_yaml(trainer_config_filepath)
    with open(os.path.join(run_dir, trainer_config_filename), 'w') as file:
        yaml.safe_dump(trainer_config, file)

    # set seed
    seed = get_seed(trainer_config)
    seed_everything(seed)

    # define network
    if network_name is None:
        network = None
    else:
        network_config_filename = network_name + "_config.yaml"
        network_dir = get_network_dir_from_name(network_name)
        network_config_filepath = os.path.join(ROOT, network_dir, network_config_filename)
        network_config_dict = read_yaml(network_config_filepath)
        with open(os.path.join(run_dir, network_config_filename), 'w') as file:
            yaml.safe_dump(network_config_dict, file)
        network_config = NetConfig(config_dict=network_config_dict)
        network_class = get_network_class_from_name(network_name)
        network = network_class(network_config)

    # define agent
    agent_class = get_class_from_name(agent_name, class_type="agent")
    discount_factor = get_discount_factor(trainer_config)
    if network is None:
        agent = agent_class(discount=discount_factor)
    else:
        full_agent_name = f"{network.name}-{agent_class.__name__}"
        agent = agent_class(network=network, name=full_agent_name, discount=discount_factor)

    # define trainer
    trainer_class = get_class_from_name(agent_name, class_type="trainer")
    trainer = trainer_class(env=Env(), agent=agent, agent_results_filepath=results_filepath, log_dir=run_dir,
                            checkpoints_dir=checkpoints_dir, **trainer_config)

    # log run parameters
    logger.info(f"---------  Running experiment with agent {agent_name} and network {network_name} ---------")
    logger.info(f"Saving run results and logs at {run_dir}")
    logger.info(f"Running with random seed {seed}")
    logger.info(f"Running with discount factor {discount_factor}")

    trainer.train()


def set_up_files_dirs_and_paths(run_dir: str) -> (str, str, str):
    os.makedirs(run_dir, exist_ok=True)

    log_filepath = os.path.join(run_dir, LOG_FILENAME)
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIRNAME)
    results_filepath = os.path.join(run_dir, RESULTS_FILENAME)

    return log_filepath, checkpoints_dir, results_filepath


def get_seed(config_dict: dict) -> int:
    if ("seed" not in config_dict) or (config_dict["seed"] is None):
        return SEED
    else:
        return config_dict.pop("seed")


def get_discount_factor(config_dict: dict) -> float:
    if ("discount" not in config_dict.keys()) or (config_dict["discount"] is None):
        return DEFAULT_DISCOUNT_FACTOR
    else:
        return config_dict.pop("discount")


if __name__ == "__main__":
    # network_name = "conv_policy_value"
    network_name = "fc_policy_value"
    run(agent_name='actor_critic', network_name=network_name)
