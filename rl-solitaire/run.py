import os
from datetime import datetime
from pytorch_lightning import seed_everything

from env.env import Env
from utils.tools import strp_datetime


ROOT = "./"

RUNS_DIRNAME = "runs"
RUN_CONFIG_FILENAME = "run_config.yaml"
LOG_FILENAME = "log.txt"
CHECKPOINTS_DIRNAME = "checkpoints"
RESULTS_FILENAME = "agent_results.pickle"
SEED = 42


def run(agent_name: str, network_name: str):
    agent_dir = os.path.join(ROOT, "agents", agent_name)
    run_dirname = strp_datetime(datetime.now())
    run_dir = os.path.join(agent_dir, RUNS_DIRNAME, run_dirname)
    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, LOG_FILENAME)
    checkpoints_dir = os.path.join(run_dir, CHECKPOINTS_DIRNAME)
    results_file = os.path.join(run_dir, RESULTS_FILENAME)


    seed_everything(SEED)
