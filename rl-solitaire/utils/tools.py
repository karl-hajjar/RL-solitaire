import os
import logging
import numpy as np
import torch
import yaml
import pickle
from datetime import datetime


def set_up_logger(path: str):
    # first remove handlers if there were some already defined
    logger = logging.getLogger()  # root logger
    for handler in logger.handlers:  # remove all old handlers
        handler.close()
        logger.removeHandler(handler)

    # set new handlers
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] -- %(module)s - %(funcName)s  %(message)s")
    file_handler = logging.FileHandler(path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # add new handlers
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_yaml(path: str) -> dict:
    with open(path, 'r') as stream:
        try:
            d = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise Exception("Exception while reading yaml file {} : {}".format(d, e))
    return d


def create_dir(path):
    """
    Creates a directory if it does not exist.
    :param string path: the path at which to create the directory. According to the specifications of os.makedirs, all
    the intermediate directories will be created if needed.
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_pickle(path, single=False):
    results = []
    with open(path, "rb") as f:
        if single:
            results = pickle.load(f)
        else:
            while True:
                try:
                    results.append(pickle.load(f))
                except EOFError:
                    break
    return results


def strp_datetime(date: datetime) -> str:
    """
    Transforms a datetime object like 'Year-Month-Day hour:minute:second:microsecond' into a string
    'Year_Month_Day-hour_minute'.
    :param date: datetime object
    :return: reformatted string date
    """
    split_date_seconds = str(date).split(':')[:-1]
    return '_'.join(split_date_seconds).replace('-', '_').replace(' ', '-')
