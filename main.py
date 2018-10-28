import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm

from env.env import Env
from agent import ActorCriticAgent


def main():
	config = read_config("config/config.yaml")
	network_config = config['Network']
	training_config = config['Training']
	files_config = config['Files']

	print('\t\t --------------------------------------------')
	print('\t\t ------  Parameters of the experiment  ------')
	print('\t\t --------------------------------------------\n')

	print('## Network Params')
	print('Network used : ' + network_config['name'])
	print('Number of filters : ', network_config['n_filters'])
	print('activation function : ' + network_config['activation'])
	print('state embedding size : ', network_config["state_embedding_size"])
	print('')

	print('## Training params')
	print('Number of iteration : ', training_config['n_iter'])
	print('Learning rate : ', training_config["lr"])
	print('Number of workers : ', training_config["n_workers"])
	print('')

	# Init files and tensorboard

	# Init results list

	env = Env()
	agent = ActorCriticAgent(env=env, net_config=network_config)

	for it in tqdm(range(training_config["n_iter"]), desc="parallel gameplay iterations"):
		env.reset()
		tf_logs = agent.train(env, training_config["n_workers"], training_config["display_every"])


if __name__ == '__main__':
	main()