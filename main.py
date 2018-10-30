from __future__ import print_function
import numpy as np
import tensorflow as tf
from time import time
from tqdm import tqdm
import os
import pickle
import logging

from env.env import Env
from agent import ActorCriticAgent
from util import read_config, flush_or_create


def main():
	config = read_config("config.yaml")
	network_config = config['Network']
	training_config = config['Training']
	files_config = config['Files']
	eval_config = config['Evaluation']

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
	print('Learning rate : ', network_config["lr"])
	print('Number of workers : ', training_config["n_workers"])
	print('')

	# Init files and tensorboard
	model_name = network_config['name']
	checkpoints_dir = os.path.join(model_name, files_config["checkpoints_dir"])
	tensorboard_log_dir = os.path.join(model_name, files_config["tensorboard_log_dir"])
	results_log_path = os.path.join(model_name, files_config["results_log_path"])

	env = Env()
	agent = ActorCriticAgent(network_config, checkpoints_dir, tensorboard_log_dir)

	# if train from scratch
	if training_config["init_checkpoint"] == 0:
	    # initialize dir for tensorboard 
	    flush_or_create(tensorboard_log_dir)
	    # initialize iteration number
	    start = 0

	# else restart training from last checkpoint
	else:
	    latest_checkpoint = get_latest_checkpoint(os.path.join(checkpoints_dir, "checkpoint"))
	    net.restore(os.path.join(checkpoints_dir, "checkpoint_{}".format(latest_checkpoint)))
	    print('\nnetwork restored from checkpoint # ', latest_checkpoint)
	    print('')
	    start = latest_checkpoint

	# intialize the summary writer and results log file
	log_file = open(results_log_path, "wb+") # open log file to write in during evaluation

	display_every = training_config["display_every"]
	n_games_train = training_config["n_games"]
	n_workers_train = training_config["n_workers"]
	n_games_eval = eval_config["n_games"]
	n_workers_eval = eval_config["n_workers"]
	summary_dict = dict({})

	logger = logging.getLogger(__name__)

	for it in tqdm(np.arange(start, training_config["n_iter"]), desc="parallel gameplay iterations"):
		# play games to generate data and train the network
		env.reset()
		try:
			agent.train(env, n_games_train, n_workers_train, display_every, it)
		except Exception as error:
			net.summary_writer.close()
			agent.net.sess.close()
			log_file.close()
			logger.error(error)
			raise 
		agent.net.save_checkpoint(checkpoints_dir, it=it)

		# play games with latest checkpoint and track average final reward
		results = agent.evaluate(env, n_games_eval, n_workers_eval)
	    # save results
		pickle.dump(results, log_file)
		print('')

	net.summary_writer.close()
	agent.net.sess.close()
	log_file.close()
	print('End of training')

if __name__ == '__main__':
	main()