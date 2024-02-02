from __future__ import print_function
import numpy as np
from time import sleep
from tqdm import tqdm
import os
import pickle
import logging
import warnings
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle 

from env.env import Env
from agent import ActorCriticAgent, RandomAgent
from util import read_config, flush_or_create
from utils.buffer import Buffer

warnings.filterwarnings("ignore")

def collect_random_data(agent):
	env = Env()
	random_agent = RandomAgent()
	end = False
	states = []
	actions = []
	rewards = []
	data = []
	discount_G = 1.0
	G = 0.
	t = 0
	while not end:
		states.append(env.state)
		action = random_agent.select_action(env.feasible_actions)
		action_index = 4*action[0] + action[1]
		actions.append(action_index)
		reward, _, end = env.step(action)
		rewards.append(reward)
		# discount = gamma
		# for s in range(t):
		# 	values[t-s-1] += discount * reward
		# 	discount = discount * gamma
		t += 1
		G += discount_G * reward
		discount_G = discount_G * agent.gamma

	R = 0.

	# evaluate state values of all states encountered in a batch to save time
	state_values = agent.net.get_value(np.array(states).reshape(-1,7,7,agent.state_channels)).reshape(-1) 

	for s in range(t):
		R = rewards[t-s-1] + agent.gamma * R
		advantage = R - state_values[t-s-1]
		data = [dict({"state" : states[t-s-1], 
				 	  "advantage" : advantage, 
				 	  "action" : actions[t-s-1],
				 	  "critic_target" : R})] + data 


	assert(G == R)
	assert(len(state_values) == len(states) == len(actions) == len(rewards) == t)

	# data = []
	# for s in range(len(states)-1):
	# 	advantage = rewards[s] + values[s+1] - values[s]
	# 	data.append(dict({"state" : states[s], 
	# 					  "advantage" : advantage,
	# 					  "critic_target" : values[s],
	# 					  "action" : actions[s]}))

	# T = len(states)-1
	# advantage = rewards[T] - values[T] # next state value is 0 because it is terminal
	# data.append(dict({"state" : states[T], 
	# 				  "advantage" : advantage,
	# 				  "critic_target" : values[T],
	# 				  "action" : actions[T]}))

	return data	


def populate_buffer(agent, n_workers, buffer):
	env = Env()
	agents = [agent for _ in range(n_workers)]
	pool = ThreadPool(n_workers)
	while len(buffer.buffer) < buffer.capacity:
		results = pool.map(collect_random_data, agents)
		for data in results:
			shuffle(data)
			buffer.add_list(data)
	pool.close()
	pool.join()



def main():
	config = read_config("config.yaml")
	agent_config = config['Agent']
	network_config = agent_config['Network']
	training_config = config['Training']
	files_config = config['Files']
	eval_config = config['Evaluation']

	print('\t\t --------------------------------------------')
	print('\t\t ------  Parameters of the experiment  ------')
	print('\t\t --------------------------------------------\n')

	print('## Agent params')
	print('Agent : ' + agent_config['name'])
	print('Gamma : ', agent_config['gamma'])
	print('')

	print('## Network Params')
	print('Network used : ' + network_config['name'])
	print('Number of filters : ', network_config['n_filters'])
	print('activation function : ' + network_config['activation'])
	print('state embedding size : ', network_config['state_embedding_size'])
	print('')

	print('## Training params')
	print('Number of iteration : ', training_config['n_iter'])
	print('Learning rate : ', network_config['lr'])
	print('Number of games per iteration : ', training_config['n_games'])
	print('Number of workers : ', training_config['n_workers'])
	print('Batch size : ', training_config['batch_size'])
	print('Buffer size : ', training_config['buffer_size'])
	print('')

	print('## Evaluation params')
	print('Number of games per iteration : ', eval_config['n_games'])
	print('Number of workers : ', eval_config['n_workers'])
	print('')

	sleep(2.0)

	# Init files and tensorboard
	model_name = agent_config['name']
	checkpoints_dir = os.path.join(model_name, files_config['checkpoints_dir'])
	tensorboard_log_dir = os.path.join(model_name, files_config['tensorboard_log_dir'])
	results_log_path = os.path.join(model_name, files_config['results_log_path'])

	# fix random seed
	if config['Seed'] is None:
		np.random.seed(seed=42)
	else:
		np.random.seed(int(seed))

	print('\n\n')
	env = Env()

	# if train from scratch
	if training_config["init_checkpoint"] == 0:
		# initialize dir for tensorboard 
		flush_or_create(tensorboard_log_dir)
	    # initialize dir for checkpoitns
		flush_or_create(checkpoints_dir)
	    # init agent and network from scratch
		agent = ActorCriticAgent(agent_config, network_config, checkpoints_dir, tensorboard_log_dir)
	    # initialize iteration number
		start = 0

	# else restart training from last checkpoint
	else:
	    agent = ActorCriticAgent(agent_config, network_config, checkpoints_dir, tensorboard_log_dir, restore=True)
	    print('\nnetwork restored from checkpoint # ', latest_checkpoint)
	    print('')
	    start = latest_checkpoint

	# intialize the summary writer and results log file
	log_file = open(results_log_path, "wb+") # open log file to write in during evaluation

	display_every = training_config["display_every"]
	n_games_train = training_config["n_games"]
	n_workers_train = training_config["n_workers"]
	T_update_net = training_config["T_update_net"]
	T_update_target_net = training_config["T_update_target_net"]
	n_games_eval = eval_config["n_games"]
	n_workers_eval = eval_config["n_workers"]
	prefill_buffer = training_config["prefill_buffer"]
	# gamma = agent_config['gamma']

	summary_dict = dict({})
	data_buffer = Buffer(capacity=training_config['buffer_size'])

	logger = logging.getLogger(__name__)

	if prefill_buffer:
		# populate buffer with intial data from random games
		print('\nPopulating Buffer ... \n')
		populate_buffer(agent, n_workers_train, data_buffer)

	print('\n\n')
	print('Starting training\n\n')
	batch_size = training_config['batch_size']
	for it in tqdm(np.arange(start, training_config["n_iter"]), desc="parallel gameplay iterations"):
		# play games to generate data and train the network
		env.reset()
		try:
			agent.train(env, n_games_train, data_buffer, batch_size, n_workers_train, display_every, T_update_net)
		except Exception as error:
			print('\n\n#### AN ERROR OCCURED WHILE TRAINING ####\n\n')
			agent.net.summary_writer.close()
			agent.net.sess.close()
			log_file.close()
			logger.error(error)
			raise 
		agent.net.save_checkpoint(checkpoints_dir, it=it+1)

		# play games with latest checkpoint and track average final reward
		results = agent.evaluate(env, n_games_eval, n_workers_eval)
	    # save results
		pickle.dump(results, log_file)
		print('')

	agent.net.summary_writer.close()
	agent.net.sess.close()
	log_file.close()
	print('End of training')

if __name__ == '__main__':
	main()