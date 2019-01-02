# RL-solitaire
Solving the game of peg solitaire with a Reinforcement Learning (RL) Algorithm. 

I used an adapted version of Asynchronous Advantage Actor Critic (A3C) which I implemented from scratch myself to train an RL agent to solve the game of peg solitaire. The game consists of 32 marbles (or pegs) set out in a cross shape. There are 33 positions in the cross-shaped board, and the start position of the game contains all 32 marbles but one is missing in the center position of the cross. The goal is remove the marbles one by one until there is only one left. To remove a marble, another marble has to move to an empty space and pass over the marble to remove. 


# Files Description

- The file <it>env.py</it> contains the implementation of the soliatire environment as a Python Class <b>Env</b> and the basic functions (init, step, reset, etc) that will be used to interact with it. It also contains a function to visualize the environment.

- The file <it>agent.py</it> contains the implementation of different classes of agents. The basic core class and its methods are described first, then the classes <b>RandomAgent</b> and <b>ActorCriticAgent</b> are implemented using the base methods from the parent class <b>Agent</b>. The actor-critic agent implements A3C and uses and consists of a neural network implemented in the file <it>network.py</it> found in the folder network.

- The folder <it>network</it> contains two python files <it>network.py</it> and <it>build.py</it>. The former contains a Python Class <b>Net</b> implementing a neural network in TensorFlow with a shared representation of the state which then splits into two heads : the policy head and the value head. The latter contains the functions to build the different blocks of the network. 

- The file <it>buffer.py</it> contains a small Python Class implementing a buffer structure. This buffer will be used as a memory replay buffer during the training of the agent. 

- The file <it>util.py</it> contains utility functions to handle files and directories and other such handy functions.

- The file <it>config.yaml</it> contains the configuration parameters (directory names, hyperparameters of the network, number of workers, etc) for training the agent.

- The file <it>main.py</it> contains the main file for training the agent. The config file is read and then the training of the agent can start with the parameters found in the configuration file.


# Description of the Method

I used a slightly adapted version of A3C in which a certain number of games (here 16) are played simultaneously using the same agent (i.e. the same policy network). The data from those games is collected and stored in a memory buffer as a list of dictionnaries where the keys are <b>state</b>, <b>advantage</b>, <b>action selected</b>, and <b>critic target</b>. After every 4 moves played by the agents, data are sampled from the buffer and used to update the policy-value network of the agent using mini-batch gradient descent. One iteration of training consists of playing out until the end the 16 games simultaneously and updating the network every time all of the 16 agents have taken 4 moves. At the end of each iteration, the network weights are saved, and an evualtion phase starts where the results of 30 games (played simultaneously with the latest update of the network) are collecetd and stored in a results file for later analysis. 

The network design has been kept simple, although a more complex architecture would have yielded a faster learning. The input to the network is the state of the environment represented by a 7x7x3 cube, i.e. a 7x7 image with 3 channels (I have used the NHWC covention for TensorFlow tensors). The first channel contains integers 1 and 0 to indicate presence or abscence of a marble at each position. The positions outside the cross-shaped board are automatically filled with zeros. The two other channels contain each a single value broadcasted to the whole channel matrix. The first of those channels contains the percentage of marbles that have been removed so far, and the last contains the percentage of marbles left to remove in order to solve the game. 

The policy-value network first processes that input using three 2d-convolutions. Then this state representation is processed separately by the value head and the policy head. The value head consists of a 1x1 convolution  with stride 1, followed by a dense layer and then the output layer. The policy head consists of a 1x1 convolution with stride 1 followed by a dense layer giving the logits of a softmax distribution. 

At each state of a game, we store the cube representation of the state, the critic target for this state is computed using the rewards cumulated during the 4 moves in which this state was observed as well as the value network for bootstrapping. The value network is used both to evaluate the value of the last state reached after the 4 moves and thus to obtain the critic target values, but also to evaluate each of the 4 states encountered, whose values will serve as baseline when computing the advantage for each of these for states. The action selected by the agent is also stored in order to train the actor (policy network). 
