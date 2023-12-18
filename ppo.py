import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from myTeam import Agent1
import game, capture, json, pickle
from collections import OrderedDict

def createTeam(firstIndex, secondIndex, isRed,
               first = 'PPOAgent', second = 'Agent1', numTraining=0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
#   m1,a1 = mlp(13,[64, 64],5), mlp(13,[64, 64],1)
#   m2,a2 = mlp(13,[64, 64],5), mlp(13,[64, 64],1)
  m1, a1 = default_net(20,5), default_net(20,1)
  m2, a2 = default_net(20,5), default_net(20,1)
  return PPOCNN(firstIndex, actor=m1, critic=a1, obs_dim=13, act_dim=5),\
        PPOCNN(secondIndex, actor=m2, critic=a2, obs_dim=13, act_dim=5)

directions = list(game.Actions._directions.keys())

def load_parameters(nnet, parameters):
    nnet.load_state_dict(torchify(parameters))

def jsonify(state_dict):
    if isinstance(state_dict,dict):
        return {k:jsonify(v) for k,v in state_dict.items()}
    elif isinstance(state_dict,torch.Tensor):
        return state_dict.detach().numpy().tolist()
    
def torchify(state_dict):
    if isinstance(state_dict,dict):
        return {k:torchify(v) for k,v in state_dict.items()}
    elif isinstance(state_dict,list):
        return torch.Tensor(state_dict)

def print_parameters(nnet):
    state_dict = nnet.state_dict()
    print(json.dumps(jsonify(state_dict)))

def dump_nnet(nnet, file):
    state_dict = nnet.state_dict()
    with open(file, 'wb') as f:
        pickle.dump(jsonify(state_dict), f)

def load_nnet(nnet, file):
    with open(file, 'r') as f:
        nnet.load_state_dict(pickle.load(f))

def load_parameters(file):
    with open(file, 'rb') as f:
        return pickle.load(f)



class mlp(nn.Module):
    def __init__(self, fan_in, layers, fan_out):
        super().__init__()
        layers = [fan_in]+layers+[fan_out]
        self.layers = nn.ModuleList([nn.Linear(layers[i-1],layers[i]) for i in range(1,len(layers))])

    def forward(self, x):
        x = x if isinstance(x,torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)
    
class default_net(nn.Module):
    def __init__(self,aux_inputs, outputs):
        super().__init__()
        self.c1 = nn.Conv2d(7,16,5,stride=3)
        self.c2 = nn.Conv2d(16,32,3,stride=2)
        self.c3 = nn.Conv2d(32,16,2,stride=1)
        self.l4 = nn.Linear(320, 128)
        self.l5 = nn.Linear(128,outputs)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(aux_inputs, 160)

    def forward(self, input):
        x,y = input
        x = x if isinstance(x,torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        y = y if isinstance(y,torch.Tensor) else torch.tensor(y, dtype=torch.float32)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = self.flatten(x)

        y = F.relu(self.l1(y))

        z = torch.cat((x,y),dim=1)
        z = F.relu(self.l4(z))
        return self.l5(z)

class numpy_mlp:
    def __init__(self, state_dict, layers):
        self.state_dict=self.numpify(state_dict) 
        self.layers = layers 

    def numpify(self, d):
        if isinstance(d, dict):
            return {k:self.numpify(v) for k,v in d.items()}
        else: 
            return np.array(d)

    def forward(self, x):
        for l in range(self.layers-1):
            weight = self.state_dict[f"layers.{l}.weight"]
            bias = self.state_dict[f"layers.{l}.bias"]
            x = np.maximum(0,np.matmul(x,np.transpose(weight))+bias)

        weight = self.state_dict[f"layers.{self.layers-1}.weight"]
        bias = self.state_dict[f"layers.{self.layers-1}.bias"]
        x = np.matmul(x,np.transpose(weight))+bias
        return x


# m = mlp(2,[5], 2)

# dump_nnet(m, "temp.out")

# m_ = numpy_mlp(load_parameters("temp.out"),2)
    
# print(m(torch.tensor([1.,1])))
# print(m_.forward(np.array([1.,1])))


class PPOAgent(Agent1):
    def __init__(self, 
                 index, 
                 timeForComputing = .1, 
                 actor=lambda x:[1.,1,1,1,1], 
                 critic=lambda x:1, 
                 obs_dim=10,
                 act_dim=5):
        super().__init__(index, timeForComputing = timeForComputing)
        self._init_hyperparameters()
        self.obs_dim=obs_dim 
        self.act_dim=act_dim 
        self.actor=actor 
        self.critic = critic 
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.reset()

        self.batch_obs = []
        self.batch_acts = []
        self.batch_log_probs = []
        self.batch_rtgs = []
        self.batch_lens = []

    def _init_hyperparameters(self):
        """
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
        self.timesteps_per_batch = 10                 # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.01                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
        self.render = True                              # If we should render during rollout
        self.render_every_i = 10                        # Only render every n iterations
        self.save_freq = 10                             # How often we save in number of iterations


    def construct_input(self, gameState:capture.GameState) -> np.array:
        features = self.getFeatures(gameState)
        return np.array(list(features.values()))
    
    def get_policy(self, gameState: capture.GameState):
        input = self.construct_input(gameState)
        output = self.actor(input)
        return output.detach().numpy()
           
    def get_action(self, gameState: capture.GameState, policy=None):
        # o1,o2 = self.construct_input(gameState)
        # print(f"o1 {o1.shape} o2 {o2.shape}")
        actions = gameState.getLegalActions(self.index)
        if policy is None:
            policy = self.get_policy(gameState)
        policy = policy.reshape([-1])
        policy_ = policy
        policy = np.array([policy[directions.index(a)] for a in actions])
        policy = np.e**policy
        policy = policy/np.sum(policy)
        # print(f"POLICY: {policy}")
        rng = np.random.default_rng()
        action = rng.choice(len(actions),p=policy)
        log_prob = policy_[directions.index(actions[action])]
        # print(f"CHOSEN ACTIONS: {actions[action]}")
        return actions[action], log_prob, action
    
    def chooseAction(self, gameState: capture.GameState):
        action, log_prob, idx = self.get_action(gameState)
        return action
    
    def reset(self):
        self.logger = {
            "observations":[],
            "log probs":[], 
            "actions":[], 't_so_far': 0,          # timesteps so far
            'i_so_far': 0,          # iterations so far
            'batch_lens': [],       # episodic lengths in batch
            'batch_rews': [],       # episodic returns in batch
            'actor_losses': [],     # losses of actor network in current iteration
        }

    def compute_rtgs(self, ep_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        ep_rtgs = []

        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * self.gamma
            ep_rtgs.insert(0, discounted_reward)

        print(ep_rtgs)
        return ep_rtgs

    

    def learn(self):
        print("LEARN")
        """
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		""" 

        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far                                                                  # ALG STEP 2
        # Autobots, roll out (just kidding, we're collecting our batch simulations here)
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  
        # print(f"OBS {batch_obs[0].shape} {batch_obs[1].shape} ACTS {batch_acts.shape} LOG {batch_log_probs.shape} RTG {batch_rtgs.shape} LENS {batch_lens}")                   # ALG STEP 3
        # Calculate how many timesteps we collected this batch
        t_so_far += np.sum(batch_lens)

        # Increment the number of iterations
        i_so_far += 1

        # Logging timesteps so far and iterations so far
        self.logger['t_so_far'] = t_so_far
        self.logger['i_so_far'] = i_so_far

        # Calculate advantage at k-th iteration
        V, _ = self.evaluate_obs(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

        # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases the variance of 
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it.
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # This is the loop where we update our network for some n epochs
        for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate_obs(batch_obs, batch_acts)

            # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            # NOTE: we just subtract the logs, which is the same as
            # dividing the values and then canceling the log with e^log.
            # For why we use log probabilities instead of actual probabilities,
            # here's a great explanation: 
            # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
            # TL;DR makes gradient ascent easier behind the scenes.
            ratios = torch.exp(curr_log_probs - batch_log_probs)

            # Calculate surrogate losses.
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            # Calculate actor and critic losses.
            # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
            # the performance function, but Adam minimizes the loss. So minimizing the negative
            # performance function maximizes it.
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)

            # Calculate gradients and perform backward propagation for actor network
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Log actor loss
            self.logger['actor_losses'].append(actor_loss.detach())

        # Print a summary of our training so far
        # self._log_summary()

        # # Save our model if it's time
        # if i_so_far % self.save_freq == 0:
        #     torch.save(self.actor.state_dict(), './ppo_actor.pth')
        #     torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def evaluate_obs(self, batch_obs, batch_acts):
        """
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
        actions = self.actor(batch_obs)
        # dist = MultivariateNormal(mean, self.cov_mat)
        # log_probs = dist.log_prob(batch_acts)
        log_probs = (actions*batch_acts).sum(axis=-1)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
        return V, log_probs

    # def _log_summary(self):
    #     """
	# 		Print to stdout what we've logged so far in the most recent batch.

	# 		Parameters:
	# 			None

	# 		Return:
	# 			None
	# 	"""
	# 	# Calculate logging values. I use a few python shortcuts to calculate each value
	# 	# without explaining since it's not too important to PPO; feel free to look it over,
	# 	# and if you have any questions you can email me (look at bottom of README)
    #     delta_t = self.logger['delta_t']
    #     self.logger['delta_t'] = time.time_ns()
    #     delta_t = (self.logger['delta_t'] - delta_t) / 1e9
    #     delta_t = str(round(delta_t, 2))

    #     t_so_far = self.logger['t_so_far']
    #     i_so_far = self.logger['i_so_far']
    #     avg_ep_lens = np.mean(self.logger['batch_lens'])
    #     avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
    #     avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

	# 	# Round decimal places for more aesthetic logging messages
    #     avg_ep_lens = str(round(avg_ep_lens, 2))
    #     avg_ep_rews = str(round(avg_ep_rews, 2))
    #     avg_actor_loss = str(round(avg_actor_loss, 5))

	# 	# Print logging statements
    #     print(flush=True)
    #     print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
    #     print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    #     print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    #     print(f"Average Loss: {avg_actor_loss}", flush=True)
    #     print(f"Timesteps So Far: {t_so_far}", flush=True)
    #     print(f"Iteration took: {delta_t} secs", flush=True)
    #     print(f"------------------------------------------------------", flush=True)
    #     print(flush=True)

	# 	# Reset batch-specific logging data
    #     self.logger['batch_lens'] = []
    #     self.logger['batch_rews'] = []
    #     self.logger['actor_losses'] = []

    def store_episode(self, ep_obs, ep_acts, ep_log_probs, ep_rews, ep_lens):
        self.batch_obs.extend(ep_obs)
        self.batch_acts.extend(ep_acts)
        self.batch_log_probs.extend(ep_log_probs)
        self.batch_rtgs.extend(self.compute_rtgs(ep_rews))
        self.batch_lens.append(ep_lens[0])
        if sum(self.batch_lens)>self.timesteps_per_batch:
            self.learn()
            self.reset_batch()


    def rollout(self):
        return torch.tensor(np.array(self.batch_obs),dtype=torch.float32), \
                F.one_hot(torch.tensor(np.array(self.batch_acts),dtype=torch.long),num_classes=self.act_dim), \
                torch.tensor(np.array(self.batch_log_probs),dtype=torch.float32), \
                torch.tensor(np.array(self.batch_rtgs),dtype=torch.float32), \
                self.batch_lens 
    
    def reset_batch(self):
        self.batch_obs = [] 
        self.batch_acts = []
        self.batch_rtgs = []
        self.batch_log_probs = []
        self.batch_lens=[]


class PPOCNN(PPOAgent):
    def rollout(self):
        batch0 = [obs[0] for obs in self.batch_obs]
        batch1 = [obs[1] for obs in self.batch_obs]
        return (torch.tensor(np.concatenate(batch0),dtype=torch.float32), torch.tensor(np.concatenate(batch1),dtype=torch.float32)), \
                F.one_hot(torch.tensor(np.array(self.batch_acts),dtype=torch.long),num_classes=self.act_dim), \
                torch.tensor(np.array(self.batch_log_probs),dtype=torch.float32), \
                torch.tensor(np.array(self.batch_rtgs),dtype=torch.float32), \
                self.batch_lens  
    
    def construct_input(self, gameState:capture.GameState) -> tuple[np.array]:
        image_input = np.expand_dims(self.one_hot(gameState),0)
        #x,y position -> 8 items
        positions = self.extract_positions(gameState)
        #scared?,timer -> 8 items
        scared = self.scared(gameState)
        #food needed -> 4 items
        food = self.percent_food_needed(gameState)
        return image_input, np.expand_dims(np.concatenate((positions, scared, food)),0)

def num_params(model):
    n = 0
    for v in model.parameters():
        n+=v.numel()
    return n


# n = default_net(12)
# input1 = torch.randn(1,7,46,26)
# input2 = torch.randn(1,12)
# n(input1,input2)

# t = time.time()

# input = torch.randn(1,7,46,26)
# c1 = nn.Conv2d(7,16,5,stride=3)
# c2 = nn.Conv2d(16,32,3,stride=2)
# c3 = nn.Conv2d(32,16,2,stride=1)
# c4 = nn.Linear(320, 128)
# c5 = nn.Linear(128,5)



# print(num_params(c1))
# print(num_params(c2))
# print(num_params(c3))
# print(num_params(c4))
# print(num_params(c5))

# res = c1(input)
# res = c2(res)
# res = c3(res)
# res = nn.Flatten()(res)
# res = c4(res)
# res = c5(res)


# print(f"TIME {time.time()-t}")
# print(res.size())
# print(np.prod(res.size()))

# t = time.time()

# input = torch.randn(1,4,84,84)
# c1 = nn.Conv2d(4,32,8,stride=4)
# c2 = nn.Conv2d(32,64,4,stride=2)
# c3 = nn.Conv2d(64,64,3,stride=1)
# c4 = nn.Linear(3136, 512)
# c5 = nn.Linear(512,18)

# print(num_params(c1))
# print(num_params(c2))
# print(num_params(c3))
# print(num_params(c4))
# print(num_params(c5))

# res = c1(input)
# res = c2(res)
# res = c3(res)
# res = nn.Flatten()(res)
# res = c4(res)
# res = c5(res)

# n = nn.Sequential(c1,c2,c3,c4,c5)
# dump_nnet(n,"time_test.out")
# print(list(load_parameters("time_test.out").keys()))


# print(f"TIME {time.time()-t}")
# print(res.size())
# print(np.prod(res.size()))
