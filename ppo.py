import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from myTeam import Agent1
import game, capture

directions = list(game.Actions._directions.keys())
class PPOAgent(DummyAgent):
    def __init__(self, index, timeForComputing = .1, nnet=lambda x:[1.,1,1,1,1], construct_input=lambda x:np.array([1,1])):
        super().__init__(index, timeForComputing = timeForComputing)
        self.nnet=nnet 
        self.construct_input=construct_input
        self.logger = {"observations":[]}
         
    def chooseAction(self, gameState: capture.GameState):
        parser = Parser()
        actions = gameState.getLegalActions(self.index)
        # print(f"AGENT STATES {[parser.parse_state(gameState.getAgentState(i)) for i in range(gameState.getNumAgents()) if gameState.getAgentState(i).numReturned>0]}")
        input = self.construct_input(gameState)
        output = self.nnet(input)
        policy = np.array([output[directions.index(a)] for a in actions])
        policy = np.e**policy
        policy = policy/np.sum(policy)
        # print(f"POLICY: {policy}")
        rng = np.random.default_rng()
        return actions[rng.choice(len(actions),p=policy)]
    
    def reset(self):
        self.logger = {"observations":[], "log probs":[], "actions":[]}

class PPO:
    def __init__(self, policy_class):
        self.obs_dim=10
        self.act_dim=5
        self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

    
    def learn(self, total_timesteps):
        """
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:                                                                       # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
            i_so_far += 1

			# Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

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
            self._log_summary()

			# # Save our model if it's time
            # if i_so_far % self.save_freq == 0:
            #     torch.save(self.actor.state_dict(), './ppo_actor.pth')
            #     torch.save(self.critic.state_dict(), './ppo_critic.pth')

    
    def rollout(self):
        """
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

			# Reset the environment. sNote that obs is short for observation. 
            obs = self.env.reset()
            done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
                batch_obs.append(obs)

				# Calculate action and make a step in the env. 
				# Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)

				# Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
                if done:
                    break

			# Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


from capture import CaptureRules
def runGames( layouts, agents, display, length, numGames, record, numTraining, redTeamName, blueTeamName, muteAgents=True, catchExceptions=False ):

  rules = CaptureRules()
  games = []

#   if numTraining > 0:
#     print('Playing %d training games' % numTraining)

  for i in range( numGames ):
    beQuiet = i < numTraining
    layout = layouts[i]
    if beQuiet:
        # Suppress output and graphics
        import textDisplay
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        gameDisplay = display
        rules.quiet = False
    g = rules.newGame( layout, agents, gameDisplay, length, muteAgents, catchExceptions )
    g.run()
    if not beQuiet: games.append(g)

    g.record = None
    if record:
      import time, pickle, game
      #fname = ('recorded-game-%d' % (i + 1)) +  '-'.join([str(t) for t in time.localtime()[1:6]])
      #f = file(fname, 'w')
      components = {'layout': layout, 'agents': [game.Agent() for a in agents], 'actions': g.moveHistory, 'length': length, 'redTeamName': redTeamName, 'blueTeamName':blueTeamName }
      #f.close()
      print("recorded")
      g.record = pickle.dumps(components)
      with open('replay-%d'%i,'wb') as f:
        f.write(g.record)

  if numGames > 1:
    scores = [game.state.data.score for game in games]
    redWinRate = [s > 0 for s in scores].count(True)/ float(len(scores))
    blueWinRate = [s < 0 for s in scores].count(True)/ float(len(scores))
    print('Average Score:', sum(scores) / float(len(scores)))
    print('Scores:       ', ', '.join([str(score) for score in scores]))
    print('Red Win Rate:  %d/%d (%.2f)' % ([s > 0 for s in scores].count(True), len(scores), redWinRate))
    print('Blue Win Rate: %d/%d (%.2f)' % ([s < 0 for s in scores].count(True), len(scores), blueWinRate))
    print('Record:       ', ', '.join([('Blue', 'Tie', 'Red')[max(0, min(2, 1 + s))] for s in scores]))
  return games