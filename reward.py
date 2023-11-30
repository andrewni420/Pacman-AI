#Calculates a reward based on a state action transition

class Reward:
    def __init__(self):
        pass 

    def reward(self, prev_state, action, state):
        pass 

    def reverse_reward(self, reward, state):
        red_indices = state.getRedTeamIndices()
        red_reward = sum([reward[i] for i in red_indices])
        blue_reward = sum(reward)-red_reward
        rev_reward = [(-blue_reward if i in red_indices else -red_reward) for i in range(len(reward))]
        return rev_reward

    def food_reward(self, prev_state, action, state, reverse=False):
        prev_agents = [prev_state.getAgentState(i) for i in range(prev_state.getNumAgents())]
        agents = [state.getAgentState(i) for i in range(prev_state.getNumAgents())]
        food = [a.numCarrying-p.numCarrying for p,a in zip(prev_agents, agents)]
        if reverse:
            return food, self.reverse_reward(food, state)
        return food
    
    def return_reward(self, prev_state, action, state, reverse=False):
        prev_agents = [prev_state.getAgentState(i) for i in range(prev_state.getNumAgents())]
        agents = [state.getAgentState(i) for i in range(prev_state.getNumAgents())]
        returned_food = [a.numReturned-p.numReturned for p,a in zip(prev_agents, agents)]
        if reverse:
            return returned_food, self.reverse_reward(returned_food, state)
        return returned_food



    





class FoodReward(Reward):
    def reward(self, prev_state,action,state):
        return self.food_reward(prev_state, action, state)