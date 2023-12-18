# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game, capture
from util import nearestPoint
from itertools import product
import numpy as np 
from copy import deepcopy
from collections import deque, namedtuple
import pickle
import json

DEFAULT_DISTANCE=float('inf')
MAX_HEIGHT = 26
MAX_WIDTH = 46
default_width = 32
default_height = 16
directions = list(game.Actions._directions.keys())
print(f"DIRECTIONS {directions}")

# device = 'gpu' if torch.cuda.is_available() else 'cpu'

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first = 'FeatureQAgent', second = 'FeatureQAgent', numTraining = 0):
#   file1 = "temp"
#   file2 = "temp"

#   mlp1 = load_mlp(file1, 3)
#   mlp2 = load_mlp(file2, 3)
  
#   return [MLPQAgent(mlp1,firstIndex),MLPQAgent(mlp2,secondIndex)]

#   weights1 = util.Counter()+{'num_invaders': -0.013479729939564427, 'invader_distance1': 0.03629469993644022, 'defender_distance2': -0.01972775197185327, 'num_food': -0.098637537366583, 'nearest_food': 0.17136591888119238, 'num_capsules': -0.04895257882228501, 'nearest_capsule': 0.07639516409658403, 'num_returned': 0.3235903097276173, 'carried*distance': 0.44918195138694783}
#   weights2 = util.Counter()+{'num_invaders': -0.011736192788425368, 'invader_distance1': 0.04701688752960884, 'defender_distance2': -0.01972775197185327, 'num_food': -0.15432440626610708, 'nearest_food': 0.24777839420202466, 'num_capsules': -0.14258824477737306, 'nearest_capsule': 0.16537410244871983, 'num_returned': 0.5914549260014474, 'carried*distance': 0.8208652967410837}

  weights1 = util.Counter()+{"my_side":1, 'stopped':-10,'closest_pct_dist2':-2,'num_invaders': -20.02675723064652907, 'invader_distance': 0.05338341115202663, 'defender_distance2': -4*2.55172188893904676, 'num_food': -2.3890522818672748, 'nearest_food': 2.8550866808539808, 'num_capsules': -10.4712666917274702, 'nearest_capsule': 5.206300266430704, 'num_returned': 10.145592233476731, 'carried*distance': 9.9751003150469721}
  weights2=util.Counter()+{"my_side":-1,'stopped':-10,'closest_pct_dist2':-10,'closest_pct_dist':10,'num_invaders': -20.746859189935796, 'invader_distance': 2.4472217426187603, 'defender_distance2': -4*2.15172188893904676, 'num_food': -0.6054812234429122, 'nearest_food': 0.57987299691582, 'num_capsules': -3.7568271666189083, 'nearest_capsule': 0.51754101782037785, 'num_returned': 10.917579649610557, 'carried*distance': 3.23128985327646}

  return [FeatureQAgent(weights1, weights2, True, firstIndex), FeatureQAgent(weights2, weights1, False, secondIndex)]



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


def load_mlp(filename, layers):
  parameters = load_parameters(filename)
  return numpy_mlp(parameters, layers)

# class mlp(nn.Module):
#     def __init__(self, fan_in, layers, fan_out, bias=True):
#         super().__init__()
#         layers = [fan_in]+layers+[fan_out]
#         self.layers = nn.ModuleList([nn.Linear(layers[i-1],layers[i],bias=bias) for i in range(1,len(layers))])
#         with torch.no_grad():
#           for c in self.layers:
#             nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity="relu")
#             c.bias*=0

#     def reinitialize(self):
#       with torch.no_grad():
#           for c in self.layers:
#             nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity="relu")
#             c.bias*=0


#     def forward(self, x):
#         x = x if isinstance(x,torch.Tensor) else torch.tensor(x, dtype=torch.float32)
#         for l in self.layers[:-1]:
#             x = F.relu(l(x))
#         return self.layers[-1](x)
    
#     def zero(self):
#       with torch.no_grad():
#         for v in self.parameters():
#           v*=0

def load_parameters(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def __init__( self, index, timeForComputing = .1, weights=util.Counter()):
    super().__init__(index, timeForComputing=timeForComputing)
    self.weights = weights
    self.previous_score=0
    self.total_reward=0
    self.freeze=False
    

  def registerInitialState(self, gameState: capture.GameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    # print(f"AGENT {self.index} REGISTERING INITIAL STATE")

    '''
    Your initialization code goes here, if you need any.
    '''
    self.registerTeam(self.getTeam(gameState))
    self.start = gameState.getAgentPosition(self.index)
    self.boundary=self.get_boundary(gameState, self.red)
    self.opp_boundary=self.get_boundary(gameState, not self.red)
    self.enclosed = self.get_enclosed(gameState)
    self.opp_enclosed = self.get_enclosed(gameState, not self.red)
    self.num_total_food = {True:len(gameState.getRedFood().asList()), False: len(gameState.getBlueFood().asList())}
    # print(self.enclosed)


  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)
  
  def getSuccessor(self, gameState: capture.GameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState: capture.GameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState: capture.GameState, action):
    """
    Returns a counter of features for the state
    """
    return self.default_observation(gameState, action)

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return self.weights
  
  def default_observation(self, gameState:capture.GameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features 
  
  def default_weights(self, gameState:capture.GameState, action):
    return {'successorScore': 1.0} 
  
  def nearest(self, myPos, posList, default=DEFAULT_DISTANCE, distancer=None) -> tuple[int, tuple[int, int]]:
    minDistance= float('inf')
    minPos=None 
    distancer = self.distancer if distancer is None else distancer
    for pos in posList:
      dist = distancer.getDistance(myPos, pos)
      if dist<minDistance:
        minDistance=dist 
        minPos = pos
    return (default if minDistance==float('inf') else max(0.5,minDistance)), (myPos if minPos is None else minPos)
  
  def nearest_food(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int, int]]:
    foodList = self.getFood(gameState).asList()  
    myPos = gameState.getAgentState(self.index).getPosition()
    distancer = self.distancer if distancer is None else distancer
    return self.nearest(myPos, foodList, distancer=distancer)
  
  def num_food(self, gameState:capture.GameState) -> int:
    return len(self.getFood(gameState).asList())
  
  def nearest_capsule(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int, int]]:
    capsuleList = self.getCapsules(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    distancer = self.distancer if distancer is None else distancer
    return self.nearest(myPos, capsuleList, distancer=distancer)
  
  def num_capsules(self, gameState:capture.GameState) -> int:
    return len(self.getCapsules(gameState))
  
  def isPacman(self, gameState:capture.GameState) -> bool:
    myState = gameState.getAgentState(self.index)
    return myState.isPacman
  
  def pacmanGhost(self, gameState:capture.GameState) -> bool:
    pacman = [gameState.getAgentState(i).isPacman for i in self.getTeam(gameState)]
    return len(set(pacman))!=1
  
  def invaders(self, gameState:capture.GameState, indices=False):
    invaders = [(i if indices else gameState.getAgentPosition(i)) for i in self.getOpponents(gameState) \
                    if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]
    return invaders
  
  def opp_invaders(self, gameState:capture.GameState, indices=False):
    invaders = [(i if indices else gameState.getAgentPosition(i)) for i in self.getTeam(gameState) \
                    if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]
    return invaders
  
  def defenders(self, gameState:capture.GameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a.getPosition() for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer==0]
    return defenders
  
  def opp_defenders(self, gameState:capture.GameState):
    team = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
    invaders = [a.getPosition() for a in team if a.isPacman and a.getPosition() != None]
    return invaders
  
  def scared(self, gameState:capture.GameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a.getPosition() for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer>0]
    return defenders
  
  def num_invaders(self, gameState:capture.GameState) -> int:
    return len(self.invaders(gameState))
  
  def num_defenders(self, gameState:capture.GameState) -> int:
    return len(self.defenders(gameState))
  
  def num_scared(self, gameState:capture.GameState) -> int:
    return len(self.scared(gameState))
  
  def total_defender_distance(self, gameState:capture.GameState, inverse=True, distancer=None) -> int:
     defenders = self.defenders(gameState)
     mypos = gameState.getAgentPosition(self.index)
     distancer = self.distancer if distancer is None else distancer
     dists = [distancer.getDistance(mypos, d) for d in defenders]
     return sum([(1/d if inverse else d) for d in dists])
  
  def total_invader_distance(self, gameState:capture.GameState, inverse=True, distancer=None) -> int:
     invaders = self.invaders(gameState)
     mypos = gameState.getAgentPosition(self.index)
     distancer = self.distancer if distancer is None else distancer
     dists = [distancer.getDistance(mypos, i) for i in invaders]
     return sum([(1/d if inverse else d) for d in dists])

  def weighted_invader_distance(self, gameState:capture.GameState, inverse=True, distancer=None, index=None) -> int:
     index = self.index if index is None else index
     same_team = self.red==gameState.isOnRedTeam(index)
     invaders = self.invaders(gameState, indices=True) if same_team else self.opp_invaders(gameState, indices=True)
     mypos = gameState.getAgentPosition(index)
     distancer = self.distancer if distancer is None else distancer
     dists = [distancer.getDistance(mypos, gameState.getAgentPosition(i)) for i in invaders]
     carried = [gameState.getAgentState(i).numCarrying for i in invaders]
     return sum([(c/d if inverse else d/c) for c,d in zip(carried,dists)])

  def nearest_invader(self, gameState, distancer=None)-> int:
    distancer = self.distancer if distancer is None else distancer
    invaders = [i for i in self.getOpponents(gameState) if gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None]
    mypos = gameState.getAgentPosition(self.index)
    if len(invaders)==0:
      return -1
    return min(invaders, key = lambda i:distancer.getDistance(mypos, gameState.getAgentPosition(i)))
    
  
  
  def invaderDistance(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int, int]]:
    distancer = self.distancer if distancer is None else distancer
    invaders = self.invaders(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, invaders, distancer=distancer)
  
  def oppInvaderDistance(self, gameState:capture.GameState, index, distancer=None) -> tuple[int, tuple[int, int]]:
    distancer = self.distancer if distancer is None else distancer
    invaders = self.opp_invaders(gameState)
    myPos = gameState.getAgentState(index).getPosition()
    return self.nearest(myPos, invaders, distancer=distancer)
    
  def defenderDistance(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int, int]]:
    distancer = self.distancer if distancer is None else distancer
    defenders = self.defenders(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, defenders, distancer=distancer)
  
  def oppDefenderDistance(self, gameState:capture.GameState, index, distancer=None) -> tuple[int, tuple[int, int]]:
    distancer = self.distancer if distancer is None else distancer
    defenders = self.opp_defenders(gameState)
    myPos = gameState.getAgentState(index).getPosition()
    return self.nearest(myPos, defenders, distancer=distancer)
  
  def scaredDistance(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int, int]]:
    distancer = self.distancer if distancer is None else distancer
    scared = self.scared(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, scared, distancer=distancer)
  
  def num_carried(self, gameState:capture.GameState) -> int:
    return gameState.getAgentState(self.index).numCarrying
  
  def num_returned(self, gameState:capture.GameState) -> int:
    return gameState.getAgentState(self.index).numReturned
  
  def is_scared(self, gameState:capture.GameState) -> bool:
    return gameState.getAgentState(self.index).scaredTimer>0
  
  def get_dimensions(self, gameState:capture.GameState) -> tuple[int]:
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    return (width,height)
    
  
  def get_boundary(self, gameState, red):
    width,height = self.get_dimensions(gameState)
    halfway = width//2
    if red:
      boundary = [(halfway-1,h) for h in range(height) if not gameState.hasWall(halfway-1,h)]
    else:
      boundary = [(halfway,h) for h in range(height) if not gameState.hasWall(halfway,h)]
    return boundary
  
  def distance_to_home(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int, int]]:
    myPos = gameState.getAgentState(self.index).getPosition()
    distancer = self.distancer if distancer is None else distancer
    dist,pos = self.nearest(myPos, self.get_boundary(gameState, self.red), distancer=distancer)
    return max(.5, dist), pos

  def opp_distance_to_home(self, gameState:capture.GameState, index, distancer=None) -> tuple[int, tuple[int, int]]:
    distancer = self.distancer if distancer is None else distancer
    myPos = gameState.getAgentState(index).getPosition()
    dist,pos = self.nearest(myPos, self.get_boundary(gameState, not self.red), distancer=distancer)
    return max(.5, dist), pos
  
  def distance_to_teammate(self, gameState:capture.GameState, distancer=None) -> tuple[int, tuple[int,int]]:
    distancer = self.distancer if distancer is None else distancer
    other = (self.index+2)%4
    other_pos = gameState.getAgentPosition(other)
    self_pos = gameState.getAgentPosition(self.index)
    return self.nearest(self_pos, [other_pos], distancer=distancer)
  
  def halfList(self, l, grid, red):
    halfway = grid.width // 2
    newList = []
    for x,y in l:
      if red and x < halfway: newList.append((x,y))
      elif not red and x >= halfway: newList.append((x,y))
    return newList
  
  def get_enclosed(self, gameState:capture.GameState, red=None) -> set[tuple[int,int]]:
    red = self.red if red is None else red
    enclosed = set()
    width,height = self.get_dimensions(gameState)
    positions = product(range(width),range(height))
    
    walls = gameState.getWalls()
    positions = set(self.halfList(positions, walls, not red))
    # self.positions= list(deepcopy(positions))
    def check_enclosed(i,j, enclosed, walls):
      if i<0 or j<0 or i>=width or j>=height:
        return True 
      if walls[i][j]:
        return True
      if (i,j) in enclosed:
        return True
      return False
    def propagate_enclosed(i,j, enclosed, walls):
      if walls[i][j]:
        return False
      if (i,j) in enclosed:
        return False
      if sum([check_enclosed(i+di,j+dj, enclosed, walls) for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]])>=3:
        return True
      return False
    
    def single_iteration(positions:set, enclosed:set, walls):
      new_pos = set()
      for i,j in positions:
        if propagate_enclosed(i,j, enclosed, walls):
          new_pos.add((i,j))
      positions.difference_update(new_pos)
      enclosed.update(new_pos)
      return len(new_pos), positions, enclosed
    
    while True:
       num_added, positions, enclosed = single_iteration(positions, enclosed, walls)
      #  print(f"NUM ADDED {num_added}")
       if num_added==0:
          break
    # self.positions=enclosed
    return enclosed
       

       
    

  
  def one_hot(self, gameState:capture.GameState, include_pos = False) -> np.array:
    t = time.time()
    #Should cache some of these if there's runtime issues
    red_food = np.zeros([MAX_WIDTH, MAX_HEIGHT])
    for x,y in gameState.getRedFood().asList():
      red_food[x][y]=1
    # red_food = np.array(gameState.getRedFood().data, dtype=np.float32)
    red_capsules = np.zeros_like(red_food)
    for x,y in gameState.getRedCapsules():
      red_capsules[x][y]=1

    blue_food = np.zeros_like(red_food)
    for x,y in gameState.getBlueFood().asList():
      blue_food[x][y]=1
    # blue_food = np.array(gameState.getBlueFood().data, dtype=np.float32)
    blue_capsules = np.zeros_like(blue_food)
    for x,y in gameState.getBlueCapsules():
      blue_capsules[x][y]=1

    walls = np.array(gameState.getWalls().data, dtype=np.float32)
    width, height = gameState.data.layout.width, gameState.data.layout.height
    walls = np.pad(walls, ((0,MAX_WIDTH-width), (0,MAX_HEIGHT-height)), 'constant', constant_values=1)

    red = np.array([[int(w<width/2)]*height for w in range(width)])
    red = np.pad(red, ((0,MAX_WIDTH-width), (0,MAX_HEIGHT-height)), 'constant', constant_values=0)
    blue = np.array([[int(w>=width/2)]*height for w in range(width)])
    blue = np.pad(blue, ((0,MAX_WIDTH-width), (0,MAX_HEIGHT-height)), 'constant', constant_values=0)
    if include_pos:
      pos = [np.zeros_like(red_food) for _ in range(4)]
      for i in self.agent_ordering(gameState):
         ithpos = gameState.getAgentPosition(i)
         pos[i][ithpos[0]][ithpos[1]]=1
    else:
      pos = []

         
    # print(f"ONE HOT TIME {time.time()-t}")
    if self.red:
      return np.stack([red_food, red_capsules, red, blue_food, blue_capsules, blue, walls]+pos, axis=0)
    return np.stack([blue_food, blue_capsules, blue, red_food, red_capsules, red, walls]+pos, axis=0)
  
  def agent_ordering(self, gameState:capture.GameState) -> list[int]:
    myTeam = self.getTeam(gameState)
    oppTeam = self.getOpponents(gameState)
    if self.index ==min(myTeam):
      return [min(myTeam), max(myTeam), min(oppTeam), max(oppTeam)]
    else:
      return [max(myTeam), min(myTeam), max(oppTeam), min(oppTeam)]
  def convert_order(self, gameState:capture.GameState, index=None)-> int:
    index = self.index if index is None else index
    ordering = self.agent_ordering(gameState)
    return ordering.index(index)

  def extract_positions(self, gameState:capture.GameState) -> np.array:
    indices = self.agent_ordering(gameState)
    # print(f" MY INDEX {self.index} MyTEAM {myTeam} oppTeam {oppTeam} INDICES {indices}")
    opps = [gameState.getAgentState(i).getPosition() for i in indices]
    w,h = gameState.data.layout.width, gameState.data.layout.height 
    opps = [(i/w,j/h) for i,j in opps]
    return np.array(opps, dtype=np.float32).reshape([-1])

  def scared(self, gameState:capture.GameState) -> np.array:
    scared_time=40
    indices = self.agent_ordering(gameState)
    timers = [gameState.getAgentState(i).scaredTimer for i in indices]
    return np.array([[t/scared_time,float(t>0)] for t in timers],dtype=np.float32).reshape([-1])
  
  def food_needed_to_win(self, gameState:capture.GameState, red=False) -> int:
    total_food = 60 
    min_food = 2 
    other_food = len(gameState.getRedFood().asList()) if red else len(gameState.getRedFood().asList())
    return other_food-min_food 
  
  def percent_food_needed(self, gameState:capture.GameState) -> np.array:
    indices = self.agent_ordering(gameState)
    self_food, other_food = self.food_needed_to_win(gameState, self.red), self.food_needed_to_win(gameState,not self.red)
    carried = [(gameState.getAgentState(i).numCarrying/f if f>0 else 1) for i,f in zip(indices, [self_food]*2+[other_food]*2)]
    return np.array(carried, dtype=np.float32)
  
  
  def rewardFunction(self, gameState: capture.GameState):
    if self.reward is None:
      cumulative_score = self.getScore(gameState)
      score = cumulative_score-self.previous_score
      self.previous_score=cumulative_score
      return score
    else:
      return self.reward.reward(gameState) 

class Agent1(DummyAgent):
  
  def getFeatures(self, gameState: capture.GameState, action=None, distancer=None):
    # print(self.one_hot(gameState).shape)
    # print(np.array(gameState.getRedFood().data).shape)
    """
    Returns a counter of features for the state
    """

    distancer = self.distancer if distancer is None else distancer
    mypos = gameState.getAgentPosition(self.index)
    features = util.Counter()
    if action is not None:
      successor = self.getSuccessor(gameState, action)
    else:
      successor = gameState
    next_pos = successor.getAgentPosition(self.index)
    w,h = self.get_dimensions(gameState)
    my_side = next_pos[0]/w if self.red else -next_pos[0]/w 
    features["my_side"] = my_side
    features['stopped']=action==Directions.STOP
    features["num_invaders"] = self.num_invaders(successor)-self.num_invaders(gameState)
    inv_dist = self.invaderDistance(successor, distancer=distancer)[0]
    features['invader_distance1'] = 1/inv_dist-min(0,features["num_invaders"])
    features['invader_distance2'] = 1/inv_dist**2-min(0,features["num_invaders"])
    invader_food_needed = self.food_needed_to_win(gameState,not self.red)
    features['invader_pct_carried'] = 1 if invader_food_needed==0 else sum([gameState.getAgentState(i).numCarrying for i in self.getOpponents(gameState)])/self.food_needed_to_win(gameState,not self.red)
    closest_invader = self.nearest_invader(gameState, distancer=distancer)
    features['closest_pct_dist'] = 0
    if closest_invader>-1:
      features['closest_pct_dist'] = self.percent_food_needed(successor)[self.convert_order(successor,index=closest_invader)]/ \
                                      self.opp_distance_to_home(successor, closest_invader, distancer=distancer)[0] / \
                                      np.sqrt(max(0.1,distancer.getDistance(next_pos, successor.getAgentPosition(closest_invader)))) + features["invader_pct_carried"]
    features['closest_pct_dist2'] = 0
    w,h = gameState.data.layout.width, gameState.data.layout.height
    if closest_invader>-1:
      features['closest_pct_dist2'] = self.percent_food_needed(successor)[self.convert_order(successor,index=closest_invader)]/ \
                                      np.sqrt(self.opp_distance_to_home(successor, closest_invader, distancer=distancer)[0]) * \
                                      np.sqrt(max(0.1,distancer.getDistance(next_pos, successor.getAgentPosition(closest_invader)))) + features["invader_pct_carried"]
      
    # print(f"PCT_DIST {features['closest_pct_dist']}")
    def_dist = self.defenderDistance(successor, distancer=distancer)[0]

    features['defender_distance2'] = 1/def_dist**3 if self.isPacman(successor) else 0
    features['defender_distance1'] = 1/def_dist if self.isPacman(successor) else 0
    # features['scared_distance'] = 1/self.scaredDistance(successor)[0]
    features['num_food'] = self.num_food(successor)-self.num_food(gameState)
    features['nearest_food'] = 1/self.nearest_food(successor, distancer=distancer)[0]-features['num_food']#/(gameState.data.layout.width*gameState.data.layout.height)
    features['num_capsules'] = self.num_capsules(successor)-self.num_capsules(gameState)
    features['nearest_capsule'] = 1/self.nearest_capsule(successor, distancer=distancer)[0]-features['num_capsules']
    # features["bias"]=1.0
    features["enclosed"] = (int(successor.getAgentPosition(self.index) in self.enclosed) - int(gameState.getAgentPosition(self.index) in self.enclosed)) and def_dist<4
    # teammate_distance=self.distance_to_teammate(successor, distancer=distancer)[0]
    # features["teammate_distance"]=1/teammate_distance**2 if teammate_distance<5 else 0
       
    # features['pacman_ghost'] = int(self.pacmanGhost(successor))
    # features['num_carried']= self.num_carried(successor)
    # features['distance_to_home']=1/self.distance_to_home(successor)[0]
    features['num_returned']=self.num_returned(successor)-self.num_returned(gameState)
    features['pct_returned']=self.percent_food_needed(successor)[self.convert_order(successor)]-self.percent_food_needed(gameState)[self.convert_order(successor)]
    features['carried*distance']=self.num_carried(successor)/self.distance_to_home(successor, distancer=distancer)[0]+features['num_returned']
    self_food_needed = self.food_needed_to_win(gameState,self.red)
    features['pct_carried*distance']=self.percent_food_needed(successor)[self.convert_order(successor)]/self.distance_to_home(successor, distancer=distancer)[0]+\
                                      (1 if self_food_needed==0 else features['num_returned']/self_food_needed)
    
    # features.divideAll(20.0)
    # print(f"FEATURES {features}")

    return features
  


class FeatureQAgent(Agent1):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, weights1, weights2, atk, index, timeForComputing = .1, **kwargs):
        Agent1.__init__(self, index, timeForComputing=timeForComputing, **kwargs)
        # self.weights["num_capsules"]=1.0
        self.epsilon=0.2
        self.total_reward=0
        self.weights1 = weights1 
        self.weights2 = weights2 
        self.atk = atk
        
    def switchWeights(self):
        # return
        temp = self.weights1
        self.weights1=self.weights2 
        self.weights2 = temp 
    def registerInitialState(self, gameState: capture.GameState):
      Agent1.registerInitialState(self, gameState)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        features = self.getFeatures(state,action)
        return sum([features[f]*self.weights1[f] for f in features])
    

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = state.getLegalActions(self.index)
        if len(actions)==0:
            return None
        rew = [self.getQValue(state,a) for a in actions]
        m = max(rew)
        argmax = random.choice([i for i in range(len(rew)) if rew[i]==m])
        return actions[argmax]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = state.getLegalActions(self.index)

        other_index = (self.index+2)%4 
        w,h = state.data.layout.width, state.data.layout.height

        my_width = state.getAgentPosition(self.index)[0]
        other_width = state.getAgentPosition(other_index)[0]
        if self.red:
          other_more_atk = other_width>w/2+1 and my_width<w/2-1
        else:
          other_more_atk = other_width<w/2-1 and my_width>w/2+1

        if (not self.atk and not other_more_atk) or (self.atk and other_more_atk):
          self.switchWeights()
          self.atk=not self.atk 
          # print(f"AGENT {self.index} SWITCHING WEIGHTS FROM {not self.atk} TO {self.atk}")

        if  util.flipCoin(0.01):
           return random.choice(legalActions)
        else:
           return self.computeActionFromQValues(state)

class MLPQAgent(FeatureQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, mlp, index, **kwargs):
        super().__init__(index, **kwargs)
        self.mlp = mlp

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.getFeatures(state,action)
        features = np.array(list(features.values()))
        return self.mlp(features)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = state.getLegalActions(self.index)
        if self.train and util.flipCoin(0.01):
            action= random.choice(legalActions)
        else:
            action= self.computeActionFromQValues(state)
        return action
