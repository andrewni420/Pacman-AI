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
from gameparser import Parser
from reward import Reward
from baselineTeam import ReflexCaptureAgent
from util import nearestPoint
from itertools import product
import numpy as np 
from qlearningAgent import PacmanQAgent
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from collections import deque, namedtuple

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
  {"num_invaders", "invader_distance", "defender_distance", "num_food", "num_capsules", "nearest_capsule", 
   "bias", "enclosed", "num_returned", "carried*distance"}

  # The following line is an example only; feel free to change it.
  # mlp1 = nn.Linear(10,1,bias=False)
  # mlp2 = nn.Linear(10,1,bias=False)
  # with torch.no_grad():
  #   mlp1.weight*=0
  #   mlp2.weight*=0
  # mlp1 = mlp(11,[256,256],1)
  # mlp2 = mlp(11,[256,256],1)
  # mlp1.zero()
  # mlp2.zero()
  # mlp1 = mlpwrapper(11,[],1, bias=False)
  # mlp2 = mlpwrapper(11,[],1, bias=False)
  # with torch.no_grad():
    # mlp1.layer.bias*=0
    # mlp2.layer.bias*=0
    # mlp1.layer.weight*=0
    # mlp1.layer.weight+=torch.tensor([[ -0.257,  3.6846, -0.0344,  3.6624, 12.0863, -3.2776,  7.3719,  3.0490,
    #       0.9243,  6.7572, 10.0]])#torch.tensor([[-0.0570,  0.1253,  0.496, -0.2305,  0.3851, -0.0480,  0.0940,  0.6522,0.0308,  0.4601,  0.6621]])
    # mlp2.layer.weight*=0
    # mlp2.layer.weight+=torch.tensor([[ -0.2006,  3.8246, -0.1700,  4.1418, 12.6177, -3.1362,  7.5170,  3.0073,
    #       0.9375,  6.3357, 10.0]])#torch.tensor([[-0.0378,  0.1216,  0.779, -0.1938,  0.3656, -0.1227,  0.1703,  0.8146, 0.0139,  0.7397,  1.0736]])
  # mlp1 = default_net(20,1, channels=11)
  # mlp2 = default_net(20,1, channels=11)
  # return [MultiOutputDQN(mlp1, firstIndex, cnn=True, include_pos=True), MultiOutputDQN(mlp2, secondIndex, cnn=True, include_pos=True)]
  # return [eval(first)(firstIndex), eval(second)(secondIndex)]
  return [FeatureQAgent(firstIndex), FeatureQAgent(secondIndex)]
  # return [MLPDDQN(mlp1,firstIndex, cnn=True, include_pos=True), MLPDDQN(mlp2,secondIndex, cnn=True, include_pos=True)]
  # return [MLPReflexAgent(mlp1, firstIndex), MLPReflexAgent(mlp2, secondIndex)]

class mlpwrapper(nn.Module):
    def __init__(self, fan_in, layers, fan_out, bias=True):
        super().__init__()
        self.layer = nn.Linear(fan_in,fan_out,bias=bias)
        with torch.no_grad():
           self.layer.weight*=0
          #  self.layer.bias*=0
    def forward(self, x):
        return self.layer(x)
   

class mlp(nn.Module):
    def __init__(self, fan_in, layers, fan_out, bias=True):
        super().__init__()
        layers = [fan_in]+layers+[fan_out]
        self.layers = nn.ModuleList([nn.Linear(layers[i-1],layers[i],bias=bias) for i in range(1,len(layers))])
        with torch.no_grad():
          for c in self.layers:
            nn.init.kaiming_normal_(c.weight, mode='fan_in', nonlinearity="relu")
            c.bias*=0

    def forward(self, x):
        x = x if isinstance(x,torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)
    
    def zero(self):
      with torch.no_grad():
        for v in self.parameters():
          v*=0
    
class default_net(nn.Module):
    def __init__(self,aux_inputs, outputs, channels=7):
        super().__init__()
        self.c1 = nn.Conv2d(channels,16,5,stride=3)
        self.c2 = nn.Conv2d(16,32,3,stride=2)
        self.c3 = nn.Conv2d(32,16,2,stride=1)
        self.l4 = nn.Linear(320, 128)
        self.l5 = nn.Linear(128,outputs)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(aux_inputs, 160)
        with torch.no_grad():
          for c in [self.c1,self.c2,self.c3,self.l4,self.l5,self.l1]:
            nn.init.kaiming_normal_(c.weight, mode='fan_out', nonlinearity="relu")
            c.bias*=0

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

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def __init__( self, index, timeForComputing = .1, parser:Parser=None, reward:Reward=None):
    super().__init__(index, timeForComputing=timeForComputing)
    self.parser=parser
    self.reward = reward
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

    '''
    Your initialization code goes here, if you need any.
    '''
    self.registerTeam(self.getTeam(gameState))
    self.start = gameState.getAgentPosition(self.index)
    self.boundary=self.get_boundary(gameState, self.red)
    self.opp_boundary=self.get_boundary(gameState, not self.red)
    self.enclosed = self.get_enclosed(gameState)
    self.opp_enclosed = self.get_enclosed(gameState, not self.red)
    # print(self.enclosed)


  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    # try:
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)
  
    # finally:
    #   return random.choice(actions)
  
  
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
    return self.default_weights(gameState, action)
  
  def default_observation(self, gameState:capture.GameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features 
  
  def default_weights(self, gameState:capture.GameState, action):
    return {'successorScore': 1.0} 
  
  def nearest(self, myPos, posList, default=DEFAULT_DISTANCE) -> tuple[int, tuple[int, int]]:
    minDistance= float('inf')
    minPos=None 
    for pos in posList:
      dist = self.getMazeDistance(myPos, pos)
      if dist<minDistance:
        minDistance=dist 
        minPos = pos
    return (default if minDistance==float('inf') else max(0.5,minDistance)), (myPos if minPos is None else minPos)
  
  def nearest_food(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    foodList = self.getFood(gameState).asList()  
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, foodList)
  
  def num_food(self, gameState:capture.GameState) -> int:
    return len(self.getFood(gameState).asList())
  
  def nearest_capsule(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    capsuleList = self.getCapsules(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, capsuleList)
  
  def num_capsules(self, gameState:capture.GameState) -> int:
    return len(self.getCapsules(gameState))
  
  def isPacman(self, gameState:capture.GameState) -> bool:
    myState = gameState.getAgentState(self.index)
    return myState.isPacman
  
  def pacmanGhost(self, gameState:capture.GameState) -> bool:
    pacman = [gameState.getAgentState(i).isPacman for i in self.getTeam(gameState)]
    return len(set(pacman))!=1
  
  def invaders(self, gameState:capture.GameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
    return invaders
  
  def opp_invaders(self, gameState:capture.GameState):
    team = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
    invaders = [a.getPosition() for a in team if a.isPacman and a.getPosition() != None]
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
  
  def total_defender_distance(self, gameState:capture.GameState, inverse=True) -> int:
     defenders = self.defenders(gameState)
     mypos = gameState.getAgentPosition(self.index)
     dists = [self.getMazeDistance(mypos, d) for d in defenders]
     return sum([(1/d if inverse else d) for d in dists])
  
  def total_invader_distance(self, gameState:capture.GameState, inverse=True) -> int:
     invaders = self.invaders(gameState)
     mypos = gameState.getAgentPosition(self.index)
     dists = [self.getMazeDistance(mypos, i) for i in invaders]
     return sum([(1/d if inverse else d) for d in dists])
  
  
  def invaderDistance(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    invaders = self.invaders(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, invaders)
  
  def oppInvaderDistance(self, gameState:capture.GameState, index) -> tuple[int, tuple[int, int]]:
    invaders = self.opp_invaders(gameState)
    myPos = gameState.getAgentState(index).getPosition()
    return self.nearest(myPos, invaders)
    
  def defenderDistance(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    defenders = self.defenders(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, defenders)
  
  def oppDefenderDistance(self, gameState:capture.GameState, index) -> tuple[int, tuple[int, int]]:
    defenders = self.opp_defenders(gameState)
    myPos = gameState.getAgentState(index).getPosition()
    return self.nearest(myPos, defenders)
  
  def scaredDistance(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    scared = self.scared(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, scared)
  
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
  
  def distance_to_home(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    myPos = gameState.getAgentState(self.index).getPosition()
    dist,pos = self.nearest(myPos, self.get_boundary(gameState, self.red))
    return max(.5, dist), pos

  def opp_distance_to_home(self, gameState:capture.GameState, index) -> tuple[int, tuple[int, int]]:
    myPos = gameState.getAgentState(index).getPosition()
    dist,pos = self.nearest(myPos, self.get_boundary(gameState, not self.red))
    return max(.5, dist), pos
  
  def distance_to_teammate(self, gameState:capture.GameState) -> tuple[int, tuple[int,int]]:
    other = (self.index+2)%4
    other_pos = gameState.getAgentPosition(other)
    self_pos = gameState.getAgentPosition(self.index)
    return self.nearest(self_pos, [other_pos])
  
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
  


  
# x,y,z as a percent of global coordinates? Channels? 
# Scared boolean 
# Number carried as percent of remaining needed to win
# 

  # Agent index for querying state
  #   self.index = index
  # Whether or not you're on the red team
  #   self.red = None
  # Agent objects controlling you and your teammates
  #   self.agentsOnTeam = None
  # Maze distance calculator
  #   self.distancer = None
  # A history of observations
  #   self.observationHistory = []
  # Time to spend each turn on computing maze distances
  #   self.timeForComputing = timeForComputing
  # Access to the graphics
  #   self.display = None

  # def registerInitialState(self, gameState: capture.GameState):
  # def final(self, gameState: capture.GameState):
  # def registerTeam(self, agentsOnTeam):
  # def observationFunction(self, gameState: capture.GameState):
  # def debugDraw(self, cells, color, clear=False):
  # def debugClear(self):
  # def getAction(self, gameState: capture.GameState):
  # def chooseAction(self, gameState: capture.GameState):
  # def getFood(self, gameState: capture.GameState):
  # def getFoodYouAreDefending(self, gameState: capture.GameState):
  # def getCapsules(self, gameState: capture.GameState):
  # def getCapsulesYouAreDefending(self, gameState: capture.GameState):
  # def getOpponents(self, gameState: capture.GameState):
  # def getTeam(self, gameState: capture.GameState):
  # def getScore(self, gameState: capture.GameState):
  # def getMazeDistance(self, pos1, pos2):
  # def getPreviousObservation(self):
  # def getCurrentObservation(self):

  def observationFunction(self, gameState: capture.GameState):
    if self.parser is None:
      return super().observationFunction(gameState)
    else:
      return self.parser.parse_game(gameState)
  
  def rewardFunction(self, gameState: capture.GameState):
    if self.reward is None:
      cumulative_score = self.getScore(gameState)
      score = cumulative_score-self.previous_score
      self.previous_score=cumulative_score
      return score
    else:
      return self.reward.reward(gameState) 

class Agent1(DummyAgent):
  
  def getFeatures(self, gameState: capture.GameState, action=None):
    # print(self.one_hot(gameState).shape)
    # print(np.array(gameState.getRedFood().data).shape)
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    if action is not None:
      successor = self.getSuccessor(gameState, action)
    else:
      successor = gameState
    features["num_invaders"] = self.num_invaders(successor)-self.num_invaders(gameState)
    features['invader_distance'] = 1/self.invaderDistance(successor)[0]-features["num_invaders"]
    features['defender_distance'] = 4/self.defenderDistance(successor)[0]**2 if self.isPacman else 0
    # features['scared_distance'] = 1/self.scaredDistance(successor)[0]
    features['num_food'] = self.num_food(successor)-self.num_food(gameState)
    features['nearest_food'] = 1/self.nearest_food(successor)[0]-features['num_food']#/(gameState.data.layout.width*gameState.data.layout.height)
    features['num_capsules'] = self.num_capsules(successor)-self.num_capsules(gameState)
    features['nearest_capsule'] = 1/self.nearest_capsule(successor)[0]-features['num_capsules']
    # features["bias"]=1.0
    # features["enclosed"] = (int(successor.getAgentPosition(self.index) in self.enclosed) - int(gameState.getAgentPosition(self.index) in self.enclosed)) and self.defenderDistance(gameState)[0]<4
    teammate_distance=self.distance_to_teammate(successor)[0]
    # features["teammate_distance"]=0#1/teammate_distance if teammate_distance<5 else 0
       
    # features['pacman_ghost'] = int(self.pacmanGhost(successor))
    # features['num_carried']= self.num_carried(successor)
    # features['distance_to_home']=1/self.distance_to_home(successor)[0]
    features['num_returned']=self.num_returned(successor)-self.num_returned(gameState)
    features['carried*distance']=self.num_carried(successor)/self.distance_to_home(successor)[0]+features['num_returned']#-self.num_carried(gameState)/self.distance_to_home(gameState)[0]
    
    features.divideAll(20.0)

    return features


  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {
            'num_invaders': -1000,
            'invader_distance': 100,
            'defender_distance':-10,
            'num_food':-100,
            'nearest_food':10,
            # 'num_capsules':-100,
            # 'nearest_capsule':10,
            'pacman_ghost':10
            }
  


class FeatureQAgent(PacmanQAgent, Agent1):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, index, timeForComputing = .1, parser:Parser=None, reward:Reward=None, **kwargs):
        self.featExtractor = self.getFeatures
        PacmanQAgent.__init__(self, **kwargs)
        Agent1.__init__(self, index, timeForComputing=timeForComputing, parser=parser, reward=reward)
        self.weights = util.Counter()
        # self.weights["num_capsules"]=1.0
        self.epsilon=0.2
        self.total_reward=0

    def getWeights(self):
        return self.weights
    
    def registerInitialState(self, gameState: capture.GameState):
      Agent1.registerInitialState(self, gameState)
      PacmanQAgent.registerInitialState(self, gameState)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        features = self.featExtractor(state,action)
        return sum([features[f]*self.weights[f] for f in features])

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
        # Pick Action
        t = time.time()
        # print(f"WIDTH {state.data.layout.width}, HEIGHT {state.data.layout.height}" )
        legalActions = state.getLegalActions(self.index)

        if self.train and util.flipCoin(0.2):
           return random.choice(legalActions)
        else:
           return self.computeActionFromQValues(state)
        
        
        if self.train:
            eps = self.epsilon*np.cos(self.percent*np.pi)
            if util.flipCoin(eps):
              action= random.choice(legalActions)
            else:
              action= self.computeActionFromQValues(state)
        else:
            action= self.computeActionFromQValues(state)
        # print(f"ACTION TIME {time.time()-t}")
        return action
        
    def training(self, percent):
      self.train=True
      self.percent = percent
    def eval(self):
      self.train=False
        
        

    def update(self, state, action, nextState, reward: float):
        # print(f"UPDATING")
        """
           Should update your weights based on transition
        """
        # print(f"AGENT {self.index} REWARD {reward}")
        self.total_reward+=reward

        difference = reward+self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state,action)
        features = self.featExtractor(state,action)

        self.alpha=0.1
        self.discount=0.9

        # print(f"PARAMETERS: {self.alpha} {self.discount} {difference}")
        # print("BEFORE")
        # print(self.weights)

        for f in features.keys():
            # print(features[f])
            self.weights[f]+=self.alpha*difference*features[f]
        # print("AFTER")
        # print(self.weights)


    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)
        print(f"AGENT {self.index} WEIGHTS {self.weights} REWARD {self.total_reward}")
        self.total_reward=0

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
        

class MLPQAgent(FeatureQAgent):
    def __init__(self, mlp, index, **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = self.getFeatures
        self.mlp = mlp
        self.mlp_copy = deepcopy(self.mlp)
        # self.mlp_copy = self.mlp.weight.data.clone().detach()
        # self.optim = optim.SGD(self.mlp.parameters(), lr=0.1)
        self.optim = optim.Adam(self.mlp.parameters(), lr=0.001)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        features = self.featExtractor(state,action)
        features = torch.tensor(list(features.values()))
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
        # Pick Action
        t = time.time()
        # print(f"WIDTH {state.data.layout.width}, HEIGHT {state.data.layout.height}" )
        self.epsilon=0.2
        legalActions = state.getLegalActions(self.index)
        if self.train and util.flipCoin(self.epsilon):
            action= random.choice(legalActions)
        else:
            action= self.computeActionFromQValues(state)
        # print(f"ACTION TIME {time.time()-t}")
        return action
        
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = state.getLegalActions(self.index)
        if len(actions)==0:
            return 0.0
        rew = [self.getQValue(state,a) for a in actions]
        m = torch.cat(rew).max()
        # argmax = random.choice([i for i in range(len(rew)) if rew[i]==m])
        return m

    def update(self, state, action, nextState, reward: float):
        # print(f"UPDATING")
        """
           Should update your weights based on transition
        """
        self.optim.zero_grad()

        difference = reward+self.discount*self.computeValueFromQValues(nextState).detach() - self.getQValue(state,action)
        # print(f"DIFFERENCE {difference}")
        features = self.featExtractor(state,action)
        features = torch.tensor(list(features.values()))

        self.alpha=0.1
        self.discount=0.9

        difference=difference**2/2
        difference.backward()
        self.optim.step()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)
        print(f"AGENT {self.index} WEIGHT {self.mlp.layer.weight} REWARD {self.total_reward}")
        self.total_reward=0
        diff = [abs(v1-v2) for v1,v2 in zip(self.mlp.parameters(),self.mlp_copy.parameters())]
        print(f"AGENT {self.index} DIFF {torch.cat([d.flatten() for d in diff]).mean()}")
        self.mlp_copy = deepcopy(self.mlp)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
        
class MLPReflexAgent(MLPQAgent):
  def getAction(self, state):
    return self.computeActionFromQValues(state)
  def update(*args, **kwargs):
    pass 
        
class ReplayBuffer:
  def __init__(self, size):
    self.size = size 
    self.buffer = deque(maxlen=size)
    self.rng = np.random.default_rng()
# AGENT 1 WEIGHT Parameter containing:
# tensor([[-0.0570,  0.1253,  0.0296, -0.2305,  0.3851, -0.0480,  0.0940,  0.6522,
#           0.0308,  0.4601,  0.6621]], requires_grad=True) REWARD 0
# AGENT 1 DIFF 0.028332185000181198
# AGENT 3 WEIGHT Parameter containing:
# tensor([[-0.0378,  0.1216,  0.0579, -0.1938,  0.3656, -0.1227,  0.1703,  0.8146,
#           0.0139,  0.7397,  1.0736]], requires_grad=True) REWARD 0

  def append(self, transition):
    self.buffer.append(transition)
  def __len__(self):
     return len(self.buffer)

  def sample(self,size=32):
    size = min(size,len(self.buffer))
    sample = self.rng.choice(len(self.buffer),size,False)
    sample = [self.buffer[i] for i in sample]
    # batched_sample=[0]*len(sample[0])
    return [[s[i] for s in sample] for i in range(len(sample[0]))]
    # for i in range(len(sample[0])):
    #    batched_sample[i]=np.stack([s[i] for s in sample],axis=0)
    # return [torch.tensor(b) for b in batched_sample]
       

class MLPDDQN(MLPQAgent):
    def __init__(self, mlp, index, cnn=False, include_pos = False,**kwargs):
        super().__init__(mlp, index, **kwargs)
        self.target_mlp = deepcopy(mlp)
        self.cnn = cnn
        self.target_update_freq = 1000
        self.target_update_counter = 0
        self.replay_buffer_size = 100000
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size) 
        self.update_freq=4
        self.update_counter=0
        self.loss = nn.HuberLoss(reduction='sum')
        self.discount=0.99
        self.mlp_copy = deepcopy(self.mlp)
        self.include_pos=include_pos
        


    def construct_input(self, gameState:capture.GameState) -> tuple[np.array]:
        image_input = np.expand_dims(self.one_hot(gameState, include_pos=self.include_pos),0)
        #x,y position -> 8 items
        positions = self.extract_positions(gameState)
        #scared?,timer -> 8 items
        scared = self.scared(gameState)
        #food needed -> 4 items
        food = self.percent_food_needed(gameState)
        return image_input, np.expand_dims(np.concatenate((positions, scared, food)),0)
    def getQValue(self, state, action, target=False):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        mlp = self.target_mlp if target else self.mlp
        if self.cnn:
          # print(f"HI")
          succ = self.getSuccessor(state,action)
          features = self.construct_input(succ)
          # print([f.shape for f in features])
          return mlp(features)
        else:
          features = self.featExtractor(state,action)
          features = torch.tensor(list(features.values()))
          return mlp(features)
        
    def computeActionFromQValues(self, state):
      """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
      """
      actions = state.getLegalActions(self.index)
      if self.percent<0.2:
          actions.remove(Directions.STOP)
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
        # print(self.enclosed)
        # self.debugDraw(list(self.positions), (1,1,1))
        # Pick Action
        # print(f"AGENT {self.index} PERCENT {self.percent}")
        legalActions = state.getLegalActions(self.index)
        if self.percent<0.2:
          legalActions.remove(Directions.STOP)

        
        # t = time.time()
        # if util.flipCoin(0.2):
        #    return random.choice(legalActions)
        # else:
        #    return self.computeActionFromQValues(state)
        self.epsilon=0.2
        eps = self.epsilon*np.cos(self.percent*np.pi)
        if (self.train and util.flipCoin(eps)) or self.percent<0.2:
            action= random.choice(legalActions)
        else:
            action= self.computeActionFromQValues(state)
        # print(f"ACTION TIME {time.time()-t}")
        return action
        
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = state.getLegalActions(self.index)
        if len(actions)==0:
            return 0.0
        rew = [self.getQValue(state,a) for a in actions]
        idx = torch.cat(rew).argmax()

        return self.getQValue(state, actions[idx], target=True)

        features = self.featExtractor(state,actions[idx])
        features = torch.tensor(list(features.values()))
        m = self.target_mlp(features)
        # print(f"M: {m}")
        # argmax = random.choice([i for i in range(len(rew)) if rew[i]==m])
        return m

    def update(self, state, action, nextState, reward: float):
        # print(f"UPDATING")
        """
           Should update your weights based on transition
        """
        # features = self.featExtractor(state,action)
        # features = torch.tensor(list(features.values()))
        # print(f"AGENT {self.index} REWARD {reward}")
        self.replay_buffer.append((state,action,nextState,reward))
        self.update_counter+=1
        if self.update_counter>self.update_freq:
           self.optim.zero_grad()
           states, actions, next_states, rewards = self.replay_buffer.sample()
           target = torch.cat([r+self.discount*self.computeValueFromQValues(s).detach() for r,s in zip(rewards,next_states)])
           predicted = torch.cat([self.getQValue(s,a) for s,a in zip(states,actions)])
          #  print(f"TARGET {target} PREDICTED {predicted}")
           loss = self.loss(predicted,target)
           loss.backward()
           self.optim.step()
           self.update_counter=0

        self.target_update_counter+=1
        if self.target_update_counter>=self.target_update_freq:
           self.target_mlp.load_state_dict(self.mlp.state_dict())
           self.target_update_counter=0

        self.total_reward+=reward

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)
        # print(f"AGENT {self.index} WEIGHT {[v for v in self.mlp.parameters()]} REWARD {self.total_reward}")
        print(f"AGENT {self.index} REWARD {self.total_reward}")
        diff = [abs(v1-v2) for v1,v2 in zip(self.mlp.parameters(),self.mlp_copy.parameters())]
        print(f"AGENT {self.index} DIFF {torch.cat([d.flatten() for d in diff]).mean()}")
        self.mlp_copy = deepcopy(self.mlp)
        self.total_reward=0
        # self.mlp = default_net(20,1)
        

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
        

class MultiOutputDQN(MLPQAgent):
    def __init__(self, mlp, index, cnn=False, include_pos = False, **kwargs):
        super().__init__(mlp, index, **kwargs)
        self.target_mlp = deepcopy(mlp)
        self.cnn = cnn
        self.target_update_freq = 20
        self.target_update_counter = 0
        self.replay_buffer_size = 1000
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size) 
        self.update_freq=4
        self.update_counter=0
        self.loss = nn.HuberLoss(reduction='sum')
        self.alpha=0.1
        self.discount=0.99
        self.feature_agent=FeatureQAgent(self.index)
        self.include_pos = include_pos
        self.percent=0

    def registerInitialState(self, gameState: capture.GameState):
      super().registerInitialState(gameState)
      self.feature_agent.registerInitialState(gameState)
      self.feature_agent.training(0)


    def construct_input(self, gameState:capture.GameState) -> tuple[np.array]:
        image_input = np.expand_dims(self.one_hot(gameState, include_pos = self.include_pos),0)
        #x,y position -> 8 items
        positions = self.extract_positions(gameState)
        #scared?,timer -> 8 items
        scared = self.scared(gameState)
        #food needed -> 4 items
        food = self.percent_food_needed(gameState)
        return image_input, np.expand_dims(np.concatenate((positions, scared, food)),0)
    
    def getQValue(self, state, target=False):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        mlp = self.target_mlp if target else self.mlp
        # print(f"HI")
        features = self.construct_input(state)
        # print([f.shape for f in features])
        return mlp(features)

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
        # print(self.enclosed)
        # self.debugDraw(list(self.positions), (1,1,1))
        # Pick Action
        t = time.time()
        # print(f"WIDTH {state.data.layout.width}, HEIGHT {state.data.layout.height}" )
        eps = self.epsilon*np.cos(self.percent*np.pi)
        legalActions = state.getLegalActions(self.index)
        return random.choice(legalActions)
        if len(self.replay_buffer)<self.replay_buffer_size:
           return random.choice(legalActions)
        # if self.train:
        #    return self.feature_agent.getAction(state)
        if self.train and util.flipCoin(eps):
            action= random.choice(legalActions)
        else:
            q = self.getQValue(state).reshape([-1])
            if random.random()<0.01:
              print(f"AGENT {self.index} Q {q}")
            q = [[q[directions.index(a)],a] for a in legalActions]
            q,action = max(q, key=lambda x:x[0])
        # print(f"ACTION TIME {time.time()-t}")
        return action
        
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = state.getLegalActions(self.index)
        if len(actions)==0:
            return 0.0
        
        idx = self.getQValue(state).argmax()

        return self.getQValue(state,target=True)[0,idx]

        features = self.featExtractor(state,actions[idx])
        features = torch.tensor(list(features.values()))
        m = self.target_mlp(features)
        # print(f"M: {m}")
        # argmax = random.choice([i for i in range(len(rew)) if rew[i]==m])
        return m

    def update(self, state, action, nextState, reward: float):
        # print(f"UPDATING")
        """
           Should update your weights based on transition
        """
        # features = self.featExtractor(state,action)
        # features = torch.tensor(list(features.values()))
        if self.train:
           self.feature_agent.update(state,action,nextState,reward)
        self.replay_buffer.append((state,action,nextState,reward))
        self.update_counter+=1
        if self.update_counter>self.update_freq:
           self.optim.zero_grad()
           states, actions, next_states, rewards = self.replay_buffer.sample()
           target = torch.cat([r+self.discount*self.computeValueFromQValues(s).reshape([-1]) for r,s in zip(rewards,next_states)])
           predicted = torch.cat([self.getQValue(s)[0,directions.index(a)].reshape([-1]) for s,a in zip(states,actions)])
          #  print(f"TARGET {target} PREDICTED {predicted}")
           loss = self.loss(predicted,target)
           loss.backward()
           self.optim.step()
           self.update_counter=0

        self.target_update_counter+=1
        if self.target_update_counter>=self.target_update_freq:
           self.target_mlp.load_state_dict(self.mlp.state_dict())
           self.target_update_counter=0

        self.total_reward+=reward

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)
        # print(f"AGENT {self.index} WEIGHT {[v for v in self.mlp.parameters()]} REWARD {self.total_reward}")
        print(f"AGENT {self.index} REWARD {self.total_reward}")
        self.total_reward=0
        # self.mlp = default_net(20,1)
        

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

        
# import os
# maxH = 0
# maxW = 0
# for filename in os.listdir("layouts"):
  
#   with open("layouts/"+filename, "r") as f:
#     f = f.read()
#     f = f.split("\n")
#     print(f"NAME {filename} HEIGHT {len(f)} WIDTH {len(f[0])}")
#     maxH = max(maxH, len(f))
#     maxW = max(maxW, len(f[0]))
# print(f"MAX HEIGHT {maxH} WIDTH {maxW}")


