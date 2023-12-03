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


DEFAULT_DISTANCE=float('inf')

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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

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

    # print(f"AM RED? {self.red}")
    # for b in self.boundary:
    #   print(f"-1 {gameState.isRed((b[0]-1,b[1]))} 0 {gameState.isRed((b[0],b[1]))} 1 {gameState.isRed((b[0]+1,b[1]))}")

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
    return (default if minDistance==float('inf') else minDistance), (myPos if minPos is None else minPos)
  
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
  
  def defenders(self, gameState:capture.GameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a.getPosition() for a in enemies if (not a.isPacman) and a.getPosition() != None and a.scaredTimer==0]
    return defenders
  
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
  
  def invaderDistance(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    invaders = self.invaders(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
    return self.nearest(myPos, invaders)
    
  def defenderDistance(self, gameState:capture.GameState) -> tuple[int, tuple[int, int]]:
    defenders = self.defenders(gameState)
    myPos = gameState.getAgentState(self.index).getPosition()
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
  
  def get_boundary(self, gameState, red):
    width = gameState.data.layout.width
    height = gameState.data.layout.height
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

  def one_hot(self, gameState:capture.GameState) -> list[list[float]]:
    #Should cache some of these if there's runtime issues
    red_food = np.array(gameState.getRedFood().data, dtype=np.float32)
    red_capsules = np.zeros_like(red_food)
    for x,y in gameState.getRedCapsules():
      red_capsules[x][y]=1
    blue_food = np.array(gameState.getBlueFood().data, dtype=np.float32)
    blue_capsules = np.zeros_like(blue_food)
    for x,y in gameState.getBlueCapsules():
      blue_capsules[x][y]=1
    walls = np.array(gameState.getWalls().data, dtype=np.float32)
    width, height = gameState.data.layout.width, gameState.data.layout.height
    red = np.array([[int(w<width/2)]*height for w in range(width)])
    blue = np.array([[int(w>=width/2)]*height for w in range(width)])
    if self.red:
      return np.stack([red_food, red_capsules, red, blue_food, blue_capsules, blue, walls], axis=0)
    return np.stack([blue_food, blue_capsules, blue, red_food, red_capsules, red, walls], axis=0)

  

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
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    if action is not None:
      successor = self.getSuccessor(gameState, action)
    else:
      successor = gameState
    features["num_invaders"] = self.num_invaders(successor)
    features['invader_distance'] = 1/self.invaderDistance(successor)[0]
    features['defender_distance'] = 1/self.defenderDistance(successor)[0]
    features['scared_distance'] = 1/self.scaredDistance(successor)[0]
    features['num_food'] = self.num_food(successor)
    features['nearest_food'] = 1/self.nearest_food(successor)[0]
    features['num_capsules'] = self.num_capsules(successor)
    features['nearest_capsule'] = 1/self.nearest_capsule(successor)[0]
    features['pacman_ghost'] = int(self.pacmanGhost(successor))
    features['num_carried']= self.num_carried(successor)
    features['distance_to_home']=1/self.distance_to_home(successor)[0]
    features['carried*distance']=self.num_carried(successor)/self.distance_to_home(successor)[0]
    features['num_returned']=self.num_returned(successor)

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
        legalActions = state.getLegalActions(self.index)
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """

        difference = reward+self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state,action)
        features = self.featExtractor(state,action)

        for f in features.keys():
            self.weights[f]+=self.alpha*difference*features[f]


    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass