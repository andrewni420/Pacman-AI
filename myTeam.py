# myTeam.py
# ---------
# jizz
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

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
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
    self.registerTeam(self.getTeam())

  def chooseAction(self, gameState: capture.GameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)
  
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



