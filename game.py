# game.py
# -------
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


# game.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import *
import time, os
import traceback
import sys
from environment import Environment
import numpy as np 
from collections import deque

#######################
# Parts worth reading #
#######################

class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()

class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'

    LEFT =       {NORTH: WEST,
                   SOUTH: EAST,
                   EAST:  NORTH,
                   WEST:  SOUTH,
                   STOP:  STOP}

    RIGHT =      dict([(y,x) for x, y in list(LEFT.items())])

    REVERSE = {NORTH: SOUTH,
               SOUTH: NORTH,
               EAST: WEST,
               WEST: EAST,
               STOP: STOP}

class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def getPosition(self):
        return (self.pos)

    def getDirection(self):
        return self.direction

    def isInteger(self):
        x,y = self.pos
        return x == int(x) and y == int(y)

    def __eq__(self, other):
        if other == None: return False
        return (self.pos == other.pos and self.direction == other.direction)

    def __hash__(self):
        x = hash(self.pos)
        y = hash(self.direction)
        return hash(x + 13 * y)

    def __str__(self):
        return "(x,y)="+str(self.pos)+", "+str(self.direction)

    def generateSuccessor(self, vector):
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        x, y= self.pos
        dx, dy = vector
        direction = Actions.vectorToDirection(vector)
        if direction == Directions.STOP:
            direction = self.direction # There is no stop direction
        return Configuration((x + dx, y+dy), direction)

class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, scared, etc).
    """

    def __init__( self, startConfiguration, isPacman ):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.isPacman = isPacman
        self.scaredTimer = 0
        self.numCarrying = 0
        self.numReturned = 0

    def __str__( self ):
        if self.isPacman:
            return "Pacman: " + str( self.configuration )
        else:
            return "Ghost: " + str( self.configuration )

    def __eq__( self, other ):
        if other == None:
            return False
        return self.configuration == other.configuration and self.scaredTimer == other.scaredTimer

    def __hash__(self):
        return hash(hash(self.configuration) + 13 * hash(self.scaredTimer))

    def copy( self ):
        state = AgentState( self.start, self.isPacman )
        state.configuration = self.configuration
        state.scaredTimer = self.scaredTimer
        state.numCarrying = self.numCarrying
        state.numReturned = self.numReturned
        return state

    def getPosition(self):
        if self.configuration == None: return None
        return self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection()

class Grid:
    """
    A 2-dimensional array of objects backed by a list of lists.  Data is accessed
    via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal,
    y vertical and the origin (0,0) in the bottom left corner.

    The __str__ method constructs an output that is oriented like a pacman board.
    """
    def __init__(self, width, height, initialValue=False, bitRepresentation=None):
        if initialValue not in [False, True]: raise Exception('Grids can only contain booleans')
        self.CELLS_PER_INT = 30

        self.width = width
        self.height = height
        self.data = [[initialValue for y in range(height)] for x in range(width)]
        if bitRepresentation:
            self._unpackBits(bitRepresentation)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, key, item):
        self.data[key] = item

    def __str__(self):
        out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
        out.reverse()
        return '\n'.join([''.join(x) for x in out])

    def __eq__(self, other):
        if other == None: return False
        return self.data == other.data

    def __hash__(self):
        # return hash(str(self))
        base = 1
        h = 0
        for l in self.data:
            for i in l:
                if i:
                    h += base
                base *= 2
        return hash(h)

    def copy(self):
        g = Grid(self.width, self.height)
        g.data = [x[:] for x in self.data]
        return g

    def deepCopy(self):
        return self.copy()

    def shallowCopy(self):
        g = Grid(self.width, self.height)
        g.data = self.data
        return g

    def count(self, item =True ):
        return sum([x.count(item) for x in self.data])

    def asList(self, key = True):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self[x][y] == key: list.append( (x,y) )
        return list

    def packBits(self):
        """
        Returns an efficient int list representation

        (width, height, bitPackedInts...)
        """
        bits = [self.width, self.height]
        currentInt = 0
        for i in range(self.height * self.width):
            bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
            x, y = self._cellIndexToPosition(i)
            if self[x][y]:
                currentInt += 2 ** bit
            if (i + 1) % self.CELLS_PER_INT == 0:
                bits.append(currentInt)
                currentInt = 0
        bits.append(currentInt)
        return tuple(bits)

    def _cellIndexToPosition(self, index):
        x = index / self.height
        y = index % self.height
        return x, y

    def _unpackBits(self, bits):
        """
        Fills in data from a bit-level representation
        """
        cell = 0
        for packed in bits:
            for bit in self._unpackInt(packed, self.CELLS_PER_INT):
                if cell == self.width * self.height: break
                x, y = self._cellIndexToPosition(cell)
                self[x][y] = bit
                cell += 1

    def _unpackInt(self, packed, size):
        bools = []
        if packed < 0: raise ValueError("must be a positive integer")
        for i in range(size):
            n = 2 ** (self.CELLS_PER_INT - i - 1)
            if packed >= n:
                bools.append(True)
                packed -= n
            else:
                bools.append(False)
        return bools

def reconstituteGrid(bitRep):
    if type(bitRep) is not type((1,2)):
        return bitRep
    width, height = bitRep[:2]
    return Grid(width, height, bitRepresentation= bitRep[2:])

####################################
# Parts you shouldn't have to read #
####################################

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions = {Directions.NORTH: (0, 1),
                   Directions.SOUTH: (0, -1),
                   Directions.EAST:  (1, 0),
                   Directions.WEST:  (-1, 0),
                   Directions.STOP:  (0, 0)}

    _directionsAsList = list(_directions.items())

    TOLERANCE = .001

    def reverseDirection(action):
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action
    reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            return Directions.NORTH
        if dy < 0:
            return Directions.SOUTH
        if dx < 0:
            return Directions.WEST
        if dx > 0:
            return Directions.EAST
        return Directions.STOP
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed = 1.0):
        dx, dy =  Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config, walls):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if not walls[next_x][next_y]: possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    def getLegalNeighbors(position, walls):
        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height: continue
            if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)

class GameStateData:
    """

    """
    def __init__( self, prevState = None ):
        """
        Generates a new data packet by copying information from its predecessor.
        """
        if prevState != None:
            self.food = prevState.food.shallowCopy()
            self.capsules = prevState.capsules[:]
            self.agentStates = self.copyAgentStates( prevState.agentStates )
            self.layout = prevState.layout
            self._eaten = prevState._eaten
            self.score = prevState.score

        self._foodEaten = None
        self._foodAdded = None
        self._capsuleEaten = None
        self._agentMoved = None
        self._lose = False
        self._win = False
        self.scoreChange = 0

    def deepCopy( self ):
        state = GameStateData( self )
        state.food = self.food.deepCopy()
        state.layout = self.layout.deepCopy()
        state._agentMoved = self._agentMoved
        state._foodEaten = self._foodEaten
        state._foodAdded = self._foodAdded
        state._capsuleEaten = self._capsuleEaten
        return state

    def copyAgentStates( self, agentStates ):
        copiedStates = []
        for agentState in agentStates:
            copiedStates.append( agentState.copy() )
        return copiedStates

    def __eq__( self, other ):
        """
        Allows two states to be compared.
        """
        if other == None: return False
        # TODO Check for type of other
        if not self.agentStates == other.agentStates: return False
        if not self.food == other.food: return False
        if not self.capsules == other.capsules: return False
        if not self.score == other.score: return False
        return True

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.
        """
        for i, state in enumerate( self.agentStates ):
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                #hash(state)
        return int((hash(tuple(self.agentStates)) + 13*hash(self.food) + 113* hash(tuple(self.capsules)) + 7 * hash(self.score)) % 1048575 )

    def __str__( self ):
        width, height = self.layout.width, self.layout.height
        map = Grid(width, height)
        if type(self.food) == type((1,2)):
            self.food = reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agentState in self.agentStates:
            if agentState == None: continue
            if agentState.configuration == None: continue
            x,y = [int( i ) for i in nearestPoint( agentState.configuration.pos )]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr( agent_dir )
            else:
                map[x][y] = self._ghostStr( agent_dir )

        for x, y in self.capsules:
            map[x][y] = 'o'

        return str(map) + ("\nScore: %d\n" % self.score)

    def _foodWallStr( self, hasFood, hasWall ):
        if hasFood:
            return '.'
        elif hasWall:
            return '%'
        else:
            return ' '

    def _pacStr( self, dir ):
        if dir == Directions.NORTH:
            return 'v'
        if dir == Directions.SOUTH:
            return '^'
        if dir == Directions.WEST:
            return '>'
        return '<'

    def _ghostStr( self, dir ):
        return 'G'
        if dir == Directions.NORTH:
            return 'M'
        if dir == Directions.SOUTH:
            return 'W'
        if dir == Directions.WEST:
            return '3'
        return 'E'

    def initialize( self, layout, numGhostAgents ):
        """
        Creates an initial game state from a layout array (see layout.py).
        """
        self.food = layout.food.copy()
        #self.capsules = []
        self.capsules = layout.capsules[:]
        self.layout = layout
        self.score = 0
        self.scoreChange = 0

        self.agentStates = []
        numGhosts = 0
        for isPacman, pos in layout.agentPositions:
            if not isPacman:
                if numGhosts == numGhostAgents: continue # Max ghosts reached already
                else: numGhosts += 1
            self.agentStates.append( AgentState( Configuration( pos, Directions.STOP), isPacman) )
        self._eaten = [False for a in self.agentStates]

try:
    import boinc
    _BOINC_ENABLED = True
except:
    _BOINC_ENABLED = False

class Game:
    """
    The Game manages the control flow, soliciting actions from agents.
    """

    def __init__( self, agents, display, rules, startingIndex=0, muteAgents=False, catchExceptions=False ):
        self.agentCrashed = False
        self.agents = agents
        self.display = display
        self.rules = rules
        self.startingIndex = startingIndex
        self.gameOver = False
        self.muteAgents = muteAgents
        self.catchExceptions = catchExceptions
        self.moveHistory = []
        self.totalAgentTimes = [0 for agent in agents]
        self.totalAgentTimeWarnings = [0 for agent in agents]
        self.agentTimeout = False
        import io
        self.agentOutput = [io.StringIO() for agent in agents]

    def getProgress(self):
        if self.gameOver:
            return 1.0
        else:
            return self.rules.getProgress(self)

    def _agentCrash( self, agentIndex, quiet=False):
        "Helper method for handling agent crashes"
        if not quiet: traceback.print_exc()
        self.gameOver = True
        self.agentCrashed = True
        self.rules.agentCrash(self, agentIndex)

    OLD_STDOUT = None
    OLD_STDERR = None

    def mute(self, agentIndex):
        if not self.muteAgents: return
        global OLD_STDOUT, OLD_STDERR
        import io
        OLD_STDOUT = sys.stdout
        OLD_STDERR = sys.stderr
        sys.stdout = self.agentOutput[agentIndex]
        sys.stderr = self.agentOutput[agentIndex]

    def unmute(self):
        if not self.muteAgents: return
        global OLD_STDOUT, OLD_STDERR
        # Revert stdout/stderr to originals
        sys.stdout = OLD_STDOUT
        sys.stderr = OLD_STDERR


    def run( self , percent=0):
    
        """
        Main control loop for game play.
        """
        self.display.initialize(self.state.data)
        self.numMoves = 0
        num_carried = [0]*len(self.agents)
        prev_states = [self.state.deepCopy() for _ in self.agents]
        prev_actions = [None for _ in self.agents]
        prev_rew = [0]*len(self.agents)
        delayed_rew = [0]*len(self.agents)
        changed_potential = [0]*len(self.agents)
        delta_rew = [0]*len(self.agents)
        
#state0 _agent0_ state1 _agent1_ state2 _agent2_ state3 _agent3_ state4 _agent0_
#state4 - state3 + state2 - state0
        state_deque = deque([(self.state.deepCopy(), None, None) for _ in range(10)], maxlen=10)
        action_deque = deque([None for _ in range(10)], maxlen=10)
        for a in self.agents:
            a.percent=percent

        ###self.display.initialize(self.state.makeObservation(1).data)
        # inform learning agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                self.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.unmute()
                self._agentCrash(i, quiet=True)
                return
            if ("registerInitialState" in dir(agent)):
                self.mute(i)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deepCopy())
                            time_taken = time.time() - start_time
                            self.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" % i, file=sys.stderr)
                            self.unmute()
                            self.agentTimeout = True
                            self._agentCrash(i, quiet=True)
                            return
                    except Exception as data:
                        self._agentCrash(i, quiet=False)
                        self.unmute()
                        return
                else:
                    agent.registerInitialState(self.state.deepCopy())
                ## TODO: could this exceed the total time
                self.unmute()

        agentIndex = self.startingIndex
        numAgents = len( self.agents )

        while not self.gameOver:
            # Fetch the next agent
            agent = self.agents[agentIndex]
            move_time = 0
            skip_action = False
            # Generate an observation of the state
            if 'observationFunction' in dir( agent ):
                self.mute(agentIndex)
                if self.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
                        try:
                            start_time = time.time()
                            observation = timed_func(self.state.deepCopy())
                        except TimeoutFunctionException:
                            skip_action = True
                        move_time += time.time() - start_time
                        self.unmute()
                    except Exception as data:
                        self._agentCrash(agentIndex, quiet=False)
                        self.unmute()
                        return
                else:
                    observation = agent.observationFunction(self.state.deepCopy())
                self.unmute()
            else:
                observation = self.state.deepCopy()

            # rew1 = self.reward1(agentIndex, prev_states[agentIndex])
            if "num_carried" in dir(agent):
                # print(f"CUR INDEX {agentIndex} PREV INDICES {[i[2] for i in list(state_deque)[::-1]]}")
                rtotal = self.calculate_reward(agentIndex, self.state,percent)-self.calculate_reward(agentIndex, state_deque[-4][0],percent)
                rother = (self.calculate_reward(agentIndex, state_deque[-1][0],percent)-self.calculate_reward(agentIndex, state_deque[-2][0],percent))
                rew1 =  rtotal-rother
                # rew1 = self.calculate_reward(agentIndex, self.state, percent)-self.calculate_reward(agentIndex, prev_states[agentIndex], percent)
                # rew1-=delta_rew[agentIndex]
                # print(f"AGENT {agentIndex} REW {rew1} RTOTAL {rtotal} ROTHER {rother}")
            else:
                rew1=0
            # rew = prev_rew[agentIndex]
            # if abs(rew-rew1)/max(abs(rew),0.001)>0.001:
            #     print(f"REWARD DIFF PREV {prev_rew} \
            #           INDEX {agentIndex} REW {rew1} \
            #             FOOD {agent.num_carried(self.state)-agent.num_carried(prev_state)}\
            #                POTENTIAL {self.food_potential(self.state, agentIndex)-self.food_potential(prev_states[agentIndex], agentIndex)} ")

            # if hasattr(agent, "update") and not agent.freeze and state_deque[-4][1] is not None:
            #     agent.update(state_deque[-4][0], state_deque[-4][1], self.state, rew1)
            # Solicit an action
            action = None
            self.mute(agentIndex)
            if self.catchExceptions:
                try:
                    timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
                    try:
                        start_time = time.time()
                        if skip_action:
                            raise TimeoutFunctionException()
                        action = timed_func( observation )
                    except TimeoutFunctionException:
                        print("Agent %d timed out on a single move!" % agentIndex, file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return

                    move_time += time.time() - start_time

                    if move_time > self.rules.getMoveWarningTime(agentIndex):
                        self.totalAgentTimeWarnings[agentIndex] += 1
                        print("Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                        if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
                            print("Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
                            self.agentTimeout = True
                            self._agentCrash(agentIndex, quiet=True)
                            self.unmute()
                            return

                    self.totalAgentTimes[agentIndex] += move_time
                    #print "Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex])
                    if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
                        print("Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex]), file=sys.stderr)
                        self.agentTimeout = True
                        self._agentCrash(agentIndex, quiet=True)
                        self.unmute()
                        return
                    self.unmute()
                except Exception as data:
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                action = agent.getAction(observation)
            self.unmute()

            

            # Execute the action
            self.moveHistory.append( (agentIndex, action) )
            if self.catchExceptions:
                try:
                    self.state = self.state.generateSuccessor( agentIndex, action )
                except Exception as data:
                    self.mute(agentIndex)
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
            else:
                #PROBABLY STORE PREVIOUS STATE HERE
                prev_state = self.state.deepCopy()
                prev_states[agentIndex]=self.state.deepCopy()
                prev_actions[agentIndex]=action
                self.state = self.state.generateSuccessor( agentIndex, action )
                if hasattr(agent, "num_carried"):
                    rew=self.calculate_reward(agentIndex, self.state,percent)-self.calculate_reward(agentIndex, prev_state,percent)
                    pos = [state.getAgentPosition(agentIndex) for state,_,_ in list(state_deque)[::-1]]
                    rew2 = self.calculate_reward(agentIndex, state_deque[-1][0],percent)-self.calculate_reward(agentIndex, state_deque[-5][0],percent)
                    # print(f"AGENT {agentIndex} REW {rew} REW2 {rew2} ACTION {action} STATES {pos}")
                state_deque.append((prev_state, action, agentIndex))
                if hasattr(agent, "num_carried"):
                    other_index = (agentIndex+2)%4
                    rself = self.calculate_reward(other_index, self.state,percent)-self.calculate_reward(other_index, state_deque[-1][0],percent)
                    # print(f"AGENT {agentIndex} RSELF {rself}")


            ## PROBABLY CALL AGENT.UPDATE HERE
            # rew = self.state.getScore()-prev_state.getScore() or something. 
            # rew = (self.state.getScore()-prev_state.getScore())*(1 if agent.red else -1)
            delayed_rew[agentIndex]=self.food_potential(self.state, agentIndex)-self.food_potential(prev_state, agentIndex)
            other_index = (agentIndex+2)%4
            changed_potential[other_index]=self.food_potential(self.state, other_index)-self.food_potential(prev_state, other_index)


            # delta_rew[other_index]=self.reward1(other_index,prev_state)
            

            rew = self.reward1(agentIndex, prev_state)
            if "num_carried" in dir(agent):
                delta_rew[other_index]=self.calculate_reward(other_index,self.state,percent)-self.calculate_reward(other_index,prev_state,percent)
                # dr = self.calculate_reward(agentIndex, self.state, 1)-self.calculate_reward(agentIndex, prev_state, 1)
                # print(f"REWARD {rew} DR {dr/10} PERCENT DIFF {abs(rew-dr/10)/max(abs(rew),0.0001)}")
            # rew+=delayed_rew[agentIndex]
            prev_rew[agentIndex]=rew

            # print(f"NEXT REW {agentIndex} {rew}")
            
                                                            
            if hasattr(agent, "update"):
                agent.update(prev_state, action, self.state, rew)

            # Change the display
            self.display.update( self.state.data )
            ###idx = agentIndex - agentIndex % 2 + 1
            ###self.display.update( self.state.makeObservation(idx).data )

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            # Track progress
            if agentIndex == numAgents + 1: self.numMoves += 1
            # Next agent
            agentIndex = ( agentIndex + 1 ) % numAgents

            if _BOINC_ENABLED:
                boinc.set_fraction_done(self.getProgress())

        for i in range(4):
            agent = self.agents[agentIndex]
            if "num_carried" in dir(agent):
                rtotal = self.calculate_reward(agentIndex, self.state,percent)-self.calculate_reward(agentIndex, state_deque[-4][0],percent)
                rother = (self.calculate_reward(agentIndex, state_deque[-1][0],percent)-self.calculate_reward(agentIndex, state_deque[-2][0],percent))
                rew1 =  rtotal-rother
            else:
                rew1=0
            if hasattr(agent, "update") and state_deque[-4][1] is not None:
                agent.update(state_deque[-4][0], state_deque[-4][1], self.state, rew1)
            state_deque.append((self.state.deepCopy(), None, agentIndex))
            agentIndex = (agentIndex+1)%4
            

        # inform a learning agent of the game result
        for agentIndex, agent in enumerate(self.agents):
            
            if "final" in dir( agent ) :
                try:
                    self.mute(agentIndex)
                    agent.final( self.state )
                    self.unmute()
                    # if hasattr(agent,"mlp"):
                    #     print(f"AGENT {agent.index} \nWEIGHT {agent.mlp.weight} \nCOPY {agent.mlp_copy} \nDIFF {agent.mlp.weight-agent.mlp_copy}")
                        
                except Exception as data:
                    if not self.catchExceptions: raise
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
        self.display.finish()

    def food_potential(self, state, agentIndex, inverse=True):
        red = self.agents[agentIndex].red 
        food = state.getBlueFood() if red else state.getRedFood()
        food = food.asList()
        pos = state.getAgentPosition(agentIndex)
        dists = [self.agents[agentIndex].getMazeDistance(pos,f) for f in food]
        total = sum([(1/d if inverse else d) for d in dists])
        return total
    
    def capsule_potential(self, state, agentIndex, inverse=True):
        red = self.agents[agentIndex].red 
        capsules = state.getBlueCapsules() if red else state.getRedCapsules()
        pos = state.getAgentPosition(agentIndex)
        dists = [self.agents[agentIndex].getMazeDistance(pos,c) for c in capsules]
        total = sum([(1/d if inverse else d) for d in dists])
        return total
    
    def calculate_reward(self, agentIndex, state, percent):
        features = self.reward_features(agentIndex, state)
        # print(f"AGENT {agentIndex} REWARD FEATURES {features}")
        weights = [self.reward1_weights(), self.reward2_weights(), self.reward3_weights()]
        modified = [1-percent+0.01, abs(percent-0.5)+0.01, percent+0.01]
        for w,m in zip(weights,modified):
            w.divideAll(1/m)
        reward = (weights[0]+weights[1]+weights[2])*features
        # if features["opp_returned"]>0:
        #     print(f"_____\n_____\n_____\nRETURNED REWARD")
        #     print(f"AGENT {agentIndex} REWARD FEATURES {features} REWARD {reward}")
        # print(f"AGENT {agentIndex} REWARD FEATURES {features} REWARD {reward}")
        return reward
    
    def reward_features(self, agentIndex, state):
        agent = self.agents[agentIndex]
        opponents = agent.getOpponents(state)
        rew = Counter()
        rew["num_carried"] = agent.num_carried(state)
        rew["opp_carried"] = sum([state.getAgentState(i).numCarrying for i in opponents])/2

        rew["num_capsules"]=len(agent.getCapsules(state))
        rew["opp_capsules"]=sum([len(self.agents[i].getCapsules(state)) for i in opponents])/2

        rew["num_returned"]=agent.num_returned(state)
        rew["opp_returned"] = sum([state.getAgentState(i).numReturned for i in opponents])/2

        rew["food_potential"]=self.food_potential(state,agentIndex)
        rew["capsule_potential"]=self.capsule_potential(state,agentIndex)
        rew["opp_food_potential"]=sum([self.food_potential(state,i) for i in opponents])/2
        rew["opp_capsule_potential"]=sum([self.capsule_potential(state,i) for i in opponents])/2

        rew["carried_distance"]=agent.num_carried(state)/agent.distance_to_home(state)[0]
        rew["opp_carried_distance"]=sum([state.getAgentState(i).numCarrying/agent.opp_distance_to_home(state,i)[0] for i in opponents])/2

        rew["invader_distance"]=1/agent.invaderDistance(state)[0]
        rew["opp_invader_distance"]=sum([1/agent.oppInvaderDistance(state,i)[0] for i in opponents])/2
        rew["num_invaders"]=agent.num_invaders(state)
        rew["opp_invaders"]=len(agent.opp_invaders(state))/2

        defender_distance = agent.defenderDistance(state)[0]
        opp_defender_distance = [agent.oppDefenderDistance(state,i)[0] for i in opponents]
        rew["defender_distance"]=1/defender_distance if defender_distance<5 else 0
        rew["opp_defender_distance"]=sum([(1/d if d<5 else 0) for d in opp_defender_distance])/2

        rew["enclosed"] = state.getAgentPosition(agentIndex) in agent.enclosed and defender_distance<3
        rew["opp_enclosed"] = sum([(state.getAgentPosition(i) in agent.opp_enclosed and opp_defender_distance[opponents.index(i)]<3) for i in opponents])/2

        rew["total_food_dist"] = self.food_potential(state, agentIndex, inverse=False)
        rew["total_capsule_dist"] = self.capsule_potential(state, agentIndex, inverse=False)
        rew["total_defender_distance"]=agent.total_defender_distance(state,inverse=False)
        rew["total_invader_distance"]=agent.total_invader_distance(state,inverse=False)
        
        return rew

    def reward1_weights(self):
        c = Counter()
        c = c+{"num_carried":4, "num_capsules":-50, "food_potential":1, "capsule_potential":2, "carried_distance":5, 
                "num_returned":20, "invader_distance":2, "num_invaders":-4, "defender_distance":-5, "enclosed":-1}
        c=c-{"opp_carried":4, "opp_capsules":-50, "opp_food_potential":1, "opp_capsule_potential":2, "opp_carried_distance":5, 
                "opp_returned":20, "opp_invader_distance":2, "opp_invaders":-4, "opp_defender_distance":-5, "opp_enclosed":-1}
        c.divideAll(len(c)/2)

        return c
        # return Counter()+{"total_food_dist":-1/600, "num_carried":4}

        # AGENT 3 REWARD FEATURES {'num_carried': 0,  'opp_carried': 0.0, 
        #                          'num_capsules': 1, 'opp_capsules': 0.0, 
        #                          'num_returned': 0, 'opp_returned': 9.0, 
        #                          'food_potential': 0.39765092477947633, 'opp_food_potential': 0.07185730935730936,
        #                          'capsule_potential': 0.02040816326530612, 'opp_capsule_potential': 0.0,
        #                         'carried_distance': 0.0, 'opp_carried_distance': 0.0, 
        #                         'invader_distance': 0.0, 'opp_invader_distance': 0.0, 
        #                         'num_invaders': 0, 'opp_invaders': 0.0, 
        #                         'defender_distance': 0, 'opp_defender_distance': 0.0, 
        #                         'enclosed': False, 'opp_enclosed': 0.0, 
        #                         'total_food_dist': 1016, 'total_capsule_dist': 49, 
        #                         'total_defender_distance': 98, 'total_invader_distance': 0} 
        # REWARD -0.22963339005804723
    
    def reward2_weights(self):
        c = Counter()
        c=c+{"num_carried":4, "num_capsules":50,"num_returned":20, "num_invaders":4, "enclosed":1}
        c.divideAll(len(c))
        return self.reward1_weights()
    
    def reward3_weights(self):
        c = Counter()
        c=c+{"num_returned":20}
        c.divideAll(len(c))
        return self.reward1_weights()

    
    def reward1(self, agentIndex, prev_state, verbose=False):
        rew=0
        agent = self.agents[agentIndex]
        if "num_carried" in dir(agent):
            # rew+=3*(len(agent.getFood(prev_state).asList())-len(agent.getFood(self.state).asList()))
            rew+=4*(agent.num_carried(self.state)-agent.num_carried(prev_state))
            rew+=50*(len(agent.getCapsules(prev_state))-len(agent.getCapsules(self.state)))
            rew += self.food_potential(self.state, agentIndex)-self.food_potential(prev_state, agentIndex)
            rew += 2*(self.capsule_potential(self.state, agentIndex)-self.capsule_potential(prev_state, agentIndex))
            rew+=5*(agent.num_carried(self.state)/agent.distance_to_home(self.state)[0]-agent.num_carried(prev_state)/agent.distance_to_home(prev_state)[0])
            rew+=20*(agent.num_returned(self.state)-agent.num_returned(prev_state))
            rew+=2*(1/agent.invaderDistance(self.state)[0]-1/agent.invaderDistance(prev_state)[0])
            rew+=4*(agent.num_invaders(prev_state)-agent.num_invaders(self.state))
            prev_def_dist = agent.defenderDistance(prev_state)[0]
            prev_def_dist = float("inf") if prev_def_dist>5 else prev_def_dist
            cur_def_dist = agent.defenderDistance(self.state)[0]
            cur_def_dist = float("inf") if cur_def_dist>5 else cur_def_dist
            rew+=5*((1)/prev_def_dist-(1)/cur_def_dist)
            rew+=(int(prev_state.getAgentPosition(agentIndex) in agent.enclosed and agent.defenderDistance(prev_state)[0]<3) - int(self.state.getAgentPosition(agentIndex) in agent.enclosed and agent.defenderDistance(prev_state)[0]<3))
            # cur_tdist = agent.distance_to_teammate(self.state)[0]
            # prev_tdist = agent.distance_to_teammate(prev_state)[0]
            # cur_tdist = 1/cur_tdist if cur_tdist<5 else 0
            # prev_tdist = 1/prev_tdist if prev_tdist<5 else 0
            # rew+=0.01*(prev_tdist-cur_tdist)
            rew=rew/10

            # if len(agent.getFood(prev_state).asList())-len(agent.getFood(self.state).asList())!=0:
            #     print(f"food reward: {4*(agent.num_carried(self.state)-agent.num_carried(prev_state))}")
            # if len(agent.getCapsules(prev_state))-len(agent.getCapsules(self.state))!=0:
            #     print(f"capsule reward: {rew}")
            # if agent.num_returned(self.state)-agent.num_returned(prev_state)!=0:
            #     print(f"returned reward: {rew}")
            # if verbose:
            #     print(f"REWARD {agentIndex} \
            #           FOOD {agent.num_carried(self.state)-agent.num_carried(prev_state)}\
            #           TOTAL {rew}")
        return rew

class TrackGame(Game):
    def __init__( self, agents, display, rules, startingIndex=0, muteAgents=False, catchExceptions=False):
        super(TrackGame, self).__init__(agents, display, rules, startingIndex=startingIndex, muteAgents=muteAgents, catchExceptions=catchExceptions)
        # self.agentCrashed = False
        # self.agents = agents
        # self.display = display
        # self.rules = rules
        # self.startingIndex = startingIndex
        # self.gameOver = False
        # self.muteAgents = muteAgents
        # self.catchExceptions = catchExceptions
        # self.moveHistory = []
        # self.totalAgentTimes = [0 for agent in agents]
        # self.totalAgentTimeWarnings = [0 for agent in agents]
        # self.agentTimeout = False
        # import io
        # self.agentOutput = [io.StringIO() for agent in agents]

    def step(self, action):
        numAgents=len(self.agents) 
        prev_state = self.state 
        self.state = self.state.generateSuccessor( self.current_agent_index, action )
        self.display.update( self.state.data )
        self.rules.process(self.state, self)
        red = self.current_agent_index in self.state.getRedTeamIndices()
        reward = (self.state.getScore()-prev_state.getScore())*(-1 if red else 1)

        if self.current_agent_index == numAgents + 1: self.numMoves += 1
        # Next agent
        
        self.current_agent_index = ( self.current_agent_index + 1 ) % numAgents
        
        return self.state.deepCopy(), reward, self.state.isOver(), None


    def reset(self):
        self.display.initialize(self.state.data)
        self.numMoves = 0

        # inform learning agents of the game start
        for i in range(len(self.agents)):
            self.agents[i].registerInitialState(self.state.deepCopy())
            self.unmute() 

        self.current_agent_index=self.startingIndex
        return self.state.deepCopy(), None
    
    def food_potential(self, state, agentIndex, inverse=True):
        red = self.agents[agentIndex].red 
        food = state.getBlueFood() if red else state.getRedFood()
        food = food.asList()
        pos = state.getAgentPosition(agentIndex)
        total = sum([1/self.agents[agentIndex].getMazeDistance(pos,f) if inverse else self.agents[agentIndex].getMazeDistance(pos,f) for f in food])
        return total

    def potential_reward(self, prev_state, state, agentIndex):
        return self.food_potential(state,agentIndex)-self.food_potential(prev_state,agentIndex) 

    def run( self ):
        ep_obs = [[] for _ in range(len(self.agents))]
        ep_acts = [[] for _ in range(len(self.agents))]
        ep_log_probs = [[] for _ in range(len(self.agents))]
        ep_rews = [[] for _ in range(len(self.agents))]
        ep_lens = [[0] for _ in range(len(self.agents))]

        """
        Main control loop for game play.
        """
        self.display.initialize(self.state.data)
        self.numMoves = 0

        # inform learning agents of the game start
        for i in range(len(self.agents)):
            self.agents[i].registerInitialState(self.state.deepCopy())
            self.unmute()

        agentIndex = self.startingIndex
        numAgents = len( self.agents )
        prev_state = self.state.deepCopy()

        while not self.gameOver:
            # Fetch the next agent
            agent = self.agents[agentIndex]
            # Solicit an action
            action = None
            self.mute(agentIndex)

            # copy_state = self.state.deepCopy()

            ep_obs[agentIndex].append(agent.construct_input(self.state.deepCopy()))
            policy = agent.get_policy(self.state.deepCopy())
            action, log_prob, idx = agent.get_action(self.state.deepCopy(), policy=policy)
            # print(f"Legal actions {self.state.getLegalActions(agentIndex)}")
            # print(f"chosen action {action}")
            # print(f"INDICES {agentIndex} {agent.index}")

            ep_acts[agentIndex].append(idx)
            ep_log_probs[agentIndex].append(log_prob)
            ep_lens[agentIndex][0]+=1
            

            self.unmute()

            # Execute the action
            self.moveHistory.append( (agentIndex, action) )
            self.state = self.state.generateSuccessor( agentIndex, action )

            red = agentIndex in self.state.getRedTeamIndices()

            rew=0
            if "num_carried" in dir(agent):
                rew+=3*(len(agent.getFood(prev_state).asList())-len(agent.getFood(self.state).asList()))
                rew+=50*(len(agent.getCapsules(prev_state))-len(agent.getCapsules(self.state)))
                rew += self.food_potential(self.state, agentIndex)-self.food_potential(prev_state, agentIndex)
                rew += 2*(self.capsule_potential(self.state, agentIndex)-self.capsule_potential(prev_state, agentIndex))
                rew+=5*(agent.num_carried(self.state)/agent.distance_to_home(self.state)[0]-agent.num_carried(prev_state)/agent.distance_to_home(prev_state)[0])
                rew+=10*(agent.num_returned(self.state)-agent.num_returned(prev_state))
                rew+=2*(1/agent.invaderDistance(self.state)[0]-1/agent.invaderDistance(prev_state)[0])
                rew+=4*(agent.num_invaders(prev_state)-agent.num_invaders(self.state))
                rew+=3*(1/agent.defenderDistance(prev_state)[0]-1/agent.defenderDistance(self.state)[0])
                rew=rew*100
            
            # reward = (self.state.getScore()-prev_state.getScore())*(1 if red else -1)
            # reward+=self.potential_reward(prev_state, self.state, agentIndex)
            # print(f"REWARD: {self.potential_reward(prev_state, self.state, agentIndex)}")
            ep_rews[agentIndex].append(rew)

            # Change the display
            self.display.update( self.state.data )

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)

            # Track progress
            if agentIndex == numAgents + 1: self.numMoves += 1
            # Next agent
            agentIndex = ( agentIndex + 1 ) % numAgents
            prev_state=self.state.deepCopy()

        # inform a learning agent of the game result
        for agentIndex, agent in enumerate(self.agents):
            agent.store_episode(ep_obs[agentIndex], ep_acts[agentIndex], ep_log_probs[agentIndex], ep_rews[agentIndex], ep_lens[agentIndex])
            if "final" in dir( agent ) :
                try:
                    self.mute(agentIndex)
                    agent.final( self.state )
                    self.unmute()
                except Exception as data:
                    if not self.catchExceptions: raise
                    self._agentCrash(agentIndex)
                    self.unmute()
                    return
        self.display.finish()





# class StepGame(Game, Environment):
#         def __init__( self, *args, **kwargs):
#             super(StepGame, self).__init__(*args, **kwargs)
#         def getCurrentState(self):
#             """
#             Returns the current state of enviornment
#             """
#             abstract

#         def getPossibleActions(self, state):
#             """
#             Returns possible actions the agent
#             can take in the given state. Can
#             return the empty list if we are in
#             a terminal state.
#             """
#             abstract

#         def doAction(self, action):
#             """
#             Performs the given action in the current
#             environment state and updates the enviornment.

#             Returns a (reward, nextState) pair
#             """
#             abstract

#         def reset(self):
#             """
#             Resets the current state to the start state
#             """
#             abstract

#         def isTerminal(self):
#             """
#             Has the enviornment entered a terminal
#             state? This means there are no successors
#             """
#             state = self.getCurrentState()
#             actions = self.getPossibleActions(state)
#             return len(actions) == 0
        
#         def step( self ):
#             """
#             Main control loop for game play.
#             """
#             self.display.initialize(self.state.data)
#             self.numMoves = 0

#             ###self.display.initialize(self.state.makeObservation(1).data)
#             # inform learning agents of the game start
#             for i in range(len(self.agents)):
#                 agent = self.agents[i]
#                 if not agent:
#                     self.mute(i)
#                     # this is a null agent, meaning it failed to load
#                     # the other team wins
#                     print("Agent %d failed to load" % i, file=sys.stderr)
#                     self.unmute()
#                     self._agentCrash(i, quiet=True)
#                     return
#                 if ("registerInitialState" in dir(agent)):
#                     self.mute(i)
#                     if self.catchExceptions:
#                         try:
#                             timed_func = TimeoutFunction(agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
#                             try:
#                                 start_time = time.time()
#                                 timed_func(self.state.deepCopy())
#                                 time_taken = time.time() - start_time
#                                 self.totalAgentTimes[i] += time_taken
#                             except TimeoutFunctionException:
#                                 print("Agent %d ran out of time on startup!" % i, file=sys.stderr)
#                                 self.unmute()
#                                 self.agentTimeout = True
#                                 self._agentCrash(i, quiet=True)
#                                 return
#                         except Exception as data:
#                             self._agentCrash(i, quiet=False)
#                             self.unmute()
#                             return
#                     else:
#                         agent.registerInitialState(self.state.deepCopy())
#                     ## TODO: could this exceed the total time
#                     self.unmute()

#             agentIndex = self.startingIndex
#             numAgents = len( self.agents )

#             while not self.gameOver:
#                 # Fetch the next agent
#                 agent = self.agents[agentIndex]
#                 move_time = 0
#                 skip_action = False
#                 # Generate an observation of the state
#                 if 'observationFunction' in dir( agent ):
#                     self.mute(agentIndex)
#                     if self.catchExceptions:
#                         try:
#                             timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
#                             try:
#                                 start_time = time.time()
#                                 observation = timed_func(self.state.deepCopy())
#                             except TimeoutFunctionException:
#                                 skip_action = True
#                             move_time += time.time() - start_time
#                             self.unmute()
#                         except Exception as data:
#                             self._agentCrash(agentIndex, quiet=False)
#                             self.unmute()
#                             return
#                     else:
#                         observation = agent.observationFunction(self.state.deepCopy())
#                     self.unmute()
#                 else:
#                     observation = self.state.deepCopy()

#                 # Solicit an action
#                 action = None
#                 self.mute(agentIndex)
#                 if self.catchExceptions:
#                     try:
#                         timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
#                         try:
#                             start_time = time.time()
#                             if skip_action:
#                                 raise TimeoutFunctionException()
#                             action = timed_func( observation )
#                         except TimeoutFunctionException:
#                             print("Agent %d timed out on a single move!" % agentIndex, file=sys.stderr)
#                             self.agentTimeout = True
#                             self._agentCrash(agentIndex, quiet=True)
#                             self.unmute()
#                             return

#                         move_time += time.time() - start_time

#                         if move_time > self.rules.getMoveWarningTime(agentIndex):
#                             self.totalAgentTimeWarnings[agentIndex] += 1
#                             print("Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
#                             if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
#                                 print("Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex]), file=sys.stderr)
#                                 self.agentTimeout = True
#                                 self._agentCrash(agentIndex, quiet=True)
#                                 self.unmute()
#                                 return

#                         self.totalAgentTimes[agentIndex] += move_time
#                         #print "Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex])
#                         if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
#                             print("Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex]), file=sys.stderr)
#                             self.agentTimeout = True
#                             self._agentCrash(agentIndex, quiet=True)
#                             self.unmute()
#                             return
#                         self.unmute()
#                     except Exception as data:
#                         self._agentCrash(agentIndex)
#                         self.unmute()
#                         return
#                 else:
#                     action = agent.getAction(observation)
#                 self.unmute()

#                 # Execute the action
#                 self.moveHistory.append( (agentIndex, action) )
#                 if self.catchExceptions:
#                     try:
#                         self.state = self.state.generateSuccessor( agentIndex, action )
#                     except Exception as data:
#                         self.mute(agentIndex)
#                         self._agentCrash(agentIndex)
#                         self.unmute()
#                         return
#                 else:
#                     self.state = self.state.generateSuccessor( agentIndex, action )

#                 # Change the display
#                 self.display.update( self.state.data )
#                 ###idx = agentIndex - agentIndex % 2 + 1
#                 ###self.display.update( self.state.makeObservation(idx).data )

#                 # Allow for game specific conditions (winning, losing, etc.)
#                 self.rules.process(self.state, self)
#                 # Track progress
#                 if agentIndex == numAgents + 1: self.numMoves += 1
#                 # Next agent
#                 agentIndex = ( agentIndex + 1 ) % numAgents

#                 if _BOINC_ENABLED:
#                     boinc.set_fraction_done(self.getProgress())

#             # inform a learning agent of the game result
#             for agentIndex, agent in enumerate(self.agents):
#                 if "final" in dir( agent ) :
#                     try:
#                         self.mute(agentIndex)
#                         agent.final( self.state )
#                         self.unmute()
#                     except Exception as data:
#                         if not self.catchExceptions: raise
#                         self._agentCrash(agentIndex)
#                         self.unmute()
#                         return
#             self.display.finish()
