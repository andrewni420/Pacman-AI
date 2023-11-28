from itertools import combinations

class Parser():
    def __init__(self, distancer=None, id=0, numAgents=2):
        self.distancer = distancer
        self.id = self.oneHot(id,numAgents)

    def parse_game(self, gameState):
        #What should we use as inputs to the neural net?
        # 6 pairwise maze distances between pacman agents
        # One-hot encoding for current agent index
        # Information about each agent:
        #   Location
        #   direction
        #   number of carried food
        #   scared timer
        #   one-hot encoding for whether it's scared
        #   one-hot encoding for ghost vs pacman
        #   maze distance to nearest red food
        #   maze distance to nearest blue food 
        #   maze distance to nearest red capsule
        #   maze distance to nearest blue capsule
        pass 
        

    def oneHot(self,i,n):
        arr =[0]*n
        arr[i]=1
        return arr

    def distances(self, agents):
        positions = [a.getPosition() for a in agents]
        return [self.distancer.getDistance(a,b) for a,b in combinations(positions,2)] 
    def locations(self, agents):
        return [a.getPosition() for a in agents]     
    def directions(self, agents):
        return [a.getDirection() for a in agents]
    def carrying(self, agents):
        return [a.numCarrying for a in agents]
    def scaredTimer(self, agents):
        return [a.scaredTimer for a in agents]
    def isScared(self, agents):
        return [self.oneHot(a.scaredTimer>0,2) for a in agents]
    
    

    def parse_state(self, agentState, distancer=None):
        position=agentState.getPosition()
        direction=agentState.getDirection()
        startConfig=agentState.start
        startPos = startConfig.getPosition()
        startDir = startConfig.getDirection()
        pacman = agentState.isPacman
        scaredTimer=agentState.scaredTimer
        numCarrying=agentState.numCarrying
        numReturned=agentState.numReturned
        return {"position": position, "direction": direction, "startPos":startPos, "startDir": startDir, 
                "pacman":pacman, "scaredTimer":scaredTimer, "numCarrying":numCarrying, "numReturned":numReturned}
