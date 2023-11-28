from typing import Any
from myTeam import DummyAgent
import game, capture
import numpy as np 
import torch
import torch.nn as nn
import json
from gameparser import Parser

directions = list(game.Actions._directions.keys())


def createTeam(firstIndex, secondIndex, isRed,
               first = 'MLPPolicy', second = 'MLPPolicy'):
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
  return [eval(first)(firstIndex, nnet=lambda x:nnet1(torch.Tensor(x)).detach().numpy()), eval(second)(secondIndex)]




class MLPPolicy(DummyAgent):
    def __init__(self, index, timeForComputing = .1, nnet=lambda x:[1.,1,1,1,1], construct_input=lambda x:np.array([1,1])):
        super().__init__(index, timeForComputing = timeForComputing)
        self.nnet=nnet 
        self.construct_input=construct_input
         
    def chooseAction(self, gameState: capture.GameState):
        parser = Parser()
        actions = gameState.getLegalActions(self.index)
        print(f"AGENT STATES {[parser.parse_state(gameState.getAgentState(i)) for i in range(gameState.getNumAgents()) if gameState.getAgentState(i).numReturned>0]}")
        input = self.construct_input(gameState)
        output = self.nnet(input)
        policy = np.array([output[directions.index(a)] for a in actions])
        policy = np.e**policy
        policy = policy/np.sum(policy)
        print(f"POLICY: {policy}")
        rng = np.random.default_rng()
        return actions[rng.choice(len(actions),p=policy)]
    
class numpyLayer():
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self):
        pass
    
class numpyLinear():
    def __init__(self, weight,bias):
        self.weight = weight
        self.bias = bias 
    def forward(self, input):
        return input@np.transpose(self.weight)+self.bias

class numpySequential():
    pass


nnet1 = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)


# Utilities for saving and loading agent neural nets
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

# load_parameters(nnet1, {"0.weight": [[0.35932496190071106, 0.4700525104999542], [-0.32676374912261963, 0.32689473032951355], [0.6029775738716125, 0.6850628852844238], [-0.4761100113391876, 0.40193358063697815], [0.2384950965642929, -0.37523600459098816], [0.6385453939437866, -0.5253885984420776], [-0.7015116214752197, -0.036123473197221756], [-0.4814959466457367, 0.5429476499557495], [0.22788774967193604, 0.3479498028755188], [-0.23838306963443756, -0.2982620298862457]], "0.bias": [0.4851861596107483, 0.5002991557121277, -0.2399613857269287, -0.04689553380012512, 0.10841728746891022, -0.4277598261833191, 0.6341562271118164, 0.5906228423118591, 0.22587144374847412, 0.06126625835895538], "2.weight": [[0.0954066663980484, -0.2707728147506714, 0.17863455414772034, 0.0643635168671608, 0.2935868501663208, -0.11360985785722733, 0.178012877702713, 0.26355937123298645, 0.27915748953819275, -0.07682307809591293], [0.21483156085014343, -0.2708161771297455, -0.15479078888893127, 0.14490124583244324, 0.2083539217710495, -0.000974097871221602, -0.11014736443758011, 0.15311171114444733, 0.2641153335571289, 0.17565888166427612], [-0.16903701424598694, -0.27078941464424133, 0.06469408422708511, -0.0004031724820379168, -0.25190383195877075, -0.2302275151014328, -0.12267225235700607, 0.021051861345767975, 0.2182309925556183, -0.05884810537099838], [-0.178777277469635, -0.24604400992393494, 0.12617142498493195, -0.1594284623861313, -0.003698028391227126, -0.09947435557842255, 0.18632404506206512, -0.14570416510105133, -0.2981168329715729, 0.16634775698184967], [-0.00031597865745425224, -0.04829738661646843, 0.27659109234809875, -0.15046080946922302, 0.27668482065200806, 0.14337989687919617, 0.11950956284999847, -0.09969138354063034, -0.28370770812034607, -0.196334108710289], [-0.2611958384513855, -0.27762946486473083, -0.1959177702665329, -0.13800837099552155, -0.2525533437728882, -0.021716879680752754, -0.1551440954208374, -0.07664001733064651, 0.1278144270181656, 0.17808115482330322], [0.19018127024173737, -0.13781848549842834, -0.21721765398979187, -0.18022149801254272, -0.04199850931763649, -0.18632479012012482, 0.17329838871955872, 0.060802292078733444, 0.09086659550666809, -0.07688117027282715], [0.05901099368929863, -0.08414053171873093, -0.3056967258453369, -0.06641926616430283, -0.25224533677101135, 0.20355509221553802, 0.14270621538162231, -0.1622934192419052, 0.027780411764979362, -0.21587461233139038], [-0.08847831934690475, 0.056939490139484406, -0.19096677005290985, 0.1947886347770691, -0.09548944979906082, -0.023153938353061676, 0.1742781698703766, 0.2823265492916107, -0.05984530970454216, 0.1289321929216385], [-0.03711720183491707, 0.033545006066560745, -0.08088129758834839, -0.20707741379737854, 0.2894207835197449, 0.25120699405670166, 0.20337693393230438, -0.09484558552503586, -0.25515833497047424, 0.19910791516304016]], "2.bias": [0.29624760150909424, -0.09798339009284973, -0.02231430634856224, -0.09409555792808533, 0.1967475712299347, -0.11388082802295685, 0.22225548326969147, -0.029290717095136642, 0.0029583321884274483, 0.1359478086233139], "4.weight": [[0.15838854014873505, 0.03690805658698082, -0.2575165033340454, -0.2740507125854492, 0.22878295183181763, 0.17594800889492035, -0.16462759673595428, -0.2007599174976349, 0.2452821433544159, 0.26959821581840515], [0.2825130522251129, -0.19781138002872467, -0.23023584485054016, -0.12011222541332245, 0.13481836020946503, -0.008632264100015163, 0.12675735354423523, -0.258272647857666, 0.29508888721466064, -0.19087165594100952], [-0.11296889185905457, 0.10732568055391312, 0.24182282388210297, -0.18943606317043304, 0.2994401454925537, 0.18427075445652008, -0.2495240420103073, -0.17782051861286163, 0.29332008957862854, 0.10257578641176224], [-0.11505924165248871, -0.1380353569984436, 0.08024199306964874, -0.22255931794643402, 0.22043974697589874, -0.11294981837272644, 0.1463940143585205, 0.09518018364906311, 0.24958544969558716, -0.03581954911351204], [0.03668598085641861, -0.24575532972812653, -0.16853469610214233, 0.05618912726640701, -0.18094350397586823, -0.13771241903305054, -0.2083449512720108, -0.07363056391477585, 0.081366166472435, 0.06627601385116577]], "4.bias": [0.0640651062130928, 0.28673404455184937, -0.2958736717700958, 0.007860185578465462, -0.05786952003836632]})
# print_parameters(nnet1)

# gpa= [[4,4,4,4,4],[4],[4,4,4,4],[4,4,4,4,3.7,4],[4,4,3.3,4,4],[4,4,4,4],[4,4,4,4]]
# credits=[[4,2,4,4,4],[4],[4,2,4,4],[4,2,4,4,4,4],[4,4,4,4,2],[4,4,4,4],[4,4,4,4]]

# print(sum([sum(c) for c in credits]))
# print((0.7*4+0.3*4)/108)

# print(sum([sum([g_*c_ for g_,c_ in zip(g,c)]) for g,c in zip(gpa,credits)])/sum([sum(c) for c in credits]))
