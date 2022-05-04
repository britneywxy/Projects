# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foods = newFood.asList()
        score = successorGameState.getScore()
        nearestFood = float("inf")
        nearestGhost = float("inf")


        # Find nearest food distance 
        for food in foods:
            nearestFood = min(nearestFood, manhattanDistance(newPos, food))

        # Find nearest ghost distance
        for ghost in newGhostStates:
            if ghost.scaredTimer == 0:
                nearestGhost = min(nearestGhost,manhattanDistance(newPos,ghost.getPosition()))
            if (manhattanDistance(newPos, ghost.getPosition()) <= 1):
                return -float('inf')

        return score + 1.0/nearestFood + 1.0/nearestGhost

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # return self.maxValue(gameState, 0, 0)[0]
        # return self.minimax(gameState, 0, 0)[1]
        return self.minimax(gameState,0,0)[1]

    def minimax(self,gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState),None
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0  #reset the value of agent index on cycle
        if agentIndex == 0: # the agent is pacman
            return self.maxVal(gameState, depth, agentIndex)
        else: # the agent is ghost
            return self.minVal(gameState, depth, agentIndex)

    def maxVal(self, gameState, depth, agentIndex):
        value = float("-inf"), Directions.STOP
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            # Recursive call
            availValue = self.minimax(gameState.generateSuccessor(agentIndex, action), depth + 1, (depth + 1)%gameState.getNumAgents())[0]
            value = max(value, (availValue, action), key=lambda x: x[0])
        return value

    def minVal(self, gameState, depth, agentIndex):
        value = float("inf"), Directions.STOP
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            # Recursive call
            availValue = self.minimax(gameState.generateSuccessor(agentIndex, action), depth + 1, (depth + 1)%gameState.getNumAgents())[0]
            value = min(value, (availValue, action), key=lambda x: x[0])
        return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf'),None
        beta = float('inf'),None
        return self.alphabeta(gameState, 0, 0, alpha, beta)[1]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
            return self.evaluationFunction(gameState),None
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0  #reset the value of agent index on cycle
        if agentIndex == 0:
            return self.maxVal(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minVal(gameState, agentIndex, depth, alpha, beta)

    def maxVal(self, gameState, agentIndex, depth, alpha, beta):
        value = -float("inf"),Directions.STOP
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            availValue = self.alphabeta(gameState.generateSuccessor(agentIndex,action),(depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta)[0]
            value = max(value,(availValue,action),key=lambda x:x[0])
            # do the prunning
            if value[0] > beta[0]: 
                return value
            else: 
                alpha = max(alpha,value,key=lambda x:x[0])

        return value


    def minVal(self, gameState, agentIndex, depth, alpha, beta):
        value = float("inf"),Directions.STOP
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            availValue = self.alphabeta(gameState.generateSuccessor(agentIndex,action),(depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta)[0]
            value = min(value,(availValue,action),key=lambda x:x[0])
            # do the prunning
            if value[0] < alpha[0]: 
                return value
            else: 
                beta = min(beta, value, key=lambda x: x[0])

        return value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, 0, maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (action, self.evaluationFunction(gameState))

        # if the agent is pacman, then return the maximum value of successor node
        if agentIndex is 0:
            return self.maxValue(gameState,action,depth,agentIndex)
        # if the agent is ghost, then return the probability
        else:
            return self.probaValue(gameState,action,depth,agentIndex)

    # find the maximum value of successor node
    def maxValue(self,gameState,action,depth,agentIndex):
        bestAction = (None, -(float('inf')))
        legalActions = gameState.getLegalActions(agentIndex)
        for legalAction in legalActions:
            newAgent = (agentIndex + 1) % gameState.getNumAgents()
            newDepth = self.depth * gameState.getNumAgents()
            if depth == newDepth:
                successorAct = legalAction
            else:
                successorAct = action
            #generate successor
            successor = gameState.generateSuccessor(agentIndex,legalAction)
            #get the best action
            successVal = self.expectimax(successor,successorAct,depth - 1,newAgent)
            bestAction = max(bestAction,successVal,key = lambda x:x[1])
        return bestAction

    # calculates the chance
    def probaValue(self,gameState,action,depth,agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        avgChance = 0
        chance = 1.0/len(legalActions)
        for legalAction in legalActions:
            newAgent = (agentIndex + 1) % gameState.getNumAgents()
            #generate succesor
            successor = gameState.generateSuccessor(agentIndex, legalAction)
            #calculate the best chance
            bestChance = self.expectimax(successor,action,depth - 1, newAgent)
            avgChance += bestChance[1] * chance
        return (action, avgChance)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
