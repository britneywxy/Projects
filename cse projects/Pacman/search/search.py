# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # starting node with the initial state of problem
    startNode = problem.getStartState()
    # if the state of the node is the goal state, then return empty path to intial state
    if problem.isGoalState(startNode):
        return []

    # construct a LIFO stack and store the initial node
    stack = util.Stack()
    stack.push((startNode, []))
    # construct an empty list to keep track of visited nodes
    visited = list()

    while not stack.isEmpty():
        item = stack.pop() # pop the current node and corresponding action
        currentNode = item[0]
        action = item[1]
        if currentNode not in visited: 
            visited.append(currentNode)
            if problem.isGoalState(currentNode):
                return action
            for successor in problem.getSuccessors(currentNode):
                nextNode = successor[0]
                nextAction = successor[1]
                actions = action + [nextAction]
                stack.push((nextNode,actions))


def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # starting node with the initial state of problem
    startNode = problem.getStartState()
    # if the state of the node is the goal state, then return empty path to intial state
    if problem.isGoalState(startNode):
        return []

    # construct a FIFO stack and store the initial node
    queue = util.Queue()
    queue.push((startNode, []))
    # construct an empty list to keep track of visited nodes
    visited = list()

    while not queue.isEmpty():
        currentNode, action = queue.pop() # pop the current node and corresponding action
        if currentNode not in visited:
            visited.append(currentNode)
            if problem.isGoalState(currentNode):
                return action
            for successor in problem.getSuccessors(currentNode):
                nextNode = successor[0]
                nextAction = successor[1] 
                actions = action + [nextAction]
                queue.push((nextNode,actions))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # starting node with the initial state of problem
    startNode = problem.getStartState()
    # if the state of the node is the goal state, then return empty path to intial state
    if problem.isGoalState(startNode):
        return []

    # construct a priority queue ordered by priority, store the intial node
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((startNode, [], 0), 0) 
    # construct an empty list to keep track of visited nodes
    visited = list()

    while not priorityQueue.isEmpty():
        currentNode, currentAction, currentPriority =  priorityQueue.pop()
        if currentNode not in visited:
            visited.append(currentNode)
            if problem.isGoalState(currentNode):
                return currentAction
            for successor in problem.getSuccessors(currentNode):
                nextNode = successor[0]
                nextAction = successor[1]
                nextPriority = successor[2]
                actions = currentAction + [nextAction]
                priority = currentPriority + nextPriority
                priorityQueue.push((nextNode, actions, priority), priority)
                

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    #starting node with the initial state of problem
    startNode = problem.getStartState()
    # if the state of the node is the goal state, then return empty path to intial state
    if problem.isGoalState(startNode):
        return []

    # construct a priority queue and store the intial node
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((startNode, [], 0), 0) 
    # construct an empty list to keep track of visited nodes
    visited = list()

    while not priorityQueue.isEmpty():
        currentNode, currentAction, currentCost = priorityQueue.pop()
        if currentNode not in visited:
            visited.append(currentNode)
            if problem.isGoalState(currentNode):
                return currentAction
            for successor in problem.getSuccessors(currentNode):
                nextNode = successor[0]
                nextAction = successor[1]
                nextCost = successor[2]    
                actions = currentAction + [nextAction]
                cost = currentCost + nextCost
                heuristicCost = cost + heuristic(nextNode, problem)
                priorityQueue.push((nextNode, actions, cost), heuristicCost)

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
