# multi_agents.py
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


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
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


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()
        if not legal_moves:
            return Directions.STOP

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        
        # Evaluates the successor state
        value = better_evaluation_function(successor_game_state)

        # Penalty for standing still
        if action == Directions.STOP:
            value -= 20

        return value

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

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

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

    # As all of the search agents are practically the same, we have generalized the code for practicality
    def _search_action(self, game_state, ghost_mode, use_pruning=False):
        """
        Return the best action using the requested adversarial search mode.
        
        The main logic we followed is:
        ghost_mode="minimum": ghosts minimize Pacman’s score.
        ghost_mode="average": ghost moves are averaged.
        use_pruning=True: enables alpha-beta pruning
        """
        num_agents = game_state.get_num_agents()

        # Recursive evaluation 
        def value(state, agent_index, depth, alpha, beta):
            
            # Base case: depth limit reached or game won/lost
            if depth >= self.depth or state.is_win() or state.is_lose():
                return self.evaluation_function(state)
            
            # Gets legal actions for the current agent
            legal_actions = state.get_legal_actions(agent_index)
            if not legal_actions:
                return self.evaluation_function(state)

            # Updates the agent's state for the next turn
            next_agent = (agent_index + 1) % num_agents
            next_depth = depth + 1 if next_agent == self.index else depth

            # Maximizes score
            if agent_index == self.index:
                result = float("-inf")
                
                # Maximizes score by iterating through all legal actions
                for action in legal_actions:
                    # Generates the successor state
                    successor = state.generate_successor(agent_index, action)
                    
                    # Updates the result
                    result = max(
                        result,
                        value(successor, next_agent, next_depth, alpha, beta),
                    )
                    
                    # If the result is greater than beta, return the result
                    if use_pruning and result > beta:
                        return result
                    # Otherwise, update the alpha
                    alpha = max(alpha, result)
                    
                return result
            
            # Minimizes score
            if ghost_mode == "minimum":
                result = float("inf")
                
                # Minimizes score by iterating through all legal actions 
                for action in legal_actions:
                    successor = state.generate_successor(agent_index, action)
                    result = min(
                        result,
                        value(successor, next_agent, next_depth, alpha, beta),
                    )
                    if use_pruning and result < alpha:
                        return result
                    beta = min(beta, result)
                return result

            # Expectimax treats each ghost action as equally likely
            # Calculates the expected value of the successor states
            successor_values = [
                # Updates the state
                value(
                    state.generate_successor(agent_index, action),
                    next_agent,
                    next_depth,
                    alpha,
                    beta,
                )
                for action in legal_actions
            ]
            
            # The average of the successor values
            return sum(successor_values) / len(successor_values)

        # Handle illegal actions
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return Directions.STOP
        
        next_agent = (self.index + 1) % num_agents
        next_depth = 1 if next_agent == self.index else 0
        
        # Initializes the alpha-beta values
        alpha, beta = float("-inf"), float("inf")
        best_action = legal_actions[0]
        best_value = float("-inf")

        # Finds the best action at the current state
        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            action_value = value(successor, next_agent, next_depth, alpha, beta)
            if action_value > best_value:
                best_value = action_value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        return self._search_action(game_state, ghost_mode="minimum")
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        return self._search_action(game_state, "minimum", use_pruning=True)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self._search_action(game_state, ghost_mode="average")

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Combines the game score with food, capsule, and ghost proximity. Active
    ghosts are treated as threats, while reachable scared ghosts are targets.
    """
    # Sets the winning as the most desired outcome
    if current_game_state.is_win():
        return 1_000_000 + current_game_state.get_score()
    if current_game_state.is_lose():
        return -1_000_000 + current_game_state.get_score()

    pacman_position = current_game_state.get_pacman_position()
    food_positions = current_game_state.get_food().as_list()
    capsules = current_game_state.get_capsules()
    ghost_states = current_game_state.get_ghost_states()

    value = current_game_state.get_score()

    # Penalizes the number of moves to reach food
    value -= 4 * len(food_positions)
    
    # Reward the nearest food
    if food_positions:
        nearest_food = min(
            manhattan_distance(pacman_position, food) for food in food_positions
        )
        value += 10 / (nearest_food + 1)

    value -= 15 * len(capsules)
    if capsules:
        nearest_capsule = min(
            manhattan_distance(pacman_position, capsule) for capsule in capsules
        )
        value += 10 / (nearest_capsule + 1)

    # Rewards the nearest capsule
    for ghost_state in ghost_states:
        ghost_distance = manhattan_distance(
            pacman_position, ghost_state.get_position()
        )
        if ghost_state.scared_timer > 0:
            # Only chase ghosts that can be reached before they recover.
            if ghost_distance <= ghost_state.scared_timer:
                value += 100 / (ghost_distance + 1)
        elif ghost_distance == 0:
            return -1_000_000
        elif ghost_distance == 1:
            value -= 200
        elif ghost_distance == 2:
            value -= 75
        else:
            value -= 20 / ghost_distance

    return value
    

# Abbreviation
better = better_evaluation_function
