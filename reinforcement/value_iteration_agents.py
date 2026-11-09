# value_iteration_agents.py
# -----------------------
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


# value_iteration_agents.py
# -----------------------
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


import mdp, util

from learning_agents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminal(state)
        """
        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.run_value_iteration()
    
    def run_value_iteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            # Batch update: every state in this sweep reads self.values from
            # the previous iteration.  Publish the new vector only afterwards.
            next_values = util.Counter()
            for state in self.mdp.get_states():
                if self.mdp.is_terminal(state):
                    continue

                actions = self.mdp.get_possible_actions(state)
                if actions:
                    next_values[state] = max(
                        self.compute_q_value_from_values(state, action)
                        for action in actions
                    )

            self.values = next_values
            
    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0.0
        for next_state, probability in self.mdp.get_transition_states_and_probs(state, action):
            reward = self.mdp.get_reward(state, action, next_state)
            q_value += probability * (
                reward + self.discount * self.values[next_state]
            )
        return q_value

    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.get_possible_actions(state)
        if not actions:
            return None

        return max(
            actions,
            key=lambda action: self.compute_q_value_from_values(state, action)
        )

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state)
              mdp.is_terminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """Run cyclic, in-place value iteration one state at a time."""
        states = self.mdp.get_states()
        if not states:
            return

        for iteration in range(self.iterations):
            state = states[iteration % len(states)]

            # A terminal selection still consumes this iteration; it simply
            # has no Bellman update to apply.
            if self.mdp.is_terminal(state):
                continue

            actions = self.mdp.get_possible_actions(state)
            if actions:
                # Publish immediately so later iterations see this update.
                self.values[state] = max(
                    self.compute_q_value_from_values(state, action)
                    for action in actions
                )

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """Run value iteration in descending order of Bellman error."""
        states = self.mdp.get_states()
        predecessors = {state: set() for state in states}

        # A state's value can directly affect only states that can transition
        # into it, so record those predecessors once from the MDP model.
        for state in states:
            if self.mdp.is_terminal(state):
                continue
            for action in self.mdp.get_possible_actions(state):
                transitions = self.mdp.get_transition_states_and_probs(state, action)
                for next_state, probability in transitions:
                    if probability > 0:
                        predecessors.setdefault(next_state, set()).add(state)

        priority_queue = util.PriorityQueue()
        for state in states:
            if self.mdp.is_terminal(state):
                continue

            actions = self.mdp.get_possible_actions(state)
            if actions:
                best_q_value = max(
                    self.compute_q_value_from_values(state, action)
                    for action in actions
                )
                diff = abs(self.values[state] - best_q_value)
                # PriorityQueue is a min-heap, so larger errors are more negative.
                priority_queue.push(state, -diff)

        for _ in range(self.iterations):
            if priority_queue.is_empty():
                break

            state = priority_queue.pop()
            actions = self.mdp.get_possible_actions(state)
            if not self.mdp.is_terminal(state) and actions:
                # Update in place so predecessor errors use the newest value.
                self.values[state] = max(
                    self.compute_q_value_from_values(state, action)
                    for action in actions
                )

            for predecessor in predecessors.get(state, set()):
                actions = self.mdp.get_possible_actions(predecessor)
                if not actions:
                    continue

                best_q_value = max(
                    self.compute_q_value_from_values(predecessor, action)
                    for action in actions
                )
                diff = abs(self.values[predecessor] - best_q_value)
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)

