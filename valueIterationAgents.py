# valueIterationAgents.py
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


# valueIterationAgents.py
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


import mdp, util, pprint

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        for state in self.mdp.getStates():
            print(f"State: {state}--------------------------------------------------------------------")
            possActions = self.mdp.getPossibleActions(state)
            print(f"Possible Actions for {state}: {possActions}")
            for action in possActions:
                print(f"Possible Values and Probs for {action}: {self.mdp.getTransitionStatesAndProbs(state,action)}")

        if(self.iterations == 0):
            print("On Iteration 1-----------------------")
            for state in self.mdp.getStates():
                print(f"State: {state} | Possible Actions: {self.mdp.getPossibleActions(state)}")
                possibleActions = self.mdp.getPossibleActions(state)
                reward = 0
                # if (('exit',) == possibleActions):
                #     reward = self.mdp.getReward(state, 'exit', 'TERMINAL_STATE')
                self.values[state] =  reward
        else:
            print("On iteration 2+=====================")
            for state in self.mdp.getStates():
                possibleVals = map((lambda act: (self.computeQValueFromValues(act),act)), self.mdp.getPossibleActions())
                print("possibleVals: {possibleVals}")
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        mdp = self.mdp
        s = state
        a = action
        actionValue = self.values[state]
        statesAndProbs = mdp.getTransitionStatesAndProbs(s,a)
        for state, prob in statesAndProbs:
            if(mdp.isTerminal(state)):
                actionValue = None
            else:
                destinationValue = 0
                for newState, prob in statesAndProbs:
                    destinationValue = self.values[newState]
                    print(f"Reward for state: {state} with action {action} | {mdp.getReward(state,action,newState)}")
                    actionValue += prob * (mdp.getReward(state, action, newState) + self.discount * destinationValue)
        print(f"actionValue: {actionValue}")
        return actionValue

        # if(mdp.isTerminal(state)):
        #     print("MDP is terminal, maybe deal with it or smth")
        #    
        #     self.values[s] = mdp.getReward(state, a, state)
        # else:
        #     statesAndProbs = mdp.getTransitionStatesAndProbs(s,a)
        #     
        #     maxQ = 0


    def getActionValue(self,state,action):
        mdp = self.mdp
        s = state
        a = action
        actionValue = 0

        statesAndProbs = mdp.getTransitionStatesAndProbs(s,a)
        for state, prob in statesAndProbs:
            if(mdp.isTerminal(state)):
                actionValue = self.values[state]
            else:
                destinationValue = 0
                for newState, prob in statesAndProbs:
                    destinationValue = self.values[newState]
                    actionValue += prob * (mdp.getReward(state, action, newState) + self.discount * destinationValue)
        

        return actionValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # TODO deal with ties
        print("In computeAction ------------------------------------------")
        possibleActions = self.mdp.getPossibleActions(state)
        print(possibleActions)
        savedAct = ""
        savedVal = 0
        for act in possibleActions:
            val = self.values[state]
            print(f"Value: {val}")
            if val > savedVal:
                savedAct = act
                savedVal = val
        return savedAct if possibleActions != () else None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
