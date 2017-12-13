import numpy as np
from scipy.special import expit
from abc import ABCMeta, abstractmethod
import os
import pickle
import run


# Abstract class for all AIs
class AI(object):

    __metaclass__ = ABCMeta

    def __init__(self, init_strategy=None):
        self.strategy = init_strategy

    @abstractmethod
    def jump(self, obstacles, score):
        return 0

    # returns the negative score and sanity checks
    # if the strategy obviously won't work then no need to run the program
    def get_performance_cost(self, strategy=None, num_runs=3):

        if strategy is not None:
            self.strategy = strategy

        empty_state = [600, 0]
        needs_to_jump = [5, 1]

        # if it jumps when nothing is on screen, that's bad
        if self.jump(empty_state, 1) > .5:
            return 0

        # it better jump if the closest obstacle if super close
        if self.jump(needs_to_jump, 1) < .5:
            return 0

        cost = 0
        for _ in range(num_runs):
            cost += -run.run(ai=self, report=False)
        cost = cost/num_runs

        return cost


class RuleBased(AI):

    file_name = 'ruleBased.p'

    # If no initial strategy given, look to see if a pickled model exists, if it doesnt make a random one
    def __init__(self, init_strategy=None):

        if init_strategy is not None:
            self.strategy = init_strategy

        elif Logistic.file_name in os.listdir('AI_files'):
            self.strategy = pickle.load(open(os.path.join('AI_files', Logistic.file_name), 'rb'))

        else:
            print('no strategy given, setting to random gaussian (0,1)')
            self.strategy = np.random.normal()

    def jump(self, obstacles, score):
        return obstacles[0] <= self.strategy


class Logistic(AI):

    file_name = 'logistic.p'

    # If no initial strategy given, look to see if a pickled model exists, if it doesnt make a random one
    def __init__(self, init_strategy=None):

        if init_strategy is not None:

            if type(init_strategy) != np.ndarray:
                init_strategy = np.array(init_strategy)

            self.strategy = init_strategy

        elif Logistic.file_name in os.listdir('AI_files'):
            print('here')
            #self.strategy = pickle.load(open(os.path.join('AI_files', Logistic.file_name), 'rb'))

        else:
            print('no strategy given, setting to random gaussian (0,1)')
            self.strategy = np.random.normal(size=(1, 8))

    def jump(self, obstacles, score):
        state = np.array([1] + obstacles + [score])  # add one for bias unit
        jump = expit(np.dot(self.strategy, state))
        return jump


class GeneticNN(AI):
    pass


if __name__ == "__main__":
    run.run(Logistic([60]))