import numpy as np


class Controller(object):

    def control(self, inputs: np.array, controller: np.array):
        action1 = np.random.choice([1, 0])
        action2 = np.random.choice([1, 0])

        return [action1, action2]
