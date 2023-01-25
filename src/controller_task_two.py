import numpy as np


class Controller(object):

    def control(self, controller: np.array):
        action1 = np.random.choice([-0.5, 0.5])
        action2 = np.random.choice([-0.5, 0.5])

        return [action1, action2]
