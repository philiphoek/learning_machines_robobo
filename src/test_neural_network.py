from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from robot_controller import robotController


def main():
    n_hidden_neurons = 10
    number_of_sensors = 6
    # left wheel and right wheel
    number_of_actions = 2

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (number_of_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 2

    print('n_vars: ' + str(n_vars))

    dom_u = 1
    dom_l = -1
    npop = 1

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    print(str(pop[0]))
    robot = pop[0]

    # create instance of controller class
    controller = robotController(n_hidden_neurons)

    controller.control(np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8]), robot)


if __name__ == "__main__":
    main()
