#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from robot_controller import robotController


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.178.66")
    rob = robobo.SimulationRobobo().connect(address='192.168.178.66', port=19997)

    rob.play_simulation()

    n_hidden_neurons = 10
    number_of_sensors = 6
    # left wheel and right wheel
    number_of_actions = 2

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (number_of_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 2

    controller = robotController(n_hidden_neurons)

    print('hallo')

    dom_u = 1
    dom_l = -1
    npop = 1

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    robot = pop[0]

    # Following code moves the robot
    for i in range(10):
        print("robobo is at {}".format(rob.position()))
        values = np.array(rob.read_irs())
        values[np.isnan(values)] = 0
        print(values)
        left, right = controller.control(np.array([1,1,1,1,1,1]), robot)
        rob.move(left, right, 2000)
        print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
        # print("Base sensor detection: ", rob.base_detects_food())

    print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()