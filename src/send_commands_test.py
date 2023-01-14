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
    number_of_sensors = 8
    # left wheel and right wheel
    number_of_actions = 2

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (number_of_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 2

    controller = robotController(n_hidden_neurons)

    dom_u = 1
    dom_l = -1
    npop = 1

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    robot = pop[0]

    # Following code moves the robot
    for i in range(100):
        # print("robobo is at {}".format(rob.position()))
        values = np.array(rob.read_irs(), float)
        # print(np.nan_to_num(values))
        # values = np.nan_to_num(values)
        # print(values)
        left, right = controller.control(np.nan_to_num(values), robot)
        print([left, right])
        rob.move(left, right, 1000)
        # print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
        # print("Base sensor detection: ", rob.base_detects_food())

    # print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()