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
    n_hidden_neurons = 10
    allowed_steps = 50
    controller = robotController(n_hidden_neurons)

    # load the best individual from folder DEAP_ES_against_enemies_7_8_for_1_runs.txt
    best_text_file = input("Enter path the best text file: ")
    with open(best_text_file, 'rb') as f:

        best = np.loadtxt(f)
        print(type(best))

    rob.play_simulation()
    for i in range(allowed_steps):
        values = np.array(rob.read_irs(), float)

        left, right = controller.control(np.nan_to_num(values), best)
        rob.move(left, right, 1000)
        # print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
        # print("Base sensor detection: ", rob.base_detects_food())

    # print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simualtion resets the environment
    rob.stop_world()
    rob.wait_for_stop()


if __name__ == "__main__":
    main()