#!/usr/bin/env python3
from __future__ import print_function

import numpy as np

import time
import robobo_task_three
import sys
import signal
from controller_task_three import robotController


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)


    allowed_steps = 100

    # hotspot iphone philip
    controller = robotController(robobo_task_three.HardwareRobobo(camera=True).connect(address="172.20.10.9"))
    controller.rob.set_phone_tilt(50, 2)

    # load the best individual from folder DEAP_ES_against_enemies_7_8_for_1_runs.txt
    # best_text_file = input("Enter path the best text file: ")
    with open('Results task 3/random_food_position/best-8.txt', 'rb') as f:
        best = np.loadtxt(f)

    print('starting')
    for i in range(allowed_steps):
        controller.makeStep(best)


if __name__ == "__main__":
    main()