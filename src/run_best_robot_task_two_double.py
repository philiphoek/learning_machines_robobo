#!/usr/bin/env python3
from __future__ import print_function

import numpy as np

import robobo
import sys
import signal
from two_robot_controller_task_two import robotController


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    controller = robotController(robobo.SimulationTwoRobobos().connect(address='127.0.0.1', port=19997))

    best_text_file = input("Enter path the best text file: ")
    with open(best_text_file, 'rb') as f:
        best = np.loadtxt(f)

    controller.rob.play_simulation()
    controller.rob.set_phone_tilt(19.6, 1)
    allowed_steps = 60
    steps_taken = 0
    times_near_object = 0
    object_hit = 0
    reward_for_seeing_object = 0
    for i in range(allowed_steps):
        controller.makeStep(best)

        if controller.rob.check_for_collision() or controller.rob.check_for_collision_two():
            object_hit = 1
            # stop the simulation ones an object is hit
            print('Object is hit')
            break

        if controller.top_left == 1 or controller.top_right == 1:
            reward_for_seeing_object += 0.5

        if controller.bottom_left == 1 or controller.bottom_right == 1 or controller.top_center == 1:
            reward_for_seeing_object += 1

        if controller.bottom_center == 1:
            reward_for_seeing_object += 1.5

        if controller.detect_object():
            times_near_object += 0.5

        steps_taken += 1

    print(f"Food collected: {controller.rob.collected_food()}")
    print(f"Reward for seeing object: {reward_for_seeing_object}")
    print(f"Object hit: {object_hit}")
    print(f"Near object penalty: {times_near_object}")
    fitness = 30 * controller.rob.collected_food() + reward_for_seeing_object - 100 * object_hit - times_near_object
    print(f"fitness: {fitness}")

    # Stopping the simualtion resets the environment
    controller.rob.stop_world()
    controller.rob.wait_for_stop()


if __name__ == "__main__":
    main()