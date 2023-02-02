#!/usr/bin/env python3
from __future__ import print_function

import numpy as np

import robobo_task_three
import cv2
import sys
import signal
from controller_task_three import robotController


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    controller = robotController(robobo_task_three.SimulationRobobo().connect(address='127.0.0.1', port=19997))

    best_text_file = input("Enter path the best text file: ")
    with open(best_text_file, 'rb') as f:
        best = np.loadtxt(f)

    controller.rob.play_simulation()
    controller.rob.set_phone_tilt(19.8, 1)
    controller.rob.randomize_position()

    allowed_steps = 100
    threshold_not_having_food = 100
    food_delivered = 0
    initial_distance_food_to_base = controller.getDistance()
    steps_taken = 0
    time_steps_food_in_gripper = 0
    object_hit = 0
    for i in range(allowed_steps):
        controller.makeStep(best)

        if controller.rob.base_detects_food():
            food_delivered = 1
            break

        if controller.food_in_gripper == 1:
            time_steps_food_in_gripper += 1

        if controller.rob.check_for_collision():
            object_hit = 1
            # stop the simulation ones an non food object is hit
            print('Object is hit')
            break

        if steps_taken > threshold_not_having_food and time_steps_food_in_gripper == 0:
            print(f"Robot has not managed to get food in gripper for {threshold_not_having_food} time steps")
            break

        steps_taken += 1

    final_distance_food_to_base = controller.getDistance()
    relative_distance_food_to_base = final_distance_food_to_base / initial_distance_food_to_base  # closer to zero means food is close to the base
    print(f"Food deliverd: {food_delivered}")
    print(f"Steps taken: {steps_taken}")
    print(f"Relative distance travelled food to base: {relative_distance_food_to_base}")
    print(f"Time steps food in gripper: {time_steps_food_in_gripper}")
    print(f"Object hit: {object_hit}")
    fitness = 0
    # fitness is highest when food is closest to base and the least amount of steps are taken
    fitness += 200 * (1 - relative_distance_food_to_base) * (1 - (steps_taken / allowed_steps))
    # small reward for having food in the gripper for most of the time
    fitness += 50 * (time_steps_food_in_gripper / steps_taken)
    # reduce fitness when object has been hit
    if object_hit == 1:
        fitness = fitness * 0.2
    print(f"fitness: {fitness}")

    controller.rob.stop_world()
    controller.rob.wait_for_stop()


if __name__ == "__main__":
    main()