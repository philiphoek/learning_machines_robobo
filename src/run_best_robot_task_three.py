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

    allowed_steps = 100
    food_delivered = 0
    initial_distance_food_to_base = controller.getDistance()
    steps_taken = 0
    times_food_in_gripper = 0
    reward_food_in_gripper_and_seeing_base = 0
    object_hit = 0
    reward_having_food_in_center_of_image = 0
    for i in range(allowed_steps):
        controller.makeStep(best)

        if controller.rob.base_detects_food():
            food_delivered = 1
            break

        if controller.food_in_gripper == 1:
            times_food_in_gripper += 0.25

        if controller.rob.check_for_collision():
            object_hit = 1
            # stop the simulation ones an non food object is hit
            print('Object is hit')
            break

        if (controller.green_left == 1 or controller.green_right) and controller.food_in_gripper:
            reward_food_in_gripper_and_seeing_base += 0.25

        if controller.green_center == 1 and controller.food_in_gripper == 1:
            reward_food_in_gripper_and_seeing_base += 0.5

        if controller.red_center == 1:
            reward_having_food_in_center_of_image += 0.25

        steps_taken += 1

    final_distance_food_to_base = controller.getDistance()
    relative_distance_food_to_base = final_distance_food_to_base / initial_distance_food_to_base  # closer to zero means food is close to the base
    print(f"Food deliverd: {food_delivered}")
    print(f"Relative distance travelled food to base: {relative_distance_food_to_base}")
    print(f"Times food in gripper: {times_food_in_gripper}")
    print(f"Reward for having food in center of image: {reward_having_food_in_center_of_image}")
    print(f"Reward seeing base while having food in gripper: {reward_food_in_gripper_and_seeing_base}")
    print(f"Object hit: {object_hit}")
    fitness = 0
    fitness += 200 * food_delivered
    fitness += 50 * (1 - relative_distance_food_to_base)
    fitness += (times_food_in_gripper * 1)
    fitness += reward_having_food_in_center_of_image
    fitness += reward_food_in_gripper_and_seeing_base
    fitness -= 200 * object_hit
    print(f"fitness: {fitness}")

    controller.rob.stop_world()
    controller.rob.wait_for_stop()


if __name__ == "__main__":
    main()