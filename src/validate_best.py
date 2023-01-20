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

def getDistance(point_one, point_two):
    p1 = np.array(point_one)
    p2 = np.array(point_two)

    squared_dist = np.sum((p1 - p2) ** 2, axis=0)
    return np.sqrt(squared_dist)


def detect_object(robobot):
    for i in range(8):
        raw_value = np.nan_to_num(robobot.read_irs())[i]
        if raw_value != 0:
            value = np.log(np.array(raw_value))
            if (value / 10 > -1) and (value / 10 < -0.20):
                return True

    return False


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

    print()
    print('Simulation started')
    rob.play_simulation()

    steps_taken = 0
    times_near_object = 0
    position_start = rob.position()
    position_after = rob.position()
    times_moved_back = 0
    intermediate_speed_difference = np.zeros(shape=(allowed_steps))
    intermediate_distance = np.zeros(shape=(allowed_steps))
    intermediate_position = np.zeros(shape=(allowed_steps, 3))
    # Following code moves the robot
    for i in range(allowed_steps):
        position_before_step = rob.position()
        values = np.array(rob.read_irs(), float)

        left, right = controller.control(np.nan_to_num(values), best)
        intermediate_speed_difference[i] = abs(left - right)
        rob.move(left, right, 1000)

        if (rob.check_for_collision()):
            # stop the simulation ones an object is hit
            print('Object is hit')
            break

        position_after_step = rob.position()
        intermediate_distance[i] = getDistance(position_before_step, position_after_step)
        intermediate_position[i] = position_after_step

        # Stop if the robot has travelled less than 0.2 meters in the last 5 time steps
        if (i > 10):
            distance_travelled_in_last_ten_steps = getDistance(intermediate_position[i - 10], intermediate_position[i])
            # print()
            # print('distance travelled:')
            # print(distance_travelled_in_last_ten_steps)
            if (distance_travelled_in_last_ten_steps < 0.20):
                # Add penalty if the robot is not moving enough

                print(f"Robot is not moving enough (travelled only {distance_travelled_in_last_ten_steps} meters)")
                break

        if (detect_object(rob)):
            times_near_object += 0.5

        steps_taken += 1

    # Stopping the simualtion resets the environment
    position_after = rob.position()
    rob.stop_world()
    rob.wait_for_stop()

    print(f"Steps taken: {steps_taken}")
    print(f"Near object penalty: {times_near_object}")
    print(f"Total distance: {np.sum(intermediate_distance)}")
    print(f"Mean speed difference: {np.mean(intermediate_speed_difference)}")
    fitness = (steps_taken - times_near_object) * np.sum(intermediate_distance) * (
                1 - (np.mean(intermediate_speed_difference) / 60))
    print(f"fitness: {fitness}")
    return fitness


if __name__ == "__main__":
    main()