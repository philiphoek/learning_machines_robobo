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

def detect_object(robobot):
    for i in range(8):
        raw_value = np.nan_to_num(robobot.read_irs())[i]
        if raw_value != 0:
            value = np.log(np.array(raw_value))
            if (value / 10 > -1) and (value / 10 < -0.20):
                return True

    return False

# https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/?ref=rp
def detect_green(image):
    # Convert the image in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(image, image,
                                mask=green_mask)

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best_x = 0
    best_y = 0
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > best_area and area > 200:
            best_area = area

            x, y, w, h = cv2.boundingRect(contour)
            best_x = int(x + (w / 2))
            best_y = int(y + h)

            image = cv2.rectangle(image, (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0), 2)

            image = cv2.rectangle(image, (best_x, best_y),
                                  (best_x + 2, best_y + 2),
                                  (0, 0, 255), 2)

            # cv2.putText(image, "Green Colour", (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1.0, (0, 255, 0))

    # cv2.imshow('Blue Detector', blue)  # to display the blue object output
    # image_rgb = image[...,::-1].copy()
    cv2.imwrite("test_pictures.png", image)

    top_left = 0
    top_center = 0
    top_right = 0
    bottom_left = 0
    bottom_center = 0
    bottom_right = 0
    if best_x < 42 and best_y <= 64:
        # print('object is in top left')
        top_left = 1
    if 42 < best_x < 84 and best_y <= 64:
        # print('object is in top center')
        top_center = 1
    if best_x > 84 and best_y <= 64:
        # print('object is in top right')
        top_right = 1
    if best_x < 42 and best_y > 64:
        # print('object is in bottom left')
        bottom_left = 1
    if 42 < best_x < 84 and best_y > 64:
        # print('object is in bottom center')
        bottom_center = 1
    if best_x > 84 and best_y > 64:
        # print('object is in bottom right')
        bottom_right = 1

    return top_left, top_center, top_right, bottom_left, bottom_center, bottom_right


def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.178.66")
    rob = robobo.SimulationRobobo().connect(address='192.168.178.66', port=19997)
    n_hidden_neurons = 10
    allowed_steps = 60
    controller = robotController(n_hidden_neurons)

    # load the best individual from folder DEAP_ES_against_enemies_7_8_for_1_runs.txt
    best_text_file = input("Enter path the best text file: ")
    with open(best_text_file, 'rb') as f:
        best = np.loadtxt(f)

    rob.play_simulation()
    rob.set_phone_tilt(19.6, 1)
    allowed_steps = 60
    steps_taken = 0
    times_near_object = 0
    object_hit = 0
    reward_for_seeing_object = 0
    for i in range(allowed_steps):
        top_left, top_center, top_right, bottom_left, bottom_center, bottom_right = detect_green(rob.get_image_front())
        values = np.array([*rob.read_irs(), top_left, top_center, top_right, bottom_left, bottom_center, bottom_right], float)

        left, right = controller.control(np.nan_to_num(values), best)
        rob.move(left, right, 1000)

        if (rob.check_for_collision()):
            object_hit = 1
            # stop the simulation ones an object is hit
            print('Object is hit')
            break

        if top_left == 1 or top_right == 1:
            reward_for_seeing_object += 0.5

        if bottom_left == 1 or bottom_right == 1 or top_center == 1:
            reward_for_seeing_object += 1

        if bottom_center == 1:
            reward_for_seeing_object += 1.5

        if (detect_object(rob)):
            times_near_object += 0.5

        steps_taken += 1

    print(f"Food collected: {rob.collected_food()}")
    print(f"Reward for seeing object: {reward_for_seeing_object}")
    print(f"Object hit: {object_hit}")
    print(f"Near object penalty: {times_near_object}")
    fitness = 30 * rob.collected_food() + reward_for_seeing_object - 100 * object_hit - times_near_object
    print(f"fitness: {fitness}")

    # Stopping the simualtion resets the environment
    rob.stop_world()
    rob.wait_for_stop()


if __name__ == "__main__":
    main()