#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def main():
    signal.signal(signal.SIGINT, terminate_program)

    # rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.208")
    # philip home: 192.168.178.66
    rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)

    rob.play_simulation()

    # Following code moves the robot
    for i in range(1):
            print("robobo is at {}".format(rob.position()))
            rob.move(5, 5, 1000)
            print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
            print("collected food: ", rob.collected_food())
   
    print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # Following code moves the phone stand
    rob.set_phone_pan(0, 100)
    rob.set_phone_tilt(30, 100)
    # time.sleep(1)
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    # # Following code makes the robot talk and be emotional
    # rob.set_emotion('happy')
    # rob.talk('Hi, my name is Robobo')
    # rob.sleep(1)
    # rob.set_emotion('sad')
    #
    # # Following code gets an image from the camera
    image = rob.get_image_front()
    image_rgb = image[...,::-1].copy()
    # IMPORTANT! `image` returned by the simulator is BGR, not RGB
    cv2.imwrite("test_pictures.png", image_rgb)

    time.sleep(0.1)

    # IR reading
    # for i in range(10000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs()))/10))
    #     time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()
    
    # Stopping the simualtion resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
