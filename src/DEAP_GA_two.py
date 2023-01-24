from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from robot_controller import robotController

from math import fabs, sqrt
import os

# DEAP imports
import random
import math
from deap import base, creator, tools


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

experiment_name = input("Enter name of experiment: ")
while (os.path.exists(experiment_name)):
    experiment_name = input("That name was already chosen, pick another: ")
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

signal.signal(signal.SIGINT, terminate_program)

# rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.178.66")
rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)


########
# Genetic algorithm
########

###########
# DO NOT CHANGE
###########
weight_upper = 1
weight_lower = -1
# cross two individuals with probability CXPB
CXPB = 0.8
# mutate an individual with probability MUTPB
MUTPB = 0.2
# mu: Mean or :term:`python:sequence` of means for the gaussian addition mutation.
MU = 0
# sigma: Standard deviation or :term:`python:sequence` of standard deviations for the gaussian addition mutation.
SIGMA = 0.1
# indpb: Independent probability for each attribute to be mutated.
INDPB = 0.1
# tournsize: The number of individuals participating in each tournament.
TOURNSIZE = 8
###########
# DO NOT CHANGE
###########
NGEN = 10
npop = 40
RUNS = 1

n_hidden_neurons = 10
controller = robotController(n_hidden_neurons)
number_of_sensors = 14
# left wheel and right wheel
number_of_actions = 2

# number of weights for multilayer with 10 hidden neurons
n_vars = (number_of_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 2

pop = np.random.uniform(weight_lower, weight_upper, (npop, n_vars))

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
        print('object is in top left')
        top_left = 1
    if 42 < best_x < 84 and best_y <= 64:
        print('object is in top center')
        top_center = 1
    if best_x > 84 and best_y <= 64:
        print('object is in top right')
        top_right = 1
    if best_x < 42 and best_y > 64:
        print('object is in bottom left')
        bottom_left = 1
    if 42 < best_x < 84 and best_y > 64:
        print('object is in bottom center')
        bottom_center = 1
    if best_x > 84 and best_y > 64:
        print('object is in bottom right')
        bottom_right = 1

    return top_left, top_center, top_right, bottom_left, bottom_center, bottom_right

def simulation(rob, robot):
    print()
    print('Simulation started')
    rob.play_simulation()

    allowed_steps = 50
    steps_taken = 0
    times_near_object = 0
    object_hit = 0
    reward_for_seeing_object = 0
    # position_start = rob.position()
    # position_after = rob.position()
    # times_moved_back = 0
    # intermediate_speed_difference = np.zeros(shape=(allowed_steps))
    # intermediate_distance = np.zeros(shape=(allowed_steps))
    # intermediate_position = np.zeros(shape=(allowed_steps, 3))
    # Following code moves the robot
    for i in range(allowed_steps):
        top_left, top_center, top_right, bottom_left, bottom_center, bottom_right = detect_green(rob.get_image_front())
        values = np.array([*rob.read_irs(), top_left, top_center, top_right, bottom_left, bottom_center, bottom_right], float)

        left, right = controller.control(np.nan_to_num(values), robot)
        # intermediate_speed_difference[i] = abs(left - right)
        rob.move(left, right, 1000)

        if (rob.check_for_collision()):
            object_hit = 1
            # stop the simulation ones an object is hit
            print('Object is hit')
            break

        if top_left == 1 or top_right == 1:
            reward_for_seeing_object += 1

        if bottom_left == 1 or bottom_right == 1 or top_center == 1:
            reward_for_seeing_object += 2

        if bottom_center == 1:
            reward_for_seeing_object += 3

        if (detect_object(rob)):
            times_near_object += 0.5

        steps_taken += 1


    print(f"Food collected: {rob.collected_food()}")
    print(f"Reward for seeing object: {reward_for_seeing_object}")
    print(f"Object hit: {object_hit}")
    print(f"Near object penalty: {times_near_object}")
    fitness = 20 * rob.collected_food() + reward_for_seeing_object - 80 * object_hit - times_near_object
    print(f"fitness: {fitness}")

    rob.stop_world()
    rob.wait_for_stop()

    return fitness


# evaluate the fitness of an individual
def evaluate(robot):
    return (simulation(rob, robot),)

def getStatistics(run, generation, pop, fits):
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print()
    print(f"Statistics of run {run} and generation {generation}")
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)

    # saves results
    # get best invididual fitness
    best_ind = tools.selBest(pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    # get best individual values

    # saves results for first pop
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write(
        '\n' + str(run) + ' ' + str(generation) + ' ' + str(round(best_fitness, 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
    file_aux.close()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, weight_lower, weight_upper)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=MU, sigma=SIGMA, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

def evolution(run):
    pop = toolbox.population(n=npop)

    print(f"Start of evolution {run}")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, np.array(pop)))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]
    getStatistics(run, 0, pop, fits)

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(1, NGEN):
        print()
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, np.array(invalid_ind))
        # zip() lets you iterate over both lists, and stops when the shortest one ends
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        getStatistics(run, g, pop, fits)

        # saves file with the best solution
        best_ind = tools.selBest(pop, 1)[0]
        print(best_ind)
        np.savetxt(experiment_name + f"/best-{run}.txt", best_ind)

        # saves simulation state
        solutions = [pop, fits]

    print("-- End of (successful) evolution --")
    # save individuals
    # return pop, fits
    # for robot in pop:
    #     simulation(rob, robot)

def main():
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('run gen best mean std')
    file_aux.close()
    for run in range(1, RUNS + 1):
        evolution(run)

if __name__ == "__main__":
    main()
