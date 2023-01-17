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


signal.signal(signal.SIGINT, terminate_program)

# rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.178.66")
rob = robobo.SimulationRobobo().connect(address='192.168.178.66', port=19997)

# dom_u = 1
# dom_l = -1
# npop = 5
#
# pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

# create instance of controller class


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
NGEN = 1
npop = 5
RUNS = 1

n_hidden_neurons = 10
controller = robotController(n_hidden_neurons)
number_of_sensors = 8
# left wheel and right wheel
number_of_actions = 2

# number of weights for multilayer with 10 hidden neurons
n_vars = (number_of_sensors + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 2

pop = np.random.uniform(weight_lower, weight_upper, (npop, n_vars))


def simulation(rob, robot):
    rob.play_simulation()

    # Following code moves the robot
    for i in range(10):
        values = np.array(rob.read_irs(), float)

        left, right = controller.control(np.nan_to_num(values), robot)
        print([left, right])
        rob.move(left, right, 1000)
        if (rob.check_for_collision()):
            # stop the simulation ones an object is hit
            print('Object is hit')
            break

    # Stopping the simualtion resets the environment
    rob.stop_world()
    rob.wait_for_stop()
    return 100


# evaluate the fitness of an individual
def evaluate(robot):
    return (simulation(rob, robot),)


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


def main():
    pop = toolbox.population(n=npop)

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, np.array(pop)))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]
    # getStatistics(run, 0, pop, fits)

    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(1, NGEN):
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

        # getStatistics(run, g, pop, fits)

        # saves file with the best solution
        best_ind = tools.selBest(pop, 1)[0]

        # saves simulation state
        solutions = [pop, fits]

    print("-- End of (successful) evolution --")
    # save individuals
    # return pop, fits
    # for robot in pop:
    #     simulation(rob, robot)


if __name__ == "__main__":
    main()
