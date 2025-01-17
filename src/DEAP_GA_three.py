from __future__ import print_function

import time
import numpy as np

import robobo_task_three
import cv2
import sys
import signal
from controller_task_three import robotController

from math import fabs, sqrt
import os

# DEAP imports
import random
import math
from deap import base, creator, tools


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


experiment_name = "Results task 3/" + input("Enter name of experiment: ")
while (os.path.exists(experiment_name)):
    experiment_name = input("That name was already chosen, pick another: ")
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

signal.signal(signal.SIGINT, terminate_program)

# rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.178.66")


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
SIGMA = 0.15
# indpb: Independent probability for each attribute to be mutated.
INDPB = 0.1
# tournsize: The number of individuals participating in each tournament.
TOURNSIZE = 8
###########
# DO NOT CHANGE
###########
NGEN = 10
npop = 50
RUNS = 4

controller = robotController(robobo_task_three.SimulationRobobo().connect(address='127.0.0.1', port=19997))

# number of weights for multilayer with 10 hidden neurons
n_vars = (controller.number_of_sensors + 1) * controller.n_hidden_neurons + (
        controller.n_hidden_neurons + 1) * controller.number_of_actions

pop = np.random.uniform(weight_lower, weight_upper, (npop, n_vars))

def simulation(controller, robot):
    print()
    print('Simulation started')
    controller.rob.play_simulation()

    controller.rob.set_phone_tilt(19.8, 1)
    controller.rob.randomize_position()

    allowed_steps = 100
    threshold_not_having_food = 100 # for now, lets do no threshold since we still have to check if it can learn with random food placement
    food_delivered = 0
    initial_distance_food_to_base = controller.getDistance()
    steps_taken = 0
    time_steps_food_in_gripper = 0
    object_hit = 0

    for i in range(allowed_steps):
        controller.makeStep(robot)

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
    fitness += 200 * (1 - relative_distance_food_to_base) * (1 - (steps_taken/allowed_steps))
    # small reward for having food in the gripper for most of the time
    fitness += 50 * (time_steps_food_in_gripper/steps_taken)
    # reduce fitness when object has been hit
    if object_hit == 1:
        fitness = fitness * 0.2
    print(f"fitness: {fitness}")

    controller.rob.stop_world()
    controller.rob.wait_for_stop()

    return fitness


# evaluate the fitness of an individual
def evaluate(robot):
    return (simulation(controller, robot),)


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
        '\n' + str(run) + ' ' + str(generation) + ' ' + str(round(best_fitness, 6)) + ' ' + str(
            round(mean, 6)) + ' ' + str(round(std, 6)))
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


def main():
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('run gen best mean std')
    file_aux.close()
    for run in range(1, RUNS + 1):
        evolution(run)


if __name__ == "__main__":
    main()
