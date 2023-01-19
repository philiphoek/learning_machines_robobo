#!/usr/bin/env python3
from __future__ import print_function

import time
# Import the necessary libraries to read
# dataset and work on that
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import robobo
import cv2
import sys
import signal
import prey
from robot_controller import robotController




def main():
    # load the best individual from folder DEAP_ES_against_enemies_7_8_for_1_runs.txt
    best_text_file = input("Enter path the result text file: ")
    df = pd.read_csv(best_text_file, sep=" ")
    # Subplots as having two types of quality
    fig, ax = plt.subplots()
    fig.suptitle('Lineplot of fitness per run', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)

    plt.grid()

    for key, group in df.groupby('run'):
        print(f"Generation {key}")
        plt.plot(group['gen'], group['mean'], label=f"Run {key}")
        plt.fill_between(group['gen'], group['mean'] - group['std'], group['mean'] + group['std'], alpha=0.5)
        # plot = group.plot('gen', 'mean',
        #            label=key, ax=ax)
        # plt.fill_between(x, y - error, y + error)

    plt.legend(loc="best")

    plt.show()




if __name__ == "__main__":
    main()