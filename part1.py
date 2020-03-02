'''
part1.py

Description:
    Run search techniques on a variety of problems

Problems:
    8-Queens problem. 8x8 chessboard with 8 queens, goal is to place queens on the chessboard such that none of them
    can attack each other.

TODO - add description of the rest

'''

# --------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------
from mlrose_hiive.generators import QueensGenerator, FlipFlopGenerator, MaxKColorGenerator, TSPGenerator, KnapsackGenerator
from Search import RHC, SA, GA, MIMIC
import numpy as np
import time



# --------------------------------------------------------------------------------------------------
# Define problems
# --------------------------------------------------------------------------------------------------
# Setup all problems
problems = []
mySeed   = 1

# Generate Queens problem
print("\n8-Queens Problem\n")
queens = QueensGenerator.generate(seed=mySeed)

# Generate Flip Flop problem
flipFlop = FlipFlopGenerator.generate(seed=mySeed)

# Generate Max K Color problem
maxK = MaxKColorGenerator.generate(seed=mySeed, max_colors=4)

# Generate Traveling Salesman problem
tsp = TSPGenerator.generate(seed=mySeed, number_of_cities=22)

# Generate Knapsack problem
knapsack = KnapsackGenerator.generate(seed=mySeed, number_of_items_types=10, max_item_count=5)


problems.append(queens)
problems.append(flipFlop)
problems.append(maxK)
problems.append(tsp)
problems.append(knapsack)

# --------------------------------------------------------------------------------------------------
# Search Techniques
# --------------------------------------------------------------------------------------------------

t0 = time.time()
for p in problems:
    # Random Hill Climbing
    restarts = [10, 25, 50, 100]
    rhc = RHC(p, restarts)
    rhc.printInfo()
    rhc.run()

    # Simulated Annealing
    temperatures = 2**np.arange(1,14)
    sa = SA(p, temperatures)
    sa.printInfo()
    sa.run()


    # Genetic Algorithm
    population = [50, 100, 150, 200]
    mutation   = [0.2, 0.3, 0.4, 0.5]
    ga = GA(p, population, mutation)
    ga.printInfo()
    ga.run()


    # MIMIC
    population = [50, 100, 150, 200]
    keep_pct   = 0.01*(2**np.arange(0,6))
    mimic = MIMIC(p, population, keep_pct)
    mimic.printInfo()
    mimic.run()

endTime = time.time() - t0
print('\n\nTotal Time: %.3f\n' % endTime)
