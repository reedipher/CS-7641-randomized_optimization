'''
Search.py

Description:
Search - Base interface class for randomized search techniques, including:
    - Randomized Hill Climbing (RHC)
    - Simulated Annealing (SA)
    - Genetic Algorithm (GA)
    - MIMIC

'''


from mlrose_hiive.runners import RHCRunner, SARunner, GARunner, MIMICRunner

import numpy as np
import operator
from functools import reduce

#
# Base Search Technique Class
#
class Search:
    '''
    Randomized Search Algorithm Base Class

    Attributes:
        problem:        Optimization problem
        name:           Name of problem
        output:         Output directory
        seed:           Random number seed
        iteration_list: Number of iterations to attempt
    '''
    def __init__(self, prob= "", label = "", outputDir = "./outputs", randomSeed=1, iter_list = 2**np.arange(1,12), attempts = 500):
        self.problem      = prob
        self.name         = label
        self.output       = outputDir
        self.seed         = randomSeed
        self.iterations   = iter_list
        self.attempts     = attempts

        # Name problem based on problem type if it was not given
        if self.name == "":
            probType = str(type(self.problem))

            if "queen" in probType:
                self.name="Queens"
            elif "flop" in probType:
                self.name="FlipFlop"
            elif "color" in probType:
                self.name="MaxKColor"
            elif "tsp" in probType:
                self.name="TravelingSalesman"
            elif "knapsack" in probType:
                self.name="Knapsack"

    def printSearchInfo(self):
        print('\tproblem:       %s' % self.problem)
        print('\tname:          %s' % self.name)
        print('\toutput:        %s' % self.output)
        print('\tseed:          %d' % self.seed)
        print('\titerations:   [', end="")
        print(*self.iterations, sep=",", end="")
        print(']')
        print('\tattempts:      %d' % self.attempts)

    @staticmethod
    def nCr(a, b):
        b = min(b, a-b)
        n = reduce(operator.mul, range(a, a-b, -1), 1)
        d = reduce(operator.mul, range(1, b+1), 1)
        return n / d

#
# Random Hill Climbing
#
class RHC(Search):
    def __init__(self, problem, restarts = [10, 25, 50, 100]):
        Search.__init__(self, problem)
        self.restarts = restarts
        self.name = self.name + '_RHC'

    def printInfo(self):
        print('\nRHC Info:')
        # print base class
        self.printSearchInfo()

        # print class specific data
        print('\trestarts:     [', end="")
        print(*self.restarts, sep=",", end="")
        print(']')

    def run(self):
        runner = RHCRunner(problem = self.problem,
                           experiment_name=self.name,
                           seed = self.seed,
                           iteration_list = self.iterations,
                           restart_list = self.restarts,
                           max_attempts = self.attempts,
                           generate_curves=True, 
                           output_directory=self.output)
        runner.run()


#
# Simulated Annealing
#
class SA(Search):
    def __init__(self, problem, temperatures = 2**np.arange(1,13)):
        Search.__init__(self, problem)
        self.temps = temperatures
        self.name  = self.name + '_SA'

    def printInfo(self):
        print('\nSA Info:')
        # print base class
        self.printSearchInfo()

        # print class specific data
        print('\ttemperatures: [', end="")
        print(*self.temps, sep=",", end="")
        print(']')

    def run(self):
        runner = SARunner(problem = self.problem,
                          experiment_name=self.name,
                          seed = self.seed,
                          iteration_list = self.iterations,
                          temperature_list = self.temps,
                          max_attempts = self.attempts,
                          generate_curves=True, 
                          output_directory=self.output)
        runner.run()




#
# Genetic Algorithm
#
class GA(Search):
    def __init__(self, problem, population_size = [50, 100, 150, 200], mutation_rates = [0.2, 0.4, 0.6, 0.8]):
        Search.__init__(self, problem)
        self.population = population_size
        self.mutation   = mutation_rates
        self.name       = self.name + '_GA'

    def printInfo(self):
        print('\nGA Info:')
        # print base class
        self.printSearchInfo()

        # print class specific data
        print('\tpopulation:   [', end="")
        print(*self.population, sep=",", end="")
        print(']')
        print('\tmutation:     [', end="")
        print(*self.mutation, sep=",", end="")
        print(']')

    def run(self):
        runner = GARunner(problem = self.problem,
                          experiment_name=self.name,
                          seed = self.seed,
                          iteration_list = self.iterations,
                          population_sizes = self.population,
                          mutation_rates = self.mutation,
                          max_attempts = self.attempts,
                          generate_curves=True, 
                          output_directory=self.output)
        runner.run()


#
# MIMIC
#
class MIMIC(Search):
    def __init__(self, problem, population_size = [50, 100, 150, 200], keep_pct = 0.01*(2**np.arange(0,7))):
        Search.__init__(self, problem)
        self.population = population_size
        self.keep_pct   = keep_pct
        self.name       = self.name + '_MIMIC'

    def printInfo(self):
        print('\nMIMIC Info:')
        # print base class
        self.printSearchInfo()

        # print class specific data
        print('\tpopulation:   [', end="")
        print(*self.population, sep=",", end="")
        print(']')
        print('\tkeep_pct:     [', end="")
        print(*self.keep_pct, sep=",", end="")
        print(']')

    def run(self):
        runner = MIMICRunner(problem = self.problem,
                          experiment_name=self.name,
                          seed = self.seed,
                          iteration_list = self.iterations,
                          population_sizes = self.population,
                          keep_percent_list = self.keep_pct,
                          max_attempts = self.attempts,
                          generate_curves=True, 
                          output_directory=self.output)
        runner.run()






#
# Test
#
from mlrose_hiive.generators import QueensGenerator

if __name__ == "__main__":
    # generate problem
    problem = QueensGenerator.generate(seed=1)

    # RHC
    restarts = [1,10,100]
    rhc = RHC(problem, restarts)
    rhc.printInfo()

    # SA
    temps = 2*np.arange(1,13)
    sa = SA(problem, temps)
    sa.printInfo()

    # GA
    population = [50, 150, 250]
    mutation   = [0.2, 0.4, 0.6]
    ga = GA(problem, population, mutation)
    ga.printInfo()

    # MIMIM
    population = [50, 150, 250]
    keep_pct   = [0.02, 0.04, 0.06]
    mimic = MIMIC(problem, population, mutation)
    mimic.printInfo()




    pass

