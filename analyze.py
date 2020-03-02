'''
analyze.py

Description:
    Run analysis on the outputs of the search techniques

'''

# --------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

outputDir = './outputs/'


# --------------------------------------------------------------------------------------------------
# Define problems
# --------------------------------------------------------------------------------------------------
# Problems analyzed
problems = []
problems.append("FlipFlop")
problems.append("Knapsack")
problems.append("MaxKColor")
problems.append("Queens")
#problems.append("TravelingSalesman")

# search techniques used
search = []
search.append("RHC")
search.append("SA")
search.append("GA")
search.append("MIMIC")


# --------------------------------------------------------------------------------------------------
# Plotting Functions
# --------------------------------------------------------------------------------------------------

def analyzeRHC(dataFile, problem):
    # read csv file into pandas dataframe
    df = pd.read_csv(dataFile)

    # see which restart values were used
    restarts = df['Restarts'].unique()

    # Get the fitness data for all the restart variations and merge them
    fitness_data = []
    timing_data  = []
    best_fitness = []
    for r in restarts:
        tmp = df.loc[(df['Restarts'] == r)].copy()
        fit = tmp[['Iteration', 'Fitness']].copy()
        best_fitness.append(fit['Fitness'].max())
        fit.rename(columns={'Fitness': 'Restarts ' + str(r)}, inplace = True)
        fitness_data.append(fit)

        time = tmp[['Time']].mean()
        timing_data.append(time[0])

    fitness = reduce(lambda a, b: pd.merge(a, b, on=['Iteration'], how='outer'), fitness_data)

    ax = fitness.plot(x='Iteration')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Best Fitness")
    ax.set_xscale('log')
    title = "RHC Fitness - " + problem
    plt.grid()
    plt.title(title)
    plt.savefig('./outputs/plots/' + problem + '_RHC_fitness.png')

    # plot the timing data
    _, ax = plt.subplots()
    ax.set_xlabel("Restarts")
    ax.set_ylabel("Average Time (s)")
    title = "RHC Timing - " + problem
    plt.plot(restarts, timing_data, 'x-')
    plt.grid()
    plt.title(title)
    plt.savefig('./outputs/plots/' + problem + '_RHC_timing.png')

    return best_fitness, timing_data


def analyzeSA(dataFile, problem):
    # read csv file into pandas dataframe
    df = pd.read_csv(dataFile)

    # see which temperatures were used
    temperatures = df['Temperature'].unique()

    # Get the fitness data for all temperatures and merge them
    fitness_data = []
    timing_data  = []
    best_fitness = []
    for t in temperatures:
        tmp = df.loc[(df['Temperature'] == t)].copy()
        fit = tmp[['Iteration', 'Fitness']].copy()
        best_fitness.append(fit['Fitness'].max())
        fit.rename(columns={'Fitness': 'Temperature ' + str(t)}, inplace=True)
        fitness_data.append(fit)

        time = tmp[['Time']].mean()
        timing_data.append(time[0])

    fitness = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), fitness_data)

    # Fitness plot
    ax = fitness.plot(x='Iteration')
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.7, pos.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.28, 1.0), ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    #ax.set_xscale('log')
    title = "SA Fitness - " + problem
    plt.title(title)
    plt.grid()
    plt.savefig('./outputs/plots/' + problem + '_SA_fitness.png')

    # Timing plot
    _, ax = plt.subplots()
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Average Time Taken (s)")
    ax.set_xscale('log')
    title = "SA Timing - " + problem
    plt.title(title)
    plt.grid()
    plt.plot(temperatures, timing_data, 'x-')
    plt.savefig('./outputs/plots/' + problem + '_SA_timing.png')

    return best_fitness, timing_data


def analyzeGA(dataFile, problem):
    # read csv file into pandas dataframe
    df = pd.read_csv(dataFile)

    # see which population sizes and mutation rates were used
    populations = df['Population Size'].unique()
    mutations   = df['Mutation Rate'].unique()

    # Get the fitness data for all temperatures and merge them
    fitness_data = []
    best_fitness = []
    timing_data  = []

    for p in populations:
        tmpTimes = []
        tmpBestFit = []
        for m in mutations:
            tmp = df.loc[(df['Population Size'] == p)].copy()
            tmp = tmp.loc[(tmp['Mutation Rate'] == m)].copy()
            fit = tmp[['Iteration', 'Fitness']].copy()
            bestFit = fit[['Fitness']].max()[0]
            fit.rename(columns={'Fitness': 'Population ' + str(p) + ' Mutation ' + str(m)}, inplace=True)

            tmpBestFit.append(bestFit)
            fitness_data.append(fit)


            time = tmp[['Time']].mean()
            tmpTimes.append(time[0])
        timing_data.append(tmpTimes)
        best_fitness.append(tmpBestFit)

    fitness = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), fitness_data)

    # Fitness plot
    ax = fitness.plot(x='Iteration')
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.6, pos.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 1.0), ncol=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness")
    title = "GA Fitness - " + problem
    plt.title(title)
    plt.grid()
    plt.savefig('./outputs/plots/' + problem + '_GA_fitness.png')

    # Fitness wireframe plot
    x, y = np.meshgrid(populations, mutations)
    z = np.array(best_fitness)
    fig = plt.figure()
    ax = Axes3D(fig)
    # interpolate for a finer grid
    f = interp2d(x, y, z, kind='linear')
    x_new = np.arange(min(populations), max(populations), (max(populations)-min(populations))/100)
    y_new = np.arange(min(mutations), max(mutations), (max(mutations)-min(mutations))/100)
    z_new = f(x_new, y_new)
    Xn, Yn = np.meshgrid(x_new, y_new)
    surf = ax.plot_surface(Xn, Yn, z_new, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Fitness')
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
    title = "GA Fitness - " + problem
    plt.title(title)
    plt.savefig('./outputs/plots/' + problem + '_GA_fitness_wireframe.png')

    # Timing surface plot
    x, y = np.meshgrid(populations, mutations)
    z = np.array(timing_data)
    fig = plt.figure()
    ax = Axes3D(fig)
    # interpolate data for a finer grid
    f = interp2d(x, y, z, kind='linear')
    x_new = np.arange(min(populations), max(populations), (max(populations)-min(populations))/100)
    y_new = np.arange(min(mutations), max(mutations), (max(mutations)-min(mutations))/100)
    z_new = f(x_new, y_new)
    Xn, Yn = np.meshgrid(x_new, y_new)
    surf = ax.plot_surface(Xn, Yn, z_new, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Time (s)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title = "GA Timing - " + problem
    plt.title(title)
    plt.savefig('./outputs/plots/' + problem + '_GA_timing_surface.png')

    return best_fitness, timing_data


def analyzeMIMIC(dataFile, problem):
    df = pd.read_csv(dataFile)
    populations = df['Population Size'].unique()
    keep_pct    = df['Keep Percent'].unique()

    fitness_data = []
    best_fitness = []
    timing_data  = []

    for p in populations:
        tmpTimes = []
        tmpBestFit = []
        for k in keep_pct:
            tmp = df.loc[(df['Population Size'] == p)].copy()
            tmp = tmp.loc[(tmp['Keep Percent'] == k)].copy()
            fit = tmp[['Iteration', 'Fitness']].copy()
            bestFit = fit[['Fitness']].max()[0]
            fit.rename(columns={'Fitness': 'Population ' + str(p) + ' Keep Percent ' + str(k)}, inplace=True)

            tmpBestFit.append(bestFit)
            fitness_data.append(fit)


            time = tmp[['Time']].mean()
            tmpTimes.append(time[0])
        timing_data.append(tmpTimes)
        best_fitness.append(tmpBestFit)

    fitness = reduce(lambda left, right: pd.merge(left, right, on=['Iteration'], how='outer'), fitness_data)

    # Fitness plot
    x, y = np.meshgrid(populations, keep_pct)
    z = np.array(best_fitness)

    fig = plt.figure()
    ax = Axes3D(fig)
    f = interp2d(x, y, z, kind='linear')
    x_new = np.arange(min(populations), max(populations), (max(populations)-min(populations))/100)
    y_new = np.arange(min(keep_pct), max(keep_pct), (max(keep_pct)-min(keep_pct))/100)
    z_new = f(x_new, y_new)
    Xn, Yn = np.meshgrid(x_new, y_new)
    surf = ax.plot_surface(Xn, Yn, z_new, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Keep Percent')
    ax.set_zlabel('Fitness')
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
    if problem == "MaxKColor":
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    title = "MIMIC Fitness - " + problem
    plt.title(title)
    plt.savefig('./outputs/plots/' + problem + '_MIMIC_fitness_surface.png')

    # Timing plot
    x, y = np.meshgrid(populations, keep_pct)
    z = np.array(timing_data)
    fig = plt.figure()
    ax = Axes3D(fig)
    f = interp2d(x, y, z, kind='linear')
    x_new = np.arange(min(populations), max(populations), (max(populations)-min(populations))/100)
    y_new = np.arange(min(keep_pct), max(keep_pct), (max(keep_pct)-min(keep_pct))/100)
    z_new = f(x_new, y_new)
    Xn, Yn = np.meshgrid(x_new, y_new)
    surf = ax.plot_surface(Xn, Yn, z_new, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=False)
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Keep Percent')
    ax.set_zlabel('Time (s)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title = "MIMIC Timing - " + problem
    plt.title(title)
    plt.savefig('./outputs/plots/' + problem + '_MIMIC_timing_surface.png')

    return best_fitness, timing_data

# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------

scores = []
for prob in problems:
    print('\t', prob)

    fit = []
    times = []
    for technique in search:
        print('\t\t', technique)

        rootFile = '__' + prob + '_' + technique
        resultsDir  = outputDir + prob + '_' + technique + '/'
        if technique == "RHC":
            resultsFile = resultsDir + "rhc" + rootFile + '__run_stats_df.csv'
            fit, times = analyzeRHC(resultsFile, prob)

        elif technique == "SA":
            resultsFile = resultsDir + "sa" + rootFile + '__run_stats_df.csv'
            fit, times = analyzeSA(resultsFile, prob)

        elif technique == "GA":
            resultsFile = resultsDir + "ga" + rootFile + '__run_stats_df.csv'
            fit, times = analyzeGA(resultsFile, prob)

        elif technique == "MIMIC":
            resultsFile = resultsDir + "mimic" + rootFile + '__run_stats_df.csv'
            fit, times = analyzeMIMIC(resultsFile, prob)

        fit = np.array(fit)
        times = np.array(times)

        # Get best fitness score
        bestFit = np.amax(fit)

        # get the minimum time the best fitness score was realized
        bestTime = times[np.where(fit == bestFit)].min()

        print('\t\t\tFitness: %d' % bestFit)
        print('\t\t\tTime:    %.2f' % bestTime)

        scores.append([prob, technique, bestFit, bestTime])


# write data to a csv file
file = open('outputs/results.csv', '+w')
file.write('Problem, Technique, Best Fit, Best Time')
file.write('\n')
for val in scores:
    file.write(str(val[0]))
    file.write(',')
    file.write(str(val[1]))
    file.write(',')
    file.write(str(val[2]))
    file.write(',')
    file.write(str(val[3]))
    file.write('\n')

