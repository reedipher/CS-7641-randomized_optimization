'''
NN.py

Description:
    Neural Network class for CS-7614 Assignment 2


'''

# --------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------
import numpy as np
import time
from mlrose_hiive import NeuralNetwork
from mlrose_hiive import GeomDecay, ArithDecay, ExpDecay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# --------------------------------------------------------------------------------------------------
# Neural Network
# --------------------------------------------------------------------------------------------------
class NN:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test  = X_test
        self.Y_test  = Y_test

        #self.iterations = 2**np.arange(0, 12)
        self.iterations = [10, 50, 100, 200]

        # RHC parameters
        self.restarts    = [10, 25, 50]

        # SA parameters
        self.decay = ['Exponential', 'Arithmetic', 'Geometric']

        # GA parameters
        self.populations = [50, 150]
        self.mutations   = [0.2, 0.4]

        self.times = []
        self.max_iteration = []

    def optimize(self, technique):
        algos = []
        legend = []
        train_score = []
        test_score = []
        times = []

        test = 0
        for i in self.iterations:
            algos.append([])
            legend.append([])

            if (technique == 'RHC'):
                for r in self.restarts:
                    print('\t\tRestart %d' % r)
                    nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                       restarts = r, max_iters=i, curve=False)
                algos[test].append(nn)
                legend[test].append('RHC')

            elif (technique == 'SA'):
                nn = NeuralNetwork(hidden_nodes=[20], algorithm='simulated_annealing',
                                   max_iters=i, schedule=ExpDecay(),
                                   curve=False)
                algos[test].append(nn)
                legend[test].append('Exponential Decay')

                nn = NeuralNetwork(hidden_nodes=[20], algorithm='simulated_annealing',
                                 max_iters=i, schedule=ArithDecay(),
                                 curve=False)
                algos[test].append(nn)
                legend[test].append('Arithmetic Decay')

                nn = NeuralNetwork(hidden_nodes=[20], algorithm='simulated_annealing',
                                 max_iters=i, schedule=GeomDecay(),
                                 curve=False)
                algos[test].append(nn)
                legend[test].append('Geometric Decay')

            elif (technique == 'GA'):
                for p in self.populations:
                    for m in self.mutations:
                        nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                           max_iters=i, pop_size=p, mutation_prob=m,
                                           curve=False)
                        algos[test].append(nn)
                        legend[test].append('Population ' + str(p) + ' Mutation ' + str(m))

            test += 1

        for l in range(len(algos[0])):
            train_score.append([])
            test_score.append([])
            times.append([])

            print('\tTrain %d/%d' % (l+1, len(algos[0])))

            for i in range(len(self.iterations)):
                print('\t\t%d/%d' % (i+1, len(self.iterations)))

                train_score[l].append(l)
                test_score[l].append(l)
                times[l].append(l)

                t0 = time.time()
                algos[i][l].fit(self.X_train, self.Y_train)
                t = (time.time() - t0)

                s_train = f1_score(algos[i][l].predict(self.X_train), self.Y_train, average='weighted')
                s_test = f1_score(algos[i][l].predict(self.X_test), self.Y_test, average='weighted')

                train_score[l][i] = (np.mean(s_train))
                test_score[l][i] = (np.mean(s_test))
                times[l][i] = (np.mean(t))

        train_score = np.array(train_score)
        test_score = np.array(test_score)
        times = np.array(times)

        for l in range(len(train_score)):
            plt.plot(self.iterations, train_score[l], label='Training ' + legend[0][l])
            plt.plot(self.iterations, test_score[l], label='Test ' + legend[0][l])

        plt.ylabel('F1 Score')
        plt.xlabel('Iterations')
        plt.title('Score for NN - ' + technique)
        plt.legend()
        plt.show()
        plt.savefig('outputs/plots/part2/' + technique + '_score.png')

        return train_score, test_score, times

    def compare(self):
        algos = []
        legend = []
        train_score = []
        test_score = []
        times = []

        test = 0
        for i in self.iterations:
            algos.append([])
            legend.append([])

            nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                      max_iters=i, schedule=GeomDecay(),
                                      curve=False)
            algos[test].append(nn)
            legend[test].append('SA')
            nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                      max_iters=i, curve=False)
            algos[test].append(nn)
            legend[test].append('RHC')
            nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                      max_iters=i, pop_size=300, mutation_prob=0.2,
                                      curve=False)
            algos[test].append(nn)
            legend[test].append('GA')

            test += 1

        for l in range(len(algos[0])):
            train_score.append([])
            test_score.append([])
            times.append([])

            print('\tCompare %d/%d' % (l+1, len(algos[0])))

            for i in range(len(self.iterations)):
                print('\t\t%d/%d' % (i+1, len(self.iterations)))

                train_score[l].append(l)
                test_score[l].append(l)
                times[l].append(l)

                t0 = time.time()
                algos[i][l].fit(self.X_train, self.Y_train)
                t = (time.time() - t0)

                s_train = f1_score(algos[i][l].predict(self.X_train), self.Y_train, average='weighted')
                s_test  = f1_score(algos[i][l].predict(self.X_test), self.Y_test, average='weighted')

                train_score[l][i] = (np.mean(s_train))
                test_score[l][i] = (np.mean(s_test))
                times[l][i] = (np.mean(t))

        train_score = np.array(train_score)
        times = np.array(time)

        for l in range(len(train_score)):
            plt.plot(self.iterations, train_score[l], label=legend[0][l])

        plt.ylabel('Score')
        plt.xlabel('Iterations')
        plt.title('Comparing Validation F1 Score for All Search Techniques')
        plt.legend()
        plt.show()
        plt.savefig('outputs/plots/part2/f1_score.png')

    def test_score(self):
        algos = []
        legend = []

        nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                  max_iters=100, curve=False)
        algos.append(nn)
        legend.append('RHC')
        nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                  max_iters=100, schedule=GeomDecay(), curve=False)
        algos.append(nn)
        legend.append('SA')
        nn = NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                  max_iters=100, pop_size=300, mutation_prob=0.2, curve=False)
        algos.append(nn)
        legend.append('GA')

        for i, algo in enumerate(algos):
            print('\tTest %d/%d' % (i, len(algos)))
            algo.fit(self.X_train, self.Y_train)
            score = f1_score(algo.predict(self.X_test), self.Y_test, average='weighted')
            print('Testing Score for' + legend[i] + ': ' + str(score))
