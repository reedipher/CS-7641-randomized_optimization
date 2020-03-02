'''
part2.py

Description:
    Use 3 Randomized Search Techniques to find good weights for a neural network


'''

# --------------------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------------------
from NN import NN
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
'''
import sklearn.model_selection as ms
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from collections import Counter
'''
from sklearn.preprocessing import normalize


# --------------------------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------------------------
def calculate_null_accuracy(y):
    _, counts = np.unique(y, return_counts=True)
    
    norm = counts / np.sum(counts)
    
    return max(norm)


# --------------------------------------------------------------------------------------------------
# Read in dataset
# --------------------------------------------------------------------------------------------------
wine = pd.read_csv('datasets/wine-quality/winequality-red.csv', sep=';')

# Preprocess the data
bins = (2, 5.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
wine['quality'].unique()

label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])

# separate the dataset as response variable and feature variables
X = wine.drop('quality', axis=1)
y = wine['quality']
labels = ['bad', 'good']

# calculate the null accuracy
wine_null = calculate_null_accuracy(y)
print("Null Accuracy: %.2f %%" % (wine_null*100))

# Split into train and test sets
test_size=0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# scale the data
scale   = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test  = scale.fit_transform(X_test)


# --------------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------------

t0 = time.time()

nn = NN(X_train, Y_train, X_test, Y_test)

nn.optimize('RHC')
nn.optimize('SA')
nn.optimize('GA')

nn.compare()

nn.test_score()

endTime = time.time() - t0
print('\n\nTotal Time: %.3f\n' % endTime)
