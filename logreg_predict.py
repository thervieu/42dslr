import os
import sys
import numpy as np
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR

# check args
if len(sys.argv) != 3:
    print('usage: python dataset_test.csv weights.csv')
    sys.exit()
if sys.argv[1] != 'dataset_test.csv':
    print('usage: python dataset_test.csv weights.csv')
    sys.exit()
if sys.argv[2] != 'weights.csv':
    print('usage: python dataset_test.csv weights.csv')
    sys.exit()
if os.path.exists('dataset_test.csv') is False:
    print('please add the dataset_test.csv')
    sys.exit()
if os.path.exists('weights.csv') is False:
    print('please first train the model')
    sys.exit()

# get weights
weights = []
with open(sys.argv[2]) as file:
    lines = file.readlines()
    for line in lines:
        weights.append(np.array(line[:-1].split(',')).reshape(-1, 1))
weights = np.array(weights)

# create log_regs w/ weights
lrs = []
for theta in weights:
    theta = theta.astype('float64')
    lrs.append(MyLR(np.array(theta)))

if os.path.exists('houses.csv'):
    os.remove('houses.csv')
f=open('houses.csv','ab')

# get data
df = pd.read_csv(sys.argv[1])
houses = ['Gryffindor','Hufflepuff','Ravenclaw','Slytherin']
df = df[['Index', 'Astronomy', 'Ancient Runes', 'Herbology', 'Charms']]

for str_ in ['Index', 'Astronomy', 'Ancient Runes', 'Herbology', 'Charms']:
    df = df[df[str_].notna()]

# get x for prediction
dataX = np.array(df[['Astronomy', 'Ancient Runes', 'Herbology', 'Charms']]).reshape(-1, 4)

# predict and write to houses.csv
col_names = np.array([['Index','Hogwarts House']])
np.savetxt(f, col_names, fmt='%s', delimiter=',')
for i in range(len(df['Index'])):
    preds = []
    for lr in lrs:
        preds.append(lr.predict_(dataX[i].reshape(-1, 4)))
    np.savetxt(f, np.array([[i, houses[np.argmax(preds)]]]), fmt='%s', delimiter=',')
f.close()
