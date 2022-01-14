import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter

df = pd.read_csv('dataset_train.csv')

houses = ['Gryffindor','Hufflepuff','Ravenclaw','Slytherin']
df = df[['Hogwarts House', 'Astronomy', 'Ancient Runes', 'Herbology', 'Charms']]

# remove lines where a column is nan
for str_ in ['Hogwarts House', 'Astronomy', 'Ancient Runes', 'Herbology', 'Charms']:
    df = df[df[str_].notna()]

dataX = np.array(df[['Astronomy', 'Ancient Runes', 'Herbology', 'Charms']]).reshape(-1, 4)
dataY = np.array(df['Hogwarts House']).reshape(-1, 1)

ds = data_spliter(dataX, dataY, 0.9)
ds0 = copy.deepcopy(ds)
ds1 = copy.deepcopy(ds)
ds2 = copy.deepcopy(ds)
ds3 = copy.deepcopy(ds)

for i in range(2, 4):
    for j in range(len(ds[i])):
        ds0[i][j][0] = 1 if ds[i][j][0] == 'Gryffindor' else 0
for i in range(2, 4):
    for j in range(len(ds[i])):
        ds1[i][j][0] = 1 if ds[i][j][0] == 'Hufflepuff' else 0
for i in range(2, 4):
    for j in range(len(ds[i])):
        ds2[i][j][0] = 1 if ds[i][j][0] == 'Ravenclaw' else 0
for i in range(2, 4):
    for j in range(len(ds[i])):
        ds3[i][j][0] = 1 if ds[i][j][0] == 'Slytherin' else 0

logreg0 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 1e-3, 10000)
logreg0.fit_(ds0[0], ds0[2])
print('log_reg: {:10} trained'.format('Gryffindor'))
logreg1 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 1e-3, 10000)
logreg1.fit_(ds1[0], ds1[2])
print('log_reg: {:10} trained'.format('Hufflepuff'))
logreg2 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 1e-3, 10000)
logreg2.fit_(ds2[0], ds2[2])
print('log_reg: {:10} trained'.format('Ravenclaw'))
logreg3 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 1e-3, 10000)
logreg3.fit_(ds3[0], ds3[2])
print('log_reg: {:10} trained'.format('Slytherin'))

rights = 0.0
wrongs = 0.0
yPred0 = logreg0.predict_(ds0[1])
yPred1 = logreg1.predict_(ds1[1])
yPred2 = logreg2.predict_(ds2[1])
yPred3 = logreg3.predict_(ds3[1])

yPredTotal = []
for i in range(len(yPred0)):
    if np.argmax([yPred0[i], yPred1[i], yPred2[i], yPred3[i]]) == houses.index(ds[3][i][0]):
        rights += 1.0
    else:
        wrongs += 1.0
    yPredTotal.append(np.argmax([yPred0[i], yPred1[i], yPred2[i], yPred3[i]]))
    
print('%% of right answer = ', rights / (rights + wrongs))

plt.show()
