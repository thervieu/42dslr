import os
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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

logreg0 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 2e-4, 70000)
theta0 = logreg0.fit_(ds0[0], ds0[2])
print('log_reg: {:10} trained\n'.format('Gryffindor'))
logreg1 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 2e-5, 50000)
theta1 = logreg1.fit_(ds1[0], ds1[2])
print('log_reg: {:10} trained\n'.format('Hufflepuff'))
logreg2 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 2e-5, 50000)
theta2 = logreg2.fit_(ds2[0], ds2[2])
print('log_reg: {:10} trained\n'.format('Ravenclaw'))
logreg3 = MyLR(np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]), 2e-4, 70000)
theta3 = logreg3.fit_(ds3[0], ds3[2])
print('log_reg: {:10} trained\n'.format('Slytherin'))

yPred0 = logreg0.predict_(ds0[1])
yPred1 = logreg1.predict_(ds1[1])
yPred2 = logreg2.predict_(ds2[1])
yPred3 = logreg3.predict_(ds3[1])

yPredTotal = []
yTrue = []
for i in range(len(yPred0)):
    yTrue.append(houses.index(ds[3][i][0]))
    yPredTotal.append(np.argmax([yPred0[i], yPred1[i], yPred2[i], yPred3[i]]))
    
print('percentage of right answers = {}%'.format(accuracy_score(yPredTotal, yTrue) * 100))

theta0 = theta0.reshape(1, 5)
theta1 = theta1.reshape(1, 5)
theta2 = theta2.reshape(1, 5)
theta3 = theta3.reshape(1, 5)

if accuracy_score(yPredTotal, yTrue) >= 0.98:
    if os.path.exists('weights.csv'):
        os.remove('weights.csv')
    print('saving model to weights.csv')
    f=open('weights.csv','ab')
    np.savetxt(f, theta0, fmt='%s', delimiter=',')
    np.savetxt(f, theta1, fmt='%s', delimiter=',')
    np.savetxt(f, theta2, fmt='%s', delimiter=',')
    np.savetxt(f, theta3, fmt='%s', delimiter=',')
    f.close()
else:
    print('model should be >= 0.98 to save, please train again!')