import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean(x):
    return sum(i for i in x) / len(x)


def std_not_div(x):
    m = mean(x)
    return math.sqrt(sum((nb - m)**2 for nb in x) )


def coef_pearson(x1, x2):
    sum_ = 0
    m1 = mean(x1)
    m2 = mean(x2)
    for i in range(len(x1)):
        sum_ += (x1[i] - m1) * (x2[i] - m2)
    return sum_ / (std_not_div(x1) * std_not_div(x2))

def minmax(x):
    if isinstance(x, np.ndarray) is False:
        return None
    if x.size == 0:
        return None
    maximum = np.max(x)
    minimum = np.min(x)
    return (x - minimum) / (maximum - minimum)

# get data
df = pd.read_csv('dataset_train.csv')
df = df[['Astronomy', 'Defense Against the Dark Arts']]
for str_ in ['Astronomy', 'Defense Against the Dark Arts']:
    df = df[df[str_].notna()]

print(coef_pearson(np.array(df['Astronomy']), np.array(df['Defense Against the Dark Arts'])))

plt.figure()
plt.scatter(df['Astronomy'], df['Defense Against the Dark Arts'])
plt.title('correlation')
plt.xlabel('Astronomy')
plt.ylabel('Defense Against the Dark Arts')
plt.show()
