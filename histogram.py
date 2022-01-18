import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_numeric_cols(df):
    numeric_cols = []
    for col_name in df.columns:
        try:
            float(df[col_name][0])
            numeric_cols.append(col_name)
        except ValueError:
            continue
    return numeric_cols[1:]


def mean(x):
    return sum(i for i in x) / len(x)


def std(x):
    m = mean(x)
    n = len(x)
    return math.sqrt(sum((nb - m)**2 for nb in x) / n)


def minmax(x):
    if isinstance(x, np.ndarray) is False:
        return None
    if x.size == 0:
        return None
    maximum = np.max(x)
    minimum = np.min(x)
    return (x - minimum) / (maximum - minimum)


def get_grades(df, house_name, col):
    dfGrades = df[df['Hogwarts House'] == house_name][col]
    return dfGrades

# get data
df = pd.read_csv('dataset_train.csv')
num_cols = get_numeric_cols(df)
for str_ in num_cols:
    df = df[df[str_].notna()]

# print each histo and normalized data
for col in num_cols:
    data_col = np.array(df[col])
    print('{:15.15s}: normalized std = {}'.format(col, std(minmax(data_col))))
    plt.figure()
    plt.hist(get_grades(df, "Gryffindor", col), bins=30, alpha=0.5, label = 'Gry', color = 'r')
    plt.hist(get_grades(df, "Ravenclaw", col), bins=30, alpha=0.5, label = 'Rav', color = 'b')
    plt.hist(get_grades(df, "Slytherin", col), bins=30, alpha=0.5, label = 'Sly', color = 'g')
    plt.hist(get_grades(df, "Hufflepuff", col), bins=30, alpha=0.5, label = 'Huf', color = 'y')
    plt.legend(loc = 'upper right')
    plt.title(col)

plt.show()
