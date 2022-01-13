import sys
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
    return numeric_cols

df = pd.read_csv('dataset_train.csv')

def get_grades(df, house_name, col):
    dfGrades = df[df['Hogwarts House'] == house_name][col]
    return dfGrades

# remove lines where a column is nan
num_cols = get_numeric_cols(df)
for str_ in num_cols:
    df = df[df[str_].notna()]

for col in num_cols[1:]:
    plt.figure()
    plt.hist(get_grades(df, "Gryffindor", col), bins=25, alpha=0.5, label = 'Gry', color = 'r')
    plt.hist(get_grades(df, "Ravenclaw", col), bins=25, alpha=0.5, label = 'Rav', color = 'b')
    plt.hist(get_grades(df, "Slytherin", col), bins=25, alpha=0.5, label = 'Sly', color = 'g')
    plt.hist(get_grades(df, "Hufflepuff", col), bins=25, alpha=0.5, label = 'Huf', color = 'y')
    plt.legend(loc = 'upper right')
    plt.title(col)
plt.show()
