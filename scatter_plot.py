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
    return numeric_cols[1:]

df = pd.read_csv('dataset_train.csv')

# remove lines where a column is nan
df = df[['Astronomy', 'Defense Against the Dark Arts']]

for str_ in ['Astronomy', 'Defense Against the Dark Arts']:
    df = df[df[str_].notna()]

for col in ['Astronomy', 'Defense Against the Dark Arts']:
    plt.figure()
    plt.scatter(df['Astronomy'], df['Defense Against the Dark Arts'])
    plt.title('correlation')
    plt.xlabel('Astronomy')
    plt.ylabel('Defense Against the Dark Arts')
plt.show()
