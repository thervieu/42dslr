import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
