import pandas as pd
import seaborn as sns
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

# get data
df = pd.read_csv('dataset_train.csv')
num_cols = get_numeric_cols(df)

num_cols.remove('Defense Against the Dark Arts') # same as Astronomy
num_cols.remove('Care of Magical Creatures') # too homogenous
num_cols.remove('Arithmancy') # too homogenous
num_cols.append('Hogwarts House')

df = df[num_cols]
# remove lines where a column is nan
for str_ in num_cols:
    df = df[df[str_].notna()]

sns.pairplot(df, hue='Hogwarts House')
plt.show()
