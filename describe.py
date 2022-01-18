import sys
import math
import numpy as np
import pandas as pd


def percentile(x, percent):
    x.sort()
    k = (len(x)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return x[int(k)]
    d0 = x[int(f)] * (c-k)
    d1 = x[int(c)] * (k-f)
    return d0+d1


def mean(x):
    return sum(i for i in x) / len(x)


def std(x):
    m = mean(x)
    n = len(x)
    return math.sqrt(sum((nb - m)**2 for nb in x) / n)


def min_(x):
    min_ = x[0]
    for elem in x:
        if elem < min_:
            min_ = elem
    return min_


def max_(x):
    max_ = x[0]
    for elem in x:
        if elem > max_:
            max_ = elem
    return max_


def get_numeric_cols(df):
    numeric_cols = []
    for col_name in df.columns:
        try:
            float(df[col_name][0])
            numeric_cols.append(col_name)
        except ValueError:
            continue
    return numeric_cols[1:]


# check arg
if len(sys.argv) != 2:
    print('usage: python <data_set.csv>')
    sys.exit()
if sys.argv[1].endswith('.csv') is False:
    print('usage: python <data_set.csv>')
    sys.exit()

# read file
df = pd.read_csv(sys.argv[1])


# remove lines where a column is nan
num_cols = get_numeric_cols(df)
df = df[num_cols]
for str_ in num_cols:
    df = df[df[str_].notna()]

# get stats unsing numpy
allFeatures = []

for str_ in num_cols:
    featureXX = []
    getMyData = df[str_]
    print(getMyData.describe())
    getMyData = np.array(getMyData)


    featureXX.append(len(getMyData))
    featureXX.append(mean(getMyData))
    featureXX.append(std(getMyData))
    featureXX.append(min_(getMyData))
    featureXX.append(percentile(getMyData, 0.25))
    featureXX.append(percentile(getMyData, 0.5))
    featureXX.append(percentile(getMyData, 0.75))
    featureXX.append(max_(getMyData))
    allFeatures.append(featureXX)

# print accordingly
print('\n{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(''.rjust(15),'Arithmancy'.rjust(15),
    'Astronomy'.rjust(15),'Herbology'.rjust(15),'Def. Ag. DA'.rjust(15),
    'Divination'.rjust(15),'Muggle Stud.'.rjust(15),'Anc. Runes'.rjust(15),
    'Hist. of Mag.'.rjust(15),'Transfigur.'.rjust(15),'Potions'.rjust(15),
    'Care M. Crea.'.rjust(15),'Charms'.rjust(15),'Flying'.rjust(15)))
for i, value in enumerate(['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']):
    print(value.ljust(15), end = '')
    for j in range(13):
        print('{:15.6f}'.format(allFeatures[j][i]), end='')
    print()


