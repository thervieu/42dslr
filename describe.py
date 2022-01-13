import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    print('usage: python <data_set.csv>')
    sys.exit()
if sys.argv[1].endswith('.csv') is False:
    print('usage: python <data_set.csv>')
    sys.exit()

df = pd.read_csv(sys.argv[1])

allFeatures = []
# remove lines where a column is nan
for str_ in ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']:
    df = df[df[str_].notna()]
# get stats unsing numpy
for str_ in ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']:
    featureXX = []
    getMyData = df[str_]
    featureXX.append(getMyData.count())
    featureXX.append(np.mean(getMyData))
    featureXX.append(np.std(getMyData))
    featureXX.append(np.min(getMyData))
    featureXX.append(np.quantile(getMyData, 0.25))
    featureXX.append(np.quantile(getMyData, 0.5))
    featureXX.append(np.quantile(getMyData, 0.75))
    featureXX.append(np.max(getMyData))
    allFeatures.append(featureXX)

# print accordingly
print('\n{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(''.rjust(15),'Arithm.'.rjust(15),
    'Astronomy'.rjust(15),'Herbology'.rjust(15),'Def. Ag. DA'.rjust(15),
    'Divination'.rjust(15),'Muggle Stud.'.rjust(15),'Anc. Runes'.rjust(15),
    'Hist. of Mag.'.rjust(15),'Transfigur.'.rjust(15),'Potions'.rjust(15),
    'Care M. Crea.'.rjust(15),'Charms'.rjust(15),'Flying'.rjust(15)))
for i, value in enumerate(['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']):
    print(value.ljust(15), end = '')
    for j in range(13):
        print('{:15.6f}'.format(allFeatures[j][i]), end='')
    print()


