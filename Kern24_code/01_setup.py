"""
Fair Algorithmic Profiling
Setup
"""

import os
#print("Current working directory:", os.getcwd())
os.makedirs("output", exist_ok=True)


# Setup

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# 01 Data Checks

siab = pd.read_csv("./data/siab.csv")

siab.describe(include = 'all')
siab.isna().sum()

grouped = siab.groupby('year')
siab_s = grouped.apply(lambda x: x.sample(n = 5000, random_state = 42)) # Sample 5000 obs from each year

siab_s = siab_s.reset_index(drop = True) # Ungroup

siab_s.groupby('year').describe(include = 'all')

# 02 Data Split

# Train with 2010 - 2015 | 2015
# Test with 2016

siab_train = siab_s[siab_s.year < 2016]
siab_train_s = siab_s[siab_s.year == 2015]
siab_test = siab[siab.year == 2016]

X_train_f = siab_train.iloc[:,4:164]
X_train_fs = siab_train_s.iloc[:,4:164]
X_train_s = X_train_f.drop(columns = ['frau1', 'maxdeutsch1', 'maxdeutsch.Missing.'])
X_train_ss = X_train_fs.drop(columns = ['frau1', 'maxdeutsch1', 'maxdeutsch.Missing.'])
y_train = siab_train.iloc[:, [3]]
y_train_s = siab_train_s.iloc[:, [3]]

X_test_f = siab_test.iloc[:,4:164]
X_test_s = X_test_f.drop(columns = ['frau1', 'maxdeutsch1', 'maxdeutsch.Missing.'])
y_test = siab_test.iloc[:, [3]]

# 03 Descriptive Stats

siab_t = siab_train.copy(deep = True)
#siab_t = siab_t.append(siab_test)
siab_t = pd.concat([siab_t, siab_test], ignore_index=True)


siab_t['nongerman'] = np.where(siab_t['maxdeutsch1'] == 0, 1, 0)
siab_t.loc[siab_t['maxdeutsch.Missing.'] == 1, 'nongerman'] = np.nan
siab_t['nongerman_male'] = np.where((siab_t['nongerman'] == 1) & (siab_t['frau1'] == 0), 1, 0)
siab_t['nongerman_female'] = np.where((siab_t['nongerman'] == 1) & (siab_t['frau1'] == 1), 1, 0)

desc1 = siab_t[['year', 'ltue']].groupby('year').mean()

desc1.to_latex('./output/desc1.tex', float_format = "%.3f") # Mean LTUE over time

desc2a = siab_t[['year', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['year']).agg(['sum', 'count'])
desc2b = siab_t[['year', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['year']).mean()
desc2c = siab_t[['year', 'ltue', 'frau1', 'nongerman', 'nongerman_male', 'nongerman_female']].groupby(['year', 'ltue']).agg(['mean', 'count'])

desc2a.to_latex('./output/desc2a.tex', float_format = "%.3f") # Number of cases over time
desc2b.to_latex('./output/desc2b.tex', float_format = "%.3f") # Socio-demo over time
desc2c.to_latex('./output/desc2c.tex', float_format = "%.3f") # Socio-demo by LTUE over time

# Save 

X_train_f.to_csv('./output/X_train_f.csv', index = False)
X_train_fs.to_csv('./output/X_train_fs.csv', index = False)
X_train_s.to_csv('./output/X_train_s.csv', index = False)
X_train_ss.to_csv('./output/X_train_ss.csv', index = False)
y_train.to_csv('./output/y_train.csv', index = False)
y_train_s.to_csv('./output/y_train_s.csv', index = False)

X_test_f.to_csv('./output/X_test_f.csv', index = False)
X_test_s.to_csv('./output/X_test_s.csv', index = False)
y_test.to_csv('./output/y_test.csv', index = False)
