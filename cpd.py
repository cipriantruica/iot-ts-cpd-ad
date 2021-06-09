# coding: utf-8

"""
 *
 * Copyright (C) 2018 Ciprian-Octavian Truică <ciprian.truica@upb.ro>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
"""

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2021, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@upb.ro"
__status__ = "Production"

import pandas as pd
import matplotlib.pyplot as plt 
import sys
import os
import numpy as np
import ruptures as rpt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Binary Segmenetation
# J. Bai. Estimating multiple breaks one at a time. Econometric Theory, 13(3):315–352, 1997.
# P. Fryzlewicz. Wild binary segmentation for multiple change-point detection. The Annals of Statistics, 42(6):2243–2281, 2014. doi:10.1214/14-AOS1245.
#
# Bottom-up segmentation
# Piotr Fryzlewicz. Unbalanced Haar Technique for Nonparametric Function Estimation. Journal of the American Statistical Association, 102(480):1318–1327, 2007. doi:10.1198/016214507000000860.
# E. Keogh, S. Chu, D. Hart, and M. Pazzani. An online algorithm for segmenting time series. In Proceedings of the IEEE International Conference on Data Mining (ICDM), 289–296. 2001.
# 

def evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    # evaluate the model
    y_hat = model.predict(X_test)
    # evaluate predictions
    mae = mean_absolute_error(y_test, y_hat)
    return mae

# opt/python-3.7.4/bin/python3.7 cpd.py file.csv
# the csv should contain numeric columns and a header
if __name__ == "__main__":
    # data file
    fn = sys.argv[1]
    # number of breaking points to search for
    df = pd.read_csv(fn, sep=',')
    # need ResultsTrend 
    ofn = "results_cpd"

    df[df.columns[:]] = df[df.columns[:]].apply(pd.to_numeric)    

    df = df.set_index(df[df.columns[0]])

    column = df.columns[-1]

    X = pd.DataFrame(df, columns = df.columns[1:] )
    indexes =  df.index.values
    
    print("Start PELT")
    algo = rpt.Pelt(model="l2", min_size=3, jump=5)
    result = algo.fit_predict(X, pen=3)
    x = []
    for idx in result[:-1]:
        x.append(indexes[idx])
    n_bkps = len(x)
    y = []
    df['PELT'] = 0
    for idx in x:
        y.append(df.loc[df.index == idx][column].values[0])
        df.at[idx, 'PELT'] = 1

    print("PELT MAE: %.3f" % evaluate(df[[column]].values.tolist(), df['PELT'].values.tolist()))

    plt.plot(df.index, df[column], label='normal')
    plt.scatter(x, y, label='outlier', color='red', marker='o')
    plt.title("Change Finder PELT " + column)
    plt.xlabel('Date Time')
    plt.ylabel(column)
    plt.savefig(ofn + "_PELT_" + column + "_full.png")
    plt.show()
    plt.close()
    print("End PELT")

    print("Start BottomUp")
    algo = rpt.BottomUp(model="l2")
    result = algo.fit_predict(X, n_bkps=n_bkps)
    x = []
    for idx in result[:-1]:
        x.append(indexes[idx])
    y = []
    df['BottomUp'] = 0
    for idx in x:
        y.append(df.loc[df.index == idx][column].values[0])
        df.at[idx, 'BottomUp'] = 1

    print("BottomUp MAE: %.3f" % evaluate(df[[column]].values.tolist(), df['BottomUp'].values.tolist()))

    plt.plot(df.index, df[column], label='normal')
    plt.scatter(x, y, label='outlier', color='red', marker='o')
    plt.title("Change Finder Bottom Up " + column)
    plt.xlabel('Date Time')
    plt.ylabel(column)
    plt.savefig( ofn + "_BottomUp_" + column + "_full.png")
    plt.show()
    plt.close()
    print("End BottomUp")
    
    print("Start Window")
    algo = rpt.Window(model="l2")
    result = algo.fit_predict(X, n_bkps=n_bkps)
    x = []
    for idx in result[:-1]:
        x.append(indexes[idx])
    y = []
    df['Window'] = 0
    for idx in x:
        y.append(df.loc[df.index == idx][column].values[0])
        df.at[idx, 'Window'] = 1

    print("Window MAE: %.3f" % evaluate(df[[column]].values.tolist(), df['Window'].values.tolist()))

    plt.plot(df.index, df[column], label='normal')
    plt.scatter(x, y, label='outlier', color='red', marker='o')
    plt.title("Change Finder Window Segmentation " + column)
    plt.xlabel('Date Time')
    plt.ylabel(column)
    plt.savefig(ofn + "_Window_" + column + "_full.png")
    plt.show()
    plt.close()
    print("End Window")
    
    print("Start BinSeg")
    algo = rpt.Binseg(model="l2")
    result = algo.fit_predict(X, n_bkps=n_bkps)
    x = []
    for idx in result[:-1]:
        x.append(indexes[idx])
    y = []
    df['BinSeg'] = 0
    for idx in x:
        y.append(df.loc[df.index == idx][column].values[0])
        df.at[idx, 'BinSeg'] = 1

    print("BinSeg MAE: %.3f" % evaluate(df[[column]].values.tolist(), df['BinSeg'].values.tolist()))

    plt.plot(df.index, df[column], label='normal')
    plt.scatter(x, y, label='outlier', color='red', marker='o')
    plt.title("Change Finder Binseg " + column)
    plt.xlabel('Date Time')
    plt.ylabel(column)
    plt.savefig(ofn + "_BinarySeg_" + column + "_full.png")
    plt.show()
    plt.close()
    print("End BinSeg")
    
    print("Start DynProg")
    algo = rpt.Dynp(model="l2", min_size=2, jump=5)
    result = algo.fit_predict(X, n_bkps=n_bkps)
    x = []
    for idx in result[:-1]:
        x.append(indexes[idx])
    y = []
    df['DynProg'] = 0
    for idx in x:
        y.append(df.loc[df.index == idx][column].values[0])
        df.at[idx, 'DynProg'] = 1

    print("DynProg MAE: %.3f" % evaluate(df[[column]].values.tolist(), df['DynProg'].values.tolist()))

    plt.plot(df.index, df[column], label='normal')
    plt.scatter(x, y, label='outlier', color='red', marker='o')
    plt.title("Change Finder DynProg " + column)
    plt.xlabel('Date Time')
    plt.ylabel(column)
    plt.savefig(ofn + "_DynProg_" + column + "_full.png")
    plt.show()
    plt.close()
    print("End DynProg")

    df.to_csv(ofn + ".csv", index=False)