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
import numpy as np
import sys

def decision(sup_ad, sup_cpd, conf):
    if (sup_ad >= 0.75 and (sup_cpd == 0 or sup_cpd <= 0.4)) or ((sup_ad == 0 or sup_ad <= 0.25) and sup_cpd >= 0.6) or (sup_ad == 0 and sup_cpd <= 0.25) or (sup_ad != 0 and sup_cpd == 0) or (sup_ad == 0 and sup_cpd != 0):
        return 'automatic'
    if (sup_ad <= 0.25 and (sup_cpd == 0 or sup_cpd <= 0.25)) or (sup_ad >= 0.75 and sup_cpd == 1) or ((sup_ad == sup_cpd) or (conf >= 0.25 and conf<=0.5)):
        return 'human'


if __name__ == "__main__":
    ad_file = sys.argv[1]
    cpd_file = sys.argv[2]
    out_file = "decision.csv"

    df_ad = pd.read_csv(ad_file, sep=',')
    df_cpd = pd.read_csv(cpd_file, sep=',')

    columns_ad = ["idx", "anomaly_kmeans", "anomaly_svm", "anomaly_isolationforest", "anomaly_gaussian", "anomaly_autoencoder"]
    columns_cpd = ["idx", "PELT", "BottomUp", "Window", "BinSeg", "DynProg"]

    df_ad = df_ad[columns_ad]
    df_cpd = df_cpd[columns_cpd]

    df_merge = pd.merge(df_ad, df_cpd, on='idx')

    
    df_merge["support_ad"] = df_merge[columns_ad[2:]].sum(axis=1) / 5
    df_merge["support_cpd"] = df_merge[columns_cpd[1:]].sum(axis=1) / 5
    df_merge["confidence"] = df_merge["support_ad"] / (df_merge["support_ad"] + df_merge["support_cpd"])

    # df_merge["confidence"] = df_merge.apply(lambda row: confidence(row["support_ad"], row["support_cpd"]), axis=1)
    df_merge["decision"] = df_merge.apply(lambda row: decision(row["support_ad"], row["support_cpd"], row["confidence"]), axis=1)
    df_merge.to_csv(out_file, index=False)
    print(df_merge)