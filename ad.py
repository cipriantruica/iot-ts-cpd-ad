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
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# from kneed import KneeLocator
import numpy as np
import matplotlib.dates as md
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
# from pyemma import msm
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
# from thundersvm import OneClassSVM
from mpl_toolkits.mplot3d import Axes3D
from pyod.models.auto_encoder import AutoEncoder
import math
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



class AnomalyDetectionModel:
    def __init__(self, data=None, algorithm='all', max_clusters=20, outliers_fraction = 0.01, anomaly_column=None, category_column=None):
        self.max_clusters=max_clusters
        self.data = data
        self.columns = self.data.columns[1:]
        self.algorithm = algorithm
        self.outliers_fraction = outliers_fraction
        self.anomaly_column = anomaly_column
        self.category_column = category_column

        
    def getD(self, x1, y1, x2, y2, x3, y3):
        return abs((y2-y1)*x3 - (x2-x1)*y3 + x2*y1-x1*y2)/math.sqrt((y2-y1)**2 + (x2-x1)**2)

    def getDistanceByPoint(self, data):
        distance = pd.Series(dtype='float64')
        for i in range(0,len(data)):
            Xa = np.array(data.loc[i])
            Xb = self.model.cluster_centers_[self.model.labels_[i]-1]
            distance.at[i] = np.linalg.norm(Xa-Xb)
        return distance

    def getOptimalClusters(self, data=None):
        if data is None:
            data = self.data[self.columns]
        n_clusters = range(1, self.max_clusters + 1)
        kmeanModels = [KMeans(n_clusters=k).fit(data).fit(data) for k in n_clusters]
        distortions = [sum(np.min(cdist(data, kmeanModels[k].cluster_centers_, 'euclidean'), axis=1)) / data.shape[0] for k in range(len(kmeanModels))]
        dist = {k: self.getD(n_clusters[0], distortions[0], n_clusters[self.max_clusters-1], distortions[self.max_clusters-1], k, distortions[k-1]) for k in n_clusters}
        self.optimalClusters = max(dist, key=dist.get)
        return self.optimalClusters

    def fitPredict(self, data=None):
        if data is None:
            data = self.data[self.columns]
        self.model.fit(data)
        
        yhat = self.model.predict(data)
        return yhat

    def getOptimalComponents(self):
        dataStd = StandardScaler().fit_transform(self.data[self.columns].values)
        # Calculating Eigenvecors and eigenvalues of Covariance matrix
        cov_mat = np.cov(dataStd.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        # Create a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [ (np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        eig_pairs.sort(key = lambda x: x[0], reverse= True)
        # Calculation of Explained Variance from the eigenvalues
        tot = sum(eig_vals)
        # Individual explained variance
        var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
        # Cumulative explained variance
        cum_var_exp = np.cumsum(var_exp)
        
        dist = {k: self.getD(1, var_exp[0], len(var_exp), var_exp[len(var_exp)-1], k, var_exp[k-1]) for k in range(1, len(var_exp)+1)}
        self.optimalComponents = max(dist, key=dist.get)

        return self.optimalComponents

    def getFeatures(self):
        # Take useful feature and standardize them
        dataStd = StandardScaler().fit_transform(self.data[self.columns])        
        dataPCA = pd.DataFrame(dataStd)
        if dataPCA.shape[1] > 1:
            # reduce to 2 important features
            pca = PCA(n_components=self.getOptimalComponents())
            dataPCA = pca.fit_transform(dataPCA)
            # standardize these 2 new features
            scaler = StandardScaler()
            np_scaled = scaler.fit_transform(dataPCA)
            dataPCA = pd.DataFrame(np_scaled)
        
        # add fearures to data
        self.data['cluster'] = self.fitPredict(data=dataPCA)
        # self.data['date_time'] = self.data.index
        self.data.index = dataPCA.index
        self.pca_columns = []
        for i in dataPCA.columns:
            col_name = 'principal_feature_' + str(i)
            self.pca_columns.append(col_name)
            self.data[col_name] = dataPCA[[i]]
        
        return self.data

    def evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        # evaluate the model
        y_hat = model.predict(X_test)
        # evaluate predictions
        mae = mean_absolute_error(y_test, y_hat)
        return mae

    def getAnomaliesKMeans(self):
        self.model = KMeans(n_clusters=self.getOptimalClusters())
        self.data = self.getFeatures()
        # get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
        distance = self.getDistanceByPoint(self.data[self.pca_columns])
        number_of_outliers = int(self.outliers_fraction * len(distance))
        threshold = distance.nlargest(number_of_outliers).min()
        # anomaly contain the anomaly result of the above method Cluster (0:normal, 1:anomaly)
        self.data['anomaly_kmeans'] = (distance >= threshold).astype(int)
        y = self.data['anomaly_kmeans'].values.tolist()
        X = self.data[self.pca_columns].values.tolist()
        print("K-Means MAE: %.3f" % self.evaluate(X, y))
        return self.data

    def getAnomaliesSVM(self):
        self.model = OneClassSVM(nu=self.outliers_fraction, kernel="rbf", gamma=0.01)
        # self.model = OneClassSVM(nu=self.outliers_fraction, kernel="rbf", gamma="scale")
        dataStd = StandardScaler().fit_transform(self.data[self.columns])
        dataStd = pd.DataFrame(dataStd)
        # train isolation forest
        self.data.index = dataStd.index
        self.data['anomaly_svm'] = pd.Series(self.fitPredict(dataStd)).replace([1, -1], [0, 1])
        y = self.data['anomaly_svm'].values.tolist()
        X = self.data[self.columns].values.tolist()
        print("SVM MAE: %.3f" % self.evaluate(X, y))
        return self.data

    def getAnomaliesIF(self):
        self.model = IsolationForest(contamination=self.outliers_fraction, bootstrap=True)
        dataStd = StandardScaler().fit_transform(self.data[self.columns])
        dataStd = pd.DataFrame(dataStd)
        # train isolation forest
        self.data.index = dataStd.index
        self.data['anomaly_isolationforest'] = pd.Series(self.fitPredict(dataStd)).replace([1, -1], [0, 1])
        y = self.data['anomaly_isolationforest'].values.tolist()
        X = self.data[self.columns].values.tolist()
        print("Isolation Forest MAE: %.3f" % self.evaluate(X, y))
        return self.data

    def getAnomaliesAutoEncoder(self):
        self.model = AutoEncoder(hidden_neurons=[len(self.columns), 64, 32, 16, 8, 4, 2, 4, 8, 16, 32, 64, len(self.columns)], epochs=100)
        dataStd = StandardScaler().fit_transform(self.data[self.columns])
        dataStd = pd.DataFrame(dataStd)
        # train isolation forest
        self.data.index = dataStd.index
        self.data['anomaly_autoencoder'] = pd.Series(self.fitPredict(dataStd))#.replace([1, -1], [0, 1])
        y = self.data['anomaly_autoencoder'].values.tolist()
        X = self.data[self.columns].values.tolist()
        print("Auto Encoder MAE: %.3f" % self.evaluate(X, y))
        return self.data

    def trainEllipticEnvelope(self, data):
        envelope =  EllipticEnvelope(contamination = self.outliers_fraction)
        X_train = data.values.reshape(-1,1)
        envelope.fit(X_train)
        data = pd.DataFrame(data)
        data['deviation_gaussian'] = envelope.decision_function(X_train)
        data['anomaly_gaussian'] = envelope.predict(X_train)
        return data

    def getAnomaliesGD(self):
        listDF = []
        if self.category_column is not None:
            for elem in self.data[self.category_column].unique():
                dfcls = self.data[self.columns].loc[self.data[self.category_column] == elem, self.anomaly_column]
                dfcls = self.trainEllipticEnvelope(dfcls)
                listDF.append(dfcls)
        else:
            dfcls = self.data[self.anomaly_column]
            dfcls = self.trainEllipticEnvelope(dfcls)
            listDF.append(dfcls)
            
        df_class = pd.concat(listDF)
        self.data['deviation_gaussian'] = df_class['deviation_gaussian'].replace([1, -1], [0, 1])
        self.data['anomaly_gaussian'] = df_class['anomaly_gaussian'].replace([1, -1], [0, 1])
        y = self.data['anomaly_gaussian'].values.tolist()
        X = self.data[self.columns].values.tolist()
        print("Gaussian MAE: %.3f" % self.evaluate(X, y))
        return self.data

    def plotAnomalies(self, algorithm, save=True, ofn=None):
        #visualize anomaly
        fig, ax = plt.subplots()
        a = self.data.loc[self.data['anomaly' + '_' + algorithm] == 1, ['idx', self.anomaly_column]]
        ax.plot(self.data['idx'], self.data[self.anomaly_column], color='blue', label='Normal')
        ax.scatter(a['idx'], a[self.anomaly_column], color='red', label='Anomaly')
        plt.title(algorithm)
        plt.xlabel('Date Time')
        plt.ylabel(self.anomaly_column)
        plt.legend()
        if save:
            if ofn is None:
                ofn = os.path.join('./figures/ad_'+ algorithm + '.png')
            plt.savefig(ofn)

        plt.show()
        plt.close()

    def plotHistogram(self, algorithm):
        # visualize histogram
        a = self.data.loc[self.data['anomaly' + '_' + algorithm] == 0, self.anomaly_column]
        b = self.data.loc[self.data['anomaly' + '_' + algorithm] == 1, self.anomaly_column]
        fig, axs = plt.subplots(figsize=(10,6))
        axs.hist([a,b], bins=32, stacked=True, color=['blue', 'red'])
        plt.show()
        plt.close()

    def getAnomalies(self, visualize=False, save=False, ofn='ad_output.csv'):
        if self.algorithm == 'all':
            self.data = self.getAnomaliesKMeans()
            self.data = self.getAnomaliesSVM()
            self.data = self.getAnomaliesIF()
            self.data = self.getAnomaliesGD()
            self.data = self.getAnomaliesAutoEncoder()
        elif self.algorithm == 'kmeans':
            self.data = self.getAnomaliesKMeans()
        elif self.algorithm == 'svm':
            self.data = self.getAnomaliesSVM()
        elif self.algorithm == "isolationforest":
            self.data = self.getAnomaliesIF()
        elif self.algorithm == "gaussian":
            self.data = self.getAnomaliesGD()
        elif self.algorithm == "autoencoder":
            self.data = self.getAnomaliesAutoEncoder()

        if visualize:
            if self.algorithm == "all":
                for algorithm in ['kmeans', 'svm', 'isolationforest', 'gaussian', 'autoencoder']:
                    self.plotAnomalies(algorithm=algorithm, ofn = ofn + '_' + algorithm + "_" + self.anomaly_column +'.png')
                    self.plotHistogram(algorithm=algorithm)
            else:
                self.plotAnomalies(algorithm=self.algorithm, ofn = ofn + '_' + self.algorithm + "_" + self.anomaly_column +'.png')
                self.plotHistogram(algorithm=self.algorithm)

        if save:
            self.data.to_csv(ofn + ".csv", index=False)

        return self.data

# opt/python-3.7.4/bin/python3.7 adf.py file.csv
# the csv should contain numeric columns and a header
# the first column is a idx
if __name__ == "__main__":
    fn = sys.argv[1]
    df = pd.read_csv(fn, sep=',')

    ofn = "results_ad"

    df[df.columns[:]] = df[df.columns[:]].apply(pd.to_numeric)

    adm = AnomalyDetectionModel(data=df, algorithm='all', anomaly_column=df.columns[-1])
    adm.getAnomalies(visualize=True, save=True, ofn=ofn)