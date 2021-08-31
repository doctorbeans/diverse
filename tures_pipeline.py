#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All the transformers I use in my pipeline.
Created on Fri Nov 22 09:02:17 2019

@author: ture.friese
"""
from scipy.stats import zscore
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from hdbscan import HDBSCAN, approximate_predict
from sklearn.decomposition import PCA

class Rotater(BaseEstimator, TransformerMixin):
    """Transforms x,y coordinates"""
    def __init__(self):
         self.pca = PCA(n_components=2)
        
    def fit(self, X, y=None, **args):
        """Fits PCA"""
        xy = pd.DataFrame(np.array([
            X['px'].values.tolist() + X['dx'].values.tolist(),
            X['py'].values.tolist() + X['dy'].values.tolist()
        ]).T, columns=['x', 'y'])
        self.pca.fit(xy)
        
    def transform(self, X, y=None, **args):
        """Add transformed features"""
        X = X.copy()
        pxyt = self.pca.transform(X[['px', 'py']]).T
        dxyt = self.pca.transform(X[['dx', 'dy']]).T
        X['pxt'], X['pyt'] = pxyt[0], pxyt[1]
        X['dxt'], X['dyt'] = dxyt[0], dxyt[1]
        X['ut'], X['vt'] = X.dxt - X.pxt, X.dyt - X.pyt
        X['mdt'] = (X.ut.abs() + X.vt.abs()) / 1000
        return X

    def fit_transform(self, X, y=None, **args):
        """Calls fit and the transform"""
        self.fit(X, y)
        return self.transform(X, y)

class Clusterer(BaseEstimator, TransformerMixin):
    """Clusters with hdbscan algorithm"""
    def __init__(self, columns, name, min_cluster_size=20, use_zscore=True):
        self.min_cluster_size = min_cluster_size
        self.columns= columns
        self.hdbscan = HDBSCAN(self.min_cluster_size, prediction_data=True, core_dist_n_jobs=-1)
        self.name = name
        self.use_zscore = use_zscore

    def fit(self, X, y=None, **args):
        """Calls inner fit method"""
        if self.use_zscore:
            self.hdbscan.fit(zscore(X[self.columns]))
        else:
            self.hdbscan.fit(X[self.columns])
        no_clusters = self.hdbscan.labels_.max()
        outlier_pct = np.count_nonzero(self.hdbscan.labels_ == -1) / float(X.shape[0]) * 100
        #print('Clusterer: Feature %s has %d clusters and %.2f%% outliers' % (self.name, no_clusters, outlier_pct))
        return self
    
    def transform(self, X, y=None, **args):
        """Calls approximate_predict method"""
        X = X.copy()
        if self.use_zscore:
            X[self.name] = approximate_predict(self.hdbscan, zscore(X[self.columns]))[0]
        else:
            X[self.name] = approximate_predict(self.hdbscan, X[self.columns])[0]
        #print('Clusterer out', X.columns)
        return X
    
    def fit_transform(self, X, y=None, **args):
        """Calls inner fit_transform method"""
        X = X.copy()
        if self.use_zscore:
            X[self.name] = self.hdbscan.fit_predict(X[self.columns])
        else:
            X[self.name] = self.hdbscan.fit_predict(zscore(X[self.columns]))
        no_clusters = self.hdbscan.labels_.max()
        outlier_pct = np.count_nonzero(self.hdbscan.labels_ == -1) / float(X.shape[0]) * 100
        #print('Clusterer: Feature %s has %d clusters and %.2f%% outliers' % (self.name, no_clusters, outlier_pct))
        return X

class Aggregater(BaseEstimator, TransformerMixin):
    """Takes a cluster (or categorical column) and generates averages"""
    def __init__(self, cluster_col, name, agg_col, func='mean'):
        self.cluster_col=cluster_col
        self.name = name
        self.agg_col = agg_col
        self.temp = None
        self.feature_names = None
        self.func = func
        
    def fit(self, X, y=None, **args):
        Xy = X.copy()
        Xy[y.name] = y
        self.temp = Xy.groupby(by=self.cluster_col)[self.agg_col].agg(self.func).rename(self.name)
        return self
    
    def transform(self, X, y=None, **args):
        X = X.copy()
        X = pd.merge(X, self.temp, how='left', left_on=self.cluster_col, right_index=True)
        #print('Aggregater out', X.columns)
        return X
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects a number of colums as features for ml, filters out the rest"""
    def __init__(self, skipcols=[]):
        self.skipcols = skipcols
        self.feature_matrix = None
        
    def fit(self, X, y=None):
        """Do nothing"""
        return self
    
    def transform(self, X, y=None):
        """Filter columns based on selected_features"""
        skipcols = [col for col in self.skipcols if col in X.columns.values.tolist()]
        X = X.drop(columns=skipcols)
        #print('FeatureSelector out', X.columns)
        return X
    
    def fit_transform(self, X, y=None, **args):
        """This method is called for training but not for testing."""
        print('FeatureSelector – before %d columns' % X.shape[1])
        X = self.transform(X, y)
        self.feature_matrix = X # side output
        #print('FeatureSelector – after %d columns' % X.shape[1])
        return X
        
    
class Imputerr(SimpleImputer):
    """Pandas Wrapper for SimpleImputer"""
    def fit(self, X, y=None):
        """Call super fit with sorted columns"""
        super(Imputerr, self).fit(X)
        return self
      
    def transform(self, X, verbose=False):
        """Call super transform with sorted columns"""
        data = super(Imputerr, self).transform(X)
        X = pd.DataFrame(data=data, columns=X.columns)
        #print('Imputerr out', X.columns)
        return X