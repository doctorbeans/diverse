#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import datetime as dt
import warnings
from datetime import datetime, timedelta
from math import sqrt
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame, Series
from shapely.geometry import Point, Polygon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, HuberRegressor, Lasso,
                                  LinearRegression, Ridge, SGDRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from xgboost import XGBRegressor

import hdbscan
import mplleaflet
import shap
from catboost import CatBoostRegressor, Pool
from IPython.core.interactiveshell import InteractiveShell

warnings.simplefilter(action='ignore', category=FutureWarning)
InteractiveShell.ast_node_interactivity = "all"


class clusterer(BaseEstimator, TransformerMixin):
    """Clusters gtfs positions with hdbscan algorithm. Calls approximate_predict from transform.
       Copied from tidsmakin longterm pipeline """
    def __init__(self, min_cluster_size, columns, name):
        self.min_cluster_size = min_cluster_size
        self.columns= columns
        self.cluster = hdbscan.HDBSCAN(self.min_cluster_size, prediction_data=True, core_dist_n_jobs=-1)
        self.name = name
        self.feature_names = None

    def fit(self, X, y=None, **args):
        """Calls inner fit method. Requires X to have columns 'lon' and 'lat'."""
        self.cluster.fit(X[self.columns])
        return self
    
    def transform(self, X, y=None, **args):
        """Calls approximate_predict method. Requires X to have columns 'lon' and 'lat'."""
        X = X.copy()
        X[self.name] = hdbscan.approximate_predict(self.cluster, X[self.columns])[0]
        return X
    
    def fit_transform(self, X, y=None, **args):
        """Calls inner fit_transform method. Requires X to have columns 'lon' and 'lat'."""
        X = X.copy()
        X[self.name] = self.cluster.fit_predict(X[self.columns])
        return X

class averager(BaseEstimator, TransformerMixin):
    """takes a cluster and cerates averages"""
    
    def __init__(self, cluster_col, name, avg_col):
        self.cluster_col=cluster_col
        self.name = name
        self.avg_col = avg_col
        self.temp = None
        self.feature_names = None
        
    def fit(self, X, y=None, **args):
        X = X.copy()
        self.temp = X.groupby(by=self.cluster_col)[self.avg_col].mean().rename(self.name)
        return self
    
    def transform(self, X, y=None, **args):
        X = X.copy()
        X = pd.merge(X, self.temp, how='left', left_on=self.cluster_col, right_index=True)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, all_features=None, selected_features=None):
        """Constructor"""
        self.all_features = all_features
        self.selected_features = selected_features
        self.mask = np.isin(all_features, selected_features)
        
    def fit(self, X, y=None):
        """Do nothing"""
        return self
    
    def transform(self, X, y=None):
        """Filter columns based on selected_features"""
        return X[self.selected_features]


# In[ ]:




