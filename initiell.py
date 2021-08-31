#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime as dt
import warnings
from datetime import datetime, timedelta
from math import sqrt
from time import sleep

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from IPython.core.interactiveshell import InteractiveShell
from pandas import DataFrame, Series
from shapely.geometry import Point, Polygon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

warnings.simplefilter(action='ignore', category=FutureWarning)

InteractiveShell.ast_node_interactivity = "all"


# ## Pre-dataloading

# In[ ]:


df_train = pd.read_csv('Train.csv')
df_test = pd.read_csv('Test.csv')
df_riders = pd.read_csv('Riders.csv')

# Join rider info to train/test
df_train = pd.merge(df_train, df_riders, how='left', left_on=['Rider Id'], right_on=['Rider Id'])
df_test = pd.merge(df_test, df_riders, how='left', left_on=['Rider Id'], right_on=['Rider Id'])


# In[ ]:


hexss = pd.read_json('travel_times/540_hexclusters.json')
dict_ = {}
for itm in hexss['features']:
    dict_[int(itm['properties']['MOVEMENT_ID'])] = Polygon(itm['geometry']['coordinates'][0])


# In[ ]:


df_test.columns


# In[ ]:


## Make pickup and destination points
df_train['pickup'] = [Point(row['Pickup Long'], row['Pickup Lat']) for i,row in df_train.iterrows()]
df_train['destination'] = [Point(row['Destination Long'], row['Destination Lat']) for i,row in df_train.iterrows()]

df_test['pickup'] = [Point(row['Pickup Long'], row['Pickup Lat']) for i,row in df_test.iterrows()]
df_test['destination'] = [Point(row['Destination Long'], row['Destination Lat']) for i,row in df_test.iterrows()]


## find the hex shape the points belong to
df_train['pickup_ID'] = df_train['pickup'].apply(lambda x: max([k if x.within(v) else -1 for k,v in dict_.items()]))
df_train['destination_ID'] = df_train['destination'].apply(lambda x: max([k if x.within(v) else -1 for k,v in dict_.items()]))

df_test['pickup_ID'] = df_test['pickup'].apply(lambda x: max([k if x.within(v) else -1 for k,v in dict_.items()]))
df_test['destination_ID'] = df_test['destination'].apply(lambda x: max([k if x.within(v) else -1 for k,v in dict_.items()]))


# ## Time 

# In[ ]:


train_time_col = ['Placement', 'Confirmation', 'Arrival at Pickup', 'Pickup', 'Arrival at Destination']
test_time_col  = ['Placement', 'Confirmation', 'Arrival at Pickup', 'Pickup']
time_cat = ['Day of Month', 'Weekday (Mo = 1) ', 'Time']

cat_col = ['Personal or Business']


# In[ ]:


for col in test_time_col:
    time_col = col + ' - Time'
    
    df_train[time_col] = pd.to_datetime(df_train[time_col])
    df_train[time_col+'_hour'] = df_train[time_col].dt.hour
    df_train[time_col] = df_train[time_col].dt.hour * 60 + df_train[time_col].dt.minute
    
    df_test[time_col] = pd.to_datetime(df_test[time_col])
    df_test[time_col+'_hour'] = df_test[time_col].dt.hour
    df_test[time_col] = df_test[time_col].dt.hour * 60 + df_test[time_col].dt.minute    


# In[ ]:


## Save Pickle dataframes
df_train.to_pickle('df_train.pkl')
df_test.to_pickle('df_test.pkl')


# ## Start here

# In[112]:


#### Read Pickled dataframes ############
df_train = pd.read_pickle('df_train.pkl')
df_test = pd.read_pickle('df_test.pkl')


# In[113]:


# ## Merge the average travel times to the pickup and destination shapes
# tt_hour = pd.read_csv('travel_times/nairobi-hexclusters-2018-4-All-HourlyAggregate.csv')

# print ('Shape before merge:', df_train.shape)
# df_train = pd.merge(df_train, tt_hour
#               , how='left'
#               , left_on=['pickup_ID','destination_ID','Pickup - Time_hour']
#               , right_on=[ 'dstid','sourceid', 'hod']
#               )
# print ('Shape after merge:', df_train.shape)
# print ('Shape before merge:', df_test.shape)
# df_test = pd.merge(df_test, tt_hour
#           , how='left'
#           , left_on=['pickup_ID','destination_ID','Pickup - Time_hour']
#           , right_on=[ 'dstid','sourceid', 'hod']
#          )
# print ('Shape after merge:', df_test.shape)

# del tt_hour


# In[114]:


## Merge the average travel times to the pickup and destination shapes
tt_weekly = pd.read_csv('travel_times/nairobi-hexclusters-2018-4-WeeklyAggregate.csv')

print ('Shape before merge:', df_train.shape)
df_train = pd.merge(df_train, tt_weekly
              , how='left'
              , left_on=['pickup_ID','destination_ID','Pickup - Weekday (Mo = 1)']
              , right_on=[ 'dstid','sourceid', 'dow']
              , suffixes=('_hour', '_week')
         )
print ('Shape after merge:', df_train.shape)
print ('Shape before merge:', df_test.shape)
df_test = pd.merge(df_test, tt_weekly
          , how='left'
              , left_on=['pickup_ID','destination_ID','Pickup - Weekday (Mo = 1)']
              , right_on=[ 'dstid','sourceid', 'dow']
              , suffixes=('_hour', '_week')
         )
print ('Shape after merge:', df_test.shape)

del tt_weekly


# In[117]:


# ## Drop columns

drop_columns = []

drop_columns_train = [  'Arrival at Destination - Time'
    
#     'User Id'
#                       ,'Vehicle Type'
#                       , 'Temperature'
#                       , 'Precipitation in millimeters'
#                       , 'Arrival at Destination - Day of Month'
#                       , 'Arrival at Destination - Weekday (Mo = 1)'
#                      ' *
                      , 'Rider Id'
#                       , 'Platform Type'
                      , 'pickup'
                      , 'destination'
#                       ,  'dow' 
#                       , 'dstid'
#                       , 'sourceid'
#                       , 'Placement - Time'
#                     #  , 'Confirmation - Time'
#                       , 'geometric_standard_deviation_travel_time'
#                     #  , 'Placement - Day of Month'
#                     #  , 'Confirmation - Day of Month'
#                     # , 'Arrival at Pickup - Day of Month'
#                     # ,'Pickup Lat','Pickup Long'
#                     # , 'Destination Lat', 'Destination Long'
                     ]

drop_columns_test = [
#                        'User Id','Vehicle Type'
#                      , 'Temperature'
#                      , 'Precipitation in millimeters'
                      'Rider Id'
#                      , 'Platform Type'
                     , 'pickup'
                     , 'destination'
#                      , 'dow'
#                      , 'dstid' 
#                      , 'sourceid' 
#                      , 'Placement - Time'
#                     # , 'Confirmation - Time'
#                      , 'geometric_standard_deviation_travel_time'
                    # , 'Placement - Day of Month'
                    # , 'Confirmation - Day of Month'
                    # , 'Arrival at Pickup - Day of Month'
                    # , 'Pickup Lat','Pickup Long'
                    # , 'Destination Lat', 'Destination Long'
                    ]

# Rider Id, pickup, destination

df_train = df_train.drop(drop_columns_train, axis=1)
df_test = df_test.drop(drop_columns_test, axis=1)


# ## Categorical

# In[118]:


df_train['Personal or Business'] = df_train['Personal or Business'].astype('category').cat.codes
df_test['Personal or Business'] = df_test['Personal or Business'].astype('category').cat.codes

df_train.set_index('Order No', inplace=True)
df_test.set_index('Order No', inplace=True)


# In[119]:


lable = df_train.pop('Time from Pickup to Arrival')


# ## Pipeline

# In[120]:


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


# In[121]:


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


# In[219]:


class FeatureSelector(BaseEstimator, TransformerMixin):
    """© ture"""
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
        if isinstance(X, DataFrame):
            return X.loc[:, self.mask]
        else:
            return X[:, self.mask]     


# In[182]:


# model = XGBRegressor()

# selected_features=['Personal or Business', 
#                    'Placement - Day of Month',
#                    'Placement - Weekday (Mo = 1)', 
#                    'Confirmation - Day of Month',
#                    'Confirmation - Weekday (Mo = 1)', 
#                    'Confirmation - Time',
#                    'Arrival at Pickup - Day of Month',
#                    'Arrival at Pickup - Weekday (Mo = 1)', 
#                    'Arrival at Pickup - Time',
#                    'Pickup - Day of Month', 
#                    'Pickup - Weekday (Mo = 1)', 
#                    'Pickup - Time',
#                    'Distance (KM)', 
#                    'Destination Lat',
#                    'Destination Long', 
#                    'No_Of_Orders', 
#                    'Age', 
#                    'Average_Rating',
#                    'No_of_Ratings', 
#                    'pickup_ID', 
#                    'destination_ID', 
#                    'Placement - Time_hour',
#                    'Confirmation - Time_hour', 
#                    'Arrival at Pickup - Time_hour',
#                    'Pickup - Time_hour', 
#                    'mean_travel_time',
#                    'standard_deviation_travel_time', 
#                    'geometric_mean_travel_time',
# #                     ,'Pickup Lat', 'Pickup Long'
#                   ]

# preprocess = Pipeline([
#     ('Pickup_clusterer', clusterer(min_cluster_size=60, columns=['Pickup Lat', 'Pickup Long'], name= 'pickup_cluster')),
#     ('Dropoff_clusterer', clusterer(min_cluster_size=60, columns=['Destination Lat', 'Destination Long'], name= 'dropoff_cluster')),
#     ('Trip_clusterer', clusterer(min_cluster_size=10, columns=['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long'], name='trip_cluster')),
#     ('Pickup_avg', averager(cluster_col=['pickup_cluster'], name='pickup_avg', avg_col='geometric_mean_travel_time')), 
#     ('featureSelector', FeatureSelector(all_features=df_train.columns.tolist(), selected_features=selected_features )),
# ])
    
# my_pipeline = Pipeline([
#     ('preprocess',preprocess ),
#     ('Estimator', XGBRegressor(n_jobs=-1))
# ])
    


# In[215]:


df_train.shape


# In[220]:


model = XGBRegressor()

selected_features=['Personal or Business', 
                   'Placement - Day of Month',
                   'Placement - Weekday (Mo = 1)', 
                   'Confirmation - Day of Month',
                   'Confirmation - Weekday (Mo = 1)', 
                   'Confirmation - Time',
#                    'Arrival at Pickup - Day of Month',
#                    'Arrival at Pickup - Weekday (Mo = 1)', 
#                    'Arrival at Pickup - Time',
                   'Pickup - Day of Month', 
                   'Pickup - Weekday (Mo = 1)', 
                   'Pickup - Time',
                   'Distance (KM)', 
                   'Destination Lat',
                   'Destination Long', 
                   'No_Of_Orders', 
                   'Age', 
                   'Average_Rating',
                   'No_of_Ratings', 
                   'pickup_ID', 
                   'destination_ID', 
                   'Placement - Time_hour',
                   'Confirmation - Time_hour', 
                   'Arrival at Pickup - Time_hour',
                   'Pickup - Time_hour', 
#                    'mean_travel_time',
#                    'standard_deviation_travel_time', 
#                    'geometric_mean_travel_time',
# #                     ,'Pickup Lat', 'Pickup Long'
                  ]

my_pipeline = Pipeline([
    ('Pickup_clusterer', clusterer(min_cluster_size=60, columns=['Pickup Lat', 'Pickup Long'], name= 'pickup_cluster')),
    ('Dropoff_clusterer', clusterer(min_cluster_size=60, columns=['Destination Lat', 'Destination Long'], name= 'dropoff_cluster')),
    ('Trip_clusterer', clusterer(min_cluster_size=10, columns=['Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long'], name='trip_cluster')),
    ('Pickup_avg', averager(cluster_col=['pickup_cluster'], name='pickup_avg', avg_col='geometric_mean_travel_time')), 
    ('featureSelector', FeatureSelector(all_features=df_train.columns.tolist(), selected_features=selected_features )),
    ('Estimator', XGBRegressor(n_jobs=-1))
])
    # all_features=df_train.columns.tolist()


# ## Cross-validate

# In[218]:


scores = cross_val_score(my_pipeline, df_train, lable, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
print('Mean Absolute Error %2f' %sqrt((-1 * scores.mean())))


# 762.318764 - 743.164917057135 
# 
# 761.367317 - 744.727784575196
# 
# 761.458746 - 741.294873650194
# 
# 764.110400 - 745.093392694508
# 
# 987.677534, 785.500839, 768.550634, 769.223604

# ## Submit

# In[202]:


# Fit model
pipe = my_pipeline.fit(df_train, lable)


# In[203]:


feature_importance = pd.DataFrame(list(zip(preprocess.named_steps['featureSelector'].selected_features, pipe.named_steps["Estimator"].feature_importances_)), columns=['features', 'importance'])
feature_importance.set_index('features', drop=True, inplace=True)
feature_importance.sort_values(by='importance', ascending=True, inplace=True)
feature_importance.plot(kind='barh', legend=False, grid=True, figsize=[15,7]);


# In[204]:


df_test_ = preprocess.transform(df_test)
explainer = shap.TreeExplainer(pipe.named_steps['Estimator'])
shap_values = explainer.shap_values(df_test_)
shap.summary_plot(shap_values, df_test_, plot_type="bar")


# In[174]:


shap.summary_plot(shap_values, df_test_)


# In[ ]:


for name in df_test_.columns:
    shap.dependence_plot(name, shap_values, df_test_, display_features=df_test_)


# In[185]:


result = my_pipeline.predict(df_test)
ids = df_test.index
submit = pd.DataFrame({'Order_No':ids,'Time from Pickup to Arrival':result.astype(int)}).set_index('Order_No')
submit.to_csv('submit.csv')


# ## Plotting

# In[ ]:


a, b = zip(*hexss['features'][0]['geometry']['coordinates'][0])


# In[ ]:


coords = hexss['features'][0]['geometry']['coordinates'][0]
df_plot = df_train.sample(500)
#plt.hold(True)
#plt.plot(df_plot['Destination Long'], df_plot['Destination Lat'], 'r.')
#plt.plot(df_plot['Pickup Long'], df_plot['Pickup Lat'], 'b.')
plt.plot(a,b)
# Create the map. Save the file to basic_plot.html. _map.html is the default
# if 'path' is not specified
#mplleaflet.show(path=mapfile)
mplleaflet.display()


# ## Grid Search

# In[ ]:


params = {
         'Pickup_clusterer__min_cluster_size': [20, 60, 80],
         'Dropoff_clusterer__min_cluster_size': [20, 60, 80],
#          'Trip_clusterer__min_cluster_size': [10, 20],
         'Estimator__max_depth': [2, 3],
         'Estimator__min_child_weight': [1, 2]
    }

grid = GridSearchCV(estimator=my_pipeline, param_grid= params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
grid.fit(df_train, lable)


# In[ ]:


print ('Best score = %s' %sqrt(-1 * grid.best_score_))
print ('Best paramter combination = ', grid.best_params_)


# In[ ]:


pred = grid.predict(df_test)

print ('rmse score = %s' %sqrt(-1 * pred.best_score_))

