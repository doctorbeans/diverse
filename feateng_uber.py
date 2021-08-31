#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:09:04 2019

@author: ture.friese
"""
import pandas as pd
import geopandas as gpd

# Read train and test data
KEYS = ['PickupLong', 'PickupLat', 'DestinationLong', 'DestinationLat', 'PickupHod', 'PickupWd']
train = pd.read_pickle('data/prepped/Train.pkl')[KEYS].drop_duplicates()
test = pd.read_pickle('data/prepped/Test.pkl')[KEYS].drop_duplicates()
print('Traning and test file', train.shape, test.shape)
df = pd.concat([train, test], axis=0).drop_duplicates()
print('Df shape', df.shape)
print('PickupWd values:', df.PickupWd.value_counts())

# Add shapely columns
df['Pickup'] = gpd.points_from_xy(df.PickupLong, df.PickupLat)
df['Destination'] = gpd.points_from_xy(df.DestinationLong, df.DestinationLat)

# Read hex clusters
h = gpd.read_file('data/downloaded/uber/540_hexclusters.json')

# Join on Pickup
gdf = gpd.GeoDataFrame(df, geometry='Pickup', crs = {'init' :'epsg:4326'})
print(gdf.shape, h.shape)
gdf = gpd.sjoin(gdf, h, how='left', op='intersects')
gdf['pickupId'] = gdf['MOVEMENT_ID']
gdf.drop(columns=['index_right', 'DISPLAY_NAME', 'MOVEMENT_ID'], inplace=True)
print(gdf.shape)

# Join in Destination
gdf.set_geometry('Destination', inplace=True)
print('geometry:', gdf.geometry.name)
gdf = gpd.sjoin(gdf, h, how='left', op='intersects')
gdf['destinationId'] = gdf['MOVEMENT_ID']
gdf.drop(columns=['index_right', 'DISPLAY_NAME', 'MOVEMENT_ID'], inplace=True)
print(gdf.shape)

print('PickupId null', gdf[gdf.pickupId.isnull()].shape)
print('DestionationId null', gdf[gdf.destinationId.isnull()].shape)
# Drop nan rows
gdf.dropna(subset=['pickupId', 'destinationId'], inplace=True)
print('After nan drop', gdf.shape)
gdf['pickupId'] = gdf['pickupId'].astype(int)
gdf['destinationId'] = gdf['destinationId'].astype(int)

uber = pd.DataFrame(gdf)
uber.drop(columns=['Pickup', 'Destination'], inplace=True)
print(uber.shape)
print(uber.columns.values.tolist())

#print(uber.head(1).T)

#Join in other uber frames...
hourly1 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-1-All-HourlyAggregate.csv')
hourly2 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-2-All-HourlyAggregate.csv')
hourly3 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-3-All-HourlyAggregate.csv')
hourly4 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-4-All-HourlyAggregate.csv')
hourly5 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2019-1-All-HourlyAggregate.csv')
hourly6 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2019-2-All-HourlyAggregate.csv')
hourly = pd.concat([hourly1, hourly2, hourly3, hourly4, hourly5, hourly6], axis=0)
hourly['variance_travel_time'] = hourly.standard_deviation_travel_time ** 2
print('hourly', hourly.shape)
print(hourly.columns.values.tolist())

hourly_agg =  hourly.groupby(by=['sourceid', 'dstid', 'hod']).agg({
    'mean_travel_time': 'mean', 
    'variance_travel_time': 'mean'
})
hourly_agg['standard_deviation_travel_time'] = hourly_agg.variance_travel_time **(1/2)
hourly_agg.drop(columns=['variance_travel_time'], inplace=True)

weekly1 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-1-WeeklyAggregate.csv')
weekly2 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-2-WeeklyAggregate.csv')
weekly3 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-3-WeeklyAggregate.csv')
weekly4 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2018-4-WeeklyAggregate.csv')
weekly5 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2019-1-WeeklyAggregate.csv')
weekly6 = pd.read_csv('data/downloaded/uber/nairobi-hexclusters-2019-2-WeeklyAggregate.csv')
weekly = pd.concat([weekly1, weekly2, weekly3, weekly4, weekly5, weekly6], axis=0)
weekly['variance_travel_time'] = weekly.standard_deviation_travel_time ** 2
print('weekly', weekly.shape)
print(weekly.columns.values.tolist())
print('dow values:', weekly.dow.value_counts())

weekly_agg =  weekly.groupby(by=['sourceid', 'dstid', 'dow']).agg({
    'mean_travel_time': 'mean', 
    'variance_travel_time': 'mean'
})
weekly_agg['standard_deviation_travel_time'] = weekly_agg.variance_travel_time **(1/2)
weekly_agg.drop(columns=['variance_travel_time'], inplace=True)

# join aggs with uber
m1 = uber.merge(hourly_agg, how='left', suffixes=['', '_uh'],
    left_on=['pickupId', 'destinationId', 'PickupHod'], 
    right_on=['sourceid', 'dstid', 'hod']
)
m1.rename(columns={'mean_travel_time': 'mean_travelTime_uh', 'standard_deviation_travel_time': 'steddev_travelTime_uh'}, inplace=True)
print('M1', m1.shape)
print(m1.columns.values.tolist())
print('Mean_travelTime_uh nan', m1[m1.mean_travelTime_uh.isnull()].shape)
m2 = m1.merge(weekly_agg, how='left', suffixes=['', '_uw'],
    left_on=['pickupId', 'destinationId', 'PickupWd'], 
    right_on=['sourceid', 'dstid', 'dow']
)
m2.rename(columns={'mean_travel_time': 'mean_travelTime_uw', 'standard_deviation_travel_time': 'steddev_travelTime_uw'}, inplace=True)
print('M2', m2.shape)
print(m2.columns.values.tolist())
print('Mean_travelTime_uw nan', m2[m2.mean_travelTime_uw.isnull()].shape)

m2 = m2[(m2.mean_travelTime_uh.notnull()) | (m2.mean_travelTime_uw.notnull())]
print('M2', m2.shape)

m2.to_pickle('data/prepped/uber_features.pkl')
