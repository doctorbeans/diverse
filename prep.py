#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:00:20 2019

@author: ture.friese
"""
from datetime import timedelta, datetime
import pandas as pd
from pyproj import Proj, transform
from sys import exit
from geopandas import points_from_xy
from shapely.geometry import Point

LONG_LAT = Proj(init='epsg:4326', no_defs=True)
WGS84UTM37S = Proj(init='epsg:21037', no_defs=True)

def prepare(file):
    """ Reads csv file and preprocesses the data """
    df = pd.read_csv(f'data/raw/{file}.csv')
    print('Shape of %s: %s' % (file, df.shape))
    df.columns = df.columns\
        .str.replace(' - Day of Month', 'Dom')\
        .str.replace(r' - Weekday \(Mo = 1\)', 'Wd')\
        .str.replace(' - Time', 'Time')\
        .str.replace(' at ', '')\
        .str.replace(' or Business', '')\
        .str.replace(' in millimeters', '')\
        .str.replace(r'\(KM\)', '')\
        .str.replace('Time from Pickup to Arrival', 'TravelTime')\
        .str.replace(' ', '')
    df['Personal'] = df.Personal.astype('category').cat.codes
    df.drop('VehicleType', axis=1, inplace=True)
    for col in ['OrderNo', 'UserId', 'RiderId']:
        df[col] = df[col].str.extract('(\d+)').astype(int)
    # Convert to datetime
    for col in ['PlacementTime', 'ConfirmationTime', 'ArrivalPickupTime', 'PickupTime', 'ArrivalDestinationTime']:
        if col in df.columns.values.tolist():
            df[col] = pd.to_datetime(df[col], format='%H:%M:%S %p')
    df['PickupHod'] = df.PickupTime.dt.round('60min').dt.hour
    # Convert to seconds since midnight     
    for col in ['PlacementTime', 'ConfirmationTime', 'ArrivalPickupTime', 'PickupTime', 'ArrivalDestinationTime']:
        if col in df.columns.values.tolist():      
            df[col] = df[col].apply(lambda t: timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
            df[col] = df[col].astype(int)
    # Join inn riders data
    ri = pd.read_csv(f'data/raw/Riders.csv')
    ri.columns = ri.columns.str.replace(' ', '')
    ri['RiderId'] = ri['RiderId'].str.extract('(\d+)').astype(int)
    df = pd.merge(df, ri, how='left', on='RiderId')
    # Adding coordinates in wgs84utm37s:
    df['px'], df['py'] = transform(LONG_LAT, WGS84UTM37S, df.PickupLong.values, df.PickupLat.values)
    df['dx'], df['dy'] = transform(LONG_LAT, WGS84UTM37S, df.DestinationLong.values, df.DestinationLat.values)
    df['u'], df['v'] = df.dx - df.px, df.dy - df.py
    df['u0'], df['v0'] = df.DestinationLong - df.PickupLong, df.DestinationLat - df.PickupLat
    df['md'] = (df.u.abs() + df.v.abs()) / 1000 # manhatten distance
    # centre distance
    odeon = (36.824850, -1.283089)
    odeon84 = transform(LONG_LAT, WGS84UTM37S, odeon[0], odeon[1])
    odeon84Pt = Point(odeon84)
    df['distance_odeon'] = [pt.distance(odeon84Pt) for pt in points_from_xy(df.px, df.py)]

    #Save result for later
    df.to_pickle(f'data/prepped/{file}.pkl')
    print(f'Pickled {file}.pkl')
    return df

def main():
    prepare('Train')
    prepare('Test')

if __name__ == '__main__':
    main()
