#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:47:10 2019

@author: trfriese
"""
import time
from openrouteservice import Client
import pandas as pd
from itertools import cycle
from openrouteservice.exceptions import ApiError

KEYS = ['PickupLong', 'PickupLat', 'DestinationLong', 'DestinationLat']
clients = [
	#Client(key='5b3ce3597851110001cf6248ce885e997fa848f2beb004a02442daff'),
	#Client(key='5b3ce3597851110001cf62488813a2e48eca4516864036ed20fd8a68'),
	#Client(key='5b3ce3597851110001cf62482838dcc2d8d9422ba541dd87c6b37632'),
	#Client(key='5b3ce3597851110001cf6248bbb2d85358b84bdcbec4764bde86fdda'),
    #Client(key='5b3ce3597851110001cf624858f7dcea42194adbb63dfe5f28999fa7')
    #Client(key='5b3ce3597851110001cf624853e24261930148f2b88177bd2ee2c01e')
    Client(key='5b3ce3597851110001cf62481424380cae024fef8464354a683fa3ed')
]

def get_directions(coordinates: list, client: Client):
    """ Call ORS with start and end cordinates and return resposnse """
    time.sleep(.2)
    #response = str(list) + str(client)
    response = client.directions(coordinates=coordinates,
        elevation=True,
        profile='driving-hgv',
        options={'vehicle_type': 'delivery'},
        instructions=True,
        extra_info=['steepness','surface','waycategory','waytype'],
    )
    return response

def filter(df, ors):
    """ Only return rows in df that are not in ors. """
    i1 = df.set_index(KEYS).index
    i2 = ors.set_index(KEYS).index
    filtered = df[~i1.isin(i2)]
    return filtered

train = pd.read_pickle('data/prepped/Train.pkl')[KEYS].drop_duplicates()
test = pd.read_pickle('data/prepped/Test.pkl')[KEYS].drop_duplicates()
print('Traning and test file', train.shape, test.shape)
df = pd.concat([train, test], axis=0).drop_duplicates()
print('Df shape', df.shape)
try:
    ors = pd.read_pickle('data/downloaded/ors.pkl')
    print('Loaded ors file', ors.shape)
except Exception as e:
    print('Unable to load ors.pkl file. Creating a new one.')
    print(e)
    ors = pd.DataFrame(columns=KEYS)

df = filter(df, ors)
print('Filtered', df.shape)

max_requests_pr_minite = 40 * len(clients)
counter = 0
start = time.time()
for (nr, row), client in zip(df.iterrows(), cycle(clients)):
    counter += 1
    print('Row', nr)
    coords = row.values.tolist()
    coords = [tuple(coords[0:2]), tuple(coords[2:])]
    try:
        response = get_directions(coords, client)
        row['response'] = response
        ors = ors.append(row, ignore_index=True)
    except ApiError as a:
        if 'Route could not be found' in str(a.message):
            row['response'] = a.message
            ors = ors.append(row, ignore_index=True)
            print('Route not found for row %d %d/%d. Continuing...' % (nr, counter, df.shape[0]))
        else:
            print('ApiError when requesting ors directions for row %d %d/%d. Aborting!' % (nr, counter, df.shape[0]))
            print(type(a))
            print(a)
            break
    except Exception as e:
        print('Exception when requesting ors directions for row %d %d/%d. Aborting!' % (nr, counter, df.shape[0]))
        print(e)
        break
    if (counter>=1000):
        print('Enough for now.')
        break
    if counter % max_requests_pr_minite == 0:
        elapsed = time.time() - start
        wait_time = 60 - elapsed + 1
        if wait_time > 0:
            print('Waiting %.0fs...' % wait_time)
            time.sleep(wait_time)
        start = time.time()

print(ors.shape)
ors.to_pickle('data/downloaded/ors.pkl')
print('Done.')
