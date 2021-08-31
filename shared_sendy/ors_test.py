#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:47:10 2019

@author: trfriese
"""
from openrouteservice import Client, convert
from pprint import pprint

client = Client(key='5b3ce3597851110001cf6248ce885e997fa848f2beb004a02442daff') # Specify your personal API key

res = client.directions(coordinates=[(8.34234,48.23424),(8.34423,48.26424)],
    elevation=True,
    profile='driving-hgv',
    options={'vehicle_type': 'delivery'},
    instructions=True,
    extra_info=['steepness','surface','waycategory','waytype'],

print('Result:')
pprint(res)
geom = res['routes'][0]['geometry']
decoded = convert.decode_polyline(geom)
print('Decoded:')
pprint(decoded)