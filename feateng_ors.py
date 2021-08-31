#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:35:00 2019

@author: trfriese
"""
from sys import exit
import pandas as pd

ors = pd.read_pickle('data/downloaded/ors.pkl')
ors.reset_index(drop=True, inplace=True)
ors['routes'] = ors.response.apply(lambda dikt: dikt.get('routes'))
nn = ors[ors.routes.notnull()]
flat = pd.io.json.json_normalize(nn.routes.apply(lambda li: li[0]))
# Flange coord-columns back on
flat[['PickupLong', 'PickupLat', 'DestinationLong', 'DestinationLat']] = ors.loc[ors.routes.notnull(), ['PickupLong', 'PickupLat', 'DestinationLong', 'DestinationLat']]

# Waytype
flat['waytype'] = flat['extras.waytypes.summary'].apply(lambda li: {d['value']: (d['distance'], d['amount']) for d in li})
flat['stateroad_dist'] = flat.waytype.apply(lambda d: d.get(1.0, [0.0, 0.0])[0])
flat['stateroad_pct'] = flat.waytype.apply(lambda d: d.get(1.0, [0.0, 0.0])[1]) / 100
flat['road_dist'] = flat.waytype.apply(lambda d: d.get(2.0, [0.0, 0.0])[0])
flat['road_pct'] = flat.waytype.apply(lambda d: d.get(2.0, [0.0, 0.0])[1]) / 100
flat['street_dist'] = flat.waytype.apply(lambda d: d.get(3.0, [0.0, 0.0])[0])
flat['street_pct'] = flat.waytype.apply(lambda d: d.get(3.0, [0.0, 0.0])[1]) / 100

# Surface
z = list(zip(flat.index.values.tolist(), flat['extras.surface.summary'].values.tolist()))
for idx, li in z:
    for d in li:
        d['row'] = idx
f = [li for idx, li in z]
fl = [item for sublist in f for item in sublist] #flatmap
mappings = {
    0.0: 'unknown', 1.0: 'paved', 2.0: 'unpaved',
    3.0: 'asphalt', 4.0: 'concrete', 5.0: 'cobblestone',
    6.0: 'metal', 7.0: 'wood', 8.0: 'compactedGravel',
    9.0: 'fineGravel', 10.0: 'gravel', 11.0: 'dirt',
    12.0: 'ground', 13.0: 'ice', 14.0: 'pavingStones',
    15.0: 'sand', 16.0: 'woodchips', 17.0: 'grass',
    18.0: 'grassPaver',
}
for d in fl:
    d['value'] = mappings[d['value']]
s = pd.DataFrame(data=fl)
s.rename(columns={'value': 'surface', 'amount': 'pct'}, inplace=True)
s = s[['row', 'surface', 'distance', 'pct']]
s['pct'] = s['pct'] / 100
#plotting
#s.groupby(by='row').size().hist()
#s.groupby(by='surface')['pct'].mean().plot(kind='bar')
p = s.pivot(index='row', columns='surface', values=['distance', 'pct'])
p = p.swaplevel(axis=1)
p.columns = ['_'.join(col).strip() for col in p.columns.values]
q = p.fillna(0)
p['propper_surface_distance'] = q.asphalt_distance + q.concrete_distance + q.paved_distance
p['propper_surface_pct'] = q.asphalt_pct + q.concrete_pct + q.paved_pct
p['unmade_surface_distance'] = q.compactedGravel_distance + q.dirt_distance + q.gravel_distance + q.ground_distance + q.pavingStones_distance + q.unpaved_distance
p['unmade_surface_pct'] = q.compactedGravel_pct + q.dirt_pct + q.gravel_pct + q.ground_pct + q.pavingStones_pct + q.unpaved_pct
flat = pd.merge(flat, p, how='left', left_index=True, right_index=True)
nan_cols = flat.columns[flat.isnull().any()].tolist()
# Setting surface columns to 0 when propper_surface_distance is set:
flat.loc[flat.propper_surface_distance.notnull(), ['asphalt_distance', 'concrete_distance', 
    'compactedGravel_distance', 'gravel_distance', 'ground_distance', 'paved_distance', 
    'pavingStones_distance', 'dirt_distance', 'unpaved_distance', 'propper_surface_distance', 
    'unmade_surface_distance', 'asphalt_pct', 'concrete_pct', 'compactedGravel_pct',
    'gravel_pct', 'ground_pct', 'paved_pct', 'pavingStones_pct', 'dirt_pct',
    'unpaved_pct', 'propper_surface_pct', 'unmade_surface_pct']].fillna(0, inplace=True)

#waycategory
z2 = list(zip(flat.index.values.tolist(), flat['extras.waycategory.summary'].values.tolist()))
for idx, li in z2:
    for d in li:
        d['row'] = idx
f2 = [li for idx, li in z2]
fl2 = [item for sublist in f2 for item in sublist] #flatmap
mappings2 = {
    0.0: 'No category', 1.0: 'Highway', 2.0: 'Steps',
    4.0: 'Ferry', 8.0: 'Unpaved road', 16.0: 'Track',
    32.0: 'Tunnel', 64.0: 'Paved road', 128.0: '	Ford',
}
s2 = pd.DataFrame(data=fl2)
s2.rename(columns={'value': 'waycategory', 'amount': 'pct'}, inplace=True)
s2 = s2[['row', 'waycategory', 'distance', 'pct']]
s2['pct'] = s2['pct'] / 100
s2.waycategory.value_counts()
# --> Waycategory is useless

#Steps
flat['no_steps'] = flat.segments.apply(lambda li: len(li[0]['steps']))

#Steepness
flat['extras.steepness.summary'].values.tolist()[0]
#TBD

flat.rename(inplace=True, columns={
    'summary.distance': 'ors_distance',
    'summary.duration': 'ors_duration',
    'summary.ascent': 'ors_ascent',
    'summary.descent': 'ors_descent',
})
flat['ors_steepness'] = flat.ors_ascent + flat.ors_descent / 2
flat['no_waypoints'] = flat.way_points.apply(lambda li: li[1])
flat.drop(columns=[
    'extras.waycategory.values', 'extras.waycategory.summary', 'extras.surface.values', 'extras.surface.summary',
    'extras.waytypes.values', 'extras.waytypes.summary', 'extras.steepness.values', 'extras.steepness.summary',
    'extras.roadaccessrestrictions.values', 'extras.roadaccessrestrictions.summary', 'waytype', 'segments', 'bbox',
    'geometry', 'way_points', 'warnings'], inplace=True)

# m -> km
distance_col = [col for col in flat.columns.values.tolist() if col.endswith('distance')]
for col in distance_col:
    flat[col] = flat[col] / 1000

print(flat.shape)
flat.to_pickle('data/prepped/ors_features.pkl')
print('Done.')
