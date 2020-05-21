"""
Script used for location modelling and visualization

Uses the coordinates supplied by twitter API in the geo field and also
the location of each tweet's user takn by their profiles after converting
it to coordinates using the Nominatim geolocator. Then creates a html
page that contains a Folium map with all the coordinates plotted
"""
import time

import folium
import matplotlib.pyplot as plt
import pandas as pd
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim

from connect_mongo import read_mongo


def swapCoords(x):
    """
    Helper function to flip a set of coordinates
    :param x: coordinate as [latitude, LONGITUDE]
    :return: coordinate as [LONGITUDE, latitude]
    """
    out = []
    for iter in x:
        if isinstance(iter, list):
            out.append(swapCoords(iter))
        else:
            return [x[1], x[0]]
    return out


def recursive_geocode(location, recursion=0):
    """
    Wrapper around the geolocator.geocode call.
    This will try to request coordinates 10 times, after a delay of 1 second in case of service failure
    :param location: the location to convert to coordinates
    :param recursion: the number of times it will request for the same location
    :return: coordinates given by geolocator
    """
    try:
        return geolocator.geocode(location, timeout=15)[1] if geolocator.geocode(value) else None
    except GeocoderTimedOut:
        print("Error: geocode failed on input %s" % value)
        # Set max recursions
        if recursion > 10:
            return None
        # Wait a bit and try again
        time.sleep(1)
        return recursive_geocode(location, recursion=recursion + 1)


# Get our initial df with columns 'geo.coordinates' and 'location'
df = read_mongo(db='twitter_db', collection='twitter_collection', query={'geo.coordinates': 1, 'location': 1})

# Initialize geolocator for later use
geolocator = Nominatim(user_agent='orestis')

# The geo.coordinates are formatted by Twitter API as [latitude, LONGITUDE]
# so we need to reverse them to [LONGITUDE, latitude]
geo_dict = (df["geo"].dropna().to_dict())
long = list()
lat = list()

for index, value in geo_dict.items():
    new_value = value['coordinates']
    reversed_coords = swapCoords(new_value)
    long.append(reversed_coords[0])
    lat.append(reversed_coords[1])
    geo_dict[index] = reversed_coords

# Create our new df with reversed coordinates and separate columns for long and lat
geo_df = pd.DataFrame(list(geo_dict.items()), columns=['id', 'coordinates'])
geo_df['longitude'] = long
geo_df['latitude'] = lat
print(geo_df)
geo_df.to_csv('geo.csv', index=False)

# Because we don't have many users' coordinates, we will try to convert the locations taken
# from the users profiles into cooordinates. For this task we will use the Nomatim geocoder
# which takes as input a location and outputs its lat and long coordinates.
# First create a mapping to avoid unnecessary geolocator failures
mapping = {'Los Angeles, CA': 'Los Angeles',
           'London/Manchester': 'London',
           'Northern California': 'California',
           'Irving, TX': 'Irving',
           'Fresno, CA': 'Fresno',
           'San Antonio, TX': 'San Antonio',
           'European Union': 'Europe',
           'European Union 🇪🇺': 'Europe',
           'Europe': 'Europe',
           'UK': 'United Kingdom',
           'Scotland, United Kingdom': 'Scotland',
           'England, United Kingdom': 'England',
           'USA': 'United States',
           }

df['location'] = df['location'].apply(lambda x: mapping[x] if x in mapping.keys() else x)
loc_dict = (df['location'].dropna().to_dict())
print(loc_dict)

# For each location send a request to Nomatim geolocator. Wrap it with a try except clause
# to catch if the location given by the user doesn't exist e.g. "Moon", "somewhere", etc.
for index, value in loc_dict.items():
    print(index)
    location = recursive_geocode(value)
    loc_dict[index] = location

# Convert to dataframe and save output to csv as the task takes a long time to complete
print(loc_dict)
loc_df = pd.DataFrame(list(loc_dict.items()), columns=['id', 'coordinates'])
loc_df['longitude'] = loc_df.coordinates.apply(lambda x: x[1])
loc_df['latitude'] = loc_df.coordinates.apply(lambda x: x[0])
loc_df.to_csv('loc.csv', index=False)

# Calculate counts of locations and print top
locs = df["location"].value_counts()
print(locs[locs >= 10])

# Keep only the city names
locs = list(locs.index)

# To plot to a map we have to define the Bounding Box. Bounding Box is the area defined
# by two longitudes and two latitudes that will include all spatial points required.
BBox = ((geo_df['longitude'].min(), geo_df['longitude'].max(),
         geo_df['latitude'].min(), geo_df['latitude'].max()))

print(BBox)

ruh_m = plt.imread('map.png')

# Plot the long and lat coordinates as scatter points
# on the map image. It is important to set up the X-axis
# and Y-axis as per the bounding box ‘BBox’

folium_map = folium.Map(
    location=[40.736851, 22.920227],
    tiles='CartoDB dark_matter',
    zoom_start=4
)
# FastMarkerCluster(data=list(zip(geo_df['latitude'].values, geo_df['longitude'].values))).add_to(folium_map)
# folium.LayerControl().add_to(folium_map)
geo_df.apply(lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1).add_to(folium_map),
             axis=1)
folium_map.save('map.html')
