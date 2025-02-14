"""
Script used for location modelling

Uses the coordinates supplied by twitter API in the geo field and also
the location of each tweet's user taken by their profiles after converting
it to coordinates using the Nominatim geolocator. Then creates a html
page that contains a Folium map with all the coordinates plotted
"""
import time

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


def get_coords_and_save_to_csv(dict):
    # For each location send a request to Nomatim geolocator. Wrap it with a try except clause
    # to catch if the location given by the user doesn't exist e.g. "Moon", "somewhere", etc.
    for index, value in dict.items():
        print(index)
        location = recursive_geocode(value)
        dict[index] = location

    # Convert to dataframe and save output to csv as the task takes a long time to complete
    print(dict)
    loc_df = pd.DataFrame(list(dict.items()), columns=['id', 'coordinates'])
    # loc_df['longitude'] = loc_df.coordinates.apply(lambda x: x[1])
    # loc_df['latitude'] = loc_df.coordinates.apply(lambda x: x[0])
    loc_df.to_csv('loc.csv', index=False)


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
geo_df.set_index('id', inplace=True)

print(geo_df)
# geo_df.to_csv('geo.csv', index=False)

# Because we don't have many users' coordinates, we will try to convert the locations taken
# from the users profiles into cooordinates. For this task we will use the Nominatim geocoder
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


# get_coords_and_save_to_csv(loc_dict)

def convert(string):
    string = (string.replace('(', ''))
    string = (string.replace(')', ''))
    li = [float(i) for i in string.split(",")]

    return li


new_loc_df = pd.read_csv('loc.csv')
new_loc_df = new_loc_df.dropna()

new_loc_df['coordinates'] = new_loc_df.coordinates.apply(lambda x: convert(x))
new_loc_df['longitude'] = new_loc_df.coordinates.apply(lambda x: x[1])
new_loc_df['latitude'] = new_loc_df.coordinates.apply(lambda x: x[0])
new_loc_df.set_index('id', inplace=True)

print(new_loc_df)

# Append one dataframe to the other to unify the locations
merged_df = new_loc_df.append(geo_df).sort_index()

# Drop duplicates from above procedure
merged_df = merged_df.groupby(merged_df.index).first()
print(merged_df)

# Save unified locations to csv file
merged_df.to_csv('../task_4/merged_locations.csv')

# To plot to a map we have to define the Bounding Box. Bounding Box is the area defined
# by two longitudes and two latitudes that will include all spatial points required.
BBox = ((merged_df['longitude'].min(), merged_df['longitude'].max(),
         merged_df['latitude'].min(), merged_df['latitude'].max()))

print(BBox)
