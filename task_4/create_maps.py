import folium
import pandas as pd
from folium.plugins import FastMarkerCluster

merged_df = pd.read_csv('merged_locations.csv')
sentiment_df = pd.read_csv('sentiment_tweets.csv')

merged_df.set_index('id', inplace=True)
sentiment_df.set_index('id', inplace=True)

joined_loc_sent_df = pd.merge(merged_df, sentiment_df, on='id', how='inner')

print(joined_loc_sent_df)

# Create a mapping for the colors
mapping = {'negative': 'red', 'neutral': 'grey', 'positive': 'green'}

# Apply said mapping
joined_loc_sent_df["color"] = joined_loc_sent_df["VADER predicted sentiment"].map(mapping)

print(joined_loc_sent_df)

# Plot the long and lat coordinates as scatter points
# on the map image. It is important to set up the X-axis
# and Y-axis as per the bounding box ‘BBox’

folium_map_1 = folium.Map(
    location=[40.736851, 22.920227],
    tiles='CartoDB dark_matter',
    zoom_start=4,
    max_zoom=20
)

folium_map_2 = folium.Map(
    location=[40.736851, 22.920227],
    tiles='CartoDB dark_matter',
    zoom_start=4,
    max_zoom=20
)

joined_loc_sent_df.apply(
    lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1,
                                    color=row["color"]).add_to(folium_map_1),
    axis=1)

FastMarkerCluster(data=list(zip(merged_df['latitude'].values, merged_df['longitude'].values))).add_to(folium_map_2)
folium.LayerControl().add_to(folium_map_2)
folium_map_1.save('map_1.html')
folium_map_2.save('map_2.html')
