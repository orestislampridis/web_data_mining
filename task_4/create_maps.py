import folium
import pandas as pd
from folium.plugins import FastMarkerCluster

merged_df = pd.read_csv('merged.csv')

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

merged_df.apply(
    lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1).add_to(folium_map_1),
    axis=1)

FastMarkerCluster(data=list(zip(merged_df['latitude'].values, merged_df['longitude'].values))).add_to(folium_map_2)
folium.LayerControl().add_to(folium_map_2)
folium_map_1.save('map_1.html')
folium_map_2.save('map_2.html')
