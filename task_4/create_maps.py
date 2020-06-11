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

feature_ea = folium.FeatureGroup(name='Entire home/apt')
feature_pr = folium.FeatureGroup(name='Private room')
feature_sr = folium.FeatureGroup(name='Shared room')

for i, v in locs_gdf.iterrows():
    popup = """
    Location id : <b>%s</b><br>
    Room type : <b>%s</b><br>
    Neighbourhood : <b>%s</b><br>
    Price : <b>%d</b><br>
    """ % (v['id'], v['room_type'], v['neighbourhood'], v['price'])

    if v['room_type'] == 'Entire home/apt':
        folium.CircleMarker(location=[v['latitude'], v['longitude']],
                            radius=1,
                            tooltip=popup,
                            color='#FFBA00',
                            fill_color='#FFBA00',
                            fill=True).add_to(feature_ea)
    elif v['room_type'] == 'Private room':
        folium.CircleMarker(location=[v['latitude'], v['longitude']],
                            radius=1,
                            tooltip=popup,
                            color='#087FBF',
                            fill_color='#087FBF',
                            fill=True).add_to(feature_pr)
    elif v['room_type'] == 'Shared room':
        folium.CircleMarker(location=[v['latitude'], v['longitude']],
                            radius=1,
                            tooltip=popup,
                            color='#FF0700',
                            fill_color='#FF0700',
                            fill=True).add_to(feature_sr)

feature_ea.add_to(locs_map)
feature_pr.add_to(locs_map)
feature_sr.add_to(locs_map)
folium.LayerControl(collapsed=False).add_to(locs_map)

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
