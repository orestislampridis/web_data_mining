import folium
import pandas as pd
from folium.plugins import FastMarkerCluster

from simple_preprocessing import clean_text

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

merged_df = pd.read_csv('merged_locations.csv')
sentiment_df = pd.read_csv('sentiment_tweets.csv')
author_text_age_df = pd.read_csv('predicted_ages.csv')
gender_df = pd.read_csv('predicted_genders.csv')

merged_df.set_index('id', inplace=True)
sentiment_df.set_index('id', inplace=True)
author_text_age_df.set_index('id', inplace=True)
gender_df.set_index('id', inplace=True)

joined_loc_sent_df = pd.merge(merged_df, sentiment_df, on='id', how='inner')
joined_loc_sent_author_text_age_df = pd.merge(joined_loc_sent_df, author_text_age_df, on='id', how='inner')
joined_loc_sent_author_text_age_gender_df = pd.merge(joined_loc_sent_author_text_age_df, gender_df, on='id',
                                                     how='inner')

print(joined_loc_sent_author_text_age_gender_df.columns)

# Create a mapping for the colors
mapping = {'negative': 'red', 'neutral': 'grey', 'positive': 'green'}

# Apply said mapping
joined_loc_sent_author_text_age_gender_df["color"] = joined_loc_sent_df["VADER predicted sentiment"].map(mapping)

# Filter text to avoid problems with popup in folium maps
joined_loc_sent_author_text_age_gender_df['filtered_text'] = joined_loc_sent_author_text_age_df.text.apply(clean_text)

# Plot the long and lat coordinates as scatter points
# on the map image. It is important to set up the X-axis
# and Y-axis as per the bounding box ‘BBox’

folium_map_1 = folium.Map(
    location=[40.736851, 22.920227],
    tiles='CartoDB dark_matter',
    zoom_start=4,
    max_zoom=20
)

popup = list()

# Create popups for each row
for i, v in joined_loc_sent_author_text_age_gender_df.iterrows():
    popup.append("""
    Username : <b>%s</b><br>
    Tweeted text : <b>%s</b><br>
    Predicted sentiment : <b>%s</b><br>
    Predicted age group : <b>%s</b><br>
    Predicted gender : <b>%s</b><br>
    """ % (
        v['original author_x'], v['filtered_text'],
        v['VADER predicted sentiment'], v['age_group'], v['gender']))

joined_loc_sent_author_text_age_gender_df['popup'] = popup

folium_map_2 = folium.Map(
    location=[40.736851, 22.920227],
    tiles='CartoDB dark_matter',
    zoom_start=4,
    max_zoom=20
)

joined_loc_sent_author_text_age_gender_df.apply(
    lambda row: folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=1, tooltip=row["popup"],
                                    color=row["color"]).add_to(folium_map_1),
    axis=1)

FastMarkerCluster(data=list(zip(merged_df['latitude'].values, merged_df['longitude'].values))).add_to(folium_map_2)
folium.LayerControl().add_to(folium_map_2)
folium_map_1.save('map_1.html')
folium_map_2.save('map_2.html')
