import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import shapely.geometry as geom
import matplotlib.patches as mpatches

#Hello
############### Read London LSOA shapefile
lsoa = gpd.read_file("LSOA_2011_London_gen_MHW.shp")

############## Filter for NWL LSOAs
nwl_boroughs = ['Brent','Ealing', 'Hammersmith and Fulham', 'Harrow', 'Hillingdon', 'Hounslow', 'Kensington and Chelsea','Westminster']
nwl_lsoa = lsoa[lsoa['LAD11NM'].isin(nwl_boroughs)]
print(nwl_lsoa.head())

############## NWL Type 1 A&E Hospitals and their coordinates
type_1_ED = ["Northwick Park Hospital", "Hillingdon Hospital", "Ealing Hospital", "Chelsea and Westminster Hospital", "Charing Cross Hospital", "West Middlesex University Hospital", "St Mary's Hospital"]
hospital_coords = pd.DataFrame({'hospital_name': type_1_ED})

hospital_data = {
    "Northwick Park Hospital": (51.57466, -0.32004), 
    "Hillingdon Hospital": (51.52568, -0.46098),    
    "Ealing Hospital": (51.50784, -0.34592),      
    "Chelsea and Westminster Hospital": (51.48476, -0.18282), 
    "Charing Cross Hospital": (51.48796, -0.22145),   
    "West Middlesex University Hospital": (51.47401, -0.32428),  
    "St Mary's Hospital": (51.51766, -0.17437)   
}

hospital_coords = pd.DataFrame.from_dict(hospital_data, orient='index', columns=['latitude', 'longitude'])
hospital_coords['hospital_name'] = hospital_coords.index

geo_hospital_coords = gpd.GeoDataFrame(hospital_coords, 
                       geometry=gpd.points_from_xy(hospital_coords['longitude'], hospital_coords['latitude']), 
                       crs='EPSG:4326').to_crs('EPSG:27700')

print(geo_hospital_coords)

############## Plot NWL LSOAs and hospitals
ax = nwl_lsoa.plot(figsize=(10, 10), color='lightgrey',
                    edgecolor='black', alpha=0.5)

hospital_colors = {
    'Northwick Park Hospital': 'red',
    'Hillingdon Hospital': 'blue', 
    'Ealing Hospital': 'green',
    'Chelsea and Westminster Hospital': 'orange',
    'Charing Cross Hospital': 'purple',
    'West Middlesex University Hospital': 'brown',
    "St Mary's Hospital": 'pink'
}

for idx, row in geo_hospital_coords.iterrows():
    color = hospital_colors[row['hospital_name']]
    ax.scatter(row.geometry.x, row.geometry.y, 
              color=color, s=100, 
              edgecolor='black', linewidth=1, alpha=0.8)
    
legend_patches = [mpatches.Patch(color=color, label=hospital) 
                 for hospital, color in hospital_colors.items()]
ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title("North West London LSOAs and Type 1 A&E Hospitals", 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
plt.savefig('nwl_hospitals_map.jpg', dpi=300, bbox_inches='tight')
plt.show()


############## Centroids of NWL LSOAs
nwl_lsoa['centroid'] = nwl_lsoa.geometry.centroid
print(nwl_lsoa.head())

############## Distance matrix from LSOA centroids to hospitals
hospital_names = geo_hospital_coords['hospital_name']

for hospital in hospital_names:
    hospital_point = geo_hospital_coords[geo_hospital_coords['hospital_name'] == hospital].geometry.iloc[0]
    nwl_lsoa[f'dist_to_{hospital}'] = nwl_lsoa.centroid.distance(hospital_point)

############## Finding the 3 nearest hospitals for each LSOA
nwl_lsoa['closest_hospital_1'] = ''
nwl_lsoa['distance_1_km'] = 0.0
nwl_lsoa['closest_hospital_2'] = ''
nwl_lsoa['distance_2_km'] = 0.0  
nwl_lsoa['closest_hospital_3'] = ''
nwl_lsoa['distance_3_km'] = 0.0

for idx, lsoa_row in nwl_lsoa.iterrows():
    lsoa_centroid = lsoa_row.centroid
    
    # Calculate distances to all hospitals
    hospital_distances = []
    for _, hospital_row in geo_hospital_coords.iterrows():
        distance = lsoa_centroid.distance(hospital_row.geometry) / 1000
        hospital_distances.append((hospital_row.hospital_name, distance))
    
    # Sort by distance and get top 3
    hospital_distances.sort(key=lambda x: x[1])
    
    nwl_lsoa.at[idx, 'closest_hospital_1'] = hospital_distances[0][0]
    nwl_lsoa.at[idx, 'distance_1_km'] = hospital_distances[0][1]
    nwl_lsoa.at[idx, 'closest_hospital_2'] = hospital_distances[1][0]
    nwl_lsoa.at[idx, 'distance_2_km'] = hospital_distances[1][1]
    nwl_lsoa.at[idx, 'closest_hospital_3'] = hospital_distances[2][0]
    nwl_lsoa.at[idx, 'distance_3_km'] = hospital_distances[2][1]

print("Sample of 3 closest hospitals for each LSOA:")
print(nwl_lsoa.head(10))

############## Plot E010029658 LSOA and its 3 closest hospitals
lsoa_code = 'E01000471'
selected_lsoa = nwl_lsoa[nwl_lsoa['LSOA11CD'] == lsoa_code]
if selected_lsoa.empty:
    print(f"LSOA code {lsoa_code} not found.")
else:
    ax = nwl_lsoa.plot(figsize=(10, 10), color='lightgrey',
                        edgecolor='darkgrey', alpha=0.5)

    selected_lsoa.plot(ax=ax, color='yellow', edgecolor='black', alpha=0.7)

    lsoa_centroid = selected_lsoa.geometry.centroid.iloc[0]
    ax.scatter(lsoa_centroid.x, lsoa_centroid.y, 
               color='gold', s=150, 
               edgecolor='black', linewidth=1, label='LSOA Centroid', marker='*')

    closest_hospitals = [
        selected_lsoa['closest_hospital_1'].iloc[0],
        selected_lsoa['closest_hospital_2'].iloc[0],
        selected_lsoa['closest_hospital_3'].iloc[0]
    ]

    for hospital in closest_hospitals:
        hospital_row = geo_hospital_coords[geo_hospital_coords['hospital_name'] == hospital].iloc[0]
        color = hospital_colors[hospital]
        ax.scatter(hospital_row.geometry.x, hospital_row.geometry.y, 
                  color=color, s=100, 
                  edgecolor='black', linewidth=1, alpha=0.8)
        
    for hospital in closest_hospitals:
        hospital_row = geo_hospital_coords[geo_hospital_coords['hospital_name'] == hospital].iloc[0]
        color = hospital_colors[hospital]
    
    ax.annotate('', xy=(hospital_row.geometry.x, hospital_row.geometry.y), 
                xytext=(lsoa_centroid.x, lsoa_centroid.y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))

    legend_patches = [mpatches.Patch(color=color, label=hospital) 
                     for hospital, color in hospital_colors.items() if hospital in closest_hospitals]
    legend_patches.append(mpatches.Patch(color='yellow', label='Example LSOA'))
    legend_patches.append(mpatches.Patch(color='black', label='LSOA Centroid'))

    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5))

    ax.set_title(f"LSOA {lsoa_code} and its 3 Closest Hospitals", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(f'lsoa_{lsoa_code}_closest_hospitals_map.jpg', dpi=300, bbox_inches='tight')
    plt.show()

print("Closest hospitals:", closest_hospitals)




#To plot LSOA column with colour gradient based on HHOLDS column
#lsoa.plot(lsoa["HHOLDS"], legend=True)

# The distance from each restaurant to the Eiffel Tower
#dist_eiffel = restaurants.distance(eiffel_tower)

# The distance to the closest restaurant
#print(dist_eiffel.min())

# Filter the restaurants for closer than 1 km
#restaurants_eiffel = restaurants[dist_eiffel < 1000]

# Make a plot of the close-by restaurants - from datacamp
#ax = restaurants_eiffel.plot()
#geopandas.GeoSeries([eiffel_tower]).plot(ax=ax, color='red')
#contextily.add_basemap(ax)
#ax.set_axis_off()
#plt.show()

