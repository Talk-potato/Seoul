import os
import folium
import pandas as pd
import numpy as np
import networkx as nx

nodes = pd.read_csv('Seoul_nodes.csv')
links = pd.read_csv('Seoul_links.csv')
nodes = nodes[['NODE_ID', 'NODE_NAME', 'latitude', 'longitude']]
links = links[['LINK_ID', 'F_NODE', 'T_NODE', 'MAX_SPD', 'LENGTH']]

# print(links.head())

# 외부지역과 연결된 다른 노드 제거
LINKID_in = links['F_NODE'].apply(lambda x: x in list(nodes['NODE_ID']))
FNODE_in = links['T_NODE'].apply(lambda x: x in list(nodes['NODE_ID']))
seoul_links = links[LINKID_in & FNODE_in]

# print(len(links)) 22935
# print(len(seoul_links)) 4210


G = nx.Graph()
R = 6371e3  # R is the Earth's radius

for idx, row in nodes.iterrows():
    # add node to Graph G
    G.add_node(row['NODE_ID'], Label=row['NODE_NAME'], latitude=row['latitude'], longitude=row['longitude'])

for idx, row in seoul_links.iterrows():
    ## Calculate the distance between Source and Target Nodes
    link_id_row = nodes[nodes['NODE_ID'] == row['F_NODE']]
    f_node_row = nodes[nodes['NODE_ID'] == row['T_NODE']]

    lon1 = float(link_id_row['longitude'].iloc[0] * np.pi / 180)
    lat1 = float(link_id_row['latitude'].iloc[0] * np.pi / 180)
    lon2 = float(f_node_row['longitude'].iloc[0] * np.pi / 180)
    lat2 = float(f_node_row['latitude'].iloc[0] * np.pi / 180)

    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(a ** 0.5, (1 - a) ** 0.5)
    d = R * c

    # Link attribute : 'LINK_ID', 'F_NODE' and weight = 'Length between them'
    G.add_edge(row['F_NODE'], row['T_NODE'], weight=d)
# print(G) #Graph with 8640 nodes and 4209 edges

# Positioning the Standard Point for our Folium Map
std_point = tuple(nodes.head(1)[['latitude', 'longitude']].iloc[0])
# print(std_point) #(37.5695820797098, 126.97695348723622)

map_osm = folium.Map(location=std_point, zoom_start=10)
for ix, row in nodes.iterrows():
    location = (row['latitude'], row['longitude'])
    folium.Circle(
        location=location,
        radius=G.degree[row['NODE_ID']] * 30,
        color='white',
        weight=1,
        fill_opacity=0.6,
        opacity=1,
        fill_color='red',
        fill=True,
        popup=str(row['NODE_NAME'])
    ).add_to(map_osm)
kw = {'opacity': 0.5, 'weight': 2}
for ix, row in seoul_links.iterrows():
    start = tuple(nodes[nodes['NODE_ID'] == row['F_NODE']][['latitude', 'longitude']].iloc[0])
    end = tuple(nodes[nodes['NODE_ID'] == row['T_NODE']][['latitude', 'longitude']].iloc[0])
    folium.PolyLine(
        locations=[start, end],
        color='blue',
        line_cap='round',
        **kw,
    ).add_to(map_osm)

map_osm.save("seoul_map.html")
