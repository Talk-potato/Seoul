import shapefile
import pandas as pd
from pyproj import Transformer

node_path = 'C:\Synthesis\[2023-09-22]NODELINKDATA\MOCT_NODE.shp'
link_path = 'C:\Synthesis\[2023-09-22]NODELINKDATA\MOCT_LINK.shp'

sf_node = shapefile.Reader(node_path, encoding='cp949')
sf_link = shapefile.Reader(link_path, encoding='cp949')

node_header = [x[0] for x in sf_node.fields][1:]
node_data = sf_node.records()
shps = [s.points[0] for s in sf_node.shapes()]

link_header = [x[0] for x in sf_link.fields][1:]
link_data = sf_link.records()

node_dataframe = pd.DataFrame(columns=node_header, data=node_data)
node_dataframe = node_dataframe.assign(coords=shps)

link_dataframe = pd.DataFrame(columns=link_header, data=link_data)

seoul_id_range = range(100, 124)
df_node = pd.DataFrame()
df_link = pd.DataFrame()
for i in seoul_id_range:
    res_node = node_dataframe[node_dataframe['NODE_ID'].map(lambda x: x[0:3] == str(i))]
    res_link = link_dataframe[link_dataframe['LINK_ID'].map(lambda x: x[0:3] == str(i))]
    df_node = pd.concat([df_node, res_node], ignore_index=True)
    df_link = pd.concat([df_link, res_link], ignore_index=True)


transformer = Transformer.from_crs('EPSG:5186', 'EPSG:4326')
df_node[['latitude', 'longitude']] = [[*transformer.transform(x, y)] for y, x in df_node['coords']]
del df_node['coords']

print(df_node)
print(df_link)

df_node.to_csv('Seoul_nodes.csv')
df_link.to_csv('Seoul_links.csv')