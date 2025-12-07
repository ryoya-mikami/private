import folium
import json

# 観光地の座標データ
landmarks = [
    {"name": "清水寺", "label": "Kiyomizu-dera (清水寺)", "lat": 34.9949, "lon": 135.7850, "color": "#FF6B6B"},
    {"name": "金閣寺", "label": "Kinkaku-ji (金閣寺)", "lat": 35.0394, "lon": 135.7292, "color": "#4ECDC4"},
    {"name": "嵐山", "label": "Arashiyama (嵐山)", "lat": 35.0094, "lon": 135.6667, "color": "#45B7D1"},
    {"name": "祇園", "label": "Gion (祇園)", "lat": 35.0037, "lon": 135.7788, "color": "#FFA07A"},
    {"name": "京都駅", "label": "Kyoto Station (京都駅)", "lat": 34.9858, "lon": 135.7588, "color": "#98D8C8"}
]

# 中心座標を計算（京都の中心）
center_lat = sum(p["lat"] for p in landmarks) / len(landmarks)
center_lon = sum(p["lon"] for p in landmarks) / len(landmarks)

# Foliumマップを作成
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=11,
    tiles='OpenStreetMap'
)

# GeoJSONファイルを読み込む
geojson_file = r'c:\Users\ryoya\OneDrive\Documents\課題\プログラミング\private\free\voronoi_kyoto.geojson'
with open(geojson_file, 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

# 各観光地の色マッピングを作成
color_map = {landmark["label"]: landmark["color"] for landmark in landmarks}

# GeoJSONの各featureに色を付けて地図に追加
for feature in geojson_data['features']:
    label = feature['properties']['label']
    color = color_map.get(label, '#CCCCCC')  # デフォルトグレー
    
    folium.GeoJson(
        feature,
        style_function=lambda x, color=color: {
            'fillColor': color,
            'color': color,
            'weight': 2,
            'fillOpacity': 0.5
        },
        popup=folium.Popup(f"<b>{label}</b>の領域", max_width=200)
    ).add_to(m)

# 観光地にマーカーを追加
for landmark in landmarks:
    folium.Marker(
        location=[landmark["lat"], landmark["lon"]],
        popup=folium.Popup(f"<b>{landmark['name']}</b>", max_width=200),
        tooltip=landmark['name'],
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # 色付きサークルマーカーも追加
    folium.CircleMarker(
        location=[landmark["lat"], landmark["lon"]],
        radius=8,
        color=landmark["color"],
        fill=True,
        fillColor=landmark["color"],
        fillOpacity=0.8,
        weight=3
    ).add_to(m)

# 凡例を追加
legend_html = '''
<div style="position: fixed; 
            top: 10px; right: 10px; width: 250px; height: auto; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; border-radius: 5px; padding: 10px">
<h4 style="margin-top:0; margin-bottom:10px;">京都観光地のボロノイ図</h4>
<p style="margin: 5px 0;">各色の領域は、その色の観光地が最も近いエリアを示します。</p>
'''

for landmark in landmarks:
    legend_html += f'''
    <p style="margin: 3px 0;">
        <span style="background-color:{landmark["color"]}; 
                     padding: 3px 8px; 
                     border-radius: 3px; 
                     color: white; 
                     font-weight: bold;">■</span> 
        {landmark['name']}
    </p>
    '''

legend_html += '</div>'

m.get_root().html.add_child(folium.Element(legend_html))

# HTMLファイルとして保存
output_file = r'c:\Users\ryoya\OneDrive\Documents\課題\プログラミング\private\free\voronoi_map.html'
m.save(output_file)

print(f"Interactive map saved to: {output_file}")
print("ブラウザで開いて確認してください!")
print("色付きのボロノイ領域が表示されます!")
