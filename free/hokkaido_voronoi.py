import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as mpatches

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# 観光地の座標データ（緯度、経度）
locations = {
    '札幌市': (43.0642, 141.3469),
    '小樽市': (43.1907, 140.9946),
    '函館市': (41.7688, 140.7289),
    '富良野市': (43.3417, 142.3833),
    '旭川市': (43.7706, 142.3650)
}

# 座標を配列に変換（経度、緯度の順）
points = np.array([[lon, lat] for lat, lon in locations.values()])
names = list(locations.keys())

# ボロノイ図の計算
vor = Voronoi(points)

# 図の作成
fig, ax = plt.subplots(figsize=(14, 12))

# 北海道の大まかな範囲を設定
hokkaido_lon_min, hokkaido_lon_max = 139.5, 145.8
hokkaido_lat_min, hokkaido_lat_max = 41.3, 45.5

# ボロノイ領域を色分けして描画
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# 各ボロノイ領域を塗りつぶし
for i, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]
    if not -1 in region and len(region) > 0:
        polygon = [vor.vertices[j] for j in region]
        ax.fill(*zip(*polygon), alpha=0.4, color=colors[i], edgecolor='black', linewidth=2)

# ボロノイ図のエッジを描画
for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-', linewidth=1.5)

# 母点（観光地）をプロット
for i, (name, (lat, lon)) in enumerate(locations.items()):
    ax.plot(lon, lat, 'o', markersize=15, color=colors[i], 
            markeredgecolor='white', markeredgewidth=3, zorder=5)
    
    # ラベルを表示（背景を白にして見やすく）
    ax.text(lon, lat + 0.15, name, fontsize=14, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=colors[i], linewidth=2, alpha=0.9))

# 軸の設定
ax.set_xlim(hokkaido_lon_min, hokkaido_lon_max)
ax.set_ylim(hokkaido_lat_min, hokkaido_lat_max)
ax.set_xlabel('経度 (Longitude)', fontsize=12, fontweight='bold')
ax.set_ylabel('緯度 (Latitude)', fontsize=12, fontweight='bold')
ax.set_title('北海道主要観光地のボロノイ図\n各地点から最も近い観光地エリアの可視化', 
             fontsize=16, fontweight='bold', pad=20)

# グリッド表示
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 凡例の作成
legend_elements = [mpatches.Patch(facecolor=colors[i], edgecolor='black', 
                                  label=name, alpha=0.6) 
                   for i, name in enumerate(names)]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
          framealpha=0.9, title='観光地エリア', title_fontsize=12)

# アスペクト比を等しく設定
ax.set_aspect('equal')

# 背景色を設定
ax.set_facecolor('#F0F8FF')
fig.patch.set_facecolor('white')

# レイアウト調整
plt.tight_layout()

# 保存
plt.savefig('hokkaido_voronoi.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("ボロノイ図を 'hokkaido_voronoi.png' として保存しました。")

# 表示
plt.show()

# 各観光地の座標情報を出力
print("\n【観光地の座標情報】")
for name, (lat, lon) in locations.items():
    print(f"{name}: 緯度 {lat}°N, 経度 {lon}°E")

# ボロノイ図の統計情報
print(f"\n【ボロノイ図の統計】")
print(f"母点の数: {len(points)}")
print(f"ボロノイ領域の数: {len(vor.regions)}")
print(f"ボロノイ頂点の数: {len(vor.vertices)}")
