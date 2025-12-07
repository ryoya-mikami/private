
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from pyproj import Transformer
import json
import os

try:
    import geopandas as gpd
except Exception:
    gpd = None

# Set Japanese font for Windows
plt.rcParams['font.family'] = 'MS Gothic'

# --------- Utility Function ----------
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Convert scipy Voronoi to finite polygons (2D).
    Returns list of regions as list of points (in same coords as input) and region indices.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max() * 2

    # Map ridge vertices to regions
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Construct finite polygons
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append([vor.vertices[v].tolist() for v in vertices])
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        pts = []
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                # both finite
                continue
            # compute the missing endpoint at a distance "radius"
            t = vor.points[p2] - vor.points[p1]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_v = len(new_vertices) - 1
            pts.append((v2, new_v))

        # collect region vertices: include existing finite verts + created far points
        region_vertices = [v for v in vertices if v >= 0]
        for v2, new_v in pts:
            region_vertices.append(v2)
            region_vertices.append(new_v)
        # order polygon vertices ccw
        coords = np.array([new_vertices[v] for v in region_vertices])
        centroid = coords.mean(axis=0)
        angles = np.arctan2(coords[:,1] - centroid[1], coords[:,0] - centroid[0])
        order = np.argsort(angles)
        polygon = coords[order].tolist()
        new_regions.append(polygon)

    return new_regions

# --------- Main Function ----------
def generate_voronoi(latlon_points, labels=None,
                     projection_epsg=32653,   # UTM zone 53N (Kyoto area)
                     clip_margin_m=2000,
                     out_png="voronoi_kyoto.png",
                     out_geojson="voronoi_kyoto.geojson"):
    """
    latlon_points: list of (lat, lon) or (lat, lon, label)
    """

    # Organize: latlon -> numpy array lon/lat
    pts_ll = []
    pts_labels = []
    for i, p in enumerate(latlon_points):
        if len(p) >= 3:
            lat, lon, lab = p[0], p[1], p[2]
            pts_ll.append((lon, lat))
            pts_labels.append(lab)
        else:
            lat, lon = p
            pts_ll.append((lon, lat))
            pts_labels.append(labels[i] if labels is not None and i < len(labels) else f"P{i+1}")

    # Projection transformer: WGS84 -> projection_epsg
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{projection_epsg}", always_xy=True)
    pts_xy = np.array([transformer.transform(lon, lat) for lon, lat in pts_ll])

    # Voronoi
    vor = Voronoi(pts_xy)

    # Convert to finite polygons
    regions = voronoi_finite_polygons_2d(vor)

    # Build shapely polygons and clip to bounding box (with margin)
    xs, ys = pts_xy[:,0], pts_xy[:,1]
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    bbox = box(minx - clip_margin_m, miny - clip_margin_m, maxx + clip_margin_m, maxy + clip_margin_m)

    poly_list = []
    for region, label, pt in zip(regions, pts_labels, pts_xy):
        poly = Polygon(region)
        poly = poly.intersection(bbox)
        if not poly.is_empty and poly.is_valid:
            poly_list.append((label, poly, pt))
        else:
            poly_list.append((label, Point(pt).buffer(10), pt))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10,10))
    # Define a color map or cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(poly_list)))

    for i, (label, poly, pt) in enumerate(poly_list):
        if hasattr(poly, 'geoms'): # MultiPolygon
            for geom in poly.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.4, fc=colors[i], edgecolor='white')
        else:
            x,y = poly.exterior.xy
            ax.fill(x, y, alpha=0.4, fc=colors[i], edgecolor='white')

        ax.plot(pt[0], pt[1], 'k.', markersize=8)
        # Add label with white halo for readability
        t = ax.text(pt[0], pt[1], f" {label}", fontsize=11, fontweight='bold', ha='left', va='bottom')
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # plot boundaries
    ax.set_xlim(bbox.bounds[0], bbox.bounds[2])
    ax.set_ylim(bbox.bounds[1], bbox.bounds[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Voronoi Diagram of Kyoto Sightseeing Spots", fontsize=16)
    
    # Hide axes ticks for cleaner map look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    # --- GeoJSON Output ---
    features = []
    for label, poly, pt in poly_list:
        def transform_coords(coords):
             # Inverse transform: UTM -> WGS84
            inverse_transformer = Transformer.from_crs(f"EPSG:{projection_epsg}", "EPSG:4326", always_xy=True)
            return [inverse_transformer.transform(x,y) for x,y in coords]

        if hasattr(poly, "geom_type") and poly.geom_type in ("Polygon", "MultiPolygon"):
            if poly.geom_type == "Polygon":
                exterior = transform_coords(list(poly.exterior.coords))
                interiors = [transform_coords(list(ring.coords)) for ring in poly.interiors]
                geom = {"type": "Polygon", "coordinates": [exterior] + interiors}
            else:
                polys = []
                for p in poly.geoms:
                    ext = transform_coords(list(p.exterior.coords))
                    ints = [transform_coords(list(ring.coords)) for ring in p.interiors]
                    polys.append([ext] + ints)
                geom = {"type": "MultiPolygon", "coordinates": polys}
        else:
            lon, lat = Transformer.from_crs(f"EPSG:{projection_epsg}", "EPSG:4326", always_xy=True).transform(pt[0], pt[1])
            geom = {"type": "Point", "coordinates": [lon, lat]}

        features.append({
            "type": "Feature",
            "properties": {"label": label},
            "geometry": geom
        })

    geo = {"type": "FeatureCollection", "features": features}
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False, indent=2)

    print(f"Saved PNG: {out_png}")
    print(f"Saved GeoJSON: {out_geojson}")
    return out_png, out_geojson

if __name__ == "__main__":
    landmarks = [
        (34.9949, 135.7850, "Kiyomizu-dera (清水寺)"),
        (35.0394, 135.7292, "Kinkaku-ji (金閣寺)"),
        (35.0094, 135.6667, "Arashiyama (嵐山)"),
        (35.0037, 135.7788, "Gion (祇園)"),
        (34.9858, 135.7588, "Kyoto Station (京都駅)")
    ]

    generate_voronoi(landmarks,
                     projection_epsg=32653,
                     clip_margin_m=3000,
                     out_png="c:\\Users\\ryoya\\OneDrive\\Documents\\課題\\プログラミング\\private\\free\\voronoi_kyoto.png",
                     out_geojson="c:\\Users\\ryoya\\OneDrive\\Documents\\課題\\プログラミング\\private\\free\\voronoi_kyoto.geojson")
