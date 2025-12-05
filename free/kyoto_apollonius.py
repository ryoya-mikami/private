import folium
import numpy as np

def create_apollonius_map():
    # 1. Coordinates
    # Kyoto Station (A)
    kyoto_station = np.array([34.9858, 135.7588])
    # Kiyomizu-dera (B)
    kiyomizu_dera = np.array([34.9949, 135.7850])

    # 2. Parameters
    # Ratio PA / PB = k
    # User specified k = 2 (PA = 2 * PB, so closer to B)
    k = 2.0

    # 3. Calculation of Apollonius Circle
    # Using the vector formula equivalent to the user's image:
    # Center C = (k^2 * B - A) / (k^2 - 1)
    # Radius R = k * |AB| / |k^2 - 1|
    
    # Note: We can use the internal/external division point method which is numerically stable and equivalent.
    # Internal division point (k:1) -> P_in divides AB in ratio k:1
    # P_in = (A + k*B) / (1 + k)
    P_in = (kyoto_station + k * kiyomizu_dera) / (1 + k)

    # External division point (k:1) -> P_out divides AB externally in ratio k:1
    # P_out = (kyoto_station - k * kiyomizu_dera) / (1 - k)
    P_out = (kyoto_station - k * kiyomizu_dera) / (1 - k)

    # Center is midpoint of P_in and P_out
    center = (P_in + P_out) / 2
    
    # Calculate radius in meters for Folium
    def get_dist_meters(pt1, pt2):
        R = 6371000 # Earth radius in meters
        lat1, lon1 = np.radians(pt1)
        lat2, lon2 = np.radians(pt2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    radius_meters = get_dist_meters(center, P_in)
    
    # Radius in degrees for generating points (approximate for plotting markers)
    radius_deg_lat = np.abs(P_in[0] - P_out[0]) / 2
    # Adjust longitude radius for latitude
    radius_deg_lon = radius_deg_lat / np.cos(np.radians(center[0]))

    # 4. Visualization
    m = folium.Map(location=center, zoom_start=14)

    # Marker A: Kyoto Station
    folium.Marker(
        kyoto_station, 
        popup='Kyoto Station (A)', 
        icon=folium.Icon(color='blue', icon='train', prefix='fa')
    ).add_to(m)

    # Marker B: Kiyomizu-dera
    folium.Marker(
        kiyomizu_dera, 
        popup='Kiyomizu-dera (B)', 
        icon=folium.Icon(color='red', icon='camera', prefix='fa')
    ).add_to(m)

    # Line AB
    folium.PolyLine(
        [kyoto_station, kiyomizu_dera], 
        color="black", 
        weight=3, 
        opacity=0.8
    ).add_to(m)

    # Apollonius Circle
    folium.Circle(
        location=center,
        radius=radius_meters,
        color='green',
        fill=True,
        fill_opacity=0.1,
        popup=f'Apollonius Circle (k={k})'
    ).add_to(m)

    # Candidate points on the circle
    # Generate 5 candidate points
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    for i, ang in enumerate(angles):
        # Calculate point on the circle
        c_lat = center[0] + radius_deg_lat * np.sin(ang)
        c_lon = center[1] + radius_deg_lon * np.cos(ang)
        
        folium.Marker(
            [c_lat, c_lon], 
            popup=f'Candidate Info Center {i+1}', 
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(m)

    # Save map
    output_file = 'kyoto_apollonius_map.html'
    m.save(output_file)
    print(f"Map saved to {output_file}")

if __name__ == "__main__":
    create_apollonius_map()
