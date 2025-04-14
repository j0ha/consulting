# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook to Streamlit app.
"""

# Uncomment the following line if you need to install packages (or install them via your environment)
# !pip install osmnx folium

import streamlit as st
import streamlit.components.v1 as components
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import folium
from shapely.geometry import box, Point, Polygon

# =============================================================================
# PARAMETERS & SETTINGS
# =============================================================================

CELL_SIZE = 250

amenity_weights = {
    "transport": 0.2,
    "retail_food": 0.2,
    "health": 0.15,
    "education": 0.15,
    "culture_leisure": 0.20,
    "green": 0.40,
    "sports": 0.20
}

category_tags = {
    # "transport": ["bus_station", "subway_entrance", "train_station"],
    # "retail_food": ["restaurant", "cafe", "fast_food", "supermarket"],
    # "health": ["hospital", "clinic", "pharmacy"],
    # "education": ["school", "university", "college"],
    "culture_leisure": ["cinema", "theatre", "museum", "library"],
    "green": {"leisure": ["park", "garden"]},
    "sports": {"leisure": ["fitness_centre", "stadium", "sports_centre"]}
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_grid(city_polygon: Polygon, cell_size: float, crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    """
    Create a grid (GeoDataFrame) of square cells of the given cell_size (in meters)
    that covers the bounding box of the provided city_polygon. The grid is then clipped
    by the original city_polygon.
    """
    minx, miny, maxx, maxy = city_polygon.bounds
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + cell_size, y + cell_size)
            grid_cells.append(cell)
            y += cell_size
        x += cell_size
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=crs)
    grid_clipped = gpd.clip(grid, city_polygon)
    grid_clipped = grid_clipped.reset_index().rename(columns={'index': 'cell_id'})
    return grid_clipped

def compute_nearest_distance(cell_geom: Polygon, amenities: gpd.GeoDataFrame) -> float:
    """
    Compute the minimum distance from the centroid of the cell (cell_geom)
    to all amenity points in the provided GeoDataFrame.
    If no amenity is available, return NaN.
    """
    centroid = cell_geom.centroid
    if amenities.empty:
        return np.nan
    distances = amenities.geometry.distance(centroid)
    return distances.min()

def min_max_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Performs min-max normalization on a pandas Series so that its values are between 0 and 1.
    If invert=True, then lower original values become higher normalized scores.
    """
    valid = series.dropna()
    if valid.empty:
        return series
    s_min, s_max = valid.min(), valid.max()
    if s_min == s_max:
        return pd.Series(np.where(series.isna(), np.nan, 0.5), index=series.index)
    if invert:
        norm = 1 - (series - s_min) / (s_max - s_min)
    else:
        norm = (series - s_min) / (s_max - s_min)
    return norm.clip(0, 1)

def ensure_point_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert geometries in a GeoDataFrame to points if they are not already points.
    For example, if an amenity is stored as a polygon, its centroid will be used.
    """
    if gdf.geom_type.unique()[0] != 'Point':
        gdf = gdf.copy()
        gdf['geometry'] = gdf.centroid
    return gdf

# =============================================================================
# MAIN ANALYSIS FUNCTION WITH PROGRESS SPINNERS
# =============================================================================

def run_city_analysis(city: str):
    # Step 1: Download city boundary polygon
    with st.spinner(f"Downloading boundary polygon for **{city}** from OSM..."):
        try:
            city_gdf = ox.geocode_to_gdf(city)
        except Exception as e:
            st.error(f"Error obtaining boundary for {city}: {e}")
            return
    city_metric = city_gdf.to_crs(epsg=3857)
    city_polygon = city_metric.unary_union

    # Step 2: Create grid covering the city area
    with st.spinner(f"Creating a {CELL_SIZE}x{CELL_SIZE} meter grid for **{city}**..."):
        grid = create_grid(city_polygon, CELL_SIZE)

    # Step 3: Download amenity data for each category
    with st.spinner("Downloading amenity data from OSM for each category..."):
        city_polygon_wgs = gpd.GeoSeries([city_polygon], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]
        amenities_by_category = dict()
        for cat, tags in category_tags.items():
            if isinstance(tags, dict):
                query_tags = tags
            else:
                query_tags = {"amenity": tags}
            try:
                st.write(f"Starting download for category: {query_tags}")
                gdf = ox.features.features_from_polygon(city_polygon_wgs, query_tags)
                if gdf is None or gdf.empty:
                    st.write(f"No data found for category **{cat}**.")
                    amenities_by_category[cat] = gpd.GeoDataFrame(columns=["geometry"], crs=grid.crs)
                else:
                    gdf = ensure_point_geometry(gdf)
                    gdf = gdf.to_crs(epsg=3857)
                    amenities_by_category[cat] = gdf[['geometry']].copy()
                    st.write(f"Found **{len(gdf)}** features for category **{cat}**.")
            except Exception as e:
                st.write(f"Error retrieving data for **{cat}**: {e}")
                amenities_by_category[cat] = gpd.GeoDataFrame(columns=["geometry"], crs=grid.crs)

    # Step 4: Compute nearest distances for each grid cell and amenity category
    with st.spinner("Computing nearest distances for each grid cell and category..."):
        for cat in amenity_weights.keys():
            col_name = f"dist_{cat}"
            grid[col_name] = grid.geometry.apply(
                lambda geom: compute_nearest_distance(geom, amenities_by_category.get(cat, gpd.GeoDataFrame()))
            )

    # Step 5: Normalize distance metrics
    with st.spinner("Normalizing distance metrics..."):
        for cat in amenity_weights.keys():
            dist_col = f"dist_{cat}"
            norm_col = f"{cat}_norm"
            grid[norm_col] = min_max_normalize(grid[dist_col], invert=True)

    # Step 6: Compute weighted amenities index
    with st.spinner("Computing weighted amenities index..."):
        def compute_amenities_index(row):
            total = 0
            for cat, weight in amenity_weights.items():
                norm_val = row.get(f"{cat}_norm")
                if pd.isna(norm_val):
                    norm_val = 0
                total += weight * norm_val
            return total
        grid["amenities_index"] = grid.apply(compute_amenities_index, axis=1)

    # Step 7: Create interactive map using Folium
    with st.spinner("Creating interactive map..."):
        grid_4326 = grid.to_crs(epsg=4326)
        city_center = gpd.GeoSeries([city_polygon], crs="EPSG:3857").to_crs(epsg=4326).centroid.iloc[0]
        m = folium.Map(location=[city_center.y, city_center.x], zoom_start=12)
        mean_score = grid_4326["amenities_index"].mean()

        def style_function(feature):
            score = feature["properties"]["amenities_index"]
            if score < mean_score:
                factor = score / mean_score if mean_score != 0 else 0
                red = 255
                green = int(factor * 200)
            else:
                factor = (score - mean_score) / (1 - mean_score) if (1 - mean_score) != 0 else 0
                red = int((1 - factor) * 255)
                green = 255
            return {
                "fillOpacity": 0.7,
                "weight": 1,
                "color": "black",
                "fillColor": f"#{red:02x}{green:02x}00"
            }
        
        geojson_data = grid_4326.to_json()
        folium.GeoJson(
            geojson_data,
            name="Amenities Index",
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["amenities_index"],
                aliases=["Amenities Index:"],
                localize=True,
            ),
        ).add_to(m)
        folium.LayerControl().add_to(m)

        st.write("**Interactive Map:**")
        components.html(m.get_root().render(), height=600)

    st.success("Analysis completed successfully!")

# =============================================================================
# STREAMLIT APP MAIN WITH LAYOUT CUSTOMIZATION
# =============================================================================

def main():
    # Inject custom CSS to enhance the appearance of the text input field.
    st.markdown(
        """
        <style>
        /* Style the text input */
        .stTextInput > div > div > input {
            background-color: #F0F8FF;
            border: 2px solid #0066CC;
            padding: 8px;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create two columns: Left for the title, explanation and input; right for logs and outputs.
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Enter a city name to run the analysis. (Example: München, Germany)")
        city = st.text_input("City", value="München, Germany")
    
    with col2:
        st.title("WaitWhat Placemaking Location Analysis")
        st.markdown("""
        **Introducing WaitWhat Placemaking Location Analysis**

        Our solution transforms complex geospatial data into actionable, strategic insights for superior placemaking. Leveraging real-time OpenStreetMap data and advanced analytics, this tool gauges the accessibility of key amenities—empowering urban planners, investors, and decision-makers to pinpoint optimal locations for development.
        """)
        st.markdown("---")
        if city:
            run_city_analysis(city)
        else:
            st.info("Please enter a valid city name.")

if __name__ == "__main__":
    main()
