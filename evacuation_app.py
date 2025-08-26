# app.py — Mumbai evacuation routing (Streamlit)
# Requirements:
# pip install streamlit streamlit-folium osmnx networkx pandas numpy geopandas folium shapely rapidfuzz fuzzywuzzy

import os
import json
import math
import io
import tempfile
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import folium
from folium import GeoJson, PolyLine, CircleMarker
from folium.plugins import MarkerCluster, MiniMap, Fullscreen, MeasureControl, MousePosition, LocateControl
from shapely.geometry import Point
import streamlit as st
from streamlit_folium import st_folium

# Fuzzy matching (rapidfuzz preferred; fallback to fuzzywuzzy or difflib)
try:
    from rapidfuzz import process as fuzzy_process
except Exception:
    try:
        from fuzzywuzzy import process as fuzzy_process
    except Exception:
        import difflib
        class _DLProc:
            @staticmethod
            def extractOne(q, choices):
                matches = difflib.get_close_matches(q, choices, n=1, cutoff=0)
                if matches:
                    score = int(difflib.SequenceMatcher(None, q, matches[0]).ratio() * 100)
                    return matches[0], score
                return None, 0
        fuzzy_process = _DLProc()

st.set_page_config(page_title="Mumbai Evacuation Routing", layout="wide")

# ----------------------------
# Config / Filenames
# ----------------------------
GRAPHML = "roads_all.graphml"   # pre-saved graph (faster than downloading)
CSV = "mumbai_ward_area_floodrisk.csv"
OUT_HTML = "mumbai_evacuation_routes.html"

ASSUMED_SPEED_KMPH = 25.0   # evacuation average speed for ETA
SAMPLE_FACTOR = 5           # sample edges 1/SAMPLE_FACTOR for lighter map
MAX_POIS_PER_CAT = 400      # cap POIs per category

# POI categories (tag dict, icon, color)
POI_CATEGORIES = {
    "hospital": ({"amenity":"hospital"}, "plus-square", "red"),
    "police": ({"amenity":"police"}, "shield", "darkblue"),
    "fire_station": ({"amenity":"fire_station"}, "fire", "orange"),
    "pharmacy": ({"amenity":"pharmacy"}, "medkit", "purple"),
    "school": ({"amenity":"school"}, "graduation-cap", "cadetblue"),
    "university": ({"amenity":"university"}, "university", "darkgreen"),
    "fuel": ({"amenity":"fuel"}, "gas-pump", "lightgray"),
    "shelter": ({"emergency":"shelter"}, "home", "green"),
    "bus": ({"amenity":"bus_station"}, "bus", "darkblue"),
    "train": ({"railway":"station"}, "train", "black"),
    "market": ({"shop":"supermarket"}, "shopping-cart", "brown"),
}

RISK_COLOR = {"low":"#1a9850","moderate":"#fc8d59","high":"#d73027","unknown":"#9e9e9e"}

# ----------------------------
# Helpers
# ----------------------------
def extract_best_match(q, choices):
    try:
        res = fuzzy_process.extractOne(q, choices)
        if res is None:
            return None, 0
        if isinstance(res, (tuple, list)):
            return res[0], int(res[1])
        return res, 100
    except Exception:
        return None, 0

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def route_length_m(G, path):
    total = 0.0
    for u,v in zip(path[:-1], path[1:]):
        ed = G.get_edge_data(u,v)
        if not ed:
            continue
        best = min(ed.values(), key=lambda d: d.get("length", float("inf")))
        total += float(best.get("length",0.0))
    return total

def nearest_node(G, lon, lat):
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except Exception:
        return ox.nearest_nodes(G, X=lon, Y=lat)

# ----------------------------
# Cached load (graph + CSV + preprocess)
# ----------------------------
@st.cache_resource
def load_graph_and_data(graphml_path: str, csv_path: str):
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"{graphml_path} not found.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    G = ox.load_graphml(graphml_path)

    try:
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
    except Exception:
        pass

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "areas" not in df.columns and "area" in df.columns:
        df.rename(columns={"area":"areas"}, inplace=True)
    df = df.rename(columns={
        "flood-risk_level":"flood_risk_level",
        "flood_risk_level":"flood_risk_level"
    })
    if "flood_risk_level" not in df.columns and "risk" in df.columns:
        df["flood_risk_level"] = df["risk"]
    df["areas"] = df["areas"].astype(str).str.strip().str.lower()
    df["flood_risk_level"] = df["flood_risk_level"].astype(str).str.strip().str.lower()
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    node_ids = np.array(list(G.nodes))
    node_x = np.array([G.nodes[n].get("x", G.nodes[n].get("lon")) for n in node_ids], dtype=float)
    node_y = np.array([G.nodes[n].get("y", G.nodes[n].get("lat")) for n in node_ids], dtype=float)
    region_lons = df["longitude"].to_numpy()
    region_lats = df["latitude"].to_numpy()
    n_regions = len(df)
    dist_stack = np.empty((n_regions, len(node_ids)), dtype=float)
    for i in range(n_regions):
        dist_stack[i] = haversine_m(region_lons[i], region_lats[i], node_x, node_y)
    nearest_region_idx_per_node = np.argmin(dist_stack, axis=0)
    node_to_region_idx = dict(zip(node_ids.tolist(), nearest_region_idx_per_node.tolist()))

    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True).reset_index()
    if "u" not in edges_gdf.columns:
        edges_gdf = edges_gdf.reset_index()
    edges_gdf["_u"] = edges_gdf["u"].astype(int)
    edges_gdf["region_idx"] = edges_gdf["_u"].map(node_to_region_idx)
    def idx_to_risk(i):
        try:
            return df.iloc[int(i)]["flood_risk_level"]
        except Exception:
            return "unknown"
    edges_gdf["risk_level"] = edges_gdf["region_idx"].apply(idx_to_risk)
    edges_sampled = edges_gdf.iloc[::SAMPLE_FACTOR].copy()

    return G, df, edges_sampled

@st.cache_data
def fetch_pois(place="Mumbai, India", categories=POI_CATEGORIES, cap=MAX_POIS_PER_CAT):
    pois = {}
    for cat, (tagdict, icon, color) in categories.items():
        try:
            g = ox.features_from_place(place, tagdict)
            if g is None or g.empty:
                pois[cat] = None
                continue
            g = g.to_crs(epsg=4326)
            g["geometry"] = g.geometry.centroid
            if len(g) > cap:
                g = g.sample(cap, random_state=1)
            pois[cat] = g
        except Exception:
            pois[cat] = None
    return pois

# ----------------------------
# UI: load data once
# ----------------------------
st.title("Mumbai — Risk-aware Evacuation Routes (Streamlit)")
st.caption("Enter a region (fuzzy search). Computes top-5 low-risk evacuation routes and shows POIs, road risk coloring and map tools.")

with st.spinner("Loading graph and CSV (this is cached)..."):
    try:
        G, flood_df, edges_sampled = load_graph_and_data(GRAPHML, CSV)
    except Exception as e:
        st.error(f"Loading error: {e}")
        st.stop()

with st.spinner("Fetching POIs (sampled/capped)..."):
    pois_by_cat = fetch_pois()

col1, col2 = st.columns([1,2])
with col1:
    user_region = st.text_input("Enter your region (area) — fuzzy match:", value="")
    compute_btn = st.button("Compute Evacuation Routes")

    st.markdown("**Map options**")
    show_roads = st.checkbox("Show risk-colored roads (sampled)", value=True)
    show_pois = st.checkbox("Show POI clusters", value=True)
    top_k = st.slider("Number of evacuation routes to display", min_value=1, max_value=5, value=5)

with col2:
    st.markdown("### Instructions")
    st.write("Type an area name (e.g. 'Sewri', 'Mulund', 'Colaba'). The app will fuzzy-match and compute top low-risk destinations, draw up to 5 evacuation routes, and display POIs and risk-colored roads.")

# helper: compute routes
def get_k_nearest_low_risk_routes(user_area: str, G, flood_df, k=5):
    all_areas = flood_df["areas"].unique().tolist()
    match, score = extract_best_match(user_area.strip().lower(), all_areas)
    if not match or score < 40:
        return None, score, []
    start_row = flood_df[flood_df["areas"] == match].iloc[0]
    start_node = nearest_node(G, float(start_row["longitude"]), float(start_row["latitude"]))

    low_df = flood_df[flood_df["flood_risk_level"] == "low"]
    if low_df.empty:
        return match, score, []

    try:
        dists = nx.single_source_dijkstra_path_length(G, start_node, weight="length")
    except Exception:
        dists = {}

    candidates = []
    for _, r in low_df.iterrows():
        node = nearest_node(G, float(r["longitude"]), float(r["latitude"]))
        d = dists.get(node, None)
        if d is not None:
            candidates.append((r["areas"], int(node), d))
    if not candidates:
        return match, score, []

    candidates.sort(key=lambda x: x[2])
    selected = []
    seen = set()
    for area, node, d in candidates:
        if area in seen:
            continue
        selected.append((area, node, d))
        seen.add(area)
        if len(selected) >= k:
            break

    routes = []
    for area, node, d in selected:
        try:
            path = nx.shortest_path(G, start_node, node, weight="length")
            lm = route_length_m(G, path)
            eta_min = (lm/1000.0)/max(ASSUMED_SPEED_KMPH,1)*60.0
            routes.append({
                "dest_region": area,
                "dest_node": int(node),
                "path": path,
                "distance_km": round(lm/1000.0, 3),
                "eta_min": round(eta_min,1)
            })
        except Exception:
            continue
    return match, score, routes

# action
if compute_btn:
    if not user_region.strip():
        st.warning("Please enter a region name.")
    else:
        with st.spinner("Computing routes..."):
            matched, score, routes = get_k_nearest_low_risk_routes(user_region, G, flood_df, k=top_k)
        if not matched:
            st.error(f"Could not match '{user_region}' (score {score}). Try a different name.")
        elif not routes:
            st.warning("No reachable low-risk destinations found.")
        else:
            st.success(f"Using region: {matched.title()} (match score {score}%)")
            for i,r in enumerate(routes,1):
                st.write(f"**Route {i}** → {r['dest_region'].title()} — {r['distance_km']:.2f} km • {r['eta_min']:.0f} min")

            start_idx = int(flood_df[flood_df["areas"]==matched].index[0])
            center = [float(flood_df.loc[start_idx,"latitude"]), float(flood_df.loc[start_idx,"longitude"])]
            m = folium.Map(location=center, zoom_start=12, tiles=None, control_scale=True)

            folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
            folium.TileLayer("cartodbpositron", name="Light", attr="© OpenStreetMap contributors © CARTO").add_to(m)
            folium.TileLayer("cartodbdark_matter", name="Dark", attr="© OpenStreetMap contributors © CARTO").add_to(m)
            folium.TileLayer("Stamen Terrain", name="Terrain", attr="Map tiles by Stamen Design, © OpenStreetMap").add_to(m)
            folium.TileLayer("Stamen Toner", name="Toner", attr="Map tiles by Stamen Design, © OpenStreetMap").add_to(m)

            MiniMap(toggle_display=True).add_to(m)
            Fullscreen().add_to(m)
            MeasureControl(primary_length_unit="kilometers").add_to(m)
            MousePosition(position="bottomright", prefix="Lat/Lon: ").add_to(m)
            LocateControl(auto_start=False).add_to(m)

            if show_roads:
                gj = GeoJson(data=edges_sampled.__geo_interface__,
                             name="Roads (risk-colored, sampled)",
                             style_function=lambda f: {"color": RISK_COLOR.get(str(f["properties"].get("risk_level","unknown")).lower(), "#9e9e9e"),
                                                       "weight":1.2, "opacity":0.9})
                gj.add_to(m)

            rc = MarkerCluster(name=f"Regions ({len(flood_df)})")
            for _, row in flood_df.iterrows():
                color = RISK_COLOR.get(str(row["flood_risk_level"]).lower(), "#9e9e9e")
                CircleMarker(location=[float(row["latitude"]), float(row["longitude"])],
                             radius=5, color=color, fill=True, fill_opacity=0.9,
                             tooltip=f"{row['areas'].title()} — Risk: {row['flood_risk_level'].title()}").add_to(rc)
            m.add_child(rc)

            if show_pois:
                for cat, gdf in pois_by_cat.items():
                    if gdf is None or gdf.empty:
                        continue
                    cluster = MarkerCluster(name=f"{cat.replace('_',' ').title()} ({len(gdf)})")
                    for _, r in gdf.iterrows():
                        try:
                            lat = float(r.geometry.y); lon = float(r.geometry.x)
                        except Exception:
                            continue
                        popup = str(r.get("name") or cat.replace("_"," ").title())
                        folium.Marker(location=[lat,lon],
                                      icon=folium.Icon(color=POI_CATEGORIES[cat][2], icon=POI_CATEGORIES[cat][1], prefix="fa"),
                                      popup=popup).add_to(cluster)
                    m.add_child(cluster)

            colors = ["#0066ff","#00cc44","#ff8800","#aa00ff","#0099cc"]
            for i, r in enumerate(routes):
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in r["path"]]
                PolyLine(coords, color=colors[i % len(colors)], weight=6, opacity=0.9,
                         tooltip=f"Route {i+1}: {r['distance_km']:.2f} km • {r['eta_min']:.0f} min → {r['dest_region'].title()}").add_to(m)

                if i == 0:
                    s = r["path"][0]
                    folium.CircleMarker(location=(G.nodes[s]["y"], G.nodes[s]["x"]),
                                        radius=7, color="#000", fill=True, fill_color="#ffffff",
                                        tooltip=f"Start: {matched.title()}").add_to(m)

                dnode = r["path"][-1]
                folium.CircleMarker(location=(G.nodes[dnode]["y"], G.nodes[dnode]["x"]),
                                    radius=7, color="#000", fill=True, fill_color="#ffd24d",
                                    tooltip=f"Destination: {r['dest_region'].title()}").add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)

            # render map in Streamlit (compatible with all versions)
            try:
                st_map = st_folium(m, width=1200, height=700, returned_objects=[])
            except TypeError:
                st_map = st_folium(m, width=1200, height=700)

            html_path = OUT_HTML
            m.save(html_path)
            with open(html_path, "rb") as fh:
                st.download_button("Download map HTML", data=fh, file_name=html_path, mime="text/html")
