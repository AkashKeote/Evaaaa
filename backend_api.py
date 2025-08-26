"""
FastAPI backend for Mumbai evacuation routing.
Reuses logic from the existing Streamlit app but exposes HTTP APIs for Flutter.
"""

import os
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ----------------------------
# Config / Filenames
# ----------------------------
GRAPHML = "roads_all.graphml"
CSV = "mumbai_ward_area_floodrisk.csv"

ASSUMED_SPEED_KMPH = 25.0
SAMPLE_FACTOR = 5
MAX_POIS_PER_CAT = 400

POI_CATEGORIES = {
    "hospital": ({"amenity": "hospital"}, "plus-square", "red"),
    "police": ({"amenity": "police"}, "shield", "darkblue"),
    "fire_station": ({"amenity": "fire_station"}, "fire", "orange"),
    "pharmacy": ({"amenity": "pharmacy"}, "medkit", "purple"),
    "school": ({"amenity": "school"}, "graduation-cap", "cadetblue"),
    "university": ({"amenity": "university"}, "university", "darkgreen"),
    "fuel": ({"amenity": "fuel"}, "gas-pump", "lightgray"),
    "shelter": ({"emergency": "shelter"}, "home", "green"),
    "bus": ({"amenity": "bus_station"}, "bus", "darkblue"),
    "train": ({"railway": "station"}, "train", "black"),
    "market": ({"shop": "supermarket"}, "shopping-cart", "brown"),
}


# ----------------------------
# Fuzzy matching
# ----------------------------
try:
    from rapidfuzz import process as fuzzy_process
except Exception:  # pragma: no cover - fallback
    try:
        from fuzzywuzzy import process as fuzzy_process
    except Exception:  # pragma: no cover - minimal fallback
        import difflib

        class _DLProc:
            @staticmethod
            def extractOne(q, choices):
                matches = difflib.get_close_matches(q, choices, n=1, cutoff=0)
                if matches:
                    score = int(
                        difflib.SequenceMatcher(None, q, matches[0]).ratio() * 100
                    )
                    return matches[0], score
                return None, 0

        fuzzy_process = _DLProc()


def extract_best_match(q: str, choices: List[str]):
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
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def route_length_m(G: nx.MultiDiGraph, path: List[int]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        ed = G.get_edge_data(u, v)
        if not ed:
            continue
        best = min(ed.values(), key=lambda d: d.get("length", float("inf")))
        total += float(best.get("length", 0.0))
    return total


def nearest_node(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    try:
        return ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except Exception:  # older osmnx
        return ox.nearest_nodes(G, X=lon, Y=lat)


# ----------------------------
# Data loading and preprocessing (cached in module globals)
# ----------------------------
_G: Optional[nx.MultiDiGraph] = None
_flood_df: Optional[pd.DataFrame] = None
_edges_sampled_geojson: Optional[Dict[str, Any]] = None


def _load_graph_and_data(graphml_path: str, csv_path: str):
    global _G, _flood_df, _edges_sampled_geojson
    if _G is not None and _flood_df is not None and _edges_sampled_geojson is not None:
        return _G, _flood_df, _edges_sampled_geojson

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
        df.rename(columns={"area": "areas"}, inplace=True)
    df = df.rename(columns={
        "flood-risk_level": "flood_risk_level",
        "flood_risk_level": "flood_risk_level",
    })
    if "flood_risk_level" not in df.columns and "risk" in df.columns:
        df["flood_risk_level"] = df["risk"]
    df["areas"] = df["areas"].astype(str).str.strip().str.lower()
    df["flood_risk_level"] = (
        df["flood_risk_level"].astype(str).str.strip().str.lower()
    )
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    node_ids = np.array(list(G.nodes))
    node_x = np.array(
        [G.nodes[n].get("x", G.nodes[n].get("lon")) for n in node_ids], dtype=float
    )
    node_y = np.array(
        [G.nodes[n].get("y", G.nodes[n].get("lat")) for n in node_ids], dtype=float
    )
    region_lons = df["longitude"].to_numpy()
    region_lats = df["latitude"].to_numpy()
    n_regions = len(df)
    dist_stack = np.empty((n_regions, len(node_ids)), dtype=float)
    for i in range(n_regions):
        dist_stack[i] = haversine_m(
            region_lons[i], region_lats[i], node_x, node_y
        )
    nearest_region_idx_per_node = np.argmin(dist_stack, axis=0)
    node_to_region_idx = dict(
        zip(node_ids.tolist(), nearest_region_idx_per_node.tolist())
    )

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
    edges_geojson = json.loads(edges_sampled.to_json())

    _G, _flood_df, _edges_sampled_geojson = G, df, edges_geojson
    return _G, _flood_df, _edges_sampled_geojson


def get_k_nearest_low_risk_routes(
    user_area: str, G: nx.MultiDiGraph, flood_df: pd.DataFrame, k: int = 5
):
    all_areas = flood_df["areas"].unique().tolist()
    match, score = extract_best_match(user_area.strip().lower(), all_areas)
    if not match or score < 40:
        return None, score, []
    start_row = flood_df[flood_df["areas"] == match].iloc[0]
    start_node = nearest_node(
        G, float(start_row["longitude"]), float(start_row["latitude"])
    )

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
            eta_min = (lm / 1000.0) / max(ASSUMED_SPEED_KMPH, 1) * 60.0
            routes.append(
                {
                    "dest_region": area,
                    "dest_node": int(node),
                    "path": path,
                    "distance_km": round(lm / 1000.0, 3),
                    "eta_min": round(eta_min, 1),
                }
            )
        except Exception:
            continue
    return match, score, routes


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Mumbai Evacuation Backend", version="1.0.0")

# Allow Flutter/web by default; adjust origins via ENV ALLOW_ORIGINS (comma-separated)
allow_origins = os.getenv("ALLOW_ORIGINS", "*")
origins = [o.strip() for o in allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_load():
    # Preload data once on startup to speed up first request
    try:
        _load_graph_and_data(GRAPHML, CSV)
    except Exception as e:
        # Do not crash app; endpoints will raise if missing
        print(f"Startup load warning: {e}")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/areas")
def list_areas():
    G, flood_df, _ = _load_graph_and_data(GRAPHML, CSV)
    return sorted(flood_df["areas"].unique().tolist())


@app.get("/match")
def match_area(q: str = Query(..., description="User-entered area name")):
    G, flood_df, _ = _load_graph_and_data(GRAPHML, CSV)
    all_areas = flood_df["areas"].unique().tolist()
    match, score = extract_best_match(q.strip().lower(), all_areas)
    if not match:
        raise HTTPException(status_code=404, detail="No match")
    return {"input": q, "match": match, "score": int(score)}


@app.get("/routes")
def compute_routes(q: str, k: int = Query(5, ge=1, le=5)):
    G, flood_df, _ = _load_graph_and_data(GRAPHML, CSV)
    matched, score, routes = get_k_nearest_low_risk_routes(q, G, flood_df, k=k)
    if not matched:
        raise HTTPException(status_code=404, detail={"message": "No match", "score": int(score)})
    return {"matched": matched, "score": int(score), "routes": routes}


@app.get("/roads_sampled_geojson")
def roads_sampled_geojson():
    G, flood_df, edges_geojson = _load_graph_and_data(GRAPHML, CSV)
    return edges_geojson


# Optional POIs endpoint (can be slow on cold start). Disabled by default.
ENABLE_POIS = bool(int(os.getenv("ENABLE_POIS", "0")))

if ENABLE_POIS:
    from functools import lru_cache

    @lru_cache(maxsize=1)
    def _fetch_pois(place: str = "Mumbai, India"):
        pois: Dict[str, Any] = {}
        for cat, (tagdict, _icon, _color) in POI_CATEGORIES.items():
            try:
                g = ox.features_from_place(place, tagdict)
                if g is None or g.empty:
                    pois[cat] = None
                    continue
                g = g.to_crs(epsg=4326)
                g["geometry"] = g.geometry.centroid
                if len(g) > MAX_POIS_PER_CAT:
                    g = g.sample(MAX_POIS_PER_CAT, random_state=1)
                pois[cat] = json.loads(g.to_json())
            except Exception:
                pois[cat] = None
        return pois

    @app.get("/pois")
    def get_pois():
        return _fetch_pois()


# Uvicorn entry for `python backend_api.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(int(os.getenv("RELOAD", "1"))),
    )


