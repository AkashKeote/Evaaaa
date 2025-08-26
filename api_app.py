"""
Streamlit-hosted API shim for Mumbai evacuation routing.
Use query params to get JSON-like responses for Flutter without extra infra cost.

Examples:
- ?api=1&endpoint=health
- ?api=1&endpoint=areas
- ?api=1&endpoint=match&q=mulund
- ?api=1&endpoint=routes&q=mulund&k=5
- ?api=1&endpoint=roads

Note: Streamlit returns text/html, but we render ONLY a JSON string in the page body
so mobile/web clients can parse the body text with jsonDecode.
"""

import json
import os
from typing import Any, Dict

import streamlit as st

# Reuse logic from backend_api
import backend_api as core


st.set_page_config(page_title="Mumbai Evacuation API (Streamlit)", layout="wide")


def _json_response(payload: Dict[str, Any]):
    # Render only a JSON block so clients can parse response body easily.
    st.markdown(
        f"<pre style='white-space: pre-wrap; word-wrap: break-word; margin:0'>{json.dumps(payload, ensure_ascii=False)}</pre>",
        unsafe_allow_html=True,
    )


def handle_api():
    endpoint = st.query_params.get("endpoint", "").strip().lower()
    q = st.query_params.get("q", "")
    k = st.query_params.get("k", "5")

    try:
        G, flood_df, edges_geo = core._load_graph_and_data(core.GRAPHML, core.CSV)
    except Exception as e:
        _json_response({"error": str(e)})
        return

    if endpoint == "health":
        _json_response({"ok": True})
        return
    if endpoint == "areas":
        _json_response(sorted(flood_df["areas"].unique().tolist()))
        return
    if endpoint == "match":
        match, score = core.extract_best_match(q.strip().lower(), flood_df["areas"].unique().tolist())
        if not match:
            _json_response({"input": q, "error": "No match", "score": int(score)})
        else:
            _json_response({"input": q, "match": match, "score": int(score)})
        return
    if endpoint == "routes":
        try:
            k_int = max(1, min(5, int(k)))
        except Exception:
            k_int = 5
        matched, score, routes = core.get_k_nearest_low_risk_routes(q, G, flood_df, k=k_int)
        if not matched:
            _json_response({"error": "No match", "score": int(score)})
        else:
            _json_response({"matched": matched, "score": int(score), "routes": routes})
        return
    if endpoint == "roads":
        _json_response(edges_geo)
        return

    _json_response({
        "error": "unknown endpoint",
        "hint": "use one of: health, areas, match, routes, roads",
    })


def render_docs():
    st.title("Mumbai Evacuation Backend — Streamlit API Mode")
    st.caption("Use query params to get JSON-like responses suitable for Flutter.")
    base = st.text_input("Base URL", value=st.get_option("browser.serverAddress") or "https://YOUR-STREAMLIT-APP-URL")
    st.markdown("### Examples")
    st.code(f"{base}?api=1&endpoint=health")
    st.code(f"{base}?api=1&endpoint=areas")
    st.code(f"{base}?api=1&endpoint=match&q=mulund")
    st.code(f"{base}?api=1&endpoint=routes&q=mulund&k=5")
    st.code(f"{base}?api=1&endpoint=roads")
    st.markdown("If you see a JSON block only, your Flutter app can parse the response body.")


# Entry
is_api = st.query_params.get("api", "0") in ("1", "true", "yes")
if is_api:
    handle_api()
else:
    render_docs()


