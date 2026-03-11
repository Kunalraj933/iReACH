"""
=============================================================================
NHS Hospital Cyberattack SI Propagation — Interactive Map
=============================================================================
Run from VS Code terminal:
    pip install folium pandas geopandas numpy
    python hospital_si_map.py

Opens:  hospital_si_map.html  (auto-launches in your browser)

Required files in the SAME folder as this script:
    hospital_communities_with_coords.csv
    hospital_si_results.csv
    hospital_communities.csv
    icb_boundaries_geodata.ipynb   (or icb_boundaries.geojson — see NOTE below)
    agg_hospital_transfers.csv     (optional — for transfer edge overlay)

NOTE on ICB boundaries:
    If you have icb_boundaries_geodata.ipynb, export the GeoJSON from it:
        import geopandas as gpd
        gdf.to_file("icb_boundaries.geojson", driver="GeoJSON")
    Then place icb_boundaries.geojson in the same folder.
    If the file is missing the script will skip the boundary layer gracefully.
=============================================================================
"""

import os, json, math, webbrowser
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster, FeatureGroupSubGroup
from collections import defaultdict

# ── 0. Config ─────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
COORDS_FILE      = os.path.join(SCRIPT_DIR, "hospital_communities_with_coords.csv")
SI_FILE          = os.path.join(SCRIPT_DIR, "hospital_si_results.csv")
COMMUNITIES_FILE = os.path.join(SCRIPT_DIR, "hospital_communities.csv")
TRANSFERS_FILE   = os.path.join(SCRIPT_DIR, "agg_hospital_transfers.csv")
ICB_FILE         = os.path.join(SCRIPT_DIR, "icb_boundaries.geojson")
OUTPUT_HTML      = os.path.join(SCRIPT_DIR, "hospital_si_map.html")

BETA              = 0.04    # SI transmission rate
EDGE_THRESHOLD    = 50      # min transfer weight to draw an edge on map
TOP_EDGES         = 300     # max edges to render (performance)
SIM_STEPS         = 40      # number of SI time steps
SIM_RUNS          = 200     # Monte Carlo runs per seed

# Community colours — 11 communities (0-10) matching Louvain output
COMMUNITY_COLOURS = {
    0:  "#38bdf8",   # sky blue    — North West
    1:  "#fb923c",   # orange      — South East
    2:  "#4ade80",   # green       — London South
    3:  "#e879f9",   # pink        — East of England
    4:  "#ef4444",   # red         — West Midlands
    5:  "#facc15",   # yellow      — East Midlands
    6:  "#06b6d4",   # cyan        — South West
    7:  "#a78bfa",   # violet      — Yorkshire
    8:  "#34d399",   # emerald     — North East
    9:  "#f97316",   # deep orange — London North
    10: "#818cf8",   # indigo      — Yorkshire South
}

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")

coords_df = pd.read_csv(COORDS_FILE)
si_df     = pd.read_csv(SI_FILE)
comm_df   = pd.read_csv(COMMUNITIES_FILE)

# Merge all data on hospital code
# coords_df uses 'Code' as the hospital code column
coords_df = coords_df.rename(columns={"Code": "hospital_code"})

# Merge SI results
merged = coords_df.merge(
    si_df[["hospital", "infection_frequency", "mean_epidemic_size",
           "mean_epidemic_size_pct", "mean_step_to_infection"]],
    left_on="hospital_code", right_on="hospital", how="left"
).drop(columns=["hospital"], errors="ignore")

merged = merged.rename(columns={"hospital_code": "hospital"})

# Fill missing SI data with 0
for col in ["infection_frequency", "mean_epidemic_size",
            "mean_epidemic_size_pct", "mean_step_to_infection"]:
    merged[col] = merged[col].fillna(0)

print(f"  Hospitals loaded:        {len(merged)}")
print(f"  Communities:             {merged['community_id'].nunique()}")
print(f"  SI data matched:         {(merged['infection_frequency'] > 0).sum()}")

# Load transfers
transfers = None
if os.path.exists(TRANSFERS_FILE):
    transfers = pd.read_csv(TRANSFERS_FILE)
    transfers = transfers[transfers["weight"] >= EDGE_THRESHOLD].copy()
    print(f"  Transfer edges loaded:   {len(transfers)} (weight >= {EDGE_THRESHOLD})")
else:
    print(f"  ⚠ Transfer file not found at {TRANSFERS_FILE} — edge layer skipped")

# Load ICB boundaries
icb_geojson = None
if os.path.exists(ICB_FILE):
    with open(ICB_FILE) as f:
        icb_geojson = json.load(f)
    print(f"  ICB boundaries loaded:   {len(icb_geojson.get('features', []))} ICBs")
else:
    print(f"  ⚠ ICB boundaries not found at {ICB_FILE} — boundary layer skipped")

# ── 2. Colour helpers ──────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def risk_colour(value, vmin=0, vmax=1):
    """Interpolate from steel-blue (low) → amber (mid) → crimson (high)."""
    t = max(0, min(1, (value - vmin) / (vmax - vmin + 1e-9)))
    if t < 0.5:
        r = int(30  + (255 - 30)  * t * 2)
        g = int(100 + (180 - 100) * t * 2)
        b = int(200 + (50  - 200) * t * 2)
    else:
        r = 255
        g = int(180 - 180 * (t - 0.5) * 2)
        b = int(50  - 50  * (t - 0.5) * 2)
    return f"#{r:02x}{g:02x}{b:02x}"

def node_radius(row, mode="infection_freq"):
    if mode == "infection_freq":
        v = row.get("infection_frequency", 0)
        return 4 + v * 22
    elif mode == "in_degree":
        v = row.get("weighted_in_degree", 0)
        return 4 + math.log1p(v) * 1.8
    elif mode == "epidemic_size":
        v = row.get("mean_epidemic_size_pct", 0)
        return 4 + v * 0.8
    return 6

# ── 3. Build map ───────────────────────────────────────────────────────────────
print("Building map...")

m = folium.Map(
    location=[52.8, -1.8],
    zoom_start=6,
    tiles=None,
    prefer_canvas=True
)

# Tile layers
folium.TileLayer(
    "CartoDB dark_matter",
    name="Dark (default)",
    attr="© CartoDB"
).add_to(m)

folium.TileLayer(
    "CartoDB positron",
    name="Light",
    attr="© CartoDB"
).add_to(m)

folium.TileLayer(
    "OpenStreetMap",
    name="Street Map",
    attr="© OpenStreetMap"
).add_to(m)

# ── 4. ICB boundary layer ──────────────────────────────────────────────────────
if icb_geojson:
    icb_layer = folium.FeatureGroup(name="ICB Boundaries", show=True)
    folium.GeoJson(
        icb_geojson,
        name="ICB Boundaries",
        style_function=lambda f: {
            "fillColor":   "#38bdf8",
            "color":       "#38bdf8",
            "weight":      1.2,
            "fillOpacity": 0.06,
            "dashArray":   "4 4"
        },
        highlight_function=lambda f: {
            "fillColor":   "#38bdf8",
            "fillOpacity": 0.18,
            "weight":      2
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["ICB23NM"],
            aliases=["ICB:"],
            style="font-family: monospace; font-size: 11px;"
        )
    ).add_to(icb_layer)
    icb_layer.add_to(m)

# ── 5. Transfer edge layer ─────────────────────────────────────────────────────
if transfers is not None:
    # Build lookup: hospital code → lat/lon
    coord_lookup = merged.set_index("hospital")[["latitude", "longitude"]].to_dict("index")

    edge_layer = folium.FeatureGroup(name="Transfer Corridors (edges)", show=False)

    # Sort by weight descending, take top N
    top_transfers = transfers.nlargest(TOP_EDGES, "weight")
    w_max = top_transfers["weight"].max()

    for _, row in top_transfers.iterrows():
        s_info = coord_lookup.get(row["source"])
        t_info = coord_lookup.get(row["target"])
        if not s_info or not t_info:
            continue

        opacity = 0.15 + 0.55 * (row["weight"] / w_max)
        weight_px = 0.5 + 3.5 * (row["weight"] / w_max)

        folium.PolyLine(
            locations=[
                [s_info["latitude"],  s_info["longitude"]],
                [t_info["latitude"],  t_info["longitude"]]
            ],
            color="#fb923c",
            weight=weight_px,
            opacity=opacity,
            tooltip=(
                f"<b>{row.get('out_provider','')[:40]}</b>"
                f" → <b>{row.get('in_provider','')[:40]}</b><br>"
                f"Transfers: <b>{int(row['weight']):,}</b>"
            )
        ).add_to(edge_layer)

    edge_layer.add_to(m)

# ── 6. Hospital node layers ────────────────────────────────────────────────────

def make_popup(row):
    c = COMMUNITY_COLOURS.get(int(row["community_id"]), "#94a3b8")
    inf_pct     = row.get("infection_frequency", 0) * 100
    ep_size     = row.get("mean_epidemic_size_pct", 0)
    step_inf    = row.get("mean_step_to_infection", 0)
    in_deg      = int(row.get("weighted_in_degree", 0))
    bridges     = int(row.get("cross_community_bridges", 0))
    comm_id     = int(row.get("community_id", 0))

    # Risk bar
    bar_w = int(inf_pct)
    bar_col = risk_colour(row.get("infection_frequency", 0))

    html = f"""
    <div style="font-family:'Courier New',monospace;width:300px;
                background:#0d1117;border:1px solid {c};
                border-radius:6px;padding:14px;color:#e2e8f0;">
      <div style="font-size:13px;font-weight:bold;color:{c};margin-bottom:4px;">
        {row['hospital']}
      </div>
      <div style="font-size:10px;color:#94a3b8;margin-bottom:10px;line-height:1.4;">
        {row.get('Name','')[:60]}
      </div>
      <div style="font-size:9px;color:#64748b;margin-bottom:2px;">
        INFECTION FREQUENCY
      </div>
      <div style="background:#1e293b;border-radius:3px;height:8px;margin-bottom:8px;overflow:hidden;">
        <div style="width:{bar_w}%;height:100%;background:{bar_col};border-radius:3px;"></div>
      </div>
      <table style="width:100%;font-size:10px;border-collapse:collapse;">
        <tr>
          <td style="color:#64748b;padding:2px 0;">Infection frequency</td>
          <td style="color:{bar_col};text-align:right;font-weight:bold;">{inf_pct:.1f}%</td>
        </tr>
        <tr>
          <td style="color:#64748b;padding:2px 0;">Mean epidemic size</td>
          <td style="color:#e2e8f0;text-align:right;">{ep_size:.1f}% of network</td>
        </tr>
        <tr>
          <td style="color:#64748b;padding:2px 0;">Mean steps to infect</td>
          <td style="color:#e2e8f0;text-align:right;">{step_inf:.1f} steps</td>
        </tr>
        <tr>
          <td style="color:#64748b;padding:2px 0;">Transfers received</td>
          <td style="color:#e2e8f0;text-align:right;">{in_deg:,}</td>
        </tr>
        <tr>
          <td style="color:#64748b;padding:2px 0;">Cross-community bridges</td>
          <td style="color:#e2e8f0;text-align:right;">{bridges}</td>
        </tr>
        <tr>
          <td style="color:#64748b;padding:2px 0;">Louvain community</td>
          <td style="text-align:right;">
            <span style="background:{c};color:#000;padding:1px 6px;
                         border-radius:3px;font-weight:bold;">{comm_id}</span>
          </td>
        </tr>
        <tr>
          <td style="color:#64748b;padding:2px 0;">Postcode</td>
          <td style="color:#e2e8f0;text-align:right;">{row.get('Postcode','')}</td>
        </tr>
      </table>
    </div>
    """
    return folium.Popup(html, max_width=320)

# Layer A: coloured by Community
community_layer = folium.FeatureGroup(name="Hospitals — by Community", show=True)

# Layer B: coloured by Infection Frequency
si_layer = folium.FeatureGroup(name="Hospitals — SI Infection Frequency", show=False)

# Layer C: coloured by Mean Epidemic Size
epidemic_layer = folium.FeatureGroup(name="Hospitals — Mean Epidemic Size", show=False)

# Layer D: Bridge hospitals only
bridge_layer = folium.FeatureGroup(name="Cross-Community Bridge Hospitals", show=False)

inf_freq_max = merged["infection_frequency"].max()
ep_size_max  = merged["mean_epidemic_size_pct"].max()
bridge_q75   = merged["cross_community_bridges"].quantile(0.75)

for _, row in merged.iterrows():
    if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
        continue

    lat, lon    = row["latitude"], row["longitude"]
    comm_col    = COMMUNITY_COLOURS.get(int(row["community_id"]), "#94a3b8")
    inf_col     = risk_colour(row["infection_frequency"], 0, inf_freq_max)
    ep_col      = risk_colour(row["mean_epidemic_size_pct"], 0, ep_size_max)
    popup       = make_popup(row)
    tooltip_txt = (
        f"<b style='font-family:monospace'>{row['hospital']}</b><br>"
        f"<span style='font-family:monospace;font-size:10px;color:#94a3b8'>"
        f"{row.get('Name','')[:40]}</span>"
    )

    # Community layer
    folium.CircleMarker(
        location=[lat, lon],
        radius=node_radius(row, "infection_freq"),
        color=comm_col,
        fill=True,
        fill_color=comm_col,
        fill_opacity=0.85,
        weight=1.5,
        popup=popup,
        tooltip=tooltip_txt
    ).add_to(community_layer)

    # SI infection frequency layer
    folium.CircleMarker(
        location=[lat, lon],
        radius=node_radius(row, "infection_freq"),
        color=inf_col,
        fill=True,
        fill_color=inf_col,
        fill_opacity=0.9,
        weight=1,
        popup=popup,
        tooltip=tooltip_txt
    ).add_to(si_layer)

    # Epidemic size layer
    folium.CircleMarker(
        location=[lat, lon],
        radius=node_radius(row, "epidemic_size"),
        color=ep_col,
        fill=True,
        fill_color=ep_col,
        fill_opacity=0.9,
        weight=1,
        popup=popup,
        tooltip=tooltip_txt
    ).add_to(epidemic_layer)

    # Bridge layer — only high bridge hospitals
    if row["cross_community_bridges"] >= bridge_q75:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4 + math.log1p(row["cross_community_bridges"]) * 2,
            color="#facc15",
            fill=True,
            fill_color="#facc15",
            fill_opacity=0.9,
            weight=2,
            popup=popup,
            tooltip=tooltip_txt
        ).add_to(bridge_layer)

community_layer.add_to(m)
si_layer.add_to(m)
epidemic_layer.add_to(m)
bridge_layer.add_to(m)

# ── 7. Legend HTML ─────────────────────────────────────────────────────────────
community_legend_rows = "".join([
    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
    f'<div style="width:10px;height:10px;border-radius:50%;background:{col};flex-shrink:0;"></div>'
    f'<span style="font-size:9px;color:#94a3b8;">Community {cid}</span></div>'
    for cid, col in sorted(COMMUNITY_COLOURS.items())
])

legend_html = f"""
<div id="map-legend" style="
    position: fixed; bottom: 30px; right: 10px; z-index: 9999;
    background: rgba(13,17,23,0.96);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px; padding: 14px 16px;
    font-family: 'Courier New', monospace;
    color: #e2e8f0; min-width: 200px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6);">

  <div style="font-size:10px;letter-spacing:0.12em;color:#64748b;margin-bottom:10px;">
    NHS CYBER PROPAGATION MAP
  </div>

  <div style="font-size:9px;color:#64748b;margin-bottom:4px;">LOUVAIN COMMUNITIES</div>
  {community_legend_rows}

  <div style="border-top:1px solid rgba(255,255,255,0.08);margin:10px 0;"></div>

  <div style="font-size:9px;color:#64748b;margin-bottom:4px;">SI INFECTION RISK</div>
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">
    <div style="width:80px;height:8px;border-radius:3px;
      background:linear-gradient(90deg,#1e64c8,#ffb400,#dc143c);"></div>
    <span style="font-size:8px;color:#64748b;">Low → High</span>
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.08);margin:10px 0;"></div>

  <div style="font-size:9px;color:#64748b;margin-bottom:4px;">NODE SIZE</div>
  <div style="font-size:8px;color:#475569;line-height:1.6;">
    Community layer: infection frequency<br>
    SI layer: infection frequency<br>
    Epidemic layer: mean epidemic size<br>
    Bridge layer: bridge count
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.08);margin:10px 0;"></div>
  <div style="font-size:8px;color:#374151;">
    β = {BETA} | {len(merged)} hospitals<br>
    Edges ≥ {EDGE_THRESHOLD} transfers shown
  </div>
</div>
"""

m.get_root().html.add_child(folium.Element(legend_html))

# ── 8. Title bar HTML ──────────────────────────────────────────────────────────
title_html = """
<div style="
    position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
    z-index: 9999;
    background: rgba(13,17,23,0.94);
    border: 1px solid rgba(255,45,85,0.35);
    border-radius: 6px; padding: 8px 20px;
    font-family: 'Courier New', monospace;
    color: #e2e8f0; text-align: center;
    box-shadow: 0 2px 16px rgba(255,45,85,0.15);">
  <span style="font-size:12px;letter-spacing:0.15em;color:#ff2d55;">●</span>
  <span style="font-size:11px;letter-spacing:0.12em;margin:0 10px;">
    NHS HOSPITAL CYBERATTACK — SI PROPAGATION NETWORK
  </span>
  <span style="font-size:12px;letter-spacing:0.15em;color:#ff2d55;">●</span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ── 9. Layer control ───────────────────────────────────────────────────────────
folium.LayerControl(collapsed=False, position="topleft").add_to(m)

# ── 10. Save and open ──────────────────────────────────────────────────────────
m.save(OUTPUT_HTML)
print(f"\n✅  Map saved to:  {OUTPUT_HTML}")
print("Opening in browser...")
webbrowser.open(f"file://{OUTPUT_HTML}")
