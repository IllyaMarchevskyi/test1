"""
07d Baseline Results Map.

Baseline branch for map export and interactive visualization.
Keeps old 07d intact and reads baseline caches only.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Stop layer: pre-compute per-stop route/headway info for the interactive map
# ──────────────────────────────────────────────────────────────────────────────

def _compute_stops_layer_data(
    cfg: dict,
    bridge_path: str,
    osm_stops_path: str,
    easyway_path: str,
    bridge_metro_path: str,
    gmetro_path: str,
    metro_easyway_path: str,
    gtfs_dir: str,
) -> list:
    """
    Повертає список dict'ів для кожної зупинки (bus/trol/metro/tram):
      {lat, lon, name, color, transport_types, tooltip_html}

    Маршрути групуються за НАСТУПНОЮ зупинкою у послідовності (sorted by index),
    а не за мітками forward/backward і не за index+1 (індекси в easyway нещільні:
    1,4,6,...). Це правильно об'єднує паралельні маршрути.
    """
    import math
    import os
    import warnings
    import pandas as pd
    from collections import defaultdict

    warnings.filterwarnings("ignore")

    def _hhmm_to_sec(v: str) -> int:
        h, m = map(int, v.split(":"))
        return h * 3600 + m * 60

    peak_windows = [
        (_hhmm_to_sec(cfg["peak_hours"]["morning_start"]),
         _hhmm_to_sec(cfg["peak_hours"]["morning_end"])),
        (_hhmm_to_sec(cfg["peak_hours"]["evening_start"]),
         _hhmm_to_sec(cfg["peak_hours"]["evening_end"])),
    ]
    total_peak_min = cfg["peak_hours"]["total_peak_hours"] * 60  # 240 хв

    def in_peak(sec: int) -> bool:
        return any(s <= sec < e for s, e in peak_windows)

    def parse_schedules(value: str) -> list:
        times = []
        for raw in str(value).strip().split(","):
            raw = raw.strip()
            if not raw or raw == r"\N":
                continue
            parts = raw.split(":")
            h, m = int(parts[0]), int(parts[1])
            times.append(h * 3600 + m * 60)
        return times

    def headway(peak_count: int) -> float:
        """Full headway (interval between trips), minutes."""
        return total_peak_min / peak_count if peak_count > 0 else float("inf")

    def expected_wait(peak_count: int) -> float:
        """Expected wait = headway / 2 (uniform random arrival assumption)."""
        return headway(peak_count) / 2.0

    def combined_headway(peak_counts: list) -> float:
        total = sum(peak_counts)
        return total_peak_min / total if total > 0 else float("inf")

    def combined_expected_wait(peak_counts: list) -> float:
        return combined_headway(peak_counts) / 2.0

    def headway_html(peak_count: int, t_color: str) -> str:
        hw = headway(peak_count)
        ew = expected_wait(peak_count)
        if math.isinf(hw):
            return '<b>—</b>'
        if hw >= 90:
            return f'<b style="color:#888">{ew:.0f}&nbsp;хв</b><span style="color:#aaa;font-size:10px">&nbsp;(рідко)</span>'
        return f'<b>{ew:.1f}&nbsp;хв</b>'

    TRANSPORT_UA = {"bus": "Авт", "trol": "Тр", "tram": "Трм", "metro": "М"}
    TRANSPORT_COLORS = {
        "bus":   "#1565C0",
        "trol":  "#2E7D32",
        "tram":  "#BF360C",
        "metro": "#6A1B9A",
    }
    MIX_COLOR = "#424242"

    def stop_color(transport_set: set) -> str:
        types = transport_set - {""}
        if len(types) == 1:
            return TRANSPORT_COLORS.get(next(iter(types)), MIX_COLOR)
        return MIX_COLOR

    records = []
    # next_stop_map: (route_id_key, direction_key, stop_id) → next_stop_id
    # Built by sorting each route's stops by index and linking neighbours.
    # This is CORRECT even when indices are non-sequential (1, 4, 6, ...).
    next_stop_map: dict = {}
    stop_names_global: dict = {}

    def _build_next_stop_map_from_df(df: pd.DataFrame, rid_col: str, dir_col: str,
                                      idx_col: str, sid_col: str) -> None:
        for (rid, dirn), grp in df.groupby([rid_col, dir_col], sort=False):
            seq = grp.sort_values(idx_col)[sid_col].astype(str).tolist()
            for i in range(len(seq) - 1):
                next_stop_map[(rid, dirn, seq[i])] = seq[i + 1]

    # ── 1. Bus + trol (easyway) ───────────────────────────────────────────
    if os.path.exists(easyway_path):
        ew = pd.read_csv(easyway_path)
        ew = ew[ew["calendar"].isin(["Weekdays", "All Week"])].copy()
        ew = ew[ew["schedules"] != r"\N"].copy()
        ew["stop_id"] = ew["stop_id"].astype(str)
        ew["route_id_str"] = ew["route_id"].astype(str)
        ew["times"] = ew["schedules"].apply(parse_schedules)
        ew["peak_count"] = ew["times"].apply(lambda t: sum(1 for s in t if in_peak(s)))

        # Collect stop names
        for row in ew.itertuples(index=False):
            sname = str(getattr(row, "stop_name", ""))
            if sname and sname not in ("nan", ""):
                stop_names_global[str(row.stop_id)] = sname

        _build_next_stop_map_from_df(ew, "route_id_str", "direction", "index", "stop_id")

        for row in ew[ew["peak_count"] > 0].itertuples(index=False):
            records.append({
                "stop_id":   str(row.stop_id),
                "transport": str(row.transport),
                "route":     str(row.route),
                "route_id":  row.route_id_str,
                "direction": str(row.direction),
                "peak_count": int(row.peak_count),
                "stop_name": str(getattr(row, "stop_name", "")),
            })

    # ── 2. Metro (easyway_metro) ──────────────────────────────────────────
    if os.path.exists(metro_easyway_path):
        em = pd.read_csv(metro_easyway_path)
        em = em[em["calendar"].isin(["Weekdays", "All Week"])].copy()
        em = em[em["schedules"] != r"\N"].copy()
        em["stop_id"] = em["stop_id"].astype(str)
        em["route_id_str"] = "metro_" + em["route_id"].astype(str)
        em["times"] = em["schedules"].apply(parse_schedules)
        em["peak_count"] = em["times"].apply(lambda t: sum(1 for s in t if in_peak(s)))

        for row in em.itertuples(index=False):
            sname = str(getattr(row, "stop_name", ""))
            if sname and sname not in ("nan", ""):
                stop_names_global[str(row.stop_id)] = sname

        _build_next_stop_map_from_df(em, "route_id_str", "direction", "index", "stop_id")

        for row in em[em["peak_count"] > 0].itertuples(index=False):
            records.append({
                "stop_id":   str(row.stop_id),
                "transport": "metro",
                "route":     str(row.route),
                "route_id":  row.route_id_str,
                "direction": str(row.direction),
                "peak_count": int(row.peak_count),
                "stop_name": str(getattr(row, "stop_name", "")),
            })

    # ── 3. Tram (GTFS) ────────────────────────────────────────────────────
    routes_path     = os.path.join(gtfs_dir, "routes.txt")
    trips_path      = os.path.join(gtfs_dir, "trips.txt")
    stop_times_path = os.path.join(gtfs_dir, "stop_times.txt")
    stops_gtfs_path = os.path.join(gtfs_dir, "stops.txt")
    calendar_path   = os.path.join(gtfs_dir, "calendar.txt")
    g_stops_full_df = None
    if all(os.path.exists(p) for p in [routes_path, trips_path, stop_times_path,
                                        stops_gtfs_path, calendar_path]):
        g_routes  = pd.read_csv(routes_path)
        tram_rids = set(g_routes[g_routes["route_type"] == 0]["route_id"].astype(str))
        g_cal     = pd.read_csv(calendar_path)
        wday_sids = set(g_cal[g_cal["monday"] == 1]["service_id"].astype(str))
        g_trips   = pd.read_csv(trips_path)
        g_trips["route_id"]   = g_trips["route_id"].astype(str)
        g_trips["service_id"] = g_trips["service_id"].astype(str)
        tram_trips = g_trips[
            g_trips["route_id"].isin(tram_rids) &
            g_trips["service_id"].isin(wday_sids)
        ][["trip_id", "route_id", "direction_id"]].copy()
        tram_trip_ids = set(tram_trips["trip_id"].astype(str))
        tram_names = dict(zip(g_routes["route_id"].astype(str),
                              g_routes["route_short_name"].astype(str)))
        trip_meta = tram_trips.set_index("trip_id")[["route_id", "direction_id"]].to_dict("index")

        g_st = pd.read_csv(stop_times_path,
                           usecols=["trip_id", "arrival_time", "stop_id", "stop_sequence"])
        g_st["trip_id"] = g_st["trip_id"].astype(str)
        g_st = g_st[g_st["trip_id"].isin(tram_trip_ids)].copy()
        g_st["stop_id"]     = g_st["stop_id"].astype(str)
        g_st["route_id"]    = g_st["trip_id"].map(lambda t: "tram_" + str(trip_meta.get(t, {}).get("route_id", "")))
        g_st["direction_id"]= g_st["trip_id"].map(lambda t: trip_meta.get(t, {}).get("direction_id", 0))

        def _gtfs_sec(t: str) -> int:
            p = str(t).split(":")
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])

        g_st["sec"] = g_st["arrival_time"].apply(_gtfs_sec)

        _build_next_stop_map_from_df(g_st, "route_id", "direction_id", "stop_sequence", "stop_id")

        g_st_peak = g_st[g_st["sec"].apply(in_peak)].copy()
        tram_agg  = (
            g_st_peak.groupby(["stop_id", "route_id", "direction_id"])["trip_id"]
            .nunique().reset_index(name="peak_count")
        )
        g_stops_full_df = pd.read_csv(
            stops_gtfs_path, usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"]
        )
        g_stops_full_df["stop_id"] = g_stops_full_df["stop_id"].astype(str)
        gtfs_name = dict(zip(g_stops_full_df["stop_id"], g_stops_full_df["stop_name"]))
        for sid, sname in gtfs_name.items():
            if sname and str(sname) not in ("nan", ""):
                stop_names_global[sid] = str(sname)

        for row in tram_agg.itertuples(index=False):
            records.append({
                "stop_id":   str(row.stop_id),
                "transport": "tram",
                "route":     tram_names.get(str(row.route_id).replace("tram_", ""), str(row.route_id)),
                "route_id":  str(row.route_id),
                "direction": int(row.direction_id),
                "peak_count": int(row.peak_count),
                "stop_name": gtfs_name.get(str(row.stop_id), ""),
            })

    if not records:
        return []

    # ── 4. Attach next_stop and group per stop ────────────────────────────
    stop_routes: dict = defaultdict(list)
    for r in records:
        sid = r["stop_id"]
        key = (r["route_id"], r["direction"], sid)
        next_sid  = next_stop_map.get(key)
        r["next_stop_id"]   = next_sid
        r["next_stop_name"] = stop_names_global.get(next_sid, next_sid) if next_sid else None
        stop_routes[sid].append(r)
        sname = r.get("stop_name", "")
        if sname and sname not in ("nan", ""):
            stop_names_global[sid] = sname

    # ── 5. Load stop coordinates ──────────────────────────────────────────
    from shapely import wkt as _wkt
    import geopandas as gpd

    coord_frames = []
    if os.path.exists(bridge_path) and os.path.exists(osm_stops_path):
        bridge  = pd.read_csv(bridge_path, usecols=["osm_id", "stop_id"]).dropna()
        bridge["osm_id"]  = bridge["osm_id"].astype(str)
        bridge["stop_id"] = bridge["stop_id"].astype(str)
        osm_raw = pd.read_csv(osm_stops_path).dropna(subset=["geometry"]).copy()
        osm_raw["geometry"] = osm_raw["geometry"].map(_wkt.loads)
        osm_raw["osm_id"]   = osm_raw.index.astype(str)
        osm_gdf = gpd.GeoDataFrame(osm_raw, geometry="geometry", crs="EPSG:4326")
        osm_gdf = osm_gdf[osm_gdf.geometry.geom_type == "Point"].copy()
        osm_gdf["lon"] = osm_gdf.geometry.x
        osm_gdf["lat"] = osm_gdf.geometry.y
        coord_frames.append(
            bridge.merge(osm_gdf[["osm_id", "lon", "lat"]], on="osm_id", how="left")
            [["stop_id", "lon", "lat"]]
        )

    if os.path.exists(bridge_metro_path) and os.path.exists(gmetro_path):
        bm     = pd.read_csv(bridge_metro_path, usecols=["osm_id", "stop_id"]).dropna()
        bm["osm_id"]  = bm["osm_id"].astype(str)
        bm["stop_id"] = bm["stop_id"].astype(str)
        gm_raw = pd.read_csv(gmetro_path).dropna(subset=["geometry"]).copy()
        gm_raw["geometry"] = gm_raw["geometry"].map(_wkt.loads)
        gm_raw["osm_id"]   = gm_raw.index.astype(str)
        gm_gdf = gpd.GeoDataFrame(gm_raw, geometry="geometry", crs="EPSG:4326")
        gm_gdf = gm_gdf[gm_gdf.geometry.geom_type == "Point"].copy()
        gm_gdf["lon"] = gm_gdf.geometry.x
        gm_gdf["lat"] = gm_gdf.geometry.y
        coord_frames.append(
            bm.merge(gm_gdf[["osm_id", "lon", "lat"]], on="osm_id", how="left")
            [["stop_id", "lon", "lat"]]
        )

    if g_stops_full_df is not None:
        coord_frames.append(
            g_stops_full_df.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
            [["stop_id", "lon", "lat"]]
        )

    if not coord_frames:
        return []

    coord_dict = {
        row.stop_id: (float(row.lat), float(row.lon))
        for row in (
            pd.concat(coord_frames, ignore_index=True)
            .dropna(subset=["lon", "lat"])
            .drop_duplicates(subset=["stop_id"])
            .itertuples()
        )
    }

    # ── 6. Build tooltip HTML per stop ────────────────────────────────────
    result = []
    for sid, routes in stop_routes.items():
        coords = coord_dict.get(sid)
        if coords is None:
            continue
        lat, lon = coords
        name = stop_names_global.get(sid, f"Зупинка {sid}")
        transport_types = set(r["transport"] for r in routes)
        color = stop_color(transport_types)

        # Group by next_stop_id (= real direction grouper)
        dir_groups: dict = defaultdict(list)
        for r in routes:
            gkey  = r["next_stop_id"] or "__terminus__"
            label = f"{TRANSPORT_UA.get(r['transport'], r['transport'])} {r['route']}"
            dir_groups[gkey].append({
                "label":          label,
                "transport":      r["transport"],
                "peak_count":     r["peak_count"],
                "next_stop_name": r["next_stop_name"],
            })

        sorted_groups = sorted(
            dir_groups.items(),
            key=lambda kv: -sum(x["peak_count"] for x in kv[1]),
        )

        type_badges = "".join(
            f'<span style="color:{TRANSPORT_COLORS.get(t, MIX_COLOR)};font-size:11px">&#9679;</span>'
            f'<span style="font-size:11px;color:#555">&nbsp;{t}</span>&ensp;'
            for t in sorted(transport_types)
        )
        html_parts = [
            '<div style="font-family:Arial,sans-serif;font-size:12px;'
            'min-width:185px;max-width:290px">',
            f'<b style="font-size:13px">{name}</b>'
            f'<span style="color:#999;font-size:10px;margin-left:5px">#{sid}</span>',
            f'<div style="margin:2px 0 4px">{type_badges}</div>',
            '<hr style="margin:3px 0;border-color:#eee">',
        ]

        for gkey, grp in sorted_groups:
            next_name = grp[0]["next_stop_name"]
            if gkey == "__terminus__":
                dir_lbl = "кінцева зупинка"
            elif next_name:
                dir_lbl = f"→ {next_name}"
            else:
                dir_lbl = f"→ {gkey}"

            html_parts.append(f'<b style="color:#333">{dir_lbl}:</b><br>')
            for item in sorted(grp, key=lambda x: -x["peak_count"]):
                t_color = TRANSPORT_COLORS.get(item["transport"], MIX_COLOR)
                hw_html = headway_html(item["peak_count"], t_color)
                html_parts.append(
                    f'&nbsp;<span style="color:{t_color}">&#9679;</span>'
                    f'&nbsp;{item["label"]}: {hw_html}<br>'
                )

            if len(grp) > 1:
                # Exclude "рідко" routes from combined calc if they'd distort it
                regular = [x["peak_count"] for x in grp if headway(x["peak_count"]) < 90]
                rare    = [x["peak_count"] for x in grp if headway(x["peak_count"]) >= 90]
                if regular:
                    comb_wait = combined_expected_wait(regular + rare)
                    html_parts.append(
                        f'&nbsp;<i style="color:#555">Разом: '
                        f'<b style="color:#000">{comb_wait:.1f}&nbsp;хв</b></i><br>'
                    )

        html_parts.append("</div>")
        result.append({
            "sid": sid,
            "lat": lat,
            "lon": lon,
            "name": name,
            "color": color,
            "transport_types": sorted(transport_types),
            "tooltip_html": "".join(html_parts),
        })

    print(f"07d_base: зупинок для карти: {len(result):,}")
    return result


# ── Multiprocessing workers for walk-path computation ──────────────────────────
_walk_G = None          # graph loaded once per worker process
_walk_nlat: dict = {}   # node → lat
_walk_nlon: dict = {}   # node → lon
_walk_sdict: dict = {}  # stop_id(str) → graph node id


def _walk_init(graph_path: str, sdict_items: list) -> None:
    """Initializer: load walk graph and stop→node mapping once per worker.

    With fork context the graph is already inherited from the parent process;
    _walk_G will be non-None so we skip the expensive pickle load.
    With spawn context _walk_G is None and we load from disk as before.
    """
    global _walk_G, _walk_nlat, _walk_nlon, _walk_sdict
    _walk_sdict = dict(sdict_items)
    if _walk_G is not None:
        return  # fork: globals inherited from parent — nothing else to do
    import pickle
    with open(graph_path, "rb") as fh:
        _walk_G = pickle.load(fh)
    _walk_nlat = {n: d["y"] for n, d in _walk_G.nodes(data=True)}
    _walk_nlon = {n: d["x"] for n, d in _walk_G.nodes(data=True)}


def _walk_task(args: tuple) -> str:
    """
    Worker: compute Dijkstra walk paths for one facility, patch its GeoJSON.

    args = (fac_id, fac_node, rows, geojson_path, CUTOFF_FAC, CUTOFF_STOP)
    rows = list of (bid, bld_node, src_stop_id, dst_stop_id, mode)

    nx.single_source_dijkstra returns (dist_dict, paths_dict) where
    paths_dict[v] = [source, ..., v] — the full node list, NOT a predecessors dict.
    We use paths_dict[v] directly (+ reverse) instead of nx.reconstruct_path.
    """
    import json
    import os
    import networkx as nx

    fac_id, fac_node, rows, geojson_path, CUTOFF_FAC, CUTOFF_STOP = args
    G = _walk_G
    nlat = _walk_nlat
    nlon = _walk_nlon
    sdict = _walk_sdict

    def pts(path: list, step: int = 3) -> list:
        """Simplify node path to [[lat,lon],...] keeping every `step`-th middle node.
        4 decimal places ≈ 11 m precision — sufficient for display, ~20 % smaller JSON.
        All graph nodes have lat/lon so no membership check needed.
        """
        if not path:
            return []
        nds = path if len(path) <= 4 else path[:1] + path[1:-1:step] + path[-1:]
        return [[round(nlat[n], 4), round(nlon[n], 4)] for n in nds]

    # pf[v] = [fac_node, ..., v]  (full path from facility to v)
    try:
        _, pf = nx.single_source_dijkstra(G, fac_node, cutoff=CUTOFF_FAC, weight="length")
    except Exception:
        return fac_id

    # ps[sid][v] = [stop_node, ..., v]  (full path from boarding stop to v)
    unique_src = {r[2] for r in rows if r[2]}
    ps: dict = {}
    for sid in unique_src:
        sn = sdict.get(sid)
        if sn and sn in G:
            try:
                _, ps[sid] = nx.single_source_dijkstra(G, sn, cutoff=CUTOFF_STOP, weight="length")
            except Exception:
                pass

    bpaths: dict = {}
    for bid, bnode, src, dst, mode in rows:
        if bnode is None:
            continue
        p: dict = {}
        if mode == "walk":
            # pf[bnode] = [fac_node, ..., bnode]; [::-1] → [bnode, ..., fac_node]
            if bnode in pf:
                c = pts(pf[bnode][::-1])
                if c:
                    p["wp"] = c
        elif mode == "transit":
            # walk_in: ps[src][bnode][::-1] → [bnode, ..., stop_node]
            if src and src in ps and bnode in ps[src]:
                c = pts(ps[src][bnode][::-1])
                if c:
                    p["wi"] = c
            # walk_out: pf[dn][::-1] → [dn, ..., fac_node]
            if dst:
                dn = sdict.get(dst)
                if dn is not None and dn in pf:
                    c = pts(pf[dn][::-1])
                    if c:
                        p["wo"] = c
        if p:
            bpaths[bid] = p

    if not bpaths or not os.path.exists(geojson_path):
        return fac_id

    with open(geojson_path, encoding="utf-8") as fh:
        gj = json.load(fh)
    for feat in gj.get("features", []):
        bid = feat.get("properties", {}).get("building_id")
        if bid is not None and int(bid) in bpaths:
            feat["properties"]["_paths"] = bpaths[int(bid)]
    with open(geojson_path, "w", encoding="utf-8") as fh:
        json.dump(gj, fh, separators=(",", ":"), ensure_ascii=False)

    return fac_id


def _add_walk_paths_to_geojson(
    geojson_dir: str,
    graph_path: str,
    catchment_buildings,   # pd.DataFrame
    facilities_df,         # pd.DataFrame  with facility_id, lat, lon
    buildings_gdf,         # gpd.GeoDataFrame with building_id, geometry (any CRS)
    stop_coords_df,        # pd.DataFrame with stop_id, lat, lon
) -> None:
    """
    Post-processes per-facility GeoJSON files to add actual Dijkstra walk paths.
    Adds '_paths' property to each building feature:
      - walk mode:    {'wp': [[lat,lon],...]}          building→facility
      - transit mode: {'wi': [...], 'wo': [...]}       building→boarding, alighting→facility
    Runs in parallel (one process per CPU core) via ProcessPoolExecutor.
    """
    import math
    import multiprocessing
    import os
    import pickle
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import osmnx as ox
    from tqdm.auto import tqdm

    if not os.path.exists(graph_path):
        print(f"_add_walk_paths: граф не знайдено ({graph_path}), пропускаємо.")
        return

    print("_add_walk_paths: завантажуємо граф...")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Convert buildings to EPSG:4326 for nearest_nodes
    print("_add_walk_paths: знаходимо вузли графу для будинків...")
    bld_4326 = buildings_gdf.to_crs("EPSG:4326")
    bld_graph_nodes = ox.distance.nearest_nodes(
        G, X=bld_4326.geometry.x.values, Y=bld_4326.geometry.y.values
    )
    bld_node_dict = {
        int(bid): int(n)
        for bid, n in zip(buildings_gdf["building_id"].values, bld_graph_nodes)
    }

    # Nearest nodes for stops
    print("_add_walk_paths: знаходимо вузли графу для зупинок...")
    stop_coords_df = stop_coords_df.dropna(subset=["lat", "lon"])
    stop_node_dict = {
        str(row.stop_id): int(n)
        for row, n in zip(
            stop_coords_df.itertuples(index=False),
            ox.distance.nearest_nodes(
                G, X=stop_coords_df["lon"].values, Y=stop_coords_df["lat"].values
            ),
        )
    }

    # Nearest nodes for facilities (batched call, not per-row)
    fac_lons = facilities_df["lon"].astype(float).values
    fac_lats = facilities_df["lat"].astype(float).values
    fac_node_dict = {
        str(fid): int(n)
        for fid, n in zip(
            facilities_df["facility_id"].values,
            ox.distance.nearest_nodes(G, X=fac_lons, Y=fac_lats),
        )
    }

    # Pre-populate module-level worker globals.
    # With fork the child processes inherit these directly (no disk reload).
    # With spawn _walk_init sees _walk_G=None and loads from disk as normal.
    global _walk_G, _walk_nlat, _walk_nlon, _walk_sdict
    _walk_G    = G
    _walk_nlat = {n: d["y"] for n, d in G.nodes(data=True)}
    _walk_nlon = {n: d["x"] for n, d in G.nodes(data=True)}
    _walk_sdict = stop_node_dict

    CUTOFF_FAC  = 30 * 75   # 30 min walk at 75 m/min
    CUTOFF_STOP = 15 * 75   # 15 min walk for boarding segment

    # Build task list (per-facility)
    cb_by_fac = catchment_buildings.groupby("facility_id", sort=False)
    tasks = []
    _nan = float("nan")
    for fac_id, fac_buildings in cb_by_fac:
        geojson_path_fac = os.path.join(geojson_dir, f"{fac_id}.geojson")
        if not os.path.exists(geojson_path_fac):
            continue
        fac_node = fac_node_dict.get(str(fac_id))
        if fac_node is None:
            continue
        rows = []
        for row in fac_buildings.itertuples(index=False):
            bid   = int(row.building_id)
            bnode = bld_node_dict.get(bid)
            mode  = getattr(row, "peak_mode", None)
            src_v = row.peak_source_stop
            dst_v = row.peak_dest_stop
            src = str(src_v) if not (isinstance(src_v, float) and math.isnan(src_v)) else None
            dst = str(dst_v) if not (isinstance(dst_v, float) and math.isnan(dst_v)) else None
            rows.append((bid, bnode, src, dst, mode))
        if rows:
            tasks.append((str(fac_id), fac_node, rows, geojson_path_fac, CUTOFF_FAC, CUTOFF_STOP))

    n_cpu = min(max(1, (os.cpu_count() or 4) - 1), 8)
    print(f"_add_walk_paths: {len(tasks)} закладів, {n_cpu} workers...")

    # fork: child processes inherit _walk_G (copy-on-write) — no graph reload.
    # spawn (Windows): _walk_init loads graph from disk as fallback.
    ctx_method = "fork" if sys.platform != "win32" else "spawn"
    ctx = multiprocessing.get_context(ctx_method)
    sdict_items = list(stop_node_dict.items())

    with ProcessPoolExecutor(
        max_workers=n_cpu,
        mp_context=ctx,
        initializer=_walk_init,
        initargs=(graph_path, sdict_items),
    ) as pool:
        futs = {pool.submit(_walk_task, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Walk paths per facility"):
            try:
                fut.result()
            except Exception as exc:
                print(f"  worker error [{futs[fut]}]: {exc}")

    print("_add_walk_paths: готово.")


def run() -> None:
    from config_loader import cfg
    import json
    import os
    import warnings

    import folium
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import pandas as pd
    from branca.element import Element
    from shapely import wkt

    from utils.catchment_map_export import export_catchment_map_data, read_parquet_with_progress

    warnings.filterwarnings("ignore")

    T_SHORT = cfg["catchment"]["threshold_short_min"]
    T_LONG = cfg["catchment"]["threshold_long_min"]
    GRP_WALK_SHORT = f"walk_{T_SHORT}min"
    GRP_TRANSIT_SHORT = f"transit_{T_SHORT}min"
    GRP_WALK_LONG = f"walk_{T_LONG}min"
    GRP_TRANSIT_LONG = f"transit_{T_LONG}min"

    PROCESSED_DIR = "./data/processed"
    OUTPUTS_DIR = "./data/outputs"
    MAP_BUILDINGS_DIR = f"{OUTPUTS_DIR}/map_buildings_baseline"
    OUT_JSON = f"{PROCESSED_DIR}/map_data_baseline.json"
    OUT_HTML = f"{OUTPUTS_DIR}/map_catchment_interactive_baseline.html"
    OUT_PNG = f"{OUTPUTS_DIR}/output.png"
    HTML_REL_GEOJSON_DIR = "map_buildings_baseline"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(MAP_BUILDINGS_DIR, exist_ok=True)

    catchment_csv = f"{PROCESSED_DIR}/catchment_results_baseline.csv"
    catchment_buildings_path = f"{PROCESSED_DIR}/catchment_buildings_baseline.parquet"
    building_weights_path = f"{PROCESSED_DIR}/building_weights_baseline.parquet"
    buildings_path = "../data/processed/buildings.parquet"
    scores_path = cfg["paths"]["scores"]
    bridge_path = "../gtfs_static/osm_easyway_data.csv"
    bridge_metro_path = "../gtfs_static/osm_easyway_metro_data.csv"
    osm_stops_path = "../gtfs_static/osm_stops.csv"
    gmetro_path = "../gtfs_static/gmetro.csv"

    required = {
        "catchment_results_baseline": catchment_csv,
        "catchment_buildings_baseline": catchment_buildings_path,
        "buildings": buildings_path,
        "scores": scores_path,
    }
    missing = [label for label, path in required.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 07d_base: {missing}")

    print("07d_base: завантаження baseline-даних...")
    catchment_results = pd.read_csv(catchment_csv)
    catchment_buildings = read_parquet_with_progress(
        catchment_buildings_path,
        desc="Завантаження baseline catchment_buildings",
    )
    if os.path.exists(building_weights_path):
        building_weights = pd.read_parquet(building_weights_path, columns=["building_id", "levels_display"])
        building_weights = building_weights.rename(columns={"levels_display": "building_levels"})
        catchment_buildings = catchment_buildings.merge(building_weights, on="building_id", how="left")
        print(f"  building_weights:    {len(building_weights):,} будинків")
    buildings = gpd.read_parquet(buildings_path, columns=["building_id", "geometry"])
    scores = pd.read_csv(scores_path, usecols=["facility_id", "facility_type", "name", "lat", "lon"])
    facilities = scores[["facility_id", "facility_type", "name", "lat", "lon"]].copy()
    stop_coords = None
    if os.path.exists(bridge_path) and os.path.exists(osm_stops_path):
        bridge = pd.read_csv(bridge_path, usecols=["osm_id", "stop_id"]).dropna()
        bridge["osm_id"] = bridge["osm_id"].astype(str)
        bridge["stop_id"] = bridge["stop_id"].astype(str)
        osm_stops_raw = pd.read_csv(osm_stops_path).dropna(subset=["geometry"]).copy()
        osm_stops_raw["geometry"] = osm_stops_raw["geometry"].map(wkt.loads)
        osm_stops_raw["osm_id"] = osm_stops_raw.index.astype(str)
        osm_stops_surface = gpd.GeoDataFrame(osm_stops_raw, geometry="geometry", crs="EPSG:4326")
        osm_stops_surface = osm_stops_surface[osm_stops_surface.geometry.geom_type == "Point"].copy()
        osm_stops_surface["lon"] = osm_stops_surface.geometry.x
        osm_stops_surface["lat"] = osm_stops_surface.geometry.y
        stop_coord_frames = [
            bridge.merge(
                osm_stops_surface[["osm_id", "lon", "lat"]],
                on="osm_id",
                how="left",
            )[["stop_id", "lon", "lat"]]
        ]

        if os.path.exists(bridge_metro_path) and os.path.exists(gmetro_path):
            bridge_metro = pd.read_csv(bridge_metro_path, usecols=["osm_id", "stop_id"]).dropna()
            bridge_metro["osm_id"] = bridge_metro["osm_id"].astype(str)
            bridge_metro["stop_id"] = bridge_metro["stop_id"].astype(str)
            gmetro_raw = pd.read_csv(gmetro_path).dropna(subset=["geometry"]).copy()
            gmetro_raw["geometry"] = gmetro_raw["geometry"].map(wkt.loads)
            gmetro_raw["osm_id"] = gmetro_raw.index.astype(str)
            gmetro = gpd.GeoDataFrame(gmetro_raw, geometry="geometry", crs="EPSG:4326")
            gmetro = gmetro[gmetro.geometry.geom_type == "Point"].copy()
            gmetro["lon"] = gmetro.geometry.x
            gmetro["lat"] = gmetro.geometry.y
            stop_coord_frames.append(
                bridge_metro.merge(
                    gmetro[["osm_id", "lon", "lat"]],
                    on="osm_id",
                    how="left",
                )[["stop_id", "lon", "lat"]]
            )
        else:
            print("07d_base: metro-місток не знайдено, preview метро-зупинок буде вимкнено.")

        stop_coords = pd.concat(stop_coord_frames, ignore_index=True)
        stop_coords = stop_coords.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["stop_id"]).reset_index(drop=True)
    else:
        print("07d_base: bridge/osm_stops не знайдені, preview зупинок буде вимкнено.")

    # Compute stops layer data (all stops with route/headway info)
    gtfs_dir = "../gtfs_static"
    stops_layer_data = _compute_stops_layer_data(
        cfg=cfg,
        bridge_path=bridge_path,
        osm_stops_path=osm_stops_path,
        easyway_path=bridge_path.replace("osm_easyway_data.csv", "easyway_routes.csv"),
        bridge_metro_path=bridge_metro_path,
        gmetro_path=gmetro_path,
        metro_easyway_path=bridge_metro_path.replace("osm_easyway_metro_data.csv", "easyway_metro.csv"),
        gtfs_dir=gtfs_dir,
    )

    print(f"  catchment_results:   {len(catchment_results)} закладів")
    print(f"  catchment_buildings: {len(catchment_buildings):,} записів")
    print(f"  buildings:           {len(buildings):,} будинків")
    print(f"  facilities:          {len(facilities)} закладів")

    # ── Stop-to-stop route lookup (for interactive routing on map) ────────
    visible_stop_ids = {s["name"].split(" ")[-1] if s["name"].startswith("Зупинка ") else None
                        for s in stops_layer_data}
    # More reliable: collect actual stop IDs from stops_layer_data via STOPS_DATA
    # stops_layer_data items have no explicit stop_id field, so use coord-based sid
    # Instead build from the parquet directly, all visible stops are in stops_layer_data
    route_lookup: dict = {}
    reach_path = f"{PROCESSED_DIR}/stop_reachability_peak_baseline.parquet"
    wait_path  = f"{PROCESSED_DIR}/wait_times_peak_baseline.parquet"
    if os.path.exists(reach_path) and os.path.exists(wait_path):
        from collections import defaultdict
        reach_df = pd.read_parquet(reach_path,
                                   columns=["stop_A", "stop_B", "transit_min", "route_options", "transport"])
        wait_df  = pd.read_parquet(wait_path,
                                   columns=["stop_A", "stop_B", "adj_wait_min"])
        reach_df["stop_A"] = reach_df["stop_A"].astype(str)
        reach_df["stop_B"] = reach_df["stop_B"].astype(str)
        wait_df["stop_A"]  = wait_df["stop_A"].astype(str)
        wait_df["stop_B"]  = wait_df["stop_B"].astype(str)
        merged = reach_df.merge(wait_df, on=["stop_A", "stop_B"], how="left")
        _rl: dict = defaultdict(dict)
        for row in merged.itertuples(index=False):
            w = float(row.adj_wait_min)
            _rl[row.stop_A][row.stop_B] = {
                "t": round(float(row.transit_min), 1),
                "w": round(w if w == w else 0.0, 1),
                "r": str(row.route_options) if row.route_options else "",
                "tp": str(row.transport) if row.transport else "",
            }
        route_lookup = dict(_rl)
        print(f"  route_lookup:        {len(merged):,} пар зупинок")
    else:
        print("  route_lookup: parquet не знайдено, маршрутизацію між зупинками вимкнено.")

    # ── Nearest stops per building (for walk-mode tooltip) ────────────────
    nearest_stops_lookup: dict = {}
    s2b_path = f"{PROCESSED_DIR}/stop_to_bld_short_baseline.parquet"
    if os.path.exists(s2b_path):
        from itertools import islice
        s2b_df = pd.read_parquet(s2b_path, columns=["stop_id", "building_id", "walk_min"])
        for bid, grp in s2b_df.sort_values("walk_min").groupby("building_id"):
            nearest_stops_lookup[str(bid)] = [
                {"s": str(r.stop_id), "w": round(float(r.walk_min), 1)}
                for r in islice(grp.itertuples(index=False), 3)
            ]
        print(f"  nearest_stops:       {len(nearest_stops_lookup):,} будинків")
    else:
        print("  nearest_stops: parquet не знайдено.")

    col_pk_sh = f"peak_total_{T_SHORT}min"
    col_pk_lg = f"peak_total_{T_LONG}min"
    col_op_sh = f"offpeak_total_{T_SHORT}min"
    col_op_lg = f"offpeak_total_{T_LONG}min"
    hosp = catchment_results[catchment_results["facility_type"] == "hospital"]
    school = catchment_results[catchment_results["facility_type"] == "school"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, facility_type, label in [
        (axes[0], "hospital", "Лікарні"),
        (axes[1], "school", "Школи"),
    ]:
        subset = catchment_results[catchment_results["facility_type"] == facility_type]
        cols = [
            f"peak_{GRP_WALK_SHORT}",
            f"peak_{GRP_TRANSIT_SHORT}",
            f"peak_{GRP_WALK_LONG}",
            f"peak_{GRP_TRANSIT_LONG}",
        ]
        means = subset[cols].mean()
        colors = ["#1FFF2E", "#EB9328", "#1B6B23", "#FF0000"]
        ax.bar(range(len(cols)), means.values, color=colors, edgecolor="white")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(
            [f"Пішки\n{T_SHORT}", f"Транспорт\n{T_SHORT}", f"Пішки\n{T_LONG}", f"Транспорт\n{T_LONG}"],
            fontsize=9,
        )
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("Будинки")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Статичний графік збережено: {OUT_PNG}")

    export_kwargs = {
        "catchment_results": catchment_results,
        "catchment_buildings": catchment_buildings,
        "buildings": buildings,
        "facilities": facilities,
        "stop_coords": stop_coords,
        "output_json_path": OUT_JSON,
        "output_geojson_dir": MAP_BUILDINGS_DIR,
        "html_rel_geojson_dir": HTML_REL_GEOJSON_DIR,
        "t_short": T_SHORT,
        "t_long": T_LONG,
        "grp_walk_short": GRP_WALK_SHORT,
        "grp_transit_short": GRP_TRANSIT_SHORT,
        "grp_walk_long": GRP_WALK_LONG,
        "grp_transit_long": GRP_TRANSIT_LONG,
        "parallel_workers": max(1, int(cfg.get("rl", {}).get("n_envs", min(8, os.cpu_count() or 1)))),
    }
    try:
        payload = export_catchment_map_data(**export_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "stop_coords" in message:
            print("Увага: helper без підтримки stop_coords, запускаємо сумісний режим без preview зупинок.")
            export_kwargs.pop("stop_coords", None)
            try:
                payload = export_catchment_map_data(**export_kwargs)
            except TypeError as inner_exc:
                if "parallel_workers" not in str(inner_exc):
                    raise
                print("Увага: helper без підтримки parallel_workers, запускаємо повністю сумісний режим.")
                export_kwargs.pop("parallel_workers", None)
                payload = export_catchment_map_data(**export_kwargs)
        elif "parallel_workers" in message:
            print("Увага: helper без підтримки parallel_workers, запускаємо сумісний режим.")
            export_kwargs.pop("parallel_workers", None)
            payload = export_catchment_map_data(**export_kwargs)
        else:
            raise

    with open(OUT_JSON, encoding="utf-8") as f:
        map_data = json.load(f)

    print(f"JSON baseline збережено: {OUT_JSON}")

    # ── Add actual walk paths to per-facility GeoJSON files ───────────────
    walk_graph_path = "../data/osm/kyiv_walk_graph.pkl"
    if stop_coords is not None and os.path.exists(walk_graph_path):
        _add_walk_paths_to_geojson(
            geojson_dir=MAP_BUILDINGS_DIR,
            graph_path=walk_graph_path,
            catchment_buildings=catchment_buildings[
                ["facility_id", "building_id", "peak_mode",
                 "peak_source_stop", "peak_dest_stop"]
            ],
            facilities_df=facilities[["facility_id", "lat", "lon"]],
            buildings_gdf=buildings,
            stop_coords_df=stop_coords,
        )
    else:
        print("07d_base: граф або координати зупинок відсутні — шляхи не обраховано.")
    print(f"  Закладів:        {len(payload['facilities'])}")
    print(f"  Будинків всього: {payload['_total_buildings']:,}")
    print(f"  GeoJSON-каталог: {payload['_geojson_dir']}")

    m = folium.Map(
        location=[cfg["city"]["center_lat"], cfg["city"]["center_lon"]],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    layer_hospitals = folium.FeatureGroup(name="Лікарні", show=True)
    layer_schools = folium.FeatureGroup(name="Школи", show=True)

    for fac in map_data["facilities"]:
        is_hosp = fac["type"] == "hospital"
        color = "#C0392B" if is_hosp else "#2980B9"
        icon = "+" if is_hosp else "B"
        layer = layer_hospitals if is_hosp else layer_schools

        popup_html = (
            f"<div style='width:230px;font-family:Arial,sans-serif;font-size:13px'>"
            f"<b style='font-size:14px'>{fac['name'][:55]}</b><br>"
            f"<span style='color:#666'>{'Лікарня' if is_hosp else 'Школа'}</span><br>"
            f"<span style='color:#666'>ID: {fac['id']}</span>"
            f"<hr style='margin:6px 0'>"
            f"<b>Пік:</b><br>"
            f"&nbsp;Пішки {T_SHORT} хв: <b>{fac['stats']['peak_walk_short']:,}</b><br>"
            f"&nbsp;Транспорт {T_SHORT} хв: <b>{fac['stats']['peak_transit_short']:,}</b><br>"
            f"&nbsp;Пішки {T_LONG} хв: <b>{fac['stats']['peak_walk_long']:,}</b><br>"
            f"&nbsp;Транспорт {T_LONG} хв: <b>{fac['stats']['peak_transit_long']:,}</b>"
            f"<hr style='margin:6px 0'>"
            f"<b>Міжпік:</b><br>"
            f"&nbsp;Пішки {T_SHORT} хв: <b>{fac['stats']['offpeak_walk_short']:,}</b><br>"
            f"&nbsp;Транспорт {T_SHORT} хв: <b>{fac['stats']['offpeak_transit_short']:,}</b><br>"
            f"&nbsp;Пішки {T_LONG} хв: <b>{fac['stats']['offpeak_walk_long']:,}</b><br>"
            f"&nbsp;Транспорт {T_LONG} хв: <b>{fac['stats']['offpeak_transit_long']:,}</b>"
            f"<hr style='margin:6px 0'>"
            f"<button onclick='showBuildings(\"{fac['id']}\")' "
            f"style='width:100%;padding:5px 0;background:{color};color:white;"
            f"border:none;border-radius:4px;cursor:pointer;font-size:12px'>"
            f"Показати будинки"
            f"</button>"
            f"</div>"
        )

        marker_html = (
            f'<div style="background:{color};color:white;'
            f'border-radius:50%;width:24px;height:24px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:13px;box-shadow:0 1px 3px rgba(0,0,0,.4);'
            f'border:2px solid white">{icon}</div>'
        )

        facility_marker = folium.Marker(
            location=[fac["lat"], fac["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=fac["name"][:40],
            icon=folium.DivIcon(html=marker_html, icon_size=(24, 24), icon_anchor=(12, 12)),
        )
        facility_marker.add_to(layer)

    layer_hospitals.add_to(m)
    layer_schools.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    map_var = m.get_name()
    grp_walk_short = map_data["grp_walk_short"]
    grp_transit_short = map_data["grp_transit_short"]
    grp_walk_long = map_data["grp_walk_long"]
    grp_transit_long = map_data["grp_transit_long"]

    js_code = f"""
    const MAP_DATA = {json.dumps(map_data, ensure_ascii=False)};
    const COLORS = {{
        '{grp_walk_short}': '#1FFF2E',
        '{grp_transit_short}': '#EB9328',
        '{grp_walk_long}': '#1B6B23',
        '{grp_transit_long}': '#FF0000',
    }};

    let currentMode = 'peak';
    let buildingsLayer = null;
    let currentFacilityId = null;
    const facilityBuildingsCache = {{}};
    let buildingsRenderer = null;
    let selectedStopsLayer = null;
    let currentFacilityHighlight = null;

    function getMapObject() {{
        return {map_var};
    }}

    function clearStopPreview() {{
        const map = getMapObject();
        if (selectedStopsLayer) {{
            map.removeLayer(selectedStopsLayer);
            selectedStopsLayer = null;
        }}
    }}

    function setFacilityHighlight(facility) {{
        const map = getMapObject();
        if (!map || !facility) return;
        if (currentFacilityHighlight) {{
            map.removeLayer(currentFacilityHighlight);
            currentFacilityHighlight = null;
        }}
        currentFacilityHighlight = L.circleMarker([facility.lat, facility.lon], {{
            radius: 16,
            color: '#111111',
            weight: 3,
            opacity: 1,
            fillOpacity: 0,
            interactive: false,
        }}).addTo(map);
    }}

    function findFacility(query) {{
        const q = String(query || '').trim().toLowerCase();
        if (!q) return null;
        return MAP_DATA.facilities.find(f => (
            String(f.id || '').toLowerCase() === q ||
            String(f.name || '').toLowerCase().includes(q)
        )) || null;
    }}

    async function focusFacility(query) {{
        const facility = findFacility(query);
        if (!facility) {{
            alert('Заклад не знайдено. Введіть facility_id або частину назви.');
            return;
        }}
        const map = getMapObject();
        map.setView([facility.lat, facility.lon], 15);
        currentFacilityId = facility.id;
        setFacilityHighlight(facility);
        await showBuildings(facility.id);
    }}

    function showStopPreview(props) {{
        const map = getMapObject();
        if (!map) return;

        clearStopPreview();

        const sourceStop    = currentMode === 'peak' ? props.peak_source_stop    : props.offpeak_source_stop;
        const destStop      = currentMode === 'peak' ? props.peak_dest_stop      : props.offpeak_dest_stop;
        const sourceStopLon = currentMode === 'peak' ? props.peak_source_stop_lon : props.offpeak_source_stop_lon;
        const sourceStopLat = currentMode === 'peak' ? props.peak_source_stop_lat : props.offpeak_source_stop_lat;
        const destStopLon   = currentMode === 'peak' ? props.peak_dest_stop_lon  : props.offpeak_dest_stop_lon;
        const destStopLat   = currentMode === 'peak' ? props.peak_dest_stop_lat  : props.offpeak_dest_stop_lat;
        const mode          = currentMode === 'peak' ? props.peak_mode           : props.offpeak_mode;
        const paths         = props._paths || {{}};

        selectedStopsLayer = L.layerGroup();

        // ── Walk paths (actual Dijkstra routes) ────────────────────────
        if (mode === 'walk' && paths.wp && paths.wp.length > 1) {{
            // Green dashed line: building → facility
            L.polyline(paths.wp, {{
                color: '#27AE60', weight: 3, opacity: 0.85,
                dashArray: '8 5', interactive: false,
            }}).addTo(selectedStopsLayer);
        }}

        if (mode === 'transit') {{
            // Orange dashed: building → boarding stop (actual walk path)
            if (paths.wi && paths.wi.length > 1) {{
                L.polyline(paths.wi, {{
                    color: '#E67E22', weight: 3, opacity: 0.85,
                    dashArray: '7 4', interactive: false,
                }}).addTo(selectedStopsLayer);
            }}
            // Orange dashed: alighting stop → facility (actual walk path)
            if (paths.wo && paths.wo.length > 1) {{
                L.polyline(paths.wo, {{
                    color: '#E67E22', weight: 3, opacity: 0.85,
                    dashArray: '7 4', interactive: false,
                }}).addTo(selectedStopsLayer);
            }}
        }}

        // ── Stop markers ───────────────────────────────────────────────
        if (typeof sourceStopLon === 'number' && typeof sourceStopLat === 'number') {{
            L.circleMarker([sourceStopLat, sourceStopLon], {{
                radius: 9, color: '#111111', fillColor: '#00BFFF',
                fillOpacity: 0.95, weight: 2, interactive: false,
            }}).bindTooltip('Зупинка посадки: ' + (sourceStop || ''), {{opacity: 0.95}})
              .addTo(selectedStopsLayer);
        }}

        if (typeof destStopLon === 'number' && typeof destStopLat === 'number') {{
            L.circleMarker([destStopLat, destStopLon], {{
                radius: 9, color: '#111111', fillColor: '#FFD700',
                fillOpacity: 0.95, weight: 2, interactive: false,
            }}).bindTooltip('Зупинка виходу: ' + (destStop || ''), {{opacity: 0.95}})
              .addTo(selectedStopsLayer);
        }}

        if (selectedStopsLayer.getLayers().length > 0) {{
            selectedStopsLayer.addTo(map);
        }} else {{
            selectedStopsLayer = null;
        }}
    }}

    function renderFacilityBuildings(facilityId, geojson) {{
        const map = getMapObject();
        if (!map) return;
        if (buildingsLayer) map.removeLayer(buildingsLayer);
        clearStopPreview();
        if (!buildingsRenderer) buildingsRenderer = L.canvas();

        const features = (geojson && geojson.features) ? geojson.features : [];
        buildingsLayer = L.layerGroup();

        features.forEach(feature => {{
            const props = feature.properties || {{}};
            const coords = feature.geometry && feature.geometry.coordinates;
            if (!coords || coords.length < 2) return;

            const group = currentMode === 'peak' ? props.group_peak : props.group_offpeak;
            if (!group) return;
            const color = COLORS[group] || '#BDC3C7';
            const mode = currentMode === 'peak' ? props.peak_mode : props.offpeak_mode;
            const totalMin = currentMode === 'peak' ? props.peak_total_min : props.offpeak_total_min;
            const walkInMin = currentMode === 'peak' ? props.peak_walk_in_min : props.offpeak_walk_in_min;
            const waitMin = currentMode === 'peak' ? props.peak_wait_min : props.offpeak_wait_min;
            const transitMin = currentMode === 'peak' ? props.peak_transit_min : props.offpeak_transit_min;
            const walkOutMin = currentMode === 'peak' ? props.peak_walk_out_min : props.offpeak_walk_out_min;
            const routeId = currentMode === 'peak' ? props.peak_route_id : props.offpeak_route_id;
            const route = currentMode === 'peak' ? props.peak_route : props.offpeak_route;
            const transport = currentMode === 'peak' ? props.peak_transport : props.offpeak_transport;
            const routeOptions = currentMode === 'peak' ? props.peak_route_options : props.offpeak_route_options;
            const buildingLevels = props.building_levels;
            const sourceStop = currentMode === 'peak' ? props.peak_source_stop : props.offpeak_source_stop;
            const destStop = currentMode === 'peak' ? props.peak_dest_stop : props.offpeak_dest_stop;
            const sourceStopLon = currentMode === 'peak' ? props.peak_source_stop_lon : props.offpeak_source_stop_lon;
            const sourceStopLat = currentMode === 'peak' ? props.peak_source_stop_lat : props.offpeak_source_stop_lat;
            const destStopLon = currentMode === 'peak' ? props.peak_dest_stop_lon : props.offpeak_dest_stop_lon;
            const destStopLat = currentMode === 'peak' ? props.peak_dest_stop_lat : props.offpeak_dest_stop_lat;

            let tooltip = 'Будинок #' + props.building_id;
            if (typeof buildingLevels === 'number') tooltip += '<br>Поверхи: ' + buildingLevels.toFixed(1);
            if (typeof totalMin === 'number') tooltip += '<br>Загальний час: ' + totalMin.toFixed(1) + ' хв';
            tooltip += '<br>Група: ' + group;
            if (mode === 'transit') {{
                const chosenRoute = [transport, route].filter(Boolean).join(' ');
                if (chosenRoute) tooltip += '<br>Обраний маршрут: ' + chosenRoute;
                if (routeId) tooltip += '<br>route_id: ' + routeId;
                if (routeOptions && routeOptions !== chosenRoute) {{
                    tooltip += '<br>Альтернативи: ' + routeOptions;
                }}
                if (typeof walkInMin === 'number') tooltip += '<br>До зупинки: ' + walkInMin.toFixed(1) + ' хв';
                if (typeof waitMin === 'number') tooltip += '<br>Очікування: ' + waitMin.toFixed(1) + ' хв';
                if (typeof transitMin === 'number') tooltip += '<br>У транспорті: ' + transitMin.toFixed(1) + ' хв';
                if (typeof walkOutMin === 'number') tooltip += '<br>Від зупинки до закладу: ' + walkOutMin.toFixed(1) + ' хв';
                if (sourceStop) tooltip += '<br>Зупинка посадки: ' + sourceStop;
                if (destStop) tooltip += '<br>Зупинка виходу: ' + destStop;
            }} else if (mode === 'walk') {{
                tooltip += '<br>Режим: пішки';
                const nearStops = NEAREST_STOPS[String(props.building_id)];
                if (nearStops && nearStops.length > 0) {{
                    tooltip += '<br><span style="color:#888;font-size:10px">──── найближчі зупинки ────</span>';
                    nearStops.forEach(function(ns) {{
                        const sname = STOP_NAME_CACHE[ns.s] || ('#' + ns.s);
                        tooltip += '<br>&#9679; ' + sname
                            + ' <span style="color:#888">#' + ns.s + '</span>'
                            + ' — ' + ns.w.toFixed(1) + ' хв';
                    }});
                }}
            }}

            const marker = L.circleMarker([coords[1], coords[0]], {{
                radius: 3,
                color: color,
                fillColor: color,
                fillOpacity: 0.8,
                weight: 0,
                interactive: true,
                renderer: buildingsRenderer,
            }});
            marker.bindTooltip(tooltip, {{sticky: false, opacity: 0.95, className: 'building-tooltip'}});
            marker.on('mouseover', function() {{
                marker.setStyle({{radius: 7, weight: 2, color: '#111111', fillColor: color, fillOpacity: 1}});
                showStopPreview(props);
            }});
            marker.on('mouseout', function() {{
                marker.setStyle({{radius: 3, weight: 0, color: color, fillColor: color, fillOpacity: 0.8}});
                clearStopPreview();
            }});
            marker.addTo(buildingsLayer);
        }});

        buildingsLayer.addTo(map);
    }}

    async function showBuildings(facilityId) {{
        currentFacilityId = facilityId;
        const facility = MAP_DATA.facilities.find(f => f.id === facilityId);
        if (!facility) return;
        setFacilityHighlight(facility);

        if (facilityBuildingsCache[facilityId]) {{
            renderFacilityBuildings(facilityId, facilityBuildingsCache[facilityId]);
            return;
        }}

        const candidates = [
            facility.buildings_geojson,
            facility.buildings_geojson.replace(/^\\.\\.\\//, ''),
            'map_buildings_baseline/' + facility.buildings_geojson.split('/').pop(),
        ];

        try {{
            let response = null;
            let lastError = null;
            for (const url of [...new Set(candidates)]) {{
                try {{
                    response = await fetch(url);
                    if (response.ok) break;
                    lastError = new Error(url + ' -> HTTP ' + response.status);
                }} catch (err) {{
                    lastError = err;
                }}
            }}
            if (!response || !response.ok) throw lastError || new Error('GeoJSON fetch failed');

            const geojson = await response.json();
            facilityBuildingsCache[facilityId] = geojson;
            renderFacilityBuildings(facilityId, geojson);
        }} catch (err) {{
            console.error('Failed to load building GeoJSON:', facilityId, err);
            alert('Не вдалося завантажити будинки: ' + err.message);
        }}
    }}

    function setMode(mode) {{
        currentMode = mode;

        const btnPeak = document.getElementById('btn-peak');
        const btnOffpeak = document.getElementById('btn-offpeak');

        if (mode === 'peak') {{
            btnPeak.style.fontWeight = 'bold';
            btnPeak.style.opacity = '1';
            btnOffpeak.style.fontWeight = 'normal';
            btnOffpeak.style.opacity = '0.65';
        }} else {{
            btnOffpeak.style.fontWeight = 'bold';
            btnOffpeak.style.opacity = '1';
            btnPeak.style.fontWeight = 'normal';
            btnPeak.style.opacity = '0.65';
        }}

        if (currentFacilityId) showBuildings(currentFacilityId);
    }}
    """

    switcher_html = f"""
    <div id="mode-switcher"
         style="position:fixed;top:112px;right:10px;z-index:1000;
                background:white;padding:8px 12px;border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3);
                font-family:Arial,sans-serif;line-height:1.6">
      <b style="font-size:13px">Час доби:</b><br>
      <button id="btn-peak" onclick="setMode('peak')"
              style="margin:3px 2px;padding:5px 10px;
                     background:#E74C3C;color:white;
                     border:none;border-radius:4px;cursor:pointer;
                     font-weight:bold;font-size:12px">
        Пік
      </button><br>
      <button id="btn-offpeak" onclick="setMode('offpeak')"
              style="margin:3px 2px;padding:5px 10px;
                     background:#3498DB;color:white;
                     border:none;border-radius:4px;cursor:pointer;
                     opacity:0.65;font-size:12px">
        Міжпік
      </button>
    </div>
    """

    search_html = """
    <div id="facility-search"
         style="position:fixed;top:10px;left:10px;z-index:1000;
                background:white;padding:10px 12px;border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3);
                font-family:Arial,sans-serif;width:280px">
      <b style="font-size:13px">Пошук закладу</b><br>
      <input id="facility-search-input" type="text"
             placeholder="facility_id або назва"
             style="margin-top:6px;width:100%;box-sizing:border-box;
                    padding:6px 8px;border:1px solid #ccc;border-radius:4px;
                    font-size:12px">
      <button onclick="focusFacility(document.getElementById('facility-search-input').value)"
              style="margin-top:8px;width:100%;padding:6px 8px;
                     background:#2C3E50;color:white;border:none;border-radius:4px;
                     cursor:pointer;font-size:12px">
        Знайти і показати
      </button>
      <div style="margin-top:6px;font-size:11px;color:#666">
        Приклад: <code>H0</code>, <code>S435</code> або частина назви
      </div>
    </div>
    """

    legend_html = f"""
    <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                background:white;padding:10px 14px;border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,0.3);
                font-family:Arial,sans-serif;font-size:12px;line-height:1.8">
      <b>Доступність будинків:</b><br>
      <span style="color:#1FFF2E;font-size:18px;vertical-align:middle">●</span>
      &nbsp;Пішки <= {T_SHORT} хв<br>
      <span style="color:#EB9328;font-size:18px;vertical-align:middle">●</span>
      &nbsp;Транспорт <= {T_SHORT} хв<br>
      <span style="color:#1B6B23;font-size:18px;vertical-align:middle">●</span>
      &nbsp;Пішки <= {T_LONG} хв<br>
      <span style="color:#FF0000;font-size:18px;vertical-align:middle">●</span>
      &nbsp;Транспорт <= {T_LONG} хв
    </div>
    """

    tooltip_style_html = """
    <style>
      .leaflet-tooltip.building-tooltip {
        background: rgba(255, 255, 255, 0.68);
        border: 1px solid rgba(60, 60, 60, 0.18);
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.10);
        color: #111111;
        backdrop-filter: blur(1px);
      }
    </style>
    """

    # ── Stops layer JS ────────────────────────────────────────────────────
    # SVG renderer is used instead of canvas so that mouse events fire only on
    # actual circle elements, not on the entire canvas area. This lets building
    # hover still work in areas between stop markers even when both layers are
    # visible. stopsPane z=450 puts stop circles visually above buildings (z=400)
    # so a stop circle directly on top of a building still gets the hover event.
    stops_js = f"""
    const STOPS_DATA = {json.dumps(stops_layer_data, ensure_ascii=False)};
    const ROUTE_LOOKUP = {json.dumps(route_lookup, ensure_ascii=False)};
    const NEAREST_STOPS = {json.dumps(nearest_stops_lookup, ensure_ascii=False)};

    // Quick name lookup: stop_id → display name (built once from STOPS_DATA)
    const STOP_NAME_CACHE = {{}};
    STOPS_DATA.forEach(function(s) {{ STOP_NAME_CACHE[s.sid] = s.name; }});

    let stopsLayerGroup = null;

    function initStopsLayer() {{
        if (stopsLayerGroup) return;
        const map = getMapObject();

        // SVG pane above buildings (overlayPane=400). SVG only captures events
        // on rendered shapes, so buildings remain hoverable between stop circles.
        if (!map.getPane('stopsPane')) {{
            map.createPane('stopsPane');
            map.getPane('stopsPane').style.zIndex = 450;
        }}
        const renderer = L.svg({{padding: 0.5, pane: 'stopsPane'}});

        stopsLayerGroup = L.layerGroup();
        STOPS_DATA.forEach(function(stop) {{
            const marker = L.circleMarker([stop.lat, stop.lon], {{
                radius: 5,
                color: stop.color,
                fillColor: stop.color,
                fillOpacity: 0.85,
                weight: 1.2,
                interactive: true,
                pane: 'stopsPane',
                renderer: renderer,
            }});
            marker.bindTooltip(stop.tooltip_html, {{
                sticky: true,
                opacity: 0.97,
                className: 'stop-tooltip',
                direction: 'top',
                offset: [0, -6],
            }});
            marker.on('mouseover', function() {{
                if (!this._isRoutingSelected)
                    this.setStyle({{radius: 8, weight: 2.5, fillOpacity: 1}});
            }});
            marker.on('mouseout', function() {{
                if (!this._isRoutingSelected)
                    this.setStyle({{radius: 5, weight: 1.2, fillOpacity: 0.85}});
            }});
            marker.on('click', function(e) {{
                L.DomEvent.stopPropagation(e);
                onStopClick(stop, this);
            }});
            marker.addTo(stopsLayerGroup);
        }});

        stopsLayerGroup.addTo(map);
    }}

    // ── Stop-to-stop routing ──────────────────────────────────────────────
    let routingFrom = null;   // {{stop, marker}}
    let routingTo   = null;   // {{stop, marker}}
    let routingLine = null;
    let routingPopup = null;

    const TRANSPORT_COLORS_RT = {{
        bus: '#1565C0', trol: '#2E7D32', tram: '#BF360C', metro: '#6A1B9A'
    }};

    function clearRouting() {{
        const map = getMapObject();
        [routingFrom, routingTo].forEach(function(sel) {{
            if (sel && sel.marker) {{
                sel.marker._isRoutingSelected = false;
                sel.marker.setStyle({{radius: 5, weight: 1.2, fillOpacity: 0.85,
                                      color: sel.stop.color, fillColor: sel.stop.color}});
            }}
        }});
        if (routingLine)  {{ map.removeLayer(routingLine);  routingLine  = null; }}
        if (routingPopup) {{ map.removeLayer(routingPopup); routingPopup = null; }}
        routingFrom = null;
        routingTo   = null;
        const hint = document.getElementById('routing-hint');
        if (hint) hint.textContent = 'Клікни на зупинку «Від»';
    }}

    function showRoutingResult(from, to) {{
        const map = getMapObject();
        const fwd = (ROUTE_LOOKUP[from.stop.sid] || {{}})[to.stop.sid];
        const bwd = (ROUTE_LOOKUP[to.stop.sid]   || {{}})[from.stop.sid];
        const info = fwd || bwd;
        const reversed = !fwd && !!bwd;

        if (routingLine)  {{ map.removeLayer(routingLine); }}
        if (routingPopup) {{ map.removeLayer(routingPopup); }}

        const fromLL = [from.stop.lat, from.stop.lon];
        const toLL   = [to.stop.lat,   to.stop.lon];

        if (!info) {{
            routingLine = L.polyline([fromLL, toLL], {{
                color: '#999', weight: 2, dashArray: '6 5', opacity: 0.7
            }}).addTo(map);
            const midLat = (from.stop.lat + to.stop.lat) / 2;
            const midLon = (from.stop.lon + to.stop.lon) / 2;
            routingPopup = L.popup({{closeButton: true, className: 'routing-popup'}})
                .setLatLng([midLat, midLon])
                .setContent('<div style="text-align:center;color:#888;padding:4px 8px">'
                    + '&#128683; Немає прямого маршруту<br>'
                    + '<span style="font-size:11px">між цими зупинками</span></div>')
                .openOn(map);
            return;
        }}

        const tColor = TRANSPORT_COLORS_RT[info.tp] || '#555';
        routingLine = L.polyline([fromLL, toLL], {{
            color: tColor, weight: 3.5, opacity: 0.85
        }}).addTo(map);

        const total = (info.w || 0) + info.t;
        const dirLabel = reversed
            ? '<span style="color:#e67e22;font-size:10px"> (зворотній напрямок)</span>'
            : '';
        const routes = info.r || '—';

        const fromName = from.stop.name || from.stop.sid;
        const toName   = to.stop.name   || to.stop.sid;

        const html = '<div style="font-family:Arial,sans-serif;font-size:12px;min-width:190px">'
            + '<b style="font-size:13px">&#128652; Маршрут</b>' + dirLabel + '<br>'
            + '<hr style="margin:4px 0;border-color:#eee">'
            + '<span style="color:#666">Від:</span> ' + fromName + '<br>'
            + '<span style="color:#666">До:</span> '  + toName   + '<br>'
            + '<hr style="margin:4px 0;border-color:#eee">'
            + '<span style="color:' + tColor + '">&#9679;</span> <b>' + routes + '</b><br>'
            + '&#9201; Очікування: <b>' + (info.w || 0).toFixed(1) + ' хв</b><br>'
            + '&#128652; У транспорті: <b>' + info.t.toFixed(1) + ' хв</b><br>'
            + '<b>Разом: ' + total.toFixed(1) + ' хв</b>'
            + '</div>';

        const midLat = (from.stop.lat + to.stop.lat) / 2;
        const midLon = (from.stop.lon + to.stop.lon) / 2;
        routingPopup = L.popup({{closeButton: true, className: 'routing-popup'}})
            .setLatLng([midLat, midLon])
            .setContent(html)
            .openOn(map);
    }}

    function onStopClick(stop, marker) {{
        if (!routingFrom) {{
            routingFrom = {{stop: stop, marker: marker}};
            marker._isRoutingSelected = true;
            marker.setStyle({{radius: 9, weight: 3, color: '#FF6F00',
                              fillColor: '#FF6F00', fillOpacity: 1}});
            const hint = document.getElementById('routing-hint');
            if (hint) hint.textContent = 'Тепер клікни зупинку «До»';
        }} else if (!routingTo) {{
            if (marker === routingFrom.marker) {{
                clearRouting();
                return;
            }}
            routingTo = {{stop: stop, marker: marker}};
            marker._isRoutingSelected = true;
            marker.setStyle({{radius: 9, weight: 3, color: '#0077B6',
                              fillColor: '#0077B6', fillOpacity: 1}});
            showRoutingResult(routingFrom, routingTo);
            const hint = document.getElementById('routing-hint');
            if (hint) hint.textContent = 'Клікни на карту щоб скинути';
        }} else {{
            clearRouting();
            routingFrom = {{stop: stop, marker: marker}};
            marker._isRoutingSelected = true;
            marker.setStyle({{radius: 9, weight: 3, color: '#FF6F00',
                              fillColor: '#FF6F00', fillOpacity: 1}});
            const hint = document.getElementById('routing-hint');
            if (hint) hint.textContent = 'Тепер клікни зупинку «До»';
        }}
    }}

    window.addEventListener('load', function() {{
        initStopsLayer();
        const map = getMapObject();
        map.on('click', function() {{ clearRouting(); }});
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') clearRouting();
        }});
    }});
    """

    stops_style_html = """
    <style>
      .leaflet-tooltip.stop-tooltip {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(80,80,80,0.22);
        box-shadow: 0 2px 8px rgba(0,0,0,0.14);
        color: #111;
        padding: 6px 10px;
        line-height: 1.55;
        max-width: 280px;
      }
      .leaflet-popup-content-wrapper.routing-popup,
      .routing-popup .leaflet-popup-content-wrapper {
        border-radius: 8px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.2);
      }
    </style>
    <div id="routing-hint-panel"
         style="position:fixed;bottom:30px;left:50%;transform:translateX(-50%);
                z-index:1000;background:rgba(30,30,30,0.82);color:#fff;
                padding:7px 18px;border-radius:20px;font-family:Arial,sans-serif;
                font-size:13px;pointer-events:none">
      <span id="routing-hint">Клікни на зупинку «Від»</span>
    </div>
    """

    m.get_root().script.add_child(Element(js_code))
    m.get_root().script.add_child(Element(stops_js))
    m.get_root().html.add_child(Element(tooltip_style_html))
    m.get_root().html.add_child(Element(stops_style_html))
    m.get_root().html.add_child(Element(search_html))
    m.get_root().html.add_child(Element(switcher_html))
    m.get_root().html.add_child(Element(legend_html))
    m.save(OUT_HTML)

    print(f"Інтерактивну baseline-карту збережено: {OUT_HTML}")
    print("GeoJSON для будинків лежать у pipeline/data/outputs/map_buildings_baseline")


if __name__ == "__main__":
    run()
