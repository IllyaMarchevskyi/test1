"""
07d Baseline Results Map.

Baseline branch for map export and interactive visualization.
Keeps old 07d intact and reads baseline caches only.
"""


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
    osm_stops_path = "../gtfs_static/osm_stops.csv"

    required = {
        "catchment_results_baseline": catchment_csv,
        "catchment_buildings_baseline": catchment_buildings_path,
        "buildings": buildings_path,
        "scores": scores_path,
        "bridge": bridge_path,
        "osm_stops": osm_stops_path,
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
    bridge = pd.read_csv(bridge_path, usecols=["osm_id", "stop_id"]).dropna()
    bridge["osm_id"] = bridge["osm_id"].astype(str)
    bridge["stop_id"] = bridge["stop_id"].astype(str)
    osm_stops_raw = pd.read_csv(osm_stops_path).dropna(subset=["geometry"]).copy()
    osm_stops_raw["geometry"] = osm_stops_raw["geometry"].map(wkt.loads)
    osm_stops_raw["osm_id"] = osm_stops_raw.index.astype(str)
    osm_stops = gpd.GeoDataFrame(osm_stops_raw, geometry="geometry", crs="EPSG:4326")
    osm_stops = osm_stops[osm_stops.geometry.geom_type == "Point"].copy()
    osm_stops["lon"] = osm_stops.geometry.x
    osm_stops["lat"] = osm_stops.geometry.y
    stop_coords = bridge.merge(osm_stops[["osm_id", "lon", "lat"]], on="osm_id", how="left")[["stop_id", "lon", "lat"]]
    stop_coords = stop_coords.dropna(subset=["lon", "lat"]).drop_duplicates(subset=["stop_id"]).reset_index(drop=True)

    print(f"  catchment_results:   {len(catchment_results)} закладів")
    print(f"  catchment_buildings: {len(catchment_buildings):,} записів")
    print(f"  buildings:           {len(buildings):,} будинків")
    print(f"  facilities:          {len(facilities)} закладів")

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
        "parallel_workers": min(8, os.cpu_count() or 1),
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
            f"<span style='color:#666'>{'Лікарня' if is_hosp else 'Школа'}</span>"
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

    function showStopPreview(props) {{
        const map = getMapObject();
        if (!map) return;

        clearStopPreview();

        const sourceStop = currentMode === 'peak' ? props.peak_source_stop : props.offpeak_source_stop;
        const destStop = currentMode === 'peak' ? props.peak_dest_stop : props.offpeak_dest_stop;
        const sourceStopLon = currentMode === 'peak' ? props.peak_source_stop_lon : props.offpeak_source_stop_lon;
        const sourceStopLat = currentMode === 'peak' ? props.peak_source_stop_lat : props.offpeak_source_stop_lat;
        const destStopLon = currentMode === 'peak' ? props.peak_dest_stop_lon : props.offpeak_dest_stop_lon;
        const destStopLat = currentMode === 'peak' ? props.peak_dest_stop_lat : props.offpeak_dest_stop_lat;

        selectedStopsLayer = L.layerGroup();

        if (typeof sourceStopLon === 'number' && typeof sourceStopLat === 'number') {{
            L.circleMarker([sourceStopLat, sourceStopLon], {{
                radius: 9,
                color: '#111111',
                fillColor: '#00BFFF',
                fillOpacity: 0.95,
                weight: 2,
                interactive: false,
            }}).bindTooltip('Зупинка посадки: ' + (sourceStop || ''), {{opacity: 0.95}}).addTo(selectedStopsLayer);
        }}

        if (typeof destStopLon === 'number' && typeof destStopLat === 'number') {{
            L.circleMarker([destStopLat, destStopLon], {{
                radius: 9,
                color: '#111111',
                fillColor: '#FFD700',
                fillOpacity: 0.95,
                weight: 2,
                interactive: false,
            }}).bindTooltip('Зупинка виходу: ' + (destStop || ''), {{opacity: 0.95}}).addTo(selectedStopsLayer);
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
                if (routeOptions) {{
                    tooltip += '<br>Транспорт: ' + routeOptions;
                }} else if (transport || route) {{
                    tooltip += '<br>Транспорт: ' + [transport, route].filter(Boolean).join(' ');
                }}
                if (typeof walkInMin === 'number') tooltip += '<br>До зупинки: ' + walkInMin.toFixed(1) + ' хв';
                if (typeof waitMin === 'number') tooltip += '<br>Очікування: ' + waitMin.toFixed(1) + ' хв';
                if (typeof transitMin === 'number') tooltip += '<br>У транспорті: ' + transitMin.toFixed(1) + ' хв';
                if (typeof walkOutMin === 'number') tooltip += '<br>Від зупинки до закладу: ' + walkOutMin.toFixed(1) + ' хв';
                if (sourceStop || destStop) tooltip += '<br>Зупинки: ' + [sourceStop, destStop].filter(Boolean).join(' -> ');
            }} else if (mode === 'walk') {{
                tooltip += '<br>Режим: пішки';
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

    m.get_root().script.add_child(Element(js_code))
    m.get_root().html.add_child(Element(tooltip_style_html))
    m.get_root().html.add_child(Element(switcher_html))
    m.get_root().html.add_child(Element(legend_html))
    m.save(OUT_HTML)

    print(f"Інтерактивну baseline-карту збережено: {OUT_HTML}")
    print("GeoJSON для будинків лежать у pipeline/data/outputs/map_buildings_baseline")


if __name__ == "__main__":
    run()
