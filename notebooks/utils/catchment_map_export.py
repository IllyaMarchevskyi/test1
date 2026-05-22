from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd


def _safe_file_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return normalized or "facility"


def _build_stats(row: pd.Series, grp_walk_short: str, grp_transit_short: str, grp_walk_long: str, grp_transit_long: str, t_short: int, t_long: int) -> Dict[str, int]:
    return {
        "peak_walk_short": int(row.get(f"peak_{grp_walk_short}", 0)),
        "peak_transit_short": int(row.get(f"peak_{grp_transit_short}", 0)),
        "peak_walk_long": int(row.get(f"peak_{grp_walk_long}", 0)),
        "peak_transit_long": int(row.get(f"peak_{grp_transit_long}", 0)),
        "peak_short": int(row.get(f"peak_total_{t_short}min", 0)),
        "peak_long": int(row.get(f"peak_total_{t_long}min", 0)),
        "offpeak_walk_short": int(row.get(f"offpeak_{grp_walk_short}", 0)),
        "offpeak_transit_short": int(row.get(f"offpeak_{grp_transit_short}", 0)),
        "offpeak_walk_long": int(row.get(f"offpeak_{grp_walk_long}", 0)),
        "offpeak_transit_long": int(row.get(f"offpeak_{grp_transit_long}", 0)),
        "offpeak_short": int(row.get(f"offpeak_total_{t_short}min", 0)),
        "offpeak_long": int(row.get(f"offpeak_total_{t_long}min", 0)),
    }


def export_catchment_map_data(
    catchment_results: pd.DataFrame,
    catchment_buildings: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    facilities: pd.DataFrame,
    output_json_path: str | Path,
    output_geojson_dir: str | Path,
    html_rel_geojson_dir: str,
    t_short: int,
    t_long: int,
    grp_walk_short: str,
    grp_transit_short: str,
    grp_walk_long: str,
    grp_transit_long: str,
) -> Dict[str, object]:
    output_json_path = Path(output_json_path)
    output_geojson_dir = Path(output_geojson_dir)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_geojson_dir.mkdir(parents=True, exist_ok=True)

    merged = catchment_buildings.merge(
        buildings[["building_id", "geometry"]],
        on="building_id",
        how="left",
    )
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=buildings.crs).to_crs("EPSG:4326")
    by_facility = {fid: grp for fid, grp in merged.groupby("facility_id")}
    results_idx = catchment_results.set_index("facility_id")

    facilities_data: List[Dict[str, object]] = []
    total_buildings = 0

    for _, facility in facilities.iterrows():
        fid = facility["facility_id"]
        facility_rows = by_facility.get(fid)
        feature_count = 0

        file_stem = _safe_file_stem(fid)
        geojson_filename = f"{file_stem}.geojson"
        geojson_path = output_geojson_dir / geojson_filename
        geojson_relpath = f"{html_rel_geojson_dir.rstrip('/')}/{geojson_filename}"

        if facility_rows is not None and not facility_rows.empty:
            features = []
            for _, row in facility_rows.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(row.geometry.x), float(row.geometry.y)],
                        },
                        "properties": {
                            "building_id": int(row["building_id"]),
                            "group_peak": row["group_peak"],
                            "group_offpeak": row["group_offpeak"],
                        },
                    }
                )
            feature_count = len(features)
            total_buildings += feature_count
            geojson_payload = {"type": "FeatureCollection", "features": features}
        else:
            geojson_payload = {"type": "FeatureCollection", "features": []}

        geojson_path.write_text(json.dumps(geojson_payload, ensure_ascii=False), encoding="utf-8")

        if fid in results_idx.index:
            row = results_idx.loc[fid]
            stats = _build_stats(
                row,
                grp_walk_short,
                grp_transit_short,
                grp_walk_long,
                grp_transit_long,
                t_short,
                t_long,
            )
        else:
            stats = {
                "peak_walk_short": 0,
                "peak_transit_short": 0,
                "peak_walk_long": 0,
                "peak_transit_long": 0,
                "peak_short": 0,
                "peak_long": 0,
                "offpeak_walk_short": 0,
                "offpeak_transit_short": 0,
                "offpeak_walk_long": 0,
                "offpeak_transit_long": 0,
                "offpeak_short": 0,
                "offpeak_long": 0,
            }

        facilities_data.append(
            {
                "id": fid,
                "name": str(facility["name"]),
                "type": facility["facility_type"],
                "lat": float(facility["lat"]),
                "lon": float(facility["lon"]),
                "stats": stats,
                "buildings_geojson": geojson_relpath,
                "n_buildings": feature_count,
            }
        )

    payload: Dict[str, object] = {
        "facilities": facilities_data,
        "t_short": t_short,
        "t_long": t_long,
        "grp_walk_short": grp_walk_short,
        "grp_transit_short": grp_transit_short,
        "grp_walk_long": grp_walk_long,
        "grp_transit_long": grp_transit_long,
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    payload["_total_buildings"] = total_buildings
    payload["_geojson_dir"] = str(output_geojson_dir)
    return payload
