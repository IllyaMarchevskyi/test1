from __future__ import annotations

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


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


def _json_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _zero_stats() -> Dict[str, int]:
    return {
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


def _export_facility_geojson(
    order_idx: int,
    facility: Dict[str, object],
    facility_rows: pd.DataFrame | None,
    stats: Dict[str, int],
    output_geojson_dir: Path,
    html_rel_geojson_dir: str,
) -> Dict[str, object]:
    fid = str(facility["facility_id"])
    file_stem = _safe_file_stem(fid)
    geojson_filename = f"{file_stem}.geojson"
    geojson_path = output_geojson_dir / geojson_filename
    geojson_relpath = f"{html_rel_geojson_dir.rstrip('/')}/{geojson_filename}"

    features = []
    if facility_rows is not None and not facility_rows.empty:
        valid_rows = facility_rows.loc[
            facility_rows["lon"].notna() & facility_rows["lat"].notna(),
            [
                "building_id",
                "group_peak",
                "group_offpeak",
                "lon",
                "lat",
                "peak_mode",
                "peak_total_min",
                "peak_walk_in_min",
                "peak_wait_min",
                "peak_transit_min",
                "peak_walk_out_min",
                "peak_route_id",
                "peak_route",
                "peak_transport",
                "peak_route_options",
                "building_levels",
                "peak_source_stop",
                "peak_dest_stop",
                "peak_n_transfers",
                "peak_transfer_stop",
                "peak_transit_leg1_min",
                "peak_transfer_wait_2_min",
                "peak_transit_leg2_min",
                "peak_transport_2",
                "peak_route_2",
                "peak_transfer_stop_lon",
                "peak_transfer_stop_lat",
                "peak_source_stop_lon",
                "peak_source_stop_lat",
                "peak_dest_stop_lon",
                "peak_dest_stop_lat",
                "offpeak_mode",
                "offpeak_total_min",
                "offpeak_walk_in_min",
                "offpeak_wait_min",
                "offpeak_transit_min",
                "offpeak_walk_out_min",
                "offpeak_route_id",
                "offpeak_route",
                "offpeak_transport",
                "offpeak_route_options",
                "offpeak_source_stop",
                "offpeak_dest_stop",
                "offpeak_n_transfers",
                "offpeak_transfer_stop",
                "offpeak_transit_leg1_min",
                "offpeak_transfer_wait_2_min",
                "offpeak_transit_leg2_min",
                "offpeak_transport_2",
                "offpeak_route_2",
                "offpeak_transfer_stop_lon",
                "offpeak_transfer_stop_lat",
                "offpeak_source_stop_lon",
                "offpeak_source_stop_lat",
                "offpeak_dest_stop_lon",
                "offpeak_dest_stop_lat",
            ],
        ]
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row.lon), float(row.lat)],
                },
                "properties": {
                    "building_id": int(row.building_id),
                    "group_peak": _json_group(row.group_peak),
                    "group_offpeak": _json_group(row.group_offpeak),
                    "peak_mode": _json_group(row.peak_mode),
                    "peak_total_min": None if pd.isna(row.peak_total_min) else float(row.peak_total_min),
                    "peak_walk_in_min": None if pd.isna(row.peak_walk_in_min) else float(row.peak_walk_in_min),
                    "peak_wait_min": None if pd.isna(row.peak_wait_min) else float(row.peak_wait_min),
                    "peak_transit_min": None if pd.isna(row.peak_transit_min) else float(row.peak_transit_min),
                    "peak_walk_out_min": None if pd.isna(row.peak_walk_out_min) else float(row.peak_walk_out_min),
                    "peak_route_id": _json_group(row.peak_route_id),
                    "peak_route": _json_group(row.peak_route),
                    "peak_transport": _json_group(row.peak_transport),
                    "peak_route_options": _json_group(row.peak_route_options),
                    "building_levels": None if pd.isna(row.building_levels) else float(row.building_levels),
                    "peak_source_stop": _json_group(row.peak_source_stop),
                    "peak_dest_stop": _json_group(row.peak_dest_stop),
                    "peak_n_transfers": int(row.peak_n_transfers) if not pd.isna(row.peak_n_transfers) else 0,
                    "peak_transfer_stop": _json_group(row.peak_transfer_stop),
                    "peak_transit_leg1_min": None if pd.isna(row.peak_transit_leg1_min) else float(row.peak_transit_leg1_min),
                    "peak_transfer_wait_2_min": None if pd.isna(row.peak_transfer_wait_2_min) else float(row.peak_transfer_wait_2_min),
                    "peak_transit_leg2_min": None if pd.isna(row.peak_transit_leg2_min) else float(row.peak_transit_leg2_min),
                    "peak_transport_2": _json_group(row.peak_transport_2),
                    "peak_route_2": _json_group(row.peak_route_2),
                    "peak_transfer_stop_lon": None if pd.isna(row.peak_transfer_stop_lon) else float(row.peak_transfer_stop_lon),
                    "peak_transfer_stop_lat": None if pd.isna(row.peak_transfer_stop_lat) else float(row.peak_transfer_stop_lat),
                    "peak_source_stop_lon": None if pd.isna(row.peak_source_stop_lon) else float(row.peak_source_stop_lon),
                    "peak_source_stop_lat": None if pd.isna(row.peak_source_stop_lat) else float(row.peak_source_stop_lat),
                    "peak_dest_stop_lon": None if pd.isna(row.peak_dest_stop_lon) else float(row.peak_dest_stop_lon),
                    "peak_dest_stop_lat": None if pd.isna(row.peak_dest_stop_lat) else float(row.peak_dest_stop_lat),
                    "offpeak_mode": _json_group(row.offpeak_mode),
                    "offpeak_total_min": None if pd.isna(row.offpeak_total_min) else float(row.offpeak_total_min),
                    "offpeak_walk_in_min": None if pd.isna(row.offpeak_walk_in_min) else float(row.offpeak_walk_in_min),
                    "offpeak_wait_min": None if pd.isna(row.offpeak_wait_min) else float(row.offpeak_wait_min),
                    "offpeak_transit_min": None if pd.isna(row.offpeak_transit_min) else float(row.offpeak_transit_min),
                    "offpeak_walk_out_min": None if pd.isna(row.offpeak_walk_out_min) else float(row.offpeak_walk_out_min),
                    "offpeak_route_id": _json_group(row.offpeak_route_id),
                    "offpeak_route": _json_group(row.offpeak_route),
                    "offpeak_transport": _json_group(row.offpeak_transport),
                    "offpeak_route_options": _json_group(row.offpeak_route_options),
                    "offpeak_source_stop": _json_group(row.offpeak_source_stop),
                    "offpeak_dest_stop": _json_group(row.offpeak_dest_stop),
                    "offpeak_n_transfers": int(row.offpeak_n_transfers) if not pd.isna(row.offpeak_n_transfers) else 0,
                    "offpeak_transfer_stop": _json_group(row.offpeak_transfer_stop),
                    "offpeak_transit_leg1_min": None if pd.isna(row.offpeak_transit_leg1_min) else float(row.offpeak_transit_leg1_min),
                    "offpeak_transfer_wait_2_min": None if pd.isna(row.offpeak_transfer_wait_2_min) else float(row.offpeak_transfer_wait_2_min),
                    "offpeak_transit_leg2_min": None if pd.isna(row.offpeak_transit_leg2_min) else float(row.offpeak_transit_leg2_min),
                    "offpeak_transport_2": _json_group(row.offpeak_transport_2),
                    "offpeak_route_2": _json_group(row.offpeak_route_2),
                    "offpeak_transfer_stop_lon": None if pd.isna(row.offpeak_transfer_stop_lon) else float(row.offpeak_transfer_stop_lon),
                    "offpeak_transfer_stop_lat": None if pd.isna(row.offpeak_transfer_stop_lat) else float(row.offpeak_transfer_stop_lat),
                    "offpeak_source_stop_lon": None if pd.isna(row.offpeak_source_stop_lon) else float(row.offpeak_source_stop_lon),
                    "offpeak_source_stop_lat": None if pd.isna(row.offpeak_source_stop_lat) else float(row.offpeak_source_stop_lat),
                    "offpeak_dest_stop_lon": None if pd.isna(row.offpeak_dest_stop_lon) else float(row.offpeak_dest_stop_lon),
                    "offpeak_dest_stop_lat": None if pd.isna(row.offpeak_dest_stop_lat) else float(row.offpeak_dest_stop_lat),
                },
            }
            for row in valid_rows.itertuples(index=False)
        ]

    geojson_payload = {"type": "FeatureCollection", "features": features}
    geojson_path.write_text(json.dumps(geojson_payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    return {
        "order_idx": order_idx,
        "id": fid,
        "name": str(facility["name"]),
        "type": facility["facility_type"],
        "lat": float(facility["lat"]),
        "lon": float(facility["lon"]),
        "stats": stats,
        "buildings_geojson": geojson_relpath,
        "n_buildings": len(features),
    }


def read_parquet_with_progress(
    path: str | Path,
    desc: str = "Завантаження parquet",
    columns: List[str] | None = None,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    tables = []
    row_groups = range(parquet_file.num_row_groups)
    total_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
    read_rows = 0

    print(
        f"{desc}: {parquet_file.num_row_groups} group(s)"
        + (f", ~{total_rows:,} rows" if total_rows is not None else ""),
        flush=True,
    )
    progress = tqdm(row_groups, total=parquet_file.num_row_groups, desc=desc, unit="grp")
    for row_group_idx in progress:
        table = parquet_file.read_row_group(row_group_idx, columns=columns)
        tables.append(table)
        read_rows += table.num_rows
        if total_rows is not None:
            progress.set_postfix({"rows": f"{read_rows:,}/{total_rows:,}"})
        else:
            progress.set_postfix({"rows": f"{read_rows:,}"})

    if not tables:
        return pd.DataFrame()

    return pa.concat_tables(tables).to_pandas()


def export_catchment_map_data(
    catchment_results: pd.DataFrame,
    catchment_buildings: pd.DataFrame,
    buildings: gpd.GeoDataFrame,
    facilities: pd.DataFrame,
    stop_coords: pd.DataFrame | None,
    output_json_path: str | Path,
    output_geojson_dir: str | Path,
    html_rel_geojson_dir: str,
    t_short: int,
    t_long: int,
    grp_walk_short: str,
    grp_transit_short: str,
    grp_walk_long: str,
    grp_transit_long: str,
    parallel_workers: int | None = None,
) -> Dict[str, object]:
    output_json_path = Path(output_json_path)
    output_geojson_dir = Path(output_geojson_dir)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_geojson_dir.mkdir(parents=True, exist_ok=True)

    started_at = perf_counter()
    print("Готуємо координати будинків для карти...", flush=True)
    building_coords = gpd.GeoDataFrame(
        buildings[["building_id", "geometry"]].copy(),
        geometry="geometry",
        crs=buildings.crs,
    ).to_crs("EPSG:4326")
    building_coords["lon"] = building_coords.geometry.x
    building_coords["lat"] = building_coords.geometry.y
    building_coords = pd.DataFrame(building_coords[["building_id", "lon", "lat"]])
    print(f"  Координати готові: {len(building_coords):,} будинків", flush=True)

    print("Приєднуємо координати до catchment_buildings...", flush=True)
    optional_detail_columns = [
        "peak_mode",
        "peak_total_min",
        "peak_walk_in_min",
        "peak_wait_min",
        "peak_transit_min",
        "peak_walk_out_min",
        "peak_route_id",
        "peak_route",
        "peak_transport",
        "peak_route_options",
        "building_levels",
        "peak_source_stop",
        "peak_dest_stop",
        "peak_n_transfers",
        "peak_transfer_stop",
        "peak_transit_leg1_min",
        "peak_transfer_wait_2_min",
        "peak_transit_leg2_min",
        "peak_transport_2",
        "peak_route_2",
        "offpeak_mode",
        "offpeak_total_min",
        "offpeak_walk_in_min",
        "offpeak_wait_min",
        "offpeak_transit_min",
        "offpeak_walk_out_min",
        "offpeak_route_id",
        "offpeak_route",
        "offpeak_transport",
        "offpeak_route_options",
        "offpeak_source_stop",
        "offpeak_dest_stop",
        "offpeak_n_transfers",
        "offpeak_transfer_stop",
        "offpeak_transit_leg1_min",
        "offpeak_transfer_wait_2_min",
        "offpeak_transit_leg2_min",
        "offpeak_transport_2",
        "offpeak_route_2",
    ]
    for column in optional_detail_columns:
        if column not in catchment_buildings.columns:
            catchment_buildings[column] = None

    merged = catchment_buildings.merge(
        building_coords,
        on="building_id",
        how="left",
        copy=False,
    )
    if stop_coords is not None and not stop_coords.empty:
        peak_src = stop_coords.rename(
            columns={"stop_id": "peak_source_stop", "lon": "peak_source_stop_lon", "lat": "peak_source_stop_lat"}
        )
        peak_dst = stop_coords.rename(
            columns={"stop_id": "peak_dest_stop", "lon": "peak_dest_stop_lon", "lat": "peak_dest_stop_lat"}
        )
        off_src = stop_coords.rename(
            columns={"stop_id": "offpeak_source_stop", "lon": "offpeak_source_stop_lon", "lat": "offpeak_source_stop_lat"}
        )
        off_dst = stop_coords.rename(
            columns={"stop_id": "offpeak_dest_stop", "lon": "offpeak_dest_stop_lon", "lat": "offpeak_dest_stop_lat"}
        )
        peak_tr = stop_coords.rename(
            columns={"stop_id": "peak_transfer_stop", "lon": "peak_transfer_stop_lon", "lat": "peak_transfer_stop_lat"}
        )
        off_tr = stop_coords.rename(
            columns={"stop_id": "offpeak_transfer_stop", "lon": "offpeak_transfer_stop_lon", "lat": "offpeak_transfer_stop_lat"}
        )
        merged = merged.merge(peak_src, on="peak_source_stop", how="left", copy=False)
        merged = merged.merge(peak_dst, on="peak_dest_stop", how="left", copy=False)
        merged = merged.merge(peak_tr, on="peak_transfer_stop", how="left", copy=False)
        merged = merged.merge(off_src, on="offpeak_source_stop", how="left", copy=False)
        merged = merged.merge(off_dst, on="offpeak_dest_stop", how="left", copy=False)
        merged = merged.merge(off_tr, on="offpeak_transfer_stop", how="left", copy=False)
    for column in [
        "peak_source_stop_lon",
        "peak_source_stop_lat",
        "peak_dest_stop_lon",
        "peak_dest_stop_lat",
        "offpeak_source_stop_lon",
        "offpeak_source_stop_lat",
        "offpeak_dest_stop_lon",
        "offpeak_dest_stop_lat",
        "peak_transfer_stop_lon",
        "peak_transfer_stop_lat",
        "offpeak_transfer_stop_lon",
        "offpeak_transfer_stop_lat",
    ]:
        if column not in merged.columns:
            merged[column] = None
    merged["facility_id"] = merged["facility_id"].astype(str)
    print(f"  Merge готовий: {len(merged):,} записів", flush=True)

    print("Групуємо записи по закладах...", flush=True)
    by_facility = {
        str(fid): grp[
            [
                "building_id",
                "group_peak",
                "group_offpeak",
                "lon",
                "lat",
                "peak_mode",
                "peak_total_min",
                "peak_walk_in_min",
                "peak_wait_min",
                "peak_transit_min",
                "peak_walk_out_min",
                "peak_route_id",
                "peak_route",
                "peak_transport",
                "peak_route_options",
                "building_levels",
                "peak_source_stop",
                "peak_dest_stop",
                "peak_n_transfers",
                "peak_transfer_stop",
                "peak_transit_leg1_min",
                "peak_transfer_wait_2_min",
                "peak_transit_leg2_min",
                "peak_transport_2",
                "peak_route_2",
                "peak_transfer_stop_lon",
                "peak_transfer_stop_lat",
                "peak_source_stop_lon",
                "peak_source_stop_lat",
                "peak_dest_stop_lon",
                "peak_dest_stop_lat",
                "offpeak_mode",
                "offpeak_total_min",
                "offpeak_walk_in_min",
                "offpeak_wait_min",
                "offpeak_transit_min",
                "offpeak_walk_out_min",
                "offpeak_route_id",
                "offpeak_route",
                "offpeak_transport",
                "offpeak_route_options",
                "offpeak_source_stop",
                "offpeak_dest_stop",
                "offpeak_n_transfers",
                "offpeak_transfer_stop",
                "offpeak_transit_leg1_min",
                "offpeak_transfer_wait_2_min",
                "offpeak_transit_leg2_min",
                "offpeak_transport_2",
                "offpeak_route_2",
                "offpeak_transfer_stop_lon",
                "offpeak_transfer_stop_lat",
                "offpeak_source_stop_lon",
                "offpeak_source_stop_lat",
                "offpeak_dest_stop_lon",
                "offpeak_dest_stop_lat",
            ]
        ]
        for fid, grp in merged.groupby("facility_id", sort=False)
    }
    print(f"  Групування готове: {len(by_facility):,} закладів", flush=True)

    print("Будуємо індекс статистики по закладах...", flush=True)
    results_idx = catchment_results.copy()
    results_idx["facility_id"] = results_idx["facility_id"].astype(str)
    results_idx = results_idx.set_index("facility_id")

    facilities_data: List[Dict[str, object]] = []
    total_buildings = 0
    workers = parallel_workers or min(8, os.cpu_count() or 1)
    workers = max(1, workers)
    print(f"Паралельна підготовка GeoJSON: {workers} worker(s)", flush=True)

    facility_tasks = []
    facilities = facilities.copy()
    facilities["facility_id"] = facilities["facility_id"].astype(str)
    for order_idx, facility in enumerate(facilities.to_dict("records")):
        fid = facility["facility_id"]
        if fid in results_idx.index:
            stats = _build_stats(
                results_idx.loc[fid],
                grp_walk_short,
                grp_transit_short,
                grp_walk_long,
                grp_transit_long,
                t_short,
                t_long,
            )
        else:
            stats = _zero_stats()
        facility_tasks.append(
            (
                order_idx,
                facility,
                by_facility.get(fid),
                stats,
                output_geojson_dir,
                html_rel_geojson_dir,
            )
        )

    progress = tqdm(total=len(facility_tasks), desc="Підготовка GeoJSON для карти", unit="заклад")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_export_facility_geojson, *task)
            for task in facility_tasks
        ]
        for future in as_completed(futures):
            item = future.result()
            facilities_data.append(item)
            total_buildings += item["n_buildings"]
            progress.update(1)
            progress.set_postfix(
                {
                    "facility_id": item["id"],
                    "features": f"{item['n_buildings']:,}",
                    "total_buildings": f"{total_buildings:,}",
                }
            )
    progress.close()
    facilities_data.sort(key=lambda item: item["order_idx"])
    facilities_data = [
        {
            "id": item["id"],
            "name": item["name"],
            "type": item["type"],
            "lat": item["lat"],
            "lon": item["lon"],
            "stats": item["stats"],
            "buildings_geojson": item["buildings_geojson"],
            "n_buildings": item["n_buildings"],
        }
        for item in facilities_data
    ]

    payload: Dict[str, object] = {
        "facilities": facilities_data,
        "t_short": t_short,
        "t_long": t_long,
        "grp_walk_short": grp_walk_short,
        "grp_transit_short": grp_transit_short,
        "grp_walk_long": grp_walk_long,
        "grp_transit_long": grp_transit_long,
    }
    output_json_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    payload["_total_buildings"] = total_buildings
    payload["_geojson_dir"] = str(output_geojson_dir)
    payload["_elapsed_sec"] = round(perf_counter() - started_at, 1)
    return payload
