"""
08c Missing Names Map.

Фільтрує заклади з порожньою назвою / NaN і виводить їх на окремій карті.
"""


def run() -> None:
    from config_loader import cfg
    import os
    import warnings

    import folium
    import pandas as pd

    warnings.filterwarnings("ignore")

    PROCESSED_DIR = "./data/processed"
    OUTPUTS_DIR = "./data/outputs"
    SCORES_PATH = cfg["paths"]["scores"]
    OUT_CSV = f"{PROCESSED_DIR}/facilities_missing_names.csv"
    OUT_HTML = f"{OUTPUTS_DIR}/facilities_missing_names_map.html"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    if not os.path.exists(SCORES_PATH):
        raise FileNotFoundError(f"Відсутній файл із закладами: {SCORES_PATH}")

    if os.path.exists(OUT_CSV) and os.path.exists(OUT_HTML):
        outputs_mtime = min(os.path.getmtime(OUT_CSV), os.path.getmtime(OUT_HTML))
        if outputs_mtime >= os.path.getmtime(SCORES_PATH):
            cached = pd.read_csv(OUT_CSV)
            print(f"08c_missing_names: кеш завантажено: {len(cached):,} закладів без назви")
            print(f"  CSV:  {OUT_CSV}")
            print(f"  HTML: {OUT_HTML}")
            return

    print("08c_missing_names: завантажуємо всі заклади...")
    scores = pd.read_csv(SCORES_PATH, usecols=["facility_id", "facility_type", "name", "lat", "lon"])
    scores["facility_id"] = scores["facility_id"].astype(str)
    scores["facility_type"] = scores["facility_type"].astype(str)

    mask_missing = (
        scores["name"].isna()
        | (scores["name"].astype(str).str.strip() == "")
        | (scores["name"].astype(str).str.lower() == "nan")
    )
    missing = scores[mask_missing].copy()
    missing["name"] = ""
    missing = missing.sort_values(["facility_type", "facility_id"]).reset_index(drop=True)
    missing.to_csv(OUT_CSV, index=False, encoding="utf-8")

    m = folium.Map(
        location=[cfg["city"]["center_lat"], cfg["city"]["center_lon"]],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    layer_hospitals = folium.FeatureGroup(name="Лікарні без назви", show=True)
    layer_schools = folium.FeatureGroup(name="Школи без назви", show=True)

    for row in missing.itertuples(index=False):
        is_hospital = row.facility_type == "hospital"
        color = "#C0392B" if is_hospital else "#2980B9"
        label = "Лікарня" if is_hospital else "Школа"
        target_layer = layer_hospitals if is_hospital else layer_schools

        popup_html = (
            f"<div style='font-family:Arial,sans-serif;font-size:13px;width:230px'>"
            f"<b>Заклад без назви</b><br>"
            f"ID: <b>{row.facility_id}</b><br>"
            f"Тип: <b>{label}</b><br>"
            f"lat: {float(row.lat):.6f}<br>"
            f"lon: {float(row.lon):.6f}"
            f"</div>"
        )

        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            weight=2,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{label} без назви: {row.facility_id}",
        ).add_to(target_layer)

    layer_hospitals.add_to(m)
    layer_schools.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 22px;
        right: 22px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px 12px;
        font-family: Arial, sans-serif;
        font-size: 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">
      <b>Заклади без назви</b><br>
      <span style="color:#C0392B;">●</span> Лікарні<br>
      <span style="color:#2980B9;">●</span> Школи
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(OUT_HTML)

    print(f"08c_missing_names: знайдено {len(missing):,} закладів без назви")
    print(f"  CSV:  {OUT_CSV}")
    print(f"  HTML: {OUT_HTML}")


if __name__ == "__main__":
    run()
