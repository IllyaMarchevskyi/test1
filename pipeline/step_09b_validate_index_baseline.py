"""
09b Validate Accessibility Index Baseline.

Готує топ-5 і боттом-5 закладів за I*_peak та будує окрему карту
для швидкої візуальної валідації змісту.
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
    INDEX_PREVIEW_PATH = f"{PROCESSED_DIR}/accessibility_index_preview_baseline.csv"
    CATCHMENT_RESULTS_PATH = f"{PROCESSED_DIR}/catchment_results_baseline.csv"
    SCORES_PATH = cfg["paths"]["scores"]
    OUT_TOP5 = f"{PROCESSED_DIR}/accessibility_index_top5_baseline.csv"
    OUT_BOTTOM5 = f"{PROCESSED_DIR}/accessibility_index_bottom5_baseline.csv"
    OUT_HTML = f"{OUTPUTS_DIR}/accessibility_index_extremes_baseline.html"

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    required = [INDEX_PREVIEW_PATH, CATCHMENT_RESULTS_PATH, SCORES_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Відсутні входи для 09_validate: {missing}")

    if all(os.path.exists(path) for path in [OUT_TOP5, OUT_BOTTOM5, OUT_HTML]):
        outputs_mtime = min(os.path.getmtime(OUT_TOP5), os.path.getmtime(OUT_BOTTOM5), os.path.getmtime(OUT_HTML))
        inputs_mtime = max(os.path.getmtime(path) for path in required)
        if outputs_mtime >= inputs_mtime:
            print("09_validate: кеш валідації вже актуальний.")
            print(f"  top5:    {OUT_TOP5}")
            print(f"  bottom5: {OUT_BOTTOM5}")
            print(f"  map:     {OUT_HTML}")
            return

    print("09_validate: завантажуємо index preview, catchment results і координати закладів...")
    index_preview = pd.read_csv(INDEX_PREVIEW_PATH)
    catchment = pd.read_csv(CATCHMENT_RESULTS_PATH)
    scores = pd.read_csv(SCORES_PATH, usecols=["facility_id", "lat", "lon"])

    index_preview["facility_id"] = index_preview["facility_id"].astype(str)
    catchment["facility_id"] = catchment["facility_id"].astype(str)
    scores["facility_id"] = scores["facility_id"].astype(str)
    index_preview["I_peak"] = pd.to_numeric(index_preview["I_peak"], errors="coerce").fillna(0.0)
    index_preview["I_offpeak"] = pd.to_numeric(index_preview["I_offpeak"], errors="coerce").fillna(0.0)
    index_preview["R"] = pd.to_numeric(index_preview["R"], errors="coerce")
    index_preview["TGI"] = pd.to_numeric(index_preview["TGI"], errors="coerce")

    merged = index_preview.merge(catchment, on=["facility_id", "facility_type", "name"], how="left")
    merged = merged.merge(scores, on="facility_id", how="left")

    top5 = merged.sort_values(["I_peak", "I_offpeak"], ascending=[False, False]).head(5).copy()
    bottom5 = merged.sort_values(["I_peak", "I_offpeak"], ascending=[True, True]).head(5).copy()

    top5.to_csv(OUT_TOP5, index=False, encoding="utf-8")
    bottom5.to_csv(OUT_BOTTOM5, index=False, encoding="utf-8")

    m = folium.Map(
        location=[cfg["city"]["center_lat"], cfg["city"]["center_lon"]],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    layer_top = folium.FeatureGroup(name="Топ-5 за I*_peak", show=True)
    layer_bottom = folium.FeatureGroup(name="Боттом-5 за I*_peak", show=True)

    def add_rows(df: pd.DataFrame, layer, color: str, label: str) -> None:
        for row in df.itertuples(index=False):
            name = str(row.name) if pd.notna(row.name) else "Без назви"
            popup_html = (
                f"<div style='font-family:Arial,sans-serif;font-size:13px;width:260px'>"
                f"<b>{name[:60]}</b><br>"
                f"ID: <b>{row.facility_id}</b><br>"
                f"Тип: <b>{'Лікарня' if row.facility_type == 'hospital' else 'Школа'}</b><br>"
                f"<hr style='margin:6px 0'>"
                f"I*_peak: <b>{float(row.I_peak):.4f}</b><br>"
                f"I*_offpeak: <b>{float(row.I_offpeak):.4f}</b><br>"
                f"R: <b>{'—' if pd.isna(row.R) else f'{float(row.R):.4f}'}</b><br>"
                f"TGI: <b>{'—' if pd.isna(row.TGI) else f'{float(row.TGI):.4f}'}</b><br>"
                f"<hr style='margin:6px 0'>"
                f"Пік 10 хв: {int(getattr(row, 'peak_total_10min', 0) or 0):,}<br>"
                f"Пік 30 хв: {int(getattr(row, 'peak_total_30min', 0) or 0):,}<br>"
                f"Міжпік 10 хв: {int(getattr(row, 'offpeak_total_10min', 0) or 0):,}<br>"
                f"Міжпік 30 хв: {int(getattr(row, 'offpeak_total_30min', 0) or 0):,}"
                f"</div>"
            )
            folium.CircleMarker(
                location=[float(row.lat), float(row.lon)],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                weight=2,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{label}: {name[:45]}",
            ).add_to(layer)

    add_rows(top5, layer_top, "#1B9E3E", "Топ")
    add_rows(bottom5, layer_bottom, "#C0392B", "Боттом")

    layer_top.add_to(m)
    layer_bottom.add_to(m)
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
      <b>Валідація I*_peak</b><br>
      <span style="color:#1B9E3E;">●</span> Топ-5<br>
      <span style="color:#C0392B;">●</span> Боттом-5
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(OUT_HTML)

    print("09_validate: підготовлено файли для валідації.")
    print(f"  top5:    {OUT_TOP5}")
    print(f"  bottom5: {OUT_BOTTOM5}")
    print(f"  map:     {OUT_HTML}")

    print("\nТоп-5 за I*_peak:")
    for row in top5.itertuples(index=False):
        print(f"  {str(row.name)[:55]:<55} {row.facility_type:<8} I*={float(row.I_peak):.4f}")

    print("\nБоттом-5 за I*_peak:")
    for row in bottom5.itertuples(index=False):
        print(f"  {str(row.name)[:55]:<55} {row.facility_type:<8} I*={float(row.I_peak):.4f}")


if __name__ == "__main__":
    run()
