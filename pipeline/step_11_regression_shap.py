"""
11 Regression + SHAP + Transport Deserts.

Крок 3 методології дипломної роботи — «пояснити»:
  - GradientBoostingRegressor: I_peak ~ f(дистанція, тип, будівлі, ентропія, TGI, R, ...)
  - SHAP TreeExplainer: пояснення важливості ознак для кожного закладу
  - Транспортні пустелі: bottom-15% I_peak + R < 0.85

Виходи
------
data/processed/
  regression_features_baseline.csv        — ознаки + target (для відтворюваності)
  regression_metrics_baseline.json        — RMSE, MAE, R² на тесті
  regression_predictions_baseline.csv     — y_true, y_pred, residual на тесті
  regression_feature_importance_baseline.csv
  regression_shap_values_baseline.csv     — SHAP-значення для всіх об'єктів і ознак
  transport_deserts_baseline.csv          — «критичні зони»

data/outputs/
  regression_feature_importance_baseline.png
  regression_shap_summary_baseline.png
  regression_residuals_baseline.png
  transport_deserts_map_baseline.html
"""

from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# Допоміжні функції
# ──────────────────────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Відстань між двома точками у кілометрах (формула Гаверсина)."""
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _build_features(cfg: dict) -> "pd.DataFrame":
    """Збирає зведену таблицю ознак + target для всіх закладів."""
    import os
    import pandas as pd

    PROCESSED = "./data/processed"

    # ── завантаження ────────────────────────────────────────────────────────
    idx = pd.read_csv(f"{PROCESSED}/accessibility_index_baseline.csv", dtype={"facility_id": str})
    scores_path = cfg["paths"]["scores"]
    if not os.path.exists(scores_path):
        scores_path = "../data/processed/accessibility_scores.csv"
    scores = pd.read_csv(scores_path, usecols=["facility_id", "facility_type", "name", "lat", "lon"],
                         dtype={"facility_id": str})
    entropy = pd.read_csv(f"{PROCESSED}/facility_entropy_baseline.csv", dtype={"facility_id": str})

    # ── об'єднання ──────────────────────────────────────────────────────────
    df = (
        idx[["facility_id", "I_peak", "I_offpeak", "R", "H_norm_peak",
             "n_buildings_peak", "TGI"]]
        .merge(scores[["facility_id", "facility_type", "name", "lat", "lon"]], on="facility_id", how="left")
        .merge(entropy[["facility_id", "n_routes_peak", "stop_departures_peak"]], on="facility_id", how="left")
    )

    # ── нові ознаки ─────────────────────────────────────────────────────────
    center_lat = cfg["city"]["center_lat"]
    center_lon = cfg["city"]["center_lon"]
    df["distance_to_center_km"] = df.apply(
        lambda r: _haversine_km(r["lat"], r["lon"], center_lat, center_lon)
        if pd.notna(r["lat"]) and pd.notna(r["lon"]) else float("nan"),
        axis=1,
    )
    df["is_hospital"] = (df["facility_type"] == "hospital").astype(int)

    df = df.dropna(subset=["I_peak", "distance_to_center_km"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Основні підфункції
# ──────────────────────────────────────────────────────────────────────────────

def run_regression() -> None:
    """GradientBoostingRegressor: навчання, оцінка, SHAP, графіки."""
    from config_loader import cfg
    import json
    import os
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")

    PROCESSED = "./data/processed"
    OUTPUTS = "./data/outputs"
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(OUTPUTS, exist_ok=True)

    # ── дані ────────────────────────────────────────────────────────────────
    df = _build_features(cfg)
    FEATURE_COLS = [
        "distance_to_center_km",
        "is_hospital",
        "n_buildings_peak",
        "H_norm_peak",
        "TGI",
        "R",
        "n_routes_peak",
        "stop_departures_peak",
    ]
    # Назви для читабельних графіків
    FEATURE_LABELS = {
        "distance_to_center_km": "Відстань до центру (км)",
        "is_hospital": "Тип: лікарня",
        "n_buildings_peak": "Будинки в зоні (пік)",
        "H_norm_peak": "Ентропія маршрутів H_norm",
        "TGI": "TGI (транзитний індекс)",
        "R": "Коеф. пік/міжпік (R)",
        "n_routes_peak": "Кількість маршрутів (пік)",
        "stop_departures_peak": "Відправлення на зупинках (пік)",
    }

    # Заповнюємо можливі NaN медіаною
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df_feat = df.dropna(subset=FEATURE_COLS + ["I_peak"]).reset_index(drop=True)
    df_feat.to_csv(f"{PROCESSED}/regression_features_baseline.csv", index=False)
    print(f"11_regression: закладів у вибірці: {len(df_feat):,}")

    X = df_feat[FEATURE_COLS].values
    y = df_feat["I_peak"].values

    rs = cfg["regression"]["random_state"]
    test_size = cfg["regression"]["test_size"]
    n_est = cfg["regression"]["n_estimators"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_feat.index, test_size=test_size, random_state=rs
    )

    # ── модель ──────────────────────────────────────────────────────────────
    model = GradientBoostingRegressor(
        n_estimators=n_est,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=rs,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2,
               "n_train": int(len(y_train)), "n_test": int(len(y_test))}
    with open(f"{PROCESSED}/regression_metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"11_regression: RMSE={rmse:.6f}  MAE={mae:.6f}  R²={r2:.4f}")

    # predictions CSV
    pred_df = df_feat.loc[idx_test, ["facility_id", "name", "facility_type", "I_peak"]].copy()
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = pred_df["I_peak"] - pred_df["y_pred"]
    pred_df.to_csv(f"{PROCESSED}/regression_predictions_baseline.csv", index=False)

    # ── feature importance ───────────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "label": [FEATURE_LABELS[c] for c in FEATURE_COLS],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(f"{PROCESSED}/regression_feature_importance_baseline.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(fi["label"][::-1], fi["importance"][::-1], color="#2196F3", edgecolor="white")
    ax.set_xlabel("Важливість ознаки (GBM)", fontsize=11)
    ax.set_title("Важливість ознак — регресія I*_peak", fontsize=13, fontweight="bold")
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in fi["importance"][::-1]], padding=3, fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{OUTPUTS}/regression_feature_importance_baseline.png", dpi=150)
    plt.close(fig)
    print(f"11_regression: графік важливості → {OUTPUTS}/regression_feature_importance_baseline.png")

    # ── SHAP ────────────────────────────────────────────────────────────────
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)   # для всієї вибірки

    shap_df = pd.DataFrame(shap_values.values, columns=FEATURE_COLS)
    shap_df.insert(0, "facility_id", df_feat["facility_id"].values)
    shap_df.to_csv(f"{PROCESSED}/regression_shap_values_baseline.csv", index=False)

    # summary plot (beeswarm)
    fig_shap, ax_shap = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values.values,
        X,
        feature_names=[FEATURE_LABELS[c] for c in FEATURE_COLS],
        show=False,
        plot_size=None,
    )
    plt.title("SHAP — вплив ознак на I*_peak (beeswarm)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig_shap.savefig(f"{OUTPUTS}/regression_shap_summary_baseline.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"11_regression: SHAP-графік → {OUTPUTS}/regression_shap_summary_baseline.png")

    # ── residuals plot ────────────────────────────────────────────────────────
    fig_r, ax_r = plt.subplots(figsize=(7, 5))
    ax_r.scatter(y_pred, pred_df["residual"], alpha=0.45, s=18, color="#FF7043")
    ax_r.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_r.set_xlabel("Прогноз I*_peak", fontsize=11)
    ax_r.set_ylabel("Залишок (факт − прогноз)", fontsize=11)
    ax_r.set_title("Залишки регресії", fontsize=13, fontweight="bold")
    ax_r.text(0.98, 0.97, f"RMSE={rmse:.5f}\nMAE={mae:.5f}\nR²={r2:.3f}",
              transform=ax_r.transAxes, ha="right", va="top", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.tight_layout()
    fig_r.savefig(f"{OUTPUTS}/regression_residuals_baseline.png", dpi=150)
    plt.close(fig_r)
    print(f"11_regression: залишки → {OUTPUTS}/regression_residuals_baseline.png")


def run_transport_deserts() -> None:
    """Виявляє транспортні пустелі й будує інтерактивну карту."""
    from config_loader import cfg
    import os
    import warnings

    import folium
    import pandas as pd

    warnings.filterwarnings("ignore")

    PROCESSED = "./data/processed"
    OUTPUTS = "./data/outputs"
    os.makedirs(PROCESSED, exist_ok=True)
    os.makedirs(OUTPUTS, exist_ok=True)

    df = _build_features(cfg)

    # ── критерії ────────────────────────────────────────────────────────────
    threshold_ipeak = df["I_peak"].quantile(0.15)   # bottom 15%
    threshold_r = 0.85                              # деградація поза піком

    df["desert_critical"] = df["I_peak"] <= threshold_ipeak          # низький індекс
    df["desert_offpeak"]  = df["R"] < threshold_r                    # погіршується поза піком
    df["is_desert"]       = df["desert_critical"] | df["desert_offpeak"]

    deserts = df[df["is_desert"]].copy()
    deserts["desert_reason"] = deserts.apply(
        lambda r: (
            "критично низький I_peak + погіршення поза піком"
            if r["desert_critical"] and r["desert_offpeak"]
            else ("критично низький I_peak" if r["desert_critical"] else "погіршення поза піком")
        ),
        axis=1,
    )

    out_cols = ["facility_id", "name", "facility_type", "lat", "lon",
                "I_peak", "R", "desert_critical", "desert_offpeak", "desert_reason"]
    deserts[out_cols].to_csv(f"{PROCESSED}/transport_deserts_baseline.csv", index=False)
    print(f"11_deserts: {len(deserts):,} транспортних пустель")
    print(f"  └ критично низький I_peak (≤{threshold_ipeak:.6f}): "
          f"{deserts['desert_critical'].sum():,}")
    print(f"  └ R < {threshold_r}: {deserts['desert_offpeak'].sum():,}")

    # ── карта ────────────────────────────────────────────────────────────────
    center_lat = cfg["city"]["center_lat"]
    center_lon = cfg["city"]["center_lon"]
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                      tiles="CartoDB positron")

    # всі заклади — сірим
    all_valid = df.dropna(subset=["lat", "lon"])
    for row in all_valid[~all_valid["is_desert"]].itertuples():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color="#9E9E9E",
            fill=True,
            fill_opacity=0.35,
            weight=0.5,
            popup=folium.Popup(
                f"<b>{row.name}</b><br>{row.facility_type}<br>"
                f"I_peak: {row.I_peak:.5f}<br>R: {row.R:.3f}",
                max_width=220,
            ),
        ).add_to(fmap)

    # пустелі — кольором за типом
    COLOR_MAP = {
        "критично низький I_peak + погіршення поза піком": "#D32F2F",
        "критично низький I_peak": "#F57C00",
        "погіршення поза піком": "#1976D2",
    }
    for row in deserts.dropna(subset=["lat", "lon"]).itertuples():
        color = COLOR_MAP.get(row.desert_reason, "#D32F2F")
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=1.2,
            popup=folium.Popup(
                f"<b>{row.name}</b><br>{row.facility_type}<br>"
                f"<b style='color:{color}'>{row.desert_reason}</b><br>"
                f"I_peak: {row.I_peak:.5f}<br>R: {row.R:.3f}",
                max_width=260,
            ),
        ).add_to(fmap)

    # Легенда
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        border: 1px solid #ccc; font-size: 13px; line-height: 1.8;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
    <b>Транспортні пустелі</b><br>
    <span style="color:#D32F2F">&#9679;</span> Критично низький + деградація вночі<br>
    <span style="color:#F57C00">&#9679;</span> Критично низький I_peak (bottom 15%)<br>
    <span style="color:#1976D2">&#9679;</span> Погіршення поза піком (R &lt; 0.85)<br>
    <span style="color:#9E9E9E">&#9679;</span> Решта закладів
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    out_map = f"{OUTPUTS}/transport_deserts_map_baseline.html"
    fmap.save(out_map)
    print(f"11_deserts: карта → {out_map}")


# ──────────────────────────────────────────────────────────────────────────────

def run() -> None:
    run_regression()
    run_transport_deserts()


if __name__ == "__main__":
    run()
