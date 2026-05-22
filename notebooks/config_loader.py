"""Завантажує config.toml і надає зручний доступ до параметрів."""
import tomli
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

with open(_CONFIG_PATH, "rb") as f:
    cfg = tomli.load(f)

# Зручні аліаси для найчастіше вживаних параметрів
CITY          = cfg["city"]["name"]
CRS_METRIC    = cfg["city"]["crs_metric"]
KYIV_CENTER   = [cfg["city"]["center_lat"], cfg["city"]["center_lon"]]

WALK_10MIN_M  = cfg["isochrone"]["walk_10min_m"]
WALK_30MIN_M  = cfg["isochrone"]["walk_30min_m"]
ACC_RADIUS    = cfg["isochrone"]["accessibility_radius"]   # "10min" або "30min"

W_STOPS       = cfg["accessibility"]["weight_stops"]
W_ROUTES      = cfg["accessibility"]["weight_routes"]
W_FREQ        = cfg["accessibility"]["weight_frequency"]

KMEANS_K      = cfg["clustering"]["kmeans_k"]
RANDOM_STATE  = cfg["clustering"]["random_state"]
