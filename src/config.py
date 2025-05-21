# src/config.py
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
RAW_DIR     = BASE_DIR.parent / "data" / "raw"

# CSV WDPA (attributs)
WDPA_CSV        = RAW_DIR / "WDPA_WDOECM_May2025_Public_marine_csv.csv"

# Shapefile split en 3 dossiers (_shp_0, _shp_1, _shp_2)
# Le loader va scanner tous les "*polygons.shp" sous RAW_DIR
WDPA_SHP_DIR = RAW_DIR / "WDPA_WDOECM_May2025_Public_marine_shp"

# Geodatabase WDPA
WDPA_GDB        = RAW_DIR / "WDPA_WDOECM_May2025_Public_marine.gdb"
WDPA_GDB_LAYER  = "WDPA_WDOECM_poly_May2025_marine"
