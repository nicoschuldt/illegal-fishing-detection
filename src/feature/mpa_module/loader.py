# src/feature/mpa_module/loader.py

import geopandas as gpd
import pandas as pd
from pathlib import Path
import glob
import fiona


class MPALoader:
    """
    Charge et prépare la couche MPA à partir de :
     - un CSV WDPA (pour filtrer les zones marines),
     - un dossier Shapefile *polygons.shp*,
     - ou une Geodatabase (.gdb) et son layer.
    """

    def __init__(self,
                 shp_path: Path = None,
                 gdb_path: Path = None,
                 csv_path: Path = None,
                 layer_name: str = None,
                 crs: str = "EPSG:4326"):
        # 1) Charger CSV WDPA et extraire les WDPAID de toutes les zones à composante marine
        marine_ids = None
        if csv_path:
            df = pd.read_csv(csv_path, low_memory=False)
            # MARINE ∈ {0,1,2} : 1 = mixte, 2 = tout marine :contentReference[oaicite:0]{index=0}
            df["MARINE"] = df["MARINE"].astype(int)
            marine_ids = set(df.loc[df["MARINE"] > 0, "WDPAID"])

        # 2) Charger la géométrie polygons depuis GDB ou Shapefile
        if gdb_path and layer_name:
            # Valider l’existence du layer
            layers = fiona.listlayers(str(gdb_path))
            if layer_name not in layers:
                raise FileNotFoundError(f"Layer '{layer_name}' non trouvé dans {gdb_path}")
            # Lire manuellement la couche GDB via fiona.open pour éviter l'appel à fiona.path
            with fiona.open(str(gdb_path), layer=layer_name) as src:
                # src est un itérable de features, src.crs donne la CRS
                self.gdf = gpd.GeoDataFrame.from_features(src, crs=src.crs)

        elif shp_path:
            shp_path = Path(shp_path)
            shp_files = list(shp_path.rglob("*polygons.shp"))
            parts = []
            for fp in shp_files:
                with fiona.open(fp) as src:
                    meta = src.meta
                    feats = (
                        feat for feat in src
                        if int(feat["properties"]["MARINE"]) > 0
                    )
                    parts.append(
                        gpd.GeoDataFrame.from_features(feats, crs=meta["crs"])
                    )
            self.gdf = pd.concat(parts, ignore_index=True)
            self.gdf = gpd.GeoDataFrame(self.gdf, crs=parts[0].crs)

        else:
            raise ValueError("Il faut fournir soit shp_path, soit gdb_path + layer_name.")

        # 3) Vérifier la présence des champs minimaux
        required = {"WDPAID", "MARINE", "PA_DEF"}
        missing = required - set(self.gdf.columns)
        if missing:
            raise ValueError(f"Champs obligatoires manquants : {missing}")

        # 4) Filtrer sur les MPAs marines si on a un CSV
        if marine_ids is not None:
            self.gdf = self.gdf[self.gdf["WDPAID"].isin(marine_ids)]

        # 5) Reprojection en WGS84 et création de l’index spatial
        self.gdf = self.gdf.to_crs(crs)
        self.sindex = self.gdf.sindex
