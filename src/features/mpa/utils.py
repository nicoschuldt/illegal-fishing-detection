# src/feature/mpa_module/utils.py

from shapely.geometry import Point
from typing import Optional, Dict
import pandas as pd


class MPAQuery:
    """
    Fournit des méthodes pour interroger la couche MPA chargée par MPALoader :
     - existence d’une MPA à un point,
     - attributs réglementaires,
     - activités de pêche interdites,
     - distance à la MPA la plus proche.
    """

    def __init__(self, loader):
        self.gdf = loader.gdf
        self.sindex = loader.sindex

    def _find_candidates(self, pt: Point):
        # Recherche rapide via index puis filtrage précis
        idx = list(self.sindex.query(pt, predicate="intersects"))
        return self.gdf.iloc[idx]

    def is_in_mpa(self, lat: float, lon: float) -> bool:
        pt = Point(lon, lat)
        cands = self._find_candidates(pt)
        hits = cands[cands.contains(pt)]
        return not hits.empty

    def get_mpa_attributes(self, lat: float, lon: float) -> Optional[Dict]:
        pt = Point(lon, lat)
        cands = self._find_candidates(pt)
        hits = cands[cands.contains(pt)]
        if hits.empty:
            return None
        row = hits.iloc[0]
        # Champs clés – complète selon besoin
        return {
            "WDPAID": row["WDPAID"],
            "NAME": row.get("NAME"),
            "DESIG": row.get("DESIG"),
            "DESIG_TYPE": row.get("DESIG_TYPE"),
            "IUCN_CAT": row.get("IUCN_CAT"),
            "NO_TAKE": row.get("NO_TAKE"),       # All/Part/None/Not Applicable
            "GIS_AREA": row.get("GIS_AREA"),
        }

    def get_prohibited_activities(self, lat: float, lon: float) -> Optional[list]:
        """
        Renvoie la liste des activités interdites, si champ réglementaire dispo
        (ex. ACTIVITE_REG, REGULATI_1). Les listes devraient être délimitées
        par point-virgule selon la doc WDPA. 
        """
        pt = Point(lon, lat)
        cands = self._find_candidates(pt)
        hits = cands[cands.contains(pt)]
        if hits.empty:
            return None
        raw = hits.iloc[0].get("ACTIVITE_REG") or hits.iloc[0].get("REGULATI_1")
        if not raw or pd.isna(raw):
            return None
        return [act.strip() for act in str(raw).split(";") if act.strip()]

    def distance_to_nearest_mpa(self, lat: float, lon: float) -> float:
        pt = Point(lon, lat)
        # Distance minimale (en degrés) au polygone le plus proche
        return float(self.gdf.distance(pt).min())
