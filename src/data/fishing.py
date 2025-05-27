from datetime import datetime, timedelta
import pandas as pd
from .gfw import gfw_client

def fetch_ais(mmsi: str, hours: int = 12) -> pd.DataFrame:
    """
    Récupère les points AIS d'un navire via GFW API pour les dernières `hours` heures.
    Retourne un DataFrame pandas avec au minimum :
      - timestamp (datetime)
      - lat, lon
      - speed (anciennement sog)
      - course (anciennement cog)
      - is_fishing (confiance fishing)
      - mmsi (str)
    """
    # valider MMSI
    try:
        mmsi_int = int(mmsi)
    except ValueError:
        raise ValueError(f"MMSI invalide : {mmsi}")

    # bornes temporelles
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)

    # appel à l'API GFW
    resp = gfw_client.vessels.get_tracks(
        mmsi = mmsi_int,
        start = start.isoformat() + "Z",
        end   = end.isoformat()   + "Z"
    )
    data = resp.data or []
    if not data:
        # aucun point AIS pour cette période
        return pd.DataFrame()

    # construction du DataFrame
    df = pd.DataFrame(data)

    # normalisation colonnes
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    rename_map = {}
    if "sog" in df.columns:
        rename_map["sog"] = "speed"
    if "cog" in df.columns:
        rename_map["cog"] = "course"
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # ajout du MMSI en str
    df["mmsi"] = str(mmsi_int)

    # tri chronologique et reset index
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
