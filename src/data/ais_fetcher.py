import pandas as pd
from datetime import datetime
from typing import Optional

AIS_COLUMNS = [
    'mmsi',
    'timestamp',
    'lat',
    'lon',
    'speed',
    'course',
    'distance_from_shore',
    'distance_from_port'
]


def load_ais_from_csv(
    mmsi: str,
    csv_path: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Conversion : timestamp UNIX en float → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["mmsi"] = df["mmsi"].astype(str)
    filtered_df = df[df["mmsi"] == str(mmsi)]

    if start:
        filtered_df = filtered_df[filtered_df["timestamp"] >= start]
    if end:
        filtered_df = filtered_df[filtered_df["timestamp"] <= end]

    return filtered_df[AIS_COLUMNS].copy()
