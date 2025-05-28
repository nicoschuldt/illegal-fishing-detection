from datetime import datetime
from ais_fetcher import load_ais_from_csv
from pathlib import Path

# === CONFIGURATION ===
CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "drifting_longlines.csv"
MMSI_TEST = "12639560807591.0"
START_DATE = datetime(2012, 1, 1)
END_DATE = datetime(2013, 1, 15)

def test_fetch_ais():
    print("Chargement des données AIS...")
    df = load_ais_from_csv(
        mmsi=MMSI_TEST,
        csv_path=CSV_PATH,
        start=START_DATE,
        end=END_DATE
    )

    print("\nAperçu du DataFrame résultant :")
    print(df.head())

    print("\nVérification des colonnes :")
    expected_columns = [
        'mmsi',
        'timestamp',
        'lat',
        'lon',
        'speed',
        'course',
        'distance_from_shore',
        'distance_from_port'
    ]
    assert list(df.columns) == expected_columns, f"Colonnes incorrectes : {df.columns}"

    print("\nNombre de lignes :", len(df))
    print("Test manuel terminé avec succès.")

if __name__ == "__main__":
    test_fetch_ais()
