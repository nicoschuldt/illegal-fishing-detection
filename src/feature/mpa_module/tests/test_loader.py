# src/feature/mpa_module/tests/test_loader.py
import pytest
from pathlib import Path
from feature.mpa_module.loader import MPALoader
from feature.mpa_module.utils import MPAQuery

DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "tests"

@pytest.fixture
def query():
    loader = MPALoader(
        shp_path=DATA_DIR / "mini_shp",
        csv_path=DATA_DIR / "mini_csv.csv"
    )
    return MPAQuery(loader)

def test_inside(query):
    # Coordonnées connues à l’intérieur d’une MPA
    assert query.is_in_mpa(48.1, -4.5)

def test_outside(query):
    # Un point en pleine mer
    assert not query.is_in_mpa(0.0, 0.0)

def test_attributes(query):
    attrs = query.get_mpa_attributes(48.1, -4.5)
    assert attrs is not None
    assert attrs["WDPAID"] == 100001

def test_prohibited(query):
    assert query.get_prohibited_activities(48.1, -4.5) is None
