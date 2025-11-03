import os
import pandas as pd
import pytest
from pathlib import Path

from src.train import transform_dataframe, engineer_features, load_data

@pytest.fixture
def sample_df():
    """Fixture to create a small sample dataframe similar to California Housing."""
    return pd.DataFrame({
        "AveRooms": [3.0, 5.0, 7.0],
        "AveOccup": [2.0, 2.5, 3.5],
        "MedInc": [5.0, 3.2, 8.1],
        "target_MedHouseVal": [2.5, 1.8, 4.0],
    })


def test_engineer_features(sample_df):
    """Test that the engineered feature rooms_per_occupant is added correctly."""
    df_feat = engineer_features(sample_df)
    assert "rooms_per_occupant" in df_feat.columns
    expected = sample_df["AveRooms"] / sample_df["AveOccup"]
    assert pytest.approx(df_feat["rooms_per_occupant"].iloc[0]) == expected.iloc[0]


def test_transform_dataframe_shape(sample_df):
    """Test that transform_dataframe produces non-empty dataframe and same row count."""
    df_out = transform_dataframe(sample_df, add_poly=False)
    assert not df_out.empty
    assert df_out.shape[0] == sample_df.shape[0]


def test_load_data_default():
    """Test that load_data(None) loads the California dataset."""
    df = load_data(None)
    assert isinstance(df, pd.DataFrame)
    assert "target_MedHouseVal" in df.columns
    assert len(df) > 1000


def test_output_file(tmp_path):
    """Test writing transformed data to a CSV file."""
    df = pd.DataFrame({
        "AveRooms": [1.0, 2.0],
        "AveOccup": [1.0, 2.0],
        "MedInc": [3.0, 4.0],
    })
    out_path = tmp_path / "out.csv"
    df_out = transform_dataframe(df, add_poly=False)
    df_out.to_csv(out_path, index=False)
    assert out_path.exists()
    df_reloaded = pd.read_csv(out_path)
    assert not df_reloaded.empty
