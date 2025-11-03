import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

# Import symbols from your main
from src.train import transform_dataframe, engineer_features, load_data

class TestTransform(unittest.TestCase):
    def setUp(self):
        # a tiny sample resembling California Housing columns
        self.sample = pd.DataFrame({
            "AveRooms": [3.0, 5.0, 7.0],
            "AveOccup": [2.0, 2.5, 3.5],
            "MedInc": [5.0, 3.2, 8.1],
            "target_MedHouseVal": [2.5, 1.8, 4.0],
        })

    def test_engineer_features_adds_ratio(self):
        df_feat = engineer_features(self.sample)
        self.assertIn("rooms_per_occupant", df_feat.columns)
        expected0 = self.sample["AveRooms"].iloc[0] / self.sample["AveOccup"].iloc[0]
        self.assertAlmostEqual(df_feat["rooms_per_occupant"].iloc[0], expected0, places=7)

    def test_transform_dataframe_rows_preserved(self):
        out = transform_dataframe(self.sample, add_poly=False)
        self.assertFalse(out.empty)
        self.assertEqual(len(out), len(self.sample))

    @patch("src.train.fetch_california_housing")
    def test_load_data_default_uses_california_and_renames_target(self, mock_fetch):
        # Build a fake return object with `.frame`
        fake_df = pd.DataFrame({
            "MedInc": [1.0, 2.0, 3.0],
            "MedHouseVal": [0.5, 0.6, 0.7],
        })
        fake_obj = MagicMock()
        fake_obj.frame = fake_df
        mock_fetch.return_value = fake_obj

        df = load_data(None)  # no path → should use mocked dataset
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("target_MedHouseVal", df.columns)
        self.assertNotIn("MedHouseVal", df.columns)
        self.assertEqual(len(df), 3)
        mock_fetch.assert_called_once()

    def test_writes_output_csv(self):
        # Use a temp directory provided by unittest
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.csv"
            out_df = transform_dataframe(
                pd.DataFrame({
                    "AveRooms": [1.0, 2.0],
                    "AveOccup": [1.0, 2.0],
                    "MedInc": [3.0, 4.0],
                }),
                add_poly=False
            )
            out_df.to_csv(out_path, index=False)
            self.assertTrue(out_path.exists())

            reloaded = pd.read_csv(out_path)
            self.assertFalse(reloaded.empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
