from __future__ import annotations

import unittest

import pandas as pd

from predictx.raw import parse_spxraw_frame


class RawParserSyntheticFixtureTest(unittest.TestCase):
    def test_synthetic_spxraw_string_is_cumsumed_and_btps_adjusted(self) -> None:
        column_map = {
            "fields": {
                "participant_id": {"E": "SEQN"},
                "raw_curve": {"E": "SPXRAW"},
                "n_points": {"E": "SPXPTS"},
                "btps_factor": {"E": "SPAFACT"},
                "test_type": {"E": "SPATTYPE"},
                "maneuver_num": {"E": "SPAMANU"},
                "acceptable": {"E": "SPAACC"},
                "position": {"E": "SPAPOS"},
                "plateau": {"E": "SPAPLAT"},
                "raw_effort_grade": {"E": "SPAQEFF"},
            }
        }
        synthetic = pd.DataFrame(
            [
                {
                    "SEQN": 1,
                    "SPXRAW": "1,2,3",
                    "SPXPTS": 3,
                    "SPAFACT": 2.0,
                    "SPATTYPE": 1,
                    "SPAMANU": 7,
                    "SPAACC": 1,
                    "SPAPOS": 0,
                    "SPAPLAT": 1,
                    "SPAQEFF": 2,
                }
            ]
        )

        parsed = parse_spxraw_frame(synthetic, column_map, "E")

        self.assertEqual(parsed["volume_mL"].tolist(), [2.0, 6.0, 12.0])
        self.assertEqual(parsed["flow_interval_mL"].tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(parsed["flow_mL_s"].tolist(), [200.0, 400.0, 600.0])
        self.assertEqual(parsed["time_s"].tolist(), [0.01, 0.02, 0.03])
        self.assertEqual(parsed["maneuver_num"].unique().tolist(), [7])


if __name__ == "__main__":
    unittest.main()
