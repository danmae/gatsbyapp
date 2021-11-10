import logging
import unittest

import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
import random
from ill_feature_generation.Feature_Generator.FeatureGeneratorRefCycle import (
    FeatureGeneratorRefCycle,
)
from ill_commons.cycle.RawCycle import RawCycle
import pyarrow as pa


class TestFeatureGenerator(unittest.TestCase):
    def setUp(self):
        """ SetUp """

        self.logger = logging.getLogger()
        self.machine_id = "MachineID"
        self.config = {
            "MachineID": {
                "ignore": [],
                "timenorm": [],
                "resampling": {"upper_limit": 200, "lower_limit": 0},
            }
        }
        timestamps = [
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:20.061000"),
            pd.Timestamp("2019-09-01 22:39:23.161000"),
            pd.Timestamp("2019-09-01 22:39:24.490000"),
            pd.Timestamp("2019-09-01 22:39:25.598000"),
        ]
        data_1 = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "timestamp": timestamps * 2,
                    "parameter_name": ["a"] * 5 + ["b"] * 5,
                    "value": [1, 1, 2, 2, 1, 0, 0, 0, 0, 0],
                    "uuid": ["uuid1"] * 5 + ["uuid2"] * 5,
                    "uuid": ["uuid1"] * 5 + ["uuid2"] * 5,
                }
            )
        )
        data_2 = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "timestamp": timestamps + timestamps[:2] + [timestamps[-1]],
                    "parameter_name": ["a"] * 5 + ["b"] * 3,
                    "value": [1, 1, 2, 2, 1, 0, 1, 1],
                    "uuid": ["uuid1"] * 5 + ["uuid2"] * 3,
                }
            )
        )
        data_3 = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "timestamp": timestamps * 2,
                    "parameter_name": ["a"] * 5 + ["b"] * 5,
                    "value": [1, 1, 3, 3, 1, np.nan, 1, 0, 1, 0],
                    "uuid": ["uuid1"] * 5 + ["uuid2"] * 5,
                }
            )
        )
        data_4 = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "timestamp": timestamps[:4] * 2,
                    "parameter_name": ["a"] * 4 + ["b"] * 4,
                    "value": [1, 1, 0, 2, 0, 0, 1, 0],
                    "uuid": ["uuid1"] * 4 + ["uuid2"] * 4,
                }
            )
        )
        data_5 = pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "timestamp": [
                        pd.Timestamp("2019-09-01 22:39:19.161000"),
                        pd.Timestamp("2019-09-01 22:39:20.061000"),
                        pd.Timestamp("2019-09-01 22:39:23.161000"),
                        pd.Timestamp("2019-09-01 22:39:24.490000"),
                        pd.Timestamp("2019-09-01 22:39:25.598000"),
                        pd.Timestamp("2019-09-01 22:39:26.598000"),
                        pd.Timestamp("2019-09-01 22:39:28.598000"),
                    ]
                    * 2,
                    "parameter_name": ["a"] * 7 + ["b"] * 7,
                    "value": [1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "uuid": ["uuid1"] * 7 + ["uuid2"] * 7,
                }
            )
        )

        self.cycle_1 = RawCycle(
            timestamps[0],
            timestamps[-1],
            "machineID",
            "variant1",
            data_1,
            "id1",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_2 = RawCycle(
            timestamps[0],
            timestamps[-1],
            "machineID",
            "variant1",
            data_2,
            "id2",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_3 = RawCycle(
            timestamps[0],
            timestamps[-1],
            "machineID",
            "variant1",
            data_3,
            "id3",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_4 = RawCycle(
            timestamps[0],
            timestamps[3],
            "machineID",
            "variant1",
            data_4,
            "id4",
            continuous=["a", "b"],
            binary=[],
        )

        self.cycle_5 = RawCycle(
            timestamps[0],
            pd.Timestamp("2019-09-01 22:39:28.598000"),
            "machineID",
            "variant1",
            data_5,
            "id5",
            continuous=["a", "b"],
            binary=[],
        )

    def test_fit_no_input(self):
        # construct input data
        no_cycles = []
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        with self.assertRaises(Exception) as context:
            ref.fit(no_cycles, self.machine_id, self.config["MachineID"], self.logger)
        self.assertTrue("No Cycles passed into fit()" in str(context.exception))

    def test_fit_single_input(self):
        # construct input data
        cycles = [self.cycle_1]
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        ref.fit(cycles, self.machine_id, self.config["MachineID"], self.logger)
        observed = ref.reference_object
        expected = cycles[0].data.to_pandas()[
            ["timestamp", "value", "parameter_name", "uuid"]
        ]
        assert_frame_equal(
            observed[["timestamp", "value", "parameter_name", "uuid"]], expected
        )

    def test_fit(self):

        # select FeatureGenerator
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        # construct input data
        cycles = [self.cycle_1, self.cycle_2, self.cycle_3, self.cycle_4]
        # calculate data with function and store result in observed
        ref.fit(cycles, self.machine_id, self.config["MachineID"], self.logger)
        observed = ref.reference_object
        # define expected outcome
        expected = cycles[1].data.to_pandas()[
            ["timestamp", "value", "parameter_name", "uuid"]
        ]
        assert_frame_equal(
            observed[["timestamp", "value", "parameter_name", "uuid"]], expected
        )

    def test_score_same_cycle(self):
        # construct input data
        input_cycle = self.cycle_1
        cycles = [self.cycle_1]
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        ref.fit(cycles, self.machine_id, self.config["MachineID"], self.logger)
        input_cycle_df = input_cycle.data.to_pandas()
        input_cycle_df["c_id"] = input_cycle.id

        # calculate data with fucntion -> observed
        observed_prediction, observed_alignment, observed_score = ref.predict(
            input_cycle_df, logging.getLogger()
        )
        # define expected outcome
        expected_prediction = pd.DataFrame(
            {
                "p_uuid": ["uuid1", "uuid2"],
                "feature_value": [1.0, 1.0],
                "c_id": [input_cycle.id] * 2,
            }
        )

        expected_alignment = pd.DataFrame(
            {
                "n_interval": [1, 1],
                "c_id": [input_cycle.id] * 2,
                "p_uuid": ["uuid1"] + ["uuid2"],
                "ratio": [1, 1],
                "alignment_offset": [1, 1],
                "duration": [6.437, 6.437],
                "similarity_score": [1.0, 1.0],
            },
            index=[0, 0],
        )
        expected_score = 1.0

        assert_frame_equal(
            observed_prediction.sort_values("p_uuid").reset_index(drop=True),
            expected_prediction[observed_prediction.columns]
            .sort_values("p_uuid")
            .reset_index(drop=True),
            check_exact=True,
        )
        assert_frame_equal(
            observed_alignment,
            expected_alignment[observed_alignment.columns],
            check_dtype=False,
        )
        assert observed_score == expected_score

    def test_score_shorter_cycle(self):
        # construct input data
        input_cycle = self.cycle_4
        cycles = [self.cycle_1]
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        ref.fit(cycles, self.machine_id, self.config["MachineID"], self.logger)

        input_cycle_df = input_cycle.data.to_pandas()
        input_cycle_df["c_id"] = input_cycle.id
        # calculate data with fucntion -> observed
        observed_prediction, observed_alignment, observed_score = ref.predict(
            input_cycle_df, logging.getLogger()
        )
        # define expected outcome
        expected_prediction = pd.DataFrame(
            {
                "p_uuid": ["uuid1", "uuid2"],
                "feature_value": [0.6, 0.8],
                "c_id": [input_cycle.id] * 2,
            }
        )
        expected_alignment = pd.DataFrame(
            {
                "n_interval": [1, 2, 3, 1, 2, 3],
                "c_id": [input_cycle.id] * 6,
                "p_uuid": ["uuid1"] * 3 + ["uuid2"] * 3,
                "ratio": [0.8] * 6,
                "alignment_offset": [
                    1.0,
                    0.00990099009900991,
                    1,
                    1,
                    0.00990099009900991,
                    1,
                ],
                "duration": [0, 0.9, 4.429, 0, 0.9, 4.429],
                "similarity_score": [0.6, 0.6, 0.6, 0.8, 0.8, 0.8],

            },
            index=[0, 1,2, 0, 1,2],
        )
        expected_score = 0.7

        assert_frame_equal(
            observed_prediction.sort_values("p_uuid").reset_index(drop=True),
            expected_prediction[observed_prediction.columns]
            .sort_values("p_uuid")
            .reset_index(drop=True),
            check_exact=True,
        )
        assert_frame_equal(
            observed_alignment,
            expected_alignment[observed_alignment.columns],
            check_dtype=False,
        )
        assert observed_score == expected_score

    def test_score_longer_cycle(self):
        # construct input data
        input_cycle = self.cycle_5
        cycles = [self.cycle_1]
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        ref.fit(cycles, self.machine_id, self.config["MachineID"], self.logger)
        input_cycle_df = input_cycle.data.to_pandas()
        input_cycle_df["c_id"] = input_cycle.id
        # calculate data with fucntion -> observed
        observed_prediction, observed_alignment, observed_score = ref.predict(
            input_cycle_df, logging.getLogger()
        )
        # define expected outcome
        expected_prediction = pd.DataFrame(
            {
                "p_uuid": ["uuid1", "uuid2"],
                "feature_value": [0.6, 1.0],
                "c_id": [input_cycle.id] * 2,
            }
        )
        expected_alignment = pd.DataFrame(
            {
                "n_interval": [1, 2, 1],
                "c_id": [input_cycle.id] * 3,
                "p_uuid": ["uuid1"] * 2 + ["uuid2"] * 1,
                "ratio": [1.4, 1.4, 1],
                "alignment_offset": [1.0, 0.00990099009900991, 1],
                "duration": [5.329, 4.108, 9.437],
                "similarity_score": [0.6, 0.6, 1],
            },
            index=[0, 1, 0],
        )
        expected_score = 0.8

        assert_frame_equal(
            observed_prediction.sort_values("p_uuid").reset_index(drop=True),
            expected_prediction[observed_prediction.columns]
            .sort_values("p_uuid")
            .reset_index(drop=True),
            check_exact=True,
        )
        assert_frame_equal(
            observed_alignment,
            expected_alignment[observed_alignment.columns],
            check_dtype=False,
        )
        assert observed_score == expected_score

    def test_calculate_cycle_score(self):
        ref = FeatureGeneratorRefCycle({"dtw_pattern": "symmetric1"})
        similarity_df = pd.DataFrame(
            {"p_uuid": ["uuid1", "uuid2"], "feature_value": [0.9, 0.83333]}
        )
        # calculate value
        observed = ref.calculate_cycle_score(similarity_df)
        # expected
        expected = 0.86666
        assert observed == expected

    def test_calculate_cycle_score_multivariate(self):
        # select FeatureGenerator
        ref = FeatureGeneratorRefCycle(
            {"dtw_pattern": "symmetric1", "multivariate": True}
        )
        # construct input data
        cycles = [self.cycle_1, self.cycle_2, self.cycle_3, self.cycle_4]
        # calculate fit with other similarity function
        ref.fit(cycles, self.machine_id, self.config["MachineID"], self.logger)

        # check for right output
        observed = ref.reference_object
        # summed distances -> c1: 10.2 c2: 10 c3: 10.6 c4: 10.2
        expected = cycles[3].data.to_pandas()[
            ["timestamp", "value", "parameter_name", "uuid"]
        ]
        assert_frame_equal(
            observed[["timestamp", "value", "parameter_name", "uuid"]], expected
        )