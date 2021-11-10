import unittest

import pandas as pd
import numpy as np
from pandas._testing import assert_frame_equal
import random
from ill_feature_generation.Feature_Generator.FeatureGeneratorRefCycle import (
    FeatureGeneratorRefCycle,
)
from ill_commons.cycle.RawCycle import RawCycle


class TestFeatureGenerator(unittest.TestCase):
    def setUp(self):
        """ etUp """

        timestamps = [
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.270000"),
            pd.Timestamp("2019-09-01 22:39:19.380000"),
            pd.Timestamp("2019-09-01 22:39:19.490000"),
            pd.Timestamp("2019-09-01 22:39:19.598000"),
        ]
        data_1 = pd.DataFrame(
            {
                "timestamp": timestamps * 2,
                "parameter_name": ["a"] * 5 + ["b"] * 5,
                "value": [1, 1, 2, 2, 1, 0, 0, 0, 0, 0],
            }
        )
        data_2 = pd.DataFrame(
            {
                "timestamp": timestamps * 2
                + [pd.Timestamp("2019-09-01 22:39:19.698000")]
                + [pd.Timestamp("2019-09-01 22:39:19.898000")]
                + [pd.Timestamp("2019-09-01 22:39:20.108000")],
                "parameter_name": ["a"] * 5 + ["b"] * 8,
                "value": [1, 1, 2, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            }
        )
        data_3 = pd.DataFrame(
            {
                "timestamp": timestamps * 2,
                "parameter_name": ["a"] * 5 + ["b"] * 5,
                "value": [1, 1, 3, 3, 1, 0, 1, 1, 0, 0],
            }
        )
        data_4 = pd.DataFrame(
            {
                "timestamp": timestamps * 2
                + [pd.Timestamp("2019-09-01 22:39:19.698000")],
                "parameter_name": ["a"] * 5 + ["b"] * 6,
                "value": [1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 1],
            }
        )
        data_5 = pd.DataFrame(
            {
                "timestamp": timestamps * 2
                + [pd.Timestamp("2019-09-01 22:39:19.698000")]
                + [pd.Timestamp("2019-09-01 22:39:12.698000")],
                "parameter_name": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                "value": [0,0,0,0,1,1,1,1,0,0,0,0],
            }
        )
        data_6 = pd.DataFrame(
            {
                "timestamp": timestamps * 2
                + [pd.Timestamp("2019-09-01 22:39:19.698000")]
                + [pd.Timestamp("2019-09-01 22:39:12.698000")],
                "parameter_name": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                "value": [4,2,1,4,2,3,1,4,3,2,1,1],
            }
        )
        data_7 = pd.DataFrame(
            {
                "timestamp": timestamps * 2
                + [pd.Timestamp("2019-09-01 22:39:19.698000")]
                + [pd.Timestamp("2019-09-01 22:39:12.698000")],
                "parameter_name": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                "value": [2,1,3,2,1,2,3,4,1,2,4,2],
            }
        )

        self.cycle_1 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_1,
            "id1",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_2 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_2,
            "id2",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_3 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_3,
            "id3",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_4 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_4,
            "id4",
            continuous=["a", "b"],
            binary=[],
        )
        self.cycle_5 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_5,
            "id5",
            continuous=["a", "b", "c"],
            binary=[],
        )
        self.cycle_6 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_6,
            "id6",
            continuous=["a", "b", "c"],
            binary=[],
        )
        self.cycle_7 = RawCycle(
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            pd.Timestamp("2019-09-01 22:39:19.161000"),
            "machineID",
            "variant1",
            data_7,
            "id7",
            continuous=["a", "b", "c"],
            binary=[],
        ) 
    def test_fit_no_input(self):
        # construct input data
        no_cycles = []
        ref = FeatureGeneratorRefCycle()
        with self.assertRaises(Exception) as context:
            ref.fit(no_cycles)
        self.assertTrue("No Cycles passed into fit()" in str(context.exception))

    def test_fit_sigle_input(self):
        # construct input data
        cycles = [self.cycle_1]
        ref = FeatureGeneratorRefCycle()
        ref.fit(cycles)
        observed = ref.reference_object
        expected = cycles[0].data
        assert_frame_equal(observed, expected)

    def test_fit(self):
        # select FeatureGenerator
        ref = FeatureGeneratorRefCycle()
        # construct input data
        cycles = [self.cycle_1, self.cycle_2, self.cycle_3, self.cycle_4]
        # calculate data with fucntion and store result in observed
        ref.fit(cycles)
        observed = ref.reference_object
        # define expected outcome
        expected = cycles[3].data
        assert_frame_equal(observed, expected)
 
    def test_fit_same_distance(self):
        #when there is the same overall-distance. the first is taken
        ref = FeatureGeneratorRefCycle()
        cycles = [self.cycle_1, self.cycle_2, self.cycle_3]
        ref.fit(cycles)
        observed = ref.reference_object
        expected = cycles[0].data
        assert_frame_equal(observed, expected)

    def test_three_parameters(self):
        #test with additional parameter ["a","b","c"]
        ref = FeatureGeneratorRefCycle()
        cycles = [self.cycle_5, self.cycle_6, self.cycle_7]
        ref.fit(cycles)
        observed = ref.reference_object
        expected = cycles[2].data
        assert_frame_equal(observed, expected)

    def test_score(self):
        # construct input data
        input_cycle = self.cycle_4
        cycles = [self.cycle_1, self.cycle_2]
        ref = FeatureGeneratorRefCycle()
        random.seed(900)
        ref.fit(cycles)
        input_cycle.data["c_id"] = input_cycle.id
        # calculate data with fucntion -> observed
        observed_prediction, observed_alignment, observed_score = ref.predict(
            input_cycle.data
        )
        # define expected outcome
        expected_prediction = pd.DataFrame(
            {"feature_name": ["a", "b"], "feature_value": [0.9, 0.83333]}
        )
        expected_alignment = pd.DataFrame(
            {
                "n_interval": [1, 2, 3, 4, 1, 2],
                "c_id": [input_cycle.id] * 6,
                "p_name": ["a"] * 4 + ["b"] * 2,
                "alignment_offset": [1, 2, 1, 3, 1, 3],
                "duration": [2, 1, 1, 1, 4, 1],
            },
            index=[0, 1, 2, 3, 0, 1],
        )
        expected_score = 0.86666

        assert_frame_equal(observed_prediction, expected_prediction, check_exact=True)
        assert_frame_equal(observed_alignment, expected_alignment, check_dtype=False)
        assert observed_score == expected_score

    def test_calculate_cycle_score(self):
        ref = FeatureGeneratorRefCycle()
        similarity_df = pd.DataFrame(
            {"feature_name": ["a", "b"], "feature_value": [0.9, 0.83333]}
        )
        # calculate value
        observed = ref.calculate_cycle_score(similarity_df)
        # expected
        expected = 0.86666
        assert observed == expected

if __name__ == "__main__":
    unittest.main()