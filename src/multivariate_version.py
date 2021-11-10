import tempfile
from logging import Logger
from typing import Callable, Dict, List, Tuple

import mlflow
import os
import pandas as pd
import numpy as np
from operator import itemgetter, attrgetter
from dtw import dtw, DTW
from ill_commons.cycle.RawCycle import RawCycle

from ill_feature_generation.Feature_Generator.FeatureGenerator import FeatureGenerator

from itertools import permutations

from ill_feature_generation.Feature_Generator.utils import calculate_similarity
from ill_feature_generation.transformation.prepare_parameters import (
    apply_transformations,
    load_configuration_file,
)

from ill_feature_generation.transformation.scaling import Scaler


class FeatureGeneratorRefCycle(FeatureGenerator):
    """
    Generate DTW with a reference Object
    """

    reference_object: pd.DataFrame
    dtw_pattern: str
    scaler: Scaler
    config: Dict
    machine_id: str
    multivariate: bool

    def __init__(self, params: Dict[str, str] = {}):
        self.dtw_pattern = (
            params["dtw_pattern"] if "dtw_pattern" in params.keys() else "symmetric1"
        )
        self.multivariate = (
            params["multivariate"] if "multivariate" in params.keys() else False
        )

    def fit(
        self,
        train_data: List[RawCycle],
        machine_id: str,
        config: Dict,
        logger: Logger,
        custom_similarity_function: Callable[
            [List[RawCycle], FeatureGenerator], pd.DataFrame
        ] = None,
    ) -> None:
        """
        Function to fit a new Reference Object that can depend on a single or multiple Cycle
        Input: self, train_data = Raw Cycle Data in long format
        Return: -

        calculates all distances (DTW) to every cycle from every cycle
        and chooses the cycle with the lowest distance to every other cycle.
        It is stored in self.reference.object
        """
        self.config = config
        self.machine_id = machine_id
        number_of_provided_cycles = len(train_data)
        # check if there are cycles, if not raise Exception
        if number_of_provided_cycles == 0:
            raise Exception(f"No Cycles passed into fit()")

        if number_of_provided_cycles == 1:
            self.ref_cycle_id = train_data[0].id
            self.scaler = Scaler().fit(data=[train_data[0].data], logger=logger)
            self.reference_object = train_data[0].data.to_pandas()
            return

        self.scaler = Scaler().fit(data=[x.data for x in train_data], logger=logger)

        if custom_similarity_function is None:
            similarities = self.default_similarity_function(train_data, logger)
        else:
            similarities = custom_similarity_function(train_data, self)

        logger.debug(similarities)

        # df = result of similarity between all cycle combinations
        # Sum the similarity for every cycle to every other cycle
        # get the cycle with the highest similarity
        x = similarities.groupby(["queryid"]).sum().reset_index()
        max_cycle = x[x["similarity"] == x["similarity"].max()].iloc[0, :]
        logger.info("Sum of similarities for every cycle: {}".format(x))
        logger.info(max_cycle)
        logger.info(f"Using Cycle {max_cycle['queryid']} as ref Cycle")
        self.ref_cycle_id = max_cycle["queryid"]
        ref_cycle = [x for x in train_data if x.id == self.ref_cycle_id][0]

        self.reference_object = ref_cycle.data.to_pandas()

    def default_similarity_function(
        self,
        cycles: List[RawCycle],
        logger: Logger,
        combinations: List[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        result = []
        data = {x.id: x for x in cycles}
        if combinations is None:
            combinations = list(permutations([x.id for x in cycles], 2))
        # calculate the distance of every cycle to every cycle
        for x, y in combinations:
            x_cycle = data[x]
            y_cycle = data[y]
            # distance of identical cycle is always 0
            if x == y:
                result.append((x, y, 1))
            else:
                logger.debug(f"Comparing cycles {x} and {y}")
                similarity_df, _ = calculate_similarity(
                    x_cycle.data.to_pandas(),
                    y_cycle.data.to_pandas(),
                    False,
                    x,
                    logger,
                    self.scaler,
                    self.dtw_pattern,
                    self.config,
                    self.multivariate,
                )
                similarity = similarity_df["feature_value"].mean()

                # feature_distance stands for the distances between one cycle to every other cycle
                result.append((x, y, similarity))
        result_df = pd.DataFrame(result, columns=["queryid", "refid", "similarity"])
        return result_df

    def predict(
        self, cycle_data: pd.DataFrame, logger: Logger
    ) -> (pd.DataFrame, pd.DataFrame, float):
        """
        Function to generate Features for a given cycle (the input is the raw cycle data).
        Gives back a data frame with features. To be flexible this frame is in long format.
        """
        cycle_id = cycle_data["c_id"].iloc[0]
        similarity_df, alignment_df = calculate_similarity(
            cycle_data,
            self.reference_object,
            True,
            cycle_id,
            logger,
            self.scaler,
            self.dtw_pattern,
            self.config,
        )
        score = self.calculate_cycle_score(similarity_df)
        return similarity_df, alignment_df, score

    def calculate_cycle_score(self, similarity_df: pd.DataFrame) -> float:
        """
        Function to aggregate the Features to one score per cycle. This is meant to serve as traffic light colour in the overview table.
        """
        return np.round(
            similarity_df.query("feature_value >= 0")["feature_value"].mean(), decimals=5,
        )

    def log_to_mlflow(self):
        """
        Function to save data specific to the model to mlflow.
        :return:
        """
        tempdir = tempfile.mkdtemp()
        self.reference_object.to_parquet(
            os.path.join(tempdir, "reference_data.snappy.parquet"),
            compression="snappy",
            use_deprecated_int96_timestamps=True,
        )
        mlflow.log_artifact(os.path.join(tempdir, "reference_data.snappy.parquet"))
        mlflow.log_param("ref_cycle_id", self.ref_cycle_id)
