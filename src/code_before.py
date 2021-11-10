import tempfile
from typing import List

#import mlflow
import os
import pandas as pd
import random
import numpy as np
from dtw import dtw
from ill_commons.cycle.RawCycle import RawCycle

from ill_feature_generation.Feature_Generator.FeatureGenerator import FeatureGenerator


class FeatureGeneratorRefCycle(FeatureGenerator):
    """
    Generate DTW with a reference Object
    """

    reference_object: pd.DataFrame
    dtw_pattern: str

    def fit(self, train_data: List[RawCycle]) -> None:
        """
        Function to fit a new Reference Object that can depend on a single or multiple Cycle
        Input: self, train_data = Raw Cycle Data in long format
        Return: -
        
        calculates all distances (DTW) to every cycle from every cycle 
        and chooses the cycle with the lowest distance to every other cycle.
        It is stored in self.reference.object
        """
        number_of_provided_cycles = len(train_data)

        # check if there are cycles, if not raise Exception
        if number_of_provided_cycles == 0:
            raise Exception(f"No Cycles passed into fit()")
        # if there is one cycle, take it as reference object
        elif number_of_provided_cycles == 1:
            self.reference_object = train_data[0].data
            return 
            
        # if there are more than one cycle, calculate the best fitting reference cycle
        self.dtw_pattern = "symmetric2"
        feature_distance = np.array([])
        total_distance = np.array([])
        feature_sum = 0
        #calculate the distance of every cycle to every cycle
        for x in range(number_of_provided_cycles):
            #calculate the distance of every cycle
            for y in range(number_of_provided_cycles): 
                #distance of identical cycle is always 0
                if x == y:
                    continue
                #calculate the distance of every feature
                groups = pd.DataFrame(train_data[x].data).groupby("parameter_name").groups
                distance = 0
                for group in groups: 
                    query_one = train_data[x].data[train_data[x].data['parameter_name'] == group]["value"].values
                    query_two = train_data[y].data[train_data[y].data['parameter_name'] == group]["value"].values
                    #distance means the distance between two given features/parameter of a cycle, calculated by the dtw
                    distance += dtw(query_one, query_two, step_pattern=self.dtw_pattern).distance 
                #feature_distance stands for the distances between one cycle to every other cycle 
                feature_distance = np.append(feature_distance, distance)
            feature_sum = feature_distance.sum()
            # total_distance stores the summed distance of every cycle to every other cycle in one value
            total_distance = np.append(total_distance, feature_sum)
            feature_distance = np.array([])
            feature_sum = 0
        #choose the cycle with the lowest distance to every other cycle
        minimum = np.argmin(total_distance)
        self.reference_object = train_data[minimum].data

    def calculate_alignment(
        self, stepstaken: [], parameter: str, c_id: str
    ) -> pd.DataFrame:
        # get positions of changes (separating the intervals)
        changes = np.where(stepstaken[:-1] != stepstaken[1:])[0]
        # get scores at the different intervals
        scores = np.append(stepstaken[changes], stepstaken[len(stepstaken) - 1])

        # get number of values in each interval
        if len(changes) == 0:
            duration = np.array([len(stepstaken)])
        else:
            add_first = np.append(np.array([changes[0] + 1]), np.diff(changes))
            duration = np.append(add_first, np.array([len(stepstaken) - changes[-1] - 1]))

        n_intervals = range(1, len(changes) + 2)

        intervals_df = pd.DataFrame(n_intervals, columns=["n_interval"])
        intervals_df["c_id"] = c_id
        intervals_df["p_name"] = parameter
        intervals_df["alignment_offset"] = scores
        intervals_df["duration"] = duration
        return intervals_df

    def predict(self, cycle_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, float):
        """
        Function to generate Features for a given cycle (the input is the raw cycle data).
        Gives back a data frame with features. To be flexible this frame is in long format.
        """
        raw_data = cycle_data
        cycle_id = cycle_data["c_id"].iloc[0]
        parameters = raw_data["parameter_name"].unique()
        similarity_df = pd.DataFrame(parameters, columns=["feature_name"])
        similarity_df["feature_value"] = 0.0
        alignment_dfs = []
        # sortieren mit timestamp
        for param in parameters:
            # param = row[index]['parameter_name']
            # if index not in cycle_list[i].columns:
            #  print('Column: ' + str(index) + 'not found')
            #  continue
            query = np.array(raw_data.loc[raw_data["parameter_name"] == param]["value"])
            query = query[~np.isnan(query.astype(float))]

            ref = np.array(
                self.reference_object.loc[
                    self.reference_object["parameter_name"] == param
                ]["value"]
            )
            ref = ref[~np.isnan(ref.astype(float))]

            alignment = dtw(
                query, ref, step_pattern=self.dtw_pattern
            )
            alignment_dfs.append(
                self.calculate_alignment(
                    alignment.stepsTaken, parameter=param, c_id=cycle_id
                )
            )

            m_x = max(abs(ref)) * len(ref)
            if m_x == 0:
                if alignment.distance == 0:
                    similarity_df.loc[
                        similarity_df["feature_name"] == param, "feature_value"
                    ] = 1.0
                    # similarity_df.at[index, feature_name] = 1.0
                else:
                    m_x = max(abs(query)) * len(query)
                    similarity_df.loc[
                        similarity_df["feature_name"] == param, "feature_value"
                    ] = np.round((m_x - alignment.distance) / m_x, decimals=5)
            else:
                similarity_df.loc[
                    similarity_df["feature_name"] == param, "feature_value"
                ] = np.round((m_x - alignment.distance) / m_x, decimals=5)
        alignment_df = pd.concat(alignment_dfs)
        score = self.calculate_cycle_score(similarity_df)
        return similarity_df, alignment_df, score

    def calculate_cycle_score(self, similarity_df: pd.DataFrame) -> float:
        """
        Function to aggregate the Features to one score per cycle. This is meant to serve as traffic light colour in the overview table.
        """
        return np.round(similarity_df["feature_value"].mean(), decimals=5)

    def log_to_mlflow(self):
        """
        Function to save data specific to the model to mlflow.
        :return:
        """
        tempdir = tempfile.mkdtemp()
        self.reference_object.to_parquet(os.path.join(tempdir, "reference_data.snappy.parquet"),
                                         compression="snappy")
        mlflow.log_artifact(os.path.join(tempdir, "reference_data.snappy.parquet"))
