from typing import Dict

import pandas as pd


class Dataset:
    def __init__(self, full_df: pd.DataFrame, column_mapping: Dict, window_size: int):

        self.full_df = full_df
        self.column_mapping = column_mapping
        self.window_size = window_size
        self.set_splits()

    def set_splits(self):
        """Use the specified window_size to set an attribute that holds corresponding index splits"""
        idx = self.window_size

        splits = []
        while idx < len(self.full_df):
            splits.append(idx)
            idx += self.window_size

        self.splits = splits

    def get_split_idx(self, window_idx):
        """Given a window_idx from an experiment, lookup the split_idx"""
        return self.splits[window_idx]

    def get_window_data(self, window_idx, split_labels=True):
        """
        Given a window_idx corresponding to a split_idx, return the data up to that
        split value starting from the split_idx - 1 value.

        Args:
            window_idx (int) - index corresponding to the end point of the desired data window
            split_labels (bool) - return features and labels separately vs. as one dataframe

        Returns:
            features (pd.DataFrame)
            labels (pd.Series)

        TO-DO: add test to make sure this function gets the expected window data
        """

        end_idx = self.splits[window_idx]

        if window_idx == 0:
            window_data = self.full_df[:end_idx]
        else:
            start_idx = self.splits[window_idx - 1]
            window_data = self.full_df[start_idx:end_idx]

        if split_labels:
            features, labels = self.split_df(window_data, self.column_mapping["target"])
            return features, labels
        else:
            return window_data

    def get_data_by_idx(self, start_idx, end_idx, split_labels=True):
        """
        Given an index into the full_df, return all records up to that observation.

        Args:
            start_idx (int) - index corresponding to the row in full_df
            end_idx (int) - index corresponding to the row in full_df
            split_labels (bool) - return features and labels separately vs. as one dataframe

        Returns:
            features (pd.DataFrame)
            labels (pd.Series)

        TO-DO: should this skip over the first full window that was trained on.. meaning eval data only?

        """

        window_data = self.full_df[start_idx:end_idx]

        if split_labels:
            features, labels = self.split_df(window_data, self.column_mapping["target"])
            return features, labels
        else:
            return window_data

    @staticmethod
    def split_df(df, label_col):
        """Splits the features from labels in a dataframe, returns both"""
        return df.drop(label_col, axis=1), df[label_col]
