import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from test_harness.experiments.baseline_experiment import BaselineExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class ToplineExperiment(BaselineExperiment):
    def __init__(self, model, dataset, k, param_grid=None):
        super().__init__(model, dataset, param_grid)
        self.name = "Topline"
        self.k = k

    @staticmethod
    def make_kfold_predictions(X, y, model, dataset, k):
        """A KFold version of LeaveOneOut predictions.

        Rather than performing exhaustive leave-one-out methodology to get predictions
        for each observation, we use a less exhaustive KFold approach.

        When k == len(X), this is equivalent to LeaveOneOut: expensive, but robust. Reducing k
        saves computation, but reduces robustness of model.

        Args:
            X (pd.Dataframe) - features in evaluation window
            y (pd.Series) - labels in evaluation window
            k (int) - number of folds
            type (str) - specified kfold or LeaveOneOut split methodology

        Returns:
            preds (np.array) - an array of predictions for each X in the input (NOT IN ORDER OF INPUT)

        """

        splitter = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

        split_ACCs = np.array([])

        for train_indicies, test_indicies in splitter.split(X, y):

            # create column transformer
            column_transformer = ColumnTransformer(
                [
                    (
                        "continuous",
                        StandardScaler(),
                        dataset.column_mapping["numerical_features"],
                    ),
                    (
                        "categorical",
                        "passthrough",
                        dataset.column_mapping["categorical_features"],
                    ),
                ]
            )

            # instantiate training pipeline
            pipe = Pipeline(
                steps=[
                    ("scaler", column_transformer),
                    ("clf", model),
                ]
            )

            # fit it
            pipe.fit(X.iloc[train_indicies], y.iloc[train_indicies])

            # get accuracy for split
            split_ACC = pipe.score(X.iloc[test_indicies], y.iloc[test_indicies])
            split_ACCs = np.append(split_ACCs, split_ACC)

        return split_ACCs

    def get_reference_response_distribution(self):

        # get data in reference window
        window_idx = self.reference_window_idx
        print(f"GETTING REFERENCE DISTRIBUTION FOR WINDOW: {window_idx}")
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        print(f"SELF MODEL: {self.model}")

        # perform kfoldsplits to get predictions
        split_ACCs = self.make_kfold_predictions(
            X_train, y_train, self.model, self.dataset, self.k
        )

        ref_ACC = np.mean(split_ACCs)
        ref_ACC_SD = np.std(split_ACCs)

        return ref_ACC, ref_ACC_SD

    def calculate_errors(self):

        self.false_positives = [
            True if self.drift_signals[i] and not self.drift_occurences[i] else False
            for i in range(len(self.drift_signals))
        ]
        self.false_negatives = [
            True if not self.drift_signals[i] and self.drift_occurences[i] else False
            for i in range(len(self.drift_signals))
        ]

    def run(self):
        """The Topline Experiment retrains a model on each incremental reference window.

        This serves as the most greedy possible scenario and should incur high label cost.
            - Train on initial window
            - Evaluate on detection window
            - Update reference window and retrain
            - Repeat until finished

        """
        logger.info(f"-------------------- Started Topline Run --------------------")
        self.train_model_gscv(window="reference", gscv=True)

        for i, split in enumerate(self.dataset.splits):

            if i > self.reference_window_idx:

                self.experiment_metrics["scores"].extend(
                    self.evaluate_model_incremental(n=10)
                )

                ref_ACC, ref_ACC_SD = self.get_reference_response_distribution()
                det_ACC = self.evaluate_model_aggregate(window="detection")

                # compare accuracies to see if detection was false alarm
                delta_ACC = np.absolute(det_ACC - ref_ACC)
                threshold_ACC = 3 * ref_ACC_SD  # considering outside 3 SD significant
                significant_ACC_change = True if delta_ACC > threshold_ACC else False
                self.drift_occurences.append(significant_ACC_change)

                self.drift_signals.append(True)  # every iteration is a retrain

                self.update_reference_window()
                self.update_detection_window()

                self.train_model_gscv(window="reference", gscv=True)

        self.calculate_label_expense()
        self.calculate_train_expense()
        self.calculate_errors()
