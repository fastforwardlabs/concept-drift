import time
import logging
from collections import defaultdict

import pandas as pd
from river import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV

from test_harness.experiments.base_experiment import Experiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class BaselineExperiment(Experiment):
    def __init__(self, model, dataset, param_grid=None):

        self.name = "baseline"
        self.dataset = dataset
        self.model = model
        self.trained_model = None
        self.reference_window_idx = 0
        self.detection_window_idx = 1
        self.experiment_metrics = defaultdict(list)
        self.incremental_metric = metrics.Accuracy()
        self.metric = "accuracy"
        self.param_grid = param_grid

    def update_reference_window(self, split_idx=None):
        """Increments reference window by 1 index, unless split_idx is provided,
        in which case that index becomes the reference window index."""
        self.reference_window_idx += 1 if not split_idx else split_idx

    def update_detection_window(self, split_idx=None):
        """Increments detection window by 1 index, unless split_idx is provided,
        in which case that index becomes the detection window index."""
        self.detection_window_idx += 1 if not split_idx else split_idx

    def train_model(self, window="reference"):
        """Trains model on specified window and updates 'trained_model' attribute."""

        # instantiate training pipeline
        pipe = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("clf", self.model),
            ]
        )

        # gather training data
        window_idx = (
            self.reference_window_idx
            if window == "reference"
            else self.detection_window_idx
        )
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        # fit model
        logger.info(f"Trained Model at Index: {window_idx}")
        start_time = time.time()
        self.trained_model = pipe.fit(X_train, y_train)
        end_time = time.time()

        train_time = end_time - start_time

        # train evaluation
        eval_score = self.evaluate_model_aggregate(window=window)

        # save metrics
        metrics = {
            "window_idx": window_idx,
            "num_train_examples": len(y_train),
            "train_time": train_time,
            "eval_score": eval_score,
        }
        self.experiment_metrics["training"].append(metrics)

    def train_model_gscv(self, window="reference", gscv=False):
        """Trains model on specified window and updates 'trained_model' attribute."""

        # gather training data
        window_idx = (
            self.reference_window_idx
            if window == "reference"
            else self.detection_window_idx
        )
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        # create column transformer
        column_transformer = ColumnTransformer(
            [
                (
                    "continuous",
                    StandardScaler(),
                    self.dataset.column_mapping["numerical_features"],
                ),
                (
                    "categorical",
                    "passthrough",
                    self.dataset.column_mapping["categorical_features"],
                ),
            ]
        )

        # instantiate training pipeline
        pipe = Pipeline(
            steps=[
                ("scaler", column_transformer),
                ("clf", self.model),
            ]
        )

        # to help ensure we don't overfit, we perform GridsearchCV eachtime a new
        # model is fit on the reference window since this is how it would be done in prod
        # vs. blindly fitting with fixed hyperparameters
        if gscv:
            if self.param_grid is None:
                raise AttributeError("Training with GSCV, but no param_grid provided.")

            gs = GridSearchCV(
                estimator=pipe,
                param_grid=self.param_grid,
                cv=5,
                scoring=self.metric,
                n_jobs=-1,
                refit=True,
                return_train_score=True,
            )

            gs.fit(X_train, y_train)

            self.trained_model = gs.best_estimator_
            train_time = gs.refit_time_
            eval_score = gs.cv_results_["mean_train_score"][gs.best_index_]
            gscv_test_score = gs.best_score_
            best_params = gs.best_params_

        else:

            # fit model
            start_time = time.time()
            self.trained_model = pipe.fit(X_train, y_train)
            end_time = time.time()
            train_time = end_time - start_time

            # train evaluation
            eval_score = self.evaluate_model_aggregate(window=window)
            gscv_test_score = None
            best_params = None

        logger.info(f"Trained Model at Index: {window_idx} | GridsearchCV: {gscv}")
        logger.info(f"GSCV Best Params: {best_params}")
        logger.info(f"Train Score: {eval_score} | GSCV Test Score: {gscv_test_score}")

        # save metrics
        metrics = {
            "window_idx": window_idx,
            "num_train_examples": len(y_train),
            "train_time": train_time,
            "eval_score": eval_score,
            "gscv_test_score": gscv_test_score,
        }
        self.experiment_metrics["training"].append(metrics)

    def evaluate_model_aggregate(self, window="detection"):
        """
        Evaluates the saved model on all data in the specified window

        Args:
            window (str) - specifies full window to evaluate on (detection/reference)

        Returns:
            test_accuracy (float) - single metric describing aggregate score on window

        """

        # gather evaluation data
        window_idx = (
            self.reference_window_idx
            if window == "reference"
            else self.detection_window_idx
        )

        X_test, y_test = self.dataset.get_window_data(window_idx, split_labels=True)

        test_accuracy = self.trained_model.score(X_test, y_test)

        logger.info(
            f"Evaluated Model in Aggregate on {window} Window: {window_idx} | Score: {test_accuracy}"
        )

        return test_accuracy

    def evaluate_model_incremental(self, n):
        """
        Evaluates the saved model in a incremental/cumulative manner by incrementally scoring the model
        every k observations in the 'detection' window. K is dermined by specifying the total number
        of equally spaced evaluation points (n) desired within the window.

        Args:
            n (int) - number of splits within a window to log scores for

        Returns:
            test_accuracy (list) - list of tuples specifying index and cumulative score
        """

        step = int(self.dataset.window_size / n)

        window_start = self.dataset.get_split_idx(
            window_idx=(self.detection_window_idx - 1)
        )

        logger.info(f"Reference Evaluation at Window Start: {window_start}")

        test_scores = []
        window_end = window_start + step  # get window end

        for i in range(n):
            X_test, y_test = self.dataset.get_data_by_idx(
                window_start, window_end, split_labels=True
            )

            y_pred = self.trained_model.predict(X_test)

            for yt, yp in zip(y_test, y_pred):
                self.incremental_metric.update(yt, yp)

            inc_score = round(self.incremental_metric.get(), 4)

            test_scores.append((window_end, inc_score))

            logger.info(
                f"Evaluated Model at Window Increment: {window_start} - {window_end} | {i+1}/{n} | {inc_score}"
            )

            window_start += step
            window_end += step

        return test_scores

    def calculate_label_expense(self):
        """A postprocessing step to aggregate and save label expense metrics"""

        logger.info(f"Calculating Label Expense")

        num_labels_requested = sum(
            [
                train_run["num_train_examples"]
                for train_run in self.experiment_metrics["training"]
            ]
        )
        percent_total_labels = round(
            num_labels_requested / len(self.dataset.full_df), 4
        )

        label_metrics = {
            "num_labels_requested": num_labels_requested,
            "percent_total_labels": percent_total_labels,
        }

        self.experiment_metrics["label_expense"] = label_metrics

        logger.info(f"Label Metrics: {label_metrics}")

    def calculate_train_expense(self):
        """A postprocessing step to aggregate and save training expense metrics"""

        logger.info(f"Calculating Train Expense")

        total_train_time = round(
            sum(
                [
                    train_run["train_time"]
                    for train_run in self.experiment_metrics["training"]
                ]
            ),
            2,
        )

        self.experiment_metrics["total_train_time"] = total_train_time

        logger.info(f"Train Metrics: {total_train_time}")

    def run(self):
        """The Baseline Experiment simply trains a model on the initial reference window
        and then evaluates on each incremental detection window with NO retraining.

        This serves as the least accurate scenario and should incur minimal label cost at the expense
        of accuracy.
            - Train on initial window
            - Evaluate on detection window
            - Update detection window
            - Repeat until finished

        """
        logger.info(f"-------------------- Started Baseline Run --------------------")
        self.train_model_gscv(window="reference", gscv=True)

        for i, split in enumerate(self.dataset.splits):

            if i > self.reference_window_idx:

                self.experiment_metrics["scores"].extend(
                    self.evaluate_model_incremental(n=10)
                )
                self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
