import time
import logging

import numpy as np
from scipy.stats import ks_2samp, describe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from test_harness.experiments.baseline_experiment import BaselineExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class MD3_Experiment(BaselineExperiment):
    def __init__(self, model, dataset, sensitivity=1):
        '''
         :param dataset
         :param model: initially trained classifier
         :param md_ref: reference margin density
         :param md_sigma: margin density stddev
         :param acc_ref: reference accuracy
         :param acc_sigma: reference accuracy stddev
         :param sensitivity: sensitivity
         '''
        super().__init__(model, dataset)
        self.name = "md3-svm"
        self.sensitivity = sensitivity

        self.lambda_ = (dataset.window_size-1)/dataset.window_size
        self.currently_drifting = False
        self.labeled_samples = 0
        self.total_labeled_samples = dataset.window_size

        self.drifted_indices = []

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
        md_ref = len(self.trained_model.support_) / len(y_test)
        md_sigma = round((2.5 / 100) * md_ref, 5)
        acc_ref = test_accuracy
        acc_sigma = round((2.5 / 100) * acc_ref, 5)

        logger.info(
            f"Evaluated Model in Aggregate on {window} Window: {window_idx} | Score: {test_accuracy} | MD: {md_ref}"
        )

        return test_accuracy, md_ref, md_sigma, acc_ref, acc_sigma

    def train_model(self, window="reference"):
        """Trains model on specified window and updates 'trained_model' attribute."""

        # instantiate training pipeline
        '''
        pipe = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("clf", self.model),
            ]
        )
        '''

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
        param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        param_grid = [{'C': param_range}]

        gs = GridSearchCV(
            estimator=self.model,
            scoring='accuracy',
            cv=5,
            param_grid=param_grid,
            n_jobs=-1
        )
        gs = gs.fit(X_train, y_train)
        self.trained_model = gs.best_estimator_
        end_time = time.time()

        train_time = end_time - start_time

        # train evaluation
        eval_score, md_ref, md_sigma, acc_ref, acc_sigma = self.evaluate_model_aggregate(window=window)

        # save metrics
        metrics = {
            "window_idx": window_idx,
            "num_train_examples": len(y_train),
            "train_time": train_time,
            "eval_score": eval_score,
            "md_ref": md_ref,
            "md_sigma": md_sigma,
            "acc_ref": acc_ref,
            "acc_sigma": acc_sigma
        }
        self.experiment_metrics["training"].append(metrics)

    def run(self):
        """MD3 Model Replacement Experiment

        This experiment uses a KS test to detect changes in the target/response distribution between
        the reference and detection windows.

        Logic flow:
            - Train on initial reference window
            - Perform Stratified LeaveOneOut/KFold to obtain prediction distribution on reference window
            - Use trained model to generate predictions on detection window
            - Perform statistical test between reference and detection window response distributions
                - If different, retrain and update both windows
                - If from same distribution, update detection window and repeat

        """
        logger.info(f"Started MD3 Model Replacement Run")
        self.train_model(window="reference")

        CALC_REF_RESPONSE = True

        '''
        for i, split in enumerate(self.dataset.splits):

            if i > self.reference_window_idx:

                logger.info(
                    f"Need to calculate Reference response distribution? - {CALC_REF_RESPONSE}"
                )

                # log actual score on detection window
                self.experiment_metrics["scores"].extend(
                    self.evaluate_model_incremental(n=10)
                )

                # get reference window response distribution with kfold + detection response distribution
                if CALC_REF_RESPONSE:
                    ref_response_dist = self.get_reference_response_distribution()
                det_response_dist = self.get_detection_response_distribution()

                logger.info(
                    f"REFERENCE STATS: {describe(ref_response_dist)} | DETECTION STATS: {describe(det_response_dist)}"
                )

                # compare distributions
                ks_result = self.perform_ks_test(
                    dist1=ref_response_dist, dist2=det_response_dist
                )

                logger.info(f"KS Test: {ks_result}")

                if ks_result[1] < self.significance_thresh:
                    # reject null hyp, distributions are NOT identical --> retrain
                    self.train_model(window="detection")
                    self.update_reference_window()
                    CALC_REF_RESPONSE = True
                else:
                    CALC_REF_RESPONSE = False

                self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
        '''



