import logging
from time import perf_counter

import numpy as np
from scipy.stats import ks_2samp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from test_harness.experiments.baseline_experiment import BaselineExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class SQSI_MRExperiment(BaselineExperiment):
    def __init__(self, model, dataset, k, significance_thresh):
        super().__init__(model, dataset)
        self.name = "sqsi-mr"
        self.k = k
        self.significance_thresh = significance_thresh

    @staticmethod
    def make_kfold_predictions(X, y, k, model):
        """A KFold version of LeaveOneOut predictions.

        Rather than performing exhaustive leave-one-out methodology to get predictions
        for each observation, we use a less exhaustive KFold approach.

        When k == len(X), this is equivalent to LeaveOneOut: expensive, but robust. Reducing k
        saves computation, but reduces robustness of model.

        Args:
            X (pd.Dataframe) - features in evaluation window
            y (pd.Series) - labels in evaluation window
            k (int) - number of folds

        Returns:
            preds (np.array) - an array of predictions for each X in the input (NOT IN ORDER OF INPUT)

        """
        # NOTE - need to think through if this should be a pipeline with MinMaxScaler...???

        skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

        preds = np.array([])
        for train_indicies, test_indicies in skf.split(X, y):

            # fit it
            model.fit(X.iloc[train_indicies], y.iloc[train_indicies])

            # score it on this Kfold's test data
            y_preds_split = model.predict_proba(X.iloc[test_indicies])
            y_preds_split_posclass_proba = y_preds_split[:, 1]

            preds = np.append(preds, y_preds_split_posclass_proba)

        return preds

    def get_reference_response_distribution(self):

        # get data in reference window
        window_idx = self.reference_window_idx
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        # perform kfoldsplits to get predictions
        preds = self.make_kfold_predictions(X_train, y_train, self.k, self.model)

        return preds

    def get_detection_response_distribution(self):

        # get data in prediction window
        window_idx = self.detection_window_idx
        X_test, y_test = self.dataset.get_window_data(window_idx, split_labels=True)

        # use trained model to get response distribution
        preds = self.trained_model.predict_proba(X_test)[:, 1]

        return preds

    @staticmethod
    def perform_ks_test(dist1, dist2):
        return ks_2samp(dist1, dist2)

    def run(self):
        """SQSI Model Replacement Experiment

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
        logger.info(f"Started SQSI Model Replacement Run")
        self.train_model(window="reference")

        CALC_REF_RESPONSE = True

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

                # compare distributions
                ks_result = self.perform_ks_test(
                    dist1=ref_response_dist, dist2=det_response_dist
                )

                logger.info(f"KS Test: {ks_result}")

                if ks_result[1] < self.significance_thresh:
                    # reject null hyp, distributions are NOT identical
                    self.train_model(window="reference")
                    self.update_reference_window()
                    CALC_REF_RESPONSE = True
                else:
                    CALC_REF_RESPONSE = False

                self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()
