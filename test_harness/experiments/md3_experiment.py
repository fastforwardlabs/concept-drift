import time
import logging

import numpy as np
from scipy.stats import ks_2samp, describe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from test_harness.experiments.baseline_experiment import BaselineExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class MD3_Experiment(BaselineExperiment):
    def __init__(self, model, dataset, param_grid=None, sensitivity=1):
        '''
         :param dataset
         :param model: initially trained classifier
         :param md_ref: reference margin density
         :param md_sigma: margin density stddev
         :param acc_ref: reference accuracy
         :param acc_sigma: reference accuracy stddev
         :param sensitivity: sensitivity
         '''
        super().__init__(model, dataset, param_grid)
        self.name = "md3-svm"
        self.sensitivity = sensitivity

        self.lambda_ = (dataset.window_size-1)/dataset.window_size
        self.currently_drifting = False
        self.labeled_samples = 0
        self.total_labeled_samples = dataset.window_size

        self.drifted_indices = []
        self.md_ref = None
        self.md_sigma = None
        self.acc_ref = None
        self.acc_sigma = None

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

        # gather training data
        window_idx = (
            self.reference_window_idx
            if window == "reference"
            else self.detection_window_idx
        )
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        # fit model
        logger.info(f"Trained Model at Index: {window_idx}")
        self.trained_model = self.model.fit(X_train, y_train)

        start_time = time.time()
        '''
        param_range = [0.001, 0.01, 0.1, 1.0, 10.0]
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
        '''

        end_time = time.time()

        train_time = end_time - start_time

        # train evaluation
        eval_score, md_ref, md_sigma, acc_ref, acc_sigma = self.evaluate_model_aggregate(window=window)
        if window == "reference":
            self.md_ref = md_ref
            self.md_sigma = md_sigma
            self.acc_ref = acc_ref
            self.acc_sigma = acc_sigma

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

    def get_reference_response_distribution(self):

        # get data in reference window
        window_idx = self.reference_window_idx
        print(f"GETTING REFERENCE DISTRIBUTION FOR WINDOW: {window_idx}")
        X_train, y_train = self.dataset.get_window_data(window_idx, split_labels=True)

        # perform kfoldsplits to get predictions
        #preds = self.make_kfold_predictions(X_train, y_train, self.model, self.k)

        return preds

    def get_detection_response_distribution(self):

        # get data in prediction window
        window_idx = self.detection_window_idx
        print(f"GETTING DETECTION DISTRIBUTION FOR WINDOW: {window_idx}")
        X_test, y_test = self.dataset.get_window_data(window_idx, split_labels=True)
        X_test_array = X_test.values
        print(X_test_array[:5])
        # use trained model to get response distribution
        preds = self.trained_model.decision_function(X_test_array) #.reshape(1, -1)[:,1]
        print(preds[:5])
        #preds = self.trained_model.predict_proba(X_test)[:, 1]

        return preds

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

        #self.train_model_gscv(window="reference", gscv=True)

        CALC_REF_RESPONSE = False

        for i, split in enumerate(self.dataset.splits):
            MD = self.md_ref
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
                preds = self.get_detection_response_distribution()

                margin_inclusion_signal = np.where(abs(preds) <= 1, 1, 0) #1 if abs(preds) <= 1 else 0
                MD = self.lambda_ * MD + (1 - self.lambda_) * margin_inclusion_signal

                if abs(MD - self.md_ref) > self.sensitivity * self.md_sigma:
                    # drift suspected
                    self.train_model(window="detection")
                    self.update_reference_window()
                    CALC_REF_RESPONSE = True
                else:
                    CALC_REF_RESPONSE = False
                self.update_detection_window()

        self.calculate_label_expense()
        self.calculate_train_expense()




