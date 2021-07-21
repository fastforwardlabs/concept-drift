import logging

from test_harness.experiments.baseline_experiment import BaselineExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("../logs/app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


class ToplineExperiment(BaselineExperiment):
    def __init__(self, model, dataset, param_grid=None):
        super().__init__(model, dataset, param_grid)
        self.name = "topline"

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

                self.update_reference_window()
                self.update_detection_window()

                self.train_model_gscv(window="reference", gscv=True)

        self.calculate_label_expense()
        self.calculate_train_expense()
