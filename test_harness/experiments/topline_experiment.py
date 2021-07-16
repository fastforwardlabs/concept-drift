from test_harness.experiments.baseline_experiment import BaselineExperiment


class ToplineExperiment(BaselineExperiment):
    def __init__(self, model, dataset):
        super().__init__(model, dataset)

    def run_topline(self):
        """The Topline Experiment retrains a model on each incremental reference window.

        This serves as the most greedy possible scenario and should incur high label cost.
            - Train on initial window
            - Evaluate on detection window
            - Update reference window and retrain
            - Repeat until finished

        """
        self.train_model(window="reference")

        for i, split in enumerate(self.dataset.splits):

            if i > self.reference_window_idx:

                self.experiment_metrics["scores"].extend(
                    self.evaluate_model_incremental(n=10)
                )

                self.update_reference_window()
                self.update_detection_window()

                self.train_model(window="reference")

        self.calculate_label_expense()
        self.calculate_train_expense()
