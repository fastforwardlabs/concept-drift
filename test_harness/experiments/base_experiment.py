from abc import ABC, abstractmethod


class Experiment(ABC):
    """Abstract base class to serve as template for other experiments.

    In addition to the abstract methods specified below, each Experiment subclass should maintain
    the following attributes at a minimum:
        experiment_metrics (defaultdict) - collection of experiment metrics including
            including keys for the following items:
                - 'training': List[Dict] where each dict saves details of training job
                - 'scores': List[Tuple] where each tuple is index and evaluation score
                - 'label_expense': Dict with number of labels requested and percent of total
                - 'total_train_time': float aggregate amount of training time

    """

    @abstractmethod
    def update_reference_window(self):
        pass

    @abstractmethod
    def update_detection_window(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def run(self):
        pass
