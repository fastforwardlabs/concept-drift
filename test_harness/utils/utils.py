import pandas as pd


def plot_experiment_error(experiment):
    return (
        pd.DataFrame(experiment.experiment_metrics["scores"])
        .set_index(0)
        .plot(
            figsize=(15, 7),
            title=f"Cumulative Score over Experiment \n \
            Label Expense: {experiment.experiment_metrics['label_expense']['num_labels_requested']} \n \
            Total Train Time: {experiment.experiment_metrics['total_train_time']}",
        )
    )
