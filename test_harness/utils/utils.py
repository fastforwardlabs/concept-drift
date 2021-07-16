import pandas as pd


def format_experimental_scores(experiment):
    return (
        pd.DataFrame(experiment.experiment_metrics["scores"])
        .set_index(0)
        .rename(columns={1: experiment.name})
    )


def plot_experiment_error(experiment):
    scores_df = format_experimental_scores(experiment)
    return scores_df.plot(
        figsize=(15, 7),
        title=f"Cumulative Score over Experiment \n \
            Label Expense: {experiment.experiment_metrics['label_expense']['num_labels_requested']} \n \
            Total Train Time: {experiment.experiment_metrics['total_train_time']}",
    )


def plot_multiple_experiments(experiments):

    exp_dfs = [format_experimental_scores(experiment) for experiment in experiments]

    return pd.concat(exp_dfs, axis=1).plot(
        figsize=(15, 7), title="Overall Score by Experiment"
    )
