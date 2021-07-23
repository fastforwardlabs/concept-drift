import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def format_experimental_scores(experiment):
    return (
        pd.DataFrame(experiment.experiment_metrics["scores"])
        .set_index(0)
        .rename(columns={1: experiment.name})
    )


def plot_experiment_error(experiment, retrain_points=True):
    scores_df = format_experimental_scores(experiment)
    ax = scores_df.plot(
        figsize=(15, 7),
        title=f"Cumulative Score over Experiment \n \
            Label Expense: {experiment.experiment_metrics['label_expense']['num_labels_requested']} \n \
            Total Train Time: {experiment.experiment_metrics['total_train_time']}",
    )

    retrainings = [
        experiment.dataset.get_split_idx(window_idx)
        for window_idx in [
            train_record["window_idx"]
            for train_record in experiment.experiment_metrics["training"]
        ]
    ]

    if retrain_points:
        [
            ax.axvline(i, color="black", linestyle=":", linewidth=0.75)
            for i in retrainings
            if i != 0
        ]

    plt.show()


def plot_multiple_experiments(experiments, change_points=None):

    exp_dfs = [format_experimental_scores(experiment) for experiment in experiments]

    ax = pd.concat(exp_dfs, axis=1).plot(
        figsize=(15, 7), title="Overall Score by Experiment"
    )

    if change_points:
        [
            ax.axvline(i, color="black", linestyle=":", linewidth=0.75)
            for i in change_points
            if i != 0
        ]
    plt.show()


def aggregate_experiment_metrics(experiments):

    metrics = []
    for exp in experiments:

        metrics.append(
            {
                "experiment": exp.name,
                "times_retrained": len(exp.experiment_metrics["training"]),
                "percent_total_labels": exp.experiment_metrics["label_expense"][
                    "percent_total_labels"
                ],
                "total_train_time": exp.experiment_metrics["total_train_time"],
            }
        )

    return pd.DataFrame(metrics).set_index("experiment")


def plot_response_distributions_bysplit(sqsi_exp):

    df = pd.DataFrame()

    for i in range(len(sqsi_exp.ref_distributions)):
        dists = pd.DataFrame(
            np.stack([sqsi_exp.ref_distributions[i], sqsi_exp.det_distributions[i]]).T,
            columns=["Reference", "Detection"],
        )
        dists["Split"] = i

        df = df.append(dists)

    df_melt = df.melt(id_vars=["Split"], var_name="Window Type")

    g = sns.FacetGrid(df_melt, col="Split", hue="Window Type", col_wrap=4)
    g.map_dataframe(sns.kdeplot, "value", fill=True)
    g.add_legend()


def calculate_distances_window_distances(experiment, distance_func):

    splits = []

    for i in range(len(experiment.ref_distributions)):

        ref_dist = experiment.ref_distributions[i]
        det_dist = experiment.det_distributions[i]

        distance = distance_func(ref_dist, det_dist)

        splits.append((i, distance))

    return pd.DataFrame(splits, columns=["Split", "Distance"])
