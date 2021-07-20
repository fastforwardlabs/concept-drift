import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from test_harness.utils.utils import plot_experiment_error, plot_multiple_experiments

from test_harness.datasets.dataset import Dataset
from test_harness.experiments.baseline_experiment import BaselineExperiment
from test_harness.experiments.topline_experiment import ToplineExperiment
from test_harness.experiments.sqsi_experiment import SQSI_MRExperiment
from test_harness.experiments.md3_experiment import MD3_Experiment

data = pd.read_csv("../../data/electricity-normalized.csv")

cols = (
    ("day", True),
    ("period", True),
    ("nswdemand", False),
    ("vicdemand", False),
    ("transfer", False),
    ("class", True),
)

data_clean = data[[col for col, _ in cols]].copy(deep=True)

# label encode categorical
catcols = (col for col, iscat in cols if iscat)

for col in catcols:
    data_clean[col] = LabelEncoder().fit_transform(data_clean[col])

# convert to categorical
def categorize(df, cols):
    catcols = (col for col, iscat in cols if iscat)
    for col in catcols:
        df[col] = pd.Categorical(df[col])
    return df


data_clean = categorize(data_clean, cols)

print(data_clean.head())





column_mapping = {
    "target": "class",
    "numerical_features": ["nswdemand", "vicdemand"],
    "categorical_features": ["day", "period"],
}

ED_dataset = Dataset(
    full_df=data_clean, column_mapping=column_mapping, window_size=2500
)
'''
model = RandomForestClassifier(n_estimators=10, random_state=42)
baseline = BaselineExperiment(model=model, dataset=ED_dataset)
topline = ToplineExperiment(model=model, dataset=ED_dataset)
# sqsi_mr = SQSI_MRExperiment(model=model, dataset=ED_dataset, k=20, significance_thresh=0.001)
baseline.run()
topline.run()
'''

#plot_experiment_error(baseline)
#plot_experiment_error(topline)
svc = SVC(kernel='linear', C=10.0, random_state=42)
X_train, y_train = ED_dataset.get_window_data(window_idx=0, split_labels=True)
md3 = MD3_Experiment(model=svc, dataset=ED_dataset)
md3.run()