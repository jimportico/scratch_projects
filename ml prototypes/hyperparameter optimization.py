"""
Creating a script to test different approaches to optimize hyperparameters
    video: https://www.youtube.com/watch?v=5nYqK-HaoKY
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline
import sklearn
from skopt.space.space import Categorical

# List directory with data
# This dataset comes from Kaggle
# https://www.kaggle.com/iabhishekofficial/mobile-price-classification
# ---------------------------------------------------------------------
data_dir = "C:\\data\\kaggle\\mobile_price\\"
files = os.listdir(data_dir)

train_df = pd.read_csv(data_dir + files[1])
test_df = pd.read_csv(data_dir + files[0])

# List features & target
# ----------------------
X = train_df.drop("price_range", axis=1).values
y = train_df["price_range"].values


# Construct a pipeline
# --------------------
scl = preprocessing.StandardScaler()
pca = decomposition.PCA()
rf_clf = ensemble.RandomForestClassifier(n_jobs=-1)
classifier = pipeline.Pipeline([("scaling", scl), ("pca", pca), ("rf", rf_clf)])


# Gridsearch
# ----------
param_grid = {
    "pca__n_components": np.arange(5, 10),
    "rf__n_estimators": [100, 200, 300, 400],
    "rf__max_depth": [1, 3, 5, 7],
    "rf__criterion": ["gini", "entropy"],
}

grid_model = model_selection.GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    scoring="accuracy",
    verbose=10,
    n_jobs=1,
    cv=5,
)

grid_model.fit(X, y)
print(grid_model.best_score_)
print(grid_model.best_estimator_.get_params())

# Random search
# -------------
random_grid = {
    "pca__n_components": np.arange(5, 10),
    "rf__n_estimators": np.arange(100, 1500, 100),
    "rf__max_depth": np.arange(1, 20),
    "rf__criterion": ["gini", "entropy"],
}

random_model = model_selection.RandomizedSearchCV(
    estimator=classifier,
    param_distributions=random_grid,
    n_iter=10,
    scoring="accuracy",
    verbose=10,
    n_jobs=1,
    cv=5,
)

random_model.fit(X, y)
print(random_model.best_score_)
print(random_model.best_estimator_.get_params())


# Bayesian Search - skopt
# -----------------------
from functools import partial
from skopt import space
from skopt import gp_minimize

# create a function
def optimize(params, param_names, x, y):
    """Optimization function for bayesian search"""
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(x, y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1 * np.mean(accuracies)


param_space = [
    space.Integer(3, 15, name="max_depth"),
    space.Integer(100, 600, name="n_estimators"),
    space.Categorical(["gini", "entropy"], name="criterion"),
    space.Real(0.01, 1, prior="uniform", name="max_features"),
]

param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

result = gp_minimize(
    optimization_function,
    dimensions=param_space,
    n_calls=15,
    n_random_starts=10,
    verbose=10,
)

print(dict(zip(param_names, result.x)))

# Hyperopt
# --------
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll import scope

# create a function
def hyp_optimize(params, x, y):
    """Optimization function for bayesian search"""
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for idx in kf.split(x, y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1 * np.mean(accuracies)


hyp_param_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "max_features": hp.uniform("max_features", 0.01, 1),
}

hyp_optimization_function = partial(hyp_optimize, x=X, y=y)

trials = Trials()

result = fmin(
    fn=hyp_optimization_function,
    space=hyp_param_space,
    max_evals=15,
    trials=trials,
    algo=tpe.suggest,
)


print(result)
