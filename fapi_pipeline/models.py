"""
The ten prespecified classifiers and their Optuna search spaces.

AdaBoost, which had been run in an earlier exploratory version of the code but
was not part of the prespecified model list, is not included.

Deviation from the previous analysis (2026-10-22): both SVMs have a solver iteration cap
(max_iter = SVM_MAX_ITER = 1,000,000; previously unlimited). In the first run on the updated data, one
linear-SVM fit during inner-fold tuning (Exp 1, outer fold 3) did not finish within 2 hours. The search spaces are
unchanged; a fit that reaches the cap stops with a ConvergenceWarning and is evaluated as usual.
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

SVM_MAX_ITER = 1_000_000

MODEL_NAMES = [
    "Logistic Regression", "Linear SVM", "RBF-SVM", "Random Forest", "Extra Trees",
    "Gradient Boosting", "XGBoost", "LightGBM", "KNN", "Gaussian Naive Bayes",
]

TREE_MODELS = {"Random Forest", "Extra Trees", "Gradient Boosting", "XGBoost", "LightGBM"}


def build_model(trial, name: str, rs: int):
    if name == "Logistic Regression":
        return LogisticRegression(C=trial.suggest_float("C", 1e-4, 10.0, log=True),
                                  l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                                  penalty="elasticnet", solver="saga", max_iter=5000, random_state=rs,
                                  class_weight="balanced")
    if name == "Linear SVM":
        return SVC(kernel="linear", C=trial.suggest_float("C", 1e-3, 50.0, log=True),
                   probability=True, random_state=rs, class_weight="balanced", max_iter=SVM_MAX_ITER)
    if name == "RBF-SVM":
        return SVC(kernel="rbf", C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                   gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                   probability=True, random_state=rs, class_weight="balanced", max_iter=SVM_MAX_ITER)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
                                      max_depth=trial.suggest_int("max_depth", 2, 4),
                                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 8),
                                      max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                                      random_state=rs, n_jobs=1, class_weight="balanced")
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 500, step=50),
                                    max_depth=trial.suggest_int("max_depth", 2, 10),
                                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 20),
                                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                                    random_state=rs, n_jobs=1, class_weight="balanced")
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=trial.suggest_int("n_estimators", 25, 100, step=25),
                                          max_depth=trial.suggest_int("max_depth", 1, 3),
                                          learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                                          subsample=trial.suggest_float("subsample", 0.7, 1.0),
                                          random_state=rs)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
                             max_depth=trial.suggest_int("max_depth", 2, 8),
                             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                             subsample=trial.suggest_float("subsample", 0.6, 1.0),
                             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                             reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                             reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                             random_state=rs, eval_metric="logloss", verbosity=0, n_jobs=1)
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
                              max_depth=trial.suggest_int("max_depth", 2, 8),
                              learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                              num_leaves=trial.suggest_int("num_leaves", 8, 64),
                              subsample=trial.suggest_float("subsample", 0.6, 1.0),
                              subsample_freq=1,
                              colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                              reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                              reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                              random_state=rs, verbose=-1, n_jobs=1, class_weight="balanced")
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=trial.suggest_int("n_neighbors", 1, 15, step=2),
                                    weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                                    metric=trial.suggest_categorical("metric", ["euclidean", "manhattan"]))
    if name == "Gaussian Naive Bayes":
        return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True))
    raise ValueError(name)
