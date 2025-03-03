import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
from ast import literal_eval
import numpy as np
import joblib
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import optuna


@dataclass
class DataEntry:
    value: np.ndarray
    label: str


def train_clf(dataset: Dict[str, Dict[str, DataEntry]], clf_params: Dict[str, Any]):
    # train classifier with target params
    estimator = clf_params.pop('estimator')
    if estimator == "svc":
        clf = SVC(**clf_params)
    elif estimator == "mlp":
        # convert str from optuna to expected sklearn value
        clf_params['hidden_layer_sizes'] = literal_eval(clf_params['hidden_layer_sizes'])
        clf_params['solver'] = clf_params.pop('solver_mpl')
        clf = MLPClassifier(**clf_params)
    elif estimator == 'logres':
        clf_params['solver'] = clf_params.pop('solver_logres')
        clf = LogisticRegression(**clf_params)
    else:
        raise ValueError(f'Unknown estimator: {estimator}')

    X_train = np.array([entry.value for entry in dataset['train'].values()])
    y_train = np.array([entry.label for entry in dataset['train'].values()])
    clf.fit(X_train, y_train)
    return clf


def eval_clf(clf, dataset: Dict[str, Dict[str, DataEntry]], class_map: List[str],
             target_split='test') -> Dict[str, Any]:
    raw_preds, y_test_oh, _ = clf_predict(clf, dataset, target_split, class_map, True)
    clf_oh_preds = raw_preds >= raw_preds.max(axis=1).reshape(-1, 1)

    accuracy = metrics.accuracy_score(y_test_oh, clf_oh_preds)
    p, r, fscore, support = metrics.precision_recall_fscore_support(y_test_oh, clf_oh_preds)
    eval_metrics = {
        'accuracy': accuracy,
        'precision': p,
        'recall': r,
        'fscore': fscore,
        'support': support
    }
    return eval_metrics


def clf_predict(clf, dataset: Dict[str, Dict[str, DataEntry]], target_split: str, class_map, one_hot_encoding=False):
    keys = list(dataset[target_split].keys())
    x = np.array([entry.value for entry in dataset[target_split].values()])
    y = np.array([entry.label for entry in dataset[target_split].values()])
    if one_hot_encoding:
        y = get_one_hot_encoding(y, class_map)
    raw_preds = clf.predict_proba(x)
    return raw_preds, y, keys


def clf_objective(trial, max_iter, dataset: Dict[str, Dict[str, DataEntry]],
                  class_map: List[str]) -> float:
    """Optuna objective function that optimises classifier accuracy score."""
    # Parameter tuning
    estimator = trial.suggest_categorical("estimator", ["svc", "mlp", "logres"])
    trial.set_user_attr("random_state", 0)
    trial.set_user_attr("max_iter", max_iter)
    if estimator == "svc":
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        trial.set_user_attr("probability", True)
        trial.suggest_float("C", 1e0, 1e3, log=True),
        if kernel == "poly":
            trial.suggest_int("degree", 2, 4)
        if kernel in ["poly", "rbf", "sigmoid"]:
            trial.suggest_categorical("gamma", ['scale', 'auto'])
    elif estimator == "mlp":
        trial.suggest_categorical("solver_mpl", ["adam", "lbfgs", "sgd"])
        trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
        trial.suggest_categorical("hidden_layer_sizes", ["(100,)", "(256, 128)"])
    elif estimator == "logres":
        trial.suggest_categorical("solver_logres", ["newton-cg", "lbfgs", "sag", "saga"])
        trial.suggest_float("C", 1e-3, 1e3, log=True)

    # for trials repeated (same params), return already computed results
    complete_trials = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in complete_trials[::-1]:
        if trial.params == t.params:
            return t.value

    clf = train_clf(dataset, clf_params={**trial.params, **trial.user_attrs})
    eval_metrics = eval_clf(clf, dataset, class_map, target_split='valid')
    return eval_metrics['accuracy']


def tune_train_eval_clf(num_trials, max_iter, dataset: Dict[str, Dict[str, DataEntry]],
                        class_map: List[str]):
    # Perform hyper-parameter tuning using optuna
    logging.info(f'Running Optuna study...')
    clf_study = optuna.create_study(study_name='clf_study', direction='maximize')
    clf_study.optimize(
        lambda trial: clf_objective(trial, max_iter, dataset, class_map), n_trials=num_trials)
    display_optuna_study_results(clf_study, logging)

    # Train classifier
    logging.info('Training shallow classifier...')
    clf = train_clf(dataset, {**clf_study.best_trial.user_attrs, **clf_study.best_trial.params})

    # Evaluate classifier
    logging.info('Evaluating shallow classifier...')
    eval_metrics = eval_clf(clf, dataset, class_map, target_split='test')

    return clf, clf_study, eval_metrics


def display_optuna_study_results(study, logger):
    logger.info("Optuna study results:")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"score: {trial.value}")
    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info(f"{key}: {value}")


def export_model(clf, model_dir: Path, model_name, class_map, model_config=None):
    logging.info(f'Writing model artifacts to : {model_dir}')
    model_dir.mkdir(exist_ok=False, parents=True)
    joblib.dump(clf, f'{model_dir}/model.joblib')

    # create and save model config
    model_config = model_config or {}
    model_config.update({
        "model_name": model_name,
        "class_map": class_map
    })
    joblib.dump(model_config, f'{model_dir}/config.joblib')
    with open(f'{model_dir}/config.json', 'w') as outfile:
        json.dump(model_config, outfile, indent=2)


def get_one_hot_encoding(values, class_map: List[str]):
    # Transform list of class-indices to one-hot encoding
    one_hot = np.zeros((len(values), len(class_map)+1))
    one_hot[np.arange(len(values)), values] = 1
    return one_hot[:, :-1]