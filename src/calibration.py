
import numpy as np
import pandas as pd
from .expected_cost.calibration import train_calibrator, calibrate_scores, calibration_train_on_heldout, calibration_with_crossval
from .psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal


calmethod_name2fn = {
    "AffineCalLogLoss": AffineCalLogLoss,
    "AffineCalBrier": AffineCalBrier,
    "HistogramBinningCal": HistogramBinningCal
}

class Reestimator:

    def __init__(self):
        self.W = None
        self.b = None

    def train(self, probs_train):
        num_classes = probs_train.shape[1]
        mean_probs = probs_train.mean(axis=0)

        self.W = np.identity(num_classes) * mean_probs
        self.b = np.zeros([num_classes, 1])
    
    def reestimate(self, probs_test):
        transformed_probs = np.matmul(np.linalg.inv(self.W), probs_test.T).T + self.b.T
        transformed_probs /= transformed_probs.sum(axis=1, keepdims=True)
        return transformed_probs


def train_calibrator_from_probs(
    probs_train,
    targets_train,
    calmethod='AffineCalLogLoss', 
    calparams={}
):
    calmethod_fn = calmethod_name2fn[calmethod]
    train_scores = np.log(probs_train)
    calmodel = train_calibrator(train_scores, targets_train, calparams=calparams, calmethod=calmethod_fn)
    return calmodel


def calibrate_probs_from_trained_model(
    probs_test,
    calmodel
):
    test_scores = np.log(probs_test)
    test_scores_cal = calibrate_scores(test_scores, calmodel)
    test_probs_cal = np.exp(test_scores_cal)
    test_probs_cal /= test_probs_cal.sum(axis=1, keepdims=True)
    return test_probs_cal


def train_reestimator_from_probs(
    probs_train
):
    reestimator = Reestimator()
    reestimator.train(probs_train)
    return reestimator


def reestimate_probs_from_trained_model(
    probs_test,
    reestimator
):
    probs_test_reest = reestimator.reestimate(probs_test)
    return probs_test_reest


def calibrate_from_train_probs(
    probs_train, 
    targets_train, 
    probs_test, 
    targets_test, 
    calmethod='AffineCalLogLoss', 
    calparams={}, 
    cross_val=False,
    boots_idx=None
):
    
    calmethod_fn = calmethod_name2fn[calmethod]
    
    test_scores = np.log(probs_test)
    if not cross_val:
        train_scores = np.log(probs_train)
        test_scores_cal = calibration_train_on_heldout(test_scores, train_scores, targets_train, calparams=calparams,calmethod=calmethod_fn, return_model=False)
    else:
        # train set not used in cross-validation
        test_scores_cal = calibration_with_crossval(test_scores, targets_test, calparams=calparams, calmethod=calmethod_fn, condition_ids=boots_idx, stratified=False)
    test_probs_cal = np.exp(test_scores_cal)
    test_probs_cal /= test_probs_cal.sum(axis=1, keepdims=True)
    return test_probs_cal






