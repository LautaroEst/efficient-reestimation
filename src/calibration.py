
import numpy as np
import pandas as pd
from .expected_cost.calibration import calibration_with_crossval, calibration_train_on_heldout
from .psrcal.calibration import AffineCalLogLoss, AffineCalBrier, HistogramBinningCal


calmethod_name2fn = {
    "AffineCalLogLoss": AffineCalLogLoss,
    "AffineCalBrier": AffineCalBrier,
    "HistogramBinningCal": HistogramBinningCal
}


def calibrate(
    probs_train, 
    targets_train, 
    probs_test, 
    targets_test, 
    calmethod='AffineCalLogLoss', 
    use_bias=True, 
    deploy_priors=None, 
    cross_val=False,
    boots_idx=None
):
    
    calmethod_fn = calmethod_name2fn[calmethod]
    
    test_scores = np.log(probs_test)
    train_scores = np.log(probs_train)

    if not cross_val:
        test_scores_cal = calibration_train_on_heldout(test_scores, train_scores, targets_train, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn, return_model=False)
    else:
        # train set not used in cross-validation
        test_scores_cal = calibration_with_crossval(test_scores, targets_test, condition_ids=boots_idx, stratified=False, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn)
    test_probs_cal = np.exp(test_scores_cal)

    return test_probs_cal

def reestimate(
    probs_train, 
    probs_test, 
):
    
    num_classes = probs_train.shape[1]
    mean_probs = probs_train.mean(axis=0)

    W = np.identity(num_classes) * mean_probs
    b = np.zeros([num_classes, 1])

    transformed_probs = np.matmul(np.linalg.inv(W), probs_test.T).T + b.T
    return transformed_probs





# def calibrate_and_evaluate_psr(scores, targets, metric="LogLoss", calmethod='AffineCalLogLoss', use_bias=True, deploy_priors=None, cross_val=False, bootstrap=False, random_state=0):

#     if bootstrap:
#         rs = np.random.RandomState(random_state)
#         boots_idx = rs.choice(len(targets), size=len(targets), replace=True)
#         scores = scores[boots_idx]
#         targets = targets[boots_idx]
#     else:
#         boots_idx = None
    
#     metric_fn = metric_name2fn[metric]
#     calmethod_fn = calmethod_name2fn[calmethod]
    
#     if not cross_val:
#         # If you have held-out data or want to do train-on-test, in which case, use the line
#         # below with scores_trn=scores_tst, target_trn=target_tst
#         scores_tst_cal, calmodel = calibration_train_on_heldout(scores, scores, targets, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod_fn, return_model=True)
#     else:
#         # Alternatively, for cross-validation on the test data
#         scores_tst_cal = calibration_with_crossval(scores, targets, conditions_ids=boots_idx, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod)
#         _, calmodel = calibration_train_on_heldout(scores, scores, targets, use_bias=use_bias, priors=deploy_priors, calmethod=calmethod, return_model=True)

#     # Finally, compute the metric before and after calibration
#     overall_perf = metric_fn(scores, targets, priors=deploy_priors, norm=True)
#     overall_perf_after_cal = metric_fn(scores_tst_cal, targets, priors=deploy_priors, norm=True)
#     cal_loss = overall_perf-overall_perf_after_cal
#     rel_cal_loss = 100*cal_loss/overall_perf

#     results = {
#         "scores": scores,
#         "targets": targets,
#         "scores_cal": scores_tst_cal,
#         "overall_perf": float(overall_perf),
#         "overall_perf_after_cal": float(overall_perf_after_cal),
#         "cal_loss": float(cal_loss),
#         "rel_cal_loss": float(rel_cal_loss),
#     }
#     return results
    

    # print(f"Overall performance before calibration ({metric}) = {overall_perf:4.2f}" ) 
    # print(f"Overall performance after calibration ({metric}) = {overall_perf_after_cal:4.2f}" ) 
    # print(f"Calibration loss = {cal_loss:4.2f}" ) 
    # print(f"Relative calibration loss = {rel_cal_loss:4.1f}" ) 