from ..psrcal.calibration import calibrate, AffineCalLogLoss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold
import torch
import numpy as np

def calibration_with_crossval(logpost, targets, use_bias=True, priors=None, calmethod=AffineCalLogLoss, seed=None, 
                              condition_ids=None, stratified=True, nfolds=5):
    
    """ 
    This is a utility method for performing calibration on the test scores using cross-validation.
    If calmethod is AffineCalLogLoss of AffineCalBrier, the calibration is done using an affine
    transformation of the form:
    
    logpostcal_i = logsoftmax ( scale * logpostraw_i + bias_i)

    The scale is the same for all classes, but the bias is class-dependent. If use_bias = False,
    this method does the usual temp-scaling.

    If calmethod is HistogramBinningCal, histogram binning is done instead.

    The method expects (potentially misscalibrated) log posteriors or log scaled likelihoods as
    input. In both cases, the output should be well-calibrated log posteriors. Note, though that
    when use_bias is False, it is probably better to feed log posteriors since, without a bias term,
    the calibration cannot introduce the required priors to obtain a good log-posterior. 

    The priors variable allows you to set external priors which overwrite the ones in the test data.
    This is useful when the test data has priors that do not reflect those we expect to see when we
    deploy the system.

    The condition_ids variable is used to determine the folds per condition. This is used when the
    data presents correlations due to some factor other than the class. In that case, the
    condition_ids variable should have indexes for the condition of each sample in the logpost
    array.

    Set stratified to True if you want to assume that the priors are always known with certainty.
    Else, if you want to consider the possible random variation in priors due to the sampling of the
    data, then set stratified to False.
    """

    logpostcal = np.zeros_like(logpost)
    
    if stratified:
        if condition_ids is not None:
            skf = StratifiedGroupKFold(n_splits=nfolds, shuffle=True, random_state=seed)
        else:
            # Use StratifiedKFold in this case for backward compatibility
            skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    else:
        skf = GroupKFold(n_splits=nfolds)

    for trni, tsti in skf.split(logpost, targets, condition_ids):

        trnf = torch.as_tensor(logpost[trni], dtype=torch.float32)
        tstf = torch.as_tensor(logpost[tsti], dtype=torch.float32)
        trnt = torch.as_tensor(targets[trni], dtype=torch.int64)

        calmodel = calmethod(trnf, trnt, bias=use_bias, priors=priors)
        calmodel.train()
        tstf_cal = calmodel.calibrate(tstf)

        logpostcal[tsti] = tstf_cal.detach().numpy()

    return logpostcal


def calibration_train_on_heldout(logpost_tst, logpost_trn, targets_trn, use_bias=True, priors=None, calmethod=AffineCalLogLoss, return_model=False):
    """ Same as calibration_with_crossval but doing cheating calibration.
    """

    trnf = torch.as_tensor(logpost_trn, dtype=torch.float32)
    tstf = torch.as_tensor(logpost_tst, dtype=torch.float32)
    trnt = torch.as_tensor(targets_trn, dtype=torch.int64)

    calmodel = calmethod(trnf, trnt, bias=use_bias, priors=priors)
    calmodel.train()
    tstf_cal = calmodel.calibrate(tstf)

    logpostcal = tstf_cal.detach().numpy()

    if return_model:
        return logpostcal, calmodel
    else:
        return logpostcal

def calibration_train_on_test(logpost, targets, use_bias=True, priors=None, calmethod=AffineCalLogLoss, return_model=False):

    return calibration_train_on_heldout(logpost, logpost, targets, use_bias, priors, calmethod, return_model)

