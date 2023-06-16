import torch
from .optim.vecmodule import Parameter, LBFGS_Objective, lbfgs
from . import losses 
from IPython import embed
import numpy as np

class AffineCal(LBFGS_Objective):

    def __init__(self, scores, labels, bias=True, scale=True, priors=None):    
        # If priors are provided, ignore the data priors and use those ones instead
        # In this case, the scores are taken to be log scaled likelihoods

        super().__init__()

        if priors is not None:
            self.priors = torch.Tensor(priors)
        else:
            self.priors = None

        self.has_scale = scale
        if scale:
            self.temp = Parameter(torch.tensor(1.0, dtype=torch.float64))
        else:
            self.temp = 1.0

        self.has_bias = bias
        if bias:
            if self.priors is not None:
                # If external priors are provided, initialize the bias this way
                # so that if the scores are perfectly calibrated for those priors
                # the training process does not need to do anything.
                self.bias = Parameter(-torch.log(self.priors))
            else:
                self.bias = Parameter(torch.zeros(scores.shape[1], dtype=torch.float64))
        else:
            self.bias = 0

        self.scores = scores
        self.labels = labels

    def train(self, quiet=True):

        return lbfgs(self, 100, quiet=quiet)


    def calibrate(self, scores):
        self.cal_scores = self.temp * scores + self.bias
        if self.priors is not None:
            self.cal_scores += torch.log(self.priors)

        self.log_probs = self.cal_scores - torch.logsumexp(self.cal_scores, axis=-1, keepdim=True) 
        return self.log_probs

    def loss(self):
        pass

class AffineCalLogLoss(AffineCal):
    def loss(self):
        return losses.LogLoss(self.calibrate(self.scores), self.labels, priors=self.priors, norm=False)
        

class AffineCalECE(AffineCal):
    def loss(self):
        return losses.ECE(self.calibrate(self.scores), self.labels)

class AffineCalLogLossPlusECE(AffineCal):

    def __init__(self, scores, labels, ece_weight=0.5, bias=True):
        super().__init__(scores, labels, bias)
        self.ece_weight = ece_weight

    def loss(self):
        return (1-self.ece_weight) * losses.LogLoss(self.calibrate(self.scores), self.labels) + self.ece_weight * losses.ECE(self.calibrate(self.scores), self.labels)


class AffineCalBrier(AffineCal):
    def loss(self):
        return losses.Brier(self.calibrate(self.scores), self.labels, norm=False)


class HistogramBinningCal():

    def __init__(self, scores, labels, M=15, **kwargs):

        # Histogram binning, as implemented here, only applies to binary classification.

        if scores.ndim != 1 and scores.shape[1]!=2:
            raise Exception("Histogram binning only implemented for binary classification")

        # The method assumes the scores are log probs, but we bin the probs, so take the exp.
        self.scores = torch.exp(scores)
        self.labels = labels.double()
        self.M = M


    def train(self):

        # Take the second score for binning
        if self.scores.ndim == 2:
            scores = self.scores[:,1]
        else:
            scores = self.scores

        # Generate intervals
        limits = np.linspace(0, 1, num=self.M+1)
        self.lows, self.highs = limits[:-1], limits[1:]
        self.cal_transform = []
        self.ave_score_per_bin = []

        # Obtain the proportion of samples of class 2 for each bin
        # This is the calibration transform to be applied for any posterior
        # within that bin.
        for low, high in zip(self.lows, self.highs):
            ix = (low < scores) & (scores <= high)
            n = torch.sum(ix)
            self.cal_transform.append(torch.mean(self.labels[ix]) if n!=0 else 0.0)
            self.ave_score_per_bin.append(torch.mean(scores[ix]) if n!= 0 else 0.0)


    def calibrate(self, scores):

        scores = torch.exp(scores)
        if scores.ndim == 2:
            scores = scores[:,1]
            return_both_probs = True
        else:
            return_both_probs = False

        cal_scores = torch.zeros_like(scores)
        binned_scores = torch.zeros_like(scores)

        # Obtain the proportion of samples of class 2 for each bin
        # This is the calibration transform to be applied for any posterior
        # within that bin.
        for i, (low, high) in enumerate(zip(self.lows, self.highs)):
            ix = (low < scores) & (scores <= high)
            cal_scores[ix] = self.cal_transform[i]
            binned_scores[ix] = self.ave_score_per_bin[i]

        if return_both_probs:
            cal_scores = torch.stack([1-cal_scores, cal_scores]).T

        # Save the binned scores for when we use this calibration to compute the ECE
        self.binned_scores = binned_scores

        # Go back to log domain
        return torch.log(cal_scores)



def calibrate(trnscores, trnlabels, tstscores, calclass, quiet=True, **kwargs):

    obj = calclass(trnscores, trnlabels, **kwargs)

    if calclass == HistogramBinningCal:

        obj.train()
        return obj.calibrate(tstscores), [obj.binned_scores, obj.lows, obj.highs, obj.cal_transform, obj.ave_score_per_bin]

    else:

        paramvec, value, curve, success = obj.train(quiet=quiet)
        
        if not success:
           raise Exception("LBFGS was unable to converge")
            
        return obj.calibrate(tstscores), [obj.temp, obj.bias] if obj.has_bias else [obj.temp]



