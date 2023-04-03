import pickle
import os
import torch
import numpy as np
import math
import abc
import joblib
import joblib
import torch.nn.functional as F
# import torchfunctions as t_fn
import numpyfunctions as np_fn

def Uentropy(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = F.softmax(logits, dim=1)  # 1 4 240 240 155
    logpc = F.log_softmax(logits, dim=1)  # 1 4 240 240 155
    # u_all1 = -pc * logpc / c
    u_all = -pc * logpc / math.log(c)
    # max_u = torch.max(u_all)
    # min_u = torch.min(u_all)
    NU = torch.sum(u_all, dim=1)
    return NU

class UncertaintyAndCorrectionEvalNumpy():
    def __init__(self, uncertainty_threshold):
        super(UncertaintyAndCorrectionEvalNumpy, self).__init__()
        self.uncertainty_threshold = uncertainty_threshold

    def __call__(self, to_evaluate=None, results=None):
        # entropy should be normalized [0,1] before calling this evaluation
        target = to_evaluate['target'].astype(np.bool)
        prediction = to_evaluate['prediction'].astype(np.bool)
        uncertainty = to_evaluate['uncertainty']

        thresholded_uncertainty = uncertainty > self.uncertainty_threshold
        tp, tn, fp, fn, tpu, tnu, fpu, fnu = \
            np_fn.uncertainty(prediction, target, thresholded_uncertainty)

        results['tpu'] = tpu
        results['tnu'] = tnu
        results['fpu'] = fpu
        results['fnu'] = fnu

        results['tp'] = tp
        results['tn'] = tn
        results['fp'] = fp
        results['fn'] = fn

        tpu_fpu_ratio = results['tpu'] / results['fpu']
        jaccard_index = results['tp'] / (results['tp'] + results['fp'] + results['fn'])
        results['dice_benefit'] = tpu_fpu_ratio < jaccard_index
        results['accuracy_benefit'] = tpu_fpu_ratio < 1

        results['dice'] = np_fn.dice(prediction, target)
        results['accuracy'] = np_fn.accuracy(prediction, target)

        corrected_prediction = prediction.copy()
        # correct to background
        corrected_prediction[thresholded_uncertainty] = 0

        results['corrected_dice'] = np_fn.dice(corrected_prediction, target)
        results['corrected_accuracy'] = np_fn.accuracy(corrected_prediction, target)

        results['dice_benefit_correct'] = (results['corrected_dice'] > results['dice']) == results['dice_benefit']
        results['accuracy_benefit_correct'] = (results['corrected_accuracy'] > results['accuracy']) == results[
            'accuracy_benefit']

        corrected_prediction = prediction.copy()
        # correct to foreground
        corrected_prediction[thresholded_uncertainty] = 1

        results['corrected_add_dice'] = np_fn.dice(corrected_prediction, target)
        results['corrected_add_accuracy'] = np_fn.accuracy(corrected_prediction, target)

def binary_calibration(probabilities, target, n_bins=10, threshold_range = None, mask=None):
    if probabilities.ndim > target.ndim:
        if probabilities.shape[-1] > 2:
            raise ValueError('can only evaluate the calibration for binary classification')
        elif probabilities.shape[-1] == 2:
            probabilities = probabilities[..., 1]
        else:
            probabilities = np.squeeze(probabilities, axis=-1)

    if mask is not None:
        probabilities = probabilities[mask]
        target = target[mask]

    if threshold_range is not None:
        low_thres, up_thres = threshold_range
        mask = np.logical_and(probabilities < up_thres, probabilities > low_thres)
        probabilities = probabilities[mask]
        target = target[mask]

    pos_frac, mean_confidence, bin_count, non_zero_bins = \
        _binary_calibration(target.flatten(), probabilities.flatten(), n_bins)

    return pos_frac, mean_confidence, bin_count, non_zero_bins

def _binary_calibration(target, probs_positive_cls, n_bins=10):
    # same as sklearn.calibration calibration_curve but with the bin_count returned
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(probs_positive_cls, bins) - 1

    # # note: this is the original formulation which has always n_bins + 1 as length
    # bin_sums = np.bincount(binids, weights=probs_positive_cls, minlength=len(bins))
    # bin_true = np.bincount(binids, weights=target, minlength=len(bins))
    # bin_total = np.bincount(binids, minlength=len(bins))

    bin_sums = np.bincount(binids, weights=probs_positive_cls, minlength=n_bins)
    bin_true = np.bincount(binids, weights=target, minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)

    nonzero = bin_total != 0
    prob_true = (bin_true[nonzero] / bin_total[nonzero])
    prob_pred = (bin_sums[nonzero] / bin_total[nonzero])

    return prob_true, prob_pred, bin_total[nonzero], nonzero

def _get_proportion(bin_weighting, bin_count, non_zero_bins, n_dim):
    if bin_weighting == 'proportion':
        bin_proportions = bin_count / bin_count.sum()
    elif bin_weighting == 'log_proportion':
        bin_proportions = np.log(bin_count) / np.log(bin_count).sum()
    elif bin_weighting == 'power_proportion':
        bin_proportions = bin_count**(1/n_dim) / (bin_count**(1/n_dim)).sum()
    elif bin_weighting == 'mean_proportion':
        bin_proportions = 1 / non_zero_bins.sum()
    else:
        raise ValueError('unknown bin weighting "{}"'.format(bin_weighting))
    return bin_proportions

def ece_binary(probabilities, target, n_bins=10, threshold_range= None, mask=None, out_bins=None,
               bin_weighting='proportion'):
# input: 1. probabilities (np) 2. target (np) 3. threshold_range (tuple[low,high]) 4. mask

    n_dim = target.ndim

    pos_frac, mean_confidence, bin_count, non_zero_bins = \
        binary_calibration(probabilities, target, n_bins, threshold_range, mask)

    bin_proportions = _get_proportion(bin_weighting, bin_count, non_zero_bins, n_dim)

    if out_bins is not None:
        out_bins['bins_count'] = bin_count
        out_bins['bins_avg_confidence'] = mean_confidence
        out_bins['bins_positive_fraction'] = pos_frac
        out_bins['bins_non_zero'] = non_zero_bins

    ece = (np.abs(mean_confidence - pos_frac) * bin_proportions).sum()
    return ece

def pkload(fname):
    with open(fname, 'rb') as f:
        return joblib.load(f)
        # return pickle.load(f)

def cal_u(uncertainty):
    mean_uncertainty_total = 0
    sum_uncertainty_total = 0
    for i in range(len(uncertainty)):
        u  = uncertainty[i][0]
        total_uncertainty = torch.sum(u, -1, keepdim=True)
        # print('current_sum_certainty:{} ; current_mean_certainty:{}'.format(torch.mean(total_uncertainty),
        #                                                                   torch.mean(u)))
        sum_uncertainty_total += torch.mean(total_uncertainty)
        mean_uncertainty_total += torch.mean(u)
    num = len(uncertainty)
    return sum_uncertainty_total/num,mean_uncertainty_total/num

def cal_ece(logits,targets):
    ece_total = 0
    for i in range(len(logits)):
        # pc = torch.zeros(1,1,128,128,128)
        logit = logits[i][0]
        target = targets[i][..., :155]
        pred = F.softmax(logit, dim=0)
        pc = pred.cpu().detach().numpy()
        pc = pc.argmax(0)
        ece_total += ece_binary(pc, target)
    num = len(logits)
    return ece_total/num

def cal_ueo(to_evaluate,thresholds):
    UEO = []
    for threshold in thresholds:
        metric = UncertaintyAndCorrectionEvalNumpy(threshold)
        UEO.append(list(metric(to_evaluate)))
    return UEO