import numpy as np
import sklearn.metrics as metrics
import pymia.evaluation.metric as m


def ece_binary(probabilities, target, n_bins=10, threshold_range: tuple = None, mask=None, out_bins: dict = None,
               bin_weighting='proportion'):

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


def binary_calibration(probabilities, target, n_bins=10, threshold_range: tuple = None, mask=None):
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


def _get_proportion(bin_weighting: str, bin_count: np.ndarray, non_zero_bins: np.ndarray, n_dim: int):
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


def uncertainty(prediction, target, thresholded_uncertainty, mask=None):
    if mask is not None:
        prediction = prediction[mask]
        target = target[mask]
        thresholded_uncertainty = thresholded_uncertainty[mask]

    tps = np.logical_and(target, prediction)
    tns = np.logical_and(~target, ~prediction)
    fps = np.logical_and(~target, prediction)
    fns = np.logical_and(target, ~prediction)

    tpu = np.logical_and(tps, thresholded_uncertainty).sum()
    tnu = np.logical_and(tns, thresholded_uncertainty).sum()
    fpu = np.logical_and(fps, thresholded_uncertainty).sum()
    fnu = np.logical_and(fns, thresholded_uncertainty).sum()

    tp = tps.sum()
    tn = tns.sum()
    fp = fps.sum()
    fn = fns.sum()

    return tp, tn, fp, fn, tpu, tnu, fpu, fnu


def error_dice(fp, fn, tpu, tnu, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fn + fp + fnu + fpu + tnu + tpu) == 0):
        return 1.
    return (2 * (fnu + fpu)) / (fn + fp + fnu + fpu + tnu + tpu)


def error_recall(fp, fn, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fn + fp) == 0):
        return 1.
    return (fnu + fpu) / (fn + fp)


def error_precision(tpu, tnu, fpu, fnu):
    if ((fnu + fpu) == 0) and ((fnu + fpu + tpu + tnu) == 0):
        return 1.
    return (fnu + fpu) / (fnu + fpu + tpu + tnu)


def dice(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    d = m.DiceCoefficient()
    d.confusion_matrix = m.ConfusionMatrix(prediction, target)
    return d.calculate()


def confusion_matrx(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    cm = m.ConfusionMatrix(prediction, target)
    return cm.tp, cm.tn, cm.fp, cm.fn, cm.n


def accuracy(prediction, target):
    _check_ndarray(prediction)
    _check_ndarray(target)

    a = m.Accuracy()
    a.confusion_matrix = m.ConfusionMatrix(prediction, target)
    return a.calculate()


def log_loss_sklearn(probabilities, target, labels=None):
    _check_ndarray(probabilities)
    _check_ndarray(target)

    if probabilities.shape[-1] != target.shape[-1]:
        probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    else:
        probabilities = probabilities.reshape(-1)
    target = target.reshape(-1)
    return metrics.log_loss(target, probabilities, labels=labels)


def entropy(p, dim=-1, keepdims=False):
    # exactly the same as scipy.stats.entropy()
    return -np.where(p > 0, p * np.log(p), [0.0]).sum(axis=dim, keepdims=keepdims)


def _check_ndarray(obj):
    if not isinstance(obj, np.ndarray):
        raise ValueError("object of type '{}' must be '{}'".format(type(obj).__name__, np.ndarray.__name__))
