import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
from torch.distributions import Normal


def callibration_bin_probs(target_cdf, n_bins, cummulative=False):
    uniform_bins = np.linspace(0, 1, n_bins + 1)
    uniform_idxs = np.digitize(target_cdf, uniform_bins, right=False)

    bin_counts = np.zeros(n_bins)
    for idx in range(n_bins + 1):
        if idx == n_bins:
            bin_counts[idx-1] += np.sum((uniform_idxs == idx+1).astype(int))
        else:
            bin_counts[idx] = np.sum((uniform_idxs == idx+1).astype(int))

    assert bin_counts.sum() == target_cdf.shape[0]
    bin_prop = bin_counts / target_cdf.shape[0]

    if cummulative:
        bin_prop = np.cumsum(bin_prop)
    return bin_prop, bin_counts, uniform_bins


def gauss_callibration(pred_means, pred_stds, targets, n_bins, cummulative=False, two_sided=False):
    if two_sided:
        norm_pred_err = (pred_means - targets) / pred_stds
        uniform_pred_err = norm.cdf(norm_pred_err)
    else:
        norm_pred_err = np.abs(pred_means - targets) / pred_stds
        uniform_pred_err = norm.cdf(norm_pred_err) * 2 - 1

    bin_prop, bin_counts, uniform_bins = callibration_bin_probs(target_cdf=uniform_pred_err,
                                                                n_bins=n_bins, cummulative=cummulative)

    bin_centers = uniform_bins[1:] - 0.5 / n_bins
    bin_width = 1 / n_bins

    if not cummulative:
        reference = np.ones(len(bin_centers)) * bin_width
    else:
        reference = np.arange(0, 1 + 1 / len(bin_centers), 1 / len(bin_centers))
    # TODO: ensure reference is correct, I think it should just be bin centers. Also return counts
    return bin_prop, bin_centers, bin_width, bin_counts, reference


def expected_callibration_error(bin_probs, reference, bin_counts, tail=False):
    bin_abs_error = np.abs(bin_probs - reference)
    if tail:
        tail_count = bin_counts[0] + bin_counts[-1]
        ECE = (bin_abs_error[0] * bin_counts[0] + bin_abs_error[-1] * bin_counts[-1]) / tail_count
    else:
        ECE = (bin_abs_error * bin_counts / bin_counts.sum(axis=0)).sum(axis=0)
    assert not np.isnan(ECE)
    return ECE


def rms(x, y):
    return F.mse_loss(x, y, reduction='mean').sqrt()


def get_rms(mu, y, y_means, y_stds):
    x_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    assert x_un.shape[1] == 1
    assert y_un.shape[1] == 1
    return rms(x_un, y_un)


def get_gauss_loglike(mu, sigma, y, y_means, y_stds):
    mu_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    sigma_un = sigma * y_stds
    assert mu_un.shape[1] == 1
    assert y_un.shape[1] == 1
    assert sigma_un.shape[1] == 1
    dist = Normal(mu_un, sigma_un)
    return dist.log_prob(y_un).mean(axis=0).item()  # mean over datapoints