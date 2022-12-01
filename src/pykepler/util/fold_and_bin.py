
__all__ = ["phasefold", "binning"]

import numpy as np
from scipy.stats import binned_statistic


def phasefold(time, flux, err, t0, period, pcenter=0.5):
    t_fold = (time - t0  + pcenter * period) % period - pcenter * period
    index = np.argsort(t_fold)
    return t_fold[index], flux[index], err[index]


def binning(x, y, binwidth, statistic='median'):
    bins = np.arange(np.min(x), np.max(x), binwidth)
    vals, _, _ = binned_statistic(x, y, statistic=statistic, bins=bins)
    stds, _, _ = binned_statistic(x, y, statistic="std", bins=bins)
    counts, _, _ = binned_statistic(x, y, statistic="count", bins=bins)
    return 0.5 * (bins[1:] + bins[:-1]), vals, stds / np.sqrt(counts), counts
