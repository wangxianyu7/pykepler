""" GP detrending for KOIs """

__all__ = ["detrend_by_quarter"]

import numpy as np
import matplotlib.pyplot as plt
from . import gpfit_with_mask

def get_transit_mask(t, dkoi, mask_half_width):
    """ mask in-transit data based on KOI catalog t0 and period
        w/o using koi_duration

        Args:
            t: times (array)
            dkoi: pandas dataframe containing KOI catalog
            mask_half_width: half width of the mask (same units as period)

        Returns:
            mask array (masked if True)

    """
    mask = np.zeros_like(t)
    for i in range(len(dkoi)):
        t0, p = np.array(dkoi.iloc[i][['koi_time0bk', 'koi_period']])
        phase = (t - t0) / p
        phase -= np.round(phase)
        idx_in_transit = np.abs(phase) * p < mask_half_width
        mask[idx_in_transit] += 1.
    mask = mask > 0.
    return mask


def get_koi_transit_mask(t, dkoi, duration_margin):
    """ mask in-transit data based on KOI catalog t0, period, and duration

        Args:
            t: times (array)
            dkoi: pandas dataframe containing KOI catalog
            duration_margin: mask half width given by duration_margin * koi_duration

        Returns:
            mask array (masked if True)

    """
    mask = np.zeros_like(t)
    for i in range(len(dkoi)):
        t0, p, dur = np.array(dkoi.iloc[i][['koi_time0bk', 'koi_period', 'koi_duration']])
        mask_half_width = 0.5 * dur / 24. * duration_margin
        phase = (t - t0) / p
        phase -= np.round(phase)
        idx_in_transit = np.abs(phase) * p < mask_half_width
        mask[idx_in_transit] += 1.
    mask = mask > 0.
    return mask


def detrend_by_quarter(data, dkoi, remove_bad_flux=True, plot=True, duration_margin=2.,midtimes=None,durations=None):
    """ detrend KOI light curves

        Args:
            data: KOI dataframe (keys: time, flux, error, quarter, quality)
            dkoi: pandas dataframe containing KOI catalog
            remove_bad_flux: if True, remove data points with non-zero quality flag
            plot: if True plot the results
            duration_margin: argument for get_koi_transit_mask

        Returns:
            t_out, f_out, e_out, q_out: time, flux, error, quarter (bad flux removed; otherwise the same!)
            fbase_out: fitted baseline

    """
    t_out, f_out, e_out, q_out = np.array([]), np.array([]), np.array([]), np.array([])
    fbase_out = np.array([])
    for q in list(set(data.quarter)):
        idx = data.quarter == q
        if remove_bad_flux:
            idx &= (data.quality == 0)
        t, f, e = np.array(data.time[idx]), np.array(data.flux[idx]), np.array(data.error[idx])

        #mask = get_transit_mask(t, dkoi, max_half_duration)
        #mask = get_transit_mask(t, dkoi, max_half_duration)
        if np.any(midtimes)==None:
            mask = get_koi_transit_mask(t, dkoi, duration_margin=duration_margin)
        else:
            mask = []
            for i in range(len(t)):
                in_time = False
                for j in range(len(midtimes)):
                    if abs(t[i] - midtimes[j]) < 0.1:
                        mask.append(True)
                        in_time = True
                if not in_time:
                    mask.append(False)
        mask = np.asarray(mask)
        fbase, _ = gpfit_with_mask(t, f, e, mask=mask)

        t_out = np.r_[t_out, t]
        f_out = np.r_[f_out, f]
        e_out = np.r_[e_out, e]
        q_out = np.r_[q_out, np.ones_like(t)*q]
        fbase_out = np.r_[fbase_out, fbase]

        if plot:
            plt.figure(figsize=(14,5))
            plt.title("quarter %d"%(q))
            plt.plot(t, f, '.', alpha=0.1, color='C0')
            plt.plot(t[~mask], f[~mask], '.', alpha=1, color='C0')
            plt.plot(t, fbase, '-', color='C1')
            plt.plot(t, f-fbase+np.min(f)-2*np.std(f), '.', alpha=0.4, color='C0')

    return t_out, f_out, e_out, q_out.astype(int), fbase_out
