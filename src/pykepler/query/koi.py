
__all__ = ["check_koi_ephemeris"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from scipy.stats import binned_statistic, median_abs_deviation


def phasefold(time, flux, t0, period):
    t_fold = (time - t0  + 0.5 * period) % period - 0.5 * period
    index = np.argsort(t_fold)
    return t_fold[index], flux[index]


def binning(x, y, binwidth, statistic='median'):
    bins = np.arange(np.min(x), np.max(x), binwidth)
    vals, _, _ = binned_statistic(x, y, statistic=statistic, bins=bins)
    return 0.5*(bins[1:]+bins[:-1]), vals


def check_koi_ephemeris(kic, t, f, datadir=None, save_plot=False):
    if datadir is None:
        datadir = "kic%s"%kic

    koitable = NasaExoplanetArchive.query_criteria(table="cumulative", where="kepid like '%s'"%kic).to_pandas()
    koitable = koitable.sort_values("koi_period").reset_index(drop=True)

    fig, ax = plt.subplots(len(koitable), 2, figsize=(20,5*len(koitable)), sharey=True)
    for i in range(len(koitable)):
        df = koitable.iloc[i]
        name, t0, period, duration = df[["kepoi_name", "koi_time0bk", "koi_period", "koi_duration"]]
        duration /= 24.0
        binwidth = duration / 5.

        t_fold, f_fold = phasefold(t, f, t0, period)
        t_bin, f_bin = binning(t_fold, f_fold, binwidth)
        fmed, fmad = np.median(f_fold), median_abs_deviation(f_fold)
        depth = np.median(f_bin) - np.min(f_bin)
        fmin = min(fmed-10*fmad, fmed-2*depth)
        fmax = max(fmed+10*fmad, fmed+1*depth)
        #print (df[["kepoi_name", "koi_time0bk", "koi_period", "koi_duration"]])

        ax[i,0].set_title(name + " full orbit")
        ax[i,0].set_xlim(np.min(t_fold), np.max(t_fold))
        ax[i,0].plot(t_fold, f_fold, '.', ms=3, alpha=0.5, label="folded data (period: %fdays)"%period)
        ax[i,0].plot(t_bin, f_bin, '.', ms=6, label="binned data (bin width: %fdays)"%binwidth)
        ax[i,0].set_xlabel("time")
        ax[i,0].set_ylabel("normalized flux")
        #ax[i,0].legend(loc='best');

        ax[i,1].set_title(name + " transit")
        ax[i,1].set_xlim(-duration*5, duration*5)
        #ax[i,1].set_ylim(fmed-10*fmad, fmed+10*fmad)
        ax[i,1].set_ylim(fmin, fmax)
        ax[i,1].plot(t_fold, f_fold, '.', ms=3, alpha=0.5, label="folded data (period: %.4fdays)"%period)
        ax[i,1].plot(t_bin, f_bin, 'o', ms=6, label="binned data (bin width: %.4fdays)"%binwidth)
        ax[i,1].set_xlabel("time")
        ax[i,1].legend(loc='best', bbox_to_anchor=(1,1))
    fig.tight_layout(h_pad=2.0)

    if save_plot:
        if not os.path.exists(datadir):
            os.system("mkdir %s"%datadir)
        plt.savefig(datadir+"/kic%s_fold.png"%kic, dpi=200, bbox_inches="tight")

    return koitable
