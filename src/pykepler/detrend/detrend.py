
__all__ = ["harmonic_search", "design_matrix_fourier", "irls_fit", "extend_edge", "detrend_by_quarter"]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from astropy.timeseries import LombScargle
from sklearn import linear_model
from scipy.ndimage import median_filter
from ..gaussianize import *


def design_matrix_fourier(x, df=None, fmax=None):
    if df is None:
        df = 1./(x[-1]-x[0])
    if fmax is None:
        fmax = 0.5/(np.min(np.diff(x)))
    nfou = int(np.round(fmax/df))
    ndata = len(x)
    X = np.zeros((ndata, 2*nfou+1))
    X[:,0] = np.ones(ndata)
    xmed = np.median(x)
    for i in range(nfou):
        _f = df * (i+1)
        X[:,1+2*i] = np.cos(2*np.pi*_f*(x-xmed))
        X[:,2+2*i] = np.sin(2*np.pi*_f*(x-xmed))
    return X


def irls_fit(x, y, X, n_itr=10):
    clf = linear_model.LinearRegression(fit_intercept=False)
    sigma = np.std(y)
    delta = np.zeros_like(y)
    for i in range(n_itr):
        w = 1./np.sqrt(sigma**2+delta**2)
        w[np.abs(delta) > sigma] = 0.
        reg = clf.fit(X, y, sample_weight=w)
        ypred = reg.predict(X)
        delta = y - ypred
        sigma = np.std(delta)
    return ypred


def extend_edge(x, y, nedge, size=3):
    xhead = x[0] - (x[1:nedge]-x[0])[::-1]
    yhead = median_filter(y[1:nedge], size=size)[::-1]
    xtail = x[-1] + (x[-1]-x[-1-nedge:-1])[::-1]
    ytail = median_filter(y[-1-nedge:-1], size=size)[::-1]
    yhead, ytail = 2*yhead[-1] - yhead, 2*ytail[0] - ytail
    xret, yret = np.r_[xhead, x, xtail], np.r_[yhead, y, ytail]
    return xret, yret, (xhead[-1]<xret)&(xret<xtail[0])


def fourier_baseline(t, f, filter_period, minimum_freq_frac=0.3):
    df = min(minimum_freq_frac / filter_period, 0.5 / (np.max(t) - np.min(t)))
    X = design_matrix_fourier(t, fmax=1./filter_period, df=df)
    fbase = irls_fit(t, f, X)
    return fbase


def medfilt_baseline(t, f, filter_period, cadence):
    ksize = int((filter_period/cadence)//2 * 2 + 1)
    fbase = median_filter(f, size=ksize, mode='nearest')
    return fbase


def detrend_chunk(t, f, filter_period, edge_fraction=0.05, minimum_freq_frac=0.3, mode='sine'):
    dt = np.median(np.diff(t))
    nedge = int(edge_fraction * (np.max(t) - np.min(t)) / dt)
    t_ext, f_ext, idx_original = extend_edge(t, f, nedge)
    if mode == 'sine':
        flux_base_ext = fourier_baseline(t_ext, f_ext, filter_period, minimum_freq_frac=minimum_freq_frac)
    else:
        flux_base_ext = medfilt_baseline(t_ext, f_ext, filter_period, dt)
    return flux_base_ext[idx_original], (t_ext, f_ext, flux_base_ext, idx_original)


def harmonic_search(t, y, dy, pmin=1., pmax=10., fap_threshold=1e-2, plot=True, power_threshold=30.):
    # power vs freq for periods between pmin and pmax
    ls = LombScargle(t, y, dy)
    freq, power = ls.autopower(samples_per_peak=10)
    #freq = np.logspace(np.log10(1./pmax), np.log10(1./pmin), 1000)
    #power = ls.power(freq)
    fidx = (1./pmax < freq) & (freq < 1./pmin)
    freq, power = freq[fidx], power[fidx]

    # find significant peaks
    pfreqs, ppowers = [], []
    idx = freq > 0.
    while True:
        powermax = np.max(power[idx])
        imax = np.where(power==powermax)[0][0]
        freqmax = freq[imax]
        fap = ls.false_alarm_probability(powermax)
        if fap < fap_threshold:
            pfreqs.append(freqmax)
            ppowers.append(powermax)
            idx &= ((freq < freqmax*0.8) | (freq > freqmax*1.2))
        else:
            break
        if np.sum(idx) == 0:
            break
    pfreqs = np.array(pfreqs)
    ppowers = np.array(ppowers)

    power_base = np.median(power[(0.5<freq)&(freq<1.)])
    power_ratio = ppowers / power_base
    idxp = power_ratio > power_threshold
    if np.sum(idxp):
        _idx = np.argmin(1./pfreqs[idxp])
        p_harmonic_min = (1./pfreqs[idxp])[_idx]
        power_ratio = power_ratio[idxp][_idx]
    else:
        p_harmonic_min, power_ratio = None, None

    if plot:
        plt.figure(figsize=(20,6))
        plt.ylabel("Lomb-Scargle power")
        plt.xlabel("period (days)")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(pmin, pmax)
        plt.plot(1./freq, power, '-', alpha=0.8, color='C0', lw=1)
        plt.plot(1./freq[~idx], power[~idx], 'o', alpha=0.8, color='salmon', markersize=3)
        plt.axhline(y=ls.false_alarm_level(fap_threshold), color='gray', ls='dashed', label='FAP %.1e%%'%(fap_threshold*100))

        if p_harmonic_min is not None:
            plt.axvline(x=p_harmonic_min, color='k', lw=1,
                        label='Pmin=%.1e, power/power_base=%.1e'%(p_harmonic_min, power_ratio))
        plt.legend(loc='lower right')
        #plt.savefig(figdir+plot+"_ls.png", dpi=200, bbox_inches="tight")

    return p_harmonic_min, power_ratio


def detrend_by_quarter(data, remove_bad_flux=True, plot=True):
    t_out, f_out, e_out, q_out = np.array([]), np.array([]), np.array([]), np.array([])
    fbase_out = np.array([])
    for q in list(set(data.quarter)):
        idx = data.quarter == q
        if remove_bad_flux:
            idx &= (data.quality == 0)
        t, f, e = np.array(data.time[idx]), np.array(data.flux[idx]), np.array(data.error[idx])

        peak_period, peak_power = harmonic_search(t, f, e, plot=False)
        if peak_period is None:
            mode = 'median'
            filter_period = 3.
        else:
            mode = 'sine'
            filter_period = 0.5 * peak_period

        fbase, _ = detrend_chunk(t, f, filter_period, mode=mode)

        t_out = np.r_[t_out, t]
        f_out = np.r_[f_out, f]
        e_out = np.r_[e_out, e]
        q_out = np.r_[q_out, np.ones_like(t)*q]
        fbase_out = np.r_[fbase_out, fbase]

        if plot:
            plt.figure()
            plt.title("quarter %d\nmode: %s, filter_period: %.1fd"%(q, mode, filter_period))
            plt.plot(t, f, '.', alpha=0.6)
            plt.plot(t, fbase, '-')
            plt.plot(t, f-fbase+np.min(f)-2*np.std(f), '.', alpha=0.4)

    return t_out, f_out, e_out, q_out.astype(int), fbase_out


'''
def harmonic_search(t, y, dy, pmin=1, pmax=10., fap_threshold=1e-2, plot=True, power_threshold=30., figdir="./", save=False):
    ls = LombScargle(t, y, dy)
    freq, power = ls.autopower()
    fidx = (1./pmax < freq) & (freq < 1./pmin)
    freq, power = freq[fidx], power[fidx]

    pfreqs, ppowers = [], []
    idx = freq > 0
    while True:
        powermax = np.max(power[idx])
        imax = np.where(power==powermax)[0][0]
        freqmax = freq[imax]
        fap = ls.false_alarm_probability(powermax)
        if fap < fap_threshold:
            pfreqs.append(freqmax)
            ppowers.append(powermax)
            idx &= ((freq < freqmax*0.8) | (freq > freqmax*1.2))
        else:
            break

    pfreqs = np.array(pfreqs)
    ppowers = np.array(ppowers)
    if np.sum(1./pfreqs < 2) > 2:
        pfreqs = pfreqs[1./pfreqs>2.]
        ppowers = pfreqs[1./pfreqs>2.]
    """
    if not len(pfreqs):
        p_harmonic_min, power_ratio = None, None
    else:
        p_harmonic_min = np.min(1./pfreqs)
        power_harmonic_min = power[np.argmin(np.abs(freq-1./p_harmonic_min))]
        power_base = np.median(power[(0.5<freq)&(freq<1.)])
        #power_ratio = power_harmonic_min/power_base
        power_ratio = np.max(power)/power_base
    """
    power_base = np.median(power[(0.5<freq)&(freq<1.)])
    power_ratio = ppowers/power_base
    idxp = power_ratio > power_threshold
    if np.sum(idxp):
        _idx = np.argmin(1./pfreqs[idxp])
        p_harmonic_min = (1./pfreqs[idxp])[_idx]
        power_ratio = power_ratio[idxp][_idx]
    else:
        p_harmonic_min, power_ratio = None, None

    if plot:
        plt.figure(figsize=(20,6))
        plt.title(str(plot))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(pmin, pmax)
        plt.plot(1./freq[idx], power[idx], '-', alpha=0.8, color='C0', lw=1)
        plt.plot(1./freq[~idx], power[~idx], '.', alpha=0.8, color='salmon', markersize=3)
        plt.axhline(y=ls.false_alarm_level(fap_threshold), color='gray', ls='dashed', label='FAP %.1e%%'%(fap_threshold*100))
        plt.ylabel("Lomb-Scargle power")
        plt.xlabel("period (days)")
        #for _f in pfreqs:
        #    plt.axvline(x=1./_f, color='k', lw=0.5)
        if p_harmonic_min is not None:
            plt.axvline(x=p_harmonic_min, color='k', lw=1,
                        label='Pmin=%.1e, power/power_base=%.1e'%(p_harmonic_min, power_ratio))
        plt.legend(loc='lower right')
        if save:
            plt.savefig(figdir+plot+"_ls.png", dpi=200, bbox_inches="tight")

    return [p_harmonic_min, power_ratio]


def local_std(x, y, nseg=30, dx=1.):
    stds_local = []
    for i in range(nseg):
        xc = np.random.choice(x)
        idx = np.abs(x - xc) < dx
        if np.sum(idx):
            xi, yi = x[idx], y[idx]
            z = np.poly1d(np.polyfit(xi, yi, 1))
            stds_local.append(np.std(yi-z(xi)))
    return np.median(stds_local)


def median_filter_clipping(f, nsigma=5.0):
    fmed = median_filter(f, size=3)
    df = f - fmed
    sigma = np.std(df)
    idx = np.abs(df) < nsigma * sigma
    return idx


def get_chunks(t, _f, e, cadence, dt_threshold=1., df_threshold=0):
    #f = median_filter(_f, size=int((3./cadence)//2 * 2 + 1))
    f = median_filter(_f, size=int((1./cadence)//2 * 2 + 1))
    dt = t[1:]-t[:-1]
    df = f[1:]-f[:-1]
    #if not np.sum(dt>dt_threshold):
    #    return [t], [f], [e]
    #gap = (dt>dt_threshold) & (np.abs(df)>df_threshold)
    gap = (dt>dt_threshold)+(np.abs(df)>df_threshold)
    tnodes = t[:-1][gap]+0.5*dt_threshold
    tnodes = np.insert(tnodes, 0, t[0]-0.5*dt_threshold)
    tnodes = np.insert(tnodes, len(tnodes), t[-1]+0.5*dt_threshold)
    t_ret, f_ret, e_ret = [], [], []
    for i in range(len(tnodes)-1):
        idx = (tnodes[i]<t)&(t<tnodes[i+1])
        t_ret.append(t[idx])
        f_ret.append(_f[idx])
        e_ret.append(e[idx])
    return t_ret, f_ret, e_ret


def clean_chunks(t, f, e, cadence, dt_threshold=0.5, df_threshold=0, flatten=False, plot=False, filter_period=10., df=None, mode='sine', clip=0., n_edge=True):
    tc, fc, ec = get_chunks(t, f, e, cadence, dt_threshold=dt_threshold, df_threshold=df_threshold)
    tclean, fclean, eclean, idxs = [], [], [], []

    tmin, tmax = np.min(tc[0]), np.max(tc[-1])
    if plot:
        #plt.figure(figsize=(20,6))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18,7), sharex=True)
        #fig.suptitle(str(plot))
        ax1.set_title(str(plot))
        ax1.set_ylabel("flux")
        ax2.set_ylabel("filtered flux")
        ax2.set_xlabel("time (days)")
    for i in range(len(tc)):
        #if i==0:
        if not len(tclean):
            lb, lf = 'filter baseline\n(%s, %.2f)'%(mode, filter_period), 'filtered flux'
        else:
            lb, lf = '', ''
        tci, fci, eci = tc[i], fc[i], ec[i]

        if clip > 0:
            idx = median_filter_clipping(fci, nsigma=clip)
            tci, fci, eci = tci[idx], fci[idx], eci[idx]
            ncut = 10
            tci, fci, eci = tci[ncut:-ncut], fci[ncut:-ncut], eci[ncut:-ncut]

        ax1.plot(tci, fci, '.', color='C%d'%i, alpha=0.3, markeredgecolor='none')
        #plt.plot(tci, fbasei, '-', color='gray', label=lb, lw=2)

        if len(tci) < filter_period / cadence:
            continue

        if n_edge:
            nedge = int(0.1*(tci[-1]-tci[0])/cadence)
            #nmed = int((0.5/cadence)//2 * 2 + 1)
            nmed = int((0.25/cadence)//2 * 2 + 1)
            _tci, _fci, _idx = extend_edge(tci, fci, nedge, size=nmed)
            tmin = min(tmin, np.min(_tci))
            tmax = max(tmax, np.max(_tci))

        if mode=='sine':
            df = min(0.3/filter_period, 1./(tci[-1]-tci[0])*0.5)
            X = design_matrix_fourier(_tci, fmax=1./filter_period, df=df)
            _fbasei = irls_fit(_tci, _fci, X)
        elif mode=='median':
            ksize = int((filter_period/cadence)//2 * 2 + 1)
            _fbasei = median_filter(_fci, size=ksize, mode='nearest')
        tci, fci, fbasei = _tci[_idx], _fci[_idx], _fbasei[_idx]

        """
        _fci = fci.copy()
        for j in range(5):
            #fbasei = median_filter(_fci, size=ksize, mode=mmode)
            delta = fci - fbasei
            _idx = np.abs(delta) > np.std(delta)
            _fci[_idx] = fbasei[_idx]
            fbasei = median_filter(_fci, size=ksize, mode=mmode)
            _fci = fci.copy()
        """

        tfilti = tci
        ffilti = fci - fbasei
        efilti = eci
        idx = tfilti**2 > -1

        if plot:
            #_idx = (_tci<np.min(tci))+(_tci>np.max(tci))
            ax1.plot(_tci[~_idx], _fci[~_idx], '.', color='gray', alpha=0.2)
            #plt.plot(_tci[ncut:-ncut], _fbasei[ncut:-ncut], '-', color='gray', ls='dashed')
            ax1.plot(_tci, _fbasei, '-', color='gray', ls='dashed', label=lb)
            #plt.plot(tci, fci, '.', color='C%d'%i, alpha=0.4)
            #plt.plot(tci, fbasei, '-', color='gray', label=lb, lw=2)
            ax2.plot(tfilti, ffilti, 'o', color='tan', alpha=0.3, zorder=-1000)
            f1, f99 = np.percentile(f, [1,99])
            ax1.set_ylim(2*f1, 2*f99)
            #f1, f99 = np.percentile(ffilti, [1,99])
            #ax2.set_ylim(2*f1, 2*f99)
            ax1.set_xlim(tmin, tmax)
            ax1.legend(loc='best')
        tclean.append(tfilti)
        fclean.append(ffilti)
        eclean.append(efilti)
        idxs.append(idx)
    if plot:
        #plt.legend()
        plt.savefig(figdir+plot+"_detrend.png", dpi=200, bbox_inches="tight")
        fig.tight_layout(pad=0.02)
        plt.show()
    if flatten:
        tclean = np.concatenate(tclean)
        fclean = np.concatenate(fclean)
        eclean = np.concatenate(eclean)
        idxs = np.concatenate(idxs)
    return tclean, fclean, eclean, idxs


def clip_and_detrend(x, y, dy, cadence, q=None, df_threshold_num=0.5, dt_threshold=0.5, plot=True, mode='median', filter_period=2, flatten=True, clip=3., gaussianize_signal=True):
    y = y/np.mean(y) - 1
    dy = y/np.mean(y)

    sigma_1d = local_std(x, y)
    if plot:
        print ("# local sigma", sigma_1d)

    if q is not None:
        x_arr = [x[q==_q] for _q in set(q)]
        y_arr = [y[q==_q] for _q in set(q)]
        dy_arr = [dy[q==_q] for _q in set(q)]
    else:
        x_arr, y_arr, dy_arr = [x], [y], [dy]

    xout, yout, dyout, ystdout = np.array([]), np.array([]), np.array([]), np.array([])
    for _x, _y, _dy in zip(x_arr, y_arr, dy_arr):
        try:
            xc, yc, dyc, _ = clean_chunks(_x, _y, _dy, cadence, df_threshold=df_threshold_num*sigma_1d, dt_threshold=dt_threshold, plot=plot, mode=mode, filter_period=filter_period, flatten=flatten, clip=clip)
        except:
            mode = 'median'
            filter_period = 3.
            xc, yc, dyc, _ = clean_chunks(_x, _y, _dy, cadence, df_threshold=df_threshold_num*sigma_1d, dt_threshold=dt_threshold, plot=plot, mode=mode, filter_period=filter_period, flatten=flatten, clip=clip)

        xn, yn, dyn = normalize_lc(xc, yc, dyc)

        if gaussianize_signal:
            noise_params = derive_noise_params(yn, print_res=plot)
            yng = gaussianize(yn, noise_params, plot=plot)
        else:
            yng = yn

        if plot:
            plt.figure(figsize=(18,6))
            plt.title(str(plot))
            plt.plot(xn, yn, '.', markersize=3, label='detrended')
            if gaussianize_signal:
                plt.plot(xn, yng, 'o', markersize=7, mfc='none', lw=1, alpha=0.3, label='Gaussianized')
            plt.legend(loc='best')
            plt.show()

        xout = np.r_[xout, xn]
        yout = np.r_[yout, yng]
        dyout = np.r_[dyout, dyn]
        ystdout = np.r_[ystdout, np.ones_like(xn)*np.nanstd(yc)]

    return xout, yout, dyout, ystdout


def get_detrended_lc(xdata, ydata, dydata, cadence, q=None, power_threshold=30., gaussianize=True, plot=False, filter_period_min=0.1, df_threshold_num=1, dt_threshold=0.5):

    N0 = 90./cadence

    p = harmonic_search(xdata, ydata, dydata, plot=plot, pmin=2.*cadence, fap_threshold=5e-2, power_threshold=power_threshold*np.sqrt(len(xdata)/N0))

    if (p[0] is None):# or (p[1] < power_threshold*np.sqrt(len(xdata)/N0)):
        mode = 'median'
        filter_period = 3.
        #mode = 'sine'
        #filter_period = 8.
    else:
        mode = 'sine'
        filter_period = 0.5 * p[0]

    filter_period = max(filter_period, filter_period_min)

    print ('# detrending method:', mode)
    print ('# filtering period:', filter_period)

    return clip_and_detrend(xdata, ydata, dydata, cadence, q=q, mode=mode, filter_period=filter_period, gaussianize_signal=gaussianize, plot=plot, df_threshold_num=df_threshold_num, dt_threshold=dt_threshold)
'''

#%%
"""
from astropy.timeseries import LombScargle
N = 100000
dt = 0.02
t = np.arange(0, N*dt, dt)
y = np.sin(2*np.pi*t) + np.random.randn(len(t))*0.2

#%%
ls = LombScargle(t, y, 0.2)
freq, power = ls.autopower()
print (np.max(power)/np.std(power))
"""

#%%
"""
filename = "kic8313257_pdc.csv"
filename = "kic3835482_pdc.csv"
filename = "kic8145411_pdc.csv"
filename = "kic3342467_pdc.csv"
filename = "kic11773022_pdc.csv"
xdata, ydata, dydata, q = np.array(pd.read_csv("data/"+filename)[["time", "flux", "error", "quarter"]]).T
seglength = 150

#%%
idxq = q==12+2
idxq = q==6
#idxq = q==12
xdata, ydata, dydata = xdata[idxq], ydata[idxq], dydata[idxq]

#%%
filename = "data/epic212006344_s5.csv"
filename = "data/epic210818897_s4.csv"
xdata, ydata, dydata = np.array(pd.read_csv(filename)[["time", "flux", "error"]]).T
dydata = np.ones_like(dydata)*1e-3
q = -1 * np.ones_like(xdata)
seglength = 8.

#%%
cadence = np.median(np.diff(xdata))
print ("cadence: %.1fmin"%(cadence*1440.))

#%%
plt.plot(xdata, ydata)

#%%
p = harmonic_search(xdata, ydata, dydata)

#%%
print (p)
if p is None or p[1] < 30:
    mode = 'median'
    filter_period = 3.
    #mode = 'sine'
    #filter_period = 8.
else:
    mode = 'sine'
    filter_period = 0.5 * p[0]

#%%
print (mode, filter_period)

#%%
clip_and_detrend(xdata, ydata, dydata, cadence, mode=mode, filter_period=filter_period, gaussianize_signal=False)

#%%
from sklearn import linear_model
def design_matrix_freqs(x, freqs):
    nfou = len(freqs)
    ndata = len(x)
    X = np.zeros((ndata, 2*nfou+1))
    X[:,0] = np.ones(ndata)
    xmed = np.median(x)
    for i in range(nfou):
        _f = freqs[i]
        X[:,1+2*i] = np.cos(2*np.pi*_f*(x-xmed))
        X[:,2+2*i] = np.sin(2*np.pi*_f*(x-xmed))
    return X

def irls_fit(x, y, X, n_itr=10):
    clf = linear_model.LinearRegression(fit_intercept=False)
    sigma = np.std(y)
    delta = np.zeros_like(y)
    for i in range(n_itr):
        w = 1./np.sqrt(sigma**2+delta**2)
        w[np.abs(delta) > sigma] = 0.
        reg = clf.fit(X, y, sample_weight=w)
        ypred = reg.predict(X)
        delta = y - ypred
        sigma = np.std(delta)
    return ypred, w, delta

def design_matrix_fourier(x, df=None, fmax=None):
    if df is None:
        df = 1./(x[-1]-x[0])
    if fmax is None:
        fmax = 0.5/(np.min(np.diff(x)))
    nfou = int(np.round(fmax/df))
    ndata = len(x)
    X = np.zeros((ndata, 2*nfou+1))
    X[:,0] = np.ones(ndata)
    xmed = np.median(x)
    for i in range(nfou):
        _f = df * (i+1)
        X[:,1+2*i] = np.cos(2*np.pi*_f*(x-xmed))
        X[:,2+2*i] = np.sin(2*np.pi*_f*(x-xmed))
    return X

from scipy.ndimage import median_filter
def extend_edge(x, y, nedge):
    xhead = x[0] - (x[1:nedge]-x[0])[::-1]
    yhead = median_filter(y[1:nedge], size=3)[::-1]
    xtail = x[-1] + (x[-1]-x[-1-nedge:-1])[::-1]
    ytail = median_filter(y[-1-nedge:-1], size=3)[::-1]
    yhead, ytail = 2*yhead[-1] - yhead, 2*ytail[0] - ytail
    xret, yret = np.r_[xhead, x, xtail], np.r_[yhead, y, ytail]
    return xret, yret, (xhead[-1]<xret)&(xret<xtail[0])

#%%
nedge = int(0.1*(t[-1]-t[0])/cadence)
tfit, yfit, idx = extend_edge(t, y, nedge)

#%%
plt.plot(t, y, '.', color='C0')
plt.plot(tfit[~idx], yfit[~idx], '.', color='C0', alpha=0.1)

#%%
if not len(pfreqs):
    pfreqs = np.array([1./pmax])
print (1./pfreqs[0])
print (1./np.max(pfreqs))

#%%
#X = design_matrix_fourier(t, pfreqs)
fmax = np.max(pfreqs)*2
#fmax = min(1./3, fmax)
print (1./fmax)

#%%
X = design_matrix_fourier(tfit, fmax=fmax)
#ypred, w, d, fmed = irls_fit(tfit, yfit, X, n_itr=10)
ypred, w, d = irls_fit(tfit, yfit, X, n_itr=10)

#%%
#plt.xlim(2220, 2240)
plt.plot(t, y, '.')
plt.plot(tfit, yfit, '.', alpha=0.3)
plt.plot(tfit, ypred, '-')
#plt.plot(tfit, fmed)
#plt.ylim(0.99, 1.025)
#plt.plot(tfit, w/np.max(w))
#plt.plot(tfit, d+1)

#%%
plt.plot(t, y-ypred[idx], '.')
"""
