
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import jax
import argparse
jax.config.update('jax_enable_x64', True)
from pykepler.query import *
from pykepler.detrend import gpfit_with_mask
plt.rcParams['figure.figsize'] = (10,4)


def get_koi_transit_mask(t, dkoi, duration_margin=2., fixed_half_width=None):
    mask = np.zeros_like(t)
    for i in range(len(dkoi)):
        t0, p, dur = np.array(dkoi.iloc[i][['koi_time0bk', 'koi_period', 'koi_duration']])
        if fixed_half_width is None:
            mask_half_width = 0.5 * dur / 24. * duration_margin
        else:
            mask_half_width = fixed_half_width
        phase = (t - t0) / p
        phase -= np.round(phase)
        idx_in_transit = np.abs(phase) * p < mask_half_width
        mask[idx_in_transit] += 1.
    mask = mask > 0.
    return mask


def get_tcinfo_koi(dkoi, t):
    # transit times expected from linear ephemeris (t0 and period)
    tcs, tranums, plnames = np.array([]), np.array([]), np.array([])
    for i in range(len(dkoi)):
        name, t0, p = np.array(dkoi.iloc[i][['kepoi_name', 'koi_time0bk', 'koi_period']])
        tmin, tmax = np.min(t), np.max(t)
        tnum_min, tnum_max = int((tmin-t0)/p), int((tmax-t0)/p)
        tnum = np.arange(tnum_min, tnum_max+1)
        tc = t0 + p * tnum
        tcs = np.r_[tcs, tc]
        tranums = np.r_[tranums, tnum]
        plnames = np.r_[plnames, [name]*len(tc)]
    return tcs, tranums, plnames


def detrend_by_quarter(data, dkoi, remove_bad_flux=True, plot=True, duration_margin=2.):
    #max_half_duration = np.max(dkoi['koi_duration']) / 24. / 2.

    t_out, f_out, e_out, q_out = np.array([]), np.array([]), np.array([]), np.array([])
    fbase_out = np.array([])
    for q in list(set(data.quarter)):
        idx = data.quarter == q
        if remove_bad_flux:
            idx &= (data.quality == 0)
        t, f, e = np.array(data.time[idx]), np.array(data.flux[idx]), np.array(data.error[idx])

        #mask = get_transit_mask(t, dkoi, max_half_duration)
        mask = get_koi_transit_mask(t, dkoi, duration_margin=duration_margin)
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


#%%
def run(kic, cadence, save_outputs=False, time_window=0.5, minimum_data_frac=0.5):
    # output path
    if save_outputs:
        output_dir = "kic%s"%kic
        if not os.path.exists(output_dir):
            os.system("mkdir %s"%output_dir)
        output_name = output_dir + "/%s"%cadence + "_"

    # data download
    data, lcfile = download_kic(kic, cadence)

    # fetch KOI info
    dkoi = fetch_koi_info(kic)
    if save_outputs:
        dkoi.to_csv(output_dir+"/koiinfo.csv", index=False)

    # GP detrending
    t, f, e, q, fbase = detrend_by_quarter(data, dkoi)
    data_detrend = pd.DataFrame(data={"time": t, "flux": f, "error": e, "quarter": q, "flux_base": fbase})
    if save_outputs:
        data_detrend.to_csv(output_name+"detrend.csv", index=False)

    # check phase-folded light curves
    spath = output_name+"fold" if save_outputs else None
    _ = check_koi_ephemeris(kic, t, f-fbase, save_path=spath)

    # extract data around transits
    expected_num_points = 2 * time_window / np.median(np.diff(t))
    tcs, tranums, plnames = get_tcinfo_koi(dkoi, t)
    t_out, f_out, e_out = np.array([]), np.array([]), np.array([])
    for tc, plname, tranum in zip(tcs, plnames, tranums):
        idx = np.abs(t - tc) < time_window
        num_points = np.sum(idx)
        if num_points < minimum_data_frac * expected_num_points:
            continue

        t_out = np.r_[t_out, t[idx]]
        f_out = np.r_[f_out, (f-fbase)[idx]]
        e_out = np.r_[e_out, e[idx]]
        plt.figure()
        plt.xlabel("time")
        plt.ylabel("flux")
        plt.title("planet %s, transit %d, available fraction: %.2f"%(plname, tranum, num_points/expected_num_points))
        plt.plot(t[idx], f[idx], ".", label='raw flux')
        plt.plot(t[idx], fbase[idx], "-", lw=1, label='baseline')
        plt.plot(t[idx], (f-fbase)[idx] + np.min(f[idx]), '.', label='detrended flux')
        plt.legend(loc='best')

    # save outputs
    dout = pd.DataFrame(data={"time": t_out, "flux": f_out, "error": e_out})
    print (np.sum(dout.duplicated("time")), "overlapping points.")
    dout = dout.drop_duplicates("time", keep='first').sort_values("time").reset_index(drop=True)
    if save_outputs:
        dout.to_csv(output_name+"detrend_transits.csv", index=False)


def get_arguments():
    parser = argparse.ArgumentParser(description='download and process kepler lightcurve')
    parser.add_argument('-kic', metavar='KIC ID', required=True, help='KIC ID')
    parser.add_argument('-cadence', metavar='cadence', required=True, help='cadence (long or short)')
    parser.add_argument('-save_outputs', metavar='save_outputs', nargs=1, required=False, default=False, help='save')
    parser.add_argument('-time_window', metavar='time_window', default=0.5, type=float, help='time_window')
    parser.add_argument('-min_data_frac', metavar='min_data_frac', default=0.5, type=float, help='min_data_frac')
    return parser.parse_args()

#%%
if __name__ == '__main__':
    args = get_arguments()
    run(int(args.kic), args.cadence, args.save_outputs, args.time_window, args.min_data_frac)

    #"""
    kic = 11773022 # Kepler-51
    cadence = "long" # or "short"
    save_outputs = True * 0
    time_window = 0.5  # extract the data where |t-tc| < time_window
    minimum_data_frac = 0.5 # ignore transits with many missing data points

    run(kic, cadence, save_outputs, time_window, minimum_data_frac)
    #"""
