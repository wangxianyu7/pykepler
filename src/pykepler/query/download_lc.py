__all__ = ["download_kic"]

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import pandas as pd
import argparse, os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def lcc2df(lcc, remove_nan=True, quarter_normalize=True):
    """ convert LightCurveCollection to pandas DataFrame

        Args:
            lcc: list of LightCurveCollection objects
            remove_nan: remove nan in time, flux, and error if True
            quarter_normalize: normalize flux and error by median flux (on a quarter-by-quarter basis)

        Returns:
            dataframe containing
            {time, PDCSAP flux, PDCSAP error, quality flag, quarter}

    """
    quarters = lcc.quarter
    df_out = pd.DataFrame({})
    for lc, q in zip(lcc, quarters):
        t, f, e = lc.time.value, np.array(lc.pdcsap_flux).astype(np.float64), np.array(lc.pdcsap_flux_err).astype(np.float64)
        qual = np.array(lc.quality)
        if remove_nan:
            idx = (t==t) * (f==f) * (e==e)
            t, f, e, qual = t[idx], f[idx], e[idx], qual[idx]
        if quarter_normalize:
            fmed = np.nanmedian(f)
            f, e = f / fmed, e / fmed
        df = pd.DataFrame(data={"time": t, "flux": f, "error": e, "quality": qual})
        df['quarter'] = q
        # df_out = df_out.append(df, ignore_index=True)
        df_out = pd.concat([df_out, df], ignore_index=True)
    return df_out.sort_values("time").reset_index(drop=True)


def get_arguments():
    parser = argparse.ArgumentParser(description='download kepler lightcurve')
    parser.add_argument('-kic', metavar='KIC ID', nargs=1, required=True, help='KIC ID')
    parser.add_argument('-cadence', metavar='cadence', nargs=1, required=True, help='cadence (long or short)')
    return parser.parse_args()

exptimes = {"long": 1800, "short": 60}


def download_kic(kic, cadence):
    result_all = lk.search_lightcurve('KIC %d'%kic, author='Kepler')
    print (result_all)

    result_long = result_all[np.array(result_all.exptime) == exptimes[cadence]]
    lc_data = result_long.download_all()
    print ()
    print (lc_data)

    df = lcc2df(lc_data)
    output_dir = "kic%s"%kic
    if not os.path.exists(output_dir):
        os.system("mkdir %s"%output_dir)
    datapath = output_dir+"/kic%s_%s.csv"%(kic, cadence)
    df.to_csv(datapath, index=False)

    plt.xlabel("time")
    plt.ylabel("normalized flux")
    plt.plot(df.time, df.flux, '.', markersize=3)
    plt.show()

    return df, datapath


if __name__ == '__main__':
    args = get_arguments()
    kic, cadence = int(args.kic[0]), args.cadence[0]
    download_kic(kic, cadence)

    """
    result_all = lk.search_lightcurve('KIC %d'%kic, author='Kepler')
    print (result_all)

    result_long = result_all[np.array(result_all.exptime) == exptimes[cadence]]
    lc_data = result_long.download_all()
    print ()
    print (lc_data)

    df = lcc2df(lc_data)
    output_dir = "kic%s"%kic
    if not os.path.exists(output_dir):
        os.system("mkdir %s"%output_dir)
    df.to_csv(output_dir+"/kic%s_%s.csv"%(kic, cadence), index=False)

    plt.xlabel("time")
    plt.ylabel("normalized flux")
    plt.plot(df.time, df.flux, '.', markersize=3)
    plt.show()
    """
