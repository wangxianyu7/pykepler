
__all__ = ["derive_noise_params", "gaussianize"]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nct, norm # dof, non-centrality, location, scale
from scipy.optimize import minimize, brute


def normalize_lc(x, y, dy):
    ymean, ystd = np.nanmean(y), np.nanstd(y)
    return x, (y-ymean)/ystd, dy/ystd


def noise_pdf(x, noise_params):
    mu, sigma, a, b, c, nu, psi = noise_params
    tail = nct(nu, psi, loc=b, scale=c)
    gauss = np.exp(-0.5*(x-mu)**2/sigma**2)/np.sqrt(2*np.pi)/sigma
    return (1-a)*gauss, a*tail.pdf(x)
    #gauss = norm(loc=mu, scale=sigma)
    #return (1-a)*gauss.pdf(x), a*tail.pdf(x)


def gaussianize_1d(y, noise_params):
    mu, sigma, a, b, c, nu, psi = noise_params
    tail = nct(nu, psi, loc=b, scale=c)
    gauss = norm(loc=mu, scale=sigma)
    cdf_y = (1-a)*gauss.cdf(y) + a*tail.cdf(y)
    gauss0 = norm()
    return gauss0.ppf(cdf_y)


def gaussianize(y, noise_params, plot=False):
    yg_1d = gaussianize_1d(y, noise_params)
    g, t = noise_pdf(y, noise_params)
    prob_gauss = g / (g + t)
    prob_gauss[1:-1] = 1. - prob_gauss[2:] * prob_gauss[:-2]
    yg = prob_gauss * y + (1. - prob_gauss) * yg_1d

    if plot:
        bins = np.linspace(-10, 20, 100)
        plt.figure(figsize=(12,5), facecolor='white')
        plt.xlabel("normalized flux")
        plt.ylabel("probability density")
        plt.yscale("log")
        plt.hist(y, bins=bins, density=True, histtype='step', lw=1)
        plt.ylim(1e-5, 1)
        _g, _t = noise_pdf(bins, noise_params)
        plt.plot(bins, _g+_t, lw=1, label='normal + non-central t')
        plt.plot(bins, _g, lw=1, ls='dotted', label='normal')
        plt.legend(loc='upper right')

        plt.figure(figsize=(12,5), facecolor='white')
        plt.xlabel("normalized flux")
        plt.ylabel("Gaussianized flux")
        idx = np.argsort(y)
        x0 = np.linspace(min(yg), max(yg), 1000)
        plt.plot(y[idx], yg[idx], '.')
        #plt.plot(x0, x0, lw=1, color='gray')
    return yg


def m2loglike_noise(p, y):
    params = noise_params.copy()
    mu, sig, loga, b = p
    params[:4] = [mu, sig, 10**loga, b]
    return -2*np.sum(np.log(np.sum(noise_pdf(y, params), axis=0)))


def derive_noise_params(y, print_res=True):
    res = minimize(m2loglike_noise, [0, 1, -3, 0], method="L-BFGS-B",
        #bounds=((-0.1,0.1), (0.5,1.5), (-4,-1), (-3,3)),
        bounds=((-0.1,0.1), (0.8,1.2), (-4,-1), (-3,3)),
        args=y)
    if print_res:
        print (res)
    p = noise_params.copy()
    p[:4] = res.x
    p[2] = 10**p[2]
    return p

"""
def derive_noise_params(y, print_res=True):
    func = lambda x: m2loglike_noise(np.r_[[0,1], x], y)
    res = minimize(func, [-3,0], method="L-BFGS-B", bounds=((-4,-1), (-3,3)))
    if print_res:
        print (res)
    p = noise_params.copy()
    p[2:4] = res.x
    p[2] = 10**p[2]
    return p

noise_params = [0, 1, 1e-2, 0, 1, 1, 1]
"""

#%%
#noise_params = [0, 1, 1e-2, 1.60, 0.86, 1.94, 0.75] # Robnik
