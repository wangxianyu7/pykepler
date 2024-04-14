""" detrending with GP """

__all__ = ["gpfit_with_mask"]

import numpy as np
import jax.numpy as jnp
import jaxopt
import celerite2
from celerite2.jax import terms as jax_terms
from jax import config
config.update('jax_enable_x64', True)


def get_gpkernel(p, kernel='matern'):
    if kernel=='matern':
        ret = jax_terms.Matern32Term(sigma=jnp.exp(p['lna']), rho=jnp.exp(p['lnc']))
    elif kernel=='shoterm':
        ret = jax_terms.SHOTerm(sigma=jnp.exp(p['lna']), rho=p['rho'], tau=jnp.exp(p['lntau']))
    return ret


def gpfit_with_mask(t, f, e, mask=None, t_pred=None, method="TNC", kernel='matern', p_guess=None):
    dt, T = np.median(np.diff(t)), np.max(t) - np.min(t)
    f_mean, f_std, e_mean = np.mean(f), np.std(f), np.mean(e)
    if mask is None:
        e_mask = np.zeros_like(t)
    else:
        e_mask = np.array(mask) * 1e6 * f_std

    if kernel=='matern':
        p_init = {"mean": f_mean, "lnjitter": np.log(e_mean), "lna": np.log(f_std), "lnc": np.log(T)}
        p_low = {"mean": f_mean - f_std, "lnjitter": np.log(e_mean/100.), "lna": np.log(f_std/10.), "lnc": np.log(dt)}
        p_upp = {"mean": f_mean + f_std, "lnjitter": np.log(f_std), "lna": np.log(f_std*10.), "lnc": np.log(2*T)}
    elif kernel=='shoterm':
        if p_guess is None:
            p_guess = 5.
        p_init = {"mean": f_mean, "lnjitter": np.log(e_mean),
                  "lna": np.log(f_std), "lntau": np.log(T), "rho": np.float64(p_guess)}
        p_low = {"mean": f_mean - f_std, "lnjitter": np.log(e_mean/100.),
                 "lna": np.log(f_std/10.), "lntau": np.log(dt), "rho": np.float64(p_guess / 2.)}
        p_upp = {"mean": f_mean + f_std, "lnjitter": np.log(f_std),
                 "lna": np.log(f_std*10.), "lntau": np.log(2*T), "rho": np.float64(p_guess * 2.)}

    def objective(p):
        res = f - p['mean']
        gpkernel = get_gpkernel(p, kernel=kernel)
        gp = celerite2.jax.GaussianProcess(gpkernel, mean=0.0)
        gp.compute(t, diag=e**2 + jnp.exp(2*p['lnjitter']) + e_mask**2)
        return -2 * gp.log_likelihood(res)

    def predict(t, f, e, p, t_pred=None):
        res = f - p['mean']
        gpkernel = get_gpkernel(p, kernel=kernel)
        gp = celerite2.jax.GaussianProcess(gpkernel, mean=0.0)
        gp.compute(t, diag=e**2 + jnp.exp(2*p['lnjitter']) + e_mask**2)
        if t_pred is None:
            t = t_pred
        return gp.predict(res, t=t_pred) + p['mean']

    solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)
    res = solver.run(p_init, bounds=(p_low, p_upp))
    p_optim = res[0]
    #print (res)

    if t_pred is None:
        t_pred = t

    return predict(t, f, e, p_optim, t_pred=t_pred), p_optim
