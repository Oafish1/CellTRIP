import numpy as np
import scipy.optimize
import scipy.stats

from . import general as _general


def compute_sim_bio_mapping(
    vel_density,
    time,
    start_val=0,
    end_val=1,
    return_pdf=False,
    return_params=False,
    maxfev=10_000,
):
    "Compute mapping function from simulated to biological time based on environment velocity"
    # Get base mean cumulative density
    # vel_density = states[..., manager.dim:].mean(dim=-1)  # We take the mean since velocities in the env space are not normalized
    # vel_density = vel_density.mean(dim=-1)  # Compute mean over all cells for more accurate density estimation
    # vel_density = vel.abs().mean(dim=(-1, -2))

    # Gamma decay fit
    def gamma_dist(x, k, loc, xscale, yscale, cdf=False):
        f = scipy.stats.gamma.cdf if cdf else scipy.stats.gamma.pdf
        return yscale * f(x / xscale, k, loc=loc)
    gamma_params, _ = scipy.optimize.curve_fit(
        gamma_dist, time, vel_density, p0=[1, 0, 30, .1],
        bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=maxfev)
    
    # Scaled CDF, omit yscale
    ret = (lambda x: (end_val - start_val) * gamma_dist(x, *gamma_params[:-1], 1, cdf=True) + start_val,)
    # PDF
    if return_pdf:
        ret += (lambda x: gamma_dist(x, *gamma_params),)
    # Params
    if return_params:
        ret += (gamma_params,)

    return _general.clean_return(ret)

    # Log normal fit
    # def lognorm_dist()

    # Gamma decay fit
    # def gamma_dist(x, k, theta, yscale):
    #     return yscale * scipy.stats.gamma.pdf(x, k, scale=theta)
    # gamma_params, _ = scipy.optimize.curve_fit(gamma_dist, time, vel_density, p0=[2, 2, sum(vel_density)], bounds=([0, 0, 0], np.inf))
    
    # return lambda x: (end_val - start_val) * scipy.stats.gamma.cdf(x, *gamma_params) + start_val




def find_target_bio_time(map_func, target_time, max_iter=1000, tol=1e-5):
    "Find simulation time corresponding to a target biological time"
    # Use exponential search to find an upper bound
    low, high = 0, 1
    for _ in range(max_iter):
        if map_func(high) >= target_time: break
        low, high = high, high * 2

    # Use binary search on interval
    for _ in range(max_iter):
        mid = (low + high) / 2
        if abs(map_func(mid) - target_time) < tol or low >= high:
            return mid
        if map_func(mid) < target_time:
            low = mid
        else:
            high = mid

    return mid
