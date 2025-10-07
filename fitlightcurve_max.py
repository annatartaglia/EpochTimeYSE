'''
    Provides functions and classes for MCMC estimation of max light epoch
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import autograd
import corner
import emcee
from chainconsumer import ChainConsumer
import pickle

# Store global constants
passbands = ['g', 'r', 'i', 'z', 'ZTF_g', 'ZTF_r']
#change this:
pbcols = {
    'g': 'tab:cyan',
    'r': 'tab:red',
    'i': 'tab:purple',
    'z': 'k',
    'ZTF_g': 'tab:blue',
    'ZTF_r': 'tab:orange'
}
param_names = ['$t_0$', '$\gamma$', r'$\tau_{\mathrm{fall}}$', r'$\tau_{\mathrm{rise}}$', '$A$', r'$\beta$']
npar = len(param_names)

# load in population means and stdevs (per each parameter, per each passband) for prior
filename = "dists.pkl"
with open(filename, "rb") as file:
    dists = pickle.load(file)

def isdetection(flux, err):
    return flux/err>5

def sigmoid(t, np1 = np):
    return 1/(1+np1.exp(-t))

def get_y_pred(params, t_all, np1= np):
    t0, gamma, tau_fall, tau_rise, A, beta = params

    fearly = A * (1-(beta*(t_all-t0)/gamma)) / (1 + np1.exp(-(t_all-t0)/tau_rise))

    s = 0.2
    g = sigmoid(s*(t_all - (gamma+t0)), np1)

    flate = A*((1-beta) * np1.exp(-(t_all - (gamma+t0))/tau_fall)) / (1+np1.exp(-(t_all-t0)/tau_rise))

    y_pred = fearly * (1-g) + flate * g

    return y_pred

def log_likelihood(t_data, y_data, y_data_err, params, fit_end_time, np1=np):

    # Only use values until fit_end_time
    mask_time = (t_data < fit_end_time)
    t_data = t_data[mask_time]
    y_data = y_data[mask_time]
    y_data_err = y_data_err[mask_time]

    y_pred = get_y_pred(params, t_data, np1)

    mse = -0.5 * (y_data - y_pred)**2 / (y_data_err**2)
    sigma_trace = -0.5 * np1.log(y_data_err**2)
    log2pi = -0.5 * np1.log(2 * np1.pi)
    logL = np1.sum(mse + sigma_trace + log2pi)
    if np1.isnan(logL):
        logL = -np1.inf

    return logL

def log_prior(theta, mu, sigma, lastnondet = None, np1 = np):
    '''
        theta: [t0, gamma, tau_fall, tau_rise, A, beta]
        mu: 6d array of population parameter means [mu_t0, mu_gamma,...]
        sigma: 6d array of population parameter sigmas [sigma_t0, sigma_gamma,...]

        returns log probability of multivariate gaussian prior
    '''
    t0, gamma, tau_fall, tau_rise, A, beta = theta

    if lastnondet:
        if t0<lastnondet:
            return -np1.inf
    if tau_rise < 1 or tau_rise > 50:
        return -np1.inf
    elif tau_fall < 1 or tau_fall > 130:
        return -np1.inf
    elif beta < 0 or beta > 1:
        return -np1.inf
    elif gamma < 1 or gamma > 120:
        return -np.inf
    elif A < 0:
        return -np1.inf

    var_inv = 1.0 / (sigma ** 2)  # Inverse of variances (diagonal of Σ⁻¹)
    log_det_cov = np1.sum(np1.log(sigma ** 2))  # Log-determinant of diagonal Σ
    diff = theta - mu

    return -0.5 * (np1.sum(diff**2 * var_inv) + log_det_cov + len(mu) * np1.log(2 * np1.pi))

def log_posterior(t_data, y_data, y_data_err, params, mu, sigma, fit_end_time, lastnondet = None, np1=np):
    logL = log_likelihood(t_data, y_data, y_data_err, params, fit_end_time, np1)
    logprior = log_prior(params, mu, sigma, lastnondet, np1)

    return logL + logprior


class FitLightCurve:
    def __init__(self, lc, passbands, dists, fit_end_time, showProgress=False):
        self.lc = lc
        self.passbands = passbands
        self.fit_end_time = fit_end_time
        self.t_data, self.y_data, self.y_data_err, self.t_all = self.get_data()
        self.autodiff_numpy = False
        self.x0 = np.array([10, 50, 40, 15, 100, 0.5])
        self.dists = dists

        self.showProgress = showProgress

    def get_data(self):
        t_data, y_data, y_data_err = {}, {}, {}
        for pb in self.passbands:
            if np.all(np.isnan(self.lc[f'{pb}_flux'])):
                continue
            mask_nans = ~self.lc[f'{pb}_flux'].isnull()
            t_data[pb] = self.lc['relative_time'][mask_nans].values
            y_data[pb] = self.lc[f'{pb}_flux'][mask_nans].values
            y_data_err[pb] = self.lc[f'{pb}_uncert'][mask_nans].values
        self.passbands = y_data.keys()

        self.lastnondet = min(
            t_data[pb][np.where(isdetection(y_data[pb], y_data_err[pb]))[0][0] - 1]
            for pb in self.passbands
            if np.any(isdetection(y_data[pb], y_data_err[pb]))
        )

        self.x0 = np.array([self.lastnondet, 50, 40, 15, 100, 0.5])

        self.max = max((t_data[pb][i]
                        for pb in self.passbands
                        if len(y_data[pb]) > 0
                        and (i := np.argmax(y_data[pb])) is not None
                        and isdetection(y_data[pb][i], y_data_err[pb][i])
                        ), default=np.nan)
        t_all = self.lc['relative_time'].values

        return t_data, y_data, y_data_err, t_all

    def objective_func_opt(self, params, pb):
        if self.autodiff_numpy:
            np1 = autograd.numpy
        else:
            np1 = np
        mu, sigma = self.dists[pb]
        return -log_posterior(self.t_data[pb], self.y_data[pb], self.y_data_err[pb], params, mu, sigma, self.fit_end_time, self.lastnondet, np1=np1)

    def run_optimizer(self):
        self.autodiff_numpy = False
        opt_params, obj = {}, {}
        for pb in self.passbands:
            opt_params[pb] = minimize(self.objective_func_opt, self.x0, args = (pb), method='Nelder-Mead', options={'xatol': 1e-12, 'disp': False}).x
            obj[pb] = self.objective_func_opt(opt_params[pb], pb)/len(self.y_data[pb]) if len(self.y_data[pb]) > 4 else np.inf
        
        self.x0 = opt_params[min(obj, key=obj.get)]

        for pb in self.passbands:
            opt_params[pb] = minimize(self.objective_func_opt, self.x0, args = (pb), method='Nelder-Mead', options={'xatol': 1e-12, 'disp': False}).x
        return opt_params

    def check_enough_data(self, pb, npoints=6, length = 30):
        # Check that there are at least n points between t0 and maximum
        window = (self.t_data[pb] >= self.max-length) & (self.t_data[pb] <= self.max+length)
        num_points = np.sum(~np.isnan(self.y_data[pb][window]))
        return num_points >= npoints

    def objective_func_mcmc(self, params, pb):
        mu, sigma = self.dists[pb]
        return log_posterior(self.t_data[pb], self.y_data[pb], self.y_data_err[pb], params, mu, sigma, self.fit_end_time, self.lastnondet)

    def get_mcmc_samples(self, params, fig_corner={}, nwalkers=100, nsteps=1000, nburn=200, parallelize=False, plot_corner = False):
        samples = {}
        
        for pb in self.passbands:
            if pb not in fig_corner.keys():
                fig_corner[pb] = None
            opt_params = params[pb]
            pos = np.nan_to_num(opt_params) + 1e-4 * np.random.randn(nwalkers, len(opt_params))
            nwalkers, ndim = pos.shape

            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.objective_func_mcmc, args = {pb})
            sampler.run_mcmc(pos, nsteps=nsteps, progress=self.showProgress);
            
            samples[pb] = sampler.get_chain(discard=nburn, flat=True)
        
            param_meds = np.median(samples[pb], axis=0)
            if plot_corner:
                fig_corner[pb] = corner.corner(samples[pb], labels=param_names, truths=param_meds, fig=fig_corner[pb],
                                   color='tab:blue', truth_color='black', label="MCMC Samples");
        return samples, fig_corner
    
    def good_fit(self, samples, pb, use_prod = True, min_pts = 10):
        num_pts = len(self.y_data[pb])
        if num_pts < min_pts:
            return False
        obj = self.objective_func_opt(np.median(samples[pb], axis=0), pb)/num_pts
        prod = np.prod(np.std(samples[pb], axis=0))
        if obj > 12:
            return False
        if use_prod and prod > 2000000:
            return False
        if not np.any((self.t_data[pb] > self.lastnondet-15) & (self.t_data[pb] < self.lastnondet+15)):
            return False
        return True

    def make_chainconsumer_plot(self, mcmc_samples, laplace_samples, opt_params, param_names, save_name):
        c = ChainConsumer()
        if mcmc_samples is not None:
            c.add_chain(mcmc_samples, parameters=param_names, name='MCMC Samples', color='#1f77b4')
        if laplace_samples is not None:
            c.add_chain(laplace_samples, parameters=param_names, name='Laplace Approximation', color='#9467bd')
        fig = c.plotter.plot(truth=opt_params)
        fig.savefig(save_name)
        return fig
    

def fit_light_curve(lc, fit_end_time = 5000):
    """ 
    Fit one light curve using optimizer 
    fit_end_time should only be changed for testing. default is entire LC 
    """

    fitlc = FitLightCurve(lc, passbands, dists, fit_end_time)
    opt_params = fitlc.run_optimizer()


    #Run MCMC twice, using first fit as a prior for the second
    og_mcmc_samples, og_fig_corner = fitlc.get_mcmc_samples(opt_params) # first run
    try:
        all_samples = np.vstack([og_mcmc_samples[pb]
                    for pb in og_mcmc_samples
                    if fitlc.good_fit(og_mcmc_samples, pb)])
    except ValueError:
        all_samples = np.vstack([og_mcmc_samples[pb]
                    for pb in og_mcmc_samples
                    ])
    fitlc.dists = {pb: [np.mean((all_samples), axis = 0), np.std((all_samples), axis = 0)] for pb in fitlc.passbands}
    
    mcmc_samples, fig_corner = fitlc.get_mcmc_samples(opt_params) # second run

    t_maxes = []
    for pb in fitlc.passbands:
        if fitlc.check_enough_data(pb):
            chosen_indices = np.random.choice(len(mcmc_samples[pb]), 100, replace=False)
            for idx in chosen_indices:
                param = mcmc_samples[pb][idx]
                t_maxes += [fitlc.t_all[np.argmax(get_y_pred(param, fitlc.t_all))]]
    max_light, stdev = np.mean(t_maxes), np.std(t_maxes)

    return opt_params, mcmc_samples, fig_corner, max_light, stdev
