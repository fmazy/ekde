import numpy as np

import ekde.ekdefunc
from ekde.ekdefunc import merge as ekdefunc_merge
import hyperclip.hyperfunc
from hyperclip import Hyperplane
from .whitening_transformer import WhiteningTransformer
import pandas as pd
from scipy.special import gamma, factorial2
from scipy.stats import norm
from joblib import Parallel, delayed, dump, load
import time
import os
import shutil
from joblib import wrap_non_picklable_objects

kernels_id = {'box' : 0,
              'gaussian' : 1}

def discretize(X, x_min, dx):
    Z = ((X - x_min) / dx).astype(int)
    return(Z)

def compute_centers(Z, x_min, dx):
    C = x_min + Z * dx + 0.5 * dx
    return(C)

class KDE():
    def __init__(self, 
                 h='scott', 
                 kernel='box',
                 q=11, 
                 bounds=[],
                 n_jobs = 1,
                 verbose=0):
        if q%2 == 0:
            raise(ValueError("Unexpected q value. q should be an odd int."))
            
        self.h = h
        self.kernel = kernel
        self._h = None
        self.q = q
        self.bounds = bounds
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def fit(self, X):
        if self.verbose > 0:
            print('KDE fit process, X.shape=', X.shape)
        # preprocessing
        if len(X.shape) == 1:
            X = X[:,None]
        
        self._real_X_min = np.min(X, axis=0)
        self._real_X_max = np.max(X, axis=0)
        
        # get data dimensions
        self._n, self._d = X.shape
        
        # preprocessing
        self._wt = WhiteningTransformer()
        X = self._wt.fit_transform(X)
        
        self._x_min = np.min(X, axis=0)
        
        # BOUNDARIES INFORMATIONS
        A, R = self._set_boundaries(x = X[0])
        self.A = A
        self.R = R
        # BANDWIDTH SELECTION
        self._compute_bandwidth(X)
        
        self._compute_support()
        
        self._x_min = X.min(axis=0)
        self._dx = self._h * self._support / self.q
        
        Z = discretize(X, self._x_min, dx=self._dx).astype(np.intc)
        Z = pd.DataFrame(Z)
        
        U_nu = Z.groupby(by=Z.columns.tolist()).size().reset_index(name="nu")
        
        self._U = U_nu[Z.columns.tolist()].values.astype(np.intc)
        self._nu = U_nu["nu"].values.astype(np.double)
                
        self._U_diff_asc = np.ones((self._U.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_asc(self._U, self._U_diff_asc)
        
        self._U_diff_desc = np.ones((self._U.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_desc(self._U, self._U_diff_desc)
        
        C = compute_centers(self._U, self._x_min, self._dx).astype(np.double)
        self._nu /= np.array(hyperclip.hyperfunc.volumes(A, R, C, self._h))
        
        self._compute_normalization()
        
        return(self)
        
    def predict(self, X):
        X = self._wt.transform(X)
        
        id_out_of_bounds = np.zeros(X.shape[0]).astype(np.bool)
        for hyp in self._bounds_hyperplanes:
            id_out_of_bounds = np.any((id_out_of_bounds, ~hyp.side(X)), axis=0)
        Z = discretize(X, self._x_min, dx=self._dx)
        # sort Z
        Z = pd.DataFrame(Z)
        Z['j'] = np.arange(Z.shape[0])
        Z = Z.sort_values(by=[i for i in range(self._d)])
        Z_indices = Z['j'].values.astype(np.intc)
        Z = Z[[i for i in range(self._d)]].values.astype(np.intc)
        
        Z_diff_asc = np.ones((Z.shape[0], self._d), dtype=np.intc)
        ekde.ekdefunc.count_diff_asc(Z, Z_diff_asc)
        
        id_Z_unique = np.where(Z_diff_asc[:, self._d-1] == 1)[0]
        
        g = np.array(ekde.ekdefunc.merge(U=self._U,
                                          U_diff_asc = self._U_diff_asc,
                                          U_diff_desc = self._U_diff_desc,
                                          nu=self._nu,
                                          Z=Z[id_Z_unique],
                                          q=self.q,
                                          h=self._h,
                                          kernel_id=kernels_id[self.kernel],
                                          dx=self._dx))
        
        # folder = './.temp_joblib_memmap'
        # try:
        #     os.mkdir(folder)
        # except FileExistsError:
        #     pass
        
        # Z_filename_memmap = os.path.join(folder, 'Z_memmap')
        # dump(Z, Z_filename_memmap)
        # Z_memmap = load(Z_filename_memmap, mmap_mode='r')
        
        # U_filename_memmap = os.path.join(folder, 'U_memmap')
        # dump(self._U, U_filename_memmap)
        # U_memmap = load(U_filename_memmap, mmap_mode='r')
        
        # U_diff_asc_filename_memmap = os.path.join(folder, 'U_diff_asc_memmap')
        # dump(self._U_diff_asc, U_diff_asc_filename_memmap)
        # U_diff_asc_memmap = load(U_diff_asc_filename_memmap, mmap_mode='r')
        
        # U_diff_desc_filename_memmap = os.path.join(folder, 'U_diff_desc_memmap')
        # dump(self._U_diff_desc, U_diff_desc_filename_memmap)
        # U_diff_desc_memmap = load(U_diff_desc_filename_memmap, mmap_mode='r')
        
        # nu_filename_memmap = os.path.join(folder, 'nu_memmap')
        # dump(self._nu, nu_filename_memmap)
        # nu_memmap = load(nu_filename_memmap, mmap_mode='r')
        
        # streams = [int(id_Z_unique.size / self.n_jobs) * i for i in range(self.n_jobs + 1)]
        # streams.append(id_Z_unique.size)
        
        
        # g = Parallel(n_jobs=self.n_jobs,
        #              backend="threading", verbose=5)(delayed(testfunc)(
        #     U_memmap,
        #     U_diff_asc_memmap,
        #     U_diff_desc_memmap,
        #     nu_memmap,
        #     Z_memmap[id_Z_unique[streams[i_stream]: streams[i_stream+1]]],
        #     self.q,
        #     self._h,
        #     kernels_id[self.kernel],
        #     self._dx) for i_stream in range(self.n_jobs + 1))
        
        
        
        
        
        # try:
        #     shutil.rmtree(folder)
        # except:  # noqa
        #     print('Could not clean-up automatically.')
        
        # g = np.hstack([np.array(gi) for gi in g])
        
        # return(g)
    
        f = np.array(ekde.ekdefunc.set_estimation(Z_diff_asc=Z_diff_asc,
                                                  Z_indices=Z_indices,
                                                  g=g))
        
        f[id_out_of_bounds] = 0.0
        
        f = f / self._normalization
        
        f /= self._wt.scale_
        return(f)
    

    
    def _set_boundaries(self, x):
        self._bounds_hyperplanes = []

        for k, pos in self.bounds:
            if pos == 'left':
                self._add_boundary(k=k,
                                   value=self._real_X_min[k],
                                   x=x)
            elif pos == 'right':
                self._add_boundary(k=k,
                                   value=self._real_X_max[k],
                                   x=x)
            elif pos == 'both':
                self._add_boundary(k=k,
                                   value=self._real_X_min[k],
                                   x=x)
                self._add_boundary(k=k,
                                   value=self._real_X_max[k],
                                   x=x)
            else:
                raise(TypeError('Unexpected bounds parameters'))
        
        A = np.zeros((self._d, len(self._bounds_hyperplanes)))
        R = np.zeros(len(self._bounds_hyperplanes))
        
        for i_hyp, hyp in enumerate(self._bounds_hyperplanes):
            A[:, i_hyp] = hyp.a
            R[i_hyp] = hyp.r
        
        return(A, R)
    
    def _add_boundary(self, k, value, x):
        P = np.diag(np.ones(self._d))
        
        P[:, k] = value

        P_wt = self._wt.transform(P)

        hyp = Hyperplane().set_by_points(P_wt)
        hyp.set_positive_side(x)
        self._bounds_hyperplanes.append(hyp)
    
    def _compute_bandwidth(self, X):
        if type(self.h) is int or type(self.h) is float:
            self._h = float(self.h)

        elif type(self.h) is str:
            if self.h == 'scott' or self.h == 'silverman':
                # the scott rule is based on gaussian kernel
                
                self._h = scotts_rule(X)
                
                if self.kernel == 'box':
                    # the support of the gaussian kernel to have 99%
                    # of the density is 2.576
                    self._h *= 2.576
            else:
                raise (ValueError("Unexpected bandwidth selection method."))
        else:
            raise (TypeError("Unexpected bandwidth type."))

        if self.verbose > 0:
            print('Bandwidth selection done : h=' + str(self._h))
    
    def _compute_support(self):
        if self.kernel == 'box':
            self._support = 1.0
        elif self.kernel == 'gaussian':
            self._support = 2.576
    
    def _compute_normalization(self):
        if self.kernel == 'box':
            self._normalization = self._h ** self._d * self._n
        elif self.kernel == 'gaussian':
            # self._normalization = self._d * gauss_integral(self._d - 1) * volume_unit_ball(self._d, p=2) * self._n
            self._normalization = (2 * np.pi)**(self._d/2) * self._h**self._d * self._n
        
            
    def set_params(self, **params):
        """
        Set parameters.

        Parameters
        ----------
        **params : kwargs
            Parameters et values to set.

        Returns
        -------
        self : DensityEstimator
            The self object.

        """
        for param, value in params.items():
            setattr(self, param, value)

@delayed
@wrap_non_picklable_objects
def testfunc(U, 
         U_diff_asc,
         U_diff_desc,
         nu,
         Z,
         q,
         h,
         kernel_id,
         dx):
    return(np.array(ekdefunc_merge(U, 
                          U_diff_asc,
                          U_diff_desc,
                          nu,
                          Z,
                          q,
                          h,
                          kernel_id,
                          dx)))

def volume_unit_ball(d, p=2):
    # from KDEpy
    return 2.0 ** d * gamma(1 + 1 / p) ** d / gamma(1 + d / p)

def gauss_integral(n):
    # from KDEpy
    factor = np.sqrt(np.pi * 2)
    if n % 2 == 0:
        return factor * factorial2(n - 1) / 2
    elif n % 2 == 1:
        return factor * norm.pdf(0) * factorial2(n - 1)
    else:
        raise ValueError("n must be odd or even.")

def scotts_rule(X):
    """
    Scott's rule according to "Multivariate density estimation", Scott 2015, p.164.
    The Silverman's rule is exactly the same in "Density estimation for statistics and data analysis", Silverman 1986, p.87.
    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features).

    Returns
    -------
    h : float
        Scotts rule bandwidth. The returned bandwidth should be then factored
        by the data variance.

    """
    n = X.shape[0]
    d = X.shape[1]
    return(n**(-1/(d + 4)))